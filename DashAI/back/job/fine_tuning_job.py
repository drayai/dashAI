import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kink import di, inject

from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
    DatasetMapping,
    FineTuningMethod,
    TrainingParameters,
)
from DashAI.back.core.enums.status import (
    FineTuningBackendType,
    FineTuningStatus,
)
from DashAI.back.dependencies.database.models import Dataset, FineTuningRun
from DashAI.back.fine_tuning.base import FineTuningRequest, TrainingCanceledError
from DashAI.back.fine_tuning.dataset import prepare_dataset
from DashAI.back.fine_tuning.huggingface_backend import (
    HuggingFaceFineTuningBackend,
)
from DashAI.back.fine_tuning.model_store import ensure_model
from DashAI.back.fine_tuning.resource_lock import training_lock
from DashAI.back.job.base_job import BaseJob, JobError

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)

# How long the worker waits for the GPU lock before failing the run with an
# actionable error (smooths the transition between back-to-back jobs).
WORKER_LOCK_WAIT_SECONDS = 10.0


class FineTuningJob(BaseJob):
    """Execute a persistent fine-tuning run using a pluggable backend."""

    @inject
    def set_status_as_delivered(
        self, session_factory: "sessionmaker" = lambda di: di["session_factory"]
    ) -> None:
        with session_factory() as db:
            run = db.get(FineTuningRun, self.kwargs["run_id"])
            if not run:
                raise JobError("Fine-tuning run does not exist.")
            run.mark_queued(self.kwargs.get("huey_id"))
            db.commit()

    @inject
    def set_status_as_error(
        self, session_factory: "sessionmaker" = lambda di: di["session_factory"]
    ) -> None:
        run_id = self.kwargs.get("run_id")
        if run_id is None:
            return
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            if run and run.status not in {
                FineTuningStatus.CANCELED,
                FineTuningStatus.COMPLETED,
            }:
                run.mark_failed(run.error_message or "Fine-tuning job failed.")
                db.commit()

    def get_job_name(self) -> str:
        run_id = self.kwargs.get("run_id")
        try:
            with di["session_factory"]() as db:
                run = db.get(FineTuningRun, run_id)
                return f"Fine-tune: {run.name}" if run else "Fine-tuning"
        except Exception:
            return f"Fine-tuning #{run_id}"

    def run(self) -> None:
        session_factory = di["session_factory"]
        config = di["config"]
        run_id = self.kwargs["run_id"]
        lock = training_lock(config)
        lock_owner = f"fine_tuning_run_{run_id}"
        lock_acquired = False

        def update(
            fraction: float, message: str, metrics: dict[str, Any] | None = None
        ) -> None:
            self.report_progress(fraction, message)
            with session_factory() as callback_db:
                callback_run = callback_db.get(FineTuningRun, run_id)
                if callback_run:
                    callback_run.progress = max(0.0, min(1.0, fraction))
                    callback_run.progress_message = message
                    if metrics:
                        callback_run.metrics = metrics
                    callback_db.commit()

        def canceled() -> bool:
            with session_factory() as callback_db:
                callback_run = callback_db.get(FineTuningRun, run_id)
                return bool(callback_run and callback_run.cancellation_requested)

        try:
            with session_factory() as db:
                run = db.get(FineTuningRun, run_id)
                if not run:
                    raise JobError(f"Fine-tuning run {run_id} does not exist.")
                if canceled():
                    run.mark_canceled()
                    db.commit()
                    return
                # Re-check exclusion inside the worker: two runs enqueued at
                # the same time both pass the API check, but only one may
                # hold the GPU. The short wait smooths job transitions.
                lock.acquire(lock_owner, wait_seconds=WORKER_LOCK_WAIT_SECONDS)
                lock_acquired = True
                run.mark_running()
                run.huey_id = self.kwargs.get("huey_id", run.huey_id)
                db.commit()
                dataset = db.get(Dataset, run.dataset_id)
                if not dataset:
                    raise JobError(f"Dataset {run.dataset_id} does not exist.")
                dataset_path = dataset.file_path
                model_id = run.base_model_id
                model_revision = run.base_model_revision
                run_backend = run.backend
                mapping = DatasetMapping.model_validate(run.dataset_mapping)
                parameters = TrainingParameters.model_validate(run.training_parameters)
                method = FineTuningMethod(run.method)

            update(0.01, "Validating dataset")
            prepared = prepare_dataset(dataset_path, mapping, parameters)
            if canceled():
                raise TrainingCanceledError

            model_path, resolved_revision = ensure_model(
                Path(config["LLM_MODELS_PATH"]),
                model_id,
                model_revision,
                lambda fraction, message: update(fraction, message),
            )
            with session_factory() as db:
                run = db.get(FineTuningRun, run_id)
                run.resolved_model_revision = resolved_revision
                db.commit()

            artifact_path = Path(config["FINE_TUNING_PATH"]) / f"run-{run_id}"
            request = FineTuningRequest(
                run_id=run_id,
                model_id=model_id,
                model_revision=model_revision,
                method=method,
                mapping=mapping,
                parameters=parameters,
                dataset=prepared,
                model_path=model_path,
                output_path=artifact_path,
                resolved_revision=resolved_revision,
            )
            if run_backend == FineTuningBackendType.UNSLOTH:
                # Deferred import: Unsloth patches global torch state and is
                # only touched when a run actually requests it.
                from DashAI.back.fine_tuning.unsloth_backend import (
                    UnslothFineTuningBackend,
                )

                backend = UnslothFineTuningBackend()
            else:
                try:
                    backend = di["fine_tuning_backend"]
                except KeyError:
                    backend = HuggingFaceFineTuningBackend()
            result = backend.train(request, update, canceled)

            with session_factory() as db:
                run = db.get(FineTuningRun, run_id)
                run.artifact_path = str(result.artifact_path)
                run.metrics = result.metrics
                run.runtime_metadata = result.runtime_metadata
                run.mark_completed()
                db.commit()
        except TrainingCanceledError:
            with session_factory() as db:
                run = db.get(FineTuningRun, run_id)
                if run:
                    run.mark_canceled()
                    db.commit()
        except Exception as exc:
            logger.exception("Fine-tuning run %s failed", run_id)
            with session_factory() as db:
                run = db.get(FineTuningRun, run_id)
                if run:
                    run.mark_failed(str(exc))
                    db.commit()
            raise JobError(str(exc)) from exc
        finally:
            if lock_acquired:
                lock.release(lock_owner)
