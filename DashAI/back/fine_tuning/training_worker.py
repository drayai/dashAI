"""Isolated child process that executes one fine-tuning run.

The Huey job spawns this module (``python -m
DashAI.back.fine_tuning.training_worker <run_id>``) so training runs in its
own process. This enables strong cancellation on Windows: killing the child
destroys its CUDA context and frees VRAM immediately, and a crashed child
can never corrupt the worker.

The child talks to DashAI exclusively through the database (progress,
metrics, final status); it never touches the Huey queue. It intentionally
avoids building the full DI container and component registry so it starts
in seconds instead of paying the whole application import cost.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1


def _build_lightweight_config(local_path: Path) -> dict:
    """Build the path config without importing the component registry."""
    import DashAI.back.dependencies.config_builder as config_builder

    # get_initial_components imports every model class (torch, diffusers,
    # timm...). Training does not need the registry, so skip that cost.
    config_builder.get_initial_components = list
    config = config_builder.build_config_dict(
        local_path=local_path,
        logging_level=os.environ.get("DASHAI_LOGGING_LEVEL", "INFO"),
    )
    return config


def _resolve_backend(backend_value: str):
    """Resolve the training backend for the child process."""
    import importlib

    from DashAI.back.core.enums.status import FineTuningBackendType

    override = os.environ.get("DASHAI_FINE_TUNING_BACKEND_OVERRIDE")
    if override:
        module_path, _, attribute = override.partition(":")
        return getattr(importlib.import_module(module_path), attribute)()

    if backend_value == FineTuningBackendType.UNSLOTH.value:
        from DashAI.back.fine_tuning.unsloth_backend import UnslothFineTuningBackend

        return UnslothFineTuningBackend()

    from DashAI.back.fine_tuning.huggingface_backend import (
        HuggingFaceFineTuningBackend,
    )

    return HuggingFaceFineTuningBackend()


def execute_run(run_id: int) -> int:
    """Run one fine-tuning run to completion, cancellation or failure."""
    from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
        DatasetMapping,
        FineTuningMethod,
        TrainingParameters,
    )
    from DashAI.back.core.enums.status import FineTuningStatus
    from DashAI.back.dependencies.database import setup_sqlite_db
    from DashAI.back.dependencies.database.models import Dataset, FineTuningRun
    from DashAI.back.fine_tuning.base import (
        FineTuningRequest,
        TrainingCanceledError,
    )
    from DashAI.back.fine_tuning.dataset import prepare_dataset
    from DashAI.back.fine_tuning.model_store import ensure_model
    from DashAI.back.fine_tuning.training_process import CANCELED_MESSAGE

    local_path = Path(os.environ["DASHAI_LOCAL_PATH"])
    config = _build_lightweight_config(local_path)
    engine, session_factory = setup_sqlite_db(config)

    def update(fraction, message, metrics=None) -> None:
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            if run:
                run.progress = max(0.0, min(1.0, fraction))
                run.progress_message = message
                if metrics:
                    run.metrics = metrics
                db.commit()

    def canceled() -> bool:
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            return bool(run and run.cancellation_requested)

    def persist_failure(exc: Exception) -> None:
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            if run and run.status not in {
                FineTuningStatus.CANCELED,
                FineTuningStatus.COMPLETED,
            }:
                run.mark_failed(str(exc))
                db.commit()

    try:
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            if not run:
                logger.error("Run %s does not exist.", run_id)
                return EXIT_FAILED
            if run.status not in (
                FineTuningStatus.QUEUED,
                FineTuningStatus.RUNNING,
            ):
                logger.error("Run %s is not active (%s).", run_id, run.status)
                return EXIT_FAILED
            dataset = db.get(Dataset, run.dataset_id)
            if not dataset:
                raise RuntimeError(f"Dataset {run.dataset_id} does not exist.")
            dataset_path = dataset.file_path
            model_id = run.base_model_id
            model_revision = run.base_model_revision
            backend_value = run.backend
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

        request = FineTuningRequest(
            run_id=run_id,
            model_id=model_id,
            model_revision=model_revision,
            method=method,
            mapping=mapping,
            parameters=parameters,
            dataset=prepared,
            model_path=model_path,
            output_path=Path(config["FINE_TUNING_PATH"]) / f"run-{run_id}",
            resolved_revision=resolved_revision,
        )
        backend = _resolve_backend(backend_value)
        result = backend.train(request, update, canceled)

        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            run.artifact_path = str(result.artifact_path)
            run.metrics = result.metrics
            run.runtime_metadata = result.runtime_metadata
            run.mark_completed()
            db.commit()
        return EXIT_OK
    except TrainingCanceledError:
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            if run:
                run.mark_canceled()
                run.progress_message = CANCELED_MESSAGE
                db.commit()
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - the child reports every failure
        logger.exception("Isolated training for run %s failed", run_id)
        persist_failure(exc)
        return EXIT_FAILED
    finally:
        engine.dispose()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python -m DashAI.back.fine_tuning.training_worker <run_id>")
        return EXIT_FAILED
    try:
        run_id = int(sys.argv[1])
    except ValueError:
        print("The run id must be an integer.")
        return EXIT_FAILED
    return execute_run(run_id)


if __name__ == "__main__":
    raise SystemExit(main())
