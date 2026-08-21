"""Job for downloading a curated base model into DashAI-managed storage."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from kink import di, inject
from sqlalchemy import exc

from DashAI.back.core.enums.status import DatafileStatus
from DashAI.back.dependencies.database.models import ManagedLocalModel
from DashAI.back.fine_tuning.model_store import directory_size, ensure_model
from DashAI.back.job.base_job import BaseJob, JobError

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

log = logging.getLogger(__name__)


class ManagedModelDownloadJob(BaseJob):
    """Download a catalog base model tracked by a `ManagedLocalModel` row.

    Parameters
    ----------
    kwargs : dict
        - managed_model_id: int (DB row id)
    """

    @inject
    def set_status_as_delivered(
        self, session_factory: "sessionmaker" = lambda di: di["session_factory"]
    ) -> None:
        """No-op: model downloads don't use the delivered state."""

    @inject
    def set_status_as_error(
        self, session_factory: "sessionmaker" = lambda di: di["session_factory"]
    ) -> None:
        managed_model_id: int = self.kwargs["managed_model_id"]
        error_msg: str = self.kwargs.get("_error_message", "")
        with session_factory() as db:
            row: ManagedLocalModel = db.get(ManagedLocalModel, managed_model_id)
            if row is not None:
                row.status = DatafileStatus.ERROR
                row.error_message = error_msg
                try:
                    db.commit()
                except exc.SQLAlchemyError as e:
                    log.exception(e)

    def get_job_name(self) -> str:
        return f"Model download: {self.kwargs.get('model_key', '')}"

    @inject
    def run(self) -> None:
        config = di["config"]
        session_factory = di["session_factory"]

        managed_model_id: int = self.kwargs["managed_model_id"]

        try:
            with session_factory() as db:
                row: ManagedLocalModel = db.get(ManagedLocalModel, managed_model_id)
                if row is None:
                    raise JobError(f"Managed model row {managed_model_id} not found.")
                model_key = row.model_key
                revision = row.base_model_revision

            def report(fraction: float, message: str) -> None:
                self.report_progress(10 + fraction * 85, message)

            destination, resolved_revision = ensure_model(
                Path(config["LLM_MODELS_PATH"]),
                model_key,
                revision,
                report,
            )

            with session_factory() as db:
                row = db.get(ManagedLocalModel, managed_model_id)
                if row is None:
                    raise JobError(f"Managed model row {managed_model_id} not found.")
                row.status = DatafileStatus.READY
                row.resolved_revision = resolved_revision
                row.size_bytes = directory_size(destination)
                row.error_message = None
                try:
                    db.commit()
                except exc.SQLAlchemyError as e:
                    log.exception(e)
                    raise JobError("DB error saving model download.") from e

            self.report_progress(100, "Model ready")
            log.debug("Managed model download job %d completed.", managed_model_id)

        except Exception as e:
            err_msg = str(e)
            log.error(
                "Managed model download job %d failed: %s", managed_model_id, err_msg
            )
            self.kwargs["_error_message"] = err_msg
            self.set_status_as_error()
            if isinstance(e, JobError):
                raise
            raise JobError(err_msg) from e
