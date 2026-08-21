"""Idempotent startup reconciliation for fine-tuning state.

DashAI runs training inside a Huey worker that dies with the application, so
a restart leaves ``running`` runs without an executor and may leave
``queued`` runs without a task. This module brings the database and the
artifact directory back to a truthful, restartable state at startup:

- ``running`` runs are marked as failed with an interruption message; they
  are never completed based on the presence of a directory.
- ``queued`` runs are kept only while their task is still pending in the
  persistent queue. Otherwise they are marked as failed. A live worker that
  was executing the task immediately re-marks the run as running, so a
  spurious failure is always overwritten with the truthful outcome.
- Training temporaries (``*.tmp`` directories) inside FINE_TUNING_PATH are
  quarantined by renaming instead of deleted, and only inside that path.
- Any leftover GPU lock is cleared, because no worker survives a restart.

Every step is idempotent: reconciling twice in a row has no additional
effect.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from DashAI.back.core.enums.status import FineTuningStatus
from DashAI.back.dependencies.database.models import FineTuningRun

logger = logging.getLogger(__name__)

TEMP_SUFFIX = ".tmp"
ORPHAN_PREFIX = "orphaned-"

INTERRUPTED_RUNNING_MESSAGE = (
    "Training was interrupted because DashAI stopped while the run was "
    "active. Start the run again to retry it."
)
MISSING_TASK_MESSAGE = (
    "The queued training task no longer exists after a DashAI restart. "
    "Start the run again to retry it."
)


def _task_snapshot(job_queue: Any, huey_id: str | None) -> dict[str, Any] | None:
    """Return ``{'pending': bool}`` for a Huey task, or None if unknown.

    Delegates to ``task_snapshot`` on the queue when available (implemented
    by HueyJobQueue). Unknown queues keep their queued runs untouched.
    """
    if not huey_id or not hasattr(job_queue, "task_snapshot"):
        return None
    return job_queue.task_snapshot(huey_id)


def reconcile_fine_tuning_runs(session_factory: Any, job_queue: Any) -> dict[str, list]:
    """Mark interrupted runs as failed. Runs are never marked completed."""
    summary: dict[str, list] = {
        "failed_running": [],
        "failed_without_task": [],
        "kept_queued": [],
    }
    with session_factory() as db:
        runs = db.scalars(
            select(FineTuningRun).where(
                FineTuningRun.status.in_(
                    [FineTuningStatus.RUNNING, FineTuningStatus.QUEUED]
                )
            )
        ).all()
        for run in runs:
            if run.status == FineTuningStatus.RUNNING:
                run.mark_failed(INTERRUPTED_RUNNING_MESSAGE)
                summary["failed_running"].append(run.id)
                continue
            snapshot = _task_snapshot(job_queue, run.huey_id)
            if snapshot is None or snapshot.get("pending"):
                summary["kept_queued"].append(run.id)
                continue
            run.mark_failed(MISSING_TASK_MESSAGE)
            summary["failed_without_task"].append(run.id)
        db.commit()
    for run_id in summary["failed_running"]:
        logger.warning("Fine-tuning run %s was running during a restart.", run_id)
    for run_id in summary["failed_without_task"]:
        logger.warning("Fine-tuning run %s lost its queued task.", run_id)
    return summary


def quarantine_orphan_temp_dirs(fine_tuning_path: Path) -> list[str]:
    """Rename leftover ``*.tmp`` training directories to a quarantine name.

    Only directories directly inside FINE_TUNING_PATH are considered; they
    are renamed, never deleted, so artifacts are recoverable. The rename
    removes the ``.tmp`` suffix, making a second pass a no-op.
    """
    root = Path(fine_tuning_path)
    if not root.exists():
        return []
    quarantined: list[str] = []
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for candidate in sorted(root.glob(f"*{TEMP_SUFFIX}")):
        if not candidate.is_dir():
            continue
        target = candidate.with_name(f"{ORPHAN_PREFIX}{candidate.name}-{timestamp}")
        try:
            candidate.rename(target)
        except OSError as exc:
            logger.warning(
                "Could not quarantine orphan directory %s: %s", candidate, exc
            )
            continue
        quarantined.append(target.name)
        logger.warning(
            "Quarantined orphan training directory %s as %s.", candidate, target
        )
    return quarantined


def clear_stale_gpu_lock(config: dict[str, Any]) -> bool:
    """Remove any leftover GPU lock file at startup."""
    from DashAI.back.fine_tuning.resource_lock import gpu_lock_path

    lock_path = gpu_lock_path(config)
    if not lock_path.exists():
        return False
    try:
        lock_path.unlink()
        logger.warning("Cleared leftover GPU lock %s at startup.", lock_path)
        return True
    except OSError as exc:
        logger.warning("Could not remove GPU lock %s: %s", lock_path, exc)
        return False


def reconcile_fine_tuning(
    config: dict[str, Any], session_factory: Any, job_queue: Any
) -> dict[str, list | bool]:
    """Run every fine-tuning reconciliation step and return a summary."""
    runs_summary = reconcile_fine_tuning_runs(session_factory, job_queue)
    quarantined = quarantine_orphan_temp_dirs(Path(config["FINE_TUNING_PATH"]))
    lock_cleared = clear_stale_gpu_lock(config)
    return {
        **runs_summary,
        "quarantined_dirs": quarantined,
        "lock_cleared": lock_cleared,
    }
