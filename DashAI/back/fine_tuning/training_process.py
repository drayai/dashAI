"""Parent-side orchestration of the isolated training child process.

The Huey job keeps ownership of the GPU lock and of the run's final state;
the child only trains and reports through the database. Cancellation gives
the child a cooperative grace period (it checks the flag between steps),
then terminates it: process death releases the CUDA context immediately,
which is what makes the cancellation strong on Windows.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from DashAI.back.core.enums.status import FineTuningStatus
from DashAI.back.dependencies.database.models import FineTuningRun

logger = logging.getLogger(__name__)

CANCELED_MESSAGE = "Training canceled."
# Grace given to the child to notice the cooperative flag, and to die after
# termination, before escalating.
CANCEL_GRACE_SECONDS = 30.0
TERMINATE_GRACE_SECONDS = 10.0
POLL_INTERVAL_SECONDS = 1.0


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _child_environment(local_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["DASHAI_LOCAL_PATH"] = str(local_path)
    existing = env.get("PYTHONPATH", "")
    root = str(_package_root())
    env["PYTHONPATH"] = f"{root}{os.pathsep}{existing}" if existing else root
    return env


def _run_status(session_factory, run_id: int) -> tuple | None:
    """Return (status, error_message) for a run, or None if it is gone."""
    with session_factory() as db:
        run = db.get(FineTuningRun, run_id)
        if run is None:
            return None
        return run.status, run.error_message


def run_isolated_training(
    run_id: int,
    session_factory,
    local_path: Path,
    report: Callable[[], None],
    canceled: Callable[[], bool],
) -> FineTuningStatus:
    """Spawn the training child, forward progress, enforce cancellation.

    Returns the final status of the run. The child persists completion,
    cancellation and failure itself; this function only intervenes when the
    child was killed or died without leaving a terminal state.
    """
    child = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "DashAI.back.fine_tuning.training_worker",
            str(run_id),
        ],
        env=_child_environment(local_path),
        cwd=str(_package_root()),
    )
    logger.info("Started isolated training process %s for run %s.", child.pid, run_id)

    cancel_requested = False
    cancel_deadline = None
    return_code = None
    try:
        while True:
            return_code = child.poll()
            if return_code is not None:
                break
            if not cancel_requested and canceled():
                cancel_requested = True
                cancel_deadline = time.monotonic() + CANCEL_GRACE_SECONDS
                logger.info(
                    "Cancellation requested for run %s; child %s has %.0fs to "
                    "stop cooperatively.",
                    run_id,
                    child.pid,
                    CANCEL_GRACE_SECONDS,
                )
            if cancel_requested and time.monotonic() >= cancel_deadline:
                logger.warning(
                    "Terminating unresponsive training child %s for run %s.",
                    child.pid,
                    run_id,
                )
                child.terminate()
                try:
                    child.wait(TERMINATE_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                return_code = child.returncode
                break
            report()
            time.sleep(POLL_INTERVAL_SECONDS)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()

    snapshot = _run_status(session_factory, run_id)
    status = snapshot[0] if snapshot else None
    if status in (FineTuningStatus.COMPLETED, FineTuningStatus.CANCELED):
        return status

    if cancel_requested:
        with session_factory() as db:
            run = db.get(FineTuningRun, run_id)
            if run:
                run.mark_canceled()
                run.progress_message = CANCELED_MESSAGE
                db.commit()
        return FineTuningStatus.CANCELED

    message = f"Training process exited with code {return_code}."
    if snapshot and snapshot[1]:
        message = snapshot[1]
    with session_factory() as db:
        run = db.get(FineTuningRun, run_id)
        if run:
            run.mark_failed(message)
            db.commit()
    return FineTuningStatus.FAILED
