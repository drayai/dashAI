import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from DashAI.back.fine_tuning.resource_lock import (
    GpuLock,
    ResourceLockError,
    gpu_lock_path,
)


@pytest.fixture(name="lock")
def lock_fixture(tmp_path):
    return GpuLock(tmp_path / "fine_tuning" / "gpu.lock")


def test_gpu_lock_path_stays_inside_managed_storage():
    config = {"FINE_TUNING_PATH": "E:/DashAI-data/fine_tuning"}
    assert gpu_lock_path(config) == Path("E:/DashAI-data/fine_tuning") / "gpu.lock"


def test_second_acquire_is_rejected_with_actionable_message(lock):
    lock.acquire("fine_tuning_run_1")
    try:
        with pytest.raises(ResourceLockError) as excinfo:
            lock.acquire("fine_tuning_run_2")
        assert "GPU is busy" in str(excinfo.value)
        assert "fine_tuning_run_1" in str(excinfo.value)
    finally:
        lock.release("fine_tuning_run_1")


def test_release_allows_next_acquire(lock):
    lock.acquire("run_a")
    assert lock.release("run_a") is True
    lock.acquire("run_b")
    assert lock.is_locked()
    lock.release("run_b")
    assert not lock.is_locked()


def test_release_refuses_foreign_owner(lock):
    lock.acquire("run_a")
    assert lock.release("run_b") is False
    assert lock.is_locked()
    assert lock.release("run_a") is True


def test_stale_lock_of_dead_process_is_taken_over(lock):
    # The child has exited: a lock claiming its pid must be discarded.
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    lock.path.write_text(
        json.dumps({"owner": "dead_run", "pid": _last_child_pid()}),
        encoding="utf-8",
    )
    lock.acquire("new_run")
    try:
        assert lock.read_owner()["owner"] == "new_run"
    finally:
        lock.release("new_run")


def _last_child_pid():
    # Spawn a process just to observe a pid that is guaranteed to exit.
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = proc.pid
    proc.wait(timeout=30)
    return pid


def test_acquire_waits_until_deadline_then_fails(lock):
    lock.acquire("holder")
    started = time.monotonic()
    try:
        with pytest.raises(ResourceLockError):
            lock.acquire("waiter", wait_seconds=1.0)
        assert time.monotonic() - started >= 0.9
    finally:
        lock.release("holder")


def test_busy_message_names_the_holder(lock):
    lock.acquire("fine_tuning_run_7")
    try:
        assert "fine_tuning_run_7" in lock.busy_message()
    finally:
        lock.release("fine_tuning_run_7")
