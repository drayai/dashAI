"""Cross-process exclusion for the shared GPU training/inference resource.

Fine-tuning jobs and managed local inference compete for the same GPU. This
module provides a small file-based lock stored inside DashAI-managed storage
so separate processes (API server and Huey worker) agree on ownership.

The lock file stores the owning process id: a lock whose owner process died
is treated as stale and can be taken over. Startup reconciliation also
removes leftovers, because no DashAI worker survives an application restart.
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOCK_FILENAME = "gpu.lock"


class ResourceLockError(RuntimeError):
    """Raised when the shared GPU resource is held by another process."""


def gpu_lock_path(config: dict[str, Any]) -> Path:
    """Return the managed-storage path of the fine-tuning GPU lock."""
    return Path(config["FINE_TUNING_PATH"]) / LOCK_FILENAME


def _pid_alive(pid: int) -> bool:
    """Return True when a process with the given pid is still running."""
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
                return bool(ok) and exit_code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class GpuLock:
    """File-based lock guarding exclusive GPU use between processes."""

    def __init__(self, lock_path: Path):
        self.path = Path(lock_path)

    def read_owner(self) -> Optional[dict[str, Any]]:
        """Return the lock payload, or None when absent or unreadable."""
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def is_locked(self) -> bool:
        """True when the lock exists and its owner process is alive."""
        owner = self.read_owner()
        if owner is None:
            return False
        pid = owner.get("pid")
        if not isinstance(pid, int):
            return True
        return _pid_alive(pid)

    def busy_message(self) -> str:
        """Actionable message describing the current lock holder."""
        owner = self.read_owner()
        if owner and owner.get("owner"):
            holder = f"'{owner['owner']}' (pid {owner.get('pid', 'unknown')})"
        else:
            holder = "another process"
        return (
            "The GPU is busy with another fine-tuning job "
            f"(current holder: {holder}). Wait for it to finish or cancel it "
            "before starting a new one."
        )

    def acquire(self, owner: str, wait_seconds: float = 0.0) -> None:
        """Acquire the lock, raising ResourceLockError while it stays busy.

        A lock whose owning process died is discarded and taken over. When
        `wait_seconds` is positive the acquisition is retried until the
        deadline to smooth transitions between queued jobs.
        """
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while True:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._discard_stale()
            payload = json.dumps(
                {
                    "owner": owner,
                    "pid": os.getpid(),
                    "created": datetime.now(timezone.utc).isoformat(),
                }
            )
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise ResourceLockError(self.busy_message()) from None
                time.sleep(0.5)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
            return

    def release(self, owner: Optional[str] = None) -> bool:
        """Release the lock. Returns True when this call removed it.

        When `owner` is given the lock is only removed if it still belongs
        to that owner, so a stale takeover is never undone by a late
        release from the previous holder.
        """
        if not self.path.exists():
            return False
        current = self.read_owner()
        if (
            owner is not None
            and current is not None
            and current.get("owner") not in (None, owner)
        ):
            return False
        try:
            self.path.unlink()
        except OSError:
            logger.warning("Could not remove GPU lock file %s.", self.path)
            return False
        return True

    def clear(self) -> bool:
        """Remove the lock unconditionally (startup reconciliation)."""
        try:
            self.path.unlink()
            return True
        except OSError:
            return False

    def _discard_stale(self) -> None:
        """Rename away a lock left by a dead process (or a corrupt file)."""
        owner = self.read_owner()
        pid = owner.get("pid") if owner else None
        stale = owner is not None and isinstance(pid, int) and not _pid_alive(pid)
        corrupt = owner is None and self.path.exists()
        if stale or corrupt:
            leftover = self.path.with_name(
                f"{self.path.name}.stale-{uuid.uuid4().hex[:8]}"
            )
            try:
                self.path.replace(leftover)
                logger.warning(
                    "Discarded stale GPU lock %s (left as %s).",
                    self.path,
                    leftover.name,
                )
            except OSError:
                logger.warning("Could not discard stale GPU lock %s.", self.path)


def training_lock(config: dict[str, Any]) -> GpuLock:
    """Return the GPU lock for the current DashAI configuration."""
    return GpuLock(gpu_lock_path(config))
