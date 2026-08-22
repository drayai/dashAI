"""Stub backends for subprocess isolation tests.

This module is imported by the child training process through
DASHAI_FINE_TUNING_BACKEND_OVERRIDE, so it must not import pytest or any
test-only dependency.
"""

import time

from DashAI.back.fine_tuning.base import (
    FineTuningResult,
    TrainingCanceledError,
)


class StubSuccessBackend:
    """Completes immediately with a minimal valid artifact."""

    def train(self, request, progress, is_canceled):
        progress(0.5, "stub training", {"loss": 1.0})
        adapter = request.output_path / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (request.output_path / "manifest.json").write_text("{}", encoding="utf-8")
        return FineTuningResult(
            artifact_path=request.output_path,
            metrics={"loss": 1.0, "stub": "success"},
            runtime_metadata={"stub": True},
        )


class StubCooperativeBackend:
    """Trains slowly but honors the cancellation flag."""

    def train(self, request, progress, is_canceled):
        for step in range(300):
            if is_canceled():
                raise TrainingCanceledError
            progress(0.5, f"stub step {step}", {"step": step})
            time.sleep(0.5)
        raise AssertionError("cooperative stub should have been canceled")


class StubUnresponsiveBackend:
    """Never checks cancellation: forces the parent to terminate it."""

    def train(self, request, progress, is_canceled):
        progress(0.5, "stub ignoring cancellation", {"stub": "unresponsive"})
        time.sleep(300)
        raise AssertionError("unresponsive stub should have been killed")


class StubCrashingBackend:
    """Fails with a clear message."""

    def train(self, request, progress, is_canceled):
        raise RuntimeError("stub training exploded")
