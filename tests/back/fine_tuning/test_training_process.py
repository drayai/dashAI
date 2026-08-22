import json
import time
from pathlib import Path

import pyarrow as pa
import pytest

from DashAI.back.core.enums.status import FineTuningStatus
from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset, save_dataset
from DashAI.back.dependencies.database.models import Dataset, FineTuningRun
from DashAI.back.fine_tuning.training_process import run_isolated_training

STUB_MODULE = "tests.back.fine_tuning.stub_backends"


@pytest.fixture(name="app_container", scope="module")
def app_container_fixture(tmp_path_factory):
    from DashAI.back.app import create_app

    local_path = tmp_path_factory.mktemp("isolated-training")
    app = create_app(local_path=local_path, logging_level="ERROR", enable_seeding=False)
    yield app.container
    app.container["engine"].dispose()


def _create_dataset_row(container) -> int:
    root = Path(container["config"]["DATASETS_PATH"]) / "isolated"
    save_dataset(
        DashAIDataset(
            pa.table(
                {
                    "instruction": ["Say hi", "Say bye", "Count", "Name a color"],
                    "response": ["Hi", "Bye", "One", "Blue"],
                }
            )
        ),
        root / "dataset",
    )
    with container["session_factory"]() as db:
        entry = db.query(Dataset).filter_by(name="isolated-training-dataset").first()
        if entry is None:
            entry = Dataset(name="isolated-training-dataset", file_path=str(root))
            db.add(entry)
            db.commit()
        return entry.id


def _create_run(container, dataset_id: int, name: str) -> int:
    with container["session_factory"]() as db:
        run = FineTuningRun(
            name=name,
            dataset_id=dataset_id,
            base_model_id="qwen2.5-0.5b-instruct",
            base_model_revision="main",
            method="lora",
            dataset_mapping={
                "format": "prompt_completion",
                "prompt_column": "instruction",
                "completion_column": "response",
                "validation_split": 0.25,
            },
            training_parameters={"preset": "quick_test", "max_samples": 4},
            status=FineTuningStatus.QUEUED,
        )
        db.add(run)
        db.commit()
        return run.id


def _prepare_local_model_snapshot(container) -> None:
    destination = (
        Path(container["config"]["LLM_MODELS_PATH"]) / "qwen2.5-0.5b-instruct" / "main"
    )
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "config.json").write_text("{}", encoding="utf-8")
    (destination / ".dashai_model.json").write_text(
        json.dumps(
            {
                "model_id": "qwen2.5-0.5b-instruct",
                "repository": "Qwen/Qwen2.5-0.5B-Instruct",
                "requested_revision": "main",
                "resolved_revision": "stub-sha",
            }
        ),
        encoding="utf-8",
    )


def _wait_for_progress(container, run_id: int, timeout: float = 90.0) -> None:
    """Wait until the child reports training progress."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with container["session_factory"]() as db:
            run = db.get(FineTuningRun, run_id)
            if run and run.progress_message and run.progress_message.startswith("stub"):
                return
        time.sleep(0.5)
    raise AssertionError("The training child never reported progress.")


def _request_cancel(container, run_id: int) -> None:
    with container["session_factory"]() as db:
        run = db.get(FineTuningRun, run_id)
        run.cancellation_requested = True
        db.commit()


def _status(container, run_id: int):
    with container["session_factory"]() as db:
        run = db.get(FineTuningRun, run_id)
        db.refresh(run)
        return run.status, run.error_message, run.progress_message


def _canceled(container, run_id: int):
    def check() -> bool:
        with container["session_factory"]() as db:
            run = db.get(FineTuningRun, run_id)
            return bool(run and run.cancellation_requested)

    return check


def test_isolated_child_completes_a_run(app_container, monkeypatch):
    _prepare_local_model_snapshot(app_container)
    dataset_id = _create_dataset_row(app_container)
    run_id = _create_run(app_container, dataset_id, "isolated success")
    monkeypatch.setenv(
        "DASHAI_FINE_TUNING_BACKEND_OVERRIDE",
        f"{STUB_MODULE}:StubSuccessBackend",
    )

    status = run_isolated_training(
        run_id,
        app_container["session_factory"],
        Path(app_container["config"]["LOCAL_PATH"]),
        report=lambda: None,
        canceled=_canceled(app_container, run_id),
    )

    assert status == FineTuningStatus.COMPLETED
    final_status, error, _ = _status(app_container, run_id)
    assert final_status == FineTuningStatus.COMPLETED
    assert error is None
    with app_container["session_factory"]() as db:
        run = db.get(FineTuningRun, run_id)
        assert run.metrics["stub"] == "success"
        assert (Path(run.artifact_path) / "manifest.json").exists()


def test_isolated_child_stops_cooperatively_within_grace(app_container, monkeypatch):
    _prepare_local_model_snapshot(app_container)
    dataset_id = _create_dataset_row(app_container)
    run_id = _create_run(app_container, dataset_id, "isolated cooperative")
    monkeypatch.setenv(
        "DASHAI_FINE_TUNING_BACKEND_OVERRIDE",
        f"{STUB_MODULE}:StubCooperativeBackend",
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.training_process.CANCEL_GRACE_SECONDS", 20.0
    )

    import threading

    def cancel_when_training() -> None:
        _wait_for_progress(app_container, run_id)
        _request_cancel(app_container, run_id)

    canceller = threading.Thread(target=cancel_when_training)
    canceller.start()
    try:
        status = run_isolated_training(
            run_id,
            app_container["session_factory"],
            Path(app_container["config"]["LOCAL_PATH"]),
            report=lambda: None,
            canceled=_canceled(app_container, run_id),
        )
    finally:
        canceller.join(timeout=120)

    assert status == FineTuningStatus.CANCELED
    final_status, _error, message = _status(app_container, run_id)
    assert final_status == FineTuningStatus.CANCELED


def test_isolated_child_is_terminated_when_unresponsive(app_container, monkeypatch):
    _prepare_local_model_snapshot(app_container)
    dataset_id = _create_dataset_row(app_container)
    run_id = _create_run(app_container, dataset_id, "isolated strong cancel")
    monkeypatch.setenv(
        "DASHAI_FINE_TUNING_BACKEND_OVERRIDE",
        f"{STUB_MODULE}:StubUnresponsiveBackend",
    )
    # Keep the escalation fast: the stub ignores the cooperative flag, so
    # the parent must terminate it after a short grace.
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.training_process.CANCEL_GRACE_SECONDS", 3.0
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.training_process.POLL_INTERVAL_SECONDS", 0.2
    )

    import threading

    def cancel_when_training() -> None:
        _wait_for_progress(app_container, run_id)
        _request_cancel(app_container, run_id)

    canceller = threading.Thread(target=cancel_when_training)
    canceller.start()
    try:
        status = run_isolated_training(
            run_id,
            app_container["session_factory"],
            Path(app_container["config"]["LOCAL_PATH"]),
            report=lambda: None,
            canceled=_canceled(app_container, run_id),
        )
    finally:
        canceller.join(timeout=120)

    # Returning at all proves the child was reaped by wait(); the run must
    # be canceled, not left running.
    assert status == FineTuningStatus.CANCELED
    final_status, _error, _message = _status(app_container, run_id)
    assert final_status == FineTuningStatus.CANCELED


def test_isolated_child_failure_is_reported(app_container, monkeypatch):
    _prepare_local_model_snapshot(app_container)
    dataset_id = _create_dataset_row(app_container)
    run_id = _create_run(app_container, dataset_id, "isolated failure")
    monkeypatch.setenv(
        "DASHAI_FINE_TUNING_BACKEND_OVERRIDE",
        f"{STUB_MODULE}:StubCrashingBackend",
    )

    status = run_isolated_training(
        run_id,
        app_container["session_factory"],
        Path(app_container["config"]["LOCAL_PATH"]),
        report=lambda: None,
        canceled=_canceled(app_container, run_id),
    )

    assert status == FineTuningStatus.FAILED
    final_status, error, _message = _status(app_container, run_id)
    assert final_status == FineTuningStatus.FAILED
    assert "stub training exploded" in error
