import uuid
from pathlib import Path

import pyarrow as pa
import pytest
from fastapi.testclient import TestClient

from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset, save_dataset
from DashAI.back.dependencies.database.models import Dataset, FineTuningRun
from DashAI.back.fine_tuning.base import FineTuningResult
from DashAI.back.fine_tuning.resource_lock import GpuLock, gpu_lock_path
from DashAI.back.job.base_job import JobError
from DashAI.back.job.fine_tuning_job import FineTuningJob


class FailingBackend:
    def train(self, request, progress, is_canceled):
        raise RuntimeError("boom during training")


class SuccessfulBackend:
    def train(self, request, progress, is_canceled):
        progress(1.0, "done", {"loss": 1.0})
        adapter = request.output_path / "adapter"
        adapter.mkdir(parents=True)
        (request.output_path / "manifest.json").write_text("{}", encoding="utf-8")
        return FineTuningResult(
            artifact_path=request.output_path,
            metrics={"loss": 1.0},
            runtime_metadata={},
        )


def _create_dataset(client: TestClient) -> Dataset:
    container = client.app.container
    root = Path(container["config"]["DATASETS_PATH"]) / "robustness-test"
    dataset = DashAIDataset(
        pa.table(
            {
                "instruction": ["Say hi", "Say bye", "Count", "Name a color"],
                "response": ["Hi", "Bye", "One", "Blue"],
            }
        )
    )
    save_dataset(dataset, root / "dataset")
    with container["session_factory"]() as db:
        entry = Dataset(
            name=f"robustness_dataset_{uuid.uuid4().hex}", file_path=str(root)
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        db.expunge(entry)
        return entry


def _create_run(client: TestClient, dataset_id: int, name: str) -> int:
    created = client.post(
        "/api/v1/fine-tuning/runs",
        json={
            "name": name,
            "dataset_id": dataset_id,
            "base_model_id": "qwen2.5-0.5b-instruct",
            "base_model_revision": "main",
            "method": "lora",
            "dataset_mapping": {
                "format": "prompt_completion",
                "prompt_column": "instruction",
                "completion_column": "response",
                "validation_split": 0.25,
            },
            "training_parameters": {"preset": "quick_test", "max_steps": 1},
        },
    )
    assert created.status_code == 201
    return created.json()["id"]


def test_unsloth_run_uses_the_unsloth_backend(client: TestClient, monkeypatch):
    dataset = _create_dataset(client)
    created = client.post(
        "/api/v1/fine-tuning/runs",
        json={
            "name": "unsloth run",
            "dataset_id": dataset.id,
            "base_model_id": "qwen2.5-0.5b-instruct",
            "base_model_revision": "main",
            "method": "lora",
            "backend": "unsloth",
            "dataset_mapping": {
                "format": "prompt_completion",
                "prompt_column": "instruction",
                "completion_column": "response",
                "validation_split": 0.25,
            },
            "training_parameters": {"preset": "quick_test", "max_steps": 1},
        },
    )
    assert created.status_code == 201
    run_id = created.json()["id"]
    assert created.json()["backend"] == "unsloth"

    class FakeUnslothBackend:
        def train(self, request, progress, is_canceled):
            adapter = request.output_path / "adapter"
            adapter.mkdir(parents=True)
            (request.output_path / "manifest.json").write_text("{}", encoding="utf-8")
            from DashAI.back.fine_tuning.base import FineTuningResult

            return FineTuningResult(
                artifact_path=request.output_path,
                metrics={"backend": "unsloth"},
                runtime_metadata={"backend": "unsloth"},
            )

    monkeypatch.setattr(
        "DashAI.back.fine_tuning.unsloth_backend.UnslothFineTuningBackend",
        FakeUnslothBackend,
    )
    monkeypatch.setattr(
        "DashAI.back.job.fine_tuning_job.ensure_model",
        lambda _root, model_id, revision, progress=None: (
            Path(client.app.container["config"]["LLM_MODELS_PATH"])
            / model_id
            / revision,
            "sha",
        ),
    )
    response = client.post(f"/api/v1/fine-tuning/runs/{run_id}/start")
    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["metrics"]["backend"] == "unsloth"
    assert response.json()["runtime_metadata"]["backend"] == "unsloth"


def _lock(client: TestClient) -> GpuLock:
    return GpuLock(gpu_lock_path(client.app.container["config"]))


def _run_status(client: TestClient, run_id: int) -> dict:
    response = client.get(f"/api/v1/fine-tuning/runs/{run_id}")
    assert response.status_code == 200
    return response.json()


def test_start_is_rejected_while_gpu_is_locked(client: TestClient):
    dataset = _create_dataset(client)
    run_id = _create_run(client, dataset.id, "conflict run")
    lock = _lock(client)
    lock.acquire("fine_tuning_run_external")
    try:
        response = client.post(f"/api/v1/fine-tuning/runs/{run_id}/start")
        assert response.status_code == 409
        assert "GPU is busy" in response.json()["detail"]
        assert _run_status(client, run_id)["status"] == "not_started"
    finally:
        lock.release("fine_tuning_run_external")


def test_worker_fails_actionably_when_gpu_already_held(client: TestClient, monkeypatch):
    dataset = _create_dataset(client)
    run_id = _create_run(client, dataset.id, "racy run")
    lock = _lock(client)
    lock.acquire("external_training")
    monkeypatch.setattr("DashAI.back.job.fine_tuning_job.WORKER_LOCK_WAIT_SECONDS", 0.0)
    try:
        with client.app.container["session_factory"]() as db:
            run = db.get(FineTuningRun, run_id)
            run.mark_queued("fake-huey-id")
            db.commit()
        # Simulates the race: the API check passed before the lock existed.
        # The job re-raises as JobError so Huey marks the task as errored.
        with pytest.raises(JobError):
            FineTuningJob(run_id=run_id).run()
        state = _run_status(client, run_id)
        assert state["status"] == "failed"
        assert "GPU is busy" in state["error_message"]
        assert "external_training" in state["error_message"]
        # The worker never owned the lock: the foreign holder keeps it.
        assert lock.is_locked()
    finally:
        lock.release("external_training")


def test_worker_releases_lock_after_success(client: TestClient, monkeypatch):
    dataset = _create_dataset(client)
    run_id = _create_run(client, dataset.id, "lock release success")
    lock = _lock(client)
    client.app.container["fine_tuning_backend"] = SuccessfulBackend()
    monkeypatch.setattr(
        "DashAI.back.job.fine_tuning_job.ensure_model",
        lambda _root, model_id, revision, progress=None: (
            Path(client.app.container["config"]["LLM_MODELS_PATH"])
            / model_id
            / revision,
            "sha",
        ),
    )
    try:
        response = client.post(f"/api/v1/fine-tuning/runs/{run_id}/start")
        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        assert not lock.is_locked()
    finally:
        if lock.path.exists():
            lock.clear()


def test_worker_releases_lock_when_backend_fails(client: TestClient, monkeypatch):
    dataset = _create_dataset(client)
    run_id = _create_run(client, dataset.id, "lock release failure")
    lock = _lock(client)
    client.app.container["fine_tuning_backend"] = FailingBackend()
    monkeypatch.setattr(
        "DashAI.back.job.fine_tuning_job.ensure_model",
        lambda _root, model_id, revision, progress=None: (
            Path(client.app.container["config"]["LLM_MODELS_PATH"])
            / model_id
            / revision,
            "sha",
        ),
    )
    try:
        response = client.post(f"/api/v1/fine-tuning/runs/{run_id}/start")
        # In immediate mode Huey stores the job error in its result instead
        # of propagating it, so the endpoint still answers 200 with the run
        # already marked as failed.
        assert response.status_code == 200
        state = _run_status(client, run_id)
        assert state["status"] == "failed"
        assert "boom during training" in state["error_message"]
        assert not lock.is_locked()
    finally:
        if lock.path.exists():
            lock.clear()
