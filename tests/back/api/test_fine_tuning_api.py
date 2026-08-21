import json
import uuid
from pathlib import Path

import pyarrow as pa
from fastapi.testclient import TestClient

from DashAI.back.core.enums.status import FineTuningStatus
from DashAI.back.dataloaders.classes.dashai_dataset import (
    DashAIDataset,
    save_dataset,
)
from DashAI.back.dependencies.database.models import Dataset, FineTuningRun
from DashAI.back.fine_tuning.base import FineTuningResult


class FakeBackend:
    def train(self, request, progress, is_canceled):
        assert not is_canceled()
        progress(0.5, "Fake training", {"loss": 1.0})
        adapter = request.output_path / "adapter"
        adapter.mkdir(parents=True)
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (request.output_path / "manifest.json").write_text(
            json.dumps({"run_id": request.run_id}), encoding="utf-8"
        )
        return FineTuningResult(
            artifact_path=request.output_path,
            metrics={"loss": 0.5},
            runtime_metadata={"duration_seconds": 0.01},
        )


def _create_dataset(client: TestClient) -> Dataset:
    container = client.app.container
    root = Path(container["config"]["DATASETS_PATH"]) / "fine-tuning-test"
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
            name=f"fine_tuning_dataset_{uuid.uuid4().hex}", file_path=str(root)
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        db.expunge(entry)
        return entry


def _payload(dataset_id: int) -> dict:
    return {
        "name": "API fine-tuning prototype",
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
        "training_parameters": {
            "preset": "quick_test",
            "max_samples": 4,
            "max_steps": 3,
            "num_train_epochs": 1,
            "max_length": 128,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.0002,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "seed": 42,
            "logging_steps": 1,
            "save_steps": 1,
            "eval_steps": 1,
        },
    }


def test_fine_tuning_crud_job_inventory_and_generative_bridge(
    client: TestClient, monkeypatch
):
    dataset = _create_dataset(client)
    payload = _payload(dataset.id)

    catalog = client.get("/api/v1/fine-tuning/catalog")
    assert catalog.status_code == 200
    assert catalog.json()["capabilities"]["unsloth"] is False

    preflight = client.post(
        "/api/v1/fine-tuning/preflight", json={**payload, "download_model": False}
    )
    assert preflight.status_code == 200
    assert preflight.json()["ready"] is True
    assert preflight.json()["train_rows"] == 3

    created = client.post("/api/v1/fine-tuning/runs", json=payload)
    assert created.status_code == 201
    run_id = created.json()["id"]
    duplicate = client.post("/api/v1/fine-tuning/runs", json=payload)
    assert duplicate.status_code == 409

    models_root = Path(client.app.container["config"]["LLM_MODELS_PATH"])

    def fake_model(_root, model_id, revision, progress=None):
        path = models_root / model_id / revision
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / ".dashai_model.json").write_text(
            json.dumps(
                {
                    "model_id": model_id,
                    "repository": "Qwen/Qwen2.5-0.5B-Instruct",
                    "requested_revision": revision,
                    "resolved_revision": "test-sha",
                }
            ),
            encoding="utf-8",
        )
        return path, "test-sha"

    monkeypatch.setattr("DashAI.back.job.fine_tuning_job.ensure_model", fake_model)
    client.app.container["fine_tuning_backend"] = FakeBackend()
    started = client.post(f"/api/v1/fine-tuning/runs/{run_id}/start")
    assert started.status_code == 200
    assert started.json()["status"] == "completed"
    assert started.json()["metrics"]["loss"] == 0.5

    downloaded_preflight = client.post(
        "/api/v1/fine-tuning/preflight", json={**payload, "download_model": False}
    )
    assert downloaded_preflight.status_code == 200
    assert downloaded_preflight.json()["model_downloaded"] is True
    assert downloaded_preflight.json()["resolved_model_revision"] == "test-sha"

    inventory = client.get("/api/v1/fine-tuning/models")
    assert inventory.status_code == 200
    assert {item["kind"] for item in inventory.json()} == {"base", "adapter"}

    session = client.post(
        "/api/v1/generative-session/",
        json={
            "model_name": "PeftAdapterTextGenerationModel",
            "task_name": "TextToTextGenerationTask",
            "parameters": {
                "max_new_tokens": 32,
                "temperature": 0.0,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
                "context_window": 256,
                "device": "cpu",
            },
            "name": "fine-tuned session",
            "description": None,
            "fine_tuning_run_id": run_id,
        },
    )
    assert session.status_code == 201
    assert session.json()["fine_tuning_run_id"] == run_id
    assert client.delete(f"/api/v1/fine-tuning/runs/{run_id}").status_code == 409

    assert (
        client.delete(f"/api/v1/generative-session/{session.json()['id']}").status_code
        == 204
    )
    assert client.delete(f"/api/v1/fine-tuning/runs/{run_id}").status_code == 204


def test_running_run_uses_cooperative_cancellation(client: TestClient):
    dataset = _create_dataset(client)
    payload = _payload(dataset.id)
    payload["name"] = "Cancelable run"
    created = client.post("/api/v1/fine-tuning/runs", json=payload).json()
    with client.app.container["session_factory"]() as db:
        run = db.get(FineTuningRun, created["id"])
        run.status = FineTuningStatus.RUNNING
        db.commit()
    canceled = client.post(f"/api/v1/fine-tuning/runs/{created['id']}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["cancellation_requested"] is True
