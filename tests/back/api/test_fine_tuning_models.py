import json
from pathlib import Path

from fastapi.testclient import TestClient

MODEL_KEY = "qwen2.5-0.5b-instruct"
INVENTORY_KEY = f"base:{MODEL_KEY}:main"
# A second catalog model no other test downloads: its entry must always be
# reported as not downloaded, even with leftover state in the shared tmp dir.
PRISTINE_KEY = "base:qwen2.5-1.5b-instruct:main"

SESSION_PARAMETERS = {
    "max_new_tokens": 16,
    "temperature": 0.0,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
    "context_window": 256,
    "device": "cpu",
}


def _inventory(client: TestClient) -> dict:
    response = client.get("/api/v1/fine-tuning/models")
    assert response.status_code == 200
    return {item["key"]: item for item in response.json()}


def _fake_download(client: TestClient, monkeypatch):
    """Patch the download job to materialize a snapshot without the hub."""
    models_root = Path(client.app.container["config"]["LLM_MODELS_PATH"])

    def fake_model(_root, model_key, revision, progress=None):
        destination = models_root / model_key / revision
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text("{}", encoding="utf-8")
        (destination / ".dashai_model.json").write_text(
            json.dumps(
                {
                    "model_id": model_key,
                    "repository": "Qwen/Qwen2.5-0.5B-Instruct",
                    "requested_revision": revision,
                    "resolved_revision": "fake-sha",
                }
            ),
            encoding="utf-8",
        )
        return destination, "fake-sha"

    monkeypatch.setattr("DashAI.back.job.managed_model_job.ensure_model", fake_model)


def test_inventory_lists_catalog_models_not_downloaded(client: TestClient):
    inventory = _inventory(client)
    entry = inventory[PRISTINE_KEY]
    assert entry["status"] == "not_downloaded"
    assert entry["downloadable"] is True
    assert entry["path"] is None
    assert entry["recommended_vram_gb"] == 8
    assert entry["kind"] == "base"
    assert INVENTORY_KEY in inventory


def test_download_model_tracks_state_and_blocks_repeats(
    client: TestClient, monkeypatch
):
    _fake_download(client, monkeypatch)

    started = client.post(f"/api/v1/fine-tuning/models/{INVENTORY_KEY}/download")
    assert started.status_code == 202
    local_model_id = started.json()["local_model_id"]

    inventory = _inventory(client)
    entry = inventory[INVENTORY_KEY]
    assert entry["status"] == "ready"
    assert entry["local_model_id"] == local_model_id
    assert entry["downloadable"] is False
    assert entry["size_bytes"] > 0
    assert entry["path"] is not None

    repeated = client.post(f"/api/v1/fine-tuning/models/{INVENTORY_KEY}/download")
    assert repeated.status_code == 409
    assert "already" in repeated.json()["detail"]


def _ensure_downloaded(client: TestClient) -> int:
    """Download the base model if needed and return its local_model_id."""
    entry = _inventory(client)[INVENTORY_KEY]
    if entry["status"] == "ready" and entry["local_model_id"]:
        return entry["local_model_id"]
    started = client.post(f"/api/v1/fine-tuning/models/{INVENTORY_KEY}/download")
    assert started.status_code == 202
    return started.json()["local_model_id"]


def test_delete_managed_model_respects_sessions(client: TestClient, monkeypatch):
    _fake_download(client, monkeypatch)
    local_model_id = _ensure_downloaded(client)

    session = client.post(
        "/api/v1/generative-session/",
        json={
            "model_name": "LocalManagedTextGenerationModel",
            "task_name": "TextToTextGenerationTask",
            "parameters": SESSION_PARAMETERS,
            "name": "base model session",
            "description": None,
            "local_model_id": local_model_id,
        },
    )
    assert session.status_code == 201
    assert session.json()["local_model_id"] == local_model_id

    blocked = client.delete(f"/api/v1/fine-tuning/models/{INVENTORY_KEY}")
    assert blocked.status_code == 409
    assert "generative session" in blocked.json()["detail"]

    assert (
        client.delete(f"/api/v1/generative-session/{session.json()['id']}").status_code
        == 204
    )
    deleted = client.delete(f"/api/v1/fine-tuning/models/{INVENTORY_KEY}")
    assert deleted.status_code == 204
    inventory = _inventory(client)
    assert inventory[INVENTORY_KEY]["status"] == "not_downloaded"


def test_session_creation_validates_local_model_references(client: TestClient):
    missing = client.post(
        "/api/v1/generative-session/",
        json={
            "model_name": "LocalManagedTextGenerationModel",
            "task_name": "TextToTextGenerationTask",
            "parameters": SESSION_PARAMETERS,
            "name": "missing local model",
            "description": None,
            "local_model_id": 999999,
        },
    )
    assert missing.status_code == 404

    wrong_model = client.post(
        "/api/v1/generative-session/",
        json={
            "model_name": "Qwen25_05BInstruct",
            "task_name": "TextToTextGenerationTask",
            "parameters": {},
            "name": "wrong model for local id",
            "description": None,
            "local_model_id": 1,
        },
    )
    assert wrong_model.status_code == 400

    without_id = client.post(
        "/api/v1/generative-session/",
        json={
            "model_name": "LocalManagedTextGenerationModel",
            "task_name": "TextToTextGenerationTask",
            "parameters": SESSION_PARAMETERS,
            "name": "local model without id",
            "description": None,
        },
    )
    assert without_id.status_code == 400


def test_managed_model_injection_in_generative_job(client: TestClient, monkeypatch):
    from DashAI.back.models.base_generative_model import BaseGenerativeModel
    from DashAI.back.models.hugging_face.local_managed_text_generation_model import (
        LocalManagedTextGenerationModel,
    )

    captured = {}

    class FakeLocalModel(BaseGenerativeModel):
        # The real schema satisfies session validation; instantiation only
        # records the kwargs injected by GenerativeJob.
        SCHEMA = LocalManagedTextGenerationModel.SCHEMA

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def generate(self, prompt):
            return ["managed output"]

    # The registry is wrapped per-test and restored through the proxy's
    # delegate, mirroring test_generative_session_download_gate patterns.
    real_registry = client.app.container["component_registry"]

    class RegistryProxy:
        def __getitem__(self, name):
            if name == "LocalManagedTextGenerationModel":
                return {"class": FakeLocalModel}
            return real_registry[name]

        def __getattr__(self, attr):
            return getattr(real_registry, attr)

    client.app.container["component_registry"] = RegistryProxy()

    _fake_download(client, monkeypatch)
    local_model_id = _ensure_downloaded(client)

    session = client.post(
        "/api/v1/generative-session/",
        json={
            "model_name": "LocalManagedTextGenerationModel",
            "task_name": "TextToTextGenerationTask",
            "parameters": SESSION_PARAMETERS,
            "name": "injection session",
            "description": None,
            "local_model_id": local_model_id,
        },
    )
    assert session.status_code == 201
    session_id = session.json()["id"]

    process = client.post(
        "/api/v1/generative-process/",
        data={"session_id": session_id, "text_0": "Say something."},
    )
    assert process.status_code == 201

    job = client.post(
        "/api/v1/job/",
        data={
            "job_type": "GenerativeJob",
            "kwargs": json.dumps({"generative_process_id": process.json()["id"]}),
        },
    )
    assert job.status_code == 201

    assert "_base_model_path" in captured
    base_path = captured["_base_model_path"].replace("\\", "/")
    assert base_path.endswith(f"{MODEL_KEY}/main")
    assert "llm_models" in base_path

    generated = client.get(f"/api/v1/generative-process/{process.json()['id']}")
    assert generated.status_code == 200
    assert generated.json()["output"]

    client.app.container["component_registry"] = real_registry
    assert issubclass(
        real_registry["LocalManagedTextGenerationModel"]["class"],
        BaseGenerativeModel,
    )
