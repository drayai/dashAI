"""Validate base-model inference from the fine-tuning managed inventory.

Runs the Milestone 2 acceptance path against an isolated local path with a
model snapshot already on disk: adopt it through the download endpoint,
create a Generative session backed by LocalManagedTextGenerationModel, and
generate one response.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


def check(response, phase: str):
    if response.status_code >= 400:
        raise RuntimeError(f"{phase} failed ({response.status_code}): {response.text}")
    return response.json() if response.content else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-path", type=Path, default=Path("E:/DashAI-smoke-finetuning")
    )
    args = parser.parse_args()
    local_path = args.local_path.resolve()
    local_path.mkdir(parents=True, exist_ok=True)
    os.environ["DASHAI_LOCAL_PATH"] = str(local_path)
    os.environ["HF_HOME"] = str(local_path / "huggingface-cache")
    os.environ["HF_DATASETS_CACHE"] = str(local_path / "huggingface-cache" / "datasets")

    from fastapi.testclient import TestClient

    from DashAI.back.app import create_app

    app = create_app(local_path=local_path, logging_level="INFO", enable_seeding=False)
    app.container["job_queue"].set_test_mode(True)
    started = time.perf_counter()

    inventory_key = "base:qwen2.5-0.5b-instruct:main"
    with TestClient(app) as client:
        adopted = check(
            client.post(f"/api/v1/fine-tuning/models/{inventory_key}/download"),
            "adopt/download base model",
        )
        local_model_id = adopted["local_model_id"]

        inventory = {
            item["key"]: item
            for item in check(client.get("/api/v1/fine-tuning/models"), "inventory")
        }
        assert inventory[inventory_key]["status"] == "ready"

        session = check(
            client.post(
                "/api/v1/generative-session/",
                json={
                    "name": f"Base model inference {datetime.now().strftime('%H%M%S')}",
                    "description": "Created by scripts/local_model_inference_check.py",
                    "task_name": "TextToTextGenerationTask",
                    "model_name": "LocalManagedTextGenerationModel",
                    "local_model_id": local_model_id,
                    "parameters": {
                        "max_new_tokens": 32,
                        "temperature": 0.0,
                        "top_p": 0.9,
                        "top_k": 50,
                        "repetition_penalty": 1.05,
                        "seed": 42,
                        "context_window": 512,
                        "device": "auto",
                    },
                },
            ),
            "create session",
        )
        process = check(
            client.post(
                "/api/v1/generative-process/",
                data={
                    "session_id": session["id"],
                    "text_0": "Name one primary color and explain in one sentence.",
                },
            ),
            "create process",
        )
        check(
            client.post(
                "/api/v1/job/",
                data={
                    "job_type": "GenerativeJob",
                    "kwargs": json.dumps({"generative_process_id": process["id"]}),
                },
            ),
            "base model inference",
        )
        generated = check(
            client.get(f"/api/v1/generative-process/{process['id']}"),
            "read response",
        )

    app.container["engine"].dispose()
    result = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "local_model_id": local_model_id,
        "session_id": session["id"],
        "generated_output": generated.get("output"),
    }
    output_path = local_path / "base_inference_result.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"Result written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
