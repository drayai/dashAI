"""Compare base-model and fine-tuned-adapter inference on identical prompts.

Generates with the managed base model and with the adapter from a completed
run (default: the latest one), using the same sampling parameters, and
records both outputs plus timings as evidence for the findings document.
This is a qualitative integration comparison, not a quality benchmark.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

PROMPTS = [
    "Name one primary color and explain in one sentence.",
    "Explain what an API is in one sentence.",
    "Give one tip for writing clean code.",
]


def check(response, phase: str):
    if response.status_code >= 400:
        raise RuntimeError(f"{phase} failed ({response.status_code}): {response.text}")
    return response.json() if response.content else None


def generate(client, session: dict, prompt: str) -> tuple[str, float]:
    started = time.perf_counter()
    process = check(
        client.post(
            "/api/v1/generative-process/",
            data={"session_id": session["id"], "text_0": prompt},
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
        "run job",
    )
    generated = check(
        client.get(f"/api/v1/generative-process/{process['id']}"), "read output"
    )
    outputs = [item for item in generated.get("output", []) if not item.get("is_input")]
    text = outputs[0]["data"] if outputs else ""
    return text, round(time.perf_counter() - started, 2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-path", type=Path, default=Path("E:/DashAI-smoke-finetuning")
    )
    parser.add_argument("--run-id", type=int, default=None)
    args = parser.parse_args()
    local_path = args.local_path.resolve()
    os.environ["DASHAI_LOCAL_PATH"] = str(local_path)
    os.environ["HF_HOME"] = str(local_path / "huggingface-cache")
    os.environ["HF_DATASETS_CACHE"] = str(local_path / "huggingface-cache" / "datasets")

    from fastapi.testclient import TestClient

    from DashAI.back.app import create_app

    app = create_app(local_path=local_path, logging_level="INFO", enable_seeding=False)
    app.container["job_queue"].set_test_mode(True)

    with TestClient(app) as client:
        runs = check(client.get("/api/v1/fine-tuning/runs"), "list runs")
        completed = [
            run for run in runs if run["status"] == "completed" and run["artifact_path"]
        ]
        if not completed:
            raise RuntimeError("No completed fine-tuning runs to compare.")
        run = next(
            (r for r in completed if args.run_id is None or r["id"] == args.run_id),
            completed[0],
        )

        inventory = {
            item["key"]: item
            for item in check(client.get("/api/v1/fine-tuning/models"), "inventory")
        }
        base_entry = next(
            (
                item
                for key, item in inventory.items()
                if item["kind"] == "base"
                and item["status"] == "ready"
                and item["source"] == "Qwen/Qwen2.5-0.5B-Instruct"
            ),
            None,
        )
        if base_entry is None:
            raise RuntimeError("The base model is not ready in the inventory.")

        parameters = {
            "max_new_tokens": 48,
            "temperature": 0.0,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "seed": 42,
            "context_window": 512,
            "device": "auto",
        }
        base_session = check(
            client.post(
                "/api/v1/generative-session/",
                json={
                    "name": f"base comparison {datetime.now().strftime('%H%M%S')}",
                    "description": "Base model comparison",
                    "task_name": "TextToTextGenerationTask",
                    "model_name": "LocalManagedTextGenerationModel",
                    "local_model_id": base_entry["local_model_id"],
                    "parameters": parameters,
                },
            ),
            "create base session",
        )
        adapter_session = check(
            client.post(
                "/api/v1/generative-session/",
                json={
                    "name": f"adapter comparison {datetime.now().strftime('%H%M%S')}",
                    "description": f"Adapter from run {run['id']}",
                    "task_name": "TextToTextGenerationTask",
                    "model_name": "PeftAdapterTextGenerationModel",
                    "fine_tuning_run_id": run["id"],
                    "parameters": parameters,
                },
            ),
            "create adapter session",
        )

        results = []
        for prompt in PROMPTS:
            base_text, base_seconds = generate(client, base_session, prompt)
            adapter_text, adapter_seconds = generate(client, adapter_session, prompt)
            results.append(
                {
                    "prompt": prompt,
                    "base": {"text": base_text, "seconds": base_seconds},
                    "adapter": {"text": adapter_text, "seconds": adapter_seconds},
                }
            )

    app.container["engine"].dispose()
    report = {
        "completed_at": datetime.now().isoformat(),
        "run_id": run["id"],
        "run_name": run["name"],
        "parameters": parameters,
        "comparisons": results,
    }
    output_path = local_path / "base_vs_adapter_result.json"
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"Comparison written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
