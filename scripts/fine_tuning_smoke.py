"""Run the complete DashAI dataset -> QLoRA -> restart -> inference smoke path."""

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path


def build_payload(dataset_id: int, name: str) -> dict:
    return {
        "name": name,
        "dataset_id": dataset_id,
        "base_model_id": "qwen2.5-0.5b-instruct",
        "base_model_revision": "main",
        "method": "qlora",
        "dataset_mapping": {
            "format": "prompt_completion",
            "prompt_column": "prompt",
            "completion_column": "completion",
            "validation_split": 0.125,
        },
        "training_parameters": {
            "preset": "quick_test",
            "max_samples": 32,
            "max_steps": 3,
            "num_train_epochs": 1.0,
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


def check(response, phase: str):
    if response.status_code >= 400:
        raise RuntimeError(f"{phase} failed ({response.status_code}): {response.text}")
    return response.json() if response.content else None


def create_dolly_dataset(local_path: Path, session_factory) -> int:
    import pyarrow as pa
    from datasets import load_dataset as load_hf_dataset
    from sqlalchemy import select

    from DashAI.back.dataloaders.classes.dashai_dataset import (
        DashAIDataset,
        save_dataset,
    )
    from DashAI.back.dependencies.database.models import Dataset

    source = load_hf_dataset(
        "databricks/databricks-dolly-15k",
        split="train",
        cache_dir=str(local_path / "huggingface-cache" / "datasets"),
    )
    source = source.shuffle(seed=42).select(range(32))
    rows = []
    for row in source:
        context = row.get("context", "").strip()
        prompt = row["instruction"].strip()
        if context:
            prompt = f"{prompt}\n\nContext:\n{context}"
        rows.append(
            {
                "prompt": prompt,
                "completion": row["response"].strip(),
                "category": row.get("category", ""),
            }
        )

    dataset_root = local_path / "datasets" / "dolly-32-seed-42"
    save_dataset(DashAIDataset(pa.Table.from_pylist(rows)), dataset_root / "dataset")
    with session_factory() as db:
        entry = db.scalar(select(Dataset).where(Dataset.name == "Dolly 32 smoke"))
        if entry is None:
            entry = Dataset(name="Dolly 32 smoke", file_path=str(dataset_root))
            db.add(entry)
        else:
            entry.file_path = str(dataset_root)
        db.commit()
        db.refresh(entry)
        return entry.id


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
    from sqlalchemy import select

    from DashAI.back.core.enums.status import FineTuningStatus
    from DashAI.back.dependencies.database.models import FineTuningRun

    with app.container["session_factory"]() as db:
        unfinished = list(
            db.scalars(
                select(FineTuningRun).where(
                    FineTuningRun.status.in_(
                        [FineTuningStatus.RUNNING, FineTuningStatus.FAILED]
                    )
                )
            )
        )
        for previous_run in unfinished:
            if previous_run.status == FineTuningStatus.RUNNING:
                previous_run.mark_failed("Interrupted before the smoke test restarted.")
            stale = local_path / "fine_tuning" / f"run-{previous_run.id}.tmp"
            if stale.exists() and stale.parent == local_path / "fine_tuning":
                shutil.rmtree(stale)
        db.commit()
    dataset_id = create_dolly_dataset(local_path, app.container["session_factory"])
    run_name = f"QLoRA smoke {datetime.now().strftime('%Y%m%d-%H%M%S')}"
    payload = build_payload(dataset_id, run_name)

    started = time.perf_counter()
    with TestClient(app) as client:
        report = check(
            client.post(
                "/api/v1/fine-tuning/preflight",
                json={**payload, "download_model": False},
            ),
            "preflight",
        )
        if not report["ready"]:
            raise RuntimeError(f"Preflight blockers: {report['blockers']}")
        run = check(client.post("/api/v1/fine-tuning/runs", json=payload), "create run")
        run = check(
            client.post(f"/api/v1/fine-tuning/runs/{run['id']}/start"),
            "start run",
        )
        if run["status"] != "completed":
            raise RuntimeError(f"Training ended with status {run['status']}: {run}")

    app.container["engine"].dispose()

    restarted = create_app(
        local_path=local_path, logging_level="INFO", enable_seeding=False
    )
    restarted.container["job_queue"].set_test_mode(True)
    with TestClient(restarted) as client:
        persisted = check(
            client.get(f"/api/v1/fine-tuning/runs/{run['id']}"),
            "read after restart",
        )
        session = check(
            client.post(
                "/api/v1/generative-session/",
                json={
                    "name": f"QLoRA smoke inference {run['id']}",
                    "description": "Created by scripts/fine_tuning_smoke.py",
                    "task_name": "TextToTextGenerationTask",
                    "model_name": "PeftAdapterTextGenerationModel",
                    "fine_tuning_run_id": run["id"],
                    "parameters": {
                        "max_new_tokens": 32,
                        "temperature": 0.0,
                        "top_p": 0.9,
                        "repetition_penalty": 1.05,
                        "context_window": 512,
                        "device": "auto",
                    },
                },
            ),
            "create generative session",
        )
        process = check(
            client.post(
                "/api/v1/generative-process/",
                data={
                    "session_id": session["id"],
                    "text_0": "Name one primary color and explain in one sentence.",
                },
            ),
            "create generative process",
        )
        check(
            client.post(
                "/api/v1/job/",
                data={
                    "job_type": "GenerativeJob",
                    "kwargs": json.dumps({"generative_process_id": process["id"]}),
                },
            ),
            "adapter inference",
        )
        generated = check(
            client.get(f"/api/v1/generative-process/{process['id']}"),
            "read generated response",
        )

    result = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds_total": round(time.perf_counter() - started, 3),
        "dataset_id": dataset_id,
        "run": persisted,
        "preflight": report,
        "session_id": session["id"],
        "process_id": process["id"],
        "generated_output": generated.get("output"),
    }
    output_path = local_path / "smoke_result.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"Smoke result written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
