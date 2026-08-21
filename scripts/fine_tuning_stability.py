"""Run a real 15-step QLoRA stability test and report numeric health.

Executes the same DashAI API path as the smoke test (preflight -> create ->
start) against an isolated local path, then prints and persists the health
instrumentation added during consolidation: per-step loss, grad_norm,
learning rate, step durations, peak VRAM, trainable dtypes and structured
health warnings. Reuses the model snapshot already downloaded by
scripts/fine_tuning_smoke.py, so no re-download is performed.
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


def build_payload(dataset_id: int, name: str, max_steps: int) -> dict:
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
            "max_samples": 64,
            "max_steps": max_steps,
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
            "save_steps": 500,
            "eval_steps": 500,
        },
    }


def check(response, phase: str):
    if response.status_code >= 400:
        raise RuntimeError(f"{phase} failed ({response.status_code}): {response.text}")
    return response.json() if response.content else None


def create_dolly_dataset(local_path: Path, session_factory, samples: int) -> int:
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
    source = source.shuffle(seed=42).select(range(samples))
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

    dataset_root = local_path / "datasets" / f"dolly-{samples}-seed-42"
    save_dataset(DashAIDataset(pa.Table.from_pylist(rows)), dataset_root / "dataset")
    name = f"Dolly {samples} stability"
    with session_factory() as db:
        entry = db.scalar(select(Dataset).where(Dataset.name == name))
        if entry is None:
            entry = Dataset(name=name, file_path=str(dataset_root))
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
    parser.add_argument("--max-steps", type=int, default=15)
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
    dataset_id = create_dolly_dataset(local_path, app.container["session_factory"], 64)
    payload = build_payload(
        dataset_id,
        f"QLoRA stability {datetime.now().strftime('%Y%m%d-%H%M%S')}",
        args.max_steps,
    )

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
        created = check(client.post("/api/v1/fine-tuning/runs", json=payload), "create")
        check(client.post(f"/api/v1/fine-tuning/runs/{created['id']}/start"), "start")
        persisted = check(
            client.get(f"/api/v1/fine-tuning/runs/{created['id']}"), "read run"
        )

    app.container["engine"].dispose()

    metrics = persisted.get("metrics") or {}
    runtime = persisted.get("runtime_metadata") or {}
    health = runtime.get("health") or {}
    last_logged = metrics.get("last_logged") or {}
    result = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "status": persisted["status"],
        "run_id": persisted["id"],
        "max_steps": args.max_steps,
        "loss": last_logged.get("loss"),
        "grad_norm": last_logged.get("grad_norm"),
        "learning_rate": last_logged.get("learning_rate"),
        "epoch": last_logged.get("epoch"),
        "final_train_loss": metrics.get("train_loss"),
        "health_warnings": metrics.get("health_warnings"),
        "step_durations_seconds": health.get("step_durations_seconds"),
        "mean_step_seconds": health.get("mean_step_seconds"),
        "peak_vram_bytes": runtime.get("peak_vram_bytes"),
        "trainable_parameter_dtypes": runtime.get("trainable_parameter_dtypes"),
        "error_message": persisted.get("error_message"),
    }
    output_path = local_path / "stability_result.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"Stability result written to {output_path}")
    return 0 if persisted["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
