from pathlib import Path

import pyarrow as pa

from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
    DatasetMapping,
    FineTuningMethod,
    PreflightRequest,
    TrainingParameters,
)
from DashAI.back.dataloaders.classes.dashai_dataset import (
    DashAIDataset,
    save_dataset,
)
from DashAI.back.dependencies.database.models import Dataset
from DashAI.back.fine_tuning.preflight import DEPENDENCIES, run_preflight


def _dataset(tmp_path: Path) -> Dataset:
    path = tmp_path / "dataset"
    save_dataset(
        DashAIDataset(pa.table({"text": ["one", "two", "three", "four"]})),
        path / "dataset",
    )
    return Dataset(id=1, name="preflight", file_path=str(path))


def _request(method: FineTuningMethod = FineTuningMethod.QLORA) -> PreflightRequest:
    return PreflightRequest(
        dataset_id=1,
        base_model_id="qwen2.5-0.5b-instruct",
        base_model_revision="main",
        method=method,
        dataset_mapping=DatasetMapping(
            format="text", text_column="text", validation_split=0.25
        ),
        training_parameters=TrainingParameters(preset="quick_test", max_samples=4),
        download_model=False,
    )


def _versions(missing: str | None = None) -> dict[str, str | None]:
    return {
        dependency: None if dependency == missing else "test"
        for dependency in DEPENDENCIES
    }


def test_preflight_reports_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.dependency_versions",
        lambda: _versions("peft"),
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.hardware_info",
        lambda: {
            "cuda_available": True,
            "device": "test",
            "vram_gb": 8,
            "compute_capability": [6, 1],
        },
    )

    report = run_preflight(_request(), _dataset(tmp_path), tmp_path / "models")

    assert report.ready is False
    assert {issue.code for issue in report.blockers} == {"missing_dependency"}


def test_qlora_preflight_rejects_cpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.dependency_versions", lambda: _versions()
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.hardware_info",
        lambda: {"cuda_available": False, "device": "cpu", "vram_gb": 0},
    )

    report = run_preflight(_request(), _dataset(tmp_path), tmp_path / "models")

    assert report.ready is False
    assert "qlora_requires_cuda" in {issue.code for issue in report.blockers}


def test_qlora_preflight_rejects_old_compute_capability(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.dependency_versions", lambda: _versions()
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.hardware_info",
        lambda: {
            "cuda_available": True,
            "device": "old GPU",
            "vram_gb": 8,
            "compute_capability": [5, 2],
        },
    )

    report = run_preflight(_request(), _dataset(tmp_path), tmp_path / "models")

    assert report.ready is False
    assert "unsupported_compute_capability" in {issue.code for issue in report.blockers}


def test_pascal_qlora_preflight_is_ready_and_warns_about_download(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.dependency_versions", lambda: _versions()
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.hardware_info",
        lambda: {
            "cuda_available": True,
            "device": "GTX 1080",
            "vram_gb": 8,
            "compute_capability": [6, 1],
        },
    )

    report = run_preflight(_request(), _dataset(tmp_path), tmp_path / "models")

    assert report.ready is True
    assert report.train_rows == 3
    assert report.validation_rows == 1
    assert "model_will_download" in {issue.code for issue in report.warnings}


def test_unsloth_preflight_blocks_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.dependency_versions",
        lambda: _versions(),
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.hardware_info",
        lambda: {
            "cuda_available": True,
            "device": "test",
            "vram_gb": 8,
            "compute_capability": [6, 1],
        },
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.unsloth_installed",
        lambda: (False, None),
    )

    request = _request()
    request.backend = "unsloth"
    report = run_preflight(request, _dataset(tmp_path), tmp_path / "models")

    assert report.ready is False
    assert "unsloth_not_installed" in {issue.code for issue in report.blockers}


def test_unsloth_preflight_warns_on_pascal_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.dependency_versions",
        lambda: _versions(),
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.hardware_info",
        lambda: {
            "cuda_available": True,
            "device": "GTX 1080",
            "vram_gb": 8,
            "compute_capability": [6, 1],
        },
    )
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.preflight.unsloth_installed",
        lambda: (True, "2026.8.19"),
    )

    request = _request(method=FineTuningMethod.LORA)
    request.backend = "unsloth"
    report = run_preflight(request, _dataset(tmp_path), tmp_path / "models")

    assert report.ready is True
    assert "unsloth_experimental_gpu" in {issue.code for issue in report.warnings}
