import importlib.metadata
from pathlib import Path
from typing import Any

from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
    FineTuningMethod,
    PreflightIssue,
    PreflightReport,
    PreflightRequest,
)
from DashAI.back.core.enums.status import FineTuningBackendType
from DashAI.back.dependencies.database.models import Dataset
from DashAI.back.fine_tuning.catalog import resolve_model
from DashAI.back.fine_tuning.dataset import prepare_dataset
from DashAI.back.fine_tuning.model_store import (
    ensure_model,
    model_directory,
    read_model_metadata,
)
from DashAI.back.fine_tuning.unsloth_backend import (
    NOT_INSTALLED_MESSAGE,
    unsloth_installed,
)

DEPENDENCIES = ("transformers", "datasets", "accelerate", "trl", "peft", "bitsandbytes")


def dependency_versions() -> dict[str, str | None]:
    versions = {}
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def hardware_info() -> dict[str, Any]:
    import torch

    hardware: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        hardware.update(
            {
                "device": properties.name,
                "vram_gb": round(properties.total_memory / 1024**3, 2),
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "cuda_version": torch.version.cuda,
            }
        )
    else:
        hardware.update({"device": "cpu", "vram_gb": 0.0})
    return hardware


def run_preflight(
    request: PreflightRequest,
    dataset: Dataset,
    models_root: Path,
) -> PreflightReport:
    blockers: list[PreflightIssue] = []
    warnings: list[PreflightIssue] = []
    preview = []
    fingerprint = None
    source_rows = train_rows = validation_rows = 0

    try:
        model = resolve_model(request.base_model_id)
    except ValueError as exc:
        blockers.append(PreflightIssue(code="unsupported_model", message=str(exc)))
        model = None

    versions = dependency_versions()
    required = ["transformers", "datasets", "accelerate", "trl", "peft"]
    if request.method == FineTuningMethod.QLORA:
        required.append("bitsandbytes")
    for package in required:
        if versions[package] is None:
            blockers.append(
                PreflightIssue(
                    code="missing_dependency",
                    message=f"Required dependency '{package}' is not installed.",
                )
            )

    hardware = hardware_info()
    if request.backend == FineTuningBackendType.UNSLOTH:
        installed, _unsloth_version = unsloth_installed()
        if not installed:
            blockers.append(
                PreflightIssue(
                    code="unsloth_not_installed",
                    message=NOT_INSTALLED_MESSAGE,
                )
            )
        else:
            compute_capability = tuple(hardware.get("compute_capability", (0, 0)))
            if hardware["cuda_available"] and compute_capability < (7, 0):
                warnings.append(
                    PreflightIssue(
                        code="unsloth_experimental_gpu",
                        message=(
                            "Unsloth lists compute capability 7.0 as its "
                            "general requirement; this GPU reports "
                            f"{compute_capability}. Pascal GPUs are treated "
                            "as experimental and were not validated in this "
                            "environment."
                        ),
                    )
                )
    if request.method == FineTuningMethod.QLORA:
        if not hardware["cuda_available"]:
            blockers.append(
                PreflightIssue(
                    code="qlora_requires_cuda",
                    message="QLoRA requires a CUDA-capable NVIDIA GPU.",
                )
            )
        elif tuple(hardware.get("compute_capability", (0, 0))) < (6, 0):
            blockers.append(
                PreflightIssue(
                    code="unsupported_compute_capability",
                    message="NF4 requires NVIDIA compute capability 6.0 or newer.",
                )
            )
    elif not hardware["cuda_available"]:
        warnings.append(
            PreflightIssue(
                code="cpu_training",
                message="CPU LoRA is supported for diagnostics but will be slow.",
            )
        )

    try:
        prepared = prepare_dataset(
            dataset.file_path,
            request.dataset_mapping,
            request.training_parameters,
        )
        preview = prepared.preview
        fingerprint = prepared.fingerprint
        source_rows = prepared.source_rows
        train_rows = len(prepared.train)
        validation_rows = (
            len(prepared.validation) if prepared.validation is not None else 0
        )
    except Exception as exc:
        blockers.append(PreflightIssue(code="invalid_dataset", message=str(exc)))

    model_path = None
    resolved_revision = None
    downloaded = False
    if model:
        path = model_directory(
            models_root, request.base_model_id, request.base_model_revision
        )
        downloaded = (path / ".dashai_model.json").exists()
        if downloaded:
            metadata = read_model_metadata(path) or {}
            resolved_revision = metadata.get("resolved_revision")
        if request.download_model and not downloaded and not blockers:
            try:
                path, resolved_revision = ensure_model(
                    models_root,
                    request.base_model_id,
                    request.base_model_revision,
                )
                downloaded = True
            except Exception as exc:
                blockers.append(
                    PreflightIssue(code="model_download_failed", message=str(exc))
                )
        if downloaded:
            model_path = str(path)
        else:
            warnings.append(
                PreflightIssue(
                    code="model_will_download",
                    message="The base model will be downloaded when training starts.",
                )
            )
        if hardware.get("vram_gb", 0) < model["recommended_vram_gb"]:
            warnings.append(
                PreflightIssue(
                    code="limited_vram",
                    message=(
                        "This model recommends "
                        f"{model['recommended_vram_gb']} GB VRAM; "
                        f"{hardware.get('vram_gb', 0)} GB was detected."
                    ),
                )
            )

    return PreflightReport(
        ready=not blockers,
        blockers=blockers,
        warnings=warnings,
        preview=preview,
        dataset_fingerprint=fingerprint,
        dataset_rows=source_rows,
        train_rows=train_rows,
        validation_rows=validation_rows,
        model_path=model_path,
        model_downloaded=downloaded,
        resolved_model_revision=resolved_revision,
        hardware=hardware,
        dependencies=versions,
        estimated_vram_gb=(model["recommended_vram_gb"] if model else None),
    )
