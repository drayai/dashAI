import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from DashAI.back.fine_tuning.catalog import resolve_model


def _safe_revision(revision: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", revision)[:80]


def model_directory(models_root: Path, model_id: str, revision: str) -> Path:
    resolve_model(model_id)
    return models_root / model_id / _safe_revision(revision)


def _credential_token() -> str | None:
    try:
        from DashAI.back.credentials.huggingface_credential import (
            HuggingFaceCredential,
        )

        return HuggingFaceCredential().get_key()
    except Exception:
        return os.getenv("HF_TOKEN")


def ensure_model(
    models_root: Path,
    model_id: str,
    revision: str,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Path, str]:
    """Download a curated model into DashAI-owned storage if necessary."""
    from huggingface_hub import HfApi, snapshot_download

    model = resolve_model(model_id)
    destination = model_directory(models_root, model_id, revision)
    metadata_path = destination / ".dashai_model.json"
    if metadata_path.exists() and (destination / "config.json").exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return destination, metadata["resolved_revision"]

    token = _credential_token()
    if progress:
        progress(0.03, f"Resolving {model['repository']}")
    info = HfApi().model_info(model["repository"], revision=revision, token=token)
    resolved_revision = info.sha

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.download")
    if temporary.exists():
        shutil.rmtree(temporary)
    if progress:
        progress(0.05, f"Downloading {model['name']}")
    snapshot_download(
        repo_id=model["repository"],
        revision=resolved_revision,
        local_dir=temporary,
        token=token,
    )
    (temporary / ".dashai_model.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "repository": model["repository"],
                "requested_revision": revision,
                "resolved_revision": resolved_revision,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return destination, resolved_revision


def read_model_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path / ".dashai_model.json"
    if not metadata_path.exists():
        return None
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def directory_size(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def ensure_managed_path(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Path is outside DashAI-managed storage: {resolved_path}")
    return resolved_path
