from pathlib import Path

import pytest

from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
    DatasetMapping,
    FineTuningMethod,
    FineTuningRunCreate,
    TrainingParameters,
)
from DashAI.back.core.enums.status import FineTuningBackendType
from DashAI.back.fine_tuning.base import FineTuningRequest
from DashAI.back.fine_tuning.catalog import get_catalog
from DashAI.back.fine_tuning.dataset import PreparedDataset
from DashAI.back.fine_tuning.unsloth_backend import (
    NOT_INSTALLED_MESSAGE,
    UnslothFineTuningBackend,
    unsloth_capabilities,
)


def _minimal_request(tmp_path: Path) -> FineTuningRequest:
    return FineTuningRequest(
        run_id=1,
        model_id="qwen2.5-0.5b-instruct",
        model_revision="main",
        method=FineTuningMethod.LORA,
        mapping=DatasetMapping(format="text", text_column="text"),
        parameters=TrainingParameters(),
        dataset=PreparedDataset(
            train=[], validation=None, preview=[], fingerprint="f", source_rows=0
        ),
        model_path=tmp_path,
        output_path=tmp_path / "run-1",
        resolved_revision="sha",
    )


def test_catalog_reports_unsloth_unavailable_with_reason(monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.unsloth_backend.unsloth_installed",
        lambda: (False, None),
    )
    capabilities = get_catalog()["capabilities"]
    assert capabilities["unsloth"] is False
    assert "not installed" in capabilities["unsloth_reason"]


def test_catalog_reports_unsloth_available_when_installed(monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.unsloth_backend.unsloth_installed",
        lambda: (True, "2026.8.19"),
    )
    capabilities = get_catalog()["capabilities"]
    assert capabilities["unsloth"] is True
    assert capabilities["unsloth_version"] == "2026.8.19"
    assert capabilities["unsloth_reason"] is None


def test_backend_default_is_transformers():
    payload = FineTuningRunCreate(
        name="defaults",
        dataset_id=1,
        base_model_id="qwen2.5-0.5b-instruct",
        dataset_mapping={"format": "text", "text_column": "text"},
    )
    assert payload.backend == FineTuningBackendType.TRANSFORMERS


def test_backend_enum_values_are_stable():
    assert [value.value for value in FineTuningBackendType] == [
        "transformers",
        "unsloth",
    ]


def test_unsloth_backend_fails_clearly_when_not_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.unsloth_backend.unsloth_installed",
        lambda: (False, None),
    )
    backend = UnslothFineTuningBackend()
    with pytest.raises(RuntimeError) as excinfo:
        backend.train(_minimal_request(tmp_path), lambda *a: None, lambda: False)
    assert NOT_INSTALLED_MESSAGE in str(excinfo.value)


def test_capabilities_shape_matches_catalog_contract(monkeypatch):
    monkeypatch.setattr(
        "DashAI.back.fine_tuning.unsloth_backend.unsloth_installed",
        lambda: (False, None),
    )
    capabilities = unsloth_capabilities()
    assert set(capabilities) == {"available", "version", "reason"}
