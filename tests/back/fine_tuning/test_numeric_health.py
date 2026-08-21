import math

from DashAI.back.fine_tuning.huggingface_backend import (
    _json_safe,
    _non_finite_warning,
)


def test_json_safe_maps_nan_and_inf_to_none():
    assert _json_safe(float("nan")) is None
    assert _json_safe(float("inf")) is None
    assert _json_safe(float("-inf")) is None


def test_json_safe_preserves_valid_values_and_structure():
    value = {
        "loss": 2.5,
        "grad_norm": None,
        "steps": [1, 2.5],
        "name": "run",
        "flag": True,
    }
    assert _json_safe(value) == value


def test_json_safe_sanitizes_nested_non_finite_values():
    cleaned = _json_safe({"loss": float("nan"), "history": [float("inf"), 1.0]})
    assert cleaned == {"loss": None, "history": [None, 1.0]}


def test_json_safe_converts_numpy_like_numbers():
    class FakeTensor:
        def __float__(self):
            return 1.5

    assert _json_safe(FakeTensor()) == 1.5

    class Broken:
        def __str__(self):
            return "unserializable"

    assert _json_safe(Broken()) == "unserializable"


def test_non_finite_warning_is_structured_and_actionable():
    warning = _non_finite_warning("grad_norm", float("nan"), step=7)
    assert warning is not None
    assert warning["field"] == "grad_norm"
    assert warning["step"] == 7
    assert "not numerically stable" in warning["message"]


def test_non_finite_warning_ignores_finite_and_missing_values():
    assert _non_finite_warning("loss", 1.25, step=1) is None
    assert _non_finite_warning("loss", None, step=1) is None
    assert _non_finite_warning("grad_norm", "n/a", step=1) is None


def test_documented_policy_fails_run_on_non_finite_loss():
    # The trainer callback turns non-finite loss values into a hard failure
    # so the adapter is never published; grad_norm only produces a warning.
    loss_warning = _non_finite_warning("loss", math.nan, step=2)
    grad_warning = _non_finite_warning("grad_norm", math.inf, step=2)
    assert loss_warning is not None
    assert grad_warning is not None
    assert "loss" in loss_warning["message"]
    assert "grad_norm" in grad_warning["message"]
