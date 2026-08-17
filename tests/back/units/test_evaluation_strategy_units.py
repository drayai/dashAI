"""Tests for EvaluationStrategy's shared validate()/_resolve_search contract,
independent of an actual training run.

FitModelUnit no longer exists: HoldoutUnit and CrossValidationUnit replaced
it, both inheriting this shared behavior from EvaluationStrategy. These tests
exercise that shared base directly through a minimal concrete subclass,
rather than through either strategy, so they stay valid regardless of what
holdout- or CV-specific requirements get added to those subclasses later.
"""

import pytest
from kink import di

from DashAI.back.job.base_job import JobError
from DashAI.back.units.context import ExecutionContext, UnitContractError
from DashAI.back.units.cross_validation_unit import CrossValidationUnit
from DashAI.back.units.evaluation_strategy import EvaluationStrategy
from DashAI.back.units.holdout_unit import HoldoutUnit


class _ConcreteStrategy(EvaluationStrategy):
    """The minimal subclass needed to instantiate EvaluationStrategy.

    Deliberately does nothing beyond satisfying the abstract methods: these
    tests are about validate()/_resolve_search(), not about any strategy's
    actual training logic.
    """

    def execute(self, ctx):
        raise NotImplementedError

    def evaluate(self, model, x, y, metric, **kwargs):
        raise NotImplementedError


def _unit(optimizer_name="OptunaOptimizer", goal_metric="Accuracy"):
    return _ConcreteStrategy(
        optimizer={"component": optimizer_name, "params": {}},
        goal_metric=goal_metric,
    )


def test_evaluation_strategy_cannot_be_instantiated_directly():
    """Sanity check: it's an abstract base, not a usable strategy on its own."""
    with pytest.raises(TypeError):
        EvaluationStrategy(optimizer=None, goal_metric=None)


@pytest.mark.parametrize("strategy_class", [HoldoutUnit, CrossValidationUnit])
def test_concrete_strategies_inherit_the_shared_contract(strategy_class):
    """Locks in that both strategies replace FitModelUnit through this base,
    not through independent, duplicated validate()/_resolve_search logic.
    """
    assert issubclass(strategy_class, EvaluationStrategy)


def test_validate_raises_when_called_before_build_model_unit_has_run():
    """Regression: a missing key must not read as "nothing to optimize".

    ``optimizable_parameters`` is only absent from the context when
    ``BuildModelUnit`` hasn't run yet — a call-order mistake, not a model with
    no optimizable parameters (that case has the key present but empty).
    ``validate`` must use ``ctx.require``, not ``ctx.get``, or it would
    silently skip the optimizer/goal-metric checks it exists to run.
    """
    ctx = ExecutionContext()

    with pytest.raises(UnitContractError, match="'optimizable_parameters'"):
        _unit().validate(ctx)


def test_validate_is_a_noop_when_there_are_genuinely_no_optimizable_parameters():
    ctx = ExecutionContext()
    ctx.put("optimizable_parameters", [])

    # Should not raise, and should not need the optimizer/goal_metric to
    # resolve in the registry.
    unit = _unit(optimizer_name="DoesNotExist", goal_metric="DoesNotExist")
    unit.validate(ctx)

    assert unit._optimizer is None
    assert unit._goal_metric is None


def test_the_optimizer_is_kept_on_the_unit_not_in_the_shared_context():
    """Regression: the optimizer is this unit's own state, not an output.

    If it were written to the context by ``validate`` and read back by
    ``execute``, two evaluation strategies sharing one context (a DAG with
    two training nodes) would overwrite each other's optimizer, and the
    second would silently run the first one's.
    """

    class _Optimizer:
        def __init__(self, **params):
            pass

    registry = {
        "AnOptimizer": {"class": _Optimizer},
        "Accuracy": {"class": object, "metadata": {"maximize": True}},
    }
    di["component_registry"] = registry
    try:
        ctx = ExecutionContext()
        ctx.put("optimizable_parameters", ["lr"])

        unit = _unit(optimizer_name="AnOptimizer")
        unit.validate(ctx)

        assert isinstance(unit._optimizer, _Optimizer)
        assert not ctx.has("optimizer")
        assert not ctx.has("goal_metric")
    finally:
        del di["component_registry"]


def test_assert_model_keeps_its_runtime_state_rejects_a_detached_model():
    """Regression: an optimizer returning a fresh model must fail loudly.

    ``ModelFactory`` attaches the run id, splits and metric classes to the
    model instance. If an optimizer ever returned a different object instead
    of mutating and returning the one it received, ``calculate_metrics``
    would silently no-op and the run would finish with no metrics rather
    than fail.
    """

    class _Detached:
        run_id = None

    with pytest.raises(JobError, match="detached from its run"):
        EvaluationStrategy._assert_model_keeps_its_runtime_state(_Detached(), run_id=1)
