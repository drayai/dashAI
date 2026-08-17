import logging
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Callable, List, Optional

from DashAI.back.core.schema_fields import (
    BaseSchema,
    component_field,
    schema_field,
    string_field,
)
from DashAI.back.core.utils import MultilingualString
from DashAI.back.job.base_job import JobError
from DashAI.back.models.base_model import BaseModel
from DashAI.back.models.model_factory import ModelFactory
from DashAI.back.units.base_unit import BaseUnit
from DashAI.back.units.context import ExecutionContext

if TYPE_CHECKING:
    from DashAI.back.optimizers.base_optimizer import BaseOptimizer

log = logging.getLogger(__name__)


class EvaluationStrategySchema(BaseSchema):
    optimizer: schema_field(
        component_field(parent="BaseOptimizer"),
        placeholder=None,
        description=MultilingualString(
            en="Optimizer used to search for hyperparameters. Only used when "
            "the model declares optimizable parameters. Omit for no HPO.",
            es="Optimizador usado para buscar hiperparámetros. Solo se usa "
            "cuando el modelo declara parámetros optimizables. Omitir "
            "para no hacer HPO.",
            pt="Otimizador usado para procurar hiperparâmetros. Só é usado "
            "quando o modelo declara parâmetros otimizáveis.",
            de="Optimierer für die Hyperparametersuche. Wird nur verwendet, "
            "wenn das Modell optimierbare Parameter deklariert.",
            zh="用于搜索超参数的优化器。仅当模型声明了可优化参数时使用。",
        ),
        alias=MultilingualString(
            en="Optimizer",
            es="Optimizador",
            pt="Otimizador",
            de="Optimierer",
            zh="优化器",
        ),
    )  # type: ignore
    goal_metric: schema_field(
        string_field(),
        placeholder="Accuracy",
        description=MultilingualString(
            en="Metric the hyperparameter search optimizes.",
            es="Métrica que optimiza la búsqueda de hiperparámetros.",
            pt="Métrica que a procura de hiperparâmetros otimiza.",
            de="Metrik, die die Hyperparametersuche optimiert.",
            zh="超参数搜索所优化的指标。",
        ),
        alias=MultilingualString(
            en="Goal metric",
            es="Métrica objetivo",
            pt="Métrica objetivo",
            de="Zielmetrik",
            zh="目标指标",
        ),
    )  # type: ignore


class EvaluationStrategy(BaseUnit, metaclass=ABCMeta):
    """Abstract base class defining the interface for model evaluation strategies.

    Concrete implementations (e.g., CrossValidationUnit, HoldoutUnit) inherit from
    this class and provide specific strategies for model evaluation and, when
    configured, HPO.

    ``validate`` resolves the optimizer and the goal metric so an impossible
    configuration is rejected before the job reports that training started.

    The optimizer is configured as a component field, so its value is
    ``{"component": <name>, "params": {...}}`` and the front renders the
    chosen optimizer's own form underneath.
    """

    SCHEMA = EvaluationStrategySchema

    REQUIRES = (
        "model",
        "factory",
        "optimizable_parameters",
        "model_parameters",
        "x",
        "y",
        "run_id",
    )
    PROVIDES = ("model", "plot_paths")

    def __init__(self, **config) -> None:
        super().__init__(**config)
        self._optimizer = None
        self._goal_metric = None
        self._progress_reporter: Optional[
            Callable[[Optional[float], Optional[str]], None]
        ] = None

    def _resolve_search(self):
        """Resolve the optimizer and the goal metric, memoized on this unit.

        Kept on the instance rather than in the context on purpose. These are
        this unit's own state, not something it hands to another unit: two
        ``EvaluationStrategy`` instances sharing a context — a DAG with two training
        nodes — would otherwise overwrite each other's optimizer, and the
        second one would silently run the first one's.
        """
        if self._optimizer is not None:
            return self._optimizer, self._goal_metric

        from kink import di

        component_registry = di["component_registry"]
        goal_metric_name: str = self.config["goal_metric"]
        optimizer_name: str = self.config["optimizer"]["component"]

        try:
            # The whole registry entry, not the class: the optimizer reads
            # metadata["maximize"] from it to pick a direction.
            goal_metric = component_registry[goal_metric_name]
        except Exception as e:
            log.exception(e)
            raise JobError(
                f"Metric is not compatible with the Task. {e}",
            ) from e

        try:
            optimizer_class = component_registry[optimizer_name]["class"]
            optimizer: "BaseOptimizer" = optimizer_class(
                **self.config["optimizer"]["params"]
            )
        except Exception as e:
            log.exception(e)
            raise JobError(
                f"Error instantiating optimizer {optimizer_name}, {e}",
            ) from e

        self._goal_metric = goal_metric
        self._optimizer = optimizer

        return optimizer, goal_metric

    def set_progress_reporter(
        self,
        progress_reporter: Optional[Callable[[Optional[float], Optional[str]], None]],
    ) -> None:
        """Register a callback that will receive progress updates."""
        self._progress_reporter = progress_reporter

    def _report_progress(
        self, fraction: Optional[float], message: Optional[str] = None
    ):
        """Emit progress updates when a reporter has been registered."""
        if self._progress_reporter is not None:
            self._progress_reporter(fraction, message)

    def validate(self, ctx: ExecutionContext) -> None:
        # ctx.require, not ctx.get: "optimizable_parameters" is one of this
        # unit's REQUIRES, so its absence means BuildModelUnit hasn't run yet
        # — a call-order mistake, not "there is nothing to optimize". Only an
        # empty value (the key present, genuinely no optimizable parameters)
        # skips the optimizer/goal-metric checks below, so no registry lookup
        # is needed either.
        if not ctx.require("optimizable_parameters"):
            return

        self._resolve_search()

    @abstractmethod
    def execute(self, ctx: ExecutionContext) -> None:
        """Do the evaluation strategy's work: train, optionally run HPO, and
        put ``model`` and ``plot_paths`` back onto the context.

        Parameters
        ----------
        ctx : ExecutionContext
            The shared execution context. ``x``/``y`` shape depends on the
            splitter that ran upstream: a single DatasetDict for holdout,
            a list of per-fold DatasetDict for cross-validation.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def evaluate(self, model: BaseModel, x, y, metric, **kwargs):
        """Evaluate the model on the given data and return the score.

        This method is called during hyperparameter optimization to compute
        the objective function value for a given set of hyperparameters.
        Different strategies may compute metrics differently (e.g., across CV folds
        or on a validation split).
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _do_hpo(self, ctx: ExecutionContext) -> None:
        """Execute hyperparameter optimization using the configured optimizer.

        The optimizer uses the self.evaluate method as the objective function,
        allowing each strategy to define its own evaluation logic.
        """
        optimizer, goal_metric = self._resolve_search()

        optimizer.optimize(
            ctx.require("model"),
            ctx.require("x"),
            ctx.require("y"),
            ctx.require("optimizable_parameters"),
            goal_metric,
            strategy=self.evaluate,
        )

        model = optimizer.get_model()
        best_params = optimizer.get_best_params()

        self._assert_model_keeps_its_runtime_state(model, ctx.require("run_id"))

        factory: ModelFactory = ctx.require("factory")
        old_parameters = ctx.require("model_parameters")
        ctx.put_ref(
            "best_parameters",
            factory.update_parameters(old_parameters, best_params),
        )

        ctx.put("model", model)

    def _generate_hpo_plots(self, ctx: ExecutionContext) -> List[str]:
        """Generate and pickle the hyperparameter optimization plots to disk.

        Shared by every evaluation strategy that runs HPO, so the plot
        generation logic only needs to be maintained in one place.

        Parameters
        ----------
        ctx : ExecutionContext
            The shared execution context.

        Returns
        -------
        list[str]
            Paths to the pickled plot files, in the order produced by the
            optimizer.
        """
        import os
        import pickle

        from kink import di

        from DashAI.back.core.artifacts import normalize_artifacts

        config = di["config"]
        optimizer, goal_metric = self._resolve_search()
        run_id = ctx.require("run_id")
        plot_paths: List[str] = []

        trials = optimizer.get_trials_values()

        plot_filenames, plots = optimizer.create_plots(
            trials,
            run_id,
            n_params=len(ctx.require("optimizable_parameters")),
            goal_metric=goal_metric,
        )

        normalized_plots = normalize_artifacts(plots)

        for filename, plot in zip(plot_filenames, normalized_plots, strict=False):
            plot_path = os.path.join(config["RUNS_PATH"], filename)
            with open(plot_path, "wb") as file:
                pickle.dump(plot, file)
                plot_paths.append(plot_path)

        return plot_paths

    @staticmethod
    def _assert_model_keeps_its_runtime_state(model, run_id) -> None:
        """Fail loudly if the optimizer returned a model that cannot log metrics.

        ``ModelFactory`` attaches the run id, the data splits and the metric
        classes to the model instance, and optimizers are expected to return
        that same instance. If one ever returns a fresh object instead,
        ``calculate_metrics`` would return early and the run would finish with
        no metrics at all instead of failing.
        """
        if run_id is None:
            return

        if getattr(model, "run_id", None) is None:
            raise JobError(
                "The optimizer returned a model detached from its run: metrics "
                "could not be computed for it. Optimizers must return the same "
                "model instance they received."
            )
