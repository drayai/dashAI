from functools import partial

import numpy as np
from kink import di

from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum
from DashAI.back.core.schema_fields import schema_field
from DashAI.back.core.utils import MultilingualString
from DashAI.back.dependencies.database.models import Metric
from DashAI.back.evaluation.base_evaluation_strategy import (
    BaseEvaluationStrategy,
    EvaluationStrategySchema,
)
from DashAI.back.splitters.base_splitter import BaseSplitter
from DashAI.back.units.context import ExecutionContext
from DashAI.back.units.evaluate_model_unit import EvaluateModelUnit


class CrossValidationSchema(EvaluationStrategySchema):
    nested: schema_field(
        dict,
        placeholder=None,
        description=MultilingualString(
            en="Inner splitter configuration for nested cross-validation. "
            "Omit to run plain (non-nested) CV.",
            es="Configuración del splitter interno para validación cruzada "
            "anidada. Omitir para CV simple (no anidada).",
            pt="Configuração do splitter interno para validação cruzada "
            "aninhada. Omitir para CV simples (não aninhada).",
            de="Konfiguration des inneren Splitters für verschachtelte "
            "Kreuzvalidierung. Weglassen für einfache (nicht verschachtelte) "
            "CV.",
            zh="嵌套交叉验证的内层分割器配置。留空则执行普通（非嵌套）交叉验证。",
        ),
        alias=MultilingualString(
            en="Nested CV",
            es="CV anidada",
            pt="CV aninhada",
            de="Verschachtelte CV",
            zh="嵌套交叉验证",
        ),
    )  # type: ignore


class CrossValidationEvaluationStrategy(BaseEvaluationStrategy):
    """Evaluation strategy implementing k-fold cross-validation with optional
    nested CV and HPO.

    Metric aggregation levels:
    - FOLD / OUTER_FOLD: per-fold metrics
    - TRIAL: metrics logged during HPO trials
    - LAST / LAST_OUTER: mean and std aggregated across folds
    """

    SCHEMA = CrossValidationSchema

    def __init__(self, **config) -> None:
        super().__init__(**config)
        self.inner_splitter: "BaseSplitter" = None

    def execute(self, ctx: ExecutionContext) -> None:
        """Run k-fold cross-validation, with optional nested CV and HPO.

        ``x``/``y`` are lists of per-fold ``{"train": ..., "test": ...}``
        dicts, as produced by the CV splitter. The last element of each list
        is the complete dataset, reserved for the final training pass.
        """
        x = ctx.require("x")
        y = ctx.require("y")

        plot_paths = []

        if self._optimizer and self._goal_metric:
            nested = self.config.get("nested")
            if nested:
                try:
                    registry = di["component_registry"]
                    splitter_name = nested.get("splitter_name", None)
                    self.inner_splitter = registry[splitter_name]["class"](nested)
                except Exception as e:
                    raise ValueError(
                        f"Error configuring inner splitter for nested CV: {e}"
                    ) from e

                self._report_progress(0.1, "Nested cross-validation")
                self._nested_cv(ctx)

            self._report_progress(0.25, "Hyperparameter optimization")
            self._do_hpo(ctx)
            plot_paths = self._generate_hpo_plots(ctx)

        model = ctx.require("model")
        total_folds = len(x) - 1

        for i in range(total_folds):
            self._report_progress(
                0.4 + ((i + 1) / total_folds) * 0.4,
                f"Evaluating fold {i + 1}/{total_folds}",
            )
            x_fold = x[i]
            y_fold = y[i]

            model.x_data = x_fold
            model.y_data = y_fold
            model.train(x_fold["train"], y_fold["train"])
            ctx.put("model", model)

            EvaluateModelUnit(
                splits=["TRAIN", "TEST"],
                level=LevelEnum.FOLD,
                fold_index=i,
            )(ctx)

        self._aggregate_fold_metrics(
            run_id=ctx.get("run_id"),
            level_to_agg=LevelEnum.FOLD,
            level_to_save=LevelEnum.LAST,
        )

        self._report_progress(0.85, "Training final model")
        model.train(x[-1]["train"], y[-1]["train"])

        ctx.put("model", model)
        ctx.put_ref("plot_paths", plot_paths)

    def evaluate(self, model, input_dataset, output_dataset, metric, **kwargs):
        """Evaluate the model via k-fold CV (used as the HPO objective).

        ``fold_index`` in ``kwargs`` distinguishes the nested-CV inner loop
        (metrics not saved) from plain CV (averaged metrics saved as TRIAL).
        """
        fold_index = kwargs.get("fold_index")

        folds_results = []
        train_results = {}
        test_results = {}

        for i in range(len(input_dataset) - 1):
            x_fold = input_dataset[i]
            y_fold = output_dataset[i]

            model.x_data = x_fold
            model.y_data = y_fold

            model.train(x_fold["train"], y_fold["train"])

            train_scores = model.compute_metrics(split=SplitEnum.TRAIN)
            test_scores = model.compute_metrics(split=SplitEnum.TEST)

            folds_results.append(test_scores[metric.__name__])

            if fold_index is None:
                for results, scores in [
                    (train_results, train_scores),
                    (test_results, test_scores),
                ]:
                    for metric_name, value in scores.items():
                        results.setdefault(metric_name, []).append(value)

        if fold_index is None:
            averaged_train_results = {
                m: np.mean(values) for m, values in train_results.items()
            }
            averaged_test_results = {
                m: np.mean(values) for m, values in test_results.items()
            }

            model._save_metrics(
                results=averaged_train_results,
                split=SplitEnum.TRAIN,
                level=LevelEnum.TRIAL,
            )
            model._save_metrics(
                results=averaged_test_results,
                split=SplitEnum.TEST,
                level=LevelEnum.TRIAL,
            )

        return np.mean(folds_results)

    def _nested_cv(self, ctx: ExecutionContext) -> None:
        """Execute nested cross-validation: an inner HPO loop per outer fold."""
        input_dataset = ctx.require("x")
        output_dataset = ctx.require("y")
        optimizer, goal_metric = self._resolve_search()

        for i in range(len(input_dataset) - 1):
            x_outer = input_dataset[i]
            y_outer = output_dataset[i]

            inner_x, inner_y, _ = self.inner_splitter.split(
                x_outer["train"], y_outer["train"]
            )

            strategy_with_context = partial(self.evaluate, fold_index=i)

            optimizer.optimize(
                ctx.require("model"),
                inner_x,
                inner_y,
                ctx.require("optimizable_parameters"),
                goal_metric,
                strategy=strategy_with_context,
            )

            model = optimizer.get_model()
            model.x_data = x_outer
            model.y_data = y_outer
            model.train(x_outer["train"], y_outer["train"])

            ctx.put("model", model)

            EvaluateModelUnit(
                splits=["TRAIN", "TEST"],
                level=LevelEnum.OUTER_FOLD,
                fold_index=i,
            )(ctx)

        self._aggregate_fold_metrics(
            run_id=ctx.get("run_id"),
            level_to_agg=LevelEnum.OUTER_FOLD,
            level_to_save=LevelEnum.LAST_OUTER,
        )

    def _aggregate_fold_metrics(
        self, run_id: int, level_to_agg=LevelEnum.FOLD, level_to_save=LevelEnum.LAST
    ) -> None:
        """Aggregate (mean/std) fold-level metrics into a single summary level."""
        with di["session_factory"]() as db:
            fold_metrics = (
                db.query(Metric)
                .filter(Metric.run_id == run_id, Metric.level == level_to_agg)
                .all()
            )

            if not fold_metrics:
                return

            metrics_by_split_name = {}
            for metric in fold_metrics:
                key = (metric.split, metric.name)
                metrics_by_split_name.setdefault(key, []).append(metric.value)

            for (split, name), values in metrics_by_split_name.items():
                avg_value = np.mean(values)
                std_value = np.std(values) if len(values) > 1 else 0.0

                existing = (
                    db.query(Metric)
                    .filter_by(
                        run_id=run_id, split=split, level=level_to_save, name=name
                    )
                    .first()
                )

                if existing:
                    existing.value = avg_value
                    existing.std_value = std_value
                else:
                    db.add(
                        Metric(
                            run_id=run_id,
                            split=split,
                            level=level_to_save,
                            name=name,
                            value=avg_value,
                            std_value=std_value,
                            step=0,
                        )
                    )

            db.commit()
