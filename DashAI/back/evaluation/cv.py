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

    The strategy handles metric aggregation at multiple levels:
    - FOLD level: Individual metrics from each fold
    - TRIAL level: Metrics during HPO trials
    - LAST/LAST_OUTER: Aggregated metrics (mean and std) for simple/nested CV
    """

    SCHEMA = CrossValidationSchema

    def __init__(self, **config) -> None:
        super().__init__(**config)
        self.inner_splitter: "BaseSplitter" = None

    def execute(self, ctx: ExecutionContext) -> None:
        """Execute k-fold cross-validation with optional nested CV and HPO.

        Trains and evaluates a model using k-fold cross-validation. Optionally performs
        hyperparameter optimization and nested CV to report unbiased hyperparameters
        metrics. Aggregates metrics across folds and returns the trained model.

        Parameters
        ----------
        ctx : ExecutionContext
            The shared execution context.
            ``x``/``y`` are lists of per-fold DatasetDict's, as produced by the CV 
            splitter. The last element of each list is the complete dataset, reserved 
            for the final training pass.
        """
        x = ctx.require("x")
        y = ctx.require("y")

        plot_paths = []

        # STEP 1: Hyperparameter Optimization (if enabled)
        if self._optimizer and self._goal_metric:
            nested = self.config.get("nested")
            # Initialize nested CV if required
            if nested:
                try:
                    registry = di["component_registry"]
                    splitter_name = nested.get("splitter_name", None)
                    
                    # Create inner splitter for nested CV fold generation
                    self.inner_splitter = registry[splitter_name]["class"](nested)
                except Exception as e:
                    raise ValueError(
                        f"Error configuring inner splitter for nested CV: {e}"
                    ) from e

                # Execute nested cross-validation for HPO
                self._report_progress(0.1, "Nested cross-validation")
                self._nested_cv(ctx)

            # Perform hyperparameter optimization and generate plots
            self._report_progress(0.25, "Hyperparameter optimization")
            self._do_hpo(ctx)
            plot_paths = self._generate_hpo_plots(ctx)

        model = ctx.require("model")
        total_folds = len(x) - 1

        # STEP 2: Main k-fold Cross-Validation Loop
        # Note: Last fold (index len(x)-1) is reserved for final training,
        # not CV evaluation
        for i in range(total_folds):
            self._report_progress(
                0.4 + ((i + 1) / total_folds) * 0.4,
                f"Evaluating fold {i + 1}/{total_folds}",
            )
            x_fold = x[i]
            y_fold = y[i]

            # Set model's internal references to current fold data
            model.x_data = x_fold
            model.y_data = y_fold
            
            # Train model on fold's training partition
            model.train(x_fold["train"], y_fold["train"])
            ctx.put("model", model)

            # Compute and store metrics for this fold
            EvaluateModelUnit(
                splits=["TRAIN", "TEST"],
                level=LevelEnum.FOLD,
                fold_index=i,
            )(ctx)

        # STEP 3: Aggregate metrics across all folds
        # Compute mean and std of fold metrics and store as LAST level metrics
        self._aggregate_fold_metrics(
            run_id=ctx.get("run_id"),
            level_to_agg=LevelEnum.FOLD,
            level_to_save=LevelEnum.LAST,
        )

        # STEP 4: Final model training on complete dataset
        # Train on all available data (the last fold contains the full dataset)
        self._report_progress(0.85, "Training final model")
        model.train(x[-1]["train"], y[-1]["train"])

        ctx.put("model", model)
        ctx.put_ref("plot_paths", plot_paths)

    def evaluate(self, model, input_dataset, output_dataset, metric, **kwargs):
        """Evaluate model using k-fold cross-validation (used as HPO objective
        function).

        This method implements cross-validation evaluation for hyperparameter
        optimization. It trains and evaluates the model on k-1 folds and computes
        the average performance. When used in nested CV, it evaluates on inner folds
        within a specific outer fold.
        
        Parameters
        ----------
        model : BaseModel
            The model instance to evaluate with specific hyperparameters.
        input_dataset : list[DatasetDict]
            List of DatasetDict's for each fold, where each DatasetDict contains
            {"train": X_train, "test": X_test}.
        output_dataset : list[DatasetDict]
            List of DatasetDict's for each fold, where each DatasetDict contains
            {"train": y_train, "test": y_test}.
        metric : Metric
            The metric class to compute on predictions.
            
        Returns
        -------
        float
            The average metric score across folds, used as the objective function
            value for hyperparameter optimization.
        """
        # Extract context: fold_index indicates if we're in nested CV inner loop
        # None means simple CV; an integer means nested CV on that outer fold
        fold_index = kwargs.get("fold_index")

        # List to collect the goal metric value from each fold
        folds_results = []
        
        # Dictionaries to accumulate all metrics across folds for averaging
        train_results = {}
        test_results = {}

        # Cross-validation loop
        # Iterate through k-1 folds (last fold is the complete dataset)
        # This loop represents either:
        # - Main CV loop (if fold_index is None)
        # - Inner CV loop within outer fold i (if fold_index is set)
        for i in range(len(input_dataset) - 1):
            x_fold = input_dataset[i]
            y_fold = output_dataset[i]

            # Set model's internal data references for this fold
            model.x_data = x_fold
            model.y_data = y_fold

            # Train model on this fold's training partition
            model.train(x_fold["train"], y_fold["train"])

            # Compute metrics on both train and test sets
            train_scores = model.compute_metrics(split=SplitEnum.TRAIN)
            test_scores = model.compute_metrics(split=SplitEnum.TEST)

            # Collect the goal metric value from this fold
            folds_results.append(test_scores[metric.__name__])

            # Accumulate all metrics only if NOT in nested CV inner loop
            if fold_index is None:
                for results, scores in [
                    (train_results, train_scores),
                    (test_results, test_scores),
                ]:
                    for metric_name, value in scores.items():
                        results.setdefault(metric_name, []).append(value)

        # Save intermediate metrics (simple CV only)
        if fold_index is None:
            # Compute average metrics across all folds
            averaged_train_results = {
                m: np.mean(values) for m, values in train_results.items()
            }
            averaged_test_results = {
                m: np.mean(values) for m, values in test_results.items()
            }

            # Persist averaged metrics as TRIAL level (intermediate HPO result)
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

        # Return the mean of the goal metric across folds
        # This is the objective value used by the optimizer
        return np.mean(folds_results)

    def _nested_cv(self, ctx: ExecutionContext) -> None:
        """Execute nested cross-validation: an inner HPO loop per outer fold.
        
        Parameters
        ----------
        ctx : ExecutionContext
            The shared execution context.
        """
        input_dataset = ctx.require("x")
        output_dataset = ctx.require("y")
        optimizer, goal_metric = self._resolve_search()

        # Nested CV Outer Loop
        # For each outer fold, optimize hyperparameters on inner folds
        for i in range(len(input_dataset) - 1):
            x_outer = input_dataset[i]
            y_outer = output_dataset[i]

            inner_x, inner_y, _ = self.inner_splitter.split(
                x_outer["train"], y_outer["train"]
            )

            # Create evaluation strategy for this outer fold
            # Passes fold_index so evaluate() knows it's in nested CV context
            strategy_with_context = partial(self.evaluate, fold_index=i)

            # INNER LOOP: Optimize hyperparameters using inner CV
            optimizer.optimize(
                ctx.require("model"),
                inner_x,
                inner_y,
                ctx.require("optimizable_parameters"),
                goal_metric,
                strategy=strategy_with_context,
            )

            # Set model's data references for outer fold evaluation
            model = optimizer.get_model()
            model.x_data = x_outer
            model.y_data = y_outer
            model.train(x_outer["train"], y_outer["train"])

            ctx.put("model", model)

            # Evaluate on outer fold's test data (this is OUTER_FOLD level metric)
            EvaluateModelUnit(
                splits=["TRAIN", "TEST"],
                level=LevelEnum.OUTER_FOLD,
                fold_index=i,
            )(ctx)

        # Aggregate outer fold metrics
        # Compute mean and std of OUTER_FOLD metrics and store as LAST_OUTER level
        self._aggregate_fold_metrics(
            run_id=ctx.get("run_id"),
            level_to_agg=LevelEnum.OUTER_FOLD,
            level_to_save=LevelEnum.LAST_OUTER,
        )

    def _aggregate_fold_metrics(
        self, run_id: int, level_to_agg=LevelEnum.FOLD, level_to_save=LevelEnum.LAST
    ) -> None:
        """Aggregate and average fold metrics across cross-validation folds.

        This method computes the mean and standard deviation of metrics collected
        at the fold level and stores the aggregated results at a higher level.
        This is used to provide summary statistics for model performance.

        Typical usage patterns:
        - Aggregate FOLD metrics -> store as LAST (simple CV summary)
        - Aggregate OUTER_FOLD metrics -> store as LAST_OUTER (nested CV summary)

        Parameters
        ----------
        run_id : int
            The database run ID to aggregate metrics for.
        level_to_agg : LevelEnum, optional
            The source metric level to aggregate. Default: FOLD.
        level_to_save : LevelEnum, optional
            The destination level for aggregated metrics. Default: LAST.
        """
        with di["session_factory"]() as db:
            # Query all metrics with level=level_to_agg for this run
            fold_metrics = (
                db.query(Metric)
                .filter(Metric.run_id == run_id, Metric.level == level_to_agg)
                .all()
            )
            
            # If no metrics found, nothing to aggregate
            if not fold_metrics:
                return

            # Group metrics by (split, name) for aggregation
            metrics_by_split_name = {}
            for metric in fold_metrics:
                key = (metric.split, metric.name)
                metrics_by_split_name.setdefault(key, []).append(metric.value)

            # Aggregate and persist metrics
            for (split, name), values in metrics_by_split_name.items():
                avg_value = np.mean(values)
                std_value = np.std(values) if len(values) > 1 else 0.0

                # Check if aggregated metric already exists for this split/name/run
                existing = (
                    db.query(Metric)
                    .filter_by(
                        run_id=run_id, split=split, level=level_to_save, name=name
                    )
                    .first()
                )

                if existing:
                    # Update existing metric with aggregated values
                    existing.value = avg_value
                    existing.std_value = std_value
                else:
                    # Create new aggregated metric
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

            # Persist aggregated metrics to database
            db.commit()
