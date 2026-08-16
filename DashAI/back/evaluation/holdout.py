from DashAI.back.evaluation.base_evaluation_strategy import BaseEvaluationStrategy
from DashAI.back.units.context import ExecutionContext
from DashAI.back.units.evaluate_model_unit import EvaluateModelUnit


class HoldoutEvaluationStrategy(BaseEvaluationStrategy):
    """Evaluation strategy implementing holdout (train/validation/test split)
    validation.

    Trains on the training partition, optionally runs HPO against the
    validation partition, and leaves the final TRAIN/VALIDATION/TEST metrics
    to ``EvaluateModelUnit``
    """

    def execute(self, ctx: ExecutionContext) -> None:
        """Train on the training set, optionally optimize with the validation
        set.

        ``x``/``y`` are the ``{"train": ..., "validation": ..., "test": ...}``
        dicts produced by the holdout splitter.
        """
        x = ctx.require("x")
        y = ctx.require("y")

        plot_paths = []

        if self._optimizer and self._goal_metric:
            self._report_progress(0.2, "Hyperparameter optimization")
            self._do_hpo(ctx)
            plot_paths = self._generate_hpo_plots(ctx)

        self._report_progress(0.5, "Training")

        model = ctx.require("model")
        model.x_data = x
        model.y_data = y
        model.train(x["train"], y["train"], x["validation"], y["validation"])
        ctx.put("model", model)

        EvaluateModelUnit().execute(ctx)

        ctx.put_ref("plot_paths", plot_paths)

    def evaluate(self, model, input_dataset, output_dataset, metric, **kwargs):
        """Evaluate model on the validation set during an HPO trial.

        Used as the objective function during hyperparameter optimization:
        trains on the training set and scores on the validation set.
        """
        from DashAI.back.core.enums.metrics import LevelEnum, SplitEnum

        model.x_data = input_dataset
        model.y_data = output_dataset
        model.train(input_dataset["train"], output_dataset["train"])

        y_pred = model.predict(input_dataset["validation"])

        output_dataset_transformed = model.prepare_output(
            output_dataset["validation"], is_fit=False
        )

        model.calculate_metrics(split=SplitEnum.TRAIN, level=LevelEnum.TRIAL)
        model.calculate_metrics(split=SplitEnum.VALIDATION, level=LevelEnum.TRIAL)

        return metric.score(output_dataset_transformed, y_pred)
