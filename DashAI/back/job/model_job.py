import logging
from typing import TYPE_CHECKING

from kink import inject
from sqlalchemy import exc
from sqlalchemy.orm.attributes import flag_modified

from DashAI.back.dependencies.database.models import ModelSession, Run
from DashAI.back.evaluation.base_evaluation_strategy import BaseEvaluationStrategy
from DashAI.back.job.base_job import BaseJob, JobError
from DashAI.back.units.build_model_unit import BuildModelUnit
from DashAI.back.units.context import ExecutionContext
from DashAI.back.units.load_dataset_unit import LoadDatasetUnit
from DashAI.back.units.prepare_and_split_unit import PrepareAndSplitUnit
from DashAI.back.units.save_model_unit import SaveModelUnit

if TYPE_CHECKING:
    from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ModelJob(BaseJob):
    """ModelJob class to run the model training."""

    @inject
    def set_status_as_delivered(
        self, session_factory: "sessionmaker" = lambda di: di["session_factory"]
    ) -> None:
        """Set the status of the job as delivered."""
        run_id: int = self.kwargs["run_id"]

        with session_factory() as db:
            run: Run = db.get(Run, run_id)
            if not run:
                raise JobError(f"Run {run_id} does not exist in DB.")
            try:
                run.set_status_as_delivered()
                db.commit()
            except exc.SQLAlchemyError as e:
                log.exception(e)
                raise JobError(
                    "Internal database error",
                ) from e

    @inject
    def set_status_as_error(
        self, session_factory: "sessionmaker" = lambda di: di["session_factory"]
    ) -> None:
        """Set the status of the job as error."""
        run_id: int = self.kwargs.get("run_id")
        if run_id is None:
            return

        with session_factory() as db:
            run: Run = db.get(Run, run_id)
            if not run:
                return
            try:
                run.set_status_as_error()
                db.commit()
            except exc.SQLAlchemyError as e:
                log.exception(e)

    @inject
    def get_job_name(self) -> str:
        """Get a descriptive name for the job."""
        run_id = self.kwargs.get("run_id")
        if not run_id:
            return "Model Training"

        from kink import di

        session_factory = di["session_factory"]

        try:
            with session_factory() as db:
                run: Run = db.get(Run, run_id)
                if run and run.name:
                    return f"Train: {run.name}"
        except Exception:
            pass

        return f"Model Training ({run_id})"

    @inject
    def run(
        self,
    ) -> None:
        import gc
        import json

        from kink import di

        component_registry = di["component_registry"]
        session_factory = di["session_factory"]

        run_id: int = self.kwargs["run_id"]
        ctx = ExecutionContext(refs={"run_id": run_id})

        with session_factory() as db:
            run: Run = db.get(Run, run_id)
            if not run:
                raise JobError(f"Run {run_id} does not exist in DB.")
            run.huey_id = self.kwargs.get("huey_id", None)
            db.commit()
            self.report_progress(0.05, "Preparing data")
            try:
                model_session: ModelSession = db.get(ModelSession, run.model_session_id)
                if not model_session:
                    raise JobError(
                        f"Model session {run.model_session_id} does not exist in DB."
                    )

                LoadDatasetUnit(dataset_id=model_session.dataset_id)(ctx)

                PrepareAndSplitUnit(
                    task_name=model_session.task_name,
                    input_columns=model_session.input_columns,
                    output_columns=model_session.output_columns,
                    splits=json.loads(model_session.splits),
                )(ctx)

                run.split_indexes = json.dumps(ctx.require("split_indexes"))

                # __call__ runs validate() (the download gate) before execute()
                # for every unit, so no separate pre-check is needed here.
                BuildModelUnit(
                    model={"component": run.model_name, "params": run.parameters},
                    train_metrics=model_session.train_metrics,
                    validation_metrics=model_session.validation_metrics,
                    test_metrics=model_session.test_metrics,
                )(ctx)

                # Resolve the evaluation strategy (holdout, cross-validation, ...)
                # declared on the model session. This is the single seam that
                # lets different validation approaches plug into the same job.
                try:
                    strategy_name = model_session.evaluation_strategy
                    strategy_class = component_registry[strategy_name]["class"]
                except Exception as e:
                    log.exception(e)
                    raise JobError(
                        "Unable to find Evaluation Strategy with name "
                        f"{model_session.evaluation_strategy} in registry.",
                    ) from e

                evaluation_strategy: "BaseEvaluationStrategy" = strategy_class(
                    optimizer=(
                        {
                            "component": run.optimizer_name,
                            "params": run.optimizer_parameters,
                        }
                        if run.optimizer_name
                        else None
                    ),
                    goal_metric=run.goal_metric,
                    nested=getattr(run, "nested", None),
                )
                evaluation_strategy.set_progress_reporter(self.report_progress)

                # Resolving the optimizer before the status changes keeps an
                # invalid configuration from ever reporting that training began.
                evaluation_strategy.validate(ctx)

                try:
                    run.set_status_as_started()
                    db.commit()
                except exc.SQLAlchemyError as e:
                    log.exception(e)
                    raise JobError(
                        "Connection with the database failed",
                    ) from e
                self.report_progress(0.2, "Training")

                try:
                    evaluation_strategy(ctx)
                except Exception as e:
                    log.exception(e)
                    raise JobError(
                        f"Model training and evaluation failed {e}",
                    ) from e

                plot_paths = ctx.require("plot_paths")

                if ctx.has("best_parameters"):
                    run.parameters = ctx.get("best_parameters")
                    flag_modified(run, "parameters")
                    db.commit()

                try:
                    paths = plot_paths + [None] * (4 - len(plot_paths))
                    (
                        run.plot_history_path,
                        run.plot_slice_path,
                        run.plot_contour_path,
                        run.plot_importance_path,
                    ) = paths[:4]
                    db.commit()
                except Exception as e:
                    log.exception(e)
                    raise JobError(
                        f"Hyperparameter plot path saving failed {e}",
                    ) from e

                self.report_progress(0.95, "Saving model")
                SaveModelUnit()(ctx)

                try:
                    run.run_path = ctx.require("model_path")
                    db.commit()
                except exc.SQLAlchemyError as e:
                    log.exception(e)
                    raise JobError(
                        "Connection with the database failed",
                    ) from e
                try:
                    run.set_status_as_finished()
                    db.commit()
                except exc.SQLAlchemyError as e:
                    log.exception(e)
                    raise JobError(
                        "Connection with the database failed",
                    ) from e
            except Exception as e:
                run.set_status_as_error()
                db.commit()
                raise e
            finally:
                ctx.clear_cache()
                gc.collect()
