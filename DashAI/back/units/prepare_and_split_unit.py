"""Unit that prepares a dataset for a task and splits it using a BaseSplitter."""

import logging
from typing import TYPE_CHECKING

from DashAI.back.core.schema_fields import (
    BaseSchema,
    list_field,
    schema_field,
    string_field,
)
from DashAI.back.core.utils import MultilingualString
from DashAI.back.job.base_job import JobError
from DashAI.back.splitters.base_splitter import BaseSplitter
from DashAI.back.units.base_unit import BaseUnit
from DashAI.back.units.context import ExecutionContext

if TYPE_CHECKING:
    from DashAI.back.tasks.base_task import BaseTask

log = logging.getLogger(__name__)


class PrepareAndSplitSchema(BaseSchema):
    task_name: schema_field(
        string_field(),
        placeholder="TabularClassificationTask",
        description=MultilingualString(
            en="Name of the task the dataset is prepared for.",
            es="Nombre de la tarea para la que se prepara el conjunto de datos.",
            pt="Nome da tarefa para a qual o conjunto de dados é preparado.",
            de="Name der Aufgabe, für die der Datensatz vorbereitet wird.",
            zh="数据集所准备的任务名称。",
        ),
        alias=MultilingualString(
            en="Task", es="Tarea", pt="Tarefa", de="Aufgabe", zh="任务"
        ),
    )  # type: ignore
    input_columns: schema_field(
        list_field(string_field(), min_items=1),
        placeholder=[],
        description=MultilingualString(
            en="Names of the columns used as model input.",
            es="Nombres de las columnas usadas como entrada del modelo.",
            pt="Nomes das colunas usadas como entrada do modelo.",
            de="Namen der als Modelleingabe verwendeten Spalten.",
            zh="用作模型输入的列名。",
        ),
        alias=MultilingualString(
            en="Input columns",
            es="Columnas de entrada",
            pt="Colunas de entrada",
            de="Eingabespalten",
            zh="输入列",
        ),
    )  # type: ignore
    output_columns: schema_field(
        list_field(string_field(), min_items=1),
        placeholder=[],
        description=MultilingualString(
            en="Names of the columns the model has to predict.",
            es="Nombres de las columnas que el modelo debe predecir.",
            pt="Nomes das colunas que o modelo deve prever.",
            de="Namen der Spalten, die das Modell vorhersagen soll.",
            zh="模型需要预测的列名。",
        ),
        alias=MultilingualString(
            en="Output columns",
            es="Columnas de salida",
            pt="Colunas de saída",
            de="Ausgabespalten",
            zh="输出列",
        ),
    )  # type: ignore
    splits: schema_field(
        dict,
        placeholder={
            "splitter_name": "HoldoutSplitter",
            "random_state": 42,
            "shuffle": True,
            "train": 0.7,
            "test": 0.1,
            "validation": 0.2,
        },
        description=MultilingualString(
            en="Split configuration: the name of a registered Splitter plus its "
            "own parameters (e.g. proportions for holdout, n_splits for "
            "k-fold).",
            es="Configuración de partición: el nombre de un Splitter registrado "
            "más sus propios parámetros (p. ej. proporciones para holdout, "
            "n_splits para k-fold).",
            pt="Configuração de divisão: o nome de um Splitter registado mais "
            "os seus próprios parâmetros.",
            de="Split-Konfiguration: der Name eines registrierten Splitters "
            "plus dessen eigene Parameter.",
            zh="划分配置：已注册 Splitter 的名称及其自身参数。",
        ),
        alias=MultilingualString(
            en="Splits",
            es="Particiones",
            pt="Partições",
            de="Teilmengen",
            zh="数据划分",
        ),
    )  # type: ignore


class PrepareAndSplitUnit(BaseUnit):
    """Validate a dataset against a task and split it using a registered Splitter.

    Runs the task's own validation, counts the labels, and delegates the
    actual partitioning to whichever ``BaseSplitter`` is named in
    ``splits["splitter_name"]`` (holdout, k-fold, repeated k-fold, etc.).

    ``x``/``y`` end up with a shape that depends on the chosen splitter: a
    single ``{"train", "validation", "test"}`` dict for holdout, a list of
    per-fold dicts for cross-validation. Downstream units (``BuildModelUnit``,
    the evaluation strategies) are the ones that know which shape to expect,
    so this unit does not normalize it.
    """

    SCHEMA = PrepareAndSplitSchema

    REQUIRES = ("dataset",)
    PROVIDES = ("x", "y", "n_labels", "task", "split_indexes")

    def execute(self, ctx: ExecutionContext) -> None:
        from kink import di

        from DashAI.back.dataloaders.classes.dashai_dataset import select_columns

        component_registry = di["component_registry"]

        task_name: str = self.config["task_name"]
        input_columns = self.config["input_columns"]
        output_columns = self.config["output_columns"]
        splits_data = self.config["splits"]

        loaded_dataset = ctx.require("dataset")

        try:
            task: "BaseTask" = component_registry[task_name]["class"]()
        except Exception as e:
            log.exception(e)
            raise JobError(
                f"Unable to find Task with name {task_name} in registry",
            ) from e

        try:
            prepared_dataset = task.prepare_for_task(
                dataset=loaded_dataset,
                input_columns=input_columns,
                output_columns=output_columns,
            )
            n_labels = task.num_labels(prepared_dataset, output_columns[0])

            X, Y = select_columns(
                prepared_dataset,
                input_columns,
                output_columns,
            )
        except Exception as e:
            log.exception(e)
            raise JobError(
                f"Can not prepare Dataset {ctx.get('dataset_id')} for Task {task_name}",
            ) from e

        try:
            splitter_name = splits_data.get("splitter_name", None)
            splitter: BaseSplitter = component_registry[splitter_name]["class"](
                splits_data=splits_data,
            )
        except Exception as e:
            log.exception(e)
            raise JobError(
                f"Unable to find Splitter with name {splitter_name} in registry.",
            ) from e

        try:
            x, y, split_indexes = splitter.split(X, Y)
        except Exception as e:
            log.exception(e)
            raise JobError(
                f"Error splitting Dataset {ctx.get('dataset_id')} "
                f"with Splitter {splitter_name}: {e}",
            ) from e

        ctx.put_ref("task_name", task_name)
        ctx.put_ref("split_indexes", split_indexes)
        ctx.put("task", task)
        ctx.put("n_labels", n_labels)
        ctx.put("x", x)
        ctx.put("y", y)
