from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
    DatasetMapping,
    FineTuningMethod,
    TrainingParameters,
)
from DashAI.back.fine_tuning.dataset import PreparedDataset

ProgressCallback = Callable[[float, str, dict[str, Any]], None]
CancellationCallback = Callable[[], bool]


class TrainingCanceledError(Exception):
    """Raised when a fine-tuning run is cooperatively canceled."""


@dataclass
class FineTuningRequest:
    run_id: int
    model_id: str
    model_revision: str
    method: FineTuningMethod
    mapping: DatasetMapping
    parameters: TrainingParameters
    dataset: PreparedDataset
    model_path: Path
    output_path: Path
    resolved_revision: str


@dataclass
class FineTuningResult:
    artifact_path: Path
    metrics: dict[str, Any]
    runtime_metadata: dict[str, Any]


class FineTuningBackend(ABC):
    @abstractmethod
    def train(
        self,
        request: FineTuningRequest,
        progress: ProgressCallback,
        is_canceled: CancellationCallback,
    ) -> FineTuningResult:
        raise NotImplementedError
