from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from DashAI.back.core.enums.status import FineTuningStatus


class DatasetFormat(str, Enum):
    TEXT = "text"
    PROMPT_COMPLETION = "prompt_completion"
    MESSAGES = "messages"


class FineTuningMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"


class DatasetMapping(BaseModel):
    format: DatasetFormat
    text_column: Optional[str] = None
    prompt_column: Optional[str] = None
    completion_column: Optional[str] = None
    messages_column: Optional[str] = None
    validation_split: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_columns(self):
        required = {
            DatasetFormat.TEXT: ("text_column",),
            DatasetFormat.PROMPT_COMPLETION: (
                "prompt_column",
                "completion_column",
            ),
            DatasetFormat.MESSAGES: ("messages_column",),
        }[self.format]
        missing = [field for field in required if not getattr(self, field)]
        if missing:
            raise ValueError(f"Missing mapping fields: {', '.join(missing)}")
        return self


class TrainingParameters(BaseModel):
    preset: Literal["quick_test", "qlora_8gb", "lora_small"] = "quick_test"
    max_samples: Optional[int] = Field(default=32, ge=2)
    max_steps: int = Field(default=3, ge=1)
    num_train_epochs: float = Field(default=1.0, gt=0)
    max_length: int = Field(default=128, ge=32, le=8192)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    target_modules: str = "all-linear"
    seed: int = 42
    logging_steps: int = Field(default=1, ge=1)
    save_steps: int = Field(default=1, ge=1)
    eval_steps: int = Field(default=1, ge=1)


class FineTuningRunCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    dataset_id: int
    base_model_id: str
    base_model_revision: str = "main"
    method: FineTuningMethod = FineTuningMethod.QLORA
    dataset_mapping: DatasetMapping
    training_parameters: TrainingParameters = Field(default_factory=TrainingParameters)


class FineTuningRunRead(FineTuningRunCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    resolved_model_revision: Optional[str] = None
    status: FineTuningStatus
    huey_id: Optional[str] = None
    progress: float
    progress_message: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    runtime_metadata: Optional[dict[str, Any]] = None
    artifact_path: Optional[str] = None
    error_message: Optional[str] = None
    cancellation_requested: bool
    created: datetime
    last_modified: datetime
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class PreflightRequest(BaseModel):
    dataset_id: int
    base_model_id: str
    base_model_revision: str = "main"
    method: FineTuningMethod = FineTuningMethod.QLORA
    dataset_mapping: DatasetMapping
    training_parameters: TrainingParameters = Field(default_factory=TrainingParameters)
    download_model: bool = False


class PreflightIssue(BaseModel):
    code: str
    message: str


class PreflightReport(BaseModel):
    ready: bool
    blockers: list[PreflightIssue]
    warnings: list[PreflightIssue]
    preview: list[dict[str, Any]]
    dataset_fingerprint: Optional[str] = None
    dataset_rows: int = 0
    train_rows: int = 0
    validation_rows: int = 0
    model_path: Optional[str] = None
    model_downloaded: bool = False
    resolved_model_revision: Optional[str] = None
    hardware: dict[str, Any]
    dependencies: dict[str, Optional[str]]
    estimated_vram_gb: Optional[float] = None


class LocalModelInfo(BaseModel):
    key: str
    kind: Literal["base", "adapter"]
    name: str
    source: str
    path: Optional[str] = None
    size_bytes: int = 0
    status: str
    run_id: Optional[int] = None
    local_model_id: Optional[int] = None
    recommended_vram_gb: Optional[int] = None
    downloadable: bool = False
    in_use: bool = False
