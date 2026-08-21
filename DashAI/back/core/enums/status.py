from enum import Enum


class ExplainerStatus(Enum):
    NOT_STARTED = 0
    DELIVERED = 1
    STARTED = 2
    FINISHED = 3
    ERROR = 4


class RunStatus(Enum):
    NOT_STARTED = 0
    DELIVERED = 1
    STARTED = 2
    FINISHED = 3
    ERROR = 4


class FineTuningStatus(str, Enum):
    NOT_STARTED = "not_started"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class FineTuningBackendType(str, Enum):
    """Execution backend for a fine-tuning run.

    Unsloth is a separate execution backend for LoRA/QLoRA, not an
    alternative method: it is modeled independently from the method column.
    """

    TRANSFORMERS = "transformers"
    UNSLOTH = "unsloth"


class ExplorerStatus(Enum):
    NOT_STARTED = 0
    DELIVERED = 1
    STARTED = 2
    FINISHED = 3
    ERROR = 4


class ConverterStatus(Enum):
    NOT_STARTED = 0
    DELIVERED = 1
    STARTED = 2
    FINISHED = 3
    ERROR = 4


class PluginStatus(Enum):
    REGISTERED = 1
    INSTALLED = 2
    ERROR = 99


class DatasetStatus(Enum):
    NOT_STARTED = 0
    DELIVERED = 1
    STARTED = 2
    FINISHED = 3
    ERROR = 4


class PredictionStatus(Enum):
    NOT_STARTED = 0
    DELIVERED = 1
    STARTED = 2
    FINISHED = 3
    ERROR = 4


class DatafileStatus(Enum):
    DOWNLOADING = "downloading"
    READY = "ready"
    ERROR = "error"
