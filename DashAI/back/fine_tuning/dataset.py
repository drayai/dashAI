import hashlib
import json
from dataclasses import dataclass
from typing import Any

from datasets import Dataset as HFDataset

from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
    DatasetFormat,
    DatasetMapping,
    TrainingParameters,
)
from DashAI.back.dataloaders.classes.dashai_dataset import load_dataset


@dataclass
class PreparedDataset:
    train: HFDataset
    validation: HFDataset | None
    preview: list[dict[str, Any]]
    fingerprint: str
    source_rows: int


def _required_text(value: Any, column: str, row_number: int) -> str:
    if value is None or not str(value).strip():
        raise ValueError(f"Row {row_number}: column '{column}' is empty.")
    return str(value)


def _messages(value: Any, column: str, row_number: int) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Row {row_number}: column '{column}' contains invalid JSON."
            ) from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"Row {row_number}: messages must be a non-empty list.")
    normalized = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError(f"Row {row_number}: every message must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Row {row_number}: unsupported role '{role}'.")
        if content is None or not str(content).strip():
            raise ValueError(f"Row {row_number}: message content is empty.")
        normalized.append({"role": role, "content": str(content)})
    return normalized


def map_row(
    row: dict[str, Any], mapping: DatasetMapping, row_number: int
) -> dict[str, Any]:
    columns = set(row)
    selected = {
        mapping.text_column,
        mapping.prompt_column,
        mapping.completion_column,
        mapping.messages_column,
    }
    missing = sorted(column for column in selected if column and column not in columns)
    if missing:
        raise ValueError(f"Columns do not exist: {', '.join(missing)}")

    if mapping.format == DatasetFormat.TEXT:
        return {
            "text": _required_text(
                row[mapping.text_column], mapping.text_column, row_number
            )
        }
    if mapping.format == DatasetFormat.PROMPT_COMPLETION:
        return {
            "prompt": _required_text(
                row[mapping.prompt_column], mapping.prompt_column, row_number
            ),
            "completion": _required_text(
                row[mapping.completion_column], mapping.completion_column, row_number
            ),
        }
    return {
        "messages": _messages(
            row[mapping.messages_column], mapping.messages_column, row_number
        )
    }


def prepare_dataset(
    dataset_path: str,
    mapping: DatasetMapping,
    parameters: TrainingParameters,
) -> PreparedDataset:
    source = load_dataset(f"{dataset_path}/dataset")
    source_rows = len(source)
    limit = min(parameters.max_samples or source_rows, source_rows)
    if limit < 2:
        raise ValueError("At least two non-empty examples are required.")

    rows = [map_row(source[index], mapping, index) for index in range(limit)]
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False).encode("utf-8")
    fingerprint = hashlib.sha256(payload).hexdigest()
    dataset = HFDataset.from_list(rows)

    validation = None
    train = dataset
    if mapping.validation_split > 0 and len(dataset) > 1:
        test_size = max(1, round(len(dataset) * mapping.validation_split))
        if test_size >= len(dataset):
            test_size = 1
        split = dataset.train_test_split(
            test_size=test_size, seed=parameters.seed, shuffle=True
        )
        train = split["train"]
        validation = split["test"]

    return PreparedDataset(
        train=train,
        validation=validation,
        preview=rows[:3],
        fingerprint=fingerprint,
        source_rows=source_rows,
    )
