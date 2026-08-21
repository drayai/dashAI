import json

import pytest

from DashAI.back.api.api_v1.schemas.fine_tuning_params import (
    DatasetFormat,
    DatasetMapping,
)
from DashAI.back.fine_tuning.dataset import map_row


@pytest.mark.parametrize(
    ("mapping", "row", "expected"),
    [
        (
            DatasetMapping(format=DatasetFormat.TEXT, text_column="body"),
            {"body": "hello"},
            {"text": "hello"},
        ),
        (
            DatasetMapping(
                format=DatasetFormat.PROMPT_COMPLETION,
                prompt_column="instruction",
                completion_column="answer",
            ),
            {"instruction": "Question", "answer": "Response"},
            {"prompt": "Question", "completion": "Response"},
        ),
        (
            DatasetMapping(format=DatasetFormat.MESSAGES, messages_column="chat"),
            {
                "chat": json.dumps(
                    [
                        {"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"},
                    ]
                )
            },
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            },
        ),
    ],
)
def test_supported_mappings(mapping, row, expected):
    assert map_row(row, mapping, 0) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-json", "invalid JSON"),
        (json.dumps([{"role": "tool", "content": "x"}]), "unsupported role"),
        (json.dumps([{"role": "user", "content": ""}]), "content is empty"),
    ],
)
def test_messages_validation(value, message):
    mapping = DatasetMapping(format=DatasetFormat.MESSAGES, messages_column="chat")
    with pytest.raises(ValueError, match=message):
        map_row({"chat": value}, mapping, 0)


def test_missing_and_empty_columns_are_rejected():
    mapping = DatasetMapping(format=DatasetFormat.TEXT, text_column="body")
    with pytest.raises(ValueError, match="Columns do not exist"):
        map_row({"other": "x"}, mapping, 0)
    with pytest.raises(ValueError, match="is empty"):
        map_row({"body": "  "}, mapping, 0)
