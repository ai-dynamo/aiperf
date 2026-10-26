# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import orjson
import pytest
from pydantic import ValidationError

from aiperf.common.models import Conversation, Text, Turn
from aiperf.common.models.dataset_models import ChatToolMetadata
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.models import MultiTurn, SingleTurn
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader
from aiperf.dataset.loader.single_turn import SingleTurnDatasetLoader
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.conftest import make_run_from_cli
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


@pytest.fixture
def tool_calls() -> list[dict[str, Any]]:
    return [
        {
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": '{ "path": "main.py" }'},
        }
        for name in ("read_file", "stat_file")
    ]


@pytest.mark.parametrize(
    "loader_class", [MultiTurnDatasetLoader, SingleTurnDatasetLoader]
)
def test_tool_messages_survive_dataset_and_wire_round_trip(
    loader_class: type[MultiTurnDatasetLoader] | type[SingleTurnDatasetLoader],
    tool_calls: list[dict[str, Any]],
    tmp_path: Path,
    default_cfg: Any,
) -> None:
    turns = [
        {"role": "user", "text": "Read main.py"},
        {"role": "assistant", "text": None, "tool_calls": tool_calls},
        {"role": "tool", "text": "print('hello')", "tool_call_id": "call_read_file"},
        {"role": "tool", "text": "", "tool_call_id": "call_stat_file"},
        {"role": "user", "text": "Summarize the file", "output_length": 15},
    ]
    records = (
        [{"session_id": "tools", "turns": turns}]
        if loader_class is MultiTurnDatasetLoader
        else [{"session_id": "tools", **turn} for turn in turns]
    )
    path = tmp_path / "tools.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(record) for record in records))
    default_cfg.endpoint_type = EndpointType.CHAT
    loader = loader_class(filename=path, run=make_run_from_cli(default_cfg))
    conversations = loader.convert_to_conversations(loader.load_dataset())
    conversation = Conversation.model_validate_json(conversations[0].model_dump_json())
    model_endpoint = create_model_endpoint(EndpointType.CHAT)
    endpoint = ChatEndpoint(model_endpoint=model_endpoint)

    payload = orjson.loads(
        orjson.dumps(
            endpoint.format_payload(
                create_request_info(
                    model_endpoint=model_endpoint, turns=conversation.turns
                )
            )
        )
    )

    assert payload["messages"] == [
        {"role": "user", "content": "Read main.py"},
        {"role": "assistant", "content": None, "tool_calls": tool_calls},
        {"role": "tool", "content": "print('hello')", "tool_call_id": "call_read_file"},
        {"role": "tool", "content": "", "tool_call_id": "call_stat_file"},
        {"role": "user", "content": "Summarize the file"},
    ]
    assert payload["max_completion_tokens"] == 15


def test_tool_calls_keep_multimodal_content(
    tool_calls: list[dict[str, Any]], default_cfg: Any
) -> None:
    default_cfg.endpoint_type = EndpointType.CHAT
    data = MultiTurn.model_validate(
        {
            "turns": [
                {
                    "role": "assistant",
                    "text": "Inspecting the image",
                    "image": "https://example.com/image.png",
                    "tool_calls": tool_calls,
                }
            ]
        }
    )
    loader = MultiTurnDatasetLoader(
        filename="unused.jsonl", run=make_run_from_cli(default_cfg)
    )
    conversation = loader.convert_to_conversations({"tools": [data]})[0]
    model_endpoint = create_model_endpoint(EndpointType.CHAT)
    messages = ChatEndpoint(model_endpoint=model_endpoint).build_messages(
        conversation.turns
    )
    assert messages == [
        {
            "role": "assistant",
            "tool_calls": tool_calls,
            "content": [
                {"type": "text", "text": "Inspecting the image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.png"},
                },
            ],
        }
    ]


@pytest.mark.parametrize("fields", [{}, {"role": "assistant", "tool_calls": []}])
def test_empty_turn_still_rejected(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="At least one modality"):
        SingleTurn.model_validate(fields)


def test_plain_chat_message_has_no_tool_fields() -> None:
    endpoint = ChatEndpoint(model_endpoint=create_model_endpoint(EndpointType.CHAT))
    assert endpoint.build_messages([Turn(texts=[Text(contents=["Hello"])])]) == [
        {"role": "user", "content": "Hello"}
    ]


@pytest.mark.parametrize(
    "fields,match",
    [
        pytest.param({"role": "user", "text": "Hello", "tool_calls": []}, "requires role 'assistant'", id="tool-calls-wrong-role"),
        pytest.param({"role": "assistant", "text": "Hello", "tool_call_id": "call_1"}, "requires role 'tool'", id="tool-result-wrong-role"),
        pytest.param({"role": "tool", "text": "result", "tool_call_id": ""}, "at least 1 character", id="empty-call-id"),
        pytest.param({"role": "tool", "tool_call_id": "call_1"}, "At least one modality", id="missing-tool-content"),
    ],
)  # fmt: skip
def test_invalid_tool_message_rejected(fields: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        SingleTurn.model_validate(fields)


def test_chat_tool_metadata_rejected_by_responses_endpoint(
    tool_calls: list[dict[str, Any]],
) -> None:
    endpoint = ResponsesEndpoint(
        model_endpoint=create_model_endpoint(EndpointType.RESPONSES)
    )
    with pytest.raises(
        NotImplementedError, match="only supported with the chat endpoint"
    ):
        endpoint.build_messages(
            [
                Turn(
                    role="assistant",
                    chat_tool_metadata=ChatToolMetadata(tool_calls=tool_calls),
                )
            ]
        )


@pytest.mark.parametrize(
    "loader_class", [MultiTurnDatasetLoader, SingleTurnDatasetLoader]
)
@pytest.mark.parametrize(
    "endpoint_type",
    [EndpointType.COMPLETIONS, EndpointType.RESPONSES, EndpointType.MESSAGES],
)
def test_loaders_reject_tool_metadata_on_non_chat_endpoints(
    loader_class: type[MultiTurnDatasetLoader] | type[SingleTurnDatasetLoader],
    endpoint_type: str,
    tool_calls: list[dict[str, Any]],
) -> None:
    cfg = CLIConfig(model_names=["test-model"], endpoint_type=endpoint_type)
    loader = loader_class(filename="unused.jsonl", run=make_run_from_cli(cfg))
    turn = SingleTurn(role="assistant", text="Inspect main.py", tool_calls=tool_calls)
    data = {
        "tools": [
            MultiTurn(turns=[turn]) if loader_class is MultiTurnDatasetLoader else turn
        ]
    }
    with pytest.raises(NotImplementedError, match="require --endpoint-type chat"):
        loader.convert_to_conversations(data)
