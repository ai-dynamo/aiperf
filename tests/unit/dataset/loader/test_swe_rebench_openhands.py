# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.swe_rebench_openhands import (
    SweRebenchOpenHandsDatasetLoader,
    _strip_none,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType, PublicDatasetType
from aiperf.workers.session_manager import UserSession
from tests.harness.optional_deps import HAS_DATASETS
from tests.unit.conftest import make_run_from_cli
from tests.unit.endpoints.conftest import create_model_endpoint, create_request_info


class _CharCountTokenizer:
    """Counts characters so expected max_tokens are exact without a real vocabulary."""

    def encode_lengths_batch(self, texts: list[str]) -> list[int]:
        return [len(text) for text in texts]


def _padded(message: dict[str, Any]) -> dict[str, Any]:
    """Mirror parquet struct unification, which pads every message with every key."""
    return {
        "role": None,
        "content": None,
        "name": None,
        "tool_call_id": None,
        "tool_calls": None,
        **message,
    }


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": orjson.dumps(arguments).decode()},
    }


def _tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Run a command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command",
                            "enum": None,
                            "additionalProperties": None,
                        },
                        "file_text": None,
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "finish",
                "description": "Finish the task",
                "parameters": {
                    "type": "object",
                    "properties": {"message": {"type": "string", "description": "m"}},
                    "required": None,
                },
            },
        },
    ]


def _trajectory() -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": "Looking first.",
            "tool_calls": [_tool_call("call-1", "execute_bash", {"command": "ls"})],
        },
        {
            "role": "tool",
            "content": "a.py",
            "tool_call_id": "call-1",
            "name": "execute_bash",
        },
        {
            "role": "assistant",
            "content": "Editing.",
            "tool_calls": [_tool_call("call-2", "execute_bash", {"command": "sed"})],
        },
        {
            "role": "tool",
            "content": "ok",
            "tool_call_id": "call-2",
            "name": "execute_bash",
        },
        {"role": "assistant", "content": "Done."},
    ]


def _row(
    trajectory_id: str = "traj-1",
    *,
    trajectory: list[dict[str, Any]] | None = None,
    tools: Any = None,
    resolved: int = 1,
) -> dict[str, Any]:
    messages = trajectory if trajectory is not None else _trajectory()
    return {
        "trajectory_id": trajectory_id,
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "trajectory": [_padded(message) for message in messages],
        "tools": _tools() if tools is None else tools,
        "model_patch": "",
        "exit_status": "submitted",
        "resolved": resolved,
        "gen_tests_correct": 1.0,
        "pred_passes_gen_tests": 1.0,
    }


def _loader(
    filters: dict[str, str] | None = None,
    tokenizer: Any = None,
) -> SweRebenchOpenHandsDatasetLoader:
    return SweRebenchOpenHandsDatasetLoader(
        filters=filters,
        tokenizer=tokenizer,
        hf_dataset_name="nebius/SWE-rebench-openhands-trajectories",
        streaming=True,
    )


def test_strip_none_removes_padding_recursively() -> None:
    value = {
        "role": "tool",
        "content": None,
        "tool_calls": None,
        "function": {
            "parameters": {
                "properties": {"file_text": None, "x": {"enum": None, "type": "string"}}
            }
        },
    }

    assert _strip_none(value) == {
        "role": "tool",
        "function": {"parameters": {"properties": {"x": {"type": "string"}}}},
    }


def test_strip_none_keeps_none_inside_lists() -> None:
    assert _strip_none({"enum": ["a", None]}) == {"enum": ["a", None]}


@pytest.mark.asyncio
async def test_convert_strips_padding_from_messages_and_tools() -> None:
    conversations = await _loader().convert_to_conversations({"dataset": [_row()]})

    turn = conversations[0].turns[0]
    assert all(
        value is not None for message in turn.raw_messages for value in message.values()
    )
    assert turn.raw_tools is not None
    command_schema = turn.raw_tools[0]["function"]["parameters"]["properties"][
        "command"
    ]
    assert command_schema == {"type": "string", "description": "The command"}
    assert "file_text" not in turn.raw_tools[0]["function"]["parameters"]["properties"]
    assert turn.raw_tools[1]["function"]["parameters"] == {
        "type": "object",
        "properties": {"message": {"type": "string", "description": "m"}},
    }


@pytest.mark.asyncio
async def test_convert_one_request_per_assistant_message_splits_at_boundaries() -> None:
    conversations = await _loader().convert_to_conversations({"dataset": [_row()]})

    conversation = conversations[0]
    assert conversation.session_id == "traj-1"
    assert conversation.context_mode == ConversationContextMode.DELTAS_WITH_RESPONSES
    assert len(conversation.turns) == 3
    first, second, third = (turn.raw_messages for turn in conversation.turns)
    assert [message["role"] for message in first] == ["system", "user"]
    assert [message["role"] for message in second] == ["assistant", "tool"]
    assert second[0]["content"] == "Looking first."
    assert [message["role"] for message in third] == ["assistant", "tool"]
    assert third[0]["content"] == "Editing."


@pytest.mark.asyncio
async def test_convert_trajectory_starting_with_assistant_raises_error() -> None:
    trajectory = [{"role": "assistant", "content": "hi"}]

    with pytest.raises(
        DatasetLoaderError,
        match=r"^row 1 trajectory_id='traj-1': trajectory must start",
    ):
        await _loader().convert_to_conversations(
            {"dataset": [_row(trajectory=trajectory)]}
        )


def _session_payloads(conversation: Any) -> list[dict[str, Any]]:
    """Drive the worker's turn accumulation and build each request's chat payload."""
    model_endpoint = create_model_endpoint(EndpointType.CHAT, model_name="target")
    endpoint = ChatEndpoint(model_endpoint=model_endpoint)
    session = UserSession(
        x_correlation_id="corr",
        num_turns=len(conversation.turns),
        conversation=conversation,
        context_mode=conversation.context_mode,
    )
    payloads = []
    for turn_index in range(len(conversation.turns)):
        session.advance_turn(turn_index)
        assert not session.should_store_response()
        request = create_request_info(
            model_endpoint=model_endpoint,
            turns=session.turn_list,
            turn_index=turn_index,
        )
        payloads.append(endpoint.format_payload(request))
    return payloads


@pytest.mark.asyncio
async def test_second_request_carries_recorded_prefix_and_no_live_response() -> None:
    conversations = await _loader().convert_to_conversations({"dataset": [_row()]})

    payloads = _session_payloads(conversations[0])

    assert [message["role"] for message in payloads[2]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
    ]
    assert payloads[2]["messages"][2]["content"] == "Looking first."
    assert payloads[2]["messages"][4]["content"] == "Editing."
    assert payloads[2]["messages"][4]["tool_calls"][0]["id"] == "call-2"
    assert len(payloads[2]["messages"]) == 6


@pytest.mark.asyncio
async def test_tools_reach_every_request_from_turn_zero_only() -> None:
    conversations = await _loader().convert_to_conversations({"dataset": [_row()]})
    conversation = conversations[0]

    payloads = _session_payloads(conversation)

    assert [turn.raw_tools is not None for turn in conversation.turns] == [
        True,
        False,
        False,
    ]
    assert all(payload["tools"] == payloads[0]["tools"] for payload in payloads)
    assert payloads[2]["tools"][0]["function"]["name"] == "execute_bash"


@pytest.mark.asyncio
async def test_max_tokens_counts_content_and_tool_calls_of_answering_message() -> None:
    conversations = await _loader(
        tokenizer=_CharCountTokenizer()
    ).convert_to_conversations({"dataset": [_row()]})

    turns = conversations[0].turns
    tool_calls_json = orjson.dumps(_trajectory()[2]["tool_calls"]).decode()
    assert turns[0].max_tokens == len("Looking first.") + len(tool_calls_json)
    assert turns[1].max_tokens == len("Editing.") + len(
        orjson.dumps(_trajectory()[4]["tool_calls"]).decode()
    )
    assert turns[2].max_tokens == len("Done.")


@pytest.mark.asyncio
async def test_max_tokens_is_clamped_to_one_for_empty_answer() -> None:
    trajectory = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": ""},
    ]

    conversations = await _loader(
        tokenizer=_CharCountTokenizer()
    ).convert_to_conversations({"dataset": [_row(trajectory=trajectory)]})

    assert conversations[0].turns[0].max_tokens == 1


@pytest.mark.asyncio
async def test_max_tokens_is_unset_without_tokenizer() -> None:
    conversations = await _loader().convert_to_conversations({"dataset": [_row()]})

    assert all(turn.max_tokens is None for turn in conversations[0].turns)


@pytest.mark.asyncio
async def test_resolved_filter_keeps_only_matching_rows() -> None:
    rows = [_row("solved", resolved=1), _row("unsolved", resolved=0)]

    conversations = await _loader({"resolved": "false"}).convert_to_conversations(
        {"dataset": rows}
    )

    assert [conversation.session_id for conversation in conversations] == ["unsolved"]


@pytest.mark.asyncio
async def test_max_turns_filter_keeps_first_requests_of_each_session() -> None:
    conversations = await _loader({"max_turns": "2"}).convert_to_conversations(
        {"dataset": [_row()]}
    )

    turns = conversations[0].turns
    assert len(turns) == 2
    assert turns[0].raw_tools is not None
    assert turns[1].raw_messages[0]["content"] == "Looking first."


@pytest.mark.parametrize(
    "filters, match",
    [
        param({"unknown": "x"}, "available filters", id="unknown_key"),
        param({"resolved": "maybe"}, "available filters", id="bad_bool"),
        param({"max_turns": "0"}, "available filters", id="zero_max_turns"),
        param({"max_turns": "-3"}, "available filters", id="negative_max_turns"),
    ],
)  # fmt: skip
def test_invalid_filters_list_valid_filters(
    filters: dict[str, str], match: str
) -> None:
    with pytest.raises(DatasetLoaderError, match=match):
        _loader(filters)


@pytest.mark.asyncio
async def test_conversation_limit_stops_reading_rows_early() -> None:
    pulled: list[str] = []

    def rows():
        for index in range(5):
            pulled.append(f"traj-{index}")
            yield _row(f"traj-{index}")

    loader = _loader()
    with patch.object(loader, "_max_conversations", return_value=2):
        conversations = await loader.convert_to_conversations({"dataset": rows()})

    assert [conversation.session_id for conversation in conversations] == [
        "traj-0",
        "traj-1",
    ]
    assert pulled == ["traj-0", "traj-1"]


@pytest.mark.asyncio
async def test_rows_without_assistant_message_are_skipped() -> None:
    no_assistant = _row(
        "no-assistant",
        trajectory=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ],
    )

    conversations = await _loader().convert_to_conversations(
        {"dataset": [no_assistant, _row("keep")]}
    )

    assert [conversation.session_id for conversation in conversations] == ["keep"]


@pytest.mark.asyncio
async def test_all_rows_skipped_raises_no_matches_error() -> None:
    no_assistant = _row(
        "no-assistant",
        trajectory=[{"role": "user", "content": "u"}],
    )

    with pytest.raises(
        DatasetLoaderError, match="No SWE-rebench OpenHands trajectories"
    ):
        await _loader().convert_to_conversations({"dataset": [no_assistant]})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row_overrides, filters, match",
    [
        param({"trajectory_id": ""}, None, r"^row 1 trajectory_id='': trajectory_id must be", id="empty_id"),
        param({"trajectory": "not-a-list"}, None, r"^row 1 trajectory_id='traj-1': trajectory must be a list", id="trajectory_not_list"),
        param({"trajectory": [{"content": "no role"}]}, None, r"^row 1 trajectory_id='traj-1': message 0 must have a string role", id="missing_role"),
        param({"tools": "not-a-list"}, None, r"^row 1 trajectory_id='traj-1': tools must be a list", id="tools_not_list"),
        param({"resolved": 7}, {"resolved": "true"}, r"^row 1 trajectory_id='traj-1': resolved must be 0 or 1", id="resolved_out_of_range"),
    ],
)  # fmt: skip
async def test_malformed_row_raises_located_error(
    row_overrides: dict[str, Any],
    filters: dict[str, str] | None,
    match: str,
) -> None:
    row = {**_row(), **row_overrides}

    with pytest.raises(DatasetLoaderError, match=match):
        await _loader(filters).convert_to_conversations({"dataset": [row]})


@pytest.mark.asyncio
async def test_duplicate_trajectory_id_raises_error() -> None:
    with pytest.raises(DatasetLoaderError, match="duplicate trajectory_id"):
        await _loader().convert_to_conversations(
            {"dataset": [_row("same"), _row("same")]}
        )


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_DATASETS, reason="datasets has no Windows-on-ARM wheel")
async def test_huggingface_dataset_revision_is_pinned() -> None:
    load_dataset = MagicMock(return_value=[])
    with patch("datasets.load_dataset", load_dataset):
        _loader()._load_hf_dataset()

    load_dataset.assert_called_once_with(
        "nebius/SWE-rebench-openhands-trajectories",
        name=None,
        split="train",
        trust_remote_code=False,
        streaming=True,
        revision="35455389ab51bf5e2306bfd436ef72d0f98bf882",
    )


@pytest.mark.asyncio
async def test_requires_finite_materialization_bound() -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["target-model"],
            public_dataset=PublicDatasetType.SWE_REBENCH_OPENHANDS,
            benchmark_duration=60,
        )
    )
    loader = SweRebenchOpenHandsDatasetLoader(
        run=run,
        hf_dataset_name="nebius/SWE-rebench-openhands-trajectories",
        streaming=True,
    )

    with pytest.raises(DatasetLoaderError, match="finite entry or request count"):
        loader._max_conversations()


def _sessions_only_run(tmp_path: Path, sessions: str) -> Any:
    config_file = tmp_path / "base.yaml"
    config_file.write_text(
        f"""
schemaVersion: "2.0"
benchmark:
  model: target-model
  endpoint:
    url: http://localhost:8000
    type: chat
  dataset:
    type: public
    dataset: swe_rebench_openhands
  profiling:
    type: concurrency
    concurrency: 1
    sessions: {sessions}
"""
    )
    return make_run_from_cli(CLIConfig(config_file=config_file))


def test_profiling_sessions_bound_materialization_without_entries_or_requests(
    tmp_path: Path,
) -> None:
    loader = SweRebenchOpenHandsDatasetLoader(
        run=_sessions_only_run(tmp_path, "7"),
        hf_dataset_name="nebius/SWE-rebench-openhands-trajectories",
        streaming=True,
    )

    assert loader._max_conversations() == 7


def test_full_pass_profiling_sessions_is_not_a_materialization_bound(
    tmp_path: Path,
) -> None:
    loader = SweRebenchOpenHandsDatasetLoader(
        run=_sessions_only_run(tmp_path, "all"),
        hf_dataset_name="nebius/SWE-rebench-openhands-trajectories",
        streaming=True,
    )

    with pytest.raises(DatasetLoaderError, match="finite entry or request count"):
        loader._max_conversations()
