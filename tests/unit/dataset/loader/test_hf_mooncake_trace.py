# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent trajectories must reuse Mooncake's recorded request semantics."""

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import orjson
import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.config.flags import CLIConfig
from aiperf.config.flags.resolver import resolve_config
from aiperf.dataset.loader.hf_mooncake_trace import (
    HFMooncakeTraceDatasetLoader,
    trajectory_to_mooncake,
)
from aiperf.dataset.loader.mooncake_trace import MooncakeTraceDatasetLoader
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.workers.session_manager import UserSessionManager
from tests.unit.conftest import make_run_from_cli


@pytest.fixture
def row() -> dict[str, Any]:
    return {
        "session_id": "swe:1",
        "model": "google/gemma-4-26B-A4B-it",
        "benchmark": "SWE-Bench Multilingual",
        "messages": [
            {"role": "system", "content": "system", "tool_calls": None},
            {
                "role": "user",
                "content": "question",
                "reasoning_content": None,
                "tool_call_id": None,
                "name": None,
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "recorded reasoning",
                "tool_calls": [
                    {
                        "id": "call1",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "content": "result", "tool_call_id": "call1"},
            {"role": "assistant", "content": "recorded reply"},
            {"role": "tool", "content": "result2", "tool_call_id": "call2"},
            {"role": "assistant", "content": "recorded final reply"},
            {"role": "tool", "content": "trailing result", "tool_call_id": "call3"},
        ],
        "tools": '[{"type":"function","function":{"name":"bash"}}]',
        "output_length": 2048,
        "extra": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": -1,
            "seed": 42,
            "stream": True,
        },
    }


def loader(**kwargs: Any) -> HFMooncakeTraceDatasetLoader:
    return HFMooncakeTraceDatasetLoader(
        hf_dataset_name="example/trajectories",
        hf_split="test",
        streaming=True,
        prompt_generator=Mock(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_recorded_prefixes_preserve_payload_and_discard_live_replies(
    row: dict[str, Any],
) -> None:
    before = deepcopy(row)
    selected = loader()
    data = selected._validate_rows([row])
    conversation = (await selected.convert_to_conversations(data))[0]
    assert row == before
    assert (
        conversation.context_mode
        == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
    )
    assert len(conversation.turns) == 3
    session = UserSessionManager().create_and_store(
        x_correlation_id="test", conversation=conversation, num_turns=3
    )
    endpoint = ChatEndpoint(model_endpoint=Mock())
    for index, end in enumerate([2, 4, 6]):
        session.advance_turn(index)
        assert not session.should_store_response()
        messages = endpoint.build_messages(session.turn_list)
        assert len(messages) == end
        assert messages[:2] == row["messages"][:2]
        assert [m["role"] for m in messages] == [
            m["role"] for m in row["messages"][:end]
        ]
        turn = conversation.turns[index]
        assert turn.max_tokens == 2048
        assert turn.raw_tools == orjson.loads(row["tools"])
        assert turn.extra_body == {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": -1,
            "seed": 42,
        }
        assert turn.model is None
        if index:
            assert messages[2]["reasoning_content"] == "recorded reasoning"
            assert messages[2]["content"] is None
            assert "index" not in messages[2]["tool_calls"][0]
            assert messages[3]["tool_call_id"] == "call1"
    assert conversation.turns[0].raw_messages[0]["tool_calls"] is None


@pytest.mark.parametrize(
    "roles,lengths",
    [
        param(["user", "assistant"], [1], id="final_assistant"),
        param(["system", "user", "assistant", "tool", "assistant", "tool", "assistant", "tool"], [2, 4, 6], id="original_example"),
        param(["user", "user", "assistant", "assistant", "tool", "tool", "assistant"], [2, 3, 6], id="consecutive_roles"),
    ],
)  # fmt: skip
def test_boundary_splitting(
    roles: list[str], lengths: list[int], row: dict[str, Any]
) -> None:
    row["messages"] = [
        {"role": role, "content": str(i)} for i, role in enumerate(roles)
    ]
    assert [len(t.messages) for t in trajectory_to_mooncake(row)] == lengths


@pytest.mark.parametrize(
    "patch,error",
    [
        param({"messages": []}, "nonempty", id="empty"),
        param({"messages": [{"role": "assistant", "content": "x"}]}, "message 0", id="empty_prefix"),
        param({"messages": [{"role": "user", "content": "x"}]}, "no assistant", id="no_boundary"),
        param({"messages": [None]}, "message 0", id="invalid_message"),
        param({"tools": "invalid JSON"}, "session", id="tools_json"),
        param({"extra": '[]'}, "object", id="params_type"),
        param({"extra": '{"messages":[]}'}, "override", id="history_override"),
        param({"extra": '{"model":"other"}'}, "override", id="model_override"),
        param({"output_length": 0}, "output_length", id="invalid_cap"),
    ],
)  # fmt: skip
def test_invalid_rows_fail_with_session(
    patch: dict[str, Any], error: str, row: dict[str, Any]
) -> None:
    with pytest.raises(DatasetLoaderError, match=error) as caught:
        trajectory_to_mooncake(row | patch)
    assert "swe:1" in str(caught.value)


def test_source_column_filters_and_explicit_subset(row: dict[str, Any]) -> None:
    selected = loader(
        filters={
            "model": "google/gemma-4-26B-A4B-it",
            "benchmark": "SWE-Bench Multilingual",
        }
    )
    assert selected.hf_subset is None
    assert selected.hf_revision is None
    rows = [row | {"model": "other"}, row | {"benchmark": "other"}, row]
    assert len(selected._validate_rows(rows)) == 1
    assert loader(hf_subset="custom-config").hf_subset == "custom-config"
    with pytest.raises(DatasetLoaderError, match="no trajectories matched"):
        selected._validate_rows([row | {"benchmark": "other"}])
    with pytest.raises(DatasetLoaderError, match="source path.*missing"):
        loader(filters={"missing": "x"})._validate_rows([row])


def test_duplicate_ids_rejected(row: dict[str, Any]) -> None:
    with pytest.raises(DatasetLoaderError, match="duplicate session_id"):
        loader()._validate_rows([row, row])


def test_session_limit_does_not_truncate_export(row: dict[str, Any]) -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["target"],
            public_dataset="mooncake_hf",
            hf_mooncake_dataset="example/trajectories",
            conversation_num=1,
        )
    )

    rows = [row, row | {"session_id": "second"}]
    assert len(loader(run=run)._validate_rows(rows)) == 2


@pytest.mark.asyncio
async def test_hf_pipeline_delegates_and_matches_mooncake_file(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, Any], tmp_path: Path
) -> None:
    rows = [row, row | {"session_id": "second"}]
    selected = loader()
    monkeypatch.setattr(selected, "_load_hf_dataset", lambda: rows)
    load = Mock(wraps=selected._mooncake.load_dataset)
    convert = Mock(wraps=selected._mooncake.convert_to_conversations)
    monkeypatch.setattr(selected._mooncake, "load_dataset", load)
    monkeypatch.setattr(selected._mooncake, "convert_to_conversations", convert)

    data = await selected.load_dataset()
    assert list(data) == ["swe:1", "second"]
    assert all(len(traces) == 3 for traces in data.values())
    load.assert_called_once_with()
    assert selected._mooncake.inline_records == []
    conversations = await selected.convert_to_conversations(data)
    convert.assert_called_once_with(data)

    filename = tmp_path / "traces.jsonl"
    filename.write_bytes(
        b"".join(
            orjson.dumps(
                trace.model_dump(exclude_none=True), option=orjson.OPT_APPEND_NEWLINE
            )
            for source in rows
            for trace in trajectory_to_mooncake(source)
        )
    )
    file_loader = MooncakeTraceDatasetLoader(
        filename=filename, run=selected.run, prompt_generator=Mock()
    )
    assert conversations == file_loader.convert_to_conversations(
        file_loader.load_dataset()
    )


def test_inline_records_released_after_load_failure(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, Any]
) -> None:
    selected = loader()
    monkeypatch.setattr(
        selected._mooncake,
        "load_dataset",
        Mock(side_effect=ValueError("invalid trace")),
    )
    with pytest.raises(ValueError, match="invalid trace"):
        selected._validate_rows([row])
    assert selected._mooncake.inline_records == []


@pytest.mark.asyncio
async def test_hf_loading_uses_explicit_source(
    monkeypatch: pytest.MonkeyPatch, row: dict[str, Any]
) -> None:
    datasets = pytest.importorskip("datasets")
    load = Mock(return_value=[row])
    monkeypatch.setattr(datasets, "load_dataset", load)
    selected = loader(hf_subset="my-config")
    data = await selected.load_dataset()
    conversations = await selected.convert_to_conversations(data)
    assert len(conversations[0].turns) == 3
    assert load.call_args.args == ("example/trajectories",)
    assert load.call_args.kwargs["name"] == "my-config"
    assert "revision" not in load.call_args.kwargs
    assert load.call_args.kwargs["split"] == "test"
    assert load.call_args.kwargs["streaming"] is True


@pytest.mark.parametrize("flags,error", [
    param({"endpoint_type": "completions"}, "chat", id="endpoint"),
    param({"fixed_schedule": True}, "no recorded timing", id="timing"),
    param({"extra_inputs": {"messages": []}}, "override", id="messages"),
    param({"extra_inputs": {"model": "other"}}, "override", id="model"),
])  # fmt: skip
def test_incompatible_config(flags: dict[str, Any], error: str) -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["target"],
            public_dataset="mooncake_hf",
            hf_mooncake_dataset="example/trajectories",
            **flags,
        )
    )
    with pytest.raises(DatasetLoaderError, match=error):
        loader(run=run)


def test_filters_route_from_yaml_and_cli_override(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("""schemaVersion: "2.0"
benchmark:
  model: target
  endpoint: {type: chat, url: "http://localhost:8000"}
  dataset:
    type: public
    dataset: mooncake_hf
    hf_mooncake_dataset: example/trajectories
    filters: {model: google/gemma-4-26B-A4B-it}
  profiling: {type: concurrency, concurrency: 2, sessions: 3}
""")
    cfg = resolve_config(
        CLIConfig(
            dataset_filters=[
                "model=google/gemma-4-26B-A4B-it",
                "benchmark=SWE-Bench Multilingual",
            ]
        ),
        config,
    )
    dataset = cfg.benchmark.get_default_dataset()
    assert dataset.filters == {
        "model": "google/gemma-4-26B-A4B-it",
        "benchmark": "SWE-Bench Multilingual",
    }


@pytest.mark.parametrize("params", [
    param({"max_tokens": 8}, id="max_tokens"),
    param({"max_completion_tokens": 8}, id="max_completion_tokens"),
    param({"max_tokens": None, "max_completion_tokens": 8}, id="null_legacy"),
])  # fmt: skip
def test_output_cap_falls_back_to_request_params(
    params: dict[str, Any], row: dict[str, Any]
) -> None:
    row["output_length"] = None
    row["extra"] = orjson.dumps(params).decode()
    for trace in trajectory_to_mooncake(row):
        assert trace.output_length == 8
        assert trace.extra == {}


def test_duration_only_loads_available_rows(row: dict[str, Any]) -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["target"],
            public_dataset="mooncake_hf",
            hf_mooncake_dataset="example/trajectories",
            benchmark_duration=10,
        )
    )
    traces = loader(run=run)._validate_rows([row])
    assert len(traces) == 1
    assert len(traces[row["session_id"]]) == 3


@pytest.mark.parametrize("encoded", [False, True])
def test_mapped_columns_match_canonical_conversations(
    row: dict[str, Any], encoded: bool
) -> None:
    source = {
        "question_id": row["session_id"],
        "turns": row["messages"],
        "extras": {
            "tools": row["tools"],
            "max_tokens": row["output_length"],
            "request_params": orjson.dumps(
                row["extra"] | {"temperature": 0.1}
            ).decode(),
            "temperature": 0.7,
            "top_p": None,
        },
    }
    if encoded:
        source["extras"] = orjson.dumps(source["extras"]).decode()
        source["turns"] = orjson.dumps(source["turns"]).decode()
    before = deepcopy(source)
    mapped = loader(
        field_mapping={
            "session_id": "question_id",
            "messages": "turns",
            "tools": "extras.tools",
            "output_length": "extras.max_tokens",
            "extra": "extras.request_params",
            "extra.temperature": "extras.temperature",
            "extra.top_p": "extras.top_p",
        }
    )._validate_rows([source])
    assert mapped == loader()._validate_rows([row])
    assert source == before


@pytest.mark.parametrize("mapping,error", [
    param({"messages": "missing"}, "source path.*missing", id="missing_source"),
    param({"extra.model": "model"}, "must not override", id="mapped_model_override"),
])  # fmt: skip
def test_invalid_mapped_row_fails_clearly(
    mapping: dict[str, str], error: str, row: dict[str, Any]
) -> None:
    with pytest.raises(DatasetLoaderError, match=error):
        loader(field_mapping=mapping)._validate_rows([row])


def test_absent_session_ids_are_generated(row: dict[str, Any]) -> None:
    row.pop("session_id")
    traces = loader()._validate_rows([row, row])
    assert list(traces) == ["hf-row-0", "hf-row-1"]


def test_nested_source_filter(row: dict[str, Any]) -> None:
    row["metadata"] = '{"model":"selected"}'
    assert (
        len(loader(filters={"metadata.model": "selected"})._validate_rows([row])) == 1
    )
