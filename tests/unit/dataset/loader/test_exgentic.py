# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import orjson
import pytest
from pytest import param

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.loader.exgentic import (
    ExgenticDatasetLoader,
    canonical_source_model,
)
from aiperf.dataset.loader.exgentic_v2 import ExgenticV2DatasetLoader
from aiperf.plugin.enums import PublicDatasetType
from tests.unit.conftest import make_run_from_cli


def _message_json(messages: list[dict[str, Any]]) -> str:
    return orjson.dumps(messages).decode()


def _span(
    start: str,
    end: str,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    input_tokens: int = 100,
    output_tokens: int = 20,
    status: int = 1,
    span_type: str | None = "llm_call",
    request_max_tokens: int | None = None,
    temperature: float | None = None,
    stop_sequences: list[str] | None = None,
    system_instructions: str | None = None,
) -> dict[str, Any]:
    span = {
        "start_time": start,
        "end_time": end,
        "status": {"code": status},
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.input.messages": _message_json(messages),
            "gen_ai.output.messages": _message_json(
                [
                    {
                        "role": "assistant",
                        "parts": [{"type": "text", "content": "recorded output"}],
                    }
                ]
            ),
            "gen_ai.tool.definitions": _message_json(tools or []),
            "gen_ai.usage.input_tokens": input_tokens,
            "gen_ai.usage.output_tokens": output_tokens,
        },
    }
    if span_type is not None:
        span["type"] = span_type
    for key, value in {
        "gen_ai.request.max_tokens": request_max_tokens,
        "gen_ai.request.temperature": temperature,
        "gen_ai.request.stop_sequences": stop_sequences,
        "gen_ai.system_instructions": system_instructions,
    }.items():
        if value is not None:
            span["attributes"][key] = value
    return span


def _row(
    session_id: str,
    spans: list[dict[str, Any]],
    *,
    harness: str = "tool_calling",
    models: list[str] | None = None,
    benchmark: str | None = None,
) -> dict[str, Any]:
    row = {
        "harness": harness,
        "models": models or ["openai/azure/Kimi-K2.5"],
        "session_id": session_id,
        "spans": spans,
    }
    if benchmark is not None:
        row["benchmark"] = benchmark
    return row


def _loader(filters: dict[str, str] | None = None) -> ExgenticDatasetLoader:
    return ExgenticDatasetLoader(
        filters=filters,
        hf_dataset_name="Exgentic/agent-llm-traces",
        streaming=True,
    )


@pytest.mark.asyncio
async def test_convert_preserves_snapshots_tools_osl_order_and_delays() -> None:
    tools = [
        {
            "type": "function",
            "name": "search",
            "description": "Search records",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
    ]
    rich_messages = [
        {"role": "developer", "parts": [{"type": "text", "content": "policy"}]},
        {
            "role": "assistant",
            "parts": [
                {"type": "thinking", "thinking": "reason", "signature": None},
                {"type": "text", "content": "calling"},
                {"type": "tool_call", "id": "call-1", "name": "search", "arguments": {"q": "x"}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"type": "tool_call_response", "id": "call-1", "result": [{"type": "text", "text": "ok"}]},
                {"type": "text", "content": "reminder"},
            ],
        },
    ]  # fmt: skip
    simple_messages = [{"role": "user", "parts": [{"type": "text", "content": "next"}]}]
    spans = [
        _span(
            "2026-01-01T00:00:20Z",
            "2026-01-01T00:00:21Z",
            messages=simple_messages,
            output_tokens=30,
        ),
        _span(
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:10Z",
            messages=rich_messages,
            tools=tools,
            output_tokens=10,
        ),
        _span(
            "2026-01-01T00:00:05Z",
            "2026-01-01T00:00:07Z",
            messages=simple_messages,
            output_tokens=20,
        ),
        _span(
            "2026-01-01T00:00:30Z",
            "2026-01-01T00:00:31Z",
            messages=simple_messages,
            status=2,
        ),
        _span(
            "2026-01-01T00:00:32Z",
            "2026-01-01T00:00:33Z",
            messages=simple_messages,
            output_tokens=0,
        ),
        _span(
            "2026-01-01T00:00:34Z",
            "2026-01-01T00:00:35Z",
            messages=simple_messages,
            span_type="tool_call",
        ),
    ]

    conversations = await _loader().convert_to_conversations(
        {"dataset": [_row("session-1", spans)]}
    )

    conversation = conversations[0]
    assert (
        conversation.context_mode
        == ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
    )
    assert [turn.max_tokens for turn in conversation.turns] == [10, 20, 30]
    assert [turn.delay for turn in conversation.turns] == [None, 0, 13_000]
    first = conversation.turns[0]
    assert first.raw_messages == [
        {"role": "system", "content": "policy"},
        {
            "role": "assistant",
            "content": "calling",
            "reasoning_content": "reason",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"x"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '[{"type":"text","text":"ok"}]',
        },
        {"role": "user", "content": "reminder"},
    ]
    assert first.raw_tools == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search records",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    ]
    assert all(
        turn.extra_headers == {"x-dynamo-session-id": "session-1"}
        for turn in conversation.turns
    )
    assert all(
        "recorded output" not in orjson.dumps(turn.raw_messages).decode()
        for turn in conversation.turns
    )


@pytest.mark.asyncio
async def test_fixed_schedule_preserves_overlapping_start_times() -> None:
    messages = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
    spans = [
        _span("2026-01-01T00:00:20Z", "2026-01-01T00:00:21Z", messages=messages),
        _span("2026-01-01T00:00:00Z", "2026-01-01T00:00:10Z", messages=messages),
        _span("2026-01-01T00:00:05Z", "2026-01-01T00:00:07Z", messages=messages),
    ]
    run = make_run_from_cli(
        CLIConfig(
            model_names=["target-model"],
            public_dataset=PublicDatasetType.EXGENTIC,
            conversation_num=1,
            fixed_schedule=True,
        )
    )
    loader = ExgenticDatasetLoader(
        run=run,
        hf_dataset_name="Exgentic/agent-llm-traces",
        streaming=True,
    )

    conversations = await loader.convert_to_conversations(
        {"dataset": [_row("session-1", spans)]}
    )

    assert [conversation.session_id for conversation in conversations] == [
        "session-1:1",
        "session-1:2",
        "session-1:0",
    ]
    assert [conversation.turns[0].timestamp for conversation in conversations] == [
        0,
        5_000,
        20_000,
    ]
    assert all(len(conversation.turns) == 1 for conversation in conversations)
    assert all(conversation.turns[0].delay is None for conversation in conversations)
    assert all(
        conversation.turns[0].extra_headers == {"x-dynamo-session-id": "session-1"}
        for conversation in conversations
    )


@pytest.mark.asyncio
async def test_filters_normalize_provider_aliases_and_limit_sessions() -> None:
    messages = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
    rows = [
        _row(
            "skip",
            [_span("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", messages=messages)],
            models=["gcp/gemini-3-pro-preview"],
        ),
        _row(
            "keep",
            [_span("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", messages=messages)],
            harness="tool_calling_with_shortlisting",
            models=["azure/Kimi-K2.5", "openai/azure/Kimi-K2.5"],
        ),
    ]
    loader = _loader(
        {
            "harness": "tool_calling_with_shortlisting",
            "source_model": "openai/azure/Kimi-K2.5",
        }
    )

    conversations = await loader.convert_to_conversations({"dataset": rows})

    assert [conversation.session_id for conversation in conversations] == ["keep"]


@pytest.mark.asyncio
async def test_v2_converts_otel_spans_with_request_controls_and_benchmark_filter() -> (
    None
):
    loader = ExgenticV2DatasetLoader(
        filters={
            "benchmark": "swebench",
            "harness": "openai_solo",
            "source_model": "gpt-5.2-2025-12-11",
        },
        hf_dataset_name="Exgentic/agent-llm-traces-v2",
        streaming=True,
    )
    instructions = _message_json(
        [{"type": "text", "content": "Follow the repository policy."}]
    )
    conversations = await loader.convert_to_conversations(
        {
            "dataset": [
                _row(
                    "skip",
                    [
                        _span(
                            "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:01Z",
                            messages=[
                                {
                                    "role": "user",
                                    "parts": [{"type": "text", "content": "skip"}],
                                }
                            ],
                            span_type=None,
                        )
                    ],
                    harness="openai_solo",
                    models=["gpt-5.2-2025-12-11"],
                    benchmark="appworld",
                ),
                _row(
                    "keep",
                    [
                        _span(
                            "2026-01-01T00:00:00Z",
                            "2026-01-01T00:00:01Z",
                            messages=[
                                {
                                    "role": "user",
                                    "parts": [{"type": "text", "content": "keep"}],
                                }
                            ],
                            span_type=None,
                            request_max_tokens=512,
                            temperature=0.7,
                            stop_sequences=["END"],
                            system_instructions=instructions,
                        )
                    ],
                    harness="openai_solo",
                    models=["gpt-5.2-2025-12-11"],
                    benchmark="swebench",
                ),
            ]
        }
    )

    assert len(conversations) == 1
    turn = conversations[0].turns[0]
    assert conversations[0].session_id == "keep"
    assert turn.max_tokens == 512
    assert turn.extra_body == {"temperature": 0.7, "stop": ["END"]}
    assert turn.raw_messages == [
        {"role": "system", "content": "Follow the repository policy."},
        {"role": "user", "content": "keep"},
    ]


@pytest.mark.parametrize(
    "source, expected",
    [
        param("Azure/gpt-4.1", "gpt-4.1"),
        param("openai/Azure/gpt-4.1", "gpt-4.1"),
        param("aws/claude-opus-4-5", "claude-opus-4-5"),
        param("gcp/gemini-3-pro-preview", "gemini-3-pro-preview"),
    ],
)  # fmt: skip
def test_canonical_source_model_strips_provider_aliases(
    source: str, expected: str
) -> None:
    assert canonical_source_model(source) == expected


def test_invalid_filter_lists_typed_values() -> None:
    with pytest.raises(DatasetLoaderError, match=r"available filters:.*Kimi-K2.5"):
        _loader({"source_model": "unknown"})


def test_v2_invalid_filter_lists_benchmark() -> None:
    with pytest.raises(DatasetLoaderError) as error:
        ExgenticV2DatasetLoader(
            filters={"source_model": "unknown"},
            hf_dataset_name="Exgentic/agent-llm-traces-v2",
            streaming=True,
        )
    available_filters = str(error.value).split("; available filters: ", 1)[1]
    assert "benchmark=[appworld, browsecompplus, swebench" in available_filters
    assert "gpt-4.1" not in available_filters


def test_v1_unsupported_filter_pair_fails_before_loading() -> None:
    with pytest.raises(DatasetLoaderError, match=r"Unsupported.*available source"):
        _loader(
            {
                "harness": "tool_calling_with_shortlisting",
                "source_model": "gpt-4.1",
            }
        )


def test_v2_unsupported_filter_pair_fails_before_loading() -> None:
    with pytest.raises(DatasetLoaderError, match=r"Unsupported.*DeepSeek-V3\.2"):
        ExgenticV2DatasetLoader(
            filters={"harness": "tool_calling", "source_model": "gpt-4.1"},
            hf_dataset_name="Exgentic/agent-llm-traces-v2",
            streaming=True,
        )


def test_v2_unsupported_source_model_fails_before_loading() -> None:
    with pytest.raises(DatasetLoaderError, match="no harness supports it"):
        ExgenticV2DatasetLoader(
            filters={"source_model": "gpt-4.1"},
            hf_dataset_name="Exgentic/agent-llm-traces-v2",
            streaming=True,
        )


def test_v1_benchmark_filter_fails_before_loading() -> None:
    with pytest.raises(DatasetLoaderError, match="only supported for Exgentic v2"):
        _loader({"benchmark": "appworld"})


@pytest.mark.asyncio
async def test_boolean_requested_max_tokens_raises_dataset_error() -> None:
    with pytest.raises(DatasetLoaderError, match="must be a positive integer"):
        await _loader().convert_to_conversations(
            {
                "dataset": [
                    _row(
                        "session-1",
                        [
                            _span(
                                "2026-01-01T00:00:00Z",
                                "2026-01-01T00:00:01Z",
                                messages=[
                                    {
                                        "role": "user",
                                        "parts": [{"type": "text", "content": "hi"}],
                                    }
                                ],
                                request_max_tokens=True,
                            )
                        ],
                    )
                ]
            }
        )


@pytest.mark.asyncio
async def test_explicit_entries_win_over_request_count() -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["target-model"],
            public_dataset=PublicDatasetType.EXGENTIC,
            conversation_num=1,
            request_count=6,
        )
    )
    loader = ExgenticDatasetLoader(
        run=run,
        hf_dataset_name="Exgentic/agent-llm-traces",
        streaming=True,
    )

    assert loader._max_conversations() == 1


@pytest.mark.asyncio
async def test_requires_finite_materialization_bound() -> None:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["target-model"],
            public_dataset=PublicDatasetType.EXGENTIC,
            benchmark_duration=60,
        )
    )
    loader = ExgenticDatasetLoader(
        run=run,
        hf_dataset_name="Exgentic/agent-llm-traces",
        streaming=True,
    )

    with pytest.raises(DatasetLoaderError, match="finite entry or request count"):
        loader._max_conversations()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, value, match",
    [
        param("models", "Kimi-K2.5", "models must be a list"),
        param("spans", ["not-an-object"], "spans must be a list of objects"),
    ],
)  # fmt: skip
async def test_invalid_row_shapes_raise_clear_error(
    field: str, value: Any, match: str
) -> None:
    messages = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
    row = _row(
        "session-1",
        [_span("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", messages=messages)],
    )
    row[field] = value

    with pytest.raises(DatasetLoaderError, match=match):
        await _loader().convert_to_conversations({"dataset": [row]})


@pytest.mark.asyncio
async def test_huggingface_dataset_revision_is_pinned() -> None:
    load_dataset = MagicMock(return_value=[])
    with patch("aiperf.dataset.loader.base_hf_dataset.hf_load_dataset", load_dataset):
        _loader()._load_hf_dataset()

    load_dataset.assert_called_once_with(
        "Exgentic/agent-llm-traces",
        name=None,
        split="train",
        trust_remote_code=False,
        streaming=True,
        revision="70036b93a04e61b0ea2706a68b962f4f26774587",
    )


@pytest.mark.asyncio
async def test_v2_huggingface_dataset_revision_is_pinned() -> None:
    load_dataset = MagicMock(return_value=[])
    with patch("aiperf.dataset.loader.base_hf_dataset.hf_load_dataset", load_dataset):
        ExgenticV2DatasetLoader(
            hf_dataset_name="Exgentic/agent-llm-traces-v2",
            streaming=True,
        )._load_hf_dataset()

    load_dataset.assert_called_once_with(
        "Exgentic/agent-llm-traces-v2",
        name=None,
        split="train",
        trust_remote_code=False,
        streaming=True,
        revision="4b8ad4ab198438e5a170f9171c19c6a2cf7c1814",
    )


@pytest.mark.asyncio
async def test_unavailable_combination_lists_available_combinations() -> None:
    messages = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
    rows = [
        _row(
            "session-1",
            [_span("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", messages=messages)],
            harness="claude_code",
            models=["Azure/gpt-4.1"],
        )
    ]

    with pytest.raises(
        DatasetLoaderError, match=r"available combinations: claude_code/gpt-4.1"
    ):
        await _loader(
            {"harness": "openai_solo", "source_model": "Kimi-K2.5"}
        ).convert_to_conversations({"dataset": rows})


@pytest.mark.asyncio
async def test_max_conversations_stops_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = [{"role": "user", "parts": [{"type": "text", "content": "hi"}]}]
    rows = [
        _row(
            f"session-{index}",
            [_span("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", messages=messages)],
        )
        for index in range(3)
    ]
    loader = _loader()
    monkeypatch.setattr(loader, "_max_conversations", lambda: 1)

    conversations = await loader.convert_to_conversations({"dataset": rows})

    assert [conversation.session_id for conversation in conversations] == ["session-0"]
