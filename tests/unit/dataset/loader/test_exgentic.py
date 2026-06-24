# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import orjson
import pytest

from aiperf.common.enums import ConversationContextMode
from aiperf.common.exceptions import DatasetLoaderError
from aiperf.dataset.loader.exgentic import (
    ExgenticDatasetLoader,
    canonical_source_model,
)


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
    span_type: str = "llm_call",
) -> dict[str, Any]:
    return {
        "start_time": start,
        "end_time": end,
        "type": span_type,
        "status": {"code": status},
        "attributes": {
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


def _row(
    session_id: str,
    spans: list[dict[str, Any]],
    *,
    harness: str = "tool_calling",
    models: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "harness": harness,
        "models": models or ["openai/azure/Kimi-K2.5"],
        "session_id": session_id,
        "spans": spans,
    }


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
        "recorded output" not in orjson.dumps(turn.raw_messages).decode()
        for turn in conversation.turns
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


@pytest.mark.parametrize(
    "source, expected",
    [
        ("Azure/gpt-4.1", "gpt-4.1"),
        ("openai/Azure/gpt-4.1", "gpt-4.1"),
        ("aws/claude-opus-4-5", "claude-opus-4-5"),
        ("gcp/gemini-3-pro-preview", "gemini-3-pro-preview"),
    ],
)
def test_canonical_source_model_strips_provider_aliases(
    source: str, expected: str
) -> None:
    assert canonical_source_model(source) == expected


def test_invalid_filter_lists_typed_values() -> None:
    with pytest.raises(DatasetLoaderError, match=r"available filters:.*Kimi-K2.5"):
        _loader({"source_model": "unknown"})


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
