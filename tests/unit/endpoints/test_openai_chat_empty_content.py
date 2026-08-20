# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import orjson
import pytest
from pytest import param

from aiperf.common.models import (
    BaseResponseData,
    ReasoningResponseData,
    SSEMessage,
    TextResponseData,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint


def _endpoint(*, enabled: bool) -> ChatEndpoint:
    model_endpoint = create_model_endpoint(EndpointType.CHAT, streaming=True)
    model_endpoint.endpoint.allow_empty_content = enabled
    return ChatEndpoint(model_endpoint)


def _chunk(delta: dict[str, object], *, finish_reason: str | None = None) -> SSEMessage:
    payload = {
        "object": "chat.completion.chunk",
        "choices": [{"delta": delta, "finish_reason": finish_reason}],
    }
    return SSEMessage.parse(
        b"data: " + orjson.dumps(payload) + b"\n\n",
        perf_ns=100,
    )


@pytest.mark.parametrize(
    ("delta", "expected_type"),
    [
        param({"content": ""}, TextResponseData, id="empty-content"),
        param(
            {"reasoning_content": ""},
            ReasoningResponseData,
            id="empty-reasoning-content",
        ),
        param({"reasoning": ""}, ReasoningResponseData, id="empty-reasoning-alias"),
        param(
            {"content": "", "reasoning_content": ""},
            ReasoningResponseData,
            id="both-empty",
        ),
    ],
)  # fmt: skip
def test_enabled_retains_explicit_empty_strings(
    delta: dict[str, object], expected_type: type[BaseResponseData]
) -> None:
    parsed = _endpoint(enabled=True).parse_response(_chunk(delta))

    assert parsed is not None
    assert isinstance(parsed.data, expected_type)


@pytest.mark.parametrize(
    "delta",
    [
        param({"content": ""}, id="empty-content"),
        param({"reasoning_content": ""}, id="empty-reasoning-content"),
        param({"reasoning": ""}, id="empty-reasoning-alias"),
    ],
)  # fmt: skip
def test_disabled_discards_explicit_empty_strings(delta: dict[str, object]) -> None:
    assert _endpoint(enabled=False).parse_response(_chunk(delta)) is None


@pytest.mark.parametrize(
    "delta",
    [
        param({"role": "assistant"}, id="role-only"),
        param({"content": None}, id="null-content"),
        param({}, id="empty-delta"),
        param({"tool_calls": []}, id="empty-tool-list"),
    ],
)  # fmt: skip
def test_enabled_still_discards_non_generation_deltas(
    delta: dict[str, object],
) -> None:
    assert _endpoint(enabled=True).parse_response(_chunk(delta)) is None


def test_enabled_retains_terminal_explicit_empty_content() -> None:
    parsed = _endpoint(enabled=True).parse_response(
        _chunk({"content": ""}, finish_reason="stop")
    )

    assert parsed is not None
    assert parsed.data == TextResponseData(text="")


def test_non_streaming_empty_content_remains_excluded() -> None:
    endpoint = _endpoint(enabled=True)
    payload = {
        "object": "chat.completion",
        "choices": [{"message": {"role": "assistant", "content": ""}}],
    }
    response = SSEMessage.parse(
        b"data: " + orjson.dumps(payload) + b"\n\n", perf_ns=100
    )

    assert endpoint.parse_response(response) is None


def test_non_empty_reasoning_alias_wins_over_empty_reasoning_content() -> None:
    parsed = _endpoint(enabled=True).parse_response(
        _chunk({"content": "", "reasoning_content": "", "reasoning": "thinking"})
    )

    assert parsed is not None
    assert parsed.data == ReasoningResponseData(content="", reasoning="thinking")


def test_enabled_finish_only_chunk_remains_excluded() -> None:
    assert (
        _endpoint(enabled=True).parse_response(_chunk({}, finish_reason="stop")) is None
    )


def test_enabled_usage_only_chunk_has_no_content_data() -> None:
    payload = {
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {"completion_tokens": 7},
    }
    response = SSEMessage.parse(
        b"data: " + orjson.dumps(payload) + b"\n\n", perf_ns=100
    )

    parsed = _endpoint(enabled=True).parse_response(response)

    assert parsed is not None
    assert parsed.data is None
    assert parsed.usage is not None


def test_enabled_done_marker_remains_excluded() -> None:
    assert (
        _endpoint(enabled=True).parse_response(
            SSEMessage.parse(b"data: [DONE]\n\n", perf_ns=100)
        )
        is None
    )


@pytest.mark.parametrize(
    "payload",
    [
        param(
            {"object": "chat.completion.chunk", "choices": ["bad"]},
            id="non-object-choice",
        ),
        param(
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": "bad"}],
            },
            id="non-object-delta",
        ),
    ],
)  # fmt: skip
def test_enabled_discards_malformed_stream_events(
    payload: dict[str, object],
) -> None:
    response = SSEMessage.parse(
        b"data: " + orjson.dumps(payload) + b"\n\n", perf_ns=100
    )

    assert _endpoint(enabled=True).parse_response(response) is None
