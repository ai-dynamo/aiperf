# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BaseEndpoint.is_truncated and its per-endpoint overrides."""

from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.common.models import RequestRecord
from aiperf.common.models.record_models import SSEField, SSEMessage, TextResponse
from aiperf.endpoints.anthropic_messages import MessagesEndpoint
from aiperf.endpoints.base_endpoint import BaseEndpoint
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_model_endpoint


def _endpoint(cls: type[BaseEndpoint], endpoint_type: EndpointType) -> BaseEndpoint:
    return cls(model_endpoint=create_model_endpoint(endpoint_type))


def _body(json_data: dict[str, Any]) -> RequestRecord:
    return RequestRecord(
        responses=[
            TextResponse(
                perf_ns=1,
                text=orjson.dumps(json_data).decode(),
                content_type="application/json",
            )
        ]
    )


def _stream(*events: dict[str, Any] | str) -> RequestRecord:
    return RequestRecord(
        responses=[
            SSEMessage(
                perf_ns=i,
                packets=[
                    SSEField(
                        name="data",
                        value=e if isinstance(e, str) else orjson.dumps(e).decode(),
                    )
                ],
            )
            for i, e in enumerate(events)
        ]
    )


def _chat_chunk(
    finish_reason: str | None, delta: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }


def _completion_chunk(finish_reason: str | None) -> dict[str, Any]:
    return {
        "object": "text_completion",
        "choices": [{"index": 0, "text": "x", "finish_reason": finish_reason}],
    }


def _response_obj(
    status: str, reason: str | None = None, **extra: Any
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "status": status,
        "incomplete_details": {"reason": reason} if reason else None,
        **extra,
    }


def _anthropic_stream(stop_reason: str | None) -> RequestRecord:
    return _stream(
        {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )  # fmt: skip


@pytest.mark.parametrize(
    "record, expected",
    [
        param(_body({"object": "chat.completion", "choices": [{"index": 0, "message": {"content": "x"}, "finish_reason": "length"}]}), True, id="non_streaming_length"),
        param(_body({"object": "chat.completion", "choices": [{"index": 0, "message": {"content": "x"}, "finish_reason": "stop"}]}), False, id="non_streaming_stop"),
        param(_body({"object": "chat.completion", "choices": [{"index": 0, "message": {"content": "x"}}]}), False, id="non_streaming_no_finish_reason"),
        param(_stream(_chat_chunk(None, {"content": "a"}), _chat_chunk("length"), "[DONE]"), True, id="streaming_length"),
        param(_stream(_chat_chunk(None, {"content": "a"}), _chat_chunk("stop"), "[DONE]"), False, id="streaming_stop"),
        param(_stream(_chat_chunk(None, {"content": "a"}), _chat_chunk(None, {"content": "b"}), "[DONE]"), False, id="streaming_no_finish_reason"),
        param(_stream(_chat_chunk(None, {"content": "a"}), _chat_chunk("length"), {"object": "chat.completion.chunk", "choices": [], "usage": {"completion_tokens": 2}}, "[DONE]"), True, id="streaming_length_then_usage_chunk"),
        param(RequestRecord(responses=[]), False, id="no_responses"),
        param(_body({"object": "chat.completion", "choices": {"0": {"finish_reason": "length"}}}), False, id="malformed_choices_dict"),
        param(_body({"object": "chat.completion", "choices": 7}), False, id="malformed_choices_number"),
    ],
)  # fmt: skip
def test_is_truncated_chat_finish_reason_matches_expected(
    record: RequestRecord, expected: bool
) -> None:
    endpoint = _endpoint(ChatEndpoint, EndpointType.CHAT)
    assert endpoint.is_truncated(record) is expected


@pytest.mark.parametrize(
    "record, expected",
    [
        param(_body(_completion_chunk("length")), True, id="non_streaming_length"),
        param(_body(_completion_chunk("stop")), False, id="non_streaming_stop"),
    ],
)  # fmt: skip
def test_is_truncated_completions_finish_reason_matches_expected(
    record: RequestRecord, expected: bool
) -> None:
    endpoint = _endpoint(CompletionsEndpoint, EndpointType.COMPLETIONS)
    assert endpoint.is_truncated(record) is expected


@pytest.mark.parametrize(
    "record, expected",
    [
        param(_body(_response_obj("incomplete", "max_output_tokens")), True, id="non_streaming_max_output_tokens"),
        param(_body(_response_obj("incomplete", "content_filter")), False, id="non_streaming_incomplete_other_reason"),
        param(_body(_response_obj("completed")), False, id="non_streaming_completed"),
        param(_body({"id": "resp_1", "object": "response", "output": []}), False, id="non_streaming_no_status"),
        param(_stream(
            {"type": "response.created", "response": _response_obj("in_progress")},
            {"type": "response.output_text.delta", "delta": "a"},
            {"type": "response.incomplete", "response": _response_obj("incomplete", "max_output_tokens")},
        ), True, id="streaming_incomplete_max_output_tokens"),
        param(_stream(
            {"type": "response.created", "response": _response_obj("in_progress")},
            {"type": "response.output_text.delta", "delta": "a"},
            {"type": "response.completed", "response": _response_obj("completed")},
        ), False, id="streaming_completed"),
        param(_stream(
            {"type": "response.output_text.delta", "delta": "a"},
            {"type": "response.output_text.done", "text": "a"},
        ), False, id="streaming_no_lifecycle_event"),
        param(_stream(
            {"type": "response.created", "response": _response_obj("in_progress")},
            {"type": "response.output_text.delta", "delta": "a"},
        ), False, id="streaming_cut_before_terminal_event"),
        param(_stream(
            {"type": "response.incomplete", "response": _response_obj("incomplete", "max_output_tokens")},
            "[DONE]",
        ), True, id="streaming_non_json_tail_skipped"),
    ],
)  # fmt: skip
def test_is_truncated_responses_status_matches_expected(
    record: RequestRecord, expected: bool
) -> None:
    endpoint = _endpoint(ResponsesEndpoint, EndpointType.RESPONSES)
    assert endpoint.is_truncated(record) is expected


@pytest.mark.parametrize(
    "record, expected",
    [
        param(_body({"type": "message", "content": [{"type": "text", "text": "x"}], "stop_reason": "max_tokens"}), True, id="non_streaming_max_tokens"),
        param(_body({"type": "message", "content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"}), False, id="non_streaming_end_turn"),
        param(_body({"type": "message", "content": [{"type": "text", "text": "x"}]}), False, id="non_streaming_no_stop_reason"),
        param(_anthropic_stream("max_tokens"), True, id="streaming_max_tokens"),
        param(_anthropic_stream("end_turn"), False, id="streaming_end_turn"),
        param(_anthropic_stream(None), False, id="streaming_no_stop_reason"),
        param(_stream({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}}, "[DONE]"), True, id="streaming_non_json_tail_skipped"),
    ],
)  # fmt: skip
def test_is_truncated_anthropic_stop_reason_matches_expected(
    record: RequestRecord, expected: bool
) -> None:
    endpoint = _endpoint(MessagesEndpoint, EndpointType.MESSAGES)
    assert endpoint.is_truncated(record) is expected


def test_is_truncated_endpoint_without_override_returns_false() -> None:
    endpoint = _endpoint(EmbeddingsEndpoint, EndpointType.EMBEDDINGS)
    record = _body(
        {"object": "chat.completion", "choices": [{"finish_reason": "length"}]}
    )
    assert endpoint.is_truncated(record) is False
