# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for tool-call surfacing in parse_response and ISL extraction.

Real-world traces (Codex CLI driving Claude Code's tool surface) showed
~64% of streaming turns are tool-only — they emit ``function_call``
items but no text/reasoning. AIPerf's old behaviour returned ``None``
from ``parse_response`` on every event of those turns, which meant TTFT
never fired client-side and client-counted OSL was zero. These tests
pin the fixed behaviour: parse_response surfaces tool-call name and
arguments as text, the existing tokeniser counts them, and TTFT fires
on the first arguments delta.

Also covers ISL parity: replayed ``tool_calls`` (chat) /
``function_call``+``function_call_output`` (Responses) input items
contribute their generated tokens to ISL, plus tool-definition schemas.
"""

import orjson
import pytest

from aiperf.common.models import RequestRecord, TextResponse
from aiperf.common.models.record_models import ToolCallResponseData
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
)

_PERF_NS = 100_000_000


def _resp(json_data: dict, perf_ns: int = _PERF_NS) -> TextResponse:
    return TextResponse(
        perf_ns=perf_ns,
        text=orjson.dumps(json_data).decode(),
        content_type="application/json",
    )


def _record(*responses: TextResponse) -> RequestRecord:
    return RequestRecord(
        responses=list(responses),
        start_perf_ns=0,
        end_perf_ns=10_000_000_000,
    )


# =============================================================================
# Chat Completions: parse_response surfaces tool_calls
# =============================================================================


@pytest.fixture
def chat_endpoint():
    return create_endpoint_with_mock_transport(
        ChatEndpoint, create_model_endpoint(EndpointType.CHAT)
    )


class TestChatParseResponseToolCalls:
    """Streaming and non-streaming tool-call chunks must register as data."""

    def test_streaming_tool_call_first_delta_is_data_bearing(self, chat_endpoint):
        # Codex-shape first tool-call chunk: index, id, type, function.name.
        # Without the fix this returns None → first_token_callback never
        # fires → TTFT lost for every tool-only turn.
        mock = create_mock_response(
            _PERF_NS,
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_x",
                                    "type": "function",
                                    "function": {"name": "get_weather"},
                                }
                            ]
                        }
                    }
                ],
            },
        )
        parsed = chat_endpoint.parse_response(mock)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        # Function name registers as the tool-call text — the tokeniser
        # will count it toward client-side OSL.
        assert parsed.data.tool_call_text == "get_weather"
        assert parsed.data.content is None

    def test_streaming_tool_call_arguments_delta_concatenates(self, chat_endpoint):
        mock = create_mock_response(
            _PERF_NS,
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"city":'},
                                }
                            ]
                        }
                    }
                ],
            },
        )
        parsed = chat_endpoint.parse_response(mock)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.tool_call_text == '{"city":'

    def test_streaming_text_plus_tool_call_both_fields_populated(self, chat_endpoint):
        # When a chunk carries BOTH prose ``content`` AND a tool-call
        # delta, the result is ToolCallResponseData with both ``content``
        # and ``tool_call_text`` populated. ``get_text()`` returns
        # content + tool_call_text so the tokeniser sees what the model
        # actually generated and matches the server's
        # ``usage.completion_tokens`` (which counts both portions).
        mock = create_mock_response(
            _PERF_NS,
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "content": "Looking up. ",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"q":"x"}'},
                                }
                            ],
                        }
                    }
                ],
            },
        )
        parsed = chat_endpoint.parse_response(mock)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.content == "Looking up. "
        assert parsed.data.tool_call_text == '{"q":"x"}'
        assert parsed.data.get_text() == 'Looking up. {"q":"x"}'

    def test_non_streaming_tool_call_only(self, chat_endpoint):
        mock = create_mock_response(
            _PERF_NS,
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_a",
                                    "type": "function",
                                    "function": {
                                        "name": "search",
                                        "arguments": '{"q":"x"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )
        parsed = chat_endpoint.parse_response(mock)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.tool_call_text == 'search{"q":"x"}'
        assert parsed.data.content is None

    def test_non_streaming_text_plus_tool_call_both_fields_populated(
        self, chat_endpoint
    ):
        # Same precedence rule as the streaming test above: both
        # populated, get_text() returns the combined string.
        mock = create_mock_response(
            _PERF_NS,
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Looking it up.",
                            "tool_calls": [
                                {
                                    "id": "call_b",
                                    "type": "function",
                                    "function": {
                                        "name": "compute",
                                        "arguments": '{"a":1}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )
        parsed = chat_endpoint.parse_response(mock)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.content == "Looking it up."
        assert parsed.data.tool_call_text == 'compute{"a":1}'
        assert parsed.data.get_text() == 'Looking it up.compute{"a":1}'


# =============================================================================
# Responses API: parse_response surfaces function_call_arguments deltas
# =============================================================================


@pytest.fixture
def responses_endpoint():
    return create_endpoint_with_mock_transport(
        ResponsesEndpoint, create_model_endpoint(EndpointType.RESPONSES)
    )


class TestResponsesParseResponseFunctionCalls:
    """The Responses API streams ``response.function_call_arguments.delta``
    once per arguments fragment. Without the fix these returned None and
    TTFT/OSL went to zero for tool-only turns (~64% of agentic traffic)."""

    def test_function_call_arguments_delta_is_data_bearing(self, responses_endpoint):
        mock = create_mock_response(
            _PERF_NS,
            {
                "type": "response.function_call_arguments.delta",
                "delta": '{"q":',
            },
        )
        parsed = responses_endpoint.parse_response(mock)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.tool_call_text == '{"q":'
        assert parsed.data.content is None

    def test_function_call_arguments_delta_empty_returns_none(self, responses_endpoint):
        # Empty/missing delta string still returns None — no spurious
        # zero-token first_token events.
        mock = create_mock_response(
            _PERF_NS,
            {"type": "response.function_call_arguments.delta", "delta": ""},
        )
        assert responses_endpoint.parse_response(mock) is None

    def test_envelope_events_still_skipped(self, responses_endpoint):
        # Structural events like response.output_item.added carry no
        # generated tokens — must continue returning None so they don't
        # get counted toward OSL.
        for event in (
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "c_1",
                    "name": "get_weather",
                    "arguments": "",
                },
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "c_1",
                    "name": "get_weather",
                    "arguments": '{"city":"SF"}',
                },
            },
            {"type": "response.created"},
            {"type": "response.in_progress"},
        ):
            assert (
                responses_endpoint.parse_response(create_mock_response(_PERF_NS, event))
                is None
            ), f"unexpected non-None for {event['type']!r}"

    def test_codex_shape_streaming_tool_only_turn(self, responses_endpoint):
        # End-to-end fixture mirroring what Codex emits on a tool-only
        # turn: created → in_progress → output_item.added → arg deltas →
        # arg done → output_item.done → completed (with usage). Verify
        # the FIRST data-bearing event is a function_call_arguments
        # delta, and that summing the deltas gives the full arguments
        # string.
        events = [
            {"type": "response.created"},
            {"type": "response.in_progress"},
            {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "c_1",
                    "name": "Read",
                    "arguments": "",
                },
            },
            {"type": "response.function_call_arguments.delta", "delta": '{"file_'},
            {"type": "response.function_call_arguments.delta", "delta": 'path":"/'},
            {"type": "response.function_call_arguments.delta", "delta": "tmp/x.py"},
            {"type": "response.function_call_arguments.delta", "delta": '"}'},
            {
                "type": "response.function_call_arguments.done",
                "arguments": '{"file_path":"/tmp/x.py"}',
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "c_1",
                    "name": "Read",
                    "arguments": '{"file_path":"/tmp/x.py"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "object": "response",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "fc_1",
                            "call_id": "c_1",
                            "name": "Read",
                            "arguments": '{"file_path":"/tmp/x.py"}',
                        }
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 12},
                },
            },
        ]
        first_data_idx = None
        accumulated = ""
        for i, event in enumerate(events):
            parsed = responses_endpoint.parse_response(
                create_mock_response(_PERF_NS + i, event)
            )
            if parsed is not None and parsed.data is not None:
                if first_data_idx is None:
                    first_data_idx = i
                # Combined-text accessor — works for TextResponseData
                # AND ToolCallResponseData regardless of which fields
                # are populated.
                accumulated += parsed.data.get_text()
        # First data-bearing event is the FIRST arguments delta — that's
        # the moment client-side TTFT should fire.
        assert first_data_idx is not None
        assert events[first_data_idx]["type"] == (
            "response.function_call_arguments.delta"
        )
        # Concatenating the deltas reproduces the full arguments string.
        assert accumulated == '{"file_path":"/tmp/x.py"}'


# =============================================================================
# Non-streaming Responses: function_call items contribute to text
# =============================================================================


class TestResponsesNonStreamingFunctionCallText:
    def test_function_call_only_output(self, responses_endpoint):
        mock = create_mock_response(
            _PERF_NS,
            {
                "object": "response",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": "c_1",
                        "name": "search",
                        "arguments": '{"q":"x"}',
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
        parsed = responses_endpoint.parse_response(mock)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        # Name + arguments concatenated into client-OSL text.
        assert parsed.data.tool_call_text == 'search{"q":"x"}'
        assert parsed.data.content is None

    def test_message_plus_function_call(self, responses_endpoint):
        mock = create_mock_response(
            _PERF_NS,
            {
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Looking up. "}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "c_2",
                        "name": "get",
                        "arguments": '{"a":1}',
                    },
                ],
            },
        )
        parsed = responses_endpoint.parse_response(mock)
        assert parsed is not None
        # Both items present → ToolCallResponseData with both fields
        # populated; get_text() returns content + tool_call_text so
        # client-OSL counts everything the model emitted.
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.content == "Looking up. "
        assert parsed.data.tool_call_text == 'get{"a":1}'
        assert parsed.data.get_text() == 'Looking up. get{"a":1}'


# =============================================================================
# ISL parity: extract_payload_inputs counts replayed tool-call inputs
# =============================================================================


class TestExtractPayloadInputsToolParity:
    def test_chat_replayed_tool_calls_count_toward_isl(self, chat_endpoint):
        payload = {
            "messages": [
                {"role": "user", "content": "Find me a flight."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_flights",
                                "arguments": '{"from":"SFO","to":"JFK"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "flight ABC at $300",
                },
                {"role": "user", "content": "Book it."},
            ],
            "model": "gpt-4o",
        }
        extracted = chat_endpoint.extract_payload_inputs(payload)
        # User text + assistant tool_call name + tool_call arguments +
        # tool-result content + user text all contribute to ISL.
        joined = "".join(extracted.texts)
        assert "Find me a flight." in joined
        assert "search_flights" in joined
        assert '{"from":"SFO","to":"JFK"}' in joined
        assert "flight ABC at $300" in joined
        assert "Book it." in joined

    def test_chat_tool_definitions_count_toward_isl(self, chat_endpoint):
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gpt-4o",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "search_flights",
                        "description": "Search for flights between two cities.",
                        "parameters": {
                            "type": "object",
                            "properties": {"from": {"type": "string"}},
                        },
                    },
                }
            ],
        }
        extracted = chat_endpoint.extract_payload_inputs(payload)
        joined = "".join(extracted.texts)
        assert "search_flights" in joined
        assert "Search for flights between two cities." in joined
        # Schema serialised — the tokeniser sees the same JSON the server
        # prepends to the prompt.
        assert '"properties"' in joined
        assert '"from"' in joined

    def test_responses_input_function_call_replay_counts(self, responses_endpoint):
        payload = {
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {
                    "type": "function_call",
                    "call_id": "c_1",
                    "name": "search_flights",
                    "arguments": '{"from":"SFO"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "c_1",
                    "output": "flight ABC",
                },
            ],
            "model": "gpt-4o",
        }
        extracted = responses_endpoint.extract_payload_inputs(payload)
        joined = "".join(extracted.texts)
        assert "hi" in joined
        assert "search_flights" in joined
        assert '{"from":"SFO"}' in joined
        assert "flight ABC" in joined

    def test_chat_payload_with_no_tools_unchanged(self, chat_endpoint):
        # Pure-text chat payloads without tools should not regress.
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o",
        }
        extracted = chat_endpoint.extract_payload_inputs(payload)
        assert extracted.texts == ["Hello"]
        assert extracted.image_count == 0
