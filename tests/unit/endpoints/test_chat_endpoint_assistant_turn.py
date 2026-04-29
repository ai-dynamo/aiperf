# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ChatEndpoint.build_assistant_turn — captures the assistant
response (text + tool_calls) into a Turn that round-trips verbatim through
``build_messages`` so FORK-mode DAG children inherit the parent's full
assistant message, not just its text.
"""

import orjson
import pytest

from aiperf.common.models import RequestRecord, TextResponse, Turn
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)


@pytest.fixture
def endpoint():
    model_endpoint = create_model_endpoint(EndpointType.CHAT)
    return create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)


def _resp(json_data: dict, perf_ns: int = 100_000_000) -> TextResponse:
    """Build a TextResponse whose ``get_json`` returns ``json_data`` —
    sufficient for ``build_assistant_turn`` (it only consumes parsed JSON)."""
    return TextResponse(
        perf_ns=perf_ns,
        text=orjson.dumps(json_data).decode(),
        content_type="application/json",
    )


def _record(*responses: TextResponse) -> RequestRecord:
    return RequestRecord(
        responses=list(responses),
        start_perf_ns=100_000_000,
        end_perf_ns=200_000_000,
    )


class TestBuildAssistantTurnTextOnly:
    """Default text-only path falls back to the base implementation."""

    def test_non_streaming_text_only_returns_text_turn(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "chat.completion",
                    "choices": [{"message": {"content": "Hello there!"}}],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        assert isinstance(turn, Turn)
        assert turn.role == "assistant"
        # Text-only path uses ``texts`` (no ``raw_messages``), which is the
        # base-class behaviour preserved unchanged.
        assert turn.raw_messages is None
        assert turn.texts and turn.texts[0].contents == ["Hello there!"]

    def test_streaming_text_only_concatenates_chunks(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": "Hello"}}],
                }
            ),
            _resp(
                {
                    "object": "chat.completion.chunk",
                    "choices": [{"delta": {"content": " world"}}],
                }
            ),
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.raw_messages is None
        assert turn.texts[0].contents == ["Hello world"]

    def test_empty_record_returns_none(self, endpoint):
        assert endpoint.build_assistant_turn(_record()) is None

    def test_reasoning_only_captures_content(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "content": "The answer is 42.",
                                "reasoning_content": "step 1, step 2, ...",
                            }
                        }
                    ],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        # Reasoning ``reasoning`` is dropped by design (most chat templates
        # don't round-trip it); ``content`` survives.
        assert turn is not None
        assert turn.raw_messages is None
        assert turn.texts[0].contents == ["The answer is 42."]


class TestBuildAssistantTurnWithToolCalls:
    """Tool calls are preserved verbatim via raw_messages."""

    def test_non_streaming_tool_calls_only_no_text(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_abc",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"SF"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.role == "assistant"
        # Tool-calls path uses ``raw_messages`` so build_messages extends
        # the assistant message verbatim onto the wire.
        assert turn.raw_messages == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"SF"}',
                        },
                    }
                ],
            }
        ]

    def test_non_streaming_text_plus_tool_calls(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "Looking that up for you.",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":"NYC"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.raw_messages[0]["content"] == "Looking that up for you."
        assert (
            turn.raw_messages[0]["tool_calls"][0]["function"]["name"] == "get_weather"
        )

    def test_streaming_tool_calls_reassembled_across_chunks(self, endpoint):
        # OpenAI streaming sends tool_calls as deltas keyed by ``index``;
        # each delta may carry a partial id, type, function.name, or
        # function.arguments fragment that must be concatenated in order.
        record = _record(
            _resp(
                {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_xyz",
                                        "type": "function",
                                        "function": {"name": "compute"},
                                    }
                                ]
                            }
                        }
                    ],
                }
            ),
            _resp(
                {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '{"a":'},
                                    }
                                ]
                            }
                        }
                    ],
                }
            ),
            _resp(
                {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": "1,"},
                                    }
                                ]
                            }
                        }
                    ],
                }
            ),
            _resp(
                {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"b":2}'},
                                    }
                                ]
                            }
                        }
                    ],
                }
            ),
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.raw_messages == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {"name": "compute", "arguments": '{"a":1,"b":2}'},
                    }
                ],
            }
        ]

    def test_streaming_multiple_parallel_tool_calls(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "chat.completion.chunk",
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_a",
                                        "type": "function",
                                        "function": {
                                            "name": "fn_a",
                                            "arguments": "{}",
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "id": "call_b",
                                        "type": "function",
                                        "function": {
                                            "name": "fn_b",
                                            "arguments": "{}",
                                        },
                                    },
                                ]
                            }
                        }
                    ],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        names = [tc["function"]["name"] for tc in turn.raw_messages[0]["tool_calls"]]
        assert names == ["fn_a", "fn_b"]


class TestToolCallsRoundTripThroughBuildMessages:
    """The captured Turn must re-render as the same assistant message on
    the wire — that's how FORK-mode children see the parent's tool_calls."""

    def test_assistant_turn_extends_messages_verbatim(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
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
                }
            )
        )
        captured = endpoint.build_assistant_turn(record)
        assert captured is not None

        # Simulate a FORK child seeded with [parent_user, captured_assistant]
        # plus its own user turn — like build_messages will see at dispatch.
        parent_user = Turn(raw_messages=[{"role": "user", "content": "Find me X."}])
        child_user = Turn(
            raw_messages=[{"role": "tool", "tool_call_id": "call_1", "content": "ok"}]
        )
        wire = endpoint.build_messages([parent_user, captured, child_user])

        assert wire == [
            {"role": "user", "content": "Find me X."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q":"x"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ]
