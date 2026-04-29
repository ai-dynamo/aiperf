# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ResponsesEndpoint.build_assistant_turn — captures the full
``output[]`` array (text, reasoning, function_call, web_search_call, ...)
for replay so FORK-mode DAG children inherit the parent's complete
response, not just its text.
"""

import orjson
import pytest

from aiperf.common.models import RequestRecord, TextResponse, Turn
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)


@pytest.fixture
def endpoint():
    model_endpoint = create_model_endpoint(EndpointType.RESPONSES)
    return create_endpoint_with_mock_transport(ResponsesEndpoint, model_endpoint)


def _resp(json_data: dict, perf_ns: int = 100_000_000) -> TextResponse:
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


class TestResponsesAssistantTurnTextOnly:
    """When the response carries only text/reasoning items, fall through to
    the base class so behaviour for non-tool workloads is unchanged."""

    def test_non_streaming_text_only_falls_back_to_base(self, endpoint):
        # Even when the non-streaming response has output items, all
        # message/text items still need to round-trip — so they ARE
        # captured into raw_messages. This is the documented behaviour for
        # Responses (output items are valid input items).
        record = _record(
            _resp(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "Hello there!"}
                            ],
                        }
                    ],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.role == "assistant"
        # Captured verbatim so the next request's ``input`` array sees the
        # exact assistant message the server emitted.
        assert turn.raw_messages == [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello there!"}],
            }
        ]

    def test_empty_record_returns_none(self, endpoint):
        assert endpoint.build_assistant_turn(_record()) is None

    def test_no_output_items_falls_back_to_base(self, endpoint):
        # No ``object: "response"``, no ``response.completed``, no
        # ``response.output_item.done`` — base class kicks in. With no
        # extractable text either, the base returns None.
        record = _record(_resp({"object": "something_else"}))
        assert endpoint.build_assistant_turn(record) is None


class TestResponsesAssistantTurnNonStreamingFunctionCall:
    """Function calls in non-streaming responses are preserved verbatim."""

    def test_function_call_only(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "fc_abc",
                            "name": "get_weather",
                            "arguments": '{"city":"SF"}',
                        }
                    ],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.raw_messages == [
            {
                "type": "function_call",
                "call_id": "fc_abc",
                "name": "get_weather",
                "arguments": '{"city":"SF"}',
            }
        ]

    def test_message_plus_function_call_preserves_order(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "Looking it up."}
                            ],
                        },
                        {
                            "type": "function_call",
                            "call_id": "fc_1",
                            "name": "search",
                            "arguments": '{"q":"weather"}',
                        },
                    ],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert len(turn.raw_messages) == 2
        assert turn.raw_messages[0]["type"] == "message"
        assert turn.raw_messages[1]["type"] == "function_call"
        assert turn.raw_messages[1]["name"] == "search"

    def test_misc_output_item_types_preserved(self, endpoint):
        # web_search_call, file_search_call, image_generation_call, etc.
        # are all valid output item types that should round-trip through
        # raw_messages so a fork child sees them.
        record = _record(
            _resp(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "web_search_call",
                            "id": "ws_1",
                            "status": "completed",
                        },
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "thinking"}],
                        },
                    ],
                }
            )
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        types = [item["type"] for item in turn.raw_messages]
        assert types == ["web_search_call", "reasoning"]


class TestResponsesAssistantTurnStreaming:
    """Streaming captures via the union of ``response.completed`` and
    ``response.output_item.done`` events, deduplicated by item ``id``."""

    def test_streaming_response_completed_dedupes_against_done(self, endpoint):
        # When response.completed and output_item.done both report the same
        # item (matched by id), the item appears exactly once in the captured
        # raw_messages — completed-ordering wins.
        record = _record(
            _resp(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "partial"}],
                    },
                }
            ),
            _resp(
                {
                    "type": "response.completed",
                    "response": {
                        "object": "response",
                        "output": [
                            {
                                "id": "msg_1",
                                "type": "message",
                                "role": "assistant",
                                "content": [
                                    {"type": "output_text", "text": "final text"}
                                ],
                            },
                            {
                                "type": "function_call",
                                "call_id": "fc_1",
                                "name": "compute",
                                "arguments": "{}",
                            },
                        ],
                    },
                }
            ),
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        # Two unique ids → two items, and completed-ordering wins so the
        # message item carries the final text the API committed to.
        assert len(turn.raw_messages) == 2
        assert turn.raw_messages[0]["id"] == "msg_1"
        assert turn.raw_messages[0]["content"][0]["text"] == "final text"
        assert turn.raw_messages[1]["type"] == "function_call"

    def test_streaming_output_item_done_fallback(self, endpoint):
        # No response.completed event — fall back to collecting each
        # output_item.done event in arrival order.
        record = _record(
            _resp(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi."}],
                    },
                }
            ),
            _resp(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "fc_42",
                        "name": "do_thing",
                        "arguments": '{"x":1}',
                    },
                }
            ),
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert [item["type"] for item in turn.raw_messages] == [
            "message",
            "function_call",
        ]
        assert turn.raw_messages[1]["arguments"] == '{"x":1}'

    def test_streaming_irrelevant_events_skipped(self, endpoint):
        # response.created / response.in_progress / response.output_text.delta
        # carry no replayable item; build_assistant_turn must not invent one.
        record = _record(
            _resp({"type": "response.created", "response": {"object": "response"}}),
            _resp({"type": "response.in_progress"}),
            _resp({"type": "response.output_text.delta", "delta": "Hello"}),
        )
        # No output items were captured AND base text-only fallback also
        # finds nothing parseable in deltas-without-completion (delta events
        # are TextResponseData via parse_response, which the base captures).
        # We expect SOMETHING here — the base falls back to text from the
        # parsed delta event. So this is base-fallback territory.
        turn = endpoint.build_assistant_turn(record)
        # Base path: response.output_text.delta yields TextResponseData
        # ("Hello") → text-only Turn via super().
        assert turn is not None
        assert turn.raw_messages is None
        assert turn.texts and turn.texts[0].contents == ["Hello"]


class TestResponsesAssistantTurnRoundTrip:
    """Captured output items must extend through ``build_messages``
    verbatim onto the next request's ``input`` array."""

    def test_function_call_extends_input_verbatim(self, endpoint):
        record = _record(
            _resp(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "fc_1",
                            "name": "search",
                            "arguments": '{"q":"x"}',
                        }
                    ],
                }
            )
        )
        captured = endpoint.build_assistant_turn(record)
        assert captured is not None

        # Simulate a FORK child whose history is [parent_user, captured,
        # function_call_output, child_user]. build_messages is what populates
        # the next request's ``input`` array via ChatEndpoint-style flatten.
        parent_user = Turn(
            raw_messages=[
                {"role": "user", "content": [{"type": "input_text", "text": "Find X"}]}
            ]
        )
        function_output = Turn(
            raw_messages=[
                {
                    "type": "function_call_output",
                    "call_id": "fc_1",
                    "output": "ok",
                }
            ]
        )
        child_user = Turn(
            raw_messages=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Now what?"}],
                }
            ]
        )
        wire = endpoint.build_messages(
            [parent_user, captured, function_output, child_user]
        )

        assert wire == [
            {"role": "user", "content": [{"type": "input_text", "text": "Find X"}]},
            {
                "type": "function_call",
                "call_id": "fc_1",
                "name": "search",
                "arguments": '{"q":"x"}',
            },
            {"type": "function_call_output", "call_id": "fc_1", "output": "ok"},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Now what?"}],
            },
        ]


class TestResponsesAssistantTurnUnionRegression:
    """Real-world scenarios where ``response.completed.response.output[]``
    drops items that ``response.output_item.done`` events captured.

    Production trace analysis (~/.claude-codex traces, 8,110 streaming
    turns) showed 51 turns where ``response.completed`` arrives with an
    empty or partial ``output[]`` even though ``output_item.done`` fired
    for the items: 12 with messages, 11 with function_calls, 44 with
    encrypted reasoning. AIPerf's previous "prefer completed, ignore done
    when completed is present" rule would have lost all of those.

    These tests pin the union strategy: take ``response.completed`` for
    ordering, then merge in any ``output_item.done`` items not already
    represented (by id).
    """

    def test_completed_empty_but_done_has_message(self, endpoint):
        # Mirrors the smoke-test session 09d2aab8-...: response.completed
        # arrived with an empty output[] yet output_item.done fired with a
        # full message item. Without the union, AIPerf would lose the
        # entire assistant text.
        record = _record(
            _resp(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "msg_smoke",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Status: DONE_WITH_CONCERNS",
                            }
                        ],
                    },
                }
            ),
            _resp(
                {
                    "type": "response.completed",
                    "response": {"object": "response", "output": []},
                }
            ),
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert len(turn.raw_messages) == 1
        assert turn.raw_messages[0]["id"] == "msg_smoke"
        assert (
            turn.raw_messages[0]["content"][0]["text"] == "Status: DONE_WITH_CONCERNS"
        )

    def test_completed_empty_but_done_has_function_call(self, endpoint):
        # 11 production turns lost a function_call dispatch under the old
        # rule. Verify the captured Turn carries the call (with call_id, so
        # follow-up function_call_output can still pair).
        record = _record(
            _resp(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "fc_smoke",
                        "type": "function_call",
                        "call_id": "call_99",
                        "name": "Agent",
                        "arguments": '{"description":"Fix WGM task 4 wiring"}',
                    },
                }
            ),
            _resp(
                {
                    "type": "response.completed",
                    "response": {"object": "response", "output": []},
                }
            ),
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert len(turn.raw_messages) == 1
        item = turn.raw_messages[0]
        assert item["type"] == "function_call"
        assert item["call_id"] == "call_99"
        assert item["name"] == "Agent"

    def test_completed_partial_done_supplies_missing_reasoning(self, endpoint):
        # Common production pattern: completed has the function_call but
        # the reasoning that preceded it only fires via output_item.done.
        # Union must surface BOTH, with completed-ordering preserved.
        record = _record(
            _resp(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "rs_1",
                        "type": "reasoning",
                        "encrypted_content": "ENCRYPTED_REASONING_BLOB",
                    },
                }
            ),
            _resp(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_x",
                        "name": "compute",
                        "arguments": "{}",
                    },
                }
            ),
            _resp(
                {
                    "type": "response.completed",
                    "response": {
                        "object": "response",
                        "output": [
                            # completed only carries the function_call —
                            # reasoning was dropped from the assembled list.
                            {
                                "id": "fc_1",
                                "type": "function_call",
                                "call_id": "call_x",
                                "name": "compute",
                                "arguments": "{}",
                            }
                        ],
                    },
                }
            ),
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        # completed-ordering puts function_call first, then the unique-id
        # reasoning item from done is appended.
        types = [item["type"] for item in turn.raw_messages]
        assert types == ["function_call", "reasoning"]
        assert turn.raw_messages[1]["encrypted_content"] == "ENCRYPTED_REASONING_BLOB"

    def test_dedup_keys_off_call_id_when_id_absent(self, endpoint):
        # Some function_call items in the wild lack a top-level ``id`` and
        # carry only ``call_id``. The dedup key must fall back to
        # ``call_id`` so the same call appearing in both completed and
        # done isn't double-counted.
        record = _record(
            _resp(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call_abc",
                        "name": "compute",
                        "arguments": "{}",
                    },
                }
            ),
            _resp(
                {
                    "type": "response.completed",
                    "response": {
                        "object": "response",
                        "output": [
                            {
                                "type": "function_call",
                                "call_id": "call_abc",
                                "name": "compute",
                                "arguments": "{}",
                            }
                        ],
                    },
                }
            ),
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        # Despite appearing in both sources, the call_id collapses them to
        # one captured item.
        assert len(turn.raw_messages) == 1
        assert turn.raw_messages[0]["call_id"] == "call_abc"
