# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for FORK-child message-merging gap fixes.

Each class targets one finding from the audit so a regression triggers
a single, named test failure rather than a generic e2e drift.
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.common.models import (
    RequestRecord,
    Text,
    TextResponse,
    Turn,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture
def chat_endpoint():
    ep_info = create_model_endpoint(EndpointType.CHAT, streaming=True)
    return create_endpoint_with_mock_transport(ChatEndpoint, ep_info)


@pytest.fixture
def responses_endpoint():
    ep_info = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    return create_endpoint_with_mock_transport(ResponsesEndpoint, ep_info)


# ---------------------------------------------------------------------------
# P1 #2: ResponsesEndpoint filters output-only items out of input on replay
# ---------------------------------------------------------------------------


class TestResponsesFiltersUnsafeOutputItems:
    def test_drops_web_search_call(self, responses_endpoint):
        parent_turn = Turn(
            role="assistant",
            raw_messages=[
                {"id": "ws_1", "type": "web_search_call", "status": "completed"},
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            ],
        )
        out = responses_endpoint.build_messages([parent_turn])
        types = [item.get("type") for item in out]
        assert "web_search_call" not in types
        assert "message" in types

    def test_drops_reasoning(self, responses_endpoint):
        parent_turn = Turn(
            role="assistant",
            raw_messages=[
                {"id": "rs_1", "type": "reasoning", "summary": []},
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ],
        )
        out = responses_endpoint.build_messages([parent_turn])
        types = [item.get("type") for item in out]
        assert "reasoning" not in types
        assert "message" in types

    def test_keeps_function_call(self, responses_endpoint):
        parent_turn = Turn(
            role="assistant",
            raw_messages=[
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "name": "lookup",
                    "arguments": "{}",
                    "call_id": "call_a",
                },
            ],
        )
        out = responses_endpoint.build_messages([parent_turn])
        assert any(item.get("type") == "function_call" for item in out)


# ---------------------------------------------------------------------------
# P1 #3: conversation-level fields walk turns from the end
# ---------------------------------------------------------------------------


class TestConversationLevelFieldsInheritThroughFork:
    def test_chat_inherits_raw_tools_from_parent_turn(self, chat_endpoint):
        tools = [{"type": "function", "function": {"name": "lookup"}}]
        parent_turn = Turn(
            role="user",
            texts=[Text(contents=["help"])],
            raw_tools=tools,
        )
        child_turn = Turn(role="user", texts=[Text(contents=["follow up"])])
        request_info = create_request_info(
            model_endpoint=chat_endpoint.model_endpoint,
            turns=[parent_turn, child_turn],
        )
        payload = chat_endpoint.format_payload(request_info)
        assert payload.get("tools") == tools

    def test_chat_max_tokens_does_not_inherit_from_parent(self, chat_endpoint):
        parent_turn = Turn(
            role="user",
            texts=[Text(contents=["help"])],
            max_tokens=128,
        )
        child_turn = Turn(role="user", texts=[Text(contents=["follow up"])])
        request_info = create_request_info(
            model_endpoint=chat_endpoint.model_endpoint,
            turns=[parent_turn, child_turn],
        )
        payload = chat_endpoint.format_payload(request_info)
        assert "max_completion_tokens" not in payload

    def test_chat_model_does_not_inherit_from_parent(self, chat_endpoint):
        parent_turn = Turn(
            role="user",
            texts=[Text(contents=["help"])],
            model="parent-model",
        )
        child_turn = Turn(role="user", texts=[Text(contents=["follow up"])])
        request_info = create_request_info(
            model_endpoint=chat_endpoint.model_endpoint,
            turns=[parent_turn, child_turn],
        )
        payload = chat_endpoint.format_payload(request_info)
        assert payload["model"] == chat_endpoint.model_endpoint.primary_model_name

    def test_chat_extra_body_does_not_inherit_from_parent(self, chat_endpoint):
        parent_turn = Turn(
            role="user",
            texts=[Text(contents=["help"])],
            extra_body={"custom_flag": True},
        )
        child_turn = Turn(role="user", texts=[Text(contents=["follow up"])])
        request_info = create_request_info(
            model_endpoint=chat_endpoint.model_endpoint,
            turns=[parent_turn, child_turn],
        )
        payload = chat_endpoint.format_payload(request_info)
        assert "custom_flag" not in payload

    def test_responses_max_tokens_does_not_inherit_from_parent(
        self, responses_endpoint
    ):
        parent_turn = Turn(
            role="user",
            texts=[Text(contents=["help"])],
            max_tokens=256,
        )
        child_turn = Turn(role="user", texts=[Text(contents=["follow up"])])
        request_info = create_request_info(
            model_endpoint=responses_endpoint.model_endpoint,
            turns=[parent_turn, child_turn],
        )
        payload = responses_endpoint.format_payload(request_info)
        assert "max_output_tokens" not in payload

    def test_responses_model_does_not_inherit_from_parent(self, responses_endpoint):
        parent_turn = Turn(
            role="user",
            texts=[Text(contents=["help"])],
            model="parent-model",
        )
        child_turn = Turn(role="user", texts=[Text(contents=["follow up"])])
        request_info = create_request_info(
            model_endpoint=responses_endpoint.model_endpoint,
            turns=[parent_turn, child_turn],
        )
        payload = responses_endpoint.format_payload(request_info)
        assert payload["model"] == responses_endpoint.model_endpoint.primary_model_name


# ---------------------------------------------------------------------------
# P2 #6: missing-index fallback aligns with non-streaming path
# ---------------------------------------------------------------------------


class TestMissingIndexParallelToolCallsDoNotCollapse:
    def test_two_parallel_tool_calls_without_index_get_distinct_slots(
        self, chat_endpoint
    ):
        # One chunk, two tool_calls, neither carries ``index``. Defaulting
        # to 0 would collapse both into the same slot. Defaulting to
        # ``len(...)`` (matching non-streaming) preserves both.
        chunk = {
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {"function": {"name": "alpha", "arguments": "{}"}},
                            {"function": {"name": "beta", "arguments": "{}"}},
                        ]
                    }
                }
            ],
        }
        record = RequestRecord(
            responses=[TextResponse(perf_ns=1, text=orjson.dumps(chunk).decode())]
        )
        turn = chat_endpoint.build_assistant_turn(record)
        assert turn is not None
        msg = turn.raw_messages[0]
        names = sorted(tc["function"]["name"] for tc in msg["tool_calls"])
        assert names == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# P2 #10: legacy function_call (singular) is captured in chat
# ---------------------------------------------------------------------------


class TestLegacyFunctionCallSupport:
    def test_non_streaming_legacy_function_call_normalised(self, chat_endpoint):
        full = {
            "object": "chat.completion",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "get_weather",
                            "arguments": '{"city":"SF"}',
                        },
                    }
                }
            ],
        }
        record = RequestRecord(
            responses=[TextResponse(perf_ns=1, text=orjson.dumps(full).decode())]
        )
        turn = chat_endpoint.build_assistant_turn(record)
        assert turn is not None
        msg = turn.raw_messages[0]
        tool_calls = msg.get("tool_calls")
        assert tool_calls and len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"city":"SF"}'

    def test_streaming_legacy_function_call_concat(self, chat_endpoint):
        # name + two argument fragments split across chunks.
        chunks = [
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"function_call": {"name": "get_weather"}}}],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"function_call": {"arguments": '{"city":'}}}],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"function_call": {"arguments": '"SF"}'}}}],
            },
        ]
        record = RequestRecord(
            responses=[
                TextResponse(perf_ns=i, text=orjson.dumps(c).decode())
                for i, c in enumerate(chunks)
            ]
        )
        turn = chat_endpoint.build_assistant_turn(record)
        assert turn is not None
        msg = turn.raw_messages[0]
        tool_calls = msg.get("tool_calls")
        assert tool_calls and len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"
        assert tool_calls[0]["function"]["arguments"] == '{"city":"SF"}'


# ---------------------------------------------------------------------------
# P2 #8 (audit-numbering): failed-stream short-circuits in Responses
# ---------------------------------------------------------------------------


class TestResponsesFailedStreamShortCircuits:
    @pytest.mark.parametrize(
        "fail_event",
        [
            {"type": "response.failed"},
            {"type": "response.incomplete"},
            {"type": "response.error"},
            {"type": "error"},
        ],
    )
    def test_failed_stream_returns_none_even_with_partial_items(
        self, responses_endpoint, fail_event
    ):
        partial_done = {
            "type": "response.output_item.done",
            "item": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "partial"}],
            },
        }
        record = RequestRecord(
            responses=[
                TextResponse(perf_ns=1, text=orjson.dumps(partial_done).decode()),
                TextResponse(perf_ns=2, text=orjson.dumps(fail_event).decode()),
            ]
        )
        # On failure we fall through to base (text-only). Since
        # parse_response of these events yields no data, the base
        # implementation returns None.
        assert responses_endpoint.build_assistant_turn(record) is None


# ---------------------------------------------------------------------------
# P3 #11: chat de-dups leading system when raw_messages already starts with system
# ---------------------------------------------------------------------------


class TestChatDeDupsLeadingSystem:
    def test_authored_leading_system_wins(self, chat_endpoint):
        authored = [
            {"role": "system", "content": "authored system"},
            {"role": "user", "content": "hi"},
        ]
        turn = Turn(role="user", raw_messages=authored)
        request_info = create_request_info(
            model_endpoint=chat_endpoint.model_endpoint,
            turns=[turn],
            system_message="request_info system",
        )
        payload = chat_endpoint.format_payload(request_info)
        # Exactly one system message, and it's the authored one.
        systems = [m for m in payload["messages"] if m["role"] == "system"]
        assert len(systems) == 1
        assert systems[0]["content"] == "authored system"


# ---------------------------------------------------------------------------
# P3 #12: raw_messages truthiness (empty list falls back to synthesis)
# ---------------------------------------------------------------------------


class TestRawMessagesEmptyListFallsBack:
    def test_empty_raw_messages_renders_synthetic_turn(self, chat_endpoint):
        turn = Turn(role="user", texts=[Text(contents=["hello"])], raw_messages=[])
        out = chat_endpoint.build_messages([turn])
        assert out == [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# P3 #14: audio data-URI prefix is stripped before format extraction
# ---------------------------------------------------------------------------


class TestAudioDataUriHandling:
    def test_data_uri_prefix_stripped(self, chat_endpoint):
        part = chat_endpoint._render_audio_part("data:audio/wav;base64,QUJD")
        assert part["input_audio"]["format"] == "wav"
        assert part["input_audio"]["data"] == "QUJD"

    def test_plain_format_data_unchanged(self, chat_endpoint):
        part = chat_endpoint._render_audio_part("wav,QUJD")
        assert part["input_audio"]["format"] == "wav"
        assert part["input_audio"]["data"] == "QUJD"

    def test_responses_endpoint_uses_same_helper(self, responses_endpoint):
        part = responses_endpoint._render_audio_part("data:audio/mp3;base64,XYZ")
        assert part["input_audio"]["format"] == "mp3"
        assert part["input_audio"]["data"] == "XYZ"


# ---------------------------------------------------------------------------
# P3 #17: Responses instructions list-form contributes to extracted texts
# ---------------------------------------------------------------------------


class TestResponsesInstructionsListForm:
    def test_string_instructions_prepended(self, responses_endpoint):
        out = responses_endpoint.extract_payload_inputs(
            {"input": [], "instructions": "be helpful"}
        )
        assert out.texts == ["be helpful"]

    def test_list_of_parts_instructions(self, responses_endpoint):
        out = responses_endpoint.extract_payload_inputs(
            {
                "input": [],
                "instructions": [
                    {"type": "input_text", "text": "first"},
                    {"type": "input_text", "text": "second"},
                ],
            }
        )
        # Order preserved: first, then second, then any input texts.
        assert out.texts[:2] == ["first", "second"]


# ---------------------------------------------------------------------------
# P3 #18: pre-tokenised embeddings input contributes a token count
# ---------------------------------------------------------------------------


class TestEmbeddingsPretokenisedInput:
    def test_list_of_int_pretokenised(self, chat_endpoint):
        out = chat_endpoint.extract_payload_inputs({"input": [1, 2, 3, 4, 5]})
        assert out.pretokenised_token_count == 5
        assert out.texts == []

    def test_list_of_list_int_pretokenised(self, chat_endpoint):
        out = chat_endpoint.extract_payload_inputs({"input": [[1, 2, 3], [4, 5], [6]]})
        assert out.pretokenised_token_count == 6
        assert out.texts == []

    def test_list_of_str_still_works(self, chat_endpoint):
        out = chat_endpoint.extract_payload_inputs({"input": ["a", "b"]})
        assert out.pretokenised_token_count == 0
        assert out.texts == ["a", "b"]


# ---------------------------------------------------------------------------
# P3 #16: Responses video raises immediately rather than emitting a chat-shape
# part the server rejects
# ---------------------------------------------------------------------------


class TestResponsesVideoRejectedAtFormatTime:
    def test_render_video_part_raises_not_implemented(self, responses_endpoint):
        with pytest.raises(NotImplementedError, match="does not support video"):
            responses_endpoint._render_video_part("https://example.com/v.mp4")
