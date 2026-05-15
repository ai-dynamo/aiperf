# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hostile-input attack tests for ChatEndpoint format_payload / parse_response.

These tests intentionally feed degenerate, oversized, or malformed inputs to
the endpoint and assert specific behaviour. Real bugs found here are marked
``xfail(strict=True)`` so production code is not modified from a test file.
"""

from __future__ import annotations

import sys

import orjson
import pytest

from aiperf.common.models import (
    RequestRecord,
    Text,
    TextResponse,
    Turn,
)
from aiperf.common.models.record_models import (
    TextResponseData,
    ToolCallResponseData,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.payload_extraction import extract_inputs
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture
def endpoint():
    me = create_model_endpoint(EndpointType.CHAT)
    return create_endpoint_with_mock_transport(ChatEndpoint, me)


@pytest.fixture
def streaming_endpoint():
    me = create_model_endpoint(EndpointType.CHAT, streaming=True)
    return create_endpoint_with_mock_transport(ChatEndpoint, me)


# ---------------------------------------------------------------------------
# format_payload hostile inputs
# ---------------------------------------------------------------------------


class TestChatFormatPayloadHostile:
    def test_raw_tools_1000_entries_forwarded_verbatim(self, endpoint):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": f"desc {i}",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for i in range(1000)
        ]
        turn = Turn(texts=[Text(contents=["hi"])], raw_tools=tools)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["tools"] is tools or payload["tools"] == tools
        assert len(payload["tools"]) == 1000
        assert payload["tools"][999]["function"]["name"] == "tool_999"

    def test_raw_tools_tool_with_no_name_forwarded(self, endpoint):
        """Endpoint does not validate tool shape; forwards verbatim."""
        tools = [{"type": "function", "function": {"arguments": "{}"}}]
        turn = Turn(texts=[Text(contents=["x"])], raw_tools=tools)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["tools"] == tools
        # The tool entry has no "name" key — assert that absence is preserved.
        assert "name" not in payload["tools"][0]["function"]

    def test_extra_body_collides_with_model_key_wins(self, endpoint):
        turn = Turn(
            texts=[Text(contents=["x"])],
            model="real-model",
            extra_body={"model": "OVERRIDE-FROM-EXTRA"},
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["model"] == "OVERRIDE-FROM-EXTRA"

    def test_extra_body_collides_with_messages_wins(self, endpoint):
        new_msgs = [{"role": "user", "content": "rewritten"}]
        turn = Turn(
            texts=[Text(contents=["original"])],
            extra_body={"messages": new_msgs},
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["messages"] == new_msgs

    def test_extra_body_collides_with_stream_wins(self, endpoint):
        turn = Turn(texts=[Text(contents=["x"])], extra_body={"stream": True})
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        # endpoint.streaming=False but extra_body forces stream=True (latest-wins).
        assert payload["stream"] is True

    def test_extra_body_collides_with_tools_wins(self, endpoint):
        raw_tools = [{"type": "function", "function": {"name": "from_raw"}}]
        eb_tools = [{"type": "function", "function": {"name": "from_extra_body"}}]
        turn = Turn(
            texts=[Text(contents=["x"])],
            raw_tools=raw_tools,
            extra_body={"tools": eb_tools},
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["tools"] == eb_tools

    def test_extra_body_non_string_int_key_rejected_by_turn_model(self, endpoint):
        """extra_body is typed `dict[str, Any]` — Pydantic Turn rejects int keys
        at construction. Defensive — extra_body can never carry non-str keys."""
        with pytest.raises(Exception, match="string"):
            Turn(texts=[Text(contents=["x"])], extra_body={42: "answer"})

    def test_extra_body_none_key_rejected_by_turn_model(self, endpoint):
        with pytest.raises(Exception, match="string"):
            Turn(texts=[Text(contents=["x"])], extra_body={None: "nope"})

    def test_extra_body_circular_reference_does_not_crash_format(self, endpoint):
        """A self-referential extra_body must not blow up format_payload itself
        (orjson serialisation would, but that's downstream)."""
        a: dict = {}
        b: dict = {"a": a}
        a["b"] = b
        turn = Turn(texts=[Text(contents=["x"])], extra_body=a)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        # The cycle is faithfully reflected — payload["b"] is b, payload["b"]["a"] is a.
        assert payload["b"] is b
        assert payload["b"]["a"] is a

    def test_max_tokens_negative_one_serialised_verbatim(self, endpoint):
        turn = Turn(texts=[Text(contents=["x"])], max_tokens=-1)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["max_completion_tokens"] == -1

    def test_max_tokens_zero_emits_key(self, endpoint):
        """max_tokens=0 is falsy but not None — must still be emitted (truthy check
        would silently drop it; current code uses `is not None`)."""
        turn = Turn(texts=[Text(contents=["x"])], max_tokens=0)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["max_completion_tokens"] == 0

    def test_max_tokens_max_int(self, endpoint):
        turn = Turn(texts=[Text(contents=["x"])], max_tokens=sys.maxsize)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["max_completion_tokens"] == sys.maxsize

    def test_empty_model_falls_back_to_primary(self, endpoint):
        """``model=""`` is falsy so the ``or`` short-circuit picks
        primary_model_name. Document this behaviour."""
        turn = Turn(texts=[Text(contents=["x"])], model="")
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["model"] == endpoint.model_endpoint.primary_model_name

    def test_assistant_turn_with_content_and_tool_calls_raw_messages(self, endpoint):
        """A mixed assistant message in raw_messages goes through build_messages
        verbatim — both content and tool_calls land in the wire body."""
        raw_assistant = [
            {
                "role": "assistant",
                "content": "hi",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            }
        ]
        turn = Turn(role="assistant", raw_messages=raw_assistant)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        msgs = payload["messages"]
        assert msgs[-1]["content"] == "hi"
        assert msgs[-1]["tool_calls"][0]["function"]["name"] == "f"


# ---------------------------------------------------------------------------
# parse_response edges
# ---------------------------------------------------------------------------


class TestChatParseResponseHostile:
    def test_empty_body_dict_returns_none(self, endpoint):
        """`{}` is falsy -> parse_response short-circuits to None without
        calling extract_chat_response_data. Graceful skip."""
        assert endpoint.parse_response(create_mock_response(1, {})) is None

    def test_unknown_object_type_raises(self, endpoint):
        with pytest.raises(ValueError, match="Unsupported OpenAI object type"):
            endpoint.parse_response(
                create_mock_response(1, {"object": "garbage", "choices": []})
            )

    def test_none_json_returns_none(self, endpoint):
        """When the response body is non-JSON, get_json returns None and the
        endpoint silently skips the chunk."""
        assert endpoint.parse_response(create_mock_response(1, None)) is None

    def test_malformed_json_text_skipped_via_get_json_none(self, endpoint):
        """Real-world transport: TextResponse.get_json() returns None on parse error."""
        bad = TextResponse(perf_ns=1, text="data: not-json")
        # TextResponse provides get_json directly; assert it returns None and the
        # endpoint's parse_response then yields None.
        assert bad.get_json() is None
        # Feed it through parse_response via wrapper to confirm None-path.
        result = endpoint.parse_response(bad)
        assert result is None

    def test_streaming_tool_call_name_then_args_reassembled(self, endpoint):
        """Function name in chunk 1, arguments in chunk 5 — must reassemble."""
        chunks = [
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "lookup"},
                                }
                            ]
                        }
                    }
                ],
            },
            {"object": "chat.completion.chunk", "choices": [{"delta": {}}]},
            {"object": "chat.completion.chunk", "choices": [{"delta": {}}]},
            {"object": "chat.completion.chunk", "choices": [{"delta": {}}]},
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"q":"x"}'}}
                            ]
                        }
                    }
                ],
            },
        ]
        responses = [
            TextResponse(perf_ns=i, text=orjson.dumps(c).decode())
            for i, c in enumerate(chunks)
        ]
        record = RequestRecord(responses=responses)

        turn = endpoint.build_assistant_turn(record)

        assert turn is not None and turn.raw_messages is not None
        tc = turn.raw_messages[0]["tool_calls"][0]
        assert tc["function"]["name"] == "lookup"
        assert tc["function"]["arguments"] == '{"q":"x"}'

    def test_legacy_function_call_singular_reassembled(self, endpoint):
        """Legacy LiteLLM-style `function_call` singular — must reassemble."""
        chunks = [
            {
                "object": "chat.completion.chunk",
                "choices": [
                    {"delta": {"function_call": {"name": "look", "arguments": '{"a'}}}
                ],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"function_call": {"arguments": '":1}'}}}],
            },
        ]
        responses = [
            TextResponse(perf_ns=i, text=orjson.dumps(c).decode())
            for i, c in enumerate(chunks)
        ]
        record = RequestRecord(responses=responses)

        turn = endpoint.build_assistant_turn(record)

        assert turn is not None and turn.raw_messages is not None
        tc = turn.raw_messages[0]["tool_calls"][0]
        assert tc["function"]["name"] == "look"
        assert tc["function"]["arguments"] == '{"a":1}'

    def test_reasoning_only_response_build_assistant_turn(self, endpoint):
        """Qwen3-style reasoning-only — build_assistant_turn must accept and
        propagate the reasoning text (PR description)."""
        body = {
            "object": "chat.completion",
            "choices": [{"message": {"content": "", "reasoning": "thinking..."}}],
        }
        responses = [TextResponse(perf_ns=1, text=orjson.dumps(body).decode())]
        record = RequestRecord(responses=responses)

        turn = endpoint.build_assistant_turn(record)

        assert turn is not None
        # No tool_calls -> base behaviour with reasoning fallback into texts.
        assert turn.raw_messages is None
        assert turn.texts and turn.texts[0].contents[0] == "thinking..."

    def test_streaming_mid_done_then_extra_chunks_no_crash(self, endpoint):
        """Mid-stream [DONE] followed by more chunks: parse_response on each
        chunk handles them independently; [DONE] arrives as non-JSON to
        transport (handled elsewhere) — verify a normal chunk after-the-fact
        still parses cleanly."""
        # The [DONE] sentinel is filtered at the SSE transport layer before
        # reaching parse_response. Simulate the endpoint receiving subsequent
        # chunks: it must still parse each correctly.
        chunk = {
            "object": "chat.completion.chunk",
            "choices": [{"delta": {"content": "post-done"}}],
        }
        resp = create_mock_response(1, chunk)
        parsed = endpoint.parse_response(resp)
        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "post-done"

    def test_chunk_with_empty_choices_returns_none(self, endpoint):
        resp = create_mock_response(
            1, {"object": "chat.completion.chunk", "choices": []}
        )
        assert endpoint.parse_response(resp) is None

    def test_chunk_with_both_content_and_tool_call_emits_mixed(self, endpoint):
        """Mixed prose+tool chunk must emit ToolCallResponseData with both."""
        json_obj = {
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": {
                        "content": "hi",
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ],
        }
        resp = create_mock_response(1, json_obj)
        parsed = endpoint.parse_response(resp)
        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.content == "hi"
        assert "f" in parsed.data.tool_call_text and "{}" in parsed.data.tool_call_text


# ---------------------------------------------------------------------------
# Pre-tokenised embedding extraction (payload_extraction)
# ---------------------------------------------------------------------------


class TestPayloadExtractionPretokenisedEdges:
    def test_pretokenised_list_of_ints_10000_entries(self):
        from aiperf.common.enums import MediaType

        payload = {"input": list(range(10000))}
        result = extract_inputs(payload, {MediaType.TEXT: {"text"}})
        assert result.pretokenised_token_count == 10000

    def test_pretokenised_list_of_list_of_ints_mixed_sizes(self):
        from aiperf.common.enums import MediaType

        payload = {"input": [list(range(10)), list(range(20)), list(range(30))]}
        result = extract_inputs(payload, {MediaType.TEXT: {"text"}})
        assert result.pretokenised_token_count == 60

    def test_input_mixed_str_and_int_falls_through(self):
        """Mixed list[str|int] does not match all-str OR all-int — silently
        produces empty pretokenised + empty texts (the all() guards reject it)."""
        from aiperf.common.enums import MediaType

        payload = {"input": ["hello", 1, "world"]}
        result = extract_inputs(payload, {MediaType.TEXT: {"text"}})
        assert result.pretokenised_token_count == 0
        assert result.texts == []
