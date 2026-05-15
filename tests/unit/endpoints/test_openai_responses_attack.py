# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hostile-input attack tests for ResponsesEndpoint format_payload / parse_response."""

from __future__ import annotations

import orjson
import pytest

from aiperf.common.models import (
    Audio,
    Image,
    RequestRecord,
    Text,
    TextResponse,
    Turn,
    Video,
)
from aiperf.common.models.record_models import (
    ReasoningResponseData,
    ToolCallResponseData,
)
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


@pytest.fixture
def endpoint():
    me = create_model_endpoint(EndpointType.RESPONSES)
    return create_endpoint_with_mock_transport(ResponsesEndpoint, me)


@pytest.fixture
def streaming_endpoint():
    me = create_model_endpoint(EndpointType.RESPONSES, streaming=True)
    return create_endpoint_with_mock_transport(ResponsesEndpoint, me)


# ---------------------------------------------------------------------------
# format_payload hostile inputs
# ---------------------------------------------------------------------------


class TestResponsesFormatPayloadHostile:
    def test_raw_tools_1000_entries_forwarded(self, endpoint):
        tools = [
            {
                "type": "function",
                "name": f"tool_{i}",
                "parameters": {"type": "object", "properties": {}},
            }
            for i in range(1000)
        ]
        turn = Turn(texts=[Text(contents=["x"])], raw_tools=tools)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert len(payload["tools"]) == 1000
        assert payload["tools"][500]["name"] == "tool_500"

    def test_raw_tools_no_name_forwarded(self, endpoint):
        tools = [{"type": "function", "parameters": {}}]
        turn = Turn(texts=[Text(contents=["x"])], raw_tools=tools)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["tools"] == tools
        assert "name" not in payload["tools"][0]

    def test_extra_body_collides_with_model_wins(self, endpoint):
        turn = Turn(
            texts=[Text(contents=["x"])],
            model="real",
            extra_body={"model": "OVERRIDDEN"},
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["model"] == "OVERRIDDEN"

    def test_extra_body_collides_with_input_wins(self, endpoint):
        new_input = [{"role": "user", "content": "rewritten"}]
        turn = Turn(
            texts=[Text(contents=["original"])], extra_body={"input": new_input}
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["input"] == new_input

    def test_extra_body_collides_with_stream_wins(self, endpoint):
        turn = Turn(texts=[Text(contents=["x"])], extra_body={"stream": True})
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["stream"] is True

    def test_extra_body_collides_with_tools_wins(self, endpoint):
        raw_tools = [{"type": "function", "name": "from_raw"}]
        eb_tools = [{"type": "function", "name": "from_extra"}]
        turn = Turn(
            texts=[Text(contents=["x"])],
            raw_tools=raw_tools,
            extra_body={"tools": eb_tools},
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["tools"] == eb_tools

    def test_extra_body_int_key_rejected_by_turn_model(self, endpoint):
        """extra_body is typed dict[str, Any] — Pydantic Turn rejects int keys."""
        with pytest.raises(Exception, match="string"):
            Turn(texts=[Text(contents=["x"])], extra_body={7: "lucky"})

    def test_extra_body_circular_no_crash(self, endpoint):
        a: dict = {}
        b: dict = {"a": a}
        a["b"] = b
        turn = Turn(texts=[Text(contents=["x"])], extra_body=a)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["b"] is b

    def test_max_tokens_negative_one_serialised(self, endpoint):
        turn = Turn(texts=[Text(contents=["x"])], max_tokens=-1)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["max_output_tokens"] == -1

    @pytest.mark.xfail(
        reason="bug: Responses endpoint uses truthy `if value is not None` for "
        "max_output_tokens — verify max_tokens=0 still emits. If this passes, "
        "behaviour is correct (None-check, not truthy-check) and xfail flips.",
        strict=True,
    )
    def test_max_tokens_zero_not_emitted(self, endpoint):
        """Sanity: confirm max_tokens=0 IS emitted (None-check semantics).
        Marked xfail-strict to invert: this asserts the BAD outcome; the
        actual code does the right thing (emit zero), so the test fails =>
        xfail-strict passes => no bug. Used to detect a future regression."""
        turn = Turn(texts=[Text(contents=["x"])], max_tokens=0)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        # Bad outcome: zero would silently drop.
        assert "max_output_tokens" not in payload

    def test_empty_model_falls_back_to_primary(self, endpoint):
        turn = Turn(texts=[Text(contents=["x"])], model="")
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        assert payload["model"] == endpoint.model_endpoint.primary_model_name

    def test_video_url_turn_rejected_at_format(self, endpoint):
        """Responses API does not support video — formatter must raise."""
        turn = Turn(
            texts=[Text(contents=["describe"])],
            videos=[Video(contents=["https://example.com/clip.mp4"])],
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        with pytest.raises(NotImplementedError, match="does not support video"):
            endpoint.format_payload(req)

    def test_audio_data_uri_form_accepted(self, endpoint):
        """`data:audio/wav;base64,XYZ` URI form accepted (stripped to wav)."""
        turn = Turn(
            texts=[Text(contents=["transcribe"])],
            audios=[Audio(contents=["data:audio/wav;base64,YWJj"])],
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        parts = [
            p for p in payload["input"][0]["content"] if p["type"] == "input_audio"
        ]
        assert len(parts) == 1
        assert parts[0]["input_audio"]["format"] == "wav"
        assert parts[0]["input_audio"]["data"] == "YWJj"

    def test_audio_internal_form_accepted(self, endpoint):
        """Internal `wav,XYZ` form accepted as-is."""
        turn = Turn(
            texts=[Text(contents=["t"])],
            audios=[Audio(contents=["wav,YWJj"])],
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        parts = [
            p for p in payload["input"][0]["content"] if p["type"] == "input_audio"
        ]
        assert parts[0]["input_audio"] == {"data": "YWJj", "format": "wav"}

    def test_audio_format_content_mismatch_accepted_verbatim(self, endpoint):
        """Header says wav but b64 is technically mp3 bytes — endpoint does NOT
        validate; format and data are emitted verbatim per the parsed prefix."""
        turn = Turn(
            texts=[Text(contents=["t"])],
            audios=[Audio(contents=["wav,SUQzBA=="])],  # ID3 prefix (mp3)
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        parts = [
            p for p in payload["input"][0]["content"] if p["type"] == "input_audio"
        ]
        assert parts[0]["input_audio"]["format"] == "wav"
        assert parts[0]["input_audio"]["data"] == "SUQzBA=="

    def test_empty_text_content_filtered_in_multimodal(self, endpoint):
        turn = Turn(
            texts=[Text(contents=["", "real text", ""])],
            images=[Image(contents=["data:image/png;base64,abc"])],
        )
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        content = payload["input"][0]["content"]
        text_parts = [c for c in content if c["type"] == "input_text"]
        assert len(text_parts) == 1
        assert text_parts[0]["text"] == "real text"

    def test_system_message_list_of_parts_accepted_for_extraction(self, endpoint):
        """Responses API can have instructions as list-of-parts; extract_payload_inputs
        must accept that and yield instruction text in the texts list."""
        payload = {
            "model": "m",
            "input": [{"role": "user", "content": "hello"}],
            "instructions": [{"type": "text", "text": "be brief"}, "and helpful"],
        }
        result = endpoint.extract_payload_inputs(payload)

        # Both list-form instruction strings should be in texts.
        assert "be brief" in result.texts
        assert "and helpful" in result.texts


# ---------------------------------------------------------------------------
# FORK replay filtering of replay-unsafe items
# ---------------------------------------------------------------------------


class TestResponsesReplayFiltering:
    def test_web_search_call_filtered_when_no_tool_config(self, endpoint):
        """A FORK child with raw_messages containing a web_search_call must
        strip that item — splicing it back without ``tools=[{"type":
        "web_search"}]`` would 400."""
        raw = [
            {"type": "web_search_call", "id": "ws_1", "status": "completed"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "result"}],
            },
        ]
        turn = Turn(role="assistant", raw_messages=raw)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        types = [it.get("type") for it in payload["input"]]
        assert "web_search_call" not in types
        assert "message" in types

    def test_function_call_user_defined_tool_not_filtered(self, endpoint):
        """function_call is a user-defined tool — must survive replay."""
        raw = [
            {
                "type": "function_call",
                "name": "lookup",
                "arguments": '{"q":"x"}',
                "call_id": "c1",
            }
        ]
        turn = Turn(role="assistant", raw_messages=raw)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        types = [it.get("type") for it in payload["input"]]
        assert "function_call" in types

    def test_reasoning_item_filtered(self, endpoint):
        """`reasoning` item type requires encrypted_content/store=False; strip it."""
        raw = [
            {"type": "reasoning", "id": "r1", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "out"}],
            },
        ]
        turn = Turn(role="assistant", raw_messages=raw)
        req = create_request_info(model_endpoint=endpoint.model_endpoint, turns=[turn])

        payload = endpoint.format_payload(req)

        types = [it.get("type") for it in payload["input"]]
        assert "reasoning" not in types
        assert "message" in types

    def test_all_replay_unsafe_types_filtered(self, endpoint):
        for unsafe_type in (
            "web_search_call",
            "file_search_call",
            "image_generation_call",
            "code_interpreter_call",
            "computer_call",
            "reasoning",
        ):
            raw = [
                {"type": unsafe_type, "id": "x"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            ]
            turn = Turn(role="assistant", raw_messages=raw)
            req = create_request_info(
                model_endpoint=endpoint.model_endpoint, turns=[turn]
            )
            payload = endpoint.format_payload(req)
            types = [it.get("type") for it in payload["input"]]
            assert unsafe_type not in types, f"{unsafe_type} leaked into replay input"


# ---------------------------------------------------------------------------
# parse_response edges
# ---------------------------------------------------------------------------


class TestResponsesParseResponseHostile:
    def test_empty_body_dict_returns_none(self, endpoint):
        """{} has no `type` and no `object: response` -> None (silent skip)."""
        assert endpoint.parse_response(create_mock_response(1, {})) is None

    def test_unknown_event_type_returns_none(self, endpoint):
        assert (
            endpoint.parse_response(create_mock_response(1, {"type": "bogus.event"}))
            is None
        )

    def test_malformed_json_text_skipped_via_get_json_none(self, endpoint):
        bad = TextResponse(perf_ns=1, text="data: not-json")
        assert bad.get_json() is None
        assert endpoint.parse_response(bad) is None

    def test_streaming_function_call_args_delta_split_across_chunks(
        self, streaming_endpoint
    ):
        """function name typically in output_item.added (no delta data) and
        arguments arrive as repeated `response.function_call_arguments.delta`
        — verify each delta independently parses as tool-call text."""
        delta1 = {
            "type": "response.function_call_arguments.delta",
            "delta": '{"q":',
        }
        delta2 = {
            "type": "response.function_call_arguments.delta",
            "delta": '"x"}',
        }
        r1 = streaming_endpoint.parse_response(create_mock_response(1, delta1))
        r2 = streaming_endpoint.parse_response(create_mock_response(2, delta2))
        assert r1 is not None and isinstance(r1.data, ToolCallResponseData)
        assert r1.data.tool_call_text == '{"q":'
        assert r2 is not None and isinstance(r2.data, ToolCallResponseData)
        assert r2.data.tool_call_text == '"x"}'

    def test_reasoning_only_full_response_extracts_reasoning(self, endpoint):
        """Reasoning item only — must extract as ReasoningResponseData."""
        body = {
            "object": "response",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking..."}],
                }
            ],
        }
        parsed = endpoint.parse_response(create_mock_response(1, body))
        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.reasoning == "thinking..."
        assert parsed.data.content is None

    def test_build_assistant_turn_reasoning_only_falls_back_to_text(self, endpoint):
        """Qwen3-style — reasoning fallback into texts when content empty."""
        body = {
            "object": "response",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thoughts"}],
                }
            ],
        }
        responses = [TextResponse(perf_ns=1, text=orjson.dumps(body).decode())]
        record = RequestRecord(responses=responses)

        turn = endpoint.build_assistant_turn(record)

        # The build_assistant_turn captures the full output[] (reasoning item)
        # into raw_messages.
        assert turn is not None
        assert turn.raw_messages is not None
        assert any(it.get("type") == "reasoning" for it in turn.raw_messages)

    def test_failed_event_in_stream_falls_back_to_base(self, endpoint):
        """A `response.failed` event in the stream -> falls back to base
        text-only build_assistant_turn (no item splicing)."""
        events = [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "partial"}],
                },
            },
            {"type": "response.failed", "response": {}},
        ]
        responses = [
            TextResponse(perf_ns=i, text=orjson.dumps(e).decode())
            for i, e in enumerate(events)
        ]
        record = RequestRecord(responses=responses)

        turn = endpoint.build_assistant_turn(record)

        # With failure event, super().build_assistant_turn is invoked. That
        # walks extract_response_data; output_item.done events return None
        # (no data path), so the record yields no text -> turn is None.
        assert turn is None

    def test_streaming_done_falls_through_no_text_yields_none(self, endpoint):
        """response.output_text.done with empty text returns None."""
        ev = {"type": "response.output_text.done", "text": ""}
        assert endpoint.parse_response(create_mock_response(1, ev)) is None
