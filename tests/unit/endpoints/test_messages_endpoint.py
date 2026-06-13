# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Anthropic Messages API endpoint (``/v1/messages``)."""

import pytest

from aiperf.common.models import (
    Audio,
    Image,
    ReasoningResponseData,
    Text,
    TextResponseData,
    ToolCallResponseData,
    Turn,
    Video,
)
from aiperf.endpoints.anthropic_messages import MessagesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)

_PERF_NS = 123456789


@pytest.fixture
def model_endpoint():
    return create_model_endpoint(EndpointType.MESSAGES)


@pytest.fixture
def streaming_model_endpoint():
    return create_model_endpoint(EndpointType.MESSAGES, streaming=True)


@pytest.fixture
def endpoint(model_endpoint):
    return create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)


class TestMessagesEndpointFormatPayload:
    """Tests for MessagesEndpoint.format_payload."""

    def test_simple_text(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello, world!"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "Hello, world!"}]

    def test_max_tokens_defaulted_when_absent(self, endpoint, model_endpoint):
        """Anthropic requires max_tokens; absence falls back to the default."""
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_tokens"] == MessagesEndpoint.DEFAULT_MAX_TOKENS

    def test_max_tokens_from_turn(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Hello"])], model="test-model", max_tokens=500
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_tokens"] == 500

    def test_system_message_is_top_level_not_a_message(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="You are a helpful assistant.",
        )

        payload = endpoint.format_payload(request_info)

        assert payload["system"] == "You are a helpful assistant."
        # System must NOT leak into the messages array (Anthropic contract).
        assert all(m["role"] != "system" for m in payload["messages"])

    def test_no_system_message_omits_field(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "system" not in payload

    def test_user_context_message_prepended(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Actual prompt"])], model="test-model")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            user_context_message="Context preamble",
        )

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {
            "role": "user",
            "content": "Context preamble",
        }
        assert payload["messages"][1]["content"] == "Actual prompt"

    def test_streaming_enabled(self, streaming_model_endpoint):
        endpoint = create_endpoint_with_mock_transport(
            MessagesEndpoint, streaming_model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(
            model_endpoint=streaming_model_endpoint, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["stream"] is True

    def test_model_fallback_to_primary(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_extra_body_merged(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Hello"])],
            model="test-model",
            extra_body={"temperature": 0.5, "top_p": 0.9},
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9

    def test_endpoint_extra_merged(self):
        model_endpoint = create_model_endpoint(
            EndpointType.MESSAGES, extra=[("anthropic_version", "2023-06-01")]
        )
        endpoint = create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["anthropic_version"] == "2023-06-01"

    def test_raw_tools_forwarded(self, endpoint, model_endpoint):
        tools = [
            {
                "name": "get_weather",
                "description": "Get the weather",
                "input_schema": {"type": "object", "properties": {}},
            }
        ]
        turn = Turn(
            texts=[Text(contents=["Weather?"])], model="test-model", raw_tools=tools
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["tools"] == tools

    def test_no_raw_tools_omits_tools_key(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "tools" not in payload

    def test_image_data_uri_renders_base64_source(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Describe"])],
            images=[Image(contents=["data:image/png;base64,QUJD"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        content = payload["messages"][0]["content"]
        assert isinstance(content, list)
        image_part = next(p for p in content if p["type"] == "image")
        assert image_part["source"] == {
            "type": "base64",
            "media_type": "image/png",
            "data": "QUJD",
        }

    def test_image_url_renders_url_source(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Describe"])],
            images=[Image(contents=["https://example.com/cat.png"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        content = payload["messages"][0]["content"]
        image_part = next(p for p in content if p["type"] == "image")
        assert image_part["source"] == {
            "type": "url",
            "url": "https://example.com/cat.png",
        }

    def test_audio_input_rejected(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Listen"])],
            audios=[Audio(contents=["wav,QUJD"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(NotImplementedError, match="does not support audio"):
            endpoint.format_payload(request_info)

    def test_video_input_rejected(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Watch"])],
            videos=[Video(contents=["https://example.com/clip.mp4"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(NotImplementedError, match="does not support video"):
            endpoint.format_payload(request_info)

    def test_empty_turns_raises(self, endpoint, model_endpoint):
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="at least one turn"):
            endpoint.format_payload(request_info)

    def test_multiple_turns_build_messages(self, endpoint, model_endpoint):
        turns = [
            Turn(texts=[Text(contents=["First"])], role="user", model="test-model"),
            Turn(texts=[Text(contents=["Reply"])], role="assistant"),
            Turn(texts=[Text(contents=["Second"])], role="user"),
        ]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)

        assert [m["role"] for m in payload["messages"]] == [
            "user",
            "assistant",
            "user",
        ]


class TestMessagesEndpointParseResponseNonStreaming:
    """Tests for parsing non-streaming ``type: message`` responses."""

    def test_text_content_block(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there"}],
                "stop_reason": "end_turn",
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.perf_ns == _PERF_NS
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hi there"

    def test_multiple_text_blocks_concatenated(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "message",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
            },
        )

        parsed = endpoint.parse_response(response)

        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello world"

    def test_thinking_block_is_reasoning(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "message",
                "content": [
                    {"type": "thinking", "thinking": "Let me reason"},
                    {"type": "text", "text": "The answer is 42"},
                ],
            },
        )

        parsed = endpoint.parse_response(response)

        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.reasoning == "Let me reason"
        assert parsed.data.content == "The answer is 42"

    def test_tool_use_block(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "message",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                    }
                ],
            },
        )

        parsed = endpoint.parse_response(response)

        assert isinstance(parsed.data, ToolCallResponseData)
        assert "get_weather" in parsed.data.tool_call_text
        assert "SF" in parsed.data.tool_call_text

    def test_usage_extracted(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "message",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 25,
                    "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 6,
                },
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.completion_tokens == 25
        assert parsed.usage.prompt_cache_read_tokens == 4
        assert parsed.usage.prompt_cache_write_tokens == 6

    def test_no_json_returns_none(self, endpoint):
        parsed = endpoint.parse_response(create_mock_response(_PERF_NS, None))
        assert parsed is None

    def test_message_no_content_no_usage_returns_none(self, endpoint):
        response = create_mock_response(_PERF_NS, {"type": "message", "content": []})
        assert endpoint.parse_response(response) is None

    def test_non_list_content_with_usage_returns_usage_only(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {"type": "message", "content": None, "usage": {"input_tokens": 7}},
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage.prompt_tokens == 7


class TestMessagesEndpointParseResponseStreaming:
    """Tests for parsing Anthropic streaming SSE events."""

    def test_message_start_yields_prompt_usage(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 25, "output_tokens": 1},
                },
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage.prompt_tokens == 25

    def test_content_block_delta_text(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
        )

        parsed = endpoint.parse_response(response)

        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello"

    def test_content_block_delta_thinking(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Hmm"},
            },
        )

        parsed = endpoint.parse_response(response)

        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.reasoning == "Hmm"

    def test_content_block_delta_tool_input(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"loc":'},
            },
        )

        parsed = endpoint.parse_response(response)

        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.tool_call_text == '{"loc":'

    def test_message_delta_yields_completion_usage(self, endpoint):
        response = create_mock_response(
            _PERF_NS,
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 15},
            },
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage.completion_tokens == 15

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "content_block_start", "index": 0, "content_block": {}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
            {"type": "ping"},
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": ""},
            },
        ],
    )
    def test_structural_events_return_none(self, endpoint, event):
        assert endpoint.parse_response(create_mock_response(_PERF_NS, event)) is None


class TestMessagesEndpointExtractPayloadInputs:
    """The top-level ``system`` prompt must count toward ISL tokenisation."""

    def test_system_and_message_text_collected(self, endpoint):
        payload = {
            "system": "System preamble",
            "messages": [{"role": "user", "content": "User text"}],
        }

        result = endpoint.extract_payload_inputs(payload)

        assert "System preamble" in result.texts
        assert "User text" in result.texts

    def test_system_as_list_of_blocks_collected(self, endpoint):
        payload = {
            "system": [
                {"type": "text", "text": "Block one"},
                {"type": "text", "text": "Block two"},
            ],
            "messages": [{"role": "user", "content": "User text"}],
        }

        result = endpoint.extract_payload_inputs(payload)

        assert result.texts[:2] == ["Block one", "Block two"]
        assert "User text" in result.texts

    def test_image_counted(self, endpoint):
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "QUJD",
                            },
                        },
                    ],
                }
            ]
        }

        result = endpoint.extract_payload_inputs(payload)

        assert result.image_count == 1
