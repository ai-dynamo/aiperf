# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MessagesEndpoint (Anthropic Messages API)."""

import orjson
import pytest
from pytest import param

from aiperf.common.models import ExtractedPayload, Text, Turn
from aiperf.common.models.record_models import (
    ReasoningResponseData,
    SSEField,
    SSEMessage,
    TextResponseData,
    ToolCallResponseData,
)
from aiperf.endpoints.anthropic_messages import (
    MessagesEndpoint,
    _walk_system,
    _walk_tool_blocks,
    _walk_tool_schemas,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestAnthropicMessagesFormatPayload:
    """Tests for MessagesEndpoint format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.MESSAGES)

    @pytest.fixture
    def streaming_model_endpoint(self):
        return create_model_endpoint(EndpointType.MESSAGES, streaming=True)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)

    def test_simple_text(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello!"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "claude-sonnet-4-20250514"
        assert "stream" not in payload
        assert payload["max_tokens"] == 1024
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["content"] == "Hello!"
        assert "system" not in payload

    def test_max_tokens_from_turn(self, endpoint, model_endpoint):
        turn = Turn(
            texts=[Text(contents=["Test"])],
            model="claude-sonnet-4-20250514",
            max_tokens=500,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_tokens"] == 500

    def test_max_tokens_default(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Test"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_tokens"] == 1024

    def test_system_message_as_top_level(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Hello"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="You are a helpful assistant.",
        )

        payload = endpoint.format_payload(request_info)

        assert payload["system"] == "You are a helpful assistant."
        for msg in payload["messages"]:
            assert msg["role"] != "system"

    def test_multi_turn(self, endpoint, model_endpoint):
        turns = [
            Turn(
                texts=[Text(contents=["Hello"])],
                role="user",
                model="claude-sonnet-4-20250514",
            ),
            Turn(
                texts=[Text(contents=["Hi there!"])],
                role="assistant",
                model="claude-sonnet-4-20250514",
            ),
            Turn(
                texts=[Text(contents=["How are you?"])],
                role="user",
                model="claude-sonnet-4-20250514",
            ),
        ]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)

        payload = endpoint.format_payload(request_info)

        assert len(payload["messages"]) == 3
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][1]["role"] == "assistant"
        assert payload["messages"][2]["role"] == "user"

    def test_streaming_enabled(self, streaming_model_endpoint):
        endpoint = create_endpoint_with_mock_transport(
            MessagesEndpoint, streaming_model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Test"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(
            model_endpoint=streaming_model_endpoint, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["stream"] is True

    def test_non_streaming_omits_stream_key(self, endpoint, model_endpoint):
        # Real Claude Code clients omit ``stream`` from non-streaming requests
        # rather than sending ``stream: false``; matching that wire shape
        # avoids a minor proxy-log diff.
        turn = Turn(texts=[Text(contents=["Test"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "stream" not in payload

    def test_extra_params(self):
        extra_params = [("temperature", 0.7), ("top_p", 0.9)]
        model_endpoint = create_model_endpoint(
            EndpointType.MESSAGES, extra=extra_params
        )
        endpoint = create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)
        turn = Turn(texts=[Text(contents=["Test"])], model="claude-sonnet-4-20250514")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9

    def test_model_fallback(self, endpoint, model_endpoint):
        turn = Turn(texts=[Text(contents=["Test"])], model=None)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_empty_turns_raises(self, endpoint, model_endpoint):
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_cache_bust_system_prefix_is_top_level_system(
        self, endpoint, model_endpoint
    ):
        turn = Turn(texts=[Text(contents=["Hello"])])
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="[rid:abc123]\n\nYou are helpful.",
        )

        payload = endpoint.format_payload(request_info)

        assert payload["system"] == "[rid:abc123]\n\nYou are helpful."
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]

    def test_cache_bust_first_user_prefix_renders_anthropic_text_block(
        self, endpoint, model_endpoint
    ):
        turn = Turn(
            texts=[
                Text(contents=["[rid:abc123]", "\n\nHello"]),
            ]
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["messages"][0]["content"] == [
            {"type": "text", "text": "[rid:abc123]"},
            {"type": "text", "text": "\n\nHello"},
        ]

    def test_cache_bust_first_user_suffix_preserves_text(
        self, endpoint, model_endpoint
    ):
        turn = Turn(texts=[Text(contents=["Hello\n\n[rid:abc123]"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["messages"][0]["content"] == "Hello\n\n[rid:abc123]"


class TestAnthropicMessagesHeaders:
    """Tests for MessagesEndpoint get_endpoint_headers."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.MESSAGES)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)

    def test_default_headers(self, endpoint, model_endpoint):
        request_info = create_request_info(model_endpoint=model_endpoint)

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["content-type"] == "application/json"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in headers

    def test_api_key_as_x_api_key(self):
        from aiperf.common.enums import ModelSelectionStrategy
        from aiperf.common.models.model_endpoint_info import (
            EndpointInfo,
            ModelEndpointInfo,
            ModelInfo,
            ModelListInfo,
        )

        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.MESSAGES,
                base_url="http://localhost:8000",
                streaming=False,
                extra=[],
                api_key="sk-ant-test-key",
            ),
        )
        endpoint = create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)
        request_info = create_request_info(model_endpoint=model_endpoint)

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["x-api-key"] == "sk-ant-test-key"
        assert "Authorization" not in headers

    def test_custom_headers_merged(self):
        from aiperf.common.enums import ModelSelectionStrategy
        from aiperf.common.models.model_endpoint_info import (
            EndpointInfo,
            ModelEndpointInfo,
            ModelInfo,
            ModelListInfo,
        )

        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.MESSAGES,
                base_url="http://localhost:8000",
                streaming=False,
                extra=[],
                headers=[("anthropic-beta", "extended-thinking-2025-04-11")],
            ),
        )
        endpoint = create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)
        request_info = create_request_info(model_endpoint=model_endpoint)

        headers = endpoint.get_endpoint_headers(request_info)

        assert headers["anthropic-beta"] == "extended-thinking-2025-04-11"
        assert headers["anthropic-version"] == "2023-06-01"


class TestAnthropicMessagesParseResponseNonStreaming:
    """Tests for MessagesEndpoint parse_response (non-streaming)."""

    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.MESSAGES)
        return create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)

    def test_text_block(self, endpoint):
        mock_response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [{"type": "text", "text": "Hello, how can I help?"}],
                "usage": {"input_tokens": 10, "output_tokens": 8},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello, how can I help?"
        assert parsed.usage is not None
        assert parsed.usage.get("input_tokens") == 10
        assert parsed.usage.get("output_tokens") == 8

    def test_thinking_and_text_blocks(self, endpoint):
        mock_response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this..."},
                    {"type": "text", "text": "The answer is 42"},
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.content == "The answer is 42"
        assert parsed.data.reasoning == "Let me analyze this..."

    def test_usage_mapping(self, endpoint):
        mock_response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [{"type": "text", "text": "Hi"}],
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 10,
                    "cache_creation_input_tokens": 5,
                    "cache_read_input_tokens": 3,
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed.usage.get("input_tokens") == 25
        assert parsed.usage.get("output_tokens") == 10

    def test_empty_content(self, endpoint):
        mock_response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [],
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is not None

    def test_null_json_returns_none(self, endpoint):
        mock_response = create_mock_response(123456789, None)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None


def _make_sse_response(json_data: dict, perf_ns: int = 123456789) -> SSEMessage:
    """Helper to create an SSEMessage from a JSON payload."""
    return SSEMessage(
        perf_ns=perf_ns,
        packets=[
            SSEField(name="data", value=orjson.dumps(json_data).decode()),
        ],
    )


class TestAnthropicMessagesParseResponseStreaming:
    """Tests for MessagesEndpoint parse_response (streaming SSE)."""

    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.MESSAGES)
        return create_endpoint_with_mock_transport(MessagesEndpoint, model_endpoint)

    def test_message_start_returns_usage(self, endpoint):
        response = _make_sse_response(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "usage": {"input_tokens": 25, "output_tokens": 0},
                },
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is not None
        assert parsed.usage.get("input_tokens") == 25

    def test_text_delta(self, endpoint):
        response = _make_sse_response(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello"

    def test_thinking_delta(self, endpoint):
        response = _make_sse_response(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "thinking_delta", "thinking": "Analyzing..."},
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.reasoning == "Analyzing..."

    def test_signature_delta_returns_none(self, endpoint):
        response = _make_sse_response(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "signature_delta", "signature": "abc123"},
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_input_json_delta_returns_tool_call_data(self, endpoint):
        response = _make_sse_response(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"location": "San Fra',
                },
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.tool_call_text == '{"location": "San Fra'

    def test_empty_input_json_delta_returns_none(self, endpoint):
        response = _make_sse_response(
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": ""},
            }
        )

        assert endpoint.parse_response(response) is None

    def test_message_delta_returns_usage(self, endpoint):
        response = _make_sse_response(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 42},
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is not None
        assert parsed.usage.get("output_tokens") == 42

    @pytest.mark.parametrize(
        "event_type",
        ["ping", "content_block_start", "content_block_stop", "message_stop"],
    )
    def test_non_content_events_return_none(self, endpoint, event_type):
        response = _make_sse_response({"type": event_type})

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_error_event_returns_none(self, endpoint):
        response = _make_sse_response(
            {
                "type": "error",
                "error": {
                    "type": "overloaded_error",
                    "message": "Overloaded",
                },
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_streaming_sequence(self, endpoint):
        """Test parsing a full streaming sequence returns correct data types."""
        events = [
            {
                "type": "message_start",
                "message": {
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {"type": "ping"},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]

        results = []
        for data in events:
            parsed = endpoint.parse_response(_make_sse_response(data))
            if parsed:
                results.append(parsed)

        # message_start (usage) + 2 text_deltas + message_delta (usage)
        assert len(results) == 4
        assert results[0].usage is not None  # message_start
        assert isinstance(results[1].data, TextResponseData)
        assert results[1].data.text == "Hello"
        assert isinstance(results[2].data, TextResponseData)
        assert results[2].data.text == " world"
        assert results[3].usage is not None  # message_delta


class TestAnthropicMessagesExtractPayloadInputs:
    """Tests for the dag5 extract_payload_inputs hook on MessagesEndpoint."""

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES),
        )

    def test_simple_text_message(self, endpoint):
        out = endpoint.extract_payload_inputs(
            {"messages": [{"role": "user", "content": "hello"}]}
        )
        assert out.texts == ["hello"]
        assert out.image_count == 0

    def test_string_system_prepended(self, endpoint):
        out = endpoint.extract_payload_inputs(
            {
                "system": "you are concise",
                "messages": [{"role": "user", "content": "hello"}],
            }
        )
        assert out.texts == ["you are concise", "hello"]

    def test_list_form_system_prepended_in_order(self, endpoint):
        out = endpoint.extract_payload_inputs(
            {
                "system": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
                "messages": [{"role": "user", "content": "body"}],
            }
        )
        assert out.texts[:2] == ["first", "second"]
        assert out.texts[2] == "body"

    def test_image_content_block_uses_anthropic_shape(self, endpoint):
        out = endpoint.extract_payload_inputs(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "describe"},
                            {
                                "type": "image",
                                "source": {"type": "url", "url": "http://x/y.jpg"},
                            },
                        ],
                    }
                ]
            }
        )
        assert out.texts == ["describe"]
        assert out.image_count == 1

    def test_openai_image_url_shape_does_not_count(self, endpoint):
        # Anthropic's PART_TYPES has IMAGE={"image"}, so an OpenAI-style
        # ``image_url`` part must NOT inflate the count - it is not an
        # Anthropic-shaped block and would be rejected by the server.
        out = endpoint.extract_payload_inputs(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "http://x"}},
                        ],
                    }
                ]
            }
        )
        assert out.image_count == 0

    def test_tools_input_schema_collected(self, endpoint):
        out = endpoint.extract_payload_inputs(
            {
                "messages": [{"role": "user", "content": "search please"}],
                "tools": [
                    {
                        "name": "web_search",
                        "description": "search the web",
                        "input_schema": {
                            "type": "object",
                            "properties": {"q": {"type": "string"}},
                            "required": ["q"],
                        },
                    }
                ],
            }
        )
        # Base walk gets name + description; we add serialised input_schema.
        assert "web_search" in out.texts
        assert "search the web" in out.texts
        serialised = [t for t in out.texts if t.startswith("{")]
        assert len(serialised) == 1
        assert "properties" in serialised[0]

    def test_tool_use_content_block_collected(self, endpoint):
        out = endpoint.extract_payload_inputs(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool_1",
                                "name": "calculator",
                                "input": {"a": 2, "b": 3},
                            }
                        ],
                    }
                ]
            }
        )
        assert "calculator" in out.texts
        serialised = [t for t in out.texts if t.startswith("{")]
        assert len(serialised) == 1
        assert orjson.loads(serialised[0]) == {"a": 2, "b": 3}

    def test_tool_result_string_content(self, endpoint):
        out = endpoint.extract_payload_inputs(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_1",
                                "content": "result text",
                            }
                        ],
                    }
                ]
            }
        )
        assert "result text" in out.texts

    def test_tool_result_list_content(self, endpoint):
        out = endpoint.extract_payload_inputs(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool_1",
                                "content": [
                                    {"type": "text", "text": "first"},
                                    {"type": "text", "text": "second"},
                                ],
                            }
                        ],
                    }
                ]
            }
        )
        assert "first" in out.texts
        assert "second" in out.texts


class TestAnthropicMessagesRenderHooks:
    """Tests for content-part render hooks (image/audio/video)."""

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES),
        )

    def test_image_part_anthropic_shape(self, endpoint):
        part = endpoint._render_image_part("http://x/y.jpg")
        assert part == {
            "type": "image",
            "source": {"type": "url", "url": "http://x/y.jpg"},
        }

    def test_audio_part_raises_not_implemented(self, endpoint):
        with pytest.raises(NotImplementedError, match="audio"):
            endpoint._render_audio_part("wav,base64data")

    def test_video_part_raises_not_implemented(self, endpoint):
        with pytest.raises(NotImplementedError, match="video"):
            endpoint._render_video_part("http://x/v.mp4")

    def test_format_payload_with_image_uses_anthropic_shape(self, endpoint):
        from aiperf.common.models import Image

        model_endpoint = create_model_endpoint(EndpointType.MESSAGES)
        turn = Turn(
            texts=[Text(contents=["caption please"])],
            images=[Image(contents=["http://x/y.jpg"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)
        content = payload["messages"][0]["content"]
        assert isinstance(content, list)
        image_parts = [p for p in content if p.get("type") == "image"]
        assert len(image_parts) == 1
        assert image_parts[0]["source"] == {"type": "url", "url": "http://x/y.jpg"}


class TestAnthropicMessagesBuildAssistantTurn:
    """Tests for build_assistant_turn override (DAG/FORK replay support)."""

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES),
        )

    @staticmethod
    def _record(events: list[dict]):
        from aiperf.common.models import RequestRecord, TextResponse

        responses = [
            TextResponse(
                perf_ns=ns,
                text=orjson.dumps(event).decode(),
                content_type="application/json",
            )
            for ns, event in enumerate(events, start=1)
        ]
        return RequestRecord(
            responses=responses,
            start_perf_ns=0,
            end_perf_ns=len(events) + 1,
        )

    def test_text_only_falls_back_to_base(self, endpoint):
        record = self._record(
            [
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "hello"}],
                }
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.role == "assistant"
        # Base impl path: text in Turn.texts, no raw_messages.
        assert turn.raw_messages is None
        assert turn.texts[0].contents == ["hello"]

    def test_non_streaming_with_tool_use(self, endpoint):
        record = self._record(
            [
                {
                    "type": "message",
                    "content": [
                        {"type": "text", "text": "Sure, calling..."},
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "calculator",
                            "input": {"a": 2, "b": 3},
                        },
                    ],
                }
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.raw_messages is not None
        msg = turn.raw_messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"][0] == {"type": "text", "text": "Sure, calling..."}
        tool_use = msg["content"][1]
        assert tool_use["type"] == "tool_use"
        assert tool_use["name"] == "calculator"
        assert tool_use["input"] == {"a": 2, "b": 3}

    def test_streaming_reassembles_tool_use_input_json(self, endpoint):
        # Streaming: tool_use block arrives via content_block_start +
        # input_json_delta fragments + content_block_stop.
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "search",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"q":'},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '"hi"}'},
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        assert turn.raw_messages is not None
        tool_use = turn.raw_messages[0]["content"][0]
        assert tool_use["type"] == "tool_use"
        assert tool_use["name"] == "search"
        assert tool_use["input"] == {"q": "hi"}

    def test_streaming_text_plus_tool_use(self, endpoint):
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "thinking..."},
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "f",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        msg = turn.raw_messages[0]
        # Text block first, tool_use second.
        assert msg["content"][0] == {"type": "text", "text": "thinking..."}
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["input"] == {}

    def test_non_streaming_preserves_unknown_tool_use_fields(self, endpoint):
        # Real Claude Code traffic carries a ``caller`` field on tool_use
        # blocks beyond the spec's id/name/input. The accumulator copies
        # every field so _finalise_tool_use round-trips it verbatim.
        record = self._record(
            [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "Bash",
                            "input": {"command": "ls"},
                            "caller": {"type": "direct"},
                        }
                    ],
                }
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        tool_use = turn.raw_messages[0]["content"][0]
        assert tool_use["caller"] == {"type": "direct"}
        assert tool_use["name"] == "Bash"
        assert tool_use["input"] == {"command": "ls"}

    def test_streaming_preserves_unknown_tool_use_fields(self, endpoint):
        # Streaming content_block_start envelope carries ``caller``;
        # _absorb_content_block_start copies all envelope fields.
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "Bash",
                        "input": {},
                        "caller": {"type": "direct"},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": "{}"},
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        tool_use = turn.raw_messages[0]["content"][0]
        assert tool_use["caller"] == {"type": "direct"}

    def test_streaming_thinking_then_tool_use(self, endpoint):
        # Real Anthropic streaming sequence: thinking content_block at idx 0
        # (with thinking_delta + signature_delta), tool_use at idx 1.
        # Override path captures both; thinking comes first in output.
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Let me think"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig_part_1"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "_part_2"},
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "search",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"q":"x"}'},
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        blocks = turn.raw_messages[0]["content"]
        # Thinking block first (its index 0 < tool_use's index 1).
        assert blocks[0]["type"] == "thinking"
        assert blocks[0]["thinking"] == "Let me think"
        assert blocks[0]["signature"] == "sig_part_1_part_2"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["input"] == {"q": "x"}

    def test_non_streaming_thinking_block_captured(self, endpoint):
        record = self._record(
            [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "reasoning text",
                            "signature": "sig123",
                        },
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "f",
                            "input": {},
                        },
                    ],
                }
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        blocks = turn.raw_messages[0]["content"]
        assert blocks[0] == {
            "type": "thinking",
            "thinking": "reasoning text",
            "signature": "sig123",
        }
        assert blocks[1]["type"] == "tool_use"


class TestAnthropicMessagesRawSystem:
    """Tests for Turn.raw_system list-form system blocks (per-block cache_control)."""

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES),
        )

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.MESSAGES)

    def test_raw_system_overrides_string_system_message(self, endpoint, model_endpoint):
        system_blocks = [
            {
                "type": "text",
                "text": "you are concise",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        turn = Turn(
            texts=[Text(contents=["hi"])],
            raw_system=system_blocks,
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="this string should be ignored",
        )
        payload = endpoint.format_payload(request_info)
        assert payload["system"] == system_blocks
        # Cache_control round-trips verbatim.
        assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_raw_system_falls_back_to_system_message_when_unset(
        self, endpoint, model_endpoint
    ):
        turn = Turn(texts=[Text(contents=["hi"])])
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="be helpful",
        )
        payload = endpoint.format_payload(request_info)
        assert payload["system"] == "be helpful"

    def test_raw_system_latest_turn_wins(self, endpoint, model_endpoint):
        # Following _latest_turn_attr semantics: the most recent non-None
        # raw_system across the turn list is what gets used.
        turn1 = Turn(
            texts=[Text(contents=["t1"])],
            raw_system=[{"type": "text", "text": "first"}],
        )
        turn2 = Turn(texts=[Text(contents=["t2"])])  # raw_system=None
        turn3 = Turn(
            texts=[Text(contents=["t3"])],
            raw_system=[{"type": "text", "text": "latest"}],
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[turn1, turn2, turn3]
        )
        payload = endpoint.format_payload(request_info)
        assert payload["system"] == [{"type": "text", "text": "latest"}]


class TestAnthropicMessagesToolUseParsing:
    """Non-streaming tool_use blocks count toward client-side OSL.

    Precedence matches the chat endpoint: reasoning > text+tool > tool > text.
    """

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES),
        )

    def test_tool_use_only_returns_tool_call_data(self, endpoint):
        response = create_mock_response(
            123456789,
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
        assert parsed.data.content is None

    def test_text_and_tool_use_returns_both_portions(self, endpoint):
        # The standard agentic shape: Claude talks, then dispatches a tool.
        # Both portions are model-generated tokens that usage.output_tokens
        # counts, so both must reach client-side OSL.
        response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [
                    {"type": "text", "text": "Let me check the weather."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "get_weather",
                        "input": {"location": "SF"},
                    },
                ],
            },
        )

        parsed = endpoint.parse_response(response)

        assert isinstance(parsed.data, ToolCallResponseData)
        assert parsed.data.content == "Let me check the weather."
        assert "get_weather" in parsed.data.tool_call_text
        assert parsed.data.get_text().startswith("Let me check the weather.")

    def test_thinking_wins_over_tool_use(self, endpoint):
        # Chat-endpoint parity: reasoning takes precedence; tool text is not
        # folded into ReasoningResponseData (it has no tool field).
        response = create_mock_response(
            123456789,
            {
                "type": "message",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "answer"},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "calc",
                        "input": {"a": 1},
                    },
                ],
            },
        )

        parsed = endpoint.parse_response(response)

        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.reasoning == "hmm"
        assert parsed.data.content == "answer"


class TestAnthropicMessagesDataUriImages:
    """Data-URI images render as base64 sources; URLs stay url sources."""

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES),
        )

    def test_data_uri_renders_base64_source(self, endpoint):
        part = endpoint._render_image_part("data:image/png;base64,QUJD")
        assert part == {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "QUJD",
            },
        }

    def test_data_uri_without_media_type_defaults_png(self, endpoint):
        part = endpoint._render_image_part("data:;base64,QUJD")
        assert part["source"]["media_type"] == "image/png"

    def test_plain_url_still_renders_url_source(self, endpoint):
        part = endpoint._render_image_part("https://example.com/cat.jpg")
        assert part["source"] == {"type": "url", "url": "https://example.com/cat.jpg"}


class TestAnthropicMessagesSplitUsageMerge:
    """extract_response_data folds message_start usage into the final usage.

    Docs-canonical servers omit input_tokens from message_delta; the record
    layer keeps only the LAST non-empty usage, so without the fold the
    server-reported input tokens would be lost for streaming requests.
    """

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES, streaming=True),
        )

    @staticmethod
    def _record(events: list[dict]):
        from aiperf.common.models import RequestRecord

        record = RequestRecord()
        record.responses = [_make_sse_response(e) for e in events]
        return record

    def test_docs_canonical_split_usage_is_merged(self, endpoint):
        record = self._record(
            [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "usage": {
                            "input_tokens": 25,
                            "cache_read_input_tokens": 7,
                            "output_tokens": 1,
                        },
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hi"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 15},
                },
            ]
        )

        parsed = endpoint.extract_response_data(record)

        final_usage = [p.usage for p in parsed if p.usage][-1]
        # Final chunk keeps its own output count and inherits the input-side
        # keys message_delta omitted.
        assert final_usage.completion_tokens == 15
        assert final_usage.prompt_uncached_tokens == 25
        assert final_usage.prompt_tokens == 32  # 25 uncached + 7 cache read
        assert final_usage.prompt_cache_read_tokens == 7

    def test_cumulative_servers_untouched(self, endpoint):
        # Modern api.anthropic.com and Dynamo repeat full cumulative usage in
        # message_delta; existing keys must win over message_start values.
        record = self._record(
            [
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
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"input_tokens": 25, "output_tokens": 40},
                },
            ]
        )

        parsed = endpoint.extract_response_data(record)

        final_usage = [p.usage for p in parsed if p.usage][-1]
        assert final_usage.completion_tokens == 40
        assert final_usage.prompt_tokens == 25

    def test_single_usage_nonstreaming_is_untouched(self, endpoint):
        record = self._record(
            [
                {
                    "type": "message",
                    "content": [{"type": "text", "text": "Hi"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                }
            ]
        )

        parsed = endpoint.extract_response_data(record)

        assert [p.usage for p in parsed if p.usage][-1].prompt_tokens == 10


class TestAnthropicMessagesDefensiveBranches:
    """Malformed and edge-shape inputs must be skipped, never crash.

    One test per defensive branch in the payload walks, the replay
    accumulators, and the parse dispatch.
    """

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES),
        )

    @staticmethod
    def _record(events: list[dict]):
        from aiperf.common.models import RequestRecord, TextResponse

        responses = [
            TextResponse(
                perf_ns=ns,
                text=orjson.dumps(event).decode(),
                content_type="application/json",
            )
            for ns, event in enumerate(events, start=1)
        ]
        return RequestRecord(
            responses=responses,
            start_perf_ns=0,
            end_perf_ns=len(events) + 1,
        )

    # --- payload-input walk helpers -------------------------------------

    def test_walk_system_accepts_plain_string_parts(self):
        result = ExtractedPayload()
        _walk_system(
            {"system": ["plain part", "", {"type": "text", "text": "typed part"}]},
            result,
        )
        assert result.texts == ["plain part", "typed part"]

    def test_walk_tool_schemas_skips_non_dict_tools(self):
        result = ExtractedPayload()
        _walk_tool_schemas(
            {
                "tools": [
                    "not-a-dict",
                    {"name": "t", "input_schema": {"type": "object"}},
                ]
            },
            result,
        )
        assert len(result.texts) == 1
        assert "object" in result.texts[0]

    @pytest.mark.parametrize(
        "payload",
        [
            param({}, id="messages_absent"),
            param({"messages": "not-a-list"}, id="messages_not_list"),
            param({"messages": [42]}, id="message_not_dict"),
            param({"messages": [{"role": "user", "content": [42]}]}, id="part_not_dict"),
        ],
    )  # fmt: skip
    def test_walk_tool_blocks_tolerates_malformed_shapes(self, payload):
        result = ExtractedPayload()
        _walk_tool_blocks(payload, result)
        assert result.texts == []

    def test_tool_result_content_non_string_non_list_skipped(self):
        result = ExtractedPayload()
        _walk_tool_blocks(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "content": 42}],
                    }
                ]
            },
            result,
        )
        assert result.texts == []

    def test_tool_result_content_list_skips_non_dict_subblocks(self):
        result = ExtractedPayload()
        _walk_tool_blocks(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": [42, {"type": "text", "text": "ok"}],
                            }
                        ],
                    }
                ]
            },
            result,
        )
        assert result.texts == ["ok"]

    # --- format_payload conversation-level fields ------------------------

    def test_raw_tools_passthrough(self, endpoint):
        model_endpoint = create_model_endpoint(EndpointType.MESSAGES)
        tools = [{"name": "calc", "input_schema": {"type": "object"}}]
        turn = Turn(
            texts=[Text(contents=["hi"])],
            model="claude-sonnet-4-20250514",
            raw_tools=tools,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)
        assert payload["tools"] == tools

    def test_extra_body_merged_last(self, endpoint):
        model_endpoint = create_model_endpoint(EndpointType.MESSAGES)
        turn = Turn(
            texts=[Text(contents=["hi"])],
            model="claude-sonnet-4-20250514",
            extra_body={"temperature": 0.5},
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])
        payload = endpoint.format_payload(request_info)
        assert payload["temperature"] == 0.5

    # --- replay accumulators ---------------------------------------------

    def test_absorb_message_skips_non_dict_blocks(self, endpoint):
        record = self._record(
            [
                {
                    "type": "message",
                    "content": [
                        42,
                        {"type": "tool_use", "id": "t1", "name": "calc", "input": {}},
                    ],
                }
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn is not None
        blocks = turn.raw_messages[0]["content"]
        assert [b["type"] for b in blocks] == ["tool_use"]

    def test_delta_without_index_is_dropped(self, endpoint):
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking"},
                },
                {
                    "type": "content_block_delta",
                    "delta": {"type": "thinking_delta", "thinking": "lost"},
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        thinking_block = turn.raw_messages[0]["content"][0]
        assert thinking_block["thinking"] == ""

    def test_delta_for_unopened_index_is_dropped(self, endpoint):
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "calc"},
                },
                {
                    "type": "content_block_delta",
                    "index": 7,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"lost": 1}',
                    },
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        tool_block = turn.raw_messages[0]["content"][0]
        assert tool_block["input"] == {}

    def test_non_string_delta_fragment_is_dropped(self, endpoint):
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": 42},
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn.raw_messages[0]["content"][0]["thinking"] == ""

    def test_malformed_streamed_tool_json_preserved_as_string(self, endpoint):
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "calc"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": "not{json"},
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn.raw_messages[0]["content"][0]["input"] == "not{json"

    def test_tool_use_without_input_deltas_gets_empty_input(self, endpoint):
        record = self._record(
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "calc"},
                }
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn.raw_messages[0]["content"][0]["input"] == {}

    def test_empty_json_responses_skipped_during_replay(self, endpoint):
        record = self._record(
            [
                {},
                {
                    "type": "message",
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "calc", "input": {}}
                    ],
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        assert turn.raw_messages[0]["content"][0]["type"] == "tool_use"

    # --- parse dispatch edge shapes ---------------------------------------

    @pytest.mark.parametrize(
        "json_obj",
        [
            param({"foo": "bar"}, id="no_type_key"),
            param({"type": "message", "content": []}, id="message_no_data_no_usage"),
            param({"type": "message_start", "message": {}}, id="message_start_no_usage"),
            param({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}, id="message_delta_no_usage"),
            param({"type": "weird_event"}, id="unknown_event_type"),
            param({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": ""}}, id="empty_text_delta"),
            param({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ""}}, id="empty_thinking_delta"),
        ],
    )  # fmt: skip
    def test_parse_response_returns_none_for_contentless_shapes(
        self, endpoint, json_obj
    ):
        assert endpoint.parse_response(_make_sse_response(json_obj)) is None

    def test_non_dict_content_blocks_skipped_in_extraction(self, endpoint):
        parsed = endpoint.parse_response(
            create_mock_response(
                1,
                {
                    "type": "message",
                    "content": [42, {"type": "text", "text": "hi"}],
                },
            )
        )
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "hi"


class TestAnthropicMessagesFalsyFieldBranches:
    """Blocks with empty or wrongly-typed field values contribute nothing.

    Complements TestAnthropicMessagesDefensiveBranches: those tests cover
    wrong container shapes; these cover falsy/non-string leaf values.
    """

    @pytest.fixture
    def endpoint(self):
        return create_endpoint_with_mock_transport(
            MessagesEndpoint,
            create_model_endpoint(EndpointType.MESSAGES),
        )

    def test_walk_system_empty_string_ignored(self):
        result = ExtractedPayload()
        _walk_system({"system": ""}, result)
        assert result.texts == []

    def test_walk_system_list_skips_empty_and_non_string_text(self):
        result = ExtractedPayload()
        _walk_system(
            {
                "system": [
                    {"type": "text", "text": ""},
                    {"type": "text", "text": 42},
                    {"type": "text", "text": "kept"},
                ]
            },
            result,
        )
        assert result.texts == ["kept"]

    def test_walk_tool_schemas_skips_tools_without_schema(self):
        result = ExtractedPayload()
        _walk_tool_schemas({"tools": [{"name": "schemaless"}]}, result)
        assert result.texts == []

    def test_tool_use_without_name_and_non_dict_input_contributes_nothing(self):
        result = ExtractedPayload()
        _walk_tool_blocks(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "tool_use", "input": "not-a-dict"}],
                    }
                ]
            },
            result,
        )
        assert result.texts == []

    def test_tool_result_empty_string_content_ignored(self):
        result = ExtractedPayload()
        _walk_tool_blocks(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "content": ""}],
                    }
                ]
            },
            result,
        )
        assert result.texts == []

    def test_tool_result_list_skips_non_text_and_empty_subblocks(self):
        result = ExtractedPayload()
        _walk_tool_blocks(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "content": [
                                    {"type": "other"},
                                    {"type": "text", "text": ""},
                                ],
                            }
                        ],
                    }
                ]
            },
            result,
        )
        assert result.texts == []

    def test_replay_ignores_structural_and_wrongly_typed_events(self, endpoint):
        record = TestAnthropicMessagesDefensiveBranches._record(
            [
                {"type": "ping"},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": 42},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "unknown_delta", "x": 1},
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "text", "text": 42},
                        {"type": "unknown_block"},
                        {"type": "tool_use", "id": "t1", "name": "calc", "input": {}},
                    ],
                },
            ]
        )
        turn = endpoint.build_assistant_turn(record)
        blocks = turn.raw_messages[0]["content"]
        assert [b["type"] for b in blocks] == ["tool_use"]

    def test_extraction_skips_blocks_with_falsy_or_missing_fields(self, endpoint):
        parsed = endpoint.parse_response(
            create_mock_response(
                1,
                {
                    "type": "message",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "thinking", "thinking": ""},
                        {"type": "unknown_block"},
                        {"type": "tool_use"},
                    ],
                },
            )
        )
        assert parsed is None
