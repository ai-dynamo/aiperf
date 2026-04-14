# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pytest import param

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Audio, Image, Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import ReasoningResponseData, TextResponseData
from aiperf.common.models.usage_models import Usage
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)

_PERF_NS = 123456789
_NOT_PRESENT = object()


def _responses_model_endpoint(
    streaming: bool = False,
    extra: list[tuple] | None = None,
    use_server_token_count: bool = False,
) -> ModelEndpointInfo:
    """Create a RESPONSES ModelEndpointInfo with custom streaming/token-count options."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.RESPONSES,
            base_url="http://localhost:8000",
            streaming=streaming,
            extra=extra or [],
            use_server_token_count=use_server_token_count,
        ),
    )


def _msg(text: str) -> dict:
    """Shorthand for a message output item with a single output_text part."""
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def _reasoning(text: str) -> dict:
    """Shorthand for a reasoning output item with a single summary_text part."""
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": text}]}


class TestResponsesEndpoint:
    """Tests for ResponsesEndpoint.format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(EndpointType.RESPONSES)

    @pytest.fixture
    def streaming_model_endpoint(self):
        return create_model_endpoint(EndpointType.RESPONSES, streaming=True)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(ResponsesEndpoint, model_endpoint)

    def test_format_payload_simple_text(self, endpoint, model_endpoint):
        """Simple single text -> string shortcut content."""
        turn = Turn(texts=[Text(contents=["Hello, world!"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert len(payload["input"]) == 1
        assert payload["input"][0]["content"] == "Hello, world!"
        assert payload["input"][0]["role"] == "user"

    def test_format_payload_multimodal_text_and_images(self, endpoint, model_endpoint):
        """Multi-modal message with text and images uses content array."""
        turn = Turn(
            texts=[Text(contents=["Describe this image"]), Text(contents=["And this"])],
            images=[Image(contents=["data:image/png;base64,abc123"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        content = payload["input"][0]["content"]
        assert isinstance(content, list)
        text_parts = [c for c in content if c["type"] == "input_text"]
        image_parts = [c for c in content if c["type"] == "input_image"]
        assert len(text_parts) == 2
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"] == "data:image/png;base64,abc123"

    def test_format_payload_audio(self, endpoint, model_endpoint):
        """Audio input is formatted correctly."""
        turn = Turn(
            texts=[Text(contents=["What did they say?"])],
            audios=[Audio(contents=["wav,YWJjMTIz"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        content = payload["input"][0]["content"]
        assert isinstance(content, list)
        audio_parts = [c for c in content if c["type"] == "input_audio"]
        assert len(audio_parts) == 1
        assert audio_parts[0]["input_audio"]["format"] == "wav"
        assert audio_parts[0]["input_audio"]["data"] == "YWJjMTIz"

    def test_format_payload_invalid_audio_raises(self, endpoint, model_endpoint):
        """Invalid audio format (no comma) raises ValueError."""
        turn = Turn(
            texts=[Text(contents=["Test"])],
            audios=[Audio(contents=["invalid_no_comma"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="Audio content must be in the format"):
            endpoint.format_payload(request_info)

    def test_format_payload_system_message_becomes_instructions(
        self, endpoint, model_endpoint
    ):
        """system_message maps to top-level instructions field."""
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            system_message="You are a helpful assistant.",
        )

        payload = endpoint.format_payload(request_info)

        assert payload["instructions"] == "You are a helpful assistant."
        assert all(item.get("role") != "system" for item in payload["input"])

    def test_format_payload_no_system_message_no_instructions(
        self, endpoint, model_endpoint
    ):
        """No system_message means no instructions key."""
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "instructions" not in payload

    def test_format_payload_user_context_message(self, endpoint, model_endpoint):
        """user_context_message is prepended as a user input item."""
        turn = Turn(texts=[Text(contents=["Hello"])], model="test-model")
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn],
            user_context_message="Background context here.",
        )

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 2
        assert payload["input"][0]["role"] == "user"
        assert payload["input"][0]["content"] == "Background context here."
        assert payload["input"][1]["content"] == "Hello"

    def test_format_payload_max_tokens_becomes_max_output_tokens(
        self, endpoint, model_endpoint
    ):
        """max_tokens maps to max_output_tokens."""
        turn = Turn(
            texts=[Text(contents=["Generate"])], model="test-model", max_tokens=500
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_output_tokens"] == 500
        assert "max_tokens" not in payload
        assert "max_completion_tokens" not in payload

    def test_format_payload_no_max_tokens(self, endpoint, model_endpoint):
        """No max_tokens means no max_output_tokens key."""
        turn = Turn(
            texts=[Text(contents=["Generate"])], model="test-model", max_tokens=None
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "max_output_tokens" not in payload

    def test_format_payload_streaming(self, streaming_model_endpoint):
        """Stream flag from endpoint config."""
        endpoint = create_endpoint_with_mock_transport(
            ResponsesEndpoint, streaming_model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Test"])], model="test-model")
        request_info = create_request_info(
            model_endpoint=streaming_model_endpoint, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["stream"] is True

    def test_format_payload_extra_params(self):
        """Extra parameters are merged into payload."""
        extra = [("temperature", 0.7), ("top_p", 0.9)]
        me = create_model_endpoint(EndpointType.RESPONSES, extra=extra)
        ep = create_endpoint_with_mock_transport(ResponsesEndpoint, me)
        turn = Turn(texts=[Text(contents=["Test"])], model="test-model")
        request_info = create_request_info(model_endpoint=me, turns=[turn])

        payload = ep.format_payload(request_info)

        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9

    def test_format_payload_model_fallback(self, endpoint, model_endpoint):
        """Turn model=None falls back to primary model name."""
        turn = Turn(texts=[Text(contents=["Test"])], model=None)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_format_payload_empty_turns_raises(self, endpoint, model_endpoint):
        """Empty turns list raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint)
        request_info = request_info.model_copy(update={"turns": []})

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_multi_turn(self, endpoint, model_endpoint):
        """Multi-turn conversation includes all turns."""
        turn1 = Turn(texts=[Text(contents=["First"])], model="m1")
        turn2 = Turn(texts=[Text(contents=["Second"])], model="m2", role="assistant")
        turn3 = Turn(texts=[Text(contents=["Third"])], model="m3")
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[turn1, turn2, turn3]
        )

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 3
        assert payload["input"][0]["content"] == "First"
        assert payload["input"][1]["role"] == "assistant"
        assert payload["input"][2]["content"] == "Third"
        assert payload["model"] == "m3"

    def test_format_payload_filters_empty_multimodal_content(
        self, endpoint, model_endpoint
    ):
        """Empty strings in multi-modal content are filtered out."""
        turn = Turn(
            texts=[Text(contents=["Valid text", "", "Another valid"])],
            images=[Image(contents=["", "data:image/png;base64,img1"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        content = payload["input"][0]["content"]
        text_parts = [c for c in content if c["type"] == "input_text"]
        image_parts = [c for c in content if c["type"] == "input_image"]
        assert len(text_parts) == 2
        assert len(image_parts) == 1

    def test_format_payload_single_text_empty_contents_produces_empty_list(
        self, endpoint, model_endpoint
    ):
        """Single text with empty contents list falls through to multimodal branch."""
        turn = Turn(texts=[Text(contents=[])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"][0]["content"] == []

    def test_format_payload_explicit_role(self, endpoint, model_endpoint):
        """Explicit turn role is used instead of default 'user'."""
        turn = Turn(
            texts=[Text(contents=["I am the assistant"])],
            model="test-model",
            role="assistant",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"][0]["role"] == "assistant"

    def test_format_payload_multiple_audios_formatted(self, endpoint, model_endpoint):
        """Multiple audio contents are all formatted correctly."""
        turn = Turn(
            texts=[Text(contents=["What is this?"])],
            audios=[
                Audio(contents=["wav,audio1data"]),
                Audio(contents=["mp3,audio2data"]),
            ],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        content = payload["input"][0]["content"]
        audio_parts = [c for c in content if c["type"] == "input_audio"]
        assert len(audio_parts) == 2
        assert audio_parts[0]["input_audio"] == {"data": "audio1data", "format": "wav"}
        assert audio_parts[1]["input_audio"] == {"data": "audio2data", "format": "mp3"}

    def test_format_payload_empty_audio_contents_filtered(
        self, endpoint, model_endpoint
    ):
        """Empty strings in audio contents are filtered out."""
        turn = Turn(
            texts=[Text(contents=["Describe"])],
            audios=[Audio(contents=["", "wav,realdata"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        content = payload["input"][0]["content"]
        audio_parts = [c for c in content if c["type"] == "input_audio"]
        assert len(audio_parts) == 1
        assert audio_parts[0]["input_audio"]["data"] == "realdata"

    @pytest.mark.parametrize(
        ("streaming", "extra", "expected"),
        [
            param(True, [], {"include_usage": True}, id="auto_set"),
            param(
                True,
                [("stream_options", {"include_usage": True})],
                {"include_usage": True},
                id="preserves_existing",
            ),
            param(
                True,
                [("stream_options", {"some_option": True})],
                {"some_option": True, "include_usage": True},
                id="adds_to_existing",
            ),
            param(
                True,
                [("stream_options", None)],
                {"include_usage": True},
                id="none_replaced",
            ),
            param(False, [], _NOT_PRESENT, id="not_added_when_not_streaming"),
        ],
    )  # fmt: skip
    def test_format_payload_stream_options_with_server_token_count(
        self, streaming, extra, expected
    ):
        """stream_options behavior when use_server_token_count=True."""
        me = _responses_model_endpoint(
            streaming=streaming, extra=extra, use_server_token_count=True
        )
        ep = create_endpoint_with_mock_transport(ResponsesEndpoint, me)
        turn = Turn(texts=[Text(contents=["Test"])], model="test-model")
        request_info = create_request_info(model_endpoint=me, turns=[turn])

        payload = ep.format_payload(request_info)

        if expected is _NOT_PRESENT:
            assert "stream_options" not in payload
        else:
            assert payload["stream_options"] == expected


class TestResponsesEndpointParseResponse:
    """Tests for ResponsesEndpoint.parse_response."""

    @pytest.fixture
    def endpoint(self):
        me = create_model_endpoint(EndpointType.RESPONSES)
        return create_endpoint_with_mock_transport(ResponsesEndpoint, me)

    # --- Returns None ---

    @pytest.mark.parametrize(
        "json_data",
        [
            param(None, id="none_json"),
            param({"some_field": "value"}, id="unknown_format"),
            param(
                {"object": "response", "output": []}, id="empty_output_no_usage"
            ),
            param(
                {"type": "response.completed", "response": {}},
                id="completed_no_usage",
            ),
            param(
                {"type": "response.completed"}, id="completed_no_response_key"
            ),
            param(
                {"type": "response.output_text.delta", "delta": ""},
                id="empty_text_delta",
            ),
            param(
                {"type": "response.output_text.delta"}, id="missing_text_delta"
            ),
            param(
                {"type": "response.reasoning_text.delta", "delta": ""},
                id="empty_reasoning_delta",
            ),
            param(
                {"type": "response.reasoning_text.delta"}, id="missing_reasoning_delta"
            ),
            param({"type": "response.created"}, id="event_created"),
            param({"type": "response.in_progress"}, id="event_in_progress"),
            param(
                {"type": "response.output_item.added"}, id="event_output_item_added"
            ),
            param(
                {"type": "response.content_part.added"}, id="event_content_part_added"
            ),
            param(
                {"type": "response.output_text.done"}, id="event_output_text_done_no_text"
            ),
            param(
                {"type": "response.output_text.done", "text": ""},
                id="event_output_text_done_empty",
            ),
            param(
                {"type": "response.completed", "response": None},
                id="completed_null_response",
            ),
            param(
                {"type": "response.output_item.done"}, id="event_output_item_done"
            ),
        ],
    )  # fmt: skip
    def test_parse_response_returns_none(self, endpoint, json_data):
        """Events and responses with no extractable content return None."""
        assert (
            endpoint.parse_response(create_mock_response(_PERF_NS, json_data)) is None
        )

    # --- Extracts text ---

    @pytest.mark.parametrize(
        ("json_data", "expected_text"),
        [
            param(
                {"type": "response.output_text.delta", "delta": "Hello"},
                "Hello",
                id="streaming_delta",
            ),
            param(
                {"type": "response.output_text.done", "text": "Final text"},
                "Final text",
                id="streaming_done_fallback",
            ),
            param(
                {"object": "response", "output_text": "Fallback text"},
                "Fallback text",
                id="output_text_fallback",
            ),
            param(
                {"object": "response", "output": [_msg("Part 1. "), _msg("Part 2.")]},
                "Part 1. Part 2.",
                id="multiple_text_parts",
            ),
            param(
                {
                    "object": "response",
                    "output": ["not a dict", 42, _msg("Valid")],
                },
                "Valid",
                id="non_dict_items_filtered",
            ),
            param(
                {
                    "object": "response",
                    "output": "not a list",
                    "output_text": "Fallback",
                },
                "Fallback",
                id="non_list_output_uses_fallback",
            ),
            param(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": ""},
                                {"type": "output_text", "text": "Real content"},
                            ],
                        }
                    ],
                },
                "Real content",
                id="empty_text_parts_ignored",
            ),
            param(
                {
                    "object": "response",
                    "output": [
                        {"type": "function_call", "name": "get_weather"},
                        _msg("Sunny."),
                    ],
                },
                "Sunny.",
                id="unknown_output_type_ignored",
            ),
            param(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "refusal", "refusal": "Can't do that."},
                                {"type": "output_text", "text": "But this works."},
                            ],
                        }
                    ],
                },
                "But this works.",
                id="unknown_content_type_ignored",
            ),
        ],
    )  # fmt: skip
    def test_parse_response_extracts_text(self, endpoint, json_data, expected_text):
        """Various response formats produce correct TextResponseData."""
        parsed = endpoint.parse_response(create_mock_response(_PERF_NS, json_data))

        assert parsed is not None
        assert parsed.perf_ns == _PERF_NS
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == expected_text

    # --- Extracts reasoning ---

    @pytest.mark.parametrize(
        ("json_data", "expected_reasoning", "expected_content"),
        [
            param(
                {"type": "response.reasoning_text.delta", "delta": "Thinking..."},
                "Thinking...",
                None,
                id="streaming_delta",
            ),
            param(
                {
                    "object": "response",
                    "output": [_reasoning("Let me think..."), _msg("42")],
                },
                "Let me think...",
                "42",
                id="full_with_text",
            ),
            param(
                {"object": "response", "output": [_reasoning("Deep thought...")]},
                "Deep thought...",
                None,
                id="reasoning_only",
            ),
            param(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [
                                {"type": "summary_text", "text": "First, "},
                                {"type": "summary_text", "text": "then, "},
                            ],
                        },
                        _msg("Result"),
                    ],
                },
                "First, then, ",
                "Result",
                id="multiple_summaries",
            ),
            param(
                {
                    "object": "response",
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [
                                "not a dict",
                                {"type": "summary_text", "text": "Valid reasoning"},
                            ],
                        },
                        {
                            "type": "message",
                            "content": [
                                None,
                                {"type": "output_text", "text": "Valid text"},
                            ],
                        },
                    ],
                },
                "Valid reasoning",
                "Valid text",
                id="non_dict_parts_filtered",
            ),
        ],
    )  # fmt: skip
    def test_parse_response_extracts_reasoning(
        self, endpoint, json_data, expected_reasoning, expected_content
    ):
        """Various response formats produce correct ReasoningResponseData."""
        parsed = endpoint.parse_response(create_mock_response(_PERF_NS, json_data))

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.reasoning == expected_reasoning
        assert parsed.data.content == expected_content

    # --- Usage extraction ---

    @pytest.mark.parametrize(
        ("json_data", "expected_usage"),
        [
            param(
                {
                    "object": "response",
                    "output": [_msg("Hello")],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
                {"input_tokens": 10, "output_tokens": 5},
                id="full_response_with_text",
            ),
            param(
                {
                    "object": "response",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
                {"input_tokens": 10, "output_tokens": 0},
                id="full_response_usage_only",
            ),
            param(
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {"input_tokens": 10, "output_tokens": 20},
                    },
                },
                {"input_tokens": 10, "output_tokens": 20},
                id="streaming_completed",
            ),
        ],
    )  # fmt: skip
    def test_parse_response_extracts_usage(self, endpoint, json_data, expected_usage):
        """Usage data is correctly extracted from both streaming and non-streaming."""
        parsed = endpoint.parse_response(create_mock_response(_PERF_NS, json_data))

        assert parsed is not None
        assert parsed.usage == Usage(expected_usage)

        output = json_data.get("output", []) if isinstance(json_data, dict) else []
        has_message = any(
            isinstance(item, dict) and item.get("type") == "message"
            for item in output
            if isinstance(output, list)
        )
        if has_message:
            assert parsed.data is not None
