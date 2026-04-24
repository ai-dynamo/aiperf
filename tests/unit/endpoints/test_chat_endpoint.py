# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models import Audio, Image, Text, Turn, Video
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class TestChatEndpoint:
    """Comprehensive tests for ChatEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for chat."""
        return create_model_endpoint(EndpointType.CHAT)

    @pytest.fixture
    def streaming_model_endpoint(self):
        """Create a test ModelEndpointInfo with streaming enabled."""
        return create_model_endpoint(EndpointType.CHAT, streaming=True)

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a ChatEndpoint instance."""
        return create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)

    def test_format_payload_simple_text(self, endpoint, model_endpoint):
        """Test simple single text message formatting."""
        turn = Turn(
            texts=[Text(contents=["Hello, world!"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "test-model"
        assert payload["stream"] is False
        assert "messages" in payload
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["content"] == "Hello, world!"
        assert "name" not in payload["messages"][0]

    def test_format_payload_multi_modal(self, endpoint, model_endpoint):
        """Test multi-modal message with text and images."""
        turn = Turn(
            texts=[
                Text(contents=["Describe this image"]),
                Text(contents=["And this one too"]),
            ],
            images=[Image(contents=["data:image/png;base64,abc123"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "test-model"
        assert "messages" in payload
        assert isinstance(payload["messages"], list)
        assert len(payload["messages"]) == 1

        message = payload["messages"][0]
        assert isinstance(message["content"], list)
        # Should have 2 text parts + 1 image part
        assert len(message["content"]) == 3
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "text"
        assert message["content"][2]["type"] == "image_url"

    def test_format_payload_with_audio(self, endpoint, model_endpoint):
        """Test audio input formatting."""
        turn = Turn(
            texts=[Text(contents=["What did they say?"])],
            audios=[Audio(contents=["wav,YWJjMTIz"])],  # format,b64
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        message = payload["messages"][0]
        assert isinstance(message["content"], list)
        # Find audio content
        audio_content = [
            c for c in message["content"] if c.get("type") == "input_audio"
        ]
        assert len(audio_content) == 1
        assert audio_content[0]["input_audio"]["format"] == "wav"
        assert audio_content[0]["input_audio"]["data"] == "YWJjMTIz"

    def test_format_payload_with_invalid_audio_format(self, endpoint, model_endpoint):
        """Test that invalid audio format raises ValueError."""
        turn = Turn(
            texts=[Text(contents=["Test"])],
            audios=[Audio(contents=["invalid_no_comma"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="Audio content must be in the format"):
            endpoint.format_payload(request_info)

    def test_format_payload_with_video(self, endpoint, model_endpoint):
        """Test video input formatting."""
        turn = Turn(
            texts=[Text(contents=["Describe the video"])],
            videos=[Video(contents=["https://example.com/video.mp4"])],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        message = payload["messages"][0]
        video_content = [c for c in message["content"] if c.get("type") == "video_url"]
        assert len(video_content) == 1
        assert video_content[0]["video_url"]["url"] == "https://example.com/video.mp4"

    def test_format_payload_max_completion_tokens(self, endpoint, model_endpoint):
        """Test max_completion_tokens is set correctly (default behavior)."""
        turn = Turn(
            texts=[Text(contents=["Generate text"])],
            model="test-model",
            max_tokens=500,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_completion_tokens"] == 500
        assert "max_tokens" not in payload

    def test_format_payload_no_max_tokens(self, endpoint, model_endpoint):
        """Test that max_completion_tokens is not included when None."""
        turn = Turn(
            texts=[Text(contents=["Generate text"])],
            model="test-model",
            max_tokens=None,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "max_completion_tokens" not in payload
        assert "max_tokens" not in payload

    def test_format_payload_legacy_max_tokens(self):
        """Test legacy max_tokens field is used when flag is enabled."""
        # Create model endpoint with use_legacy_max_tokens=True
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                streaming=False,
                extra=[],
                use_legacy_max_tokens=True,
            ),
        )
        endpoint = create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)

        turn = Turn(
            texts=[Text(contents=["Generate text"])],
            model="test-model",
            max_tokens=500,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_tokens"] == 500
        assert "max_completion_tokens" not in payload

    def test_format_payload_legacy_max_tokens_disabled(self):
        """Test max_completion_tokens is used when legacy flag is explicitly disabled."""
        # Create model endpoint with use_legacy_max_tokens=False (explicit)
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                streaming=False,
                extra=[],
                use_legacy_max_tokens=False,
            ),
        )
        endpoint = create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)

        turn = Turn(
            texts=[Text(contents=["Generate text"])],
            model="test-model",
            max_tokens=500,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["max_completion_tokens"] == 500
        assert "max_tokens" not in payload

    def test_format_payload_legacy_max_tokens_none(self):
        """Test neither field is set when max_tokens is None, regardless of legacy flag."""
        # Create model endpoint with use_legacy_max_tokens=True
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                streaming=False,
                extra=[],
                use_legacy_max_tokens=True,
            ),
        )
        endpoint = create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)

        turn = Turn(
            texts=[Text(contents=["Generate text"])],
            model="test-model",
            max_tokens=None,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "max_tokens" not in payload
        assert "max_completion_tokens" not in payload

    def test_format_payload_streaming_enabled(self, streaming_model_endpoint):
        """Test stream flag is set from endpoint config."""
        endpoint = create_endpoint_with_mock_transport(
            ChatEndpoint, streaming_model_endpoint
        )

        turn = Turn(
            texts=[Text(contents=["Test streaming"])],
            model="test-model",
        )
        request_info = create_request_info(
            model_endpoint=streaming_model_endpoint, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["stream"] is True

    def test_format_payload_model_fallback(self, endpoint, model_endpoint):
        """Test that endpoint model is used when turn model is None."""
        turn = Turn(
            texts=[Text(contents=["Test"])],
            model=None,  # No model specified
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_format_payload_with_extra_params(self):
        """Test extra parameters are included in payload."""
        extra_params = [("temperature", 0.7), ("top_p", 0.9)]
        model_endpoint = create_model_endpoint(EndpointType.CHAT, extra=extra_params)
        endpoint = create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)

        turn = Turn(texts=[Text(contents=["Test"])], model="test-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["temperature"] == 0.7
        assert payload["top_p"] == 0.9

    def test_format_payload_empty_turn_contents(self, endpoint, model_endpoint):
        """Test handling of empty content in turns."""
        turn = Turn(
            texts=[Text(contents=[""])],  # Empty string
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        # Empty string should still be included in simple message
        assert "messages" in payload
        assert payload["messages"][0]["content"] == ""

    def test_format_payload_multiple_images(self, endpoint, model_endpoint):
        """Test multiple images are all included."""
        turn = Turn(
            texts=[Text(contents=["Describe these"])],
            images=[
                Image(contents=["data:image/png;base64,img1"]),
                Image(contents=["data:image/png;base64,img2"]),
                Image(contents=["data:image/png;base64,img3"]),
            ],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        message = payload["messages"][0]
        image_parts = [c for c in message["content"] if c.get("type") == "image_url"]
        assert len(image_parts) == 3

    def test_format_payload_filters_empty_multimodal_content(
        self, endpoint, model_endpoint
    ):
        """Test that empty strings in multi-modal content are filtered out."""
        turn = Turn(
            texts=[
                Text(contents=["Valid text", "", "Another valid"]),  # Has empty
            ],
            images=[
                Image(contents=["", "data:image/png;base64,img1"]),  # Has empty
            ],
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        message = payload["messages"][0]
        text_parts = [c for c in message["content"] if c.get("type") == "text"]
        image_parts = [c for c in message["content"] if c.get("type") == "image_url"]

        # Only non-empty content should be included
        assert len(text_parts) == 2  # "Valid text" and "Another valid"
        assert len(image_parts) == 1  # Only the valid image

    def test_format_payload_custom_role(self, endpoint, model_endpoint):
        """Test custom role is used."""
        turn = Turn(
            texts=[Text(contents=["I am an assistant"])],
            role="assistant",
            model="test-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        # For simple text, role is in the message
        assert payload.get("role") == "assistant" or (
            "messages" in payload and payload["messages"][0]["role"] == "assistant"
        )

    def test_format_payload_uses_request_info_turns(self, endpoint, model_endpoint):
        """Test that format_payload correctly uses all turns from request_info.turns."""
        turn1 = Turn(texts=[Text(contents=["First turn"])], model="model1")
        turn2 = Turn(texts=[Text(contents=["Second turn"])], model="model2")

        # Pass multiple turns - all should be included in the chat conversation
        request_info = create_request_info(
            model_endpoint=model_endpoint, turn_index=0, turns=[turn1, turn2]
        )

        payload = endpoint.format_payload(request_info)

        # Should include all turns in multi-turn conversation
        assert "First turn" in str(payload)
        assert "Second turn" in str(payload)
        # Model should be from the last turn
        assert payload["model"] == "model2"

    def test_format_payload_raw_messages_bypasses_create_messages(
        self, endpoint, model_endpoint
    ):
        """Test that raw_messages on turn bypasses _create_messages and is used directly."""
        raw_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[Turn(max_tokens=100, raw_messages=raw_messages)],
        )

        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == raw_messages
        assert payload["model"] == "test-model"
        assert payload["max_completion_tokens"] == 100

    def test_format_payload_raw_messages_without_max_tokens(
        self, endpoint, model_endpoint
    ):
        """Test that raw_messages works on a turn without max_tokens."""
        raw_messages = [
            {"role": "user", "content": "Hello"},
        ]
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[Turn(raw_messages=raw_messages)],
        )

        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == raw_messages
        assert payload["model"] == "test-model"
        assert "max_completion_tokens" not in payload
        assert "max_tokens" not in payload

    def test_format_payload_raw_messages_with_tools(self, endpoint, model_endpoint):
        """Test that tools are injected into payload when raw_tools is set."""
        raw_messages = [{"role": "user", "content": "What's the weather?"}]
        raw_tools = [
            {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
        ]
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[Turn(max_tokens=50, raw_messages=raw_messages, raw_tools=raw_tools)],
        )

        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == raw_messages
        assert payload["tools"] == raw_tools
        assert payload["max_completion_tokens"] == 50

    def test_format_payload_raw_messages_without_tools(self, endpoint, model_endpoint):
        """Test that no tools key exists when raw_tools is None."""
        raw_messages = [{"role": "user", "content": "Hello"}]
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[Turn(raw_messages=raw_messages)],
        )

        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == raw_messages
        assert "tools" not in payload

    def test_format_payload_raw_messages_with_extra_params(self):
        """Test that extra params still apply when using raw_messages."""
        extra_params = [("temperature", 0.7)]
        model_endpoint = create_model_endpoint(EndpointType.CHAT, extra=extra_params)
        endpoint = create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)

        raw_messages = [{"role": "user", "content": "Hello"}]
        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[Turn(max_tokens=50, raw_messages=raw_messages)],
        )

        payload = endpoint.format_payload(request_info)

        assert payload["messages"] == raw_messages
        assert payload["temperature"] == 0.7
        assert payload["max_completion_tokens"] == 50
