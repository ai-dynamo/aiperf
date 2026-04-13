# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Audio, Image, Text, Turn
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class TestResponsesEndpoint:
    """Tests for ResponsesEndpoint format_payload."""

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
        # system_message should NOT appear as an input item
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

    def test_format_payload_stream_options_auto_set(self):
        """stream_options.include_usage auto-set when use_server_token_count=True."""
        from aiperf.common.enums import ModelSelectionStrategy
        from aiperf.common.models.model_endpoint_info import (
            EndpointInfo,
            ModelEndpointInfo,
            ModelInfo,
            ModelListInfo,
        )

        me = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.RESPONSES,
                base_url="http://localhost:8000",
                streaming=True,
                extra=[],
                use_server_token_count=True,
            ),
        )
        ep = create_endpoint_with_mock_transport(ResponsesEndpoint, me)
        turn = Turn(texts=[Text(contents=["Test"])], model="test-model")
        request_info = create_request_info(model_endpoint=me, turns=[turn])

        payload = ep.format_payload(request_info)

        assert payload["stream_options"] == {"include_usage": True}

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
