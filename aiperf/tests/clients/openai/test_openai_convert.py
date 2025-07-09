#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest

from aiperf.clients.client_interfaces import RequestConverterFactory
from aiperf.clients.model_endpoint_info import EndpointInfo, ModelEndpointInfo
from aiperf.clients.openai.openai_convert import (
    OpenAIChatCompletionRequestConverter,
    OpenAICompletionRequestConverter,
    OpenAIEmbeddingsRequestConverter,
    OpenAIResponsesRequestConverter,
)
from aiperf.common.dataset_models import Image, Text, Turn
from aiperf.common.enums import EndpointType


class TestOpenAIChatCompletionRequestConverter:
    """Test cases for OpenAIChatCompletionRequestConverter."""

    def test_converter_initialization(self):
        """Test converter can be initialized."""
        converter = OpenAIChatCompletionRequestConverter()
        assert converter is not None
        assert hasattr(converter, "logger")

    @pytest.mark.asyncio
    async def test_format_payload_basic(self, sample_model_endpoint_info, sample_turn):
        """Test basic payload formatting for chat completions."""
        converter = OpenAIChatCompletionRequestConverter()

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["model"] == "gpt-4"
        assert payload["stream"] is False
        assert "messages" in payload
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][0]["name"] == "text"
        assert payload["messages"][0]["content"] == "Hello, how are you?"

    @pytest.mark.asyncio
    async def test_format_payload_with_extra_params(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test payload formatting with extra parameters."""
        converter = OpenAIChatCompletionRequestConverter()

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        # Check that extra parameters are included
        assert payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_format_payload_streaming(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test payload formatting with streaming enabled."""
        converter = OpenAIChatCompletionRequestConverter()

        # Enable streaming
        sample_model_endpoint_info.endpoint.streaming = True

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["stream"] is True

    @pytest.mark.asyncio
    async def test_format_payload_multiple_text_content(
        self, sample_model_endpoint_info
    ):
        """Test payload formatting with multiple text content."""
        converter = OpenAIChatCompletionRequestConverter()

        # Create turn with multiple text entries
        turn = Turn(
            text=[
                Text(name="text1", role="user", content=["Hello", "How are you?"]),
                Text(name="text2", role="assistant", content=["I'm fine", "Thank you"]),
            ]
        )

        payload = await converter.format_payload(sample_model_endpoint_info, turn)

        assert len(payload["messages"]) == 4  # 2 content items per text entry
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["messages"][1]["content"] == "How are you?"
        assert payload["messages"][2]["content"] == "I'm fine"
        assert payload["messages"][3]["content"] == "Thank you"

    @pytest.mark.asyncio
    async def test_format_payload_empty_content_filtered(
        self, sample_model_endpoint_info
    ):
        """Test that empty content is filtered out."""
        converter = OpenAIChatCompletionRequestConverter()

        # Create turn with empty content
        turn = Turn(
            text=[Text(name="text", role="user", content=["Hello", "", "World"])]
        )

        payload = await converter.format_payload(sample_model_endpoint_info, turn)

        assert len(payload["messages"]) == 2  # Empty content should be filtered out
        assert payload["messages"][0]["content"] == "Hello"
        assert payload["messages"][1]["content"] == "World"

    @pytest.mark.asyncio
    async def test_format_payload_no_role_defaults_to_user(
        self, sample_model_endpoint_info
    ):
        """Test that missing role defaults to 'user'."""
        converter = OpenAIChatCompletionRequestConverter()

        turn = Turn(text=[Text(name="text", role=None, content=["Hello"])])

        payload = await converter.format_payload(sample_model_endpoint_info, turn)

        assert payload["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_format_payload_preserves_role(self, sample_model_endpoint_info):
        """Test that custom roles are preserved."""
        converter = OpenAIChatCompletionRequestConverter()

        turn = Turn(text=[Text(name="text", role="assistant", content=["Hello"])])

        payload = await converter.format_payload(sample_model_endpoint_info, turn)

        assert payload["messages"][0]["role"] == "assistant"

    def test_converter_factory_registration(self):
        """Test that converter is registered with factory."""

        converter = RequestConverterFactory.create_instance(
            EndpointType.OPENAI_CHAT_COMPLETIONS
        )
        assert isinstance(converter, OpenAIChatCompletionRequestConverter)


class TestOpenAICompletionRequestConverter:
    """Test cases for OpenAICompletionRequestConverter."""

    def test_converter_initialization(self):
        """Test converter can be initialized."""
        converter = OpenAICompletionRequestConverter()
        assert converter is not None
        assert hasattr(converter, "logger")

    @pytest.mark.asyncio
    async def test_format_payload_basic(self, sample_model_endpoint_info, sample_turn):
        """Test basic payload formatting for completions."""
        converter = OpenAICompletionRequestConverter()

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["model"] == "gpt-4"
        assert payload["stream"] is False
        assert payload["prompt"] == sample_turn.text
        assert payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_format_payload_streaming(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test payload formatting with streaming enabled."""
        converter = OpenAICompletionRequestConverter()

        # Enable streaming
        sample_model_endpoint_info.endpoint.streaming = True

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["stream"] is True

    @pytest.mark.asyncio
    async def test_format_payload_with_extra_params(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test payload formatting with extra parameters."""
        converter = OpenAICompletionRequestConverter()

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["temperature"] == 0.7

    def test_converter_factory_registration(self):
        """Test that converter is registered with factory."""

        converter = RequestConverterFactory.create_instance(
            EndpointType.OPENAI_COMPLETIONS
        )
        assert isinstance(converter, OpenAICompletionRequestConverter)


class TestOpenAIEmbeddingsRequestConverter:
    """Test cases for OpenAIEmbeddingsRequestConverter."""

    def test_converter_initialization(self):
        """Test converter can be initialized."""
        converter = OpenAIEmbeddingsRequestConverter()
        assert converter is not None
        assert hasattr(converter, "logger")

    @pytest.mark.asyncio
    async def test_format_payload_basic(self, sample_model_endpoint_info, sample_turn):
        """Test basic payload formatting for embeddings."""
        converter = OpenAIEmbeddingsRequestConverter()

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["model"] == "gpt-4"
        assert payload["input"] == sample_turn.text
        assert payload["dimensions"] == 1536
        assert payload["encoding_format"] == "float"
        assert payload["user"] == ""
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_format_payload_with_url_params(
        self, sample_model_list_info, sample_turn
    ):
        """Test payload formatting with URL parameters."""
        converter = OpenAIEmbeddingsRequestConverter()

        # Create endpoint with URL parameters
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_EMBEDDINGS,
            url_params={
                "dimensions": 512,
                "encoding_format": "base64",
                "user": "test-user",
            },
        )
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )

        payload = await converter.format_payload(model_endpoint_info, sample_turn)

        assert payload["dimensions"] == 512
        assert payload["encoding_format"] == "base64"
        assert payload["user"] == "test-user"

    @pytest.mark.asyncio
    async def test_format_payload_no_url_params(
        self, sample_model_list_info, sample_turn
    ):
        """Test payload formatting without URL parameters."""
        converter = OpenAIEmbeddingsRequestConverter()

        # Create endpoint without URL parameters
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_EMBEDDINGS, url_params=None
        )
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )

        payload = await converter.format_payload(model_endpoint_info, sample_turn)

        # Should use defaults
        assert payload["dimensions"] == 1536
        assert payload["encoding_format"] == "float"
        assert payload["user"] == ""

    @pytest.mark.asyncio
    async def test_format_payload_streaming(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test payload formatting with streaming enabled."""
        converter = OpenAIEmbeddingsRequestConverter()

        # Enable streaming
        sample_model_endpoint_info.endpoint.streaming = True

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["stream"] is True

    def test_converter_factory_registration(self):
        """Test that converter is registered with factory."""

        converter = RequestConverterFactory.create_instance(
            EndpointType.OPENAI_EMBEDDINGS
        )
        assert isinstance(converter, OpenAIEmbeddingsRequestConverter)


class TestOpenAIResponsesRequestConverter:
    """Test cases for OpenAIResponsesRequestConverter."""

    def test_converter_initialization(self):
        """Test converter can be initialized."""
        converter = OpenAIResponsesRequestConverter()
        assert converter is not None
        assert hasattr(converter, "logger")

    @pytest.mark.asyncio
    async def test_format_payload_basic(self, sample_model_endpoint_info, sample_turn):
        """Test basic payload formatting for responses."""
        converter = OpenAIResponsesRequestConverter()

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["model"] == "gpt-4"
        assert payload["input"] == sample_turn.text
        assert payload["max_output_tokens"] == 1000
        assert payload["stream"] is False

    @pytest.mark.asyncio
    async def test_format_payload_with_url_params(
        self, sample_model_list_info, sample_turn
    ):
        """Test payload formatting with URL parameters."""
        converter = OpenAIResponsesRequestConverter()

        # Create endpoint with URL parameters
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_RESPONSES, url_params={"max_output_tokens": 2000}
        )
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )

        payload = await converter.format_payload(model_endpoint_info, sample_turn)

        assert payload["max_output_tokens"] == 2000

    @pytest.mark.asyncio
    async def test_format_payload_no_url_params(
        self, sample_model_list_info, sample_turn
    ):
        """Test payload formatting without URL parameters."""
        converter = OpenAIResponsesRequestConverter()

        # Create endpoint without URL parameters
        endpoint_info = EndpointInfo(
            type=EndpointType.OPENAI_RESPONSES, url_params=None
        )
        model_endpoint_info = ModelEndpointInfo(
            models=sample_model_list_info, endpoint=endpoint_info
        )

        payload = await converter.format_payload(model_endpoint_info, sample_turn)

        # Should use default
        assert payload["max_output_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_format_payload_streaming(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test payload formatting with streaming enabled."""
        converter = OpenAIResponsesRequestConverter()

        # Enable streaming
        sample_model_endpoint_info.endpoint.streaming = True

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["stream"] is True

    @pytest.mark.asyncio
    async def test_format_payload_with_extra_params(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test payload formatting with extra parameters."""
        converter = OpenAIResponsesRequestConverter()

        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        assert payload["temperature"] == 0.7

    def test_converter_factory_registration(self):
        """Test that converter is registered with factory."""

        converter = RequestConverterFactory.create_instance(
            EndpointType.OPENAI_RESPONSES
        )
        assert isinstance(converter, OpenAIResponsesRequestConverter)


class TestConverterIntegration:
    """Integration tests for all OpenAI converters."""

    @pytest.mark.asyncio
    async def test_all_converters_factory_registration(self):
        """Test that all converters are properly registered with factory."""
        converters = [
            (
                EndpointType.OPENAI_CHAT_COMPLETIONS,
                OpenAIChatCompletionRequestConverter,
            ),
            (EndpointType.OPENAI_COMPLETIONS, OpenAICompletionRequestConverter),
            (EndpointType.OPENAI_EMBEDDINGS, OpenAIEmbeddingsRequestConverter),
            (EndpointType.OPENAI_RESPONSES, OpenAIResponsesRequestConverter),
        ]

        for endpoint_type, converter_class in converters:
            converter = RequestConverterFactory.create_instance(endpoint_type)
            assert isinstance(converter, converter_class)

    @pytest.mark.asyncio
    async def test_all_converters_basic_functionality(
        self, sample_model_endpoint_info, sample_turn
    ):
        """Test basic functionality of all converters."""
        converters = [
            (
                EndpointType.OPENAI_CHAT_COMPLETIONS,
                OpenAIChatCompletionRequestConverter,
            ),
            (EndpointType.OPENAI_COMPLETIONS, OpenAICompletionRequestConverter),
            (EndpointType.OPENAI_EMBEDDINGS, OpenAIEmbeddingsRequestConverter),
            (EndpointType.OPENAI_RESPONSES, OpenAIResponsesRequestConverter),
        ]

        for endpoint_type, converter_class in converters:
            # Update endpoint type to match converter
            sample_model_endpoint_info.endpoint.type = endpoint_type

            converter = converter_class()
            payload = await converter.format_payload(
                sample_model_endpoint_info, sample_turn
            )

            # All converters should include these basic fields
            assert "model" in payload
            assert "stream" in payload
            assert payload["model"] == "gpt-4"
            assert isinstance(payload["stream"], bool)

    @pytest.mark.asyncio
    async def test_converters_with_different_model_names(
        self, sample_model_list_info, sample_turn
    ):
        """Test converters with different model names."""
        models = ["gpt-4", "gpt-3.5-turbo", "text-embedding-ada-002"]

        for model_name in models:
            sample_model_list_info.models[0].name = model_name

            endpoint_info = EndpointInfo(type=EndpointType.OPENAI_CHAT_COMPLETIONS)
            model_endpoint_info = ModelEndpointInfo(
                models=sample_model_list_info, endpoint=endpoint_info
            )

            converter = OpenAIChatCompletionRequestConverter()
            payload = await converter.format_payload(model_endpoint_info, sample_turn)

            assert payload["model"] == model_name

    @pytest.mark.asyncio
    async def test_converters_with_multimodal_content(self, sample_model_endpoint_info):
        """Test converters with multimodal content."""
        # Create turn with text and image
        turn = Turn(
            text=[Text(name="text", role="user", content=["Describe this image"])],
            image=[Image(name="image", content=["https://example.com/image.jpg"])],
        )

        converter = OpenAIChatCompletionRequestConverter()
        payload = await converter.format_payload(sample_model_endpoint_info, turn)

        # Should only process text content for chat completions
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["content"] == "Describe this image"

    @pytest.mark.asyncio
    async def test_converters_debug_logging(
        self, sample_model_endpoint_info, sample_turn, caplog
    ):
        """Test that converters log debug information."""
        converter = OpenAIChatCompletionRequestConverter()

        # Enable debug logging
        import logging

        logging.getLogger().setLevel(logging.DEBUG)

        _ = await converter.format_payload(sample_model_endpoint_info, sample_turn)

        # Check that debug log was created
        assert "Formatted payload" in caplog.text

    @pytest.mark.asyncio
    async def test_converters_error_handling(self, sample_turn):
        """Test converters handle errors gracefully."""
        converter = OpenAIChatCompletionRequestConverter()

        # Create invalid model endpoint info
        invalid_endpoint = Mock()
        invalid_endpoint.primary_model_name = None
        invalid_endpoint.endpoint.streaming = False
        invalid_endpoint.endpoint.extra = None

        # Should not raise an exception
        await converter.format_payload(invalid_endpoint, sample_turn)

    @pytest.mark.asyncio
    async def test_converters_with_empty_turn(self, sample_model_endpoint_info):
        """Test converters with empty turn."""
        empty_turn = Turn()

        converter = OpenAIChatCompletionRequestConverter()
        payload = await converter.format_payload(sample_model_endpoint_info, empty_turn)

        # Should handle empty turn gracefully
        assert payload["model"] == "gpt-4"
        assert payload["messages"] == []

    @pytest.mark.parametrize(
        "endpoint_type,converter_class",
        [
            (
                EndpointType.OPENAI_CHAT_COMPLETIONS,
                OpenAIChatCompletionRequestConverter,
            ),
            (EndpointType.OPENAI_COMPLETIONS, OpenAICompletionRequestConverter),
            (EndpointType.OPENAI_EMBEDDINGS, OpenAIEmbeddingsRequestConverter),
            (EndpointType.OPENAI_RESPONSES, OpenAIResponsesRequestConverter),
        ],
    )
    @pytest.mark.asyncio
    async def test_converter_consistency(
        self, endpoint_type, converter_class, sample_model_endpoint_info, sample_turn
    ):
        """Test that all converters have consistent behavior."""
        # Update endpoint type
        sample_model_endpoint_info.endpoint.type = endpoint_type

        converter = converter_class()
        payload = await converter.format_payload(
            sample_model_endpoint_info, sample_turn
        )

        # Common assertions for all converters
        assert isinstance(payload, dict)
        assert "model" in payload
        assert "stream" in payload
        assert payload["model"] == "gpt-4"
        assert isinstance(payload["stream"], bool)
