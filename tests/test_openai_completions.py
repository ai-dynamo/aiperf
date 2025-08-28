# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.clients.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.clients.openai.openai_chat import OpenAIChatCompletionRequestConverter
from aiperf.clients.openai.openai_completions import OpenAICompletionRequestConverter
from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models.dataset_models import Text, Turn


class TestOpenAIRequestConverters:
    """Test cases for TestOpenAIRequestConverters format_payload method."""

    @pytest.fixture
    def converter(self):
        """Create a converter instance."""
        return OpenAICompletionRequestConverter()

    @pytest.fixture
    def chat_converter(self):
        """Create a chat converter instance."""
        return OpenAIChatCompletionRequestConverter()

    @pytest.fixture
    def basic_model_endpoint(self):
        """Create a basic ModelEndpointInfo for testing."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="gpt-3.5-turbo")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.OPENAI_COMPLETIONS,
                base_url="http://localhost:8000/v1",
                streaming=False,
            ),
        )

    @pytest.fixture
    def basic_turn(self):
        """Create a basic Turn for testing."""
        return Turn(texts=[Text(name="prompt", contents=["Hello, world!"])])

    @pytest.mark.asyncio
    async def test_basic_payload_format(
        self, converter, basic_model_endpoint, basic_turn
    ):
        """Test that the basic payload format is correct."""
        result = await converter.format_payload(basic_model_endpoint, basic_turn)

        # Check required fields are present
        assert "prompt" in result
        assert "model" in result
        assert "stream" in result

        # Check values
        assert result["prompt"] == ["Hello, world!"]
        assert result["model"] == "gpt-3.5-turbo"
        assert result["stream"] is False

        # Check optional fields are not present when not set
        assert "max_tokens" not in result

    @pytest.mark.asyncio
    async def test_payload_with_max_tokens(self, converter, basic_model_endpoint):
        """Test that max_tokens is included when specified."""
        turn = Turn(
            texts=[Text(name="prompt", contents=["Generate text"])], max_tokens=100
        )

        result = await converter.format_payload(basic_model_endpoint, turn)

        assert result["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_chat_payload_with_max_tokens(
        self, chat_converter, basic_model_endpoint
    ):
        """Test that max_tokens renamed to max_completion_tokens in chat payload.
        Also ensures that max_tokens is not included in the payload."""
        turn = Turn(
            texts=[Text(name="prompt", contents=["Generate text"])], max_tokens=100
        )

        result = await chat_converter.format_payload(basic_model_endpoint, turn)
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 100

    @pytest.mark.asyncio
    async def test_payload_without_max_tokens(
        self, converter, basic_model_endpoint, basic_turn
    ):
        """Test that max_tokens is not included when not specified."""
        result = await converter.format_payload(basic_model_endpoint, basic_turn)

        assert "max_tokens" not in result
