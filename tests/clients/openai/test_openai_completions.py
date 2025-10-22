# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums.endpoints_enums import EndpointType
from aiperf.common.enums.model_enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.openai_completions import OpenAICompletionRequestConverter


class TestOpenAICompletionRequestConverter:
    """Test OpenAICompletionRequestConverter."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                custom_endpoint="/v1/chat/completions",
                api_key="test-api-key",
            ),
        )

    @pytest.mark.asyncio
    async def test_format_payload_basic(self, model_endpoint, sample_conversations):
        converter = OpenAICompletionRequestConverter()
        # Use the first turn from the sample_conversations fixture
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        payload = await converter.format_payload(model_endpoint, turns)
        print(f"Payload: {payload}")
        expected_payload = {
            "prompt": ["Hello, world!"],
            "model": "test-model",
            "stream": False,
        }
        assert payload == expected_payload

    @pytest.mark.asyncio
    async def test_format_payload_with_extra_options(
        self, model_endpoint, sample_conversations
    ):
        converter = OpenAICompletionRequestConverter()
        # Use the first turn from the sample_conversations fixture
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        turns[0].max_tokens = 50
        model_endpoint.endpoint.streaming = True
        model_endpoint.endpoint.extra = [("ignore_eos", True)]
        payload = await converter.format_payload(model_endpoint, turns)
        print(f"Payload: {payload}")
        expected_payload = {
            "prompt": ["Hello, world!"],
            "model": "test-model",
            "stream": True,
            "max_tokens": 50,
            "ignore_eos": True,
        }
        assert payload == expected_payload
