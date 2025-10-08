# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest import mock

import pytest

from aiperf.clients.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.clients.openai.openai_completions import OpenAICompletionRequestConverter
from aiperf.common.enums.endpoints_enums import EndpointType
from aiperf.common.enums.model_enums import ModelSelectionStrategy


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
        with mock.patch.object(converter, "debug") as debug_mock:
            payload = await converter.format_payload(model_endpoint, turns)
            print(f"Payload: {payload}")
        assert payload["prompt"] == ["Hello, world!"]
        assert payload["model"] == "test-model"
        assert payload["stream"] is True
        debug_mock.assert_called_once()
        assert "Formatted payload" in debug_mock.call_args[0][0]()
