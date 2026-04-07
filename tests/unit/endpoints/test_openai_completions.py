# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import create_request_info


class TestCompletionsEndpoint:
    """Test CompletionsEndpoint."""

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

    def test_format_payload_basic(self, model_endpoint, sample_conversations):
        endpoint = CompletionsEndpoint(model_endpoint)
        # Use the first turn from the sample_conversations fixture
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "prompt": ["Hello, world!"],
            "model": "test-model",
            "stream": False,
        }
        assert payload == expected_payload

    def test_format_payload_with_extra_options(
        self, model_endpoint, sample_conversations
    ):
        endpoint = CompletionsEndpoint(model_endpoint)
        # Use the first turn from the sample_conversations fixture
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        turns[0].max_tokens = 50
        model_endpoint.endpoint.streaming = True
        model_endpoint.endpoint.extra = [("ignore_eos", True)]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        expected_payload = {
            "prompt": ["Hello, world!"],
            "model": "test-model",
            "stream": True,
            "max_tokens": 50,
            "ignore_eos": True,
            "stream_options": {"include_usage": True},
        }
        assert payload == expected_payload

    def test_format_payload_no_stream_options_when_stream_usage_false(
        self, model_endpoint, sample_conversations
    ):
        """Verify stream_options is not added when stream_usage=False even with streaming=True."""
        endpoint = CompletionsEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        model_endpoint.endpoint.streaming = True
        model_endpoint.endpoint.stream_usage = False
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)
        payload = endpoint.format_payload(request_info)
        assert "stream_options" not in payload

    @pytest.mark.parametrize(
        "streaming,user_extra,expected_stream_options",
        [
            # Auto-add when streaming
            (True, None, {"include_usage": True}),
            # Don't add when not streaming
            (False, None, None),
            # Preserve user's include_usage=False
            (True, [("stream_options", {"include_usage": False})], {"include_usage": False}),
            # Merge with user's other options
            (True, [("stream_options", {"continuous_updates": True})], {"continuous_updates": True, "include_usage": True}),
        ],
    )  # fmt: skip
    def test_stream_options_auto_configuration(
        self,
        model_endpoint,
        sample_conversations,
        streaming,
        user_extra,
        expected_stream_options,
    ):
        """Verify stream_options.include_usage is automatically configured for all streaming requests."""
        endpoint = CompletionsEndpoint(model_endpoint)
        turn = sample_conversations["session_1"].turns[0]
        turns = [turn]
        model_endpoint.endpoint.streaming = streaming
        if user_extra:
            model_endpoint.endpoint.extra = user_extra
        request_info = create_request_info(turns=turns, model_endpoint=model_endpoint)
        payload = endpoint.format_payload(request_info)

        if expected_stream_options is None:
            assert "stream_options" not in payload
        else:
            assert "stream_options" in payload
            assert payload["stream_options"] == expected_stream_options
