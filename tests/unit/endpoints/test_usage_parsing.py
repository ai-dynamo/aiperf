# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consolidated tests for usage field parsing across endpoints."""

import pytest

from aiperf.common.models.record_models import ReasoningResponseData
from aiperf.common.models.usage_models import Usage
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
)


@pytest.mark.parametrize(
    "endpoint_type,endpoint_class,object_type,data_key",
    [
        (EndpointType.CHAT, ChatEndpoint, "chat.completion", "message"),
        (EndpointType.CHAT, ChatEndpoint, "chat.completion.chunk", "delta"),
        (EndpointType.COMPLETIONS, CompletionsEndpoint, "completion", None),
    ],
)
class TestUsageParsing:
    """Parameterized tests for usage parsing across endpoints."""

    @pytest.fixture
    def endpoint(self, endpoint_type, endpoint_class):
        """Create endpoint instance."""
        model_endpoint = create_model_endpoint(endpoint_type)
        return create_endpoint_with_mock_transport(endpoint_class, model_endpoint)

    def test_parse_with_standard_usage(
        self, endpoint, object_type, data_key, endpoint_class
    ):
        """Test parsing response with standard usage fields."""
        if endpoint_class == ChatEndpoint:
            content_data = {data_key: {"content": "Test response"}}
        else:
            content_data = {"text": "Test response"}

        mock_response = create_mock_response(
            12345,
            {
                "object": object_type,
                "choices": [content_data],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.completion_tokens == 5
        assert parsed.usage.total_tokens == 15

    def test_parse_without_usage(self, endpoint, object_type, data_key, endpoint_class):
        """Test parsing response without usage field."""
        if endpoint_class == ChatEndpoint:
            content_data = {data_key: {"content": "Test"}}
        else:
            content_data = {"text": "Test"}

        mock_response = create_mock_response(
            12345, {"object": object_type, "choices": [content_data]}
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is None

    def test_parse_with_empty_usage(
        self, endpoint, object_type, data_key, endpoint_class
    ):
        """Test parsing response with empty usage dict."""
        if endpoint_class == ChatEndpoint:
            content_data = {data_key: {"content": "Test"}}
        else:
            content_data = {"text": "Test"}

        mock_response = create_mock_response(
            12345, {"object": object_type, "choices": [content_data], "usage": {}}
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is None  # Empty dict treated as None


class TestChatEndpointUsageSpecific:
    """Chat-specific usage parsing tests."""

    @pytest.fixture
    def endpoint(self):
        """Create a ChatEndpoint instance."""
        model_endpoint = create_model_endpoint(EndpointType.CHAT)
        return create_endpoint_with_mock_transport(ChatEndpoint, model_endpoint)

    def test_parse_with_nested_reasoning_tokens(self, endpoint):
        """Test parsing response with nested reasoning tokens (o1/o3 models)."""
        mock_response = create_mock_response(
            12345,
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Thinking...",
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 60,
                    "total_tokens": 80,
                    "completion_tokens_details": {"reasoning_tokens": 50},
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.usage.root["completion_tokens_details"]["reasoning_tokens"] == 50
        assert parsed.usage.reasoning_tokens == 50

    def test_parse_with_modern_naming(self, endpoint):
        """Test parsing with input_tokens/output_tokens naming."""
        mock_response = create_mock_response(
            12345,
            {
                "object": "chat.completion",
                "choices": [{"message": {"content": "Response"}}],
                "usage": {
                    "input_tokens": 25,
                    "output_tokens": 15,
                    "total_tokens": 40,
                    "output_tokens_details": {"reasoning_tokens": 5},
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage.root["input_tokens"] == 25
        assert parsed.usage.root["output_tokens"] == 15
        assert parsed.usage.reasoning_tokens == 5

    @pytest.mark.parametrize(
        "usage_data,expected_prompt,expected_completion,expected_total",
        [
            ({"prompt_tokens": 10}, 10, None, None),
            ({"completion_tokens": 5}, None, 5, None),
            ({"prompt_tokens": 10, "total_tokens": 10}, 10, None, 10),
            (
                {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300},
                100,
                200,
                300,
            ),
        ],
    )
    def test_parse_partial_usage(
        self,
        endpoint,
        usage_data,
        expected_prompt,
        expected_completion,
        expected_total,
    ):
        """Test parsing responses with partial usage data."""
        mock_response = create_mock_response(
            12345,
            {
                "object": "chat.completion",
                "choices": [{"message": {"content": "Test"}}],
                "usage": usage_data,
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is not None
        assert parsed.usage.root.get("prompt_tokens") == expected_prompt
        assert parsed.usage.root.get("completion_tokens") == expected_completion
        assert parsed.usage.root.get("total_tokens") == expected_total


class TestUsageModelProperties:
    """Test Usage model helper properties for various provider formats."""

    @pytest.mark.parametrize(
        "usage_data,expected",
        [
            # OpenAI standard format
            (
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "reasoning_tokens": None,
                },
            ),
            # Anthropic naming (input/output tokens)
            (
                {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
                {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                    "reasoning_tokens": None,
                },
            ),
            # OpenAI with reasoning tokens
            (
                {
                    "prompt_tokens": 20,
                    "completion_tokens": 60,
                    "total_tokens": 80,
                    "completion_tokens_details": {
                        "reasoning_tokens": 50,
                    },
                },
                {
                    "prompt_tokens": 20,
                    "completion_tokens": 60,
                    "reasoning_tokens": 50,
                    "total_tokens": 80,
                },
            ),
        ],
    )
    def test_provider_specific_fields(self, usage_data, expected):
        """Test extraction of provider-specific usage fields."""
        usage = Usage(usage_data)

        assert usage.prompt_tokens == expected["prompt_tokens"]
        assert usage.completion_tokens == expected["completion_tokens"]
        assert usage.total_tokens == expected["total_tokens"]
        assert usage.reasoning_tokens == expected["reasoning_tokens"]

    @pytest.mark.parametrize(
        "usage_data,missing_fields",
        [
            # No special fields at all
            (
                {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                ["reasoning_tokens"],
            ),
        ],
    )
    def test_missing_fields_return_none(self, usage_data, missing_fields):
        """Test that missing optional fields return None."""
        usage = Usage(usage_data)

        for field in missing_fields:
            assert getattr(usage, field) is None
