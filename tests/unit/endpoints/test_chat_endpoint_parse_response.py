# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ChatEndpoint parse_response functionality."""

import pytest

from aiperf.common.models.record_models import ReasoningResponseData, TextResponseData
from aiperf.endpoints.openai_chat import ChatEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_config,
    create_endpoint_with_mock_transport,
    create_mock_response,
)


class TestChatEndpointParseResponse:
    """Tests for ChatEndpoint parse_response functionality."""

    @pytest.fixture
    def endpoint(self):
        """Create a ChatEndpoint instance for parsing tests."""
        cfg = create_config(EndpointType.CHAT)
        return create_endpoint_with_mock_transport(ChatEndpoint, cfg)

    def test_parse_response_chat_completion(self, endpoint):
        """Test parsing non-streaming chat completion response."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "chat.completion",
                "choices": [{"message": {"content": "Hello, how can I help?"}}],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello, how can I help?"

    def test_parse_response_chat_completion_chunk(self, endpoint):
        """Test parsing streaming chat completion chunk."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "Hello"}}],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello"

    def test_parse_response_with_reasoning_content(self, endpoint):
        """Test parsing response with reasoning_content (reasoning-capable models)."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "content": "The answer is 42",
                            "reasoning_content": "First, I analyzed the problem...",
                        }
                    }
                ],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.content == "The answer is 42"
        assert parsed.data.reasoning == "First, I analyzed the problem..."

    def test_parse_response_with_reasoning_field(self, endpoint):
        """Test parsing response with 'reasoning' field."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "chat.completion",
                "choices": [
                    {"message": {"content": "Answer", "reasoning": "Reasoning here"}}
                ],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.content == "Answer"
        assert parsed.data.reasoning == "Reasoning here"

    def test_parse_response_reasoning_priority(self, endpoint):
        """Test that reasoning_content takes priority over reasoning."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "chat.completion",
                "choices": [
                    {
                        "message": {
                            "content": "Answer",
                            "reasoning_content": "Should use this",
                            "reasoning": "Not this",
                        }
                    }
                ],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed.data.reasoning == "Should use this"

    def test_parse_response_only_reasoning_no_content(self, endpoint):
        """Test parsing when only reasoning is present (no content)."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "chat.completion",
                "choices": [{"message": {"reasoning": "Only reasoning"}}],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.content is None
        assert parsed.data.reasoning == "Only reasoning"

    @pytest.mark.parametrize(
        "json_data",
        [
            None,
            {"object": "chat.completion"},
            {"object": "chat.completion", "choices": [{"message": {}}]},
            {"object": "chat.completion", "choices": [{"message": {"content": None}}]},
            {"object": "chat.completion", "choices": [{"message": {"content": ""}}]},
            {"object": "chat.completion.chunk", "choices": [{"delta": {}}]},
        ],
    )
    def test_parse_response_returns_none(self, endpoint, json_data):
        """Test parsing responses that should return None."""
        mock_response = create_mock_response(123456789, json_data)
        parsed = endpoint.parse_response(mock_response)
        assert parsed is None

    def test_fast_parse_response_handles_standard_chunk_with_usage(self, endpoint):
        """Fast-path parser should handle the common streaming chunk shape."""
        parsed = endpoint._fast_parse_response(
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "Hello"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            123456789,
        )

        assert parsed is not endpoint._FAST_PARSE_FALLBACK
        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello"
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.completion_tokens == 20

    def test_fast_parse_response_falls_back_for_unexpected_shape(self, endpoint):
        """Fast-path parser should defer to the generic path for unusual responses."""
        parsed = endpoint._fast_parse_response(
            {"object": "unexpected", "choices": [{"message": {"content": "Hello"}}]},
            123456789,
        )

        assert parsed is endpoint._FAST_PARSE_FALLBACK

    def test_parse_response_streaming_multiple_chunks(self, endpoint):
        """Test parsing multiple streaming chunks."""
        chunks = [
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "Hello"}}],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": " world"}}],
            },
            {
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": "!"}}],
            },
        ]

        results = []
        for i, chunk_json in enumerate(chunks):
            mock_response = create_mock_response(123456789 + i, chunk_json)
            parsed = endpoint.parse_response(mock_response)
            if parsed:
                results.append(parsed.data.text)

        assert len(results) == 3
        assert results == ["Hello", " world", "!"]

    @pytest.mark.parametrize(
        "content_text",
        [
            "Line 1\nLine 2\nLine 3",
            "Hello 👋 World! 你好 🌍",
            '{"key": "value", "nested": {"data": [1, 2, 3]}}',
        ],
    )
    def test_parse_response_content_variations(self, endpoint, content_text):
        """Test parsing responses with various content types."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "chat.completion",
                "choices": [{"message": {"content": content_text}}],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == content_text
