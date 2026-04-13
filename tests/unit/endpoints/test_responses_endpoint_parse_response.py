# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ResponsesEndpoint parse_response functionality."""

import pytest

from aiperf.common.models.record_models import ReasoningResponseData, TextResponseData
from aiperf.common.models.usage_models import Usage
from aiperf.endpoints.openai_responses import ResponsesEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
)


class TestResponsesEndpointParseResponse:
    """Tests for ResponsesEndpoint parse_response functionality."""

    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(EndpointType.RESPONSES)
        return create_endpoint_with_mock_transport(ResponsesEndpoint, model_endpoint)

    # --- Non-streaming (full response) ---

    def test_parse_response_full_text(self, endpoint):
        """Non-streaming response with text output."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Hello, how can I help?"}
                        ],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello, how can I help?"
        assert parsed.usage == Usage({"input_tokens": 10, "output_tokens": 5})

    def test_parse_response_full_reasoning(self, endpoint):
        """Non-streaming response with reasoning content."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "response",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "reasoning", "text": "Let me think..."},
                            {"type": "output_text", "text": "The answer is 42"},
                        ],
                    }
                ],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.content == "The answer is 42"
        assert parsed.data.reasoning == "Let me think..."

    def test_parse_response_full_usage_only(self, endpoint):
        """Non-streaming response with usage but empty output."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "response",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage == Usage({"input_tokens": 10, "output_tokens": 0})

    def test_parse_response_full_empty_output(self, endpoint):
        """Non-streaming response with no output and no usage returns None."""
        mock_response = create_mock_response(
            123456789,
            {"object": "response", "output": []},
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_full_output_text_fallback(self, endpoint):
        """Non-streaming response falls back to output_text convenience field."""
        mock_response = create_mock_response(
            123456789,
            {
                "object": "response",
                "output_text": "Fallback text",
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Fallback text"

    # --- Streaming events ---

    def test_parse_response_streaming_text_delta(self, endpoint):
        """Streaming text delta event."""
        mock_response = create_mock_response(
            123456789,
            {"type": "response.output_text.delta", "delta": "Hello"},
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello"

    def test_parse_response_streaming_reasoning_delta(self, endpoint):
        """Streaming reasoning delta event."""
        mock_response = create_mock_response(
            123456789,
            {"type": "response.reasoning_text.delta", "delta": "Thinking..."},
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, ReasoningResponseData)
        assert parsed.data.reasoning == "Thinking..."

    def test_parse_response_streaming_completed_with_usage(self, endpoint):
        """Streaming response.completed event extracts usage."""
        mock_response = create_mock_response(
            123456789,
            {
                "type": "response.completed",
                "response": {
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage == Usage({"input_tokens": 10, "output_tokens": 20})

    @pytest.mark.parametrize(
        "event_type",
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.done",
            "response.output_item.done",
        ],
    )
    def test_parse_response_non_content_events_return_none(self, endpoint, event_type):
        """Non-content streaming events return None."""
        mock_response = create_mock_response(
            123456789,
            {"type": event_type},
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_none_json(self, endpoint):
        """None json returns None."""
        mock_response = create_mock_response(123456789, json_data=None)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_streaming_empty_delta(self, endpoint):
        """Empty delta in text event returns None."""
        mock_response = create_mock_response(
            123456789,
            {"type": "response.output_text.delta", "delta": ""},
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_streaming_completed_no_usage(self, endpoint):
        """Completed event without usage returns None."""
        mock_response = create_mock_response(
            123456789,
            {
                "type": "response.completed",
                "response": {},
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None
