# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CompletionsEndpoint parse_response functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import InferenceServerResponse, TextResponseData
from aiperf.endpoints.openai_completions import CompletionsEndpoint
from aiperf.plugin.enums import EndpointType


class TestCompletionsEndpointParseResponse:
    """Tests for CompletionsEndpoint parse_response functionality."""

    @pytest.fixture
    def endpoint(self):
        """Create a CompletionsEndpoint instance for parsing tests."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="completion-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.COMPLETIONS,
                base_url="http://localhost:8000",
            ),
        )
        with patch("aiperf.plugin.plugins.get_class") as mock_transport:
            mock_transport.return_value = MagicMock()
            return CompletionsEndpoint(model_endpoint=model_endpoint)

    def test_parse_response_completion_object(self, endpoint):
        """Test parsing text_completion object type."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
            "choices": [{"text": "Once upon a time"}],
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Once upon a time"

    def test_parse_response_completion_object_alternate(self, endpoint):
        """Test parsing completion object type (alternate format)."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "completion",
            "choices": [{"text": "Generated text here"}],
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Generated text here"

    def test_parse_response_multiple_choices_uses_first(self, endpoint):
        """Test that only first choice is used when multiple exist."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
            "choices": [
                {"text": "First choice"},
                {"text": "Second choice"},
                {"text": "Third choice"},
            ],
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == "First choice"

    def test_parse_response_empty_text(self, endpoint):
        """Test parsing response with empty text."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
            "choices": [{"text": ""}],
        }

        parsed = endpoint.parse_response(mock_response)

        # Empty string is falsy, should return None
        assert parsed is None

    def test_parse_response_no_text_field(self, endpoint):
        """Test parsing response without text field."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
            "choices": [{}],
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_null_text(self, endpoint):
        """Test parsing response with null text."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
            "choices": [{"text": None}],
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_no_choices_key(self, endpoint):
        """Test parsing response without choices key."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_no_json(self, endpoint):
        """Test parsing when get_json returns None."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = None

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_multiline_text(self, endpoint):
        """Test parsing response with multiline text."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
            "choices": [{"text": "Line 1\nLine 2\nLine 3"}],
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == "Line 1\nLine 2\nLine 3"

    def test_parse_response_special_characters(self, endpoint):
        """Test parsing response with special characters."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
            "choices": [{"text": "Hello 👋 World! Special: @#$%^&*()"}],
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == "Hello 👋 World! Special: @#$%^&*()"

    def test_parse_response_streaming_chunks(self, endpoint):
        """Test parsing multiple streaming completion chunks."""
        chunks = [
            {"object": "text_completion", "choices": [{"text": "Once"}]},
            {"object": "text_completion", "choices": [{"text": " upon"}]},
            {"object": "text_completion", "choices": [{"text": " a time"}]},
        ]

        results = []
        for i, chunk_json in enumerate(chunks):
            mock_response = Mock(spec=InferenceServerResponse)
            mock_response.perf_ns = 100 + i
            mock_response.get_json.return_value = chunk_json
            parsed = endpoint.parse_response(mock_response)
            if parsed:
                results.append(parsed.data.text)

        assert len(results) == 3
        assert results == ["Once", " upon", " a time"]

    def test_parse_response_very_long_text(self, endpoint):
        """Test parsing response with very long text."""
        long_text = "word " * 10000
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "object": "text_completion",
            "choices": [{"text": long_text}],
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == long_text
