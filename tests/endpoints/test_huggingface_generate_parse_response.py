# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.enums import EndpointType
from aiperf.common.models.record_models import TextResponseData
from aiperf.common.protocols import InferenceServerResponse
from aiperf.endpoints.huggingface_generate import HuggingFaceGenerateEndpoint
from tests.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)


def create_mock_response(perf_ns: int, json_data):
    """Helper to create a mock InferenceServerResponse."""
    mock_response = Mock(spec=InferenceServerResponse)
    mock_response.perf_ns = perf_ns
    mock_response.get_json.return_value = json_data
    return mock_response


class TestHuggingFaceGenerateParseResponse:
    """Tests for HuggingFaceGenerateEndpoint parse_response functionality."""

    @pytest.fixture
    def endpoint(self):
        """Create a HuggingFaceGenerateEndpoint instance for parsing tests."""
        model_endpoint = create_model_endpoint(EndpointType.HUGGINGFACE_GENERATE)
        return create_endpoint_with_mock_transport(
            HuggingFaceGenerateEndpoint, model_endpoint
        )

    def test_parse_response_single_dict(self, endpoint):
        """Parses a normal dict JSON response with generated_text."""
        mock_response = create_mock_response(111, {"generated_text": "Hello world"})
        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 111
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "Hello world"

    def test_parse_response_list_of_dicts(self, endpoint):
        """Parses a list response and concatenates all generated_text entries."""
        mock_response = create_mock_response(
            222, [{"generated_text": "Hi"}, {"generated_text": " there!"}]
        )
        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 222
        assert parsed.data.text.strip() == "Hi there!"

    def test_parse_response_list_multiple_entries(self, endpoint):
        """Handles multiple dicts and concatenates text entries."""
        mock_response = create_mock_response(
            333,
            [
                {"generated_text": "Part1"},
                {"generated_text": " Part2"},
                {"generated_text": " End"},
            ],
        )
        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text.strip() == "Part1 Part2 End"

    def test_parse_response_empty_list(self, endpoint):
        """Empty list response returns None."""
        mock_response = create_mock_response(555, [])
        parsed = endpoint.parse_response(mock_response)
        assert parsed is None

    def test_parse_response_none(self, endpoint):
        """None or invalid response returns None."""
        mock_response = create_mock_response(666, None)
        parsed = endpoint.parse_response(mock_response)
        assert parsed is None

    @pytest.mark.parametrize(
        "text_value",
        [
            "Plain text response",
            "Symbols and punctuation: @#$%^&*!",
            "Multiline\nresponse\nfrom model",
            '{"json_like": "string"}',
        ],
    )
    def test_parse_response_content_variations(self, endpoint, text_value):
        """Handle various text output formats."""
        mock_response = create_mock_response(777, {"generated_text": text_value})
        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == text_value
