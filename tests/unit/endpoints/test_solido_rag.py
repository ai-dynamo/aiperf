# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Text, Turn
from aiperf.common.models.record_models import (
    TextResponseData,
)
from aiperf.endpoints.solido_rag import SolidoEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType
from aiperf.plugin.schema.schemas import EndpointMetadata
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestSolidoEndpointMetadata:
    """Tests for SolidoEndpoint metadata."""

    def test_metadata_returns_correct_values(self):
        """Test that metadata returns expected configuration."""
        metadata = plugins.get_endpoint_metadata(EndpointType.SOLIDO_RAG)
        assert isinstance(metadata, EndpointMetadata)

        assert metadata.endpoint_path == "/rag/api/prompt"
        assert metadata.supports_streaming is True
        assert metadata.produces_tokens is True
        assert metadata.tokenizes_input is True
        assert metadata.metrics_title == "SOLIDO RAG Metrics"


class TestSolidoEndpointFormatPayload:
    """Tests for SolidoEndpoint format_payload functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for SOLIDO."""
        return create_model_endpoint(EndpointType.SOLIDO_RAG, model_name="solido-model")

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a SolidoEndpoint instance."""
        return create_endpoint_with_mock_transport(SolidoEndpoint, model_endpoint)

    def test_format_payload_basic(self, endpoint, model_endpoint):
        """Test basic payload formatting with single query."""
        request_info = create_request_info(
            model_endpoint,
            texts=["What is Solido SDE?"],
            model="solido-model",
        )

        payload = endpoint.format_payload(request_info)

        assert payload["query"] == ["What is Solido SDE?"]
        assert payload["inference_model"] == "solido-model"
        assert payload["filters"] == {"family": "Solido", "tool": "SDE"}

    def test_format_payload_multiple_text_contents(self, endpoint, model_endpoint):
        """Test payload with multiple text contents in a turn."""
        turn = Turn(
            texts=[
                Text(contents=["First query", "Second query"]),
                Text(contents=["Third query"]),
            ],
            model="solido-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["query"] == ["First query", "Second query", "Third query"]

    def test_format_payload_filters_empty_content(self, endpoint, model_endpoint):
        """Test that empty strings are filtered from query."""
        turn = Turn(
            texts=[Text(contents=["Valid query", "", "Another query"])],
            model="solido-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["query"] == ["Valid query", "Another query"]

    @pytest.mark.parametrize(
        "turn_model,expected_model",
        [
            ("custom-model", "custom-model"),
            (None, "solido-model"),
        ],
    )
    def test_format_payload_model_selection(
        self, endpoint, model_endpoint, turn_model, expected_model
    ):
        """Test model selection: turn model takes precedence, fallback to endpoint model."""
        turn = Turn(
            texts=[Text(contents=["Test query"])],
            model=turn_model,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inference_model"] == expected_model

    def test_format_payload_with_extra_params(self):
        """Test that extra parameters are merged into payload."""
        extra_params = [
            ("temperature", 0.7),
            ("max_new_tokens", 200),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.SOLIDO_RAG,
            model_name="solido-model",
            extra=extra_params,
        )
        endpoint = create_endpoint_with_mock_transport(SolidoEndpoint, model_endpoint)

        request_info = create_request_info(
            model_endpoint,
            texts=["Test query"],
            model="solido-model",
        )

        payload = endpoint.format_payload(request_info)

        assert payload["temperature"] == 0.7
        assert payload["max_new_tokens"] == 200

    @pytest.mark.parametrize(
        "custom_filters",
        [
            {"family": "CustomFamily", "tool": "CustomTool", "version": "1.0"},
            {"custom_key": "custom_value"},
        ],
    )
    def test_format_payload_custom_filters_override_defaults(self, custom_filters):
        """Test that custom filters completely replace default filters."""
        extra_params = [("filters", custom_filters)]
        model_endpoint = create_model_endpoint(
            EndpointType.SOLIDO_RAG,
            model_name="solido-model",
            extra=extra_params,
        )
        endpoint = create_endpoint_with_mock_transport(SolidoEndpoint, model_endpoint)

        request_info = create_request_info(
            model_endpoint,
            texts=["Test query"],
            model="solido-model",
        )

        payload = endpoint.format_payload(request_info)

        assert payload["filters"] == custom_filters

    def test_format_payload_no_turns_raises_error(self, endpoint, model_endpoint):
        """Test that missing turns raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(
            ValueError, match="SOLIDO endpoint requires at least one turn"
        ):
            endpoint.format_payload(request_info)

    def test_format_payload_uses_last_turn(self, endpoint, model_endpoint):
        """Test that only the last turn is used for SOLIDO payload."""
        turn1 = Turn(texts=[Text(contents=["First"])], model="model1")
        turn2 = Turn(texts=[Text(contents=["Second"])], model="model2")

        request_info = create_request_info(
            model_endpoint=model_endpoint,
            turns=[turn1, turn2],
        )

        payload = endpoint.format_payload(request_info)

        assert payload["query"] == ["Second"]
        assert payload["inference_model"] == "model2"


class TestSolidoEndpointParseResponse:
    """Tests for SolidoEndpoint parse_response functionality."""

    @pytest.fixture
    def endpoint(self):
        """Create a SolidoEndpoint instance for parsing tests."""
        model_endpoint = create_model_endpoint(EndpointType.SOLIDO_RAG)
        return create_endpoint_with_mock_transport(SolidoEndpoint, model_endpoint)

    def test_parse_response_with_content(self, endpoint):
        """Test parsing valid SOLIDO response with content."""
        mock_response = create_mock_response(
            123456789,
            {"content": "This is the RAG response with retrieved context."},
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, TextResponseData)
        assert parsed.data.text == "This is the RAG response with retrieved context."

    def test_parse_response_with_sources_dictionary(self, endpoint):
        """Test parsing SOLIDO response with sources in dictionary format."""
        mock_response = create_mock_response(
            123456789,
            {
                "content": "Response with sources",
                "sources": {
                    "file:///docs/test.pdf": {
                        "document_title": "Test Document",
                        "pages": {"1": "Page 1 content", "2": "Page 2 content"},
                        "sections": {},
                        "catalog_id": 42,
                    }
                },
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == "Response with sources"
        assert parsed.sources is not None

    def test_parse_response_with_sources_list(self, endpoint):
        """Test parsing SOLIDO response with sources in list format."""
        mock_response = create_mock_response(
            123456789,
            {
                "content": "Response with list sources",
                "sources": [
                    {"id": 1, "document": "doc1.pdf"},
                    {"id": 2, "document": "doc2.pdf"},
                ],
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == "Response with list sources"
        assert parsed.sources is not None

    def test_parse_response_with_additional_fields(self, endpoint):
        """Test parsing response with extra fields (should ignore them)."""
        mock_response = create_mock_response(
            123456789,
            {
                "content": "Response text",
                "metadata": {"source": "doc1"},
                "score": 0.95,
            },
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == "Response text"

    @pytest.mark.parametrize(
        "json_data,description",
        [
            ({"content": ""}, "empty content"),
            ({"message": "No content field"}, "missing content field"),
            ({"content": None}, "None content"),
            (None, "no JSON"),
            ({}, "empty JSON object"),
        ],
    )
    def test_parse_response_returns_none(self, endpoint, json_data, description):
        """Test parsing invalid responses returns None."""
        mock_response = create_mock_response(123456789, json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    @pytest.mark.parametrize(
        "content",
        [
            "Simple response",
            "Multi-line\nresponse\nwith\nnewlines",
            "Response with special chars: @#$%^&*()",
            "🚀 Unicode content works too! 你好",
        ],
    )
    def test_parse_response_various_content_types(self, endpoint, content):
        """Test parsing various content string formats."""
        mock_response = create_mock_response(123456789, {"content": content})

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.text == content


class TestSolidoEndpointIntegration:
    """Integration tests for SOLIDO endpoint end-to-end behavior."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for SOLIDO."""
        return create_model_endpoint(
            EndpointType.SOLIDO_RAG,
            model_name="solido-model",
            base_url="http://localhost:8080",
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a SolidoEndpoint instance."""
        return create_endpoint_with_mock_transport(SolidoEndpoint, model_endpoint)

    def test_full_request_response_cycle(self, endpoint, model_endpoint):
        """Test full cycle of formatting request and parsing response."""
        request_info = create_request_info(
            model_endpoint,
            texts=["How do I use Solido SDE?"],
            model="solido-model",
        )

        payload = endpoint.format_payload(request_info)
        assert payload["query"] == ["How do I use Solido SDE?"]

        mock_response = create_mock_response(
            123456789,
            {"content": "Solido SDE is used for..."},
        )

        parsed = endpoint.parse_response(mock_response)
        assert parsed is not None
        assert parsed.data.text == "Solido SDE is used for..."

    @pytest.mark.parametrize(
        "text,expected_result",
        [
            ("Valid text", "Valid text"),
            ("", None),
            (None, None),
        ],
    )
    def test_make_text_response_data(self, text, expected_result):
        """Test helper method for creating text response data."""
        result = SolidoEndpoint.make_text_response_data(text)

        if expected_result is None:
            assert result is None
        else:
            assert result is not None
            assert isinstance(result, TextResponseData)
            assert result.text == expected_result
