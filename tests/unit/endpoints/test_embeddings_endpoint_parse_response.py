# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EmbeddingsEndpoint parse_response functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import (
    EmbeddingResponseData,
    InferenceServerResponse,
)
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.plugin.enums import EndpointType


class TestEmbeddingsEndpointParseResponse:
    """Tests for EmbeddingsEndpoint parse_response functionality."""

    @pytest.fixture
    def endpoint(self):
        """Create an EmbeddingsEndpoint instance for parsing tests."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="embeddings-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.EMBEDDINGS,
                base_url="http://localhost:8000",
            ),
        )
        with patch("aiperf.plugin.plugins.get_class") as mock_transport:
            mock_transport.return_value = MagicMock()
            return EmbeddingsEndpoint(model_endpoint=model_endpoint)

    def test_parse_response_single_embedding(self, endpoint):
        """Test parsing response with single embedding."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "index": 0,
                }
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, EmbeddingResponseData)
        assert len(parsed.data.embeddings) == 1
        assert parsed.data.embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_parse_response_multiple_embeddings(self, endpoint):
        """Test parsing response with multiple embeddings."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1},
                {"object": "embedding", "embedding": [0.7, 0.8, 0.9], "index": 2},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings) == 3
        assert parsed.data.embeddings[0] == [0.1, 0.2, 0.3]
        assert parsed.data.embeddings[1] == [0.4, 0.5, 0.6]
        assert parsed.data.embeddings[2] == [0.7, 0.8, 0.9]

    def test_parse_response_large_dimensional_embedding(self, endpoint):
        """Test parsing high-dimensional embedding vectors."""
        embedding_vector = [
            float(i) / 1000 for i in range(1536)
        ]  # 1536-dim like OpenAI
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [{"object": "embedding", "embedding": embedding_vector, "index": 0}]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings[0]) == 1536

    def test_parse_response_empty_data_array(self, endpoint):
        """Test parsing response with empty data array."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {"data": []}

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_no_data_key(self, endpoint):
        """Test parsing response without data key."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {}

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_missing_embedding_field(self, endpoint):
        """Test parsing when embedding field is missing."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [{"object": "embedding", "index": 0}]
        }

        parsed = endpoint.parse_response(mock_response)

        # No valid embeddings, should return None
        assert parsed is None

    def test_parse_response_null_embedding(self, endpoint):
        """Test parsing when embedding is None."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [{"object": "embedding", "embedding": None, "index": 0}]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_filters_null_embeddings(self, endpoint):
        """Test that None embeddings are filtered out."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
                {"object": "embedding", "embedding": None, "index": 1},
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 2},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings) == 2
        assert parsed.data.embeddings[0] == [0.1, 0.2]
        assert parsed.data.embeddings[1] == [0.3, 0.4]

    def test_parse_response_invalid_object_type_raises(self, endpoint):
        """Test that invalid object type raises ValueError."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {"object": "embedding", "embedding": [0.1, 0.2]},
                {"object": "wrong_type", "embedding": [0.3, 0.4]},
            ]
        }

        with pytest.raises(ValueError, match="invalid list"):
            endpoint.parse_response(mock_response)

    def test_parse_response_mixed_invalid_objects_raises(self, endpoint):
        """Test that mixed valid/invalid objects raises ValueError."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {"object": "embedding", "embedding": [0.1]},
                "not a dict",  # Invalid item
            ]
        }

        with pytest.raises(ValueError, match="invalid list"):
            endpoint.parse_response(mock_response)

    def test_parse_response_no_json(self, endpoint):
        """Test parsing when get_json returns None."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = None

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_zero_dimensional_embedding(self, endpoint):
        """Test parsing embedding with zero dimensions (edge case)."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [{"object": "embedding", "embedding": [], "index": 0}]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings) == 1
        assert parsed.data.embeddings[0] == []

    def test_parse_response_negative_values(self, endpoint):
        """Test parsing embeddings with negative values."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": [-0.5, -0.3, 0.2, -0.1, 0.4],
                    "index": 0,
                }
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.embeddings[0] == [-0.5, -0.3, 0.2, -0.1, 0.4]

    def test_parse_response_preserves_order(self, endpoint):
        """Test that embeddings order is preserved."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {"object": "embedding", "embedding": [1.0], "index": 2},
                {"object": "embedding", "embedding": [2.0], "index": 0},
                {"object": "embedding", "embedding": [3.0], "index": 1},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        # Order should match data array order, not index field
        assert parsed.data.embeddings[0] == [1.0]
        assert parsed.data.embeddings[1] == [2.0]
        assert parsed.data.embeddings[2] == [3.0]

    def test_parse_response_extra_fields_ignored(self, endpoint):
        """Test that extra fields in response are ignored."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2],
                    "index": 0,
                    "extra_field": "ignored",
                    "another_field": 123,
                }
            ],
            "model": "text-embedding-ada-002",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.embeddings[0] == [0.1, 0.2]
