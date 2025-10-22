# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RankingsEndpoint parse_response functionality."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RankingsResponseData
from aiperf.common.protocols import InferenceServerResponse
from aiperf.endpoints.nim_rankings import RankingsEndpoint


class TestRankingsEndpointParseResponse:
    """Tests for RankingsEndpoint parse_response functionality."""

    @pytest.fixture
    def endpoint(self):
        """Create a RankingsEndpoint instance for parsing tests."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="rankings-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.RANKINGS,
                base_url="http://localhost:8000",
            ),
        )
        with patch(
            "aiperf.common.factories.TransportFactory.create_instance"
        ) as mock_transport:
            mock_transport.return_value = MagicMock()
            return RankingsEndpoint(model_endpoint=model_endpoint)

    def test_parse_response_basic_rankings(self, endpoint):
        """Test parsing basic rankings response."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [
                {"index": 0, "score": 0.95},
                {"index": 1, "score": 0.87},
                {"index": 2, "score": 0.72},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, RankingsResponseData)
        assert len(parsed.data.rankings) == 3
        assert parsed.data.rankings[0] == {"index": 0, "score": 0.95}
        assert parsed.data.rankings[1] == {"index": 1, "score": 0.87}
        assert parsed.data.rankings[2] == {"index": 2, "score": 0.72}

    def test_parse_response_single_ranking(self, endpoint):
        """Test parsing response with single ranking."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [{"index": 0, "score": 0.99}]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.rankings) == 1
        assert parsed.data.rankings[0]["score"] == 0.99

    def test_parse_response_many_rankings(self, endpoint):
        """Test parsing response with many rankings."""
        rankings = [{"index": i, "score": 1.0 - (i * 0.01)} for i in range(100)]
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {"rankings": rankings}

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.rankings) == 100
        assert parsed.data.rankings[0]["score"] == pytest.approx(1.0)
        assert parsed.data.rankings[99]["score"] == pytest.approx(0.01, abs=1e-9)

    def test_parse_response_empty_rankings(self, endpoint):
        """Test parsing response with empty rankings array."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {"rankings": []}

        parsed = endpoint.parse_response(mock_response)

        # Empty list is falsy in Python, so implementation returns None
        assert parsed is None

    def test_parse_response_no_rankings_key(self, endpoint):
        """Test parsing response without rankings key."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {}

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_null_rankings(self, endpoint):
        """Test parsing response with null rankings."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {"rankings": None}

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_rankings_with_extra_fields(self, endpoint):
        """Test that extra fields in rankings are preserved."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [
                {
                    "index": 0,
                    "score": 0.95,
                    "text": "Passage 1",
                    "metadata": {"source": "doc1"},
                },
                {"index": 1, "score": 0.87, "text": "Passage 2"},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.rankings[0]["text"] == "Passage 1"
        assert parsed.data.rankings[0]["metadata"] == {"source": "doc1"}
        assert parsed.data.rankings[1]["text"] == "Passage 2"

    def test_parse_response_negative_scores(self, endpoint):
        """Test parsing rankings with negative scores."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [
                {"index": 0, "score": -0.5},
                {"index": 1, "score": -0.8},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.rankings[0]["score"] == -0.5
        assert parsed.data.rankings[1]["score"] == -0.8

    def test_parse_response_scores_above_one(self, endpoint):
        """Test parsing rankings with scores above 1.0."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [
                {"index": 0, "score": 10.5},
                {"index": 1, "score": 7.3},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.rankings[0]["score"] == 10.5
        assert parsed.data.rankings[1]["score"] == 7.3

    def test_parse_response_unsorted_rankings(self, endpoint):
        """Test that rankings order is preserved as returned."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [
                {"index": 2, "score": 0.5},
                {"index": 0, "score": 0.9},
                {"index": 1, "score": 0.7},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        # Order should be preserved as is
        assert parsed.data.rankings[0]["index"] == 2
        assert parsed.data.rankings[1]["index"] == 0
        assert parsed.data.rankings[2]["index"] == 1

    def test_parse_response_no_json(self, endpoint):
        """Test parsing when get_json returns None."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = None

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_rankings_with_string_scores(self, endpoint):
        """Test rankings with string scores (should work if valid format)."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [
                {"index": 0, "score": "0.95"},
                {"index": 1, "score": "0.87"},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        # Should accept and preserve whatever format the API returns
        assert parsed is not None
        assert parsed.data.rankings[0]["score"] == "0.95"

    def test_parse_response_zero_scores(self, endpoint):
        """Test rankings with zero scores."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [
                {"index": 0, "score": 0.0},
                {"index": 1, "score": 0.0},
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.rankings[0]["score"] == 0.0
        assert parsed.data.rankings[1]["score"] == 0.0

    @pytest.mark.parametrize(
        "rankings_data",
        [
            [{"score": 0.95}, {"score": 0.87}],  # Missing index
            [{"index": 0}, {"index": 1}],  # Missing score
        ],
    )
    def test_parse_response_missing_fields(self, endpoint, rankings_data):
        """Test rankings with missing optional fields."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {"rankings": rankings_data}

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.rankings) == 2

    def test_parse_response_rankings_as_nested_objects(self, endpoint):
        """Test rankings with complex nested structure."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {
            "rankings": [
                {
                    "index": 0,
                    "score": 0.95,
                    "document": {
                        "id": "doc1",
                        "title": "Title 1",
                        "metadata": {"author": "Author 1"},
                    },
                },
            ]
        }

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.rankings[0]["document"]["title"] == "Title 1"
        assert parsed.data.rankings[0]["document"]["metadata"]["author"] == "Author 1"
