# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


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
    InferenceServerResponse,
    RankingsResponseData,
)
from aiperf.endpoints.cohere_rankings import CohereRankingsEndpoint
from aiperf.endpoints.hf_tei_rankings import HFTeiRankingsEndpoint
from aiperf.endpoints.nim_rankings import NIMRankingsEndpoint
from aiperf.plugin.enums import EndpointType


def mock_response_nim(rankings):
    """Return mock JSON for NIM-style ranking responses."""
    return {"rankings": rankings}


def mock_response_hf_tei(rankings):
    """Return mock JSON for HuggingFace TEI ranking responses."""
    return {"results": rankings}


def mock_response_cohere(rankings):
    """Return mock JSON for Cohere ranking responses (relevance_score key)."""
    converted = [
        {
            "index": r.get("index"),
            "relevance_score": r.get("score"),
            "text": r.get("text"),
            "metadata": r.get("metadata"),
        }
        for r in rankings
    ]
    return {"results": converted}


class TestRankingsEndpointParseResponse:
    """Tests for RankingsEndpoint parse_response functionality."""

    @pytest.fixture(
        params=[
            (EndpointType.NIM_RANKINGS, NIMRankingsEndpoint, mock_response_nim),
            (
                EndpointType.HF_TEI_RANKINGS,
                HFTeiRankingsEndpoint,
                mock_response_hf_tei,
            ),
            (
                EndpointType.COHERE_RANKINGS,
                CohereRankingsEndpoint,
                mock_response_cohere,
            ),
        ]
    )
    def endpoint(self, request):
        """Create endpoint instances for all ranking endpoint types."""
        endpoint_type, endpoint_cls, mock_fn = request.param
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="rankings-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=endpoint_type,
                base_url="http://localhost:8000",
            ),
        )
        with patch("aiperf.plugin.plugins.get_class") as mock_transport:
            mock_transport.return_value = MagicMock()
            endpoint_instance = endpoint_cls(model_endpoint=model_endpoint)
            # Attach mock builder to endpoint for easy access in tests
            endpoint_instance._mock_builder = mock_fn
            return endpoint_instance

    def _make_mock_response(self, endpoint, rankings):
        """Create Mock(InferenceServerResponse) for given endpoint and rankings list."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = endpoint._mock_builder(rankings)
        return mock_response

    def test_parse_response_basic_rankings(self, endpoint):
        rankings = [
            {"index": 0, "score": 0.95},
            {"index": 1, "score": 0.87},
            {"index": 2, "score": 0.72},
        ]
        mock_response = self._make_mock_response(endpoint, rankings)
        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None, f"{endpoint.__class__.__name__} returned None"
        assert isinstance(parsed.data, RankingsResponseData)
        assert len(parsed.data.rankings) == 3

    def test_parse_response_empty_rankings(self, endpoint):
        mock_response = self._make_mock_response(endpoint, [])
        parsed = endpoint.parse_response(mock_response)
        assert parsed is None

    def test_parse_response_no_rankings_key(self, endpoint):
        """Should handle missing key gracefully."""
        mock_response = Mock(spec=InferenceServerResponse)
        mock_response.perf_ns = 123456789
        mock_response.get_json.return_value = {}
        parsed = endpoint.parse_response(mock_response)
        assert parsed is None

    def test_parse_response_with_extra_fields(self, endpoint):
        rankings = [
            {"index": 0, "score": 0.9, "text": "passage1"},
            {"index": 1, "score": 0.8, "metadata": {"source": "doc1"}},
        ]
        mock_response = self._make_mock_response(endpoint, rankings)
        parsed = endpoint.parse_response(mock_response)
        assert parsed is not None
        assert isinstance(parsed.data.rankings, list)
        first = parsed.data.rankings[0]
        assert any(k in first for k in ("score", "relevance_score"))

    def test_parse_response_nested_objects(self, endpoint):
        rankings = [
            {
                "index": 0,
                "score": 0.95,
                "document": {"id": "d1", "meta": {"source": "x"}},
            }
        ]
        mock_response = self._make_mock_response(endpoint, rankings)
        parsed = endpoint.parse_response(mock_response)
        assert parsed is not None, (
            f"{endpoint.__class__.__name__} failed to parse nested objects"
        )
