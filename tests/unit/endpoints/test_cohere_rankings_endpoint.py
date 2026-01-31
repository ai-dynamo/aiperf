# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.cohere_rankings import CohereRankingsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class TestCohereRankingsEndpoint:
    """Unit tests for CohereRankingsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for Cohere rankings."""
        return create_model_endpoint(EndpointType.COHERE_RANKINGS)

    @pytest.fixture
    def converter(self, model_endpoint):
        """Create a CohereRankingsEndpoint instance."""
        return create_endpoint_with_mock_transport(
            CohereRankingsEndpoint, model_endpoint
        )

    @pytest.fixture
    def basic_turn(self):
        """Create a basic turn with query and documents."""
        return Turn(
            texts=[
                Text(name="query", contents=["What is deep learning?"]),
                Text(
                    name="passages",  # kept as 'passages' since input dataset uses that
                    contents=[
                        "Deep learning uses neural networks.",
                        "Bananas are yellow.",
                        "Machine learning is related to AI.",
                    ],
                ),
            ],
            model="test-model",
        )

    def test_format_payload_basic(self, converter, model_endpoint, basic_turn):
        """Test basic payload formatting with query and passages."""
        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[basic_turn])
        )

        assert payload["model"] == "test-model"
        assert payload["query"] == "What is deep learning?"
        assert len(payload["documents"]) == 3
        assert "Deep learning uses neural networks." in payload["documents"][0]

    def test_format_payload_single_passage(self, converter, model_endpoint):
        """Test payload formatting with single passage."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["What is Python?"]),
                Text(name="passages", contents=["Python is a programming language"]),
            ],
            model="test-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["query"] == "What is Python?"
        assert len(payload["documents"]) == 1
        assert payload["documents"][0] == "Python is a programming language"

    def test_format_payload_multiple_query_contents(
        self, converter, model_endpoint, caplog
    ):
        """Test with multiple contents in query text (uses first one)."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["First query", "Second query"]),
                Text(name="passages", contents=["Some passage"]),
            ],
            model="test-model",
        )

        with caplog.at_level(logging.WARNING):
            payload = converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "Multiple query texts found" in caplog.text
        assert payload["query"] == "First query"

    def test_format_payload_no_passages(self, converter, model_endpoint, caplog):
        """Test with query but no passages (should warn)."""
        turn = Turn(
            texts=[Text(name="query", contents=["What is AI?"])],
            model="test-model",
        )

        with caplog.at_level(logging.WARNING):
            payload = converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "no passages to rank" in caplog.text
        assert payload["query"] == "What is AI?"
        assert payload["documents"] == []

    def test_format_payload_no_query(self, converter, model_endpoint):
        """Test with no query text (should raise error)."""
        turn = Turn(
            texts=[Text(name="passages", contents=["Some passage"])],
            model="test-model",
        )

        with pytest.raises(ValueError, match="requires a text with name 'query'"):
            converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

    def test_format_payload_empty_query_contents(self, converter, model_endpoint):
        """Test with empty query contents (should raise error)."""
        turn = Turn(
            texts=[
                Text(name="query", contents=[]),
                Text(name="passages", contents=["Some passage"]),
            ],
            model="test-model",
        )

        with pytest.raises(ValueError, match="requires a text with name 'query'"):
            converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

    def test_format_payload_model_priority(self, converter, model_endpoint):
        """Test that turn model takes priority over endpoint model."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["Test query"]),
                Text(name="passages", contents=["Test passage"]),
            ],
            model="turn-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["model"] == "turn-model"

    def test_format_payload_fallback_model(self, converter, model_endpoint):
        """Test fallback to endpoint model when turn model is None."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["Test query"]),
                Text(name="passages", contents=["Test passage"]),
            ],
            model=None,
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["model"] == model_endpoint.primary_model_name

    def test_extract_rankings(self, converter):
        """Test extraction of ranking results from API response."""
        mock_json = {
            "results": [
                {"index": 0, "relevance_score": 0.95},
                {"index": 2, "relevance_score": 0.10},
            ]
        }

        rankings = converter.extract_rankings(mock_json)
        assert len(rankings) == 2
        assert rankings[0]["index"] == 0
        assert rankings[0]["score"] == 0.95
        assert rankings[1]["index"] == 2
        assert rankings[1]["score"] == 0.10
