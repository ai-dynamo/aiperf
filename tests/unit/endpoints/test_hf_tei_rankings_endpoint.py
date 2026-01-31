# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.hf_tei_rankings import HFTeiRankingsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class TestHFTeiRankingsEndpoint:
    """Test cases for HFTeiRankingsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for HF TEI rankings."""
        return create_model_endpoint(EndpointType.HF_TEI_RANKINGS)

    @pytest.fixture
    def converter(self, model_endpoint):
        """Create an HFTeiRankingsEndpoint instance."""
        return create_endpoint_with_mock_transport(
            HFTeiRankingsEndpoint, model_endpoint
        )

    @pytest.fixture
    def basic_turn(self):
        """Create a basic turn with query and passages."""
        return Turn(
            texts=[
                Text(name="query", contents=["What is artificial intelligence?"]),
                Text(
                    name="passages",
                    contents=[
                        "AI is a branch of computer science",
                        "Machine learning is a subset of AI",
                        "Deep learning uses neural networks",
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

        assert payload["query"] == "What is artificial intelligence?"
        assert len(payload["texts"]) == 3
        assert payload["texts"][0] == "AI is a branch of computer science"
        assert payload["texts"][1] == "Machine learning is a subset of AI"
        assert payload["texts"][2] == "Deep learning uses neural networks"

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
        assert payload["texts"] == ["Python is a programming language"]

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
            texts=[Text(name="query", contents=["What is AI?"])], model="test-model"
        )

        with caplog.at_level(logging.WARNING):
            payload = converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "no passages to rank" in caplog.text
        assert payload["query"] == "What is AI?"
        assert payload["texts"] == []

    def test_format_payload_no_query(self, converter, model_endpoint):
        """Test with no query text (should raise error)."""
        turn = Turn(
            texts=[Text(name="passages", contents=["Some passage"])], model="test-model"
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

    def test_format_payload_ignored_texts(self, converter, model_endpoint, caplog):
        """Test that texts with other names are ignored with warning."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["What is AI?"]),
                Text(name="passages", contents=["AI definition"]),
                Text(name="context", contents=["This should be ignored"]),
                Text(name="metadata", contents=["This too"]),
            ],
            model="test-model",
        )

        with caplog.at_level(logging.WARNING):
            payload = converter.format_payload(
                create_request_info(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "context" in caplog.text
        assert "metadata" in caplog.text
        assert payload["query"] == "What is AI?"
        assert len(payload["texts"]) == 1

    def test_format_payload_mixed_order(self, converter, model_endpoint):
        """Test that text order doesn't matter."""
        turn = Turn(
            texts=[
                Text(name="passages", contents=["Passage 1", "Passage 2"]),
                Text(name="query", contents=["What is AI?"]),
                Text(name="passages", contents=["Passage 3"]),
            ],
            model="test-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["query"] == "What is AI?"
        assert payload["texts"] == ["Passage 1", "Passage 2", "Passage 3"]

    def test_format_payload_extra_parameters(self):
        """Test that extra parameters from endpoint config are included."""
        extra_params = [("top_k", 5), ("return_scores", True)]
        test_endpoint = create_model_endpoint(
            EndpointType.HF_TEI_RANKINGS, extra=extra_params
        )
        converter = create_endpoint_with_mock_transport(
            HFTeiRankingsEndpoint, test_endpoint
        )

        turn = Turn(
            texts=[
                Text(name="query", contents=["Test query"]),
                Text(name="passages", contents=["Test passage"]),
            ],
            model="test-model",
        )

        payload = converter.format_payload(
            create_request_info(model_endpoint=test_endpoint, turns=[turn])
        )

        assert payload["top_k"] == 5
        assert payload["return_scores"] is True
        assert payload["query"] == "Test query"
