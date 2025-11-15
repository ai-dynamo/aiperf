# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.common.enums import EndpointType
from aiperf.common.models import Text, Turn
from aiperf.common.models.record_models import RequestInfo
from aiperf.endpoints.nim_rankings import NIMRankingsEndpoint
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)


class TestNIMRankingsEndpoint:
    """Test cases for NIMRankingsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for rankings."""
        return create_model_endpoint(EndpointType.NIM_RANKINGS)

    @pytest.fixture
    def converter(self, model_endpoint):
        """Create a NIMRankingsEndpoint instance."""
        return create_endpoint_with_mock_transport(NIMRankingsEndpoint, model_endpoint)

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
            RequestInfo(model_endpoint=model_endpoint, turns=[basic_turn])
        )

        assert payload["model"] == "test-model"
        assert payload["query"] == {"text": "What is artificial intelligence?"}
        assert len(payload["passages"]) == 3
        assert payload["passages"][0] == {"text": "AI is a branch of computer science"}
        assert payload["passages"][1] == {"text": "Machine learning is a subset of AI"}
        assert payload["passages"][2] == {"text": "Deep learning uses neural networks"}

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
            RequestInfo(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["query"] == {"text": "What is Python?"}
        assert len(payload["passages"]) == 1
        assert payload["passages"][0] == {"text": "Python is a programming language"}

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
                RequestInfo(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "Multiple query texts found" in caplog.text
        assert payload["query"] == {"text": "First query"}

    def test_format_payload_no_passages(self, converter, model_endpoint, caplog):
        """Test with query but no passages (should warn)."""
        turn = Turn(
            texts=[Text(name="query", contents=["What is AI?"])], model="test-model"
        )

        with caplog.at_level(logging.WARNING):
            payload = converter.format_payload(
                RequestInfo(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "no passages to rank" in caplog.text
        assert payload["query"] == {"text": "What is AI?"}
        assert payload["passages"] == []

    def test_format_payload_no_query(self, converter, model_endpoint):
        """Test with no query text (should raise error)."""
        turn = Turn(
            texts=[Text(name="passages", contents=["Some passage"])], model="test-model"
        )

        with pytest.raises(
            ValueError, match="requires a text with name 'query' or 'queries'"
        ):
            converter.format_payload(
                RequestInfo(model_endpoint=model_endpoint, turns=[turn])
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

        with pytest.raises(
            ValueError, match="requires a text with name 'query' or 'queries'"
        ):
            converter.format_payload(
                RequestInfo(model_endpoint=model_endpoint, turns=[turn])
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

        # Should warn about ignored texts
        with caplog.at_level(logging.WARNING):
            payload = converter.format_payload(
                RequestInfo(model_endpoint=model_endpoint, turns=[turn])
            )

        # Check that warnings were issued for ignored texts
        assert "context" in caplog.text
        assert "metadata" in caplog.text

        # But payload should still be correct
        assert payload["query"] == {"text": "What is AI?"}
        assert len(payload["passages"]) == 1

    def test_format_payload_mixed_order(self, converter, model_endpoint):
        """Test that text order doesn't matter."""
        turn = Turn(
            texts=[
                Text(name="passages", contents=["Passage 1", "Passage 2"]),
                Text(name="query", contents=["What is AI?"]),
                Text(name="passages", contents=["Passage 3"]),  # Multiple passage texts
            ],
            model="test-model",
        )

        payload = converter.format_payload(
            RequestInfo(model_endpoint=model_endpoint, turns=[turn])
        )

        assert payload["query"] == {"text": "What is AI?"}
        assert len(payload["passages"]) == 3
        assert payload["passages"][0] == {"text": "Passage 1"}
        assert payload["passages"][1] == {"text": "Passage 2"}
        assert payload["passages"][2] == {"text": "Passage 3"}

    def test_format_payload_model_priority(self, converter, model_endpoint):
        """Test that turn model takes priority over endpoint model."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["Test query"]),
                Text(name="passages", contents=["Test passage"]),
            ],
            model="turn-model",  # Different from endpoint model
        )

        payload = converter.format_payload(
            RequestInfo(model_endpoint=model_endpoint, turns=[turn])
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
            RequestInfo(model_endpoint=model_endpoint, turns=[turn])
        )
        assert payload["model"] == model_endpoint.primary_model_name

    def test_format_payload_max_tokens_warning(self, converter, model_endpoint, caplog):
        """Test that max_tokens generates a warning for rankings."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["Test query"]),
                Text(name="passages", contents=["Test passage"]),
            ],
            model="test-model",
            max_tokens=100,  # Should trigger warning
        )

        with caplog.at_level(logging.WARNING):
            converter.format_payload(
                RequestInfo(model_endpoint=model_endpoint, turns=[turn])
            )

        assert "not supported for rankings" in caplog.text

    def test_format_payload_extra_parameters(self):
        """Test that extra parameters from endpoint config are included."""
        extra_params = [("top_k", 5), ("return_scores", True)]
        test_endpoint = create_model_endpoint(
            EndpointType.NIM_RANKINGS, extra=extra_params
        )
        converter = create_endpoint_with_mock_transport(
            NIMRankingsEndpoint, test_endpoint
        )

        turn = Turn(
            texts=[
                Text(name="query", contents=["Test query"]),
                Text(name="passages", contents=["Test passage"]),
            ],
            model="test-model",
        )

        payload = converter.format_payload(
            RequestInfo(model_endpoint=test_endpoint, turns=[turn])
        )

        assert payload["top_k"] == 5
        assert payload["return_scores"] is True
        assert payload["model"] == "test-model"
        assert payload["query"] == {"text": "Test query"}

    def test_converter_docstring(self, converter):
        """Test that the converter has proper documentation."""
        assert "query" in converter.__class__.__doc__
        assert "passages" in converter.__class__.__doc__
        assert "rankings" in converter.__class__.__doc__.lower()

    def test_format_payload_queries_plural_backward_compatibility(
        self, converter, model_endpoint
    ):
        """Test backward compatibility with genai-perf datasets using 'queries' (plural).

        genai-perf uses queries.jsonl files which result in text name='queries'.
        The endpoint should accept both 'query' and 'queries' for backward compatibility.
        """
        turn = Turn(
            texts=[
                Text(name="queries", contents=["What is AI?"]),  # plural form
                Text(name="passages", contents=["AI definition"]),
            ],
            model="test-model",
        )

        payload = converter.format_payload(
            RequestInfo(model_endpoint=model_endpoint, turns=[turn])
        )

        # Should work just like "query" (singular)
        assert payload["query"] == {"text": "What is AI?"}
        assert len(payload["passages"]) == 1
        assert payload["passages"][0] == {"text": "AI definition"}
