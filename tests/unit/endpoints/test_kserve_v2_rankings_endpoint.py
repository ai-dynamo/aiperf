# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.kserve_v2_rankings import KServeV2RankingsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestKServeV2RankingsFormatPayload:
    """Tests for KServeV2RankingsEndpoint.format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V2_RANKINGS, model_name="rank-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV2RankingsEndpoint, model_endpoint
        )

    @pytest.fixture
    def basic_turn(self):
        return Turn(
            texts=[
                Text(name="query", contents=["What is AI?"]),
                Text(
                    name="passages",
                    contents=["AI is computer science", "ML is a subset of AI"],
                ),
            ],
        )

    def test_format_payload_basic(self, endpoint, model_endpoint, basic_turn):
        """Test basic payload with query and passages as BYTES tensors."""
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[basic_turn]
        )

        payload = endpoint.format_payload(request_info)

        assert "inputs" in payload
        inputs = payload["inputs"]
        assert len(inputs) == 2

        query_input = inputs[0]
        assert query_input["name"] == "query"
        assert query_input["shape"] == [1]
        assert query_input["datatype"] == "BYTES"
        assert query_input["data"] == ["What is AI?"]

        passages_input = inputs[1]
        assert passages_input["name"] == "passages"
        assert passages_input["shape"] == [2]
        assert passages_input["datatype"] == "BYTES"
        assert passages_input["data"] == [
            "AI is computer science",
            "ML is a subset of AI",
        ]

    def test_format_payload_tensor_shapes(self, endpoint, model_endpoint):
        """Test that tensor shapes match the number of inputs."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["query text"]),
                Text(
                    name="passages",
                    contents=["p1", "p2", "p3"],
                ),
            ],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["shape"] == [1]  # query always shape [1]
        assert payload["inputs"][1]["shape"] == [3]  # 3 passages

    def test_format_payload_custom_names(self):
        """Test custom tensor names via extra."""
        extra = [
            ("v2_query_name", "QUERY_INPUT"),
            ("v2_passages_name", "PASSAGE_INPUT"),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_RANKINGS, model_name="rank-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2RankingsEndpoint, model_endpoint
        )

        turn = Turn(
            texts=[
                Text(name="query", contents=["test query"]),
                Text(name="passages", contents=["passage"]),
            ],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["name"] == "QUERY_INPUT"
        assert payload["inputs"][1]["name"] == "PASSAGE_INPUT"

    def test_format_payload_extra_params(self):
        """Test that remaining extras become parameters."""
        extra = [("top_k", 5)]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_RANKINGS, model_name="rank-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2RankingsEndpoint, model_endpoint
        )

        turn = Turn(
            texts=[
                Text(name="query", contents=["test"]),
                Text(name="passages", contents=["p1"]),
            ],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["parameters"] == {"top_k": 5}

    def test_format_payload_no_query_raises(self, endpoint, model_endpoint):
        """Test that missing query raises ValueError."""
        turn = Turn(
            texts=[Text(name="passages", contents=["passage"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(
            ValueError, match="requires a text with name 'query' or 'queries'"
        ):
            endpoint.format_payload(request_info)

    def test_format_payload_no_passages_warns(self, endpoint, model_endpoint, caplog):
        """Test that missing passages generates a warning."""
        turn = Turn(
            texts=[Text(name="query", contents=["test query"])],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with caplog.at_level(logging.WARNING):
            payload = endpoint.format_payload(request_info)

        assert "no passages to rank" in caplog.text
        assert payload["inputs"][1]["data"] == []
        assert payload["inputs"][1]["shape"] == [0]

    def test_format_payload_queries_plural_compat(self, endpoint, model_endpoint):
        """Test backward compatibility with 'queries' (plural) name."""
        turn = Turn(
            texts=[
                Text(name="queries", contents=["What is AI?"]),
                Text(name="passages", contents=["AI definition"]),
            ],
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["data"] == ["What is AI?"]

    def test_format_payload_single_turn_validation(self, endpoint, model_endpoint):
        """Test that multiple turns raises ValueError."""
        turn1 = Turn(texts=[Text(name="query", contents=["a"])])
        turn2 = Turn(texts=[Text(name="query", contents=["b"])])
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[turn1, turn2]
        )

        with pytest.raises(ValueError, match="only supports one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_max_tokens_warning(self, endpoint, model_endpoint, caplog):
        """Test that max_tokens generates a warning."""
        turn = Turn(
            texts=[
                Text(name="query", contents=["test"]),
                Text(name="passages", contents=["p1"]),
            ],
            max_tokens=100,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with caplog.at_level(logging.WARNING):
            endpoint.format_payload(request_info)

        assert "not supported for rankings" in caplog.text


class TestKServeV2RankingsParseResponse:
    """Tests for KServeV2RankingsEndpoint.parse_response."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V2_RANKINGS, model_name="rank-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV2RankingsEndpoint, model_endpoint
        )

    def test_parse_response_basic_scores(self, endpoint):
        """Test parsing basic ranking scores."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "scores",
                        "shape": [3],
                        "datatype": "FP32",
                        "data": [0.9, 0.7, 0.3],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        rankings = parsed.data.rankings
        assert len(rankings) == 3
        assert rankings[0] == {"index": 0, "score": 0.9}
        assert rankings[1] == {"index": 1, "score": 0.7}
        assert rankings[2] == {"index": 2, "score": 0.3}

    def test_parse_response_custom_output_name(self):
        """Test parsing with custom output tensor name."""
        extra = [("v2_output_name", "RANK_SCORES")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_RANKINGS, model_name="rank-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2RankingsEndpoint, model_endpoint
        )

        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "RANK_SCORES",
                        "shape": [2],
                        "datatype": "FP32",
                        "data": [0.8, 0.2],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert len(parsed.data.rankings) == 2
        assert parsed.data.rankings[0] == {"index": 0, "score": 0.8}

    def test_parse_response_fallback_first_output(self, endpoint):
        """Test fallback to first output when named output not found."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "other_output",
                        "shape": [2],
                        "datatype": "FP32",
                        "data": [0.5, 0.4],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert len(parsed.data.rankings) == 2

    def test_parse_response_no_json(self, endpoint):
        """Test parsing response with no JSON."""
        response = create_mock_response(json_data=None)

        assert endpoint.parse_response(response) is None

    def test_parse_response_no_outputs(self, endpoint):
        """Test parsing response with no outputs field."""
        response = create_mock_response(json_data={"id": "123"})

        assert endpoint.parse_response(response) is None

    def test_parse_response_empty_data(self, endpoint):
        """Test parsing response with empty data array."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "scores",
                        "shape": [0],
                        "datatype": "FP32",
                        "data": [],
                    }
                ]
            }
        )

        assert endpoint.parse_response(response) is None
