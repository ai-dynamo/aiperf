# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.kserve_v2_embeddings import KServeV2EmbeddingsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestKServeV2EmbeddingsFormatPayload:
    """Tests for KServeV2EmbeddingsEndpoint.format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V2_EMBEDDINGS, model_name="embed-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV2EmbeddingsEndpoint, model_endpoint
        )

    def test_format_payload_basic(self, endpoint, model_endpoint):
        """Test basic payload with single text."""
        turn = Turn(texts=[Text(contents=["Hello world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "inputs" in payload
        inputs = payload["inputs"]
        assert len(inputs) == 1
        assert inputs[0]["name"] == "text_input"
        assert inputs[0]["shape"] == [1]
        assert inputs[0]["datatype"] == "BYTES"
        assert inputs[0]["data"] == ["Hello world"]

    def test_format_payload_multiple_texts_batch(self, endpoint, model_endpoint):
        """Test that multiple text contents create a batch with shape=[N]."""
        turn = Turn(texts=[Text(contents=["Hello", "World", "Test"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["shape"] == [3]
        assert payload["inputs"][0]["data"] == ["Hello", "World", "Test"]

    def test_format_payload_filters_empty(self, endpoint, model_endpoint):
        """Test that empty text contents are filtered out."""
        turn = Turn(texts=[Text(contents=["valid", "", "also valid"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["shape"] == [2]
        assert payload["inputs"][0]["data"] == ["valid", "also valid"]

    def test_format_payload_single_turn_validation(self, endpoint, model_endpoint):
        """Test that multiple turns raises ValueError."""
        turn1 = Turn(texts=[Text(contents=["a"])])
        turn2 = Turn(texts=[Text(contents=["b"])])
        request_info = create_request_info(
            model_endpoint=model_endpoint, turns=[turn1, turn2]
        )

        with pytest.raises(ValueError, match="only supports one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_empty_turns_raises(self, endpoint, model_endpoint):
        """Test that empty turns raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="only supports one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_max_tokens_warning(self, endpoint, model_endpoint, caplog):
        """Test that max_tokens generates a warning."""
        turn = Turn(texts=[Text(contents=["prompt"])], max_tokens=256)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with caplog.at_level(logging.WARNING):
            endpoint.format_payload(request_info)

        assert "not supported for embeddings" in caplog.text

    def test_format_payload_custom_input_name(self):
        """Test custom input tensor name via extra."""
        extra = [("v2_input_name", "INPUT_TEXT")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_EMBEDDINGS, model_name="embed-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2EmbeddingsEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["name"] == "INPUT_TEXT"

    def test_format_payload_extra_params(self):
        """Test that remaining extras become parameters."""
        extra = [("temperature", 0.7), ("top_k", 40)]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_EMBEDDINGS, model_name="embed-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2EmbeddingsEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["parameters"] == {"temperature": 0.7, "top_k": 40}


class TestKServeV2EmbeddingsParseResponse:
    """Tests for KServeV2EmbeddingsEndpoint.parse_response."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V2_EMBEDDINGS, model_name="embed-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV2EmbeddingsEndpoint, model_endpoint
        )

    def test_parse_response_basic_fp32(self, endpoint):
        """Test parsing a standard V2 response with FP32 embeddings."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "embedding_output",
                        "shape": [1, 3],
                        "datatype": "FP32",
                        "data": [0.1, 0.2, 0.3],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.embeddings == [[0.1, 0.2, 0.3]]

    def test_parse_response_batch_reshape(self, endpoint):
        """Test reshaping flat array into multiple embedding vectors."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "embedding_output",
                        "shape": [2, 3],
                        "datatype": "FP32",
                        "data": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert len(parsed.data.embeddings) == 2
        assert parsed.data.embeddings[0] == [0.1, 0.2, 0.3]
        assert parsed.data.embeddings[1] == [0.4, 0.5, 0.6]

    def test_parse_response_single_dim_shape(self, endpoint):
        """Test that single-dim shape treats entire array as one embedding."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "embedding_output",
                        "shape": [3],
                        "datatype": "FP32",
                        "data": [0.1, 0.2, 0.3],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.embeddings == [[0.1, 0.2, 0.3]]

    def test_parse_response_custom_output_name(self):
        """Test parsing with custom output tensor name."""
        extra = [("v2_output_name", "EMBED_OUT")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_EMBEDDINGS, model_name="embed-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2EmbeddingsEndpoint, model_endpoint
        )

        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "EMBED_OUT",
                        "shape": [1, 2],
                        "datatype": "FP32",
                        "data": [0.5, 0.6],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.embeddings == [[0.5, 0.6]]

    def test_parse_response_fallback_first_output(self, endpoint):
        """Test fallback to first output when named output not found."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "other_output",
                        "shape": [1, 2],
                        "datatype": "FP32",
                        "data": [0.7, 0.8],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.embeddings == [[0.7, 0.8]]

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
                        "name": "embedding_output",
                        "shape": [0],
                        "datatype": "FP32",
                        "data": [],
                    }
                ]
            }
        )

        assert endpoint.parse_response(response) is None
