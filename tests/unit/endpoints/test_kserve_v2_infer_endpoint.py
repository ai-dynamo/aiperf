# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.kserve_v2_infer import KServeV2InferEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestKServeV2InferFormatPayload:
    """Tests for KServeV2InferEndpoint.format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V2_INFER, model_name="triton-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV2InferEndpoint, model_endpoint
        )

    def test_format_payload_basic(self, endpoint, model_endpoint):
        """Test basic payload with single text prompt."""
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

    def test_format_payload_multiple_texts_joined(self, endpoint, model_endpoint):
        """Test that multiple text contents are joined with spaces."""
        turn = Turn(texts=[Text(contents=["Hello", "world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["data"] == ["Hello world"]

    def test_format_payload_with_max_tokens(self, endpoint, model_endpoint):
        """Test that max_tokens adds an INT32 tensor input."""
        turn = Turn(texts=[Text(contents=["prompt"])], max_tokens=256)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["inputs"]) == 2
        max_tokens_input = payload["inputs"][1]
        assert max_tokens_input["name"] == "max_tokens"
        assert max_tokens_input["shape"] == [1]
        assert max_tokens_input["datatype"] == "INT32"
        assert max_tokens_input["data"] == [256]

    def test_format_payload_no_max_tokens(self, endpoint, model_endpoint):
        """Test that no max_tokens omits the max_tokens tensor."""
        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["inputs"]) == 1
        assert "parameters" not in payload

    def test_format_payload_with_extra_params(self):
        """Test that remaining extras become parameters."""
        extra = [
            ("temperature", 0.7),
            ("top_k", 40),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_INFER, model_name="triton-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2InferEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["parameters"] == {"temperature": 0.7, "top_k": 40}

    def test_format_payload_custom_input_name(self):
        """Test custom input tensor name via extra."""
        extra = [("v2_input_name", "INPUT_TEXT")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_INFER, model_name="triton-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2InferEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["name"] == "INPUT_TEXT"

    def test_format_payload_empty_turns_raises(self, endpoint, model_endpoint):
        """Test that empty turns raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_filters_empty_text(self, endpoint, model_endpoint):
        """Test that empty text contents are filtered out."""
        turn = Turn(texts=[Text(contents=["valid", "", "also valid"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"][0]["data"] == ["valid also valid"]


class TestKServeV2InferParseResponse:
    """Tests for KServeV2InferEndpoint.parse_response."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V2_INFER, model_name="triton-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV2InferEndpoint, model_endpoint
        )

    def test_parse_response_basic(self, endpoint):
        """Test parsing a standard V2 response."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "text_output",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["Generated text"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Generated text"

    def test_parse_response_custom_output_name(self):
        """Test parsing with custom output tensor name."""
        extra = [("v2_output_name", "OUTPUT_TEXT")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V2_INFER, model_name="triton-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV2InferEndpoint, model_endpoint
        )

        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "OUTPUT_TEXT",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["Response text"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Response text"

    def test_parse_response_multiple_outputs_matches_name(self, endpoint):
        """Test that the correct output is matched by name."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {"name": "logits", "shape": [1], "datatype": "FP32", "data": [0.5]},
                    {
                        "name": "text_output",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["Correct output"],
                    },
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Correct output"

    def test_parse_response_fallback_to_first_output(self, endpoint):
        """Test fallback to first output when named output not found."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "other_output",
                        "shape": [1],
                        "datatype": "BYTES",
                        "data": ["Fallback text"],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Fallback text"

    def test_parse_response_no_json(self, endpoint):
        """Test parsing response with no JSON."""
        response = create_mock_response(json_data=None)

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_parse_response_no_outputs(self, endpoint):
        """Test parsing response with no outputs field."""
        response = create_mock_response(json_data={"id": "123"})

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_parse_response_empty_data(self, endpoint):
        """Test parsing response with empty data array."""
        response = create_mock_response(
            json_data={
                "outputs": [
                    {
                        "name": "text_output",
                        "shape": [0],
                        "datatype": "BYTES",
                        "data": [],
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(response)

        assert parsed is None
