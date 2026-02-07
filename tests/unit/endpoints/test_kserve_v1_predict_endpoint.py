# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.kserve_v1_predict import KServeV1PredictEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestKServeV1PredictFormatPayload:
    """Tests for KServeV1PredictEndpoint.format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V1_PREDICT, model_name="sklearn-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV1PredictEndpoint, model_endpoint
        )

    def test_format_payload_basic(self, endpoint, model_endpoint):
        """Test basic payload with single text prompt."""
        turn = Turn(texts=[Text(contents=["Classify this text"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload == {"instances": [{"text": "Classify this text"}]}

    def test_format_payload_multiple_texts_joined(self, endpoint, model_endpoint):
        """Test that multiple text contents are joined with spaces."""
        turn = Turn(texts=[Text(contents=["Hello", "world"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["instances"][0]["text"] == "Hello world"

    def test_format_payload_custom_input_field(self):
        """Test custom input field name via extra."""
        extra = [("v1_input_field", "input_text")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V1_PREDICT, model_name="sklearn-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV1PredictEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["prompt"])])
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert "input_text" in payload["instances"][0]
        assert payload["instances"][0]["input_text"] == "prompt"

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

        assert payload["instances"][0]["text"] == "valid also valid"


class TestKServeV1PredictParseResponse:
    """Tests for KServeV1PredictEndpoint.parse_response."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.KSERVE_V1_PREDICT, model_name="sklearn-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            KServeV1PredictEndpoint, model_endpoint
        )

    def test_parse_response_dict_predictions(self, endpoint):
        """Test parsing response with dict predictions."""
        response = create_mock_response(
            json_data={"predictions": [{"output": "Predicted text"}]}
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Predicted text"

    def test_parse_response_custom_output_field(self):
        """Test parsing with custom output field name."""
        extra = [("v1_output_field", "result")]
        model_endpoint = create_model_endpoint(
            EndpointType.KSERVE_V1_PREDICT, model_name="sklearn-model", extra=extra
        )
        endpoint = create_endpoint_with_mock_transport(
            KServeV1PredictEndpoint, model_endpoint
        )

        response = create_mock_response(
            json_data={"predictions": [{"result": "Custom output"}]}
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Custom output"

    def test_parse_response_string_predictions(self, endpoint):
        """Test parsing response with string predictions."""
        response = create_mock_response(
            json_data={"predictions": ["Direct text response"]}
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Direct text response"

    def test_parse_response_auto_detect_fallback(self, endpoint):
        """Test auto-detection fallback when predictions missing."""
        response = create_mock_response(json_data={"text": "Auto-detected text"})

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Auto-detected text"

    def test_parse_response_no_json(self, endpoint):
        """Test parsing response with no JSON."""
        response = create_mock_response(json_data=None)

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_parse_response_empty_predictions(self, endpoint):
        """Test parsing response with empty predictions list."""
        response = create_mock_response(json_data={"predictions": []})

        parsed = endpoint.parse_response(response)

        assert parsed is None

    def test_parse_response_dict_with_auto_detect(self, endpoint):
        """Test auto-detect on prediction dict when output field missing."""
        response = create_mock_response(
            json_data={"predictions": [{"text": "Found via auto-detect"}]}
        )

        parsed = endpoint.parse_response(response)

        assert parsed is not None
        assert parsed.data.get_text() == "Found via auto-detect"
