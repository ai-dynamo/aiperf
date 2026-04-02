# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ImageRetrievalEndpoint."""

import pytest

from aiperf.common.models import Image, Turn
from aiperf.common.models.record_models import ImageRetrievalResponseData
from aiperf.endpoints.nim_image_retrieval import ImageRetrievalEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)

BASE64_PNG = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


class TestImageRetrievalEndpointFormatPayload:
    """Tests for ImageRetrievalEndpoint.format_payload."""

    @pytest.fixture
    def model_endpoint(self):
        return create_model_endpoint(
            EndpointType.IMAGE_RETRIEVAL, model_name="image-retrieval-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        return create_endpoint_with_mock_transport(
            ImageRetrievalEndpoint, model_endpoint
        )

    def test_format_payload_single_image(self, endpoint, model_endpoint):
        """Test basic format_payload with a single image."""
        turn = Turn(
            images=[Image(contents=[BASE64_PNG])], model="image-retrieval-model"
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 1
        assert payload["input"][0]["type"] == "image_url"
        assert payload["input"][0]["url"] == BASE64_PNG

    def test_format_payload_multiple_images(self, endpoint, model_endpoint):
        """Test format_payload with multiple base64 images."""
        img1 = BASE64_PNG
        img2 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="
        turn = Turn(
            images=[Image(contents=[img1, img2])], model="image-retrieval-model"
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 2
        assert payload["input"][0]["url"] == img1
        assert payload["input"][1]["url"] == img2

    def test_format_payload_multiple_image_objects(self, endpoint, model_endpoint):
        """Test format_payload with multiple Image objects."""
        img2 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ=="
        turn = Turn(
            images=[
                Image(contents=[BASE64_PNG]),
                Image(contents=[img2]),
            ],
            model="image-retrieval-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 2

    def test_format_payload_with_extra(self):
        """Test format_payload applies endpoint extra fields."""
        model_endpoint_with_extra = create_model_endpoint(
            EndpointType.IMAGE_RETRIEVAL,
            model_name="image-retrieval-model",
            extra=[("threshold", 0.5), ("max_detections", 10)],
        )
        endpoint = create_endpoint_with_mock_transport(
            ImageRetrievalEndpoint, model_endpoint_with_extra
        )
        turn = Turn(
            images=[Image(contents=[BASE64_PNG])], model="image-retrieval-model"
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint_with_extra, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["threshold"] == 0.5
        assert payload["max_detections"] == 10

    def test_format_payload_no_images_raises(self, endpoint, model_endpoint):
        """Test that empty images raises ValueError."""
        turn = Turn(images=[], model="image-retrieval-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(
            ValueError, match="Image Retrieval request requires at least one image"
        ):
            endpoint.format_payload(request_info)

    def test_format_payload_multiple_turns_raises(self, endpoint, model_endpoint):
        """Test that multiple turns raises ValueError."""
        turns = [
            Turn(images=[Image(contents=[BASE64_PNG])], model="image-retrieval-model"),
            Turn(images=[Image(contents=[BASE64_PNG])], model="image-retrieval-model"),
        ]
        request_info = create_request_info(model_endpoint=model_endpoint, turns=turns)

        with pytest.raises(
            ValueError, match="Image Retrieval endpoint only supports one turn"
        ):
            endpoint.format_payload(request_info)

    def test_format_payload_empty_contents_raises(self, endpoint, model_endpoint):
        """Test that images with empty contents raises ValueError."""
        turn = Turn(images=[Image(contents=["", ""])], model="image-retrieval-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="No valid image content found"):
            endpoint.format_payload(request_info)


class TestImageRetrievalEndpointParseResponse:
    """Tests for ImageRetrievalEndpoint.parse_response."""

    @pytest.fixture
    def endpoint(self):
        model_endpoint = create_model_endpoint(
            EndpointType.IMAGE_RETRIEVAL, model_name="image-retrieval-model"
        )
        return create_endpoint_with_mock_transport(
            ImageRetrievalEndpoint, model_endpoint
        )

    def test_parse_response_basic(self, endpoint):
        """Test basic parse_response with valid bounding box data."""
        json_data = {
            "data": [
                {
                    "index": 0,
                    "bounding_boxes": {
                        "chart": [
                            {
                                "x_min": 10,
                                "y_min": 20,
                                "x_max": 100,
                                "y_max": 120,
                                "confidence": 0.95,
                            }
                        ]
                    },
                }
            ],
            "usage": {"images_size_mb": 0.5},
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, ImageRetrievalResponseData)
        assert len(parsed.data.data) == 1
        assert "chart" in parsed.data.data[0]["bounding_boxes"]

    def test_parse_response_multiple_items(self, endpoint):
        """Test parse_response with multiple data items."""
        json_data = {
            "data": [
                {"index": 0, "bounding_boxes": {"chart": []}},
                {"index": 1, "bounding_boxes": {"table": []}},
            ]
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.data) == 2

    def test_parse_response_no_json_returns_none(self, endpoint):
        """Test parse_response returns None for non-JSON response."""
        mock_response = create_mock_response(json_data=None)
        mock_response.get_raw.return_value = ""

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_no_data_field_returns_none(self, endpoint):
        """Test parse_response returns None when data field is missing."""
        mock_response = create_mock_response(json_data={"status": "ok"})

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_empty_data_returns_none(self, endpoint):
        """Test parse_response returns None when data is empty list."""
        mock_response = create_mock_response(json_data={"data": []})

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_preserves_perf_ns(self, endpoint):
        """Test that perf_ns from response is preserved."""
        json_data = {"data": [{"index": 0}]}
        mock_response = create_mock_response(perf_ns=999888777, json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 999888777

    def test_parse_response_get_text_returns_empty(self, endpoint):
        """Test that ImageRetrievalResponseData.get_text() returns empty string."""
        json_data = {"data": [{"index": 0}]}
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.get_text() == ""
