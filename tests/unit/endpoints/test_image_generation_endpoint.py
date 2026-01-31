# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ImageGenerationEndpoint."""

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.openai_image_generation import ImageGenerationEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestImageGenerationEndpoint:
    """Tests for ImageGenerationEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for image generation."""
        return create_model_endpoint(
            EndpointType.IMAGE_GENERATION, model_name="dall-e-3"
        )

    @pytest.fixture
    def streaming_model_endpoint(self):
        """Create a test ModelEndpointInfo with streaming enabled."""
        return create_model_endpoint(
            EndpointType.IMAGE_GENERATION, model_name="dall-e-3", streaming=True
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create an ImageGenerationEndpoint instance."""
        return create_endpoint_with_mock_transport(
            ImageGenerationEndpoint, model_endpoint
        )

    @pytest.fixture
    def streaming_endpoint(self, streaming_model_endpoint):
        """Create an ImageGenerationEndpoint instance with streaming."""
        return create_endpoint_with_mock_transport(
            ImageGenerationEndpoint, streaming_model_endpoint
        )

    # ===== format_payload tests =====

    def test_format_payload_simple_prompt(self, endpoint, model_endpoint):
        """Test simple prompt formatting."""
        turn = Turn(
            texts=[Text(contents=["A sunset over mountains"])],
            model="dall-e-3",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["prompt"] == "A sunset over mountains"
        assert payload["model"] == "dall-e-3"
        assert payload["response_format"] == "b64_json"
        assert payload["n"] == 1
        assert "stream" not in payload

    def test_format_payload_with_streaming(
        self, streaming_endpoint, streaming_model_endpoint
    ):
        """Test payload formatting with streaming enabled."""
        turn = Turn(
            texts=[Text(contents=["A cat in space"])],
            model="dall-e-3",
        )
        request_info = create_request_info(
            model_endpoint=streaming_model_endpoint, turns=[turn]
        )

        payload = streaming_endpoint.format_payload(request_info)

        assert payload["stream"] is True
        assert payload["prompt"] == "A cat in space"

    def test_format_payload_with_extra_inputs(self):
        """Test payload formatting with extra inputs."""
        model_endpoint_with_extra = create_model_endpoint(
            EndpointType.IMAGE_GENERATION,
            model_name="dall-e-3",
            extra=[
                ("size", "1024x1024"),
                ("quality", "hd"),
                ("style", "vivid"),
                ("n", 2),
            ],
        )
        endpoint = create_endpoint_with_mock_transport(
            ImageGenerationEndpoint, model_endpoint_with_extra
        )

        turn = Turn(
            texts=[Text(contents=["A dog"])],
            model="dall-e-3",
        )
        request_info = create_request_info(
            model_endpoint=model_endpoint_with_extra, turns=[turn]
        )

        payload = endpoint.format_payload(request_info)

        assert payload["size"] == "1024x1024"
        assert payload["quality"] == "hd"
        assert payload["style"] == "vivid"
        assert payload["n"] == 2  # Extra input overrides default

    def test_format_payload_model_from_turn(self, endpoint, model_endpoint):
        """Test that turn model overrides endpoint model."""
        turn = Turn(
            texts=[Text(contents=["A tree"])],
            model="custom-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "custom-model"

    def test_format_payload_no_turns_raises(self, endpoint, model_endpoint):
        """Test that missing turns raises ValueError."""
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[])

        with pytest.raises(ValueError, match="requires at least one turn"):
            endpoint.format_payload(request_info)

    def test_format_payload_no_text_raises(self, endpoint, model_endpoint):
        """Test that missing text raises ValueError."""
        turn = Turn(texts=[], model="dall-e-3")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="requires text prompt"):
            endpoint.format_payload(request_info)

    def test_format_payload_empty_text_contents_raises(self, endpoint, model_endpoint):
        """Test that empty text contents raises ValueError."""
        turn = Turn(texts=[Text(contents=[])], model="dall-e-3")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="requires text prompt"):
            endpoint.format_payload(request_info)

    # ===== parse_response tests =====

    @pytest.mark.parametrize(
        "json_data,expected_images",
        [
            # Single b64_json image
            (
                {"data": [{"b64_json": "iVBORw0KGgoAAAANS"}]},
                [{"b64_json": "iVBORw0KGgoAAAANS", "url": None}],
            ),
            # Single URL image
            (
                {"data": [{"url": "https://example.com/image.png"}]},
                [{"b64_json": None, "url": "https://example.com/image.png"}],
            ),
            # Multiple images
            (
                {
                    "data": [
                        {"b64_json": "image1"},
                        {"b64_json": "image2"},
                        {"b64_json": "image3"},
                    ]
                },
                [
                    {"b64_json": "image1", "url": None},
                    {"b64_json": "image2", "url": None},
                    {"b64_json": "image3", "url": None},
                ],
            ),
            # Mixed URL and b64
            (
                {
                    "data": [
                        {"url": "https://example.com/img1.png"},
                        {"b64_json": "image2"},
                    ]
                },
                [
                    {"b64_json": None, "url": "https://example.com/img1.png"},
                    {"b64_json": "image2", "url": None},
                ],
            ),
            # Empty data array
            ({"data": []}, []),
        ],
    )  # fmt: skip
    def test_parse_response_image_formats(self, endpoint, json_data, expected_images):
        """Test parsing various image format responses."""
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.images) == len(expected_images)
        for i, expected in enumerate(expected_images):
            assert parsed.data.images[i].b64_json == expected["b64_json"]
            assert parsed.data.images[i].url == expected["url"]

    def test_parse_response_with_revised_prompt(self, endpoint):
        """Test parsing response with revised prompt."""
        json_data = {
            "data": [
                {
                    "b64_json": "image_data",
                    "revised_prompt": "A beautiful sunset over mountains, high quality",
                }
            ]
        }  # fmt: skip
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.images) == 1
        assert (
            parsed.data.images[0].revised_prompt
            == "A beautiful sunset over mountains, high quality"
        )

    @pytest.mark.parametrize(
        "json_data,expected_metadata",
        [
            # Metadata fields
            (
                {
                    "data": [{"b64_json": "img"}],
                    "size": "1024x1024",
                    "quality": "hd",
                    "output_format": "png",
                    "background": "transparent",
                },
                {
                    "size": "1024x1024",
                    "quality": "hd",
                    "output_format": "png",
                    "background": "transparent",
                },
            ),
            # Partial metadata
            (
                {
                    "data": [{"b64_json": "img"}],
                    "size": "512x512",
                },
                {
                    "size": "512x512",
                    "quality": None,
                    "output_format": None,
                    "background": None,
                },
            ),
            # No metadata
            (
                {"data": [{"b64_json": "img"}]},
                {
                    "size": None,
                    "quality": None,
                    "output_format": None,
                    "background": None,
                },
            ),
        ],
    )
    def test_parse_response_metadata(self, endpoint, json_data, expected_metadata):
        """Test parsing response metadata fields."""
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.size == expected_metadata["size"]
        assert parsed.data.quality == expected_metadata["quality"]
        assert parsed.data.output_format == expected_metadata["output_format"]
        assert parsed.data.background == expected_metadata["background"]

    def test_parse_response_streaming_chunk(self, endpoint):
        """Test parsing streaming response chunk."""
        json_data = {
            "b64_json": "partial_image_data",
            "partial_image_index": 0,
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.images) == 1
        assert parsed.data.images[0].b64_json == "partial_image_data"
        assert parsed.data.images[0].partial_image_index == 0

    def test_parse_response_with_usage(self, endpoint):
        """Test parsing response with usage information."""
        json_data = {
            "data": [{"b64_json": "image_data"}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 0,
                "total_tokens": 10,
            },
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 10
        assert parsed.usage.total_tokens == 10

    def test_parse_response_perf_ns(self, endpoint):
        """Test that perf_ns is preserved."""
        json_data = {"data": [{"b64_json": "image"}]}
        mock_response = create_mock_response(perf_ns=999888777, json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 999888777

    def test_parse_response_preserves_order(self, endpoint):
        """Test that images order is preserved."""
        json_data = {
            "data": [
                {"b64_json": "first_image"},
                {"b64_json": "second_image"},
                {"b64_json": "third_image"},
            ]
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.images[0].b64_json == "first_image"
        assert parsed.data.images[1].b64_json == "second_image"
        assert parsed.data.images[2].b64_json == "third_image"

    def test_parse_response_extra_fields_ignored(self, endpoint):
        """Test that extra fields in response are ignored."""
        json_data = {
            "data": [
                {
                    "b64_json": "image_data",
                    "extra_field": "ignored",
                    "another_field": 123,
                }
            ],
            "model": "dall-e-3",
            "unknown_field": "value",
        }
        mock_response = create_mock_response(json_data=json_data)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data.images[0].b64_json == "image_data"

    def test_parse_response_no_json(self, endpoint):
        """Test parsing when get_json returns None."""
        mock_response = create_mock_response(json_data=None)
        mock_response.get_raw.return_value = ""

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None
