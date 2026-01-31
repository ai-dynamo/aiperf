# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.models import Image, Text, Turn
from aiperf.endpoints.nim_embeddings import NIMEmbeddingsEndpoint
from aiperf.plugin import plugins
from aiperf.plugin.enums import EndpointType
from aiperf.plugin.schema.schemas import EndpointMetadata
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)
from tests.unit.endpoints.test_embeddings_endpoint import TestEmbeddingsEndpoint


class TestNIMEmbeddingsEndpoint(TestEmbeddingsEndpoint):
    """Tests for NIMEmbeddingsEndpoint.

    Inherits all base EmbeddingsEndpoint tests and adds NIM-specific
    multimodal functionality tests.
    """

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for NIM embeddings."""
        return create_model_endpoint(
            EndpointType.NIM_EMBEDDINGS, model_name="nim-embeddings-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a NIMEmbeddingsEndpoint instance."""
        return create_endpoint_with_mock_transport(
            NIMEmbeddingsEndpoint, model_endpoint
        )

    # =========================================================================
    # NIM-specific multimodal tests (image support)
    # =========================================================================

    def test_format_payload_image_only(self, endpoint, model_endpoint):
        """Test embedding request with images only."""
        image_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        turn = Turn(
            images=[Image(contents=[image_data_url])],
            model="nim-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "nim-embeddings-model"
        assert payload["input"] == [image_data_url]

    def test_format_payload_multiple_images(self, endpoint, model_endpoint):
        """Test embedding request with multiple images."""
        image1 = "data:image/png;base64,abc123"
        image2 = "data:image/jpeg;base64,def456"
        turn = Turn(
            images=[Image(contents=[image1, image2])],
            model="nim-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == [image1, image2]

    def test_format_payload_text_and_image_combined(self, endpoint, model_endpoint):
        """Test embedding request with both text and images combined."""
        text = "Describe this image"
        image_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        turn = Turn(
            texts=[Text(contents=[text])],
            images=[Image(contents=[image_data_url])],
            model="nim-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "nim-embeddings-model"
        assert payload["input"] == [f"{text} {image_data_url}"]

    def test_format_payload_multiple_text_and_images(self, endpoint, model_endpoint):
        """Test embedding request with multiple texts and images paired together."""
        texts = ["First description", "Second description"]
        images = ["data:image/png;base64,img1", "data:image/png;base64,img2"]
        turn = Turn(
            texts=[Text(contents=texts)],
            images=[Image(contents=images)],
            model="nim-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == [
            f"{texts[0]} {images[0]}",
            f"{texts[1]} {images[1]}",
        ]

    def test_format_payload_text_image_count_mismatch(self, endpoint, model_endpoint):
        """Test that mismatched text and image counts raise an error."""
        turn = Turn(
            texts=[Text(contents=["Text 1", "Text 2", "Text 3"])],
            images=[Image(contents=["data:image/png;base64,img1"])],
            model="nim-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="must have the same length"):
            endpoint.format_payload(request_info)

    def test_format_payload_filters_empty_images(self, endpoint, model_endpoint):
        """Test that empty image strings are filtered from inputs."""
        turn = Turn(
            images=[
                Image(
                    contents=[
                        "data:image/png;base64,valid",
                        "",
                        "data:image/png;base64,another",
                    ]
                )
            ],
            model="nim-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == [
            "data:image/png;base64,valid",
            "data:image/png;base64,another",
        ]

    def test_metadata_returns_nim_specific_title(self, endpoint):
        """Test that metadata returns NIM-specific metrics title."""
        metadata = plugins.get_endpoint_metadata(EndpointType.NIM_EMBEDDINGS)
        assert isinstance(metadata, EndpointMetadata)

        assert metadata.metrics_title == "NIM Embeddings Metrics"
        assert metadata.endpoint_path == "/v1/embeddings"
        assert metadata.supports_streaming is False
        assert metadata.produces_tokens is False
        assert metadata.tokenizes_input is True
        assert metadata.supports_images is True

    def test_format_payload_images_from_multiple_image_objects(
        self, endpoint, model_endpoint
    ):
        """Test extracting images from multiple Image objects in a turn."""
        turn = Turn(
            images=[
                Image(contents=["data:image/png;base64,img1"]),
                Image(contents=["data:image/png;base64,img2"]),
            ],
            model="nim-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == [
            "data:image/png;base64,img1",
            "data:image/png;base64,img2",
        ]
