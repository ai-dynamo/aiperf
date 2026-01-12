# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.common.enums import EndpointType
from aiperf.common.models import Image, Text, Turn
from aiperf.common.models.record_models import RequestInfo
from aiperf.endpoints.nim_embeddings import NIMEmbeddingsEndpoint
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
)


class TestNIMEmbeddingsEndpoint:
    """Tests for NIMEmbeddingsEndpoint multimodal functionality."""

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

    def test_format_payload_text_only(self, endpoint, model_endpoint):
        """Test that text-only requests work (inherited from base class)."""
        turn = Turn(
            texts=[Text(contents=["Embed this text"])],
            model="nim-embeddings-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "nim-embeddings-model"
        assert payload["input"] == ["Embed this text"]

    def test_format_payload_image_only(self, endpoint, model_endpoint):
        """Test embedding request with images only."""
        image_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        turn = Turn(
            images=[Image(contents=[image_data_url])],
            model="nim-embeddings-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "nim-embeddings-model"
        assert len(payload["input"]) == 1
        assert payload["input"] == [f"<img src='{image_data_url}'/>"]

    def test_format_payload_multiple_images(self, endpoint, model_endpoint):
        """Test embedding request with multiple images."""
        image1 = "data:image/png;base64,abc123"
        image2 = "data:image/jpeg;base64,def456"
        turn = Turn(
            images=[Image(contents=[image1, image2])],
            model="nim-embeddings-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 2
        assert payload["input"] == [f"<img src='{image1}'/>", f"<img src='{image2}'/>"]

    def test_format_payload_text_and_image_combined(self, endpoint, model_endpoint):
        """Test embedding request with both text and images combined."""
        text = "Describe this image"
        image_data_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=="
        turn = Turn(
            texts=[Text(contents=[text])],
            images=[Image(contents=[image_data_url])],
            model="nim-embeddings-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "nim-embeddings-model"
        assert len(payload["input"]) == 1
        assert payload["input"] == [f"{text} <img src='{image_data_url}'/>"]

    def test_format_payload_multiple_text_and_images(self, endpoint, model_endpoint):
        """Test embedding request with multiple texts and images paired together."""
        texts = ["First description", "Second description"]
        images = ["data:image/png;base64,img1", "data:image/png;base64,img2"]
        turn = Turn(
            texts=[Text(contents=texts)],
            images=[Image(contents=images)],
            model="nim-embeddings-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 2
        assert payload["input"] == [
            f"{texts[0]} <img src='{images[0]}'/>",
            f"{texts[1]} <img src='{images[1]}'/>",
        ]

    def test_format_payload_text_image_count_mismatch(self, endpoint, model_endpoint):
        """Test that mismatched text and image counts raise an error."""
        turn = Turn(
            texts=[Text(contents=["Text 1", "Text 2", "Text 3"])],
            images=[Image(contents=["data:image/png;base64,img1"])],
            model="nim-embeddings-model",
        )
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

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
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 2
        assert payload["input"] == [
            "<img src='data:image/png;base64,valid'/>",
            "<img src='data:image/png;base64,another'/>",
        ]

    def test_metadata_returns_nim_specific_title(self, endpoint):
        """Test that metadata returns NIM-specific metrics title."""
        metadata = endpoint.metadata()

        assert metadata.metrics_title == "NIM Embeddings Metrics"
        assert metadata.endpoint_path == "/v1/embeddings"
        assert metadata.supports_streaming is False

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
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 2
        assert payload["input"] == [
            "<img src='data:image/png;base64,img1'/>",
            "<img src='data:image/png;base64,img2'/>",
        ]
