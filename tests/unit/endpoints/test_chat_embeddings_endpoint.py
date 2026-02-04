# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ChatEmbeddingsEndpoint."""

import pytest

from aiperf.common.models import Image, Text, Turn
from aiperf.common.models.record_models import EmbeddingResponseData
from aiperf.endpoints.chat_embeddings import ChatEmbeddingsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestChatEmbeddingsEndpoint:
    """Tests for ChatEmbeddingsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for chat embeddings."""
        return create_model_endpoint(
            EndpointType.CHAT_EMBEDDINGS, model_name="vlm2vec-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a ChatEmbeddingsEndpoint instance."""
        return create_endpoint_with_mock_transport(
            ChatEmbeddingsEndpoint, model_endpoint
        )

    # -------------------------------------------------------------------------
    # format_payload tests (inherited from ChatEndpoint)
    # -------------------------------------------------------------------------

    def test_format_payload_simple_text(self, endpoint, model_endpoint):
        """Test simple text message produces messages array."""
        turn = Turn(
            texts=[Text(contents=["Embed this text"])],
            model="vlm2vec-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "vlm2vec-model"
        assert "messages" in payload
        assert len(payload["messages"]) == 1
        assert payload["messages"][0]["content"] == "Embed this text"

    def test_format_payload_multimodal_with_images(self, endpoint, model_endpoint):
        """Test multimodal message with text and images (main vLLM use case)."""
        turn = Turn(
            texts=[Text(contents=["Describe this image"])],
            images=[Image(contents=["data:image/png;base64,abc123"])],
            model="vlm2vec-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "vlm2vec-model"
        assert "messages" in payload
        assert len(payload["messages"]) == 1

        message = payload["messages"][0]
        assert isinstance(message["content"], list)
        # Should have 1 text part + 1 image part
        assert len(message["content"]) == 2
        assert message["content"][0]["type"] == "text"
        assert message["content"][1]["type"] == "image_url"
        assert (
            message["content"][1]["image_url"]["url"] == "data:image/png;base64,abc123"
        )

    def test_format_payload_model_fallback(self, endpoint, model_endpoint):
        """Test that endpoint model is used when turn model is None."""
        turn = Turn(
            texts=[Text(contents=["Test"])],
            model=None,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_format_payload_extra_params(self):
        """Test extra parameters are included in payload."""
        extra_params = [("encoding_format", "float"), ("dimensions", 1024)]
        model_endpoint = create_model_endpoint(
            EndpointType.CHAT_EMBEDDINGS, model_name="vlm2vec-model", extra=extra_params
        )
        endpoint = create_endpoint_with_mock_transport(
            ChatEmbeddingsEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["Test"])], model="vlm2vec-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["encoding_format"] == "float"
        assert payload["dimensions"] == 1024

    # -------------------------------------------------------------------------
    # parse_response tests (overridden method)
    # -------------------------------------------------------------------------

    def test_parse_response_single_embedding(self, endpoint):
        """Test parsing response with single embedding from multimodal input."""
        mock_response = create_mock_response(
            json_data={
                "data": [
                    {
                        "object": "embedding",
                        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                        "index": 0,
                    }
                ]
            }
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.perf_ns == 123456789
        assert isinstance(parsed.data, EmbeddingResponseData)
        assert len(parsed.data.embeddings) == 1
        assert parsed.data.embeddings[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_parse_response_empty_data_returns_none(self, endpoint):
        """Test parsing response with empty data array returns None."""
        mock_response = create_mock_response(json_data={"data": []})

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_no_json_returns_none(self, endpoint):
        """Test parsing when get_json returns None."""
        mock_response = create_mock_response(json_data=None)

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None

    def test_parse_response_missing_embedding_returns_none(self, endpoint):
        """Test parsing when embedding field is missing returns None."""
        mock_response = create_mock_response(
            json_data={"data": [{"object": "embedding", "index": 0}]}
        )

        parsed = endpoint.parse_response(mock_response)

        # No valid embeddings found, should return None
        assert parsed is None

    def test_parse_response_high_dimensional(self, endpoint):
        """Test parsing high-dimensional embedding vectors."""
        embedding_vector = [float(i) / 1000 for i in range(1536)]  # 1536-dim
        mock_response = create_mock_response(
            json_data={
                "data": [
                    {"object": "embedding", "embedding": embedding_vector, "index": 0}
                ]
            }
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert len(parsed.data.embeddings[0]) == 1536
