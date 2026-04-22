# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.common.models import Image, Text, Turn
from aiperf.common.models.record_models import EmbeddingResponseData
from aiperf.endpoints.cohere_embeddings import CohereEmbeddingsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_mock_response,
    create_model_endpoint,
    create_request_info,
)


class TestCohereEmbeddingsEndpoint:
    """Tests for CohereEmbeddingsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for Cohere embeddings."""
        return create_model_endpoint(
            EndpointType.COHERE_EMBEDDINGS, model_name="cohere-embeddings-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create a CohereEmbeddingsEndpoint instance."""
        return create_endpoint_with_mock_transport(
            CohereEmbeddingsEndpoint, model_endpoint
        )

    def test_format_payload_text_inputs(self, endpoint, model_endpoint):
        """Text-only requests should use the Cohere texts field."""
        turn = Turn(
            texts=[Text(contents=["Hello world", "How are you?"])],
            model="cohere-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "cohere-embeddings-model"
        assert payload["texts"] == ["Hello world", "How are you?"]
        assert "inputs" not in payload
        assert "images" not in payload

    def test_format_payload_filters_empty_texts(self, endpoint, model_endpoint):
        """Empty text strings should be removed before sending."""
        turn = Turn(
            texts=[Text(contents=["Valid", "", "Still valid"])],
            model="cohere-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["texts"] == ["Valid", "Still valid"]

    def test_format_payload_image_only(self, endpoint, model_endpoint):
        """Image-only requests should use Cohere input objects."""
        image = "data:image/png;base64,abc123"
        turn = Turn(
            images=[Image(contents=[image])],
            model="cohere-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"] == [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image},
                    }
                ]
            }
        ]
        assert "texts" not in payload
        assert "images" not in payload

    def test_format_payload_mixed_text_and_image(self, endpoint, model_endpoint):
        """Mixed requests should pair text and image content like NIM semantics."""
        turn = Turn(
            texts=[Text(contents=["A photo of a cat"])],
            images=[Image(contents=["data:image/png;base64,abc123"])],
            model="cohere-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"] == [
            {
                "content": [
                    {"type": "text", "text": "A photo of a cat"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ]
            }
        ]
        assert "texts" not in payload
        assert "images" not in payload

    def test_format_payload_mixed_requires_matching_counts(
        self, endpoint, model_endpoint
    ):
        """Mixed requests with ambiguous pairing should fail."""
        turn = Turn(
            texts=[Text(contents=["One", "Two"])],
            images=[Image(contents=["data:image/png;base64,abc123"])],
            model="cohere-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with pytest.raises(ValueError, match="must have the same length"):
            endpoint.format_payload(request_info)

    def test_format_payload_multiple_images_use_inputs(self, endpoint, model_endpoint):
        """Multiple image embeddings should use inputs to avoid the single-image field."""
        turn = Turn(
            images=[
                Image(
                    contents=[
                        "data:image/png;base64,abc123",
                        "data:image/png;base64,def456",
                    ]
                )
            ],
            model="cohere-embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["inputs"] == [
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    }
                ]
            },
            {
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,def456"},
                    }
                ]
            },
        ]

    def test_format_payload_extra_parameters(self):
        """Endpoint extra config should merge into the Cohere payload."""
        model_endpoint = create_model_endpoint(
            EndpointType.COHERE_EMBEDDINGS,
            model_name="cohere-embeddings-model",
            extra=[
                ("input_type", "query"),
                ("embedding_types", ["float"]),
                ("output_dimension", 256),
                ("truncate", "END"),
            ],
        )
        endpoint = create_endpoint_with_mock_transport(
            CohereEmbeddingsEndpoint, model_endpoint
        )
        turn = Turn(texts=[Text(contents=["Test"])], model="cohere-embeddings-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input_type"] == "query"
        assert payload["embedding_types"] == ["float"]
        assert payload["output_dimension"] == 256
        assert payload["truncate"] == "END"

    def test_format_payload_model_fallback(self, endpoint, model_endpoint):
        """Turn model should fall back to the endpoint's primary model."""
        turn = Turn(texts=[Text(contents=["Test"])], model=None)
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_format_payload_max_tokens_warning(self, endpoint, model_endpoint, caplog):
        """Max tokens is still invalid for embedding endpoints."""
        turn = Turn(
            texts=[Text(contents=["Test"])],
            model="cohere-embeddings-model",
            max_tokens=128,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with caplog.at_level(logging.ERROR):
            endpoint.format_payload(request_info)

        assert "not supported for embeddings" in caplog.text

    def test_parse_response_float_embeddings(self, endpoint):
        """Float embeddings should parse directly."""
        mock_response = create_mock_response(
            json_data={
                "embeddings": {"float": [[0.1, 0.2], [0.3, 0.4]]},
                "meta": {"billed_units": {"input_tokens": 12}},
            }
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert isinstance(parsed.data, EmbeddingResponseData)
        assert parsed.data.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 12
        assert parsed.usage.completion_tokens == 0

    def test_parse_response_without_float_embeddings_returns_usage_only(self, endpoint):
        """Non-float embedding payloads should be ignored."""
        mock_response = create_mock_response(
            json_data={
                "embeddings": {"base64": ["AAAAAA=="]},
                "meta": {"billed_units": {"input_tokens": 6}},
            }
        )

        parsed = endpoint.parse_response(mock_response)

        assert parsed is not None
        assert parsed.data is None
        assert parsed.usage is not None
        assert parsed.usage.prompt_tokens == 6

    def test_parse_response_missing_embeddings_returns_none(self, endpoint):
        """Missing embeddings field should return None."""
        mock_response = create_mock_response(json_data={"meta": {}})

        parsed = endpoint.parse_response(mock_response)

        assert parsed is None
