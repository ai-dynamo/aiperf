# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from aiperf.common.models import Text, Turn
from aiperf.endpoints.openai_embeddings import EmbeddingsEndpoint
from aiperf.plugin.enums import EndpointType
from tests.unit.endpoints.conftest import (
    create_endpoint_with_mock_transport,
    create_model_endpoint,
    create_request_info,
)


class TestEmbeddingsEndpoint:
    """Comprehensive tests for EmbeddingsEndpoint."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo for embeddings."""
        return create_model_endpoint(
            EndpointType.EMBEDDINGS, model_name="embeddings-model"
        )

    @pytest.fixture
    def endpoint(self, model_endpoint):
        """Create an EmbeddingsEndpoint instance."""
        return create_endpoint_with_mock_transport(EmbeddingsEndpoint, model_endpoint)

    def test_format_payload_single_input(self, endpoint, model_endpoint):
        """Test single input text for embedding."""
        turn = Turn(
            texts=[Text(contents=["Embed this text"])],
            model="embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == "embeddings-model"
        assert payload["input"] == ["Embed this text"]

    def test_format_payload_multiple_inputs(self, endpoint, model_endpoint):
        """Test multiple input texts for batch embedding."""
        turn = Turn(
            texts=[
                Text(contents=["First text", "Second text"]),
                Text(contents=["Third text"]),
            ],
            model="embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 3
        assert payload["input"] == ["First text", "Second text", "Third text"]

    def test_format_payload_filters_empty_inputs(self, endpoint, model_endpoint):
        """Test that empty strings are filtered from inputs."""
        turn = Turn(
            texts=[
                Text(contents=["Valid text", "", "Another valid", ""]),
            ],
            model="embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 2
        assert payload["input"] == ["Valid text", "Another valid"]

    def test_format_payload_max_tokens_warning(self, endpoint, model_endpoint, caplog):
        """Test that max_tokens triggers an error for embeddings."""
        turn = Turn(
            texts=[Text(contents=["Test"])],
            model="embeddings-model",
            max_tokens=100,  # Not supported for embeddings
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        with caplog.at_level(logging.ERROR):
            endpoint.format_payload(request_info)

        assert "not supported for embeddings" in caplog.text

    def test_format_payload_model_fallback(self, endpoint, model_endpoint):
        """Test fallback to endpoint model when turn model is None."""
        turn = Turn(
            texts=[Text(contents=["Test"])],
            model=None,
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["model"] == model_endpoint.primary_model_name

    def test_format_payload_extra_parameters(self):
        """Test extra parameters are included."""
        extra_params = [
            ("encoding_format", "float"),
            ("dimensions", 1536),
        ]
        model_endpoint = create_model_endpoint(
            EndpointType.EMBEDDINGS, model_name="embeddings-model", extra=extra_params
        )
        endpoint = create_endpoint_with_mock_transport(
            EmbeddingsEndpoint, model_endpoint
        )

        turn = Turn(texts=[Text(contents=["Test"])], model="embeddings-model")
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["encoding_format"] == "float"
        assert payload["dimensions"] == 1536

    def test_format_payload_empty_texts_list(self, endpoint, model_endpoint):
        """Test with no text contents."""
        turn = Turn(
            texts=[],
            model="embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == []

    def test_format_payload_uses_first_turn_only(self, endpoint, model_endpoint):
        """Test that only request_info.turns[0] is used."""
        turn1 = Turn(texts=[Text(contents=["Embed me"])], model="model1")
        turn2 = Turn(texts=[Text(contents=["Not me"])], model="model2")

        request_info = create_request_info(
            model_endpoint=model_endpoint, turn_index=0, turns=[turn1, turn2]
        )

        # Should raise ValueError because embeddings endpoint only supports one turn
        with pytest.raises(
            ValueError, match="Embeddings endpoint only supports one turn"
        ):
            endpoint.format_payload(request_info)

    def test_format_payload_very_long_input(self, endpoint, model_endpoint):
        """Test handling of very long input text."""
        long_text = "word " * 10000  # Very long input
        turn = Turn(
            texts=[Text(contents=[long_text])],
            model="embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert len(payload["input"]) == 1
        assert payload["input"][0] == long_text

    def test_format_payload_special_characters(self, endpoint, model_endpoint):
        """Test handling of special characters in text."""
        special_text = "Text with émojis 🎉 and símböls & <tags>"
        turn = Turn(
            texts=[Text(contents=[special_text])],
            model="embeddings-model",
        )
        request_info = create_request_info(model_endpoint=model_endpoint, turns=[turn])

        payload = endpoint.format_payload(request_info)

        assert payload["input"] == [special_text]
