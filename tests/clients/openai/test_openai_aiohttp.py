# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

import pytest

from aiperf.clients.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.clients.openai.openai_aiohttp import OpenAIClientAioHttp
from aiperf.common.enums import EndpointType, ModelSelectionStrategy


class TestOpenAIClientAioHttpHeaders:
    """Test OpenAI client header generation with x_request_id and x_correlation_id."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                custom_endpoint="/v1/chat/completions",
                api_key="test-api-key",
            ),
        )

    @pytest.fixture
    def client(self):
        """Create a minimal OpenAI client instance without full initialization."""
        return OpenAIClientAioHttp.__new__(OpenAIClientAioHttp)

    def test_get_headers_without_request_ids(self, client, model_endpoint):
        """Test get_headers without x_request_id or x_correlation_id."""
        headers = client.get_headers(model_endpoint)

        assert "X-Request-ID" not in headers
        assert "X-Correlation-ID" not in headers
        assert headers["User-Agent"] == "aiperf/1.0"
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-api-key"

    def test_get_headers_with_x_request_id(self, client, model_endpoint):
        """Test get_headers with x_request_id."""
        request_id = str(uuid.uuid4())
        headers = client.get_headers(model_endpoint, x_request_id=request_id)

        assert headers["X-Request-ID"] == request_id
        assert "X-Correlation-ID" not in headers

    def test_get_headers_with_x_correlation_id(self, client, model_endpoint):
        """Test get_headers with x_correlation_id."""
        correlation_id = str(uuid.uuid4())
        headers = client.get_headers(model_endpoint, x_correlation_id=correlation_id)

        assert headers["X-Correlation-ID"] == correlation_id
        assert "X-Request-ID" not in headers

    def test_get_headers_with_empty_string_ids(self, client, model_endpoint):
        """Test get_headers with empty string IDs (should not add headers)."""
        headers = client.get_headers(
            model_endpoint,
            x_request_id="",
            x_correlation_id="",
        )

        assert "X-Request-ID" not in headers
        assert "X-Correlation-ID" not in headers

    def test_headers_with_streaming_endpoint(self, client):
        """Test headers for streaming endpoint."""
        streaming_model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                custom_endpoint="/v1/chat/completions",
                api_key="test-api-key",
                streaming=True,
            ),
        )

        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        headers = client.get_headers(
            streaming_model_endpoint,
            x_request_id=request_id,
            x_correlation_id=correlation_id,
        )

        assert headers["Accept"] == "text/event-stream"
        assert headers["X-Request-ID"] == request_id
        assert headers["X-Correlation-ID"] == correlation_id

    def test_headers_with_custom_headers(self, client):
        """Test that custom headers are preserved with x_request_id and x_correlation_id."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                custom_endpoint="/v1/chat/completions",
                api_key="test-api-key",
                headers=[
                    ("X-Custom-Header", "custom-value"),
                    ("X-Another", "another-value"),
                ],
            ),
        )

        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        headers = client.get_headers(
            model_endpoint,
            x_request_id=request_id,
            x_correlation_id=correlation_id,
        )

        assert headers["X-Request-ID"] == request_id
        assert headers["X-Correlation-ID"] == correlation_id
        assert headers["X-Custom-Header"] == "custom-value"
        assert headers["X-Another"] == "another-value"

    def test_headers_without_api_key(self, client):
        """Test headers without API key."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                custom_endpoint="/v1/chat/completions",
                api_key=None,
            ),
        )

        request_id = str(uuid.uuid4())
        headers = client.get_headers(model_endpoint, x_request_id=request_id)

        assert "Authorization" not in headers
        assert headers["X-Request-ID"] == request_id

    def test_custom_headers_override_id_headers(self, client):
        """Test that custom headers with same name will override x_request_id and x_correlation_id."""
        model_endpoint = ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.RANDOM,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000",
                custom_endpoint="/v1/chat/completions",
                headers=[
                    ("X-Request-ID", "custom-request-id"),
                    ("X-Correlation-ID", "custom-correlation-id"),
                ],
            ),
        )

        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        headers = client.get_headers(
            model_endpoint,
            x_request_id=request_id,
            x_correlation_id=correlation_id,
        )

        assert headers["X-Request-ID"] == "custom-request-id"
        assert headers["X-Correlation-ID"] == "custom-correlation-id"
