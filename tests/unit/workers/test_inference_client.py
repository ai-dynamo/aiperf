# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo, RequestRecord
from aiperf.workers.inference_client import InferenceClient


class TestInferenceClient:
    """Tests for InferenceClient functionality."""

    @pytest.fixture
    def model_endpoint(self):
        """Create a test ModelEndpointInfo."""
        return ModelEndpointInfo(
            models=ModelListInfo(
                models=[ModelInfo(name="test-model")],
                model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
            ),
            endpoint=EndpointInfo(
                type=EndpointType.CHAT,
                base_url="http://localhost:8000/v1/test",
            ),
        )

    @pytest.fixture
    def inference_client(self, model_endpoint):
        """Create an InferenceClient instance."""
        with (
            patch(
                "aiperf.common.factories.TransportFactory.create_instance"
            ) as mock_transport_factory,
            patch(
                "aiperf.common.factories.EndpointFactory.create_instance"
            ) as mock_endpoint_factory,
        ):
            mock_transport = MagicMock()
            mock_endpoint = MagicMock()
            mock_endpoint.get_endpoint_headers.return_value = {}
            mock_endpoint.get_endpoint_params.return_value = {}
            mock_endpoint.format_payload.return_value = {}
            mock_transport_factory.return_value = mock_transport
            mock_endpoint_factory.return_value = mock_endpoint
            return InferenceClient(model_endpoint=model_endpoint)

    @pytest.mark.asyncio
    async def test_send_request_sets_endpoint_headers(
        self, inference_client, model_endpoint
    ):
        """Test that send_request sets endpoint_headers on request_info."""
        model_endpoint.endpoint.api_key = "test-key"
        model_endpoint.endpoint.headers = [("X-Custom", "value")]

        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[])

        expected_headers = {
            "Authorization": "Bearer test-key",
            "X-Custom": "value",
        }
        inference_client.endpoint.get_endpoint_headers.return_value = expected_headers

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await inference_client.send_request(request_info)

        assert "Authorization" in request_info.endpoint_headers
        assert request_info.endpoint_headers["Authorization"] == "Bearer test-key"
        assert request_info.endpoint_headers["X-Custom"] == "value"

    @pytest.mark.asyncio
    async def test_send_request_sets_endpoint_params(
        self, inference_client, model_endpoint
    ):
        """Test that send_request sets endpoint_params on request_info."""
        model_endpoint.endpoint.url_params = {"api-version": "v1", "timeout": "30"}

        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[])

        expected_params = {"api-version": "v1", "timeout": "30"}
        inference_client.endpoint.get_endpoint_params.return_value = expected_params

        inference_client.transport.send_request = AsyncMock(
            return_value=RequestRecord()
        )

        await inference_client.send_request(request_info)

        assert request_info.endpoint_params["api-version"] == "v1"
        assert request_info.endpoint_params["timeout"] == "30"

    @pytest.mark.asyncio
    async def test_send_request_calls_transport(self, inference_client, model_endpoint):
        """Test that send_request delegates to transport."""
        request_info = RequestInfo(model_endpoint=model_endpoint, turns=[])
        expected_record = RequestRecord()

        inference_client.transport.send_request = AsyncMock(
            return_value=expected_record
        )

        record = await inference_client.send_request(request_info)

        inference_client.transport.send_request.assert_called_once()
        call_args = inference_client.transport.send_request.call_args
        assert call_args[0][0] == request_info
        assert record == expected_record
