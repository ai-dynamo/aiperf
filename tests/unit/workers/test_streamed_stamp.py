# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.enums import ModelSelectionStrategy
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import ErrorDetails, RequestRecord
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.workers.inference_client import InferenceClient


@pytest.fixture
def mock_http_transport_entry():
    """Create a mock transport entry with http/https url_schemes."""
    entry = MagicMock()
    entry.name = TransportType.HTTP.value
    entry.metadata = {"url_schemes": ["http", "https"]}
    return entry


@pytest.fixture
def model_endpoint():
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
def inference_client(model_endpoint, mock_http_transport_entry):
    """Create an InferenceClient with a mocked endpoint and transport."""
    mock_transport = MagicMock()
    mock_endpoint = MagicMock()
    mock_endpoint.get_endpoint_headers.return_value = {}
    mock_endpoint.get_endpoint_params.return_value = {}
    mock_endpoint.format_payload.return_value = {}

    def mock_get_class(protocol, name):
        if protocol == "endpoint":
            return lambda **kwargs: mock_endpoint
        if protocol == "transport":
            return lambda **kwargs: mock_transport
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.workers.inference_client.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[mock_http_transport_entry],
        ),
    ):
        return InferenceClient(
            model_endpoint=model_endpoint, service_id="test-service-id"
        )


@pytest.mark.asyncio
async def test_streaming_request_record_stamped_true(
    inference_client, sample_request_info
):
    """A per-request streaming override of True stamps ``streamed`` True."""
    request_info = sample_request_info
    request_info.stream_override = True
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=request_info)
    )

    record = await inference_client.send_request(request_info)

    assert record.streamed is True


@pytest.mark.asyncio
async def test_non_streaming_request_record_stamped_false(
    inference_client, sample_request_info
):
    """A per-request override of False wins over a global streaming=True."""
    request_info = sample_request_info
    request_info.model_endpoint.endpoint.streaming = True
    request_info.stream_override = False
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=request_info)
    )

    record = await inference_client.send_request(request_info)

    assert record.streamed is False


@pytest.mark.asyncio
async def test_error_record_still_stamped_with_effective_mode(
    inference_client, sample_request_info
):
    """An error record reflects the mode the request was SENT with.

    A mid-stream error is still a streamed send, so the ground-truth stamp
    tracks the effective wire mode rather than whether a response arrived.
    """
    request_info = sample_request_info
    request_info.stream_override = True
    error_record = RequestRecord(
        request_info=request_info,
        error=ErrorDetails(type="TestError", message="boom"),
    )
    inference_client.transport.send_request = AsyncMock(return_value=error_record)

    record = await inference_client.send_request(request_info)

    assert record.has_error
    assert record.streamed is True
