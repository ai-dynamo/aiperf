# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for endpoint tests."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import EndpointType, ModelSelectionStrategy
from aiperf.common.models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestInfo
from aiperf.common.protocols import InferenceServerResponse


def create_model_endpoint(
    endpoint_type: EndpointType,
    model_name: str = "test-model",
    streaming: bool = False,
    base_url: str = "http://localhost:8000",
    extra: list[tuple[str, Any]] | None = None,
    use_legacy_max_tokens: bool = False,
) -> ModelEndpointInfo:
    """Helper to create a ModelEndpointInfo with common defaults."""
    return ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name=model_name)],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=endpoint_type,
            base_url=base_url,
            streaming=streaming,
            extra=extra or [],
            use_legacy_max_tokens=use_legacy_max_tokens,
        ),
    )


def create_endpoint_with_mock_transport(endpoint_class, model_endpoint):
    """Helper to create an endpoint instance with mocked transport."""
    with patch(
        "aiperf.common.factories.TransportFactory.create_instance"
    ) as mock_transport:
        mock_transport.return_value = MagicMock()
        return endpoint_class(model_endpoint=model_endpoint)


def create_request_info(
    model_endpoint: ModelEndpointInfo,
    texts: list[str],
    model: str | None = None,
    max_tokens: int | None = None,
    **turn_kwargs,
) -> RequestInfo:
    """Helper to create RequestInfo with a simple text turn."""
    turn = Turn(
        texts=[Text(contents=texts)],
        model=model,
        max_tokens=max_tokens,
        **turn_kwargs,
    )
    return RequestInfo(model_endpoint=model_endpoint, turns=[turn])


def create_mock_response(
    perf_ns: int = 123456789,
    json_data: dict | None = None,
    text: str | None = None,
) -> Mock:
    """Helper to create a mock InferenceServerResponse."""
    mock_response = Mock(spec=InferenceServerResponse)
    mock_response.perf_ns = perf_ns
    mock_response.get_json.return_value = json_data
    mock_response.get_text.return_value = text
    return mock_response


@pytest.fixture
def mock_transport_factory():
    """Mock the TransportFactory to return a MagicMock."""
    with patch("aiperf.common.factories.TransportFactory.create_instance") as mock:
        mock.return_value = MagicMock()
        yield mock
