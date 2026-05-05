# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for endpoint tests."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import InferenceServerResponse, RequestInfo
from aiperf.plugin.enums import EndpointType


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
    return endpoint_class(model_endpoint=model_endpoint)


def create_request_info(
    model_endpoint: ModelEndpointInfo,
    texts: list[str] | None = None,
    turns: list[Turn] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    turn_index: int = 0,
    credit_num: int = 0,
    credit_phase: CreditPhase | None = None,
    x_request_id: str = "test-request-id",
    x_correlation_id: str = "test-correlation-id",
    conversation_id: str = "test-conversation",
    system_message: str | None = None,
    user_context_message: str | None = None,
    **turn_kwargs,
) -> RequestInfo:
    """Helper to create RequestInfo with all required fields.

    Can either provide texts (to create a simple turn) or provide turns directly.
    """
    if credit_phase is None:
        credit_phase = CreditPhase.PROFILING

    if turns is None:
        if texts is None:
            texts = ["test prompt"]
        turn = Turn(
            texts=[Text(contents=texts)],
            model=model,
            max_tokens=max_tokens,
            **turn_kwargs,
        )
        turns = [turn]

    return RequestInfo(
        model_endpoint=model_endpoint,
        turns=turns,
        turn_index=turn_index,
        credit_num=credit_num,
        credit_phase=credit_phase,
        x_request_id=x_request_id,
        x_correlation_id=x_correlation_id,
        conversation_id=conversation_id,
        system_message=system_message,
        user_context_message=user_context_message,
    )


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
def mock_transport_plugin():
    """Mock the plugin transport class to return a MagicMock."""
    with patch("aiperf.plugin.plugins.get_class") as mock:
        mock.return_value = MagicMock
        yield mock
