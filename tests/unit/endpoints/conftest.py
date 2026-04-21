# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for endpoint tests."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import Text, Turn
from aiperf.common.models.record_models import InferenceServerResponse, RequestInfo
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.plugin.enums import EndpointType

_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000"],
        "streaming": False,
    },
    "datasets": {
        "default": {
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    "phases": {"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
}


def create_config(
    endpoint_type: EndpointType = EndpointType.CHAT,
    model_name: str = "test-model",
    streaming: bool = False,
    base_url: str = "http://localhost:8000",
    extra: dict[str, Any] | None = None,
    use_legacy_max_tokens: bool = False,
    template: dict[str, Any] | None = None,
    **endpoint_overrides: Any,
) -> BenchmarkConfig:
    """Helper to create a BenchmarkConfig with common defaults."""
    endpoint = {
        "type": endpoint_type,
        "urls": [base_url],
        "streaming": streaming,
        "extra": extra or {},
        "use_legacy_max_tokens": use_legacy_max_tokens,
        **endpoint_overrides,
    }
    if template is not None:
        endpoint["template"] = template
    return BenchmarkConfig(
        **{**_MINIMAL_CONFIG_KWARGS, "models": [model_name], "endpoint": endpoint}
    )


def _wrap_run(config: BenchmarkConfig) -> BenchmarkRun:
    """Wrap a BenchmarkConfig in a BenchmarkRun for testing."""
    return BenchmarkRun(benchmark_id="test", cfg=config, artifact_dir=Path("/tmp/test"))


def create_endpoint_with_mock_transport(endpoint_class, config):
    """Helper to create an endpoint instance with mocked transport."""
    run = _wrap_run(config) if isinstance(config, BenchmarkConfig) else config
    return endpoint_class(run=run)


def create_request_info(
    config: BenchmarkConfig,
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
        credit_phase = "profiling"

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
