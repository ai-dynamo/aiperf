# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for endpoint tests."""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import RequestInfo, Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import InferenceServerResponse
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


# Alias retained for branch-side tests imported as ``create_config``.
create_config = create_model_endpoint


def _benchmark_run_from_model_endpoint(model_endpoint: ModelEndpointInfo):
    """Build a minimal ``BenchmarkRun`` whose ``cfg`` matches the given endpoint shape.

    Branch-side endpoints take ``run: BenchmarkRun`` rather than the
    pre-branch ``model_endpoint: ModelEndpointInfo`` -- so we build a
    benchmark run with mirroring config to drive the endpoint under test.
    """
    import uuid

    from aiperf.config import BenchmarkConfig, BenchmarkRun

    extra_pairs = list(model_endpoint.endpoint.extra or [])
    extra_dict: dict[str, Any] = {}
    for pair in extra_pairs:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            extra_dict[str(pair[0])] = pair[1]
        elif isinstance(pair, dict):
            extra_dict.update(pair)

    payload: dict[str, Any] = {
        "models": [m.name for m in model_endpoint.models.models],
        "endpoint": {
            "type": model_endpoint.endpoint.type,
            "urls": [model_endpoint.endpoint.base_url],
            "streaming": model_endpoint.endpoint.streaming,
            "extra": extra_dict,
            "use_legacy_max_tokens": getattr(
                model_endpoint.endpoint, "use_legacy_max_tokens", False
            ),
        },
        "datasets": [{"name": "default", "type": "synthetic"}],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 1,
            }
        ],
    }
    cfg = BenchmarkConfig.model_validate(payload)
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )


def create_endpoint_with_mock_transport(endpoint_class, model_endpoint):
    """Helper to create an endpoint instance with a mocked transport.

    Endpoints now accept either ``model_endpoint=ModelEndpointInfo`` (main
    keeper convention) or ``run=BenchmarkRun`` (branch port convention).
    Pass through whichever the caller provided; the BaseEndpoint shim
    auto-derives the other.
    """
    return endpoint_class(model_endpoint=model_endpoint)


def create_request_info(
    model_endpoint: ModelEndpointInfo | None = None,
    config: ModelEndpointInfo | None = None,  # alias retained for branch tests
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

    ``model_endpoint`` / ``config`` are accepted for source-compat with both
    main's and the K8s branch's tests but no longer flow onto the
    ``RequestInfo`` struct (msgspec shape). Endpoints read endpoint metadata
    off ``self.model_endpoint`` (auto-derived from ``run`` when needed).
    """
    _ = model_endpoint if model_endpoint is not None else config
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
