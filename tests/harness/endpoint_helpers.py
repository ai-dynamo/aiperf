# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared endpoint-test builders (config/run/request-info/response).

These live in ``tests.harness`` (not ``tests/unit/endpoints/conftest.py``) so
tests outside the endpoints package can reuse them without a cross-package
conftest import. ``tests/unit/endpoints/conftest.py`` re-exports every name here
so existing ``from tests.unit.endpoints.conftest import ...`` sites keep working.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

from aiperf.common.enums import CreditPhase, ModelSelectionStrategy
from aiperf.common.models import RequestInfo, Text, Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import InferenceServerResponse
from aiperf.config import BenchmarkConfig, BenchmarkRun
from aiperf.plugin.enums import EndpointType

_MINIMAL_CONFIG_KWARGS: dict[str, Any] = {
    "models": ["test-model"],
    "endpoint": {
        "type": "chat",
        "urls": ["http://localhost:8000"],
        "streaming": False,
    },
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 1,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {"name": "default", "type": "concurrency", "requests": 10, "concurrency": 1}
    ],
}


def create_config(
    endpoint_type: EndpointType = EndpointType.CHAT,
    model_name: str = "test-model",
    streaming: bool = False,
    base_url: str = "http://localhost:8000",
    extra: dict[str, Any] | list[tuple[str, Any]] | None = None,
    use_legacy_max_tokens: bool = False,
    template: dict[str, Any] | None = None,
    **endpoint_overrides: Any,
) -> BenchmarkConfig:
    """Branch-style helper: build a ``BenchmarkConfig`` for endpoint tests."""
    extra_dict: dict[str, Any] = {}
    if isinstance(extra, dict):
        extra_dict = dict(extra)
    elif isinstance(extra, list):
        for pair in extra:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                extra_dict[str(pair[0])] = pair[1]
    endpoint = {
        "type": endpoint_type,
        "urls": [base_url],
        "streaming": streaming,
        "extra": extra_dict,
        "use_legacy_max_tokens": use_legacy_max_tokens,
        **endpoint_overrides,
    }
    if template is not None:
        endpoint["template"] = template
    return BenchmarkConfig(
        **{**_MINIMAL_CONFIG_KWARGS, "models": [model_name], "endpoint": endpoint}
    )


def create_model_endpoint(
    endpoint_type: EndpointType,
    model_name: str = "test-model",
    streaming: bool = False,
    base_url: str = "http://localhost:8000",
    extra: list[tuple[str, Any]] | None = None,
    use_legacy_max_tokens: bool = False,
) -> ModelEndpointInfo:
    """Main-keeper helper: build a ``ModelEndpointInfo``.

    Kept for tests added against main's pre-branch endpoint API (e.g.
    image_edit, openai_chat_attack). New tests should prefer ``create_config``.
    """
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


def _wrap_run(config_or_endpoint: Any) -> BenchmarkRun:
    """Wrap a ``BenchmarkConfig`` (or main-style ``ModelEndpointInfo``) in a ``BenchmarkRun``.

    Branch endpoints take ``model_endpoint=ModelEndpointInfo``. Pre-branch
    tests that build a ``ModelEndpointInfo`` first are auto-translated to a
    matching ``BenchmarkConfig`` so they keep working without per-test rewrites.
    """
    if isinstance(config_or_endpoint, BenchmarkConfig):
        cfg = config_or_endpoint
    elif isinstance(config_or_endpoint, ModelEndpointInfo):
        cfg = _config_from_model_endpoint(config_or_endpoint)
    else:
        # Already a BenchmarkRun (idempotency) or unknown — return as-is.
        if isinstance(config_or_endpoint, BenchmarkRun):
            return config_or_endpoint
        raise TypeError(
            f"_wrap_run expected BenchmarkConfig or ModelEndpointInfo, "
            f"got {type(config_or_endpoint).__name__}"
        )
    return BenchmarkRun(
        benchmark_id="test",
        cfg=cfg,
        artifact_dir=Path("/tmp/test"),
    )


def _wrap_model_endpoint(config_or_endpoint: Any) -> ModelEndpointInfo:
    """Helper used by tests that previously passed ``model_endpoint=_wrap_model_endpoint(...)``.

    Branch endpoints take ``model_endpoint=ModelEndpointInfo`` as the sole
    canonical ctor parameter. This shim accepts whatever the test happens
    to have (``BenchmarkConfig`` / ``ModelEndpointInfo`` / ``BenchmarkRun``)
    and returns the right ``ModelEndpointInfo`` to pass through.
    """
    if isinstance(config_or_endpoint, ModelEndpointInfo):
        return config_or_endpoint
    return ModelEndpointInfo.from_run(_wrap_run(config_or_endpoint))


def _config_from_model_endpoint(model_endpoint: ModelEndpointInfo) -> BenchmarkConfig:
    """Translate a ``ModelEndpointInfo`` into a matching ``BenchmarkConfig``."""
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
    return BenchmarkConfig.model_validate(payload)


def create_endpoint_with_mock_transport(endpoint_class, config):
    """Build an endpoint instance from a ``BenchmarkConfig``, ``BenchmarkRun``, or ``ModelEndpointInfo``.

    Endpoints accept only ``model_endpoint=ModelEndpointInfo``; this helper
    fans out the various shapes test files pass in.
    """
    if isinstance(config, ModelEndpointInfo):
        return endpoint_class(model_endpoint=config)
    run = _wrap_run(config)
    return endpoint_class(model_endpoint=ModelEndpointInfo.from_run(run))


def create_request_info(
    config: BenchmarkConfig | None = None,
    model_endpoint: ModelEndpointInfo | None = None,
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
    """Build a ``RequestInfo`` for endpoint tests.

    ``config`` / ``model_endpoint`` are accepted for source-compat with branch
    and main-style tests respectively; neither flows onto the RequestInfo
    struct directly because msgspec RequestInfo has no endpoint reference.
    Endpoints read endpoint metadata off ``self.run.cfg.endpoint`` /
    ``self.model_endpoint`` from their own state.
    """
    _ = config if config is not None else model_endpoint
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
