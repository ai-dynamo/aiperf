# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache-bust endpoint capability validation."""

from __future__ import annotations

from typing import Any

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.config.config import BenchmarkConfig
from aiperf.plugin.enums import EndpointType, TimingMode

_URL = "http://localhost:8000/v1/chat/completions"


def _build(
    *,
    target: CacheBustTarget,
    endpoint_type: EndpointType = EndpointType.CHAT,
    timing_mode: TimingMode | None = None,
) -> BenchmarkConfig:
    """Construct a v2 BenchmarkConfig for an endpoint capability combination."""
    phase: dict[str, Any] = {
        "name": "profiling",
        "type": "concurrency",
        "concurrency": 1,
        "requests": 10,
    }
    if timing_mode is not None:
        phase["timing_mode"] = timing_mode

    endpoint: dict[str, Any] = {"urls": [_URL], "type": endpoint_type}
    if endpoint_type == EndpointType.TEMPLATE:
        endpoint["template"] = {
            "body": '{"prompt": "{{prompt}}"}',
            "response_field": "text",
        }

    body: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": endpoint,
        "datasets": [
            {
                "name": "main",
                "type": "synthetic",
                "prompts": {
                    "isl": 128,
                    "osl": 16,
                    "cache_bust": {"target": target},
                },
            }
        ],
        "phases": [phase],
    }
    return BenchmarkConfig.model_validate(body)


_NON_NONE_CACHE_BUST_TARGETS: list[CacheBustTarget] = [
    t for t in CacheBustTarget if t != CacheBustTarget.NONE
]
_WARMUP_ISOLATION_TARGETS: list[CacheBustTarget] = [
    CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
    CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
]
_RID_CACHE_BUST_TARGETS: list[CacheBustTarget] = [
    t for t in _NON_NONE_CACHE_BUST_TARGETS if t not in _WARMUP_ISOLATION_TARGETS
]
_TIMING_MODES: list[TimingMode] = list(TimingMode)
_NON_AGENTIC_TIMING_MODES: list[TimingMode] = [
    t for t in TimingMode if t != TimingMode.AGENTIC_REPLAY
]
_INCOMPATIBLE_ENDPOINT_TYPES: list[EndpointType] = [
    e
    for e in EndpointType
    if e not in {EndpointType.CHAT, EndpointType.RESPONSES, EndpointType.MESSAGES}
]


@pytest.mark.parametrize("timing_mode", _NON_AGENTIC_TIMING_MODES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_accepted_with_non_agentic_timing_mode(
    timing_mode: TimingMode, target: CacheBustTarget
) -> None:
    """Every non-agentic-replay timing mode accepts cache-bust with a structured chat endpoint."""
    cfg = _build(
        target=target, endpoint_type=EndpointType.CHAT, timing_mode=timing_mode
    )
    assert cfg.get_cache_bust_target() == target


@pytest.mark.parametrize("timing_mode", _TIMING_MODES)
@pytest.mark.parametrize("target", _RID_CACHE_BUST_TARGETS)
def test_rid_cache_bust_accepted_with_every_timing_mode(
    timing_mode: TimingMode, target: CacheBustTarget
) -> None:
    """RID-based cache-bust targets are accepted with every timing mode."""
    cfg = _build(
        target=target, endpoint_type=EndpointType.CHAT, timing_mode=timing_mode
    )
    assert cfg.get_cache_bust_target() == target


@pytest.mark.parametrize("target", _WARMUP_ISOLATION_TARGETS)
def test_warmup_isolation_rejected_with_agentic_replay(
    target: CacheBustTarget,
) -> None:
    """WARMUP_ISOLATION_* targets are rejected when timing_mode=agentic_replay."""
    with pytest.raises(ValueError, match="not compatible with agentic_replay"):
        _build(
            target=target,
            endpoint_type=EndpointType.CHAT,
            timing_mode=TimingMode.AGENTIC_REPLAY,
        )


@pytest.mark.parametrize("endpoint_type", _INCOMPATIBLE_ENDPOINT_TYPES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_rejected_with_unsupported_endpoint(
    endpoint_type: EndpointType, target: CacheBustTarget
) -> None:
    """Endpoints without the capability metadata reject cache-bust."""
    with pytest.raises(ValueError, match=r"not supported|chat or responses"):
        _build(
            target=target,
            endpoint_type=endpoint_type,
            timing_mode=TimingMode.AGENTIC_REPLAY,
        )


@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
@pytest.mark.parametrize(
    "endpoint_type", [EndpointType.CHAT, EndpointType.RESPONSES, EndpointType.MESSAGES]
)
def test_cache_bust_accepted_with_supported_endpoint(
    endpoint_type: EndpointType, target: CacheBustTarget
) -> None:
    """Every supported structured endpoint accepts every cache-bust target."""
    cfg = _build(
        target=target,
        endpoint_type=endpoint_type,
        timing_mode=TimingMode.REQUEST_RATE,
    )
    assert cfg.get_cache_bust_target() == target


@pytest.mark.parametrize("timing_mode", list(TimingMode))
def test_cache_bust_none_passes_all_timing_modes(timing_mode: TimingMode) -> None:
    """target=NONE never trips construction regardless of timing_mode."""
    cfg = _build(
        target=CacheBustTarget.NONE,
        endpoint_type=EndpointType.CHAT,
        timing_mode=timing_mode,
    )
    assert cfg.get_cache_bust_target() == CacheBustTarget.NONE
    assert cfg.get_profiling_phases()[0].timing_mode == timing_mode


@pytest.mark.parametrize("endpoint_type", list(EndpointType))
def test_cache_bust_none_passes_all_endpoint_types(endpoint_type: EndpointType) -> None:
    """target=NONE never trips construction regardless of endpoint type."""
    cfg = _build(
        target=CacheBustTarget.NONE,
        endpoint_type=endpoint_type,
        timing_mode=TimingMode.AGENTIC_REPLAY,
    )
    assert cfg.get_cache_bust_target() == CacheBustTarget.NONE
    assert cfg.endpoint.type == endpoint_type


def test_unsafe_override_does_not_bypass_cache_bust_endpoint_validation() -> None:
    """unsafe_override does not bypass endpoint capability validation."""
    body: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": {
            "urls": ["http://localhost:8000/v1/embeddings"],
            "type": EndpointType.EMBEDDINGS,
        },
        "unsafe_override": True,
        "datasets": [
            {
                "name": "main",
                "type": "synthetic",
                "prompts": {
                    "isl": 128,
                    "osl": 16,
                    "cache_bust": {"target": CacheBustTarget.SYSTEM_PREFIX},
                },
            }
        ],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 10,
                "timing_mode": TimingMode.REQUEST_RATE,
            }
        ],
    }
    with pytest.raises(ValueError, match=r"not supported|chat or responses"):
        BenchmarkConfig.model_validate(body)
