# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for the cache-bust *compatibility* lockdown."""

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
    """Construct a v2 BenchmarkConfig mirroring the v1 lockdown's inputs."""
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

_NON_AGENTIC_TIMING_MODES: list[TimingMode] = [
    m for m in TimingMode if m != TimingMode.AGENTIC_REPLAY
]

_INCOMPATIBLE_ENDPOINT_TYPES: list[EndpointType] = [
    e for e in EndpointType if e not in {EndpointType.CHAT, EndpointType.RESPONSES}
]


@pytest.mark.parametrize("timing_mode", _NON_AGENTIC_TIMING_MODES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_rejected_with_every_non_agentic_timing_mode(
    timing_mode: TimingMode, target: CacheBustTarget
) -> None:
    """Every non-agentic TimingMode + non-NONE cache_bust SHOULD raise."""
    with pytest.raises(ValueError, match="agentic_replay"):
        _build(
            target=target,
            endpoint_type=EndpointType.CHAT,
            timing_mode=timing_mode,
        )


@pytest.mark.parametrize("endpoint_type", _INCOMPATIBLE_ENDPOINT_TYPES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_rejected_with_every_non_chat_endpoint_type(
    endpoint_type: EndpointType, target: CacheBustTarget
) -> None:
    """Every non-chat/responses endpoint + non-NONE cache_bust SHOULD raise."""
    with pytest.raises(ValueError, match="chat or responses"):
        _build(
            target=target,
            endpoint_type=endpoint_type,
            timing_mode=TimingMode.AGENTIC_REPLAY,
        )


def test_unsafe_override_does_not_bypass_cache_bust_validation() -> None:
    """``unsafe_override`` is a scenario-lock escape hatch; it must NOT bypass"""
    body: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": {"urls": [_URL], "type": EndpointType.CHAT},
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
    with pytest.raises(ValueError, match="agentic_replay"):
        BenchmarkConfig.model_validate(body)


def test_unsafe_override_does_not_bypass_cache_bust_endpoint_validation() -> None:
    """Same as above but for the endpoint-type branch of the lockdown."""
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
                "timing_mode": TimingMode.AGENTIC_REPLAY,
            }
        ],
    }
    with pytest.raises(ValueError, match="chat or responses"):
        BenchmarkConfig.model_validate(body)


@pytest.mark.parametrize("timing_mode", list(TimingMode))
def test_cache_bust_none_passes_all_timing_modes(timing_mode: TimingMode) -> None:
    """target=NONE never trips construction -- regardless of timing_mode."""
    cfg = _build(
        target=CacheBustTarget.NONE,
        endpoint_type=EndpointType.CHAT,
        timing_mode=timing_mode,
    )
    assert cfg.get_cache_bust_target() == CacheBustTarget.NONE
    assert cfg.get_profiling_phases()[0].timing_mode == timing_mode


@pytest.mark.parametrize("endpoint_type", list(EndpointType))
def test_cache_bust_none_passes_all_endpoint_types(
    endpoint_type: EndpointType,
) -> None:
    """target=NONE never trips construction -- regardless of endpoint_type."""
    cfg = _build(
        target=CacheBustTarget.NONE,
        endpoint_type=endpoint_type,
        timing_mode=TimingMode.AGENTIC_REPLAY,
    )
    assert cfg.get_cache_bust_target() == CacheBustTarget.NONE
    assert cfg.endpoint.type == endpoint_type


@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
@pytest.mark.parametrize("endpoint_type", [EndpointType.CHAT, EndpointType.RESPONSES])
def test_cache_bust_all_targets_construct_with_chat_endpoint_and_agentic_replay(
    target: CacheBustTarget, endpoint_type: EndpointType
) -> None:
    """Every non-NONE CacheBustTarget constructs with the *compatible* combo"""
    cfg = _build(
        target=target,
        endpoint_type=endpoint_type,
        timing_mode=TimingMode.AGENTIC_REPLAY,
    )
    assert cfg.get_cache_bust_target() == target
    assert cfg.endpoint.type == endpoint_type
    assert cfg.get_profiling_phases()[0].timing_mode == TimingMode.AGENTIC_REPLAY
