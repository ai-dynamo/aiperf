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
    """Construct a v2 BenchmarkConfig for an endpoint capability combination.

    warmup_isolation_system requires a shared system prompt; this helper adds
    one automatically so callers testing timing-mode or endpoint compatibility
    don't need to repeat that setup.
    """
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

    dataset: dict[str, Any] = {
        "name": "main",
        "type": "synthetic",
        "prompts": {
            "isl": 128,
            "osl": 16,
            "cache_bust": {"target": target},
        },
    }
    if target == CacheBustTarget.WARMUP_ISOLATION_SYSTEM:
        dataset["prefix_prompts"] = {"shared_system_length": 512}

    body: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": endpoint,
        "datasets": [dataset],
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
_WARMUP_ISOLATION_INCOMPATIBLE_TIMING_MODES: list[TimingMode] = [
    TimingMode.AGENTIC_REPLAY,
    TimingMode.AGENT_GRAPH,
]
_WARMUP_ISOLATION_COMPATIBLE_TIMING_MODES: list[TimingMode] = [
    t for t in TimingMode if t not in _WARMUP_ISOLATION_INCOMPATIBLE_TIMING_MODES
]
_INCOMPATIBLE_ENDPOINT_TYPES: list[EndpointType] = [
    e
    for e in EndpointType
    if e not in {EndpointType.CHAT, EndpointType.RESPONSES, EndpointType.MESSAGES}
]


@pytest.mark.parametrize("timing_mode", _WARMUP_ISOLATION_COMPATIBLE_TIMING_MODES)
@pytest.mark.parametrize("target", _WARMUP_ISOLATION_TARGETS)
def test_warmup_isolation_accepted_with_compatible_timing_mode(
    timing_mode: TimingMode, target: CacheBustTarget
) -> None:
    """Session-isolated timing modes accept warmup-isolation targets."""
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


@pytest.mark.parametrize("timing_mode", _WARMUP_ISOLATION_INCOMPATIBLE_TIMING_MODES)
@pytest.mark.parametrize("target", _WARMUP_ISOLATION_TARGETS)
def test_warmup_isolation_rejected_with_incompatible_replay_mode(
    target: CacheBustTarget, timing_mode: TimingMode
) -> None:
    """Replay modes that cannot isolate warmup payloads reject these targets."""
    with pytest.raises(ValueError, match="not compatible with"):
        _build(
            target=target,
            endpoint_type=EndpointType.CHAT,
            timing_mode=timing_mode,
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


def _build_with_prefix(
    *,
    target: CacheBustTarget,
    shared_system_length: int | None = None,
    user_context_length: int | None = None,
    pool_size: int | None = None,
    prefix_length: int | None = None,
) -> BenchmarkConfig:
    """Build a BenchmarkConfig with prefix_prompts configured."""
    prefix_prompts: dict[str, Any] = {}
    if shared_system_length is not None:
        prefix_prompts["shared_system_length"] = shared_system_length
    if user_context_length is not None:
        prefix_prompts["user_context_length"] = user_context_length
    if pool_size is not None:
        prefix_prompts["pool_size"] = pool_size
    if prefix_length is not None:
        prefix_prompts["length"] = prefix_length

    dataset: dict[str, Any] = {
        "name": "main",
        "type": "synthetic",
        "prompts": {
            "isl": 128,
            "osl": 16,
            "cache_bust": {"target": target},
        },
    }
    if prefix_prompts:
        dataset["prefix_prompts"] = prefix_prompts

    body: dict[str, Any] = {
        "models": ["test-model"],
        "endpoint": {"urls": [_URL], "type": EndpointType.CHAT},
        "datasets": [dataset],
        "phases": [
            {
                "name": "profiling",
                "type": "concurrency",
                "concurrency": 1,
                "requests": 10,
            }
        ],
    }
    return BenchmarkConfig.model_validate(body)


def test_warmup_isolation_system_without_shared_system_prompt_rejected() -> None:
    """warmup_isolation_system is rejected when no shared_system_length is configured."""
    with pytest.raises(ValueError, match="requires a shared system prompt"):
        _build_with_prefix(target=CacheBustTarget.WARMUP_ISOLATION_SYSTEM)


def test_warmup_isolation_system_with_user_context_only_rejected() -> None:
    """warmup_isolation_system is rejected when only user_context_length is set (no system msg)."""
    with pytest.raises(ValueError, match="requires a shared system prompt"):
        _build_with_prefix(
            target=CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
            user_context_length=256,
        )


def test_warmup_isolation_system_with_prefix_pool_only_rejected() -> None:
    """warmup_isolation_system is rejected when only a prefix pool is configured (no system msg)."""
    with pytest.raises(ValueError, match="requires a shared system prompt"):
        _build_with_prefix(
            target=CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
            pool_size=4,
            prefix_length=256,
        )


def test_warmup_isolation_system_with_shared_system_prompt_accepted() -> None:
    """warmup_isolation_system is accepted when shared_system_length is set."""
    cfg = _build_with_prefix(
        target=CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
        shared_system_length=512,
    )
    assert cfg.get_cache_bust_target() == CacheBustTarget.WARMUP_ISOLATION_SYSTEM


def test_warmup_isolation_first_turn_without_shared_system_prompt_accepted() -> None:
    """warmup_isolation_first_turn is accepted with any prefix config (targets user turn)."""
    cfg = _build_with_prefix(
        target=CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
        user_context_length=256,
    )
    assert cfg.get_cache_bust_target() == CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN


def test_warmup_isolation_first_turn_with_no_prefix_accepted() -> None:
    """warmup_isolation_first_turn is accepted even without any prefix config."""
    cfg = _build_with_prefix(target=CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN)
    assert cfg.get_cache_bust_target() == CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN


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
