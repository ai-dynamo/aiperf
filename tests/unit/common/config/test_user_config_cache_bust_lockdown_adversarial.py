# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cache-bust compatibility lockdown rejects a non-NONE cache_bust target paired with a non-agentic timing mode or a non-chat/responses endpoint."""

from __future__ import annotations

from typing import Any

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.config.config import BenchmarkConfig
from aiperf.plugin.enums import EndpointType, TimingMode

# Endpoint URL paths are cosmetic here; the lockdown only cares about the
# endpoint type and the cache_bust target/timing combination.
_URL = "http://localhost:8000/v1/chat/completions"


def _build(
    *,
    target: CacheBustTarget,
    endpoint_type: EndpointType = EndpointType.CHAT,
    timing_mode: TimingMode | None = None,
) -> BenchmarkConfig:
    """Construct a v2 BenchmarkConfig from a (cache_bust target, endpoint type, timing_mode) trio."""
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
        # Template endpoint needs a payload template to construct at all.
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

_TIMING_MODES: list[TimingMode] = list(TimingMode)

_INCOMPATIBLE_ENDPOINT_TYPES: list[EndpointType] = [
    e for e in EndpointType if e not in {EndpointType.CHAT, EndpointType.RESPONSES}
]


@pytest.mark.parametrize("timing_mode", _TIMING_MODES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_accepted_with_every_timing_mode(
    timing_mode: TimingMode, target: CacheBustTarget
) -> None:
    """Every timing mode accepts cache-bust with a structured chat endpoint."""
    cfg = _build(
        target=target,
        endpoint_type=EndpointType.CHAT,
        timing_mode=timing_mode,
    )
    assert cfg.get_cache_bust_target() == target


@pytest.mark.parametrize("endpoint_type", _INCOMPATIBLE_ENDPOINT_TYPES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_rejected_with_every_non_chat_endpoint_type(
    endpoint_type: EndpointType, target: CacheBustTarget
) -> None:
    """Every non-chat/responses endpoint paired with a non-NONE cache_bust target raises."""
    with pytest.raises(ValueError, match="not supported|chat or responses"):
        _build(
            target=target,
            endpoint_type=endpoint_type,
            timing_mode=TimingMode.AGENTIC_REPLAY,
        )


def test_unsafe_override_does_not_bypass_cache_bust_validation() -> None:
    """``unsafe_override`` is a scenario-lock escape hatch and does not bypass the cache-bust compatibility lockdown."""
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
    cfg = BenchmarkConfig.model_validate(body)
    assert cfg.get_cache_bust_target() == CacheBustTarget.SYSTEM_PREFIX


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
    with pytest.raises(ValueError, match="not supported|chat or responses"):
        BenchmarkConfig.model_validate(body)


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
    """Every non-NONE CacheBustTarget constructs with the compatible agentic_replay + chat/responses combo."""
    cfg = _build(
        target=target,
        endpoint_type=endpoint_type,
        timing_mode=TimingMode.AGENTIC_REPLAY,
    )
    assert cfg.get_cache_bust_target() == target
    assert cfg.endpoint.type == endpoint_type
    assert cfg.get_profiling_phases()[0].timing_mode == TimingMode.AGENTIC_REPLAY
