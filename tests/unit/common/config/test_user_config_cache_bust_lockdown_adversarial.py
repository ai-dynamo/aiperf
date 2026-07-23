# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for the cache-bust *compatibility* lockdown.

REBASED from the v1 ``UserConfig.validate_cache_bust_compatibility`` suite onto
the v2 ``BenchmarkConfig.model_validate(body)`` construction model.

The lockdown (``BenchmarkConfig.validate_cache_bust_compatibility``) refuses
every config where a non-NONE cache-bust target is paired with an incompatible
profiling timing mode or endpoint type::

    cache_bust.target != NONE AND profiling timing_mode is explicit & not AGENTIC_REPLAY
    cache_bust.target != NONE AND endpoint.type not in {CHAT, RESPONSES}

Either combination silently drops the cache-bust marker (a benchmark that
*looks* configured for cache-busting but exercises none), so it is a HARD
config-time error. The validator is a pure-validation ``model_validator`` (it
RAISES only, never mutates) and DEFERS in two cases: when a ``scenario`` governs
the config (the scenario's own locks stamp the agentic timing_mode + cache_bust
POST-construction) and when no profiling phase carries an EXPLICIT timing_mode
(the effective mode is then runtime-derived).

The scenario-lock SIBLING of this validator (``require_cache_bust`` /
``timing_mode`` locks applied by ``apply_scenario``) is covered by
``tests/unit/common/scenario/test_scenario_validator.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.config.config import BenchmarkConfig
from aiperf.plugin.enums import EndpointType, TimingMode

# =============================================================================
# Helpers
# =============================================================================

# Endpoint URL paths are cosmetic for these tests; the lockdown only cared about
# the endpoint *type* and the cache_bust target/timing combination.
_URL = "http://localhost:8000/v1/chat/completions"


def _build(
    *,
    target: CacheBustTarget,
    endpoint_type: EndpointType = EndpointType.CHAT,
    timing_mode: TimingMode | None = None,
) -> BenchmarkConfig:
    """Construct a v2 BenchmarkConfig mirroring the v1 lockdown's inputs.

    The v1 trio (cache_bust target, timing_mode, endpoint type) maps to:

    - ``target`` -> ``datasets[0].prompts.cache_bust.target`` (synthetic);
    - ``endpoint_type`` -> ``endpoint.type``;
    - ``timing_mode`` -> the profiling phase's ``timing_mode`` override (v2 has
      no top-level timing_mode; ``None`` leaves the phase non-agentic, the
      credit pipeline derives it from ``phase.type``).
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
        # The template endpoint requires a payload template -- unrelated to the
        # cache-bust lockdown, but needed for the model to construct at all.
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


# Non-NONE cache_bust targets the (missing) validator should refuse on
# incompatible configs.
_NON_NONE_CACHE_BUST_TARGETS: list[CacheBustTarget] = [
    t for t in CacheBustTarget if t != CacheBustTarget.NONE
]

# Every TimingMode that ISN'T agentic_replay (the only mode that mints markers).
_NON_AGENTIC_TIMING_MODES: list[TimingMode] = [
    m for m in TimingMode if m != TimingMode.AGENTIC_REPLAY
]

# Every EndpointType that ISN'T chat or responses (the only formatters that
# consume the system message field that hosts the marker).
_INCOMPATIBLE_ENDPOINT_TYPES: list[EndpointType] = [
    e for e in EndpointType if e not in {EndpointType.CHAT, EndpointType.RESPONSES}
]

# =============================================================================
# Rejection: non-agentic timing modes
# =============================================================================


@pytest.mark.parametrize("timing_mode", _NON_AGENTIC_TIMING_MODES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_rejected_with_every_non_agentic_timing_mode(
    timing_mode: TimingMode, target: CacheBustTarget
) -> None:
    """Every non-agentic TimingMode + non-NONE cache_bust SHOULD raise.

    Parametrized over the FULL enum so any new TimingMode is exercised. The
    validation is not implemented yet, so this remains a strict xfail.
    """
    with pytest.raises(ValueError, match="agentic_replay"):
        _build(
            target=target,
            endpoint_type=EndpointType.CHAT,
            timing_mode=timing_mode,
        )


# =============================================================================
# Rejection: non-chat/responses endpoint types (strict xfail)
# =============================================================================


@pytest.mark.parametrize("endpoint_type", _INCOMPATIBLE_ENDPOINT_TYPES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_rejected_with_every_non_chat_endpoint_type(
    endpoint_type: EndpointType, target: CacheBustTarget
) -> None:
    """Every non-chat/responses endpoint + non-NONE cache_bust SHOULD raise.

    The validation is not implemented yet, so this remains a strict xfail.
    Endpoint type drives whether the system-message marker slot exists at all.
    """
    with pytest.raises(ValueError, match="chat or responses"):
        _build(
            target=target,
            endpoint_type=endpoint_type,
            timing_mode=TimingMode.AGENTIC_REPLAY,
        )


# =============================================================================
# Rejection: unsafe_override does NOT bypass the (missing) lockdown
# =============================================================================


def test_unsafe_override_does_not_bypass_cache_bust_validation() -> None:
    """``unsafe_override`` is a scenario-lock escape hatch; it must NOT bypass
    the cache-bust *compatibility* lockdown (fundamentally invalid combos, not
    submission-policy violations). v2 has no lockdown, so strict xfail."""
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


# =============================================================================
# Allowed: target=NONE always constructs  (REAL v2 behavior -> PASS)
# =============================================================================


@pytest.mark.parametrize("timing_mode", list(TimingMode))
def test_cache_bust_none_passes_all_timing_modes(timing_mode: TimingMode) -> None:
    """target=NONE never trips construction -- regardless of timing_mode.

    Parametrized over the FULL enum (including AGENTIC_REPLAY). This is real v2
    behavior and must pass.
    """
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


# =============================================================================
# Allowed: every non-NONE target with chat + agentic_replay  (PASS)
# =============================================================================


@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
@pytest.mark.parametrize("endpoint_type", [EndpointType.CHAT, EndpointType.RESPONSES])
def test_cache_bust_all_targets_construct_with_chat_endpoint_and_agentic_replay(
    target: CacheBustTarget, endpoint_type: EndpointType
) -> None:
    """Every non-NONE CacheBustTarget constructs with the *compatible* combo
    (agentic_replay + chat/responses) and carries the expected target.

    This is the configuration the v1 validator explicitly allowed; on v2 it is
    plain construction and must pass.
    """
    cfg = _build(
        target=target,
        endpoint_type=endpoint_type,
        timing_mode=TimingMode.AGENTIC_REPLAY,
    )
    assert cfg.get_cache_bust_target() == target
    assert cfg.endpoint.type == endpoint_type
    assert cfg.get_profiling_phases()[0].timing_mode == TimingMode.AGENTIC_REPLAY
