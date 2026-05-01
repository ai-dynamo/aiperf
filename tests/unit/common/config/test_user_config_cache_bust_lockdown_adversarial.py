# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive adversarial coverage for the ``UserConfig.validate_cache_bust_compatibility``
post-init validator.

The validator (``src/aiperf/common/config/user_config.py``) refuses every config where:

- ``input.prompt.cache_bust.target != NONE`` AND ``timing_mode != AGENTIC_REPLAY``, OR
- ``input.prompt.cache_bust.target != NONE`` AND
  ``endpoint.type not in {CHAT, RESPONSES}``.

This is a HARD config-time error (not a scenario-lock soft warning) because either
combination would silently drop the marker — a benchmark that *looks* fine but
exercises no cache-busting at all.

This file complements the basic tests in ``tests/unit/common/config/test_user_config.py``
(owned by the parallel agent). Coverage here parametrizes over EVERY enum value in
``TimingMode`` / ``EndpointType`` / non-NONE ``CacheBustTarget`` so that any new
enum addition is automatically exercised against the validator's allow-list.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aiperf.common.config import (
    EndpointConfig,
    InputConfig,
    LoadGeneratorConfig,
    UserConfig,
)
from aiperf.common.config.prompt_config import CacheBustConfig, PromptConfig
from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario.validator import ValidationOutcome
from aiperf.plugin.enums import EndpointType, TimingMode

# =============================================================================
# Helpers
# =============================================================================


def _endpoint(endpoint_type: EndpointType = EndpointType.CHAT) -> EndpointConfig:
    return EndpointConfig(
        model_names=["test-model"],
        type=endpoint_type,
        custom_endpoint="test",
        streaming=False,
    )


def _input(target: CacheBustTarget) -> InputConfig:
    return InputConfig(
        prompt=PromptConfig(cache_bust=CacheBustConfig(target=target)),
    )


def _force_timing_mode(monkeypatch, mode: TimingMode) -> None:
    """Hijack the scenario validator hook to set ``_timing_mode`` to an arbitrary value.

    ``UserConfig.validate_timing_mode`` runs first and derives ``_timing_mode``
    from loadgen fields. ``_run_scenario_validator`` runs after that and is
    allowed to overwrite ``_timing_mode``. ``validate_cache_bust_compatibility``
    runs last and reads the post-scenario value. Hijacking the scenario hook
    is the cleanest way to exercise every TimingMode without having to
    construct a different loadgen for each one.
    """

    def fake(cfg, **_kwargs):
        # Accept and ignore any kwargs the production caller passes
        # (e.g. ``timing_mode_explicit``) — the test only cares that
        # ``_timing_mode`` ends up at ``mode`` for the cache_bust validator.
        cfg._timing_mode = mode
        return ValidationOutcome(violations=[], submission_valid=True)

    monkeypatch.setattr("aiperf.common.scenario.validator.validate_scenario", fake)


# Non-NONE cache_bust targets the validator should refuse on incompatible configs.
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
    monkeypatch, timing_mode: TimingMode, target: CacheBustTarget
):
    """For every TimingMode that isn't AGENTIC_REPLAY, every non-NONE
    cache_bust target raises ValueError naming ``agentic_replay``.

    Parametrized over the FULL enum so any new TimingMode added in the future
    must explicitly opt in (by becoming AGENTIC_REPLAY) or fail this test.
    """
    _force_timing_mode(monkeypatch, timing_mode)

    with pytest.raises(ValidationError, match="agentic_replay"):
        UserConfig(
            endpoint=_endpoint(EndpointType.CHAT),
            input=_input(target),
            loadgen=LoadGeneratorConfig(concurrency=1, request_count=10),
        )


# =============================================================================
# Rejection: non-chat/responses endpoint types
# =============================================================================


@pytest.mark.parametrize("endpoint_type", _INCOMPATIBLE_ENDPOINT_TYPES)
@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
def test_cache_bust_rejected_with_every_non_chat_endpoint_type(
    monkeypatch, endpoint_type: EndpointType, target: CacheBustTarget
):
    """For every EndpointType that isn't CHAT or RESPONSES, every non-NONE
    cache_bust target raises ValueError naming ``chat or responses``.

    The ``endpoint-type`` substring requirement in the task spec is a fuzzy
    pattern; the exact validator message uses ``--cache-bust requires
    --endpoint-type chat or responses``, so we match the more specific
    ``chat or responses`` substring here. Any rewording that drops both
    "chat" and "responses" would constitute a behavior change worth catching.
    """
    _force_timing_mode(monkeypatch, TimingMode.AGENTIC_REPLAY)

    with pytest.raises(ValidationError, match="chat or responses"):
        UserConfig(
            endpoint=_endpoint(endpoint_type),
            input=_input(target),
            loadgen=LoadGeneratorConfig(concurrency=1, request_count=10),
        )


# =============================================================================
# Allowed: target=NONE always passes
# =============================================================================


@pytest.mark.parametrize("timing_mode", list(TimingMode))
def test_cache_bust_none_passes_all_timing_modes(monkeypatch, timing_mode: TimingMode):
    """target=NONE must never trip the validator — regardless of timing_mode.

    Parametrized over the FULL enum (including AGENTIC_REPLAY, which is the
    happy-case for non-NONE targets but must also be a no-op for NONE).
    """
    _force_timing_mode(monkeypatch, timing_mode)

    cfg = UserConfig(
        endpoint=_endpoint(EndpointType.CHAT),
        input=_input(CacheBustTarget.NONE),
        loadgen=LoadGeneratorConfig(concurrency=1, request_count=10),
    )
    assert cfg.input.prompt.cache_bust.target == CacheBustTarget.NONE
    assert cfg.timing_mode == timing_mode


@pytest.mark.parametrize("endpoint_type", list(EndpointType))
def test_cache_bust_none_passes_all_endpoint_types(
    monkeypatch, endpoint_type: EndpointType
):
    """target=NONE must never trip the validator — regardless of endpoint_type."""
    _force_timing_mode(monkeypatch, TimingMode.AGENTIC_REPLAY)

    cfg = UserConfig(
        endpoint=_endpoint(endpoint_type),
        input=_input(CacheBustTarget.NONE),
        loadgen=LoadGeneratorConfig(concurrency=1, request_count=10),
    )
    assert cfg.input.prompt.cache_bust.target == CacheBustTarget.NONE
    assert cfg.endpoint.type == endpoint_type


# =============================================================================
# Allowed: every non-NONE target with chat + agentic_replay
# =============================================================================


@pytest.mark.parametrize("target", _NON_NONE_CACHE_BUST_TARGETS)
@pytest.mark.parametrize("endpoint_type", [EndpointType.CHAT, EndpointType.RESPONSES])
def test_cache_bust_all_targets_pass_with_chat_endpoint_and_agentic_replay(
    monkeypatch, target: CacheBustTarget, endpoint_type: EndpointType
):
    """Regression: every non-NONE CacheBustTarget passes validation with
    ``timing_mode=AGENTIC_REPLAY`` AND ``endpoint_type in {CHAT, RESPONSES}``.

    Locks the validator's allow-list: if a future change accidentally narrows
    the allowed set (e.g. accepts CHAT but not RESPONSES), this catches it.
    """
    _force_timing_mode(monkeypatch, TimingMode.AGENTIC_REPLAY)

    cfg = UserConfig(
        endpoint=_endpoint(endpoint_type),
        input=_input(target),
        loadgen=LoadGeneratorConfig(concurrency=1, request_count=10),
    )
    assert cfg.input.prompt.cache_bust.target == target
    assert cfg.endpoint.type == endpoint_type
    assert cfg.timing_mode == TimingMode.AGENTIC_REPLAY


# =============================================================================
# unsafe_override does NOT bypass cache_bust validation
# =============================================================================


def test_unsafe_override_does_not_bypass_cache_bust_validation(monkeypatch):
    """``--unsafe-override`` is a scenario-lock escape hatch (downgrades
    scenario violations to warnings). It must NOT bypass
    ``validate_cache_bust_compatibility``: that validator catches
    *fundamentally invalid* combinations (marker would be silently dropped),
    not submission-policy violations.
    """
    _force_timing_mode(monkeypatch, TimingMode.REQUEST_RATE)

    with pytest.raises(ValidationError, match="agentic_replay"):
        UserConfig(
            endpoint=_endpoint(EndpointType.CHAT),
            input=_input(CacheBustTarget.SYSTEM_PREFIX),
            loadgen=LoadGeneratorConfig(concurrency=1, request_count=10),
            unsafe_override=True,
        )


def test_unsafe_override_does_not_bypass_cache_bust_endpoint_validation(monkeypatch):
    """Same idea but for the endpoint-type branch of the validator."""
    _force_timing_mode(monkeypatch, TimingMode.AGENTIC_REPLAY)

    with pytest.raises(ValidationError, match="chat or responses"):
        UserConfig(
            endpoint=_endpoint(EndpointType.EMBEDDINGS),
            input=_input(CacheBustTarget.SYSTEM_PREFIX),
            loadgen=LoadGeneratorConfig(concurrency=1, request_count=10),
            unsafe_override=True,
        )


# =============================================================================
# AgentX-MVP scenario integration
# =============================================================================


def test_inferencex_agentx_mvp_scenario_with_explicit_compatible_settings(monkeypatch):
    """Smoke-test: when ``--scenario inferencex-agentx-mvp`` resolves with
    ``timing_mode=AGENTIC_REPLAY`` (as designed) and the user picks a chat
    endpoint with ``cache_bust=SYSTEM_PREFIX``, the cache_bust validator
    must NOT raise.

    Stubs the scenario validator (same pattern as test_user_config.py) to
    isolate the cache_bust validator from the unrelated scenario invariants
    (loader, benchmark_duration, etc.) — this test exists to prove the
    AgentX-MVP shape passes the *cache_bust* check, not the full scenario
    lock.
    """
    _force_timing_mode(monkeypatch, TimingMode.AGENTIC_REPLAY)

    cfg = UserConfig(
        endpoint=_endpoint(EndpointType.CHAT),
        input=_input(CacheBustTarget.SYSTEM_PREFIX),
        loadgen=LoadGeneratorConfig(concurrency=1, request_count=10),
        scenario="inferencex-agentx-mvp",
    )

    assert cfg.timing_mode == TimingMode.AGENTIC_REPLAY
    assert cfg.input.prompt.cache_bust.target == CacheBustTarget.SYSTEM_PREFIX
    assert cfg.endpoint.type == EndpointType.CHAT
    assert cfg.scenario == "inferencex-agentx-mvp"
    assert cfg._scenario_outcome is not None
