# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AGENTIC_REPLAY warmup/profiling CreditPhaseConfig construction."""

import pydantic

from aiperf.common.enums import CreditPhase
from aiperf.config.phases import PhaseConfig
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import (
    _build_agentic_warmup_config,
    _build_profiling_config,
)
from aiperf.timing.request_cancellation import RequestCancellationConfig

_PHASE_ADAPTER = pydantic.TypeAdapter(PhaseConfig)


def _ar_profiling_phase(concurrency: int = 10, duration: float = 900) -> PhaseConfig:
    """Build the AGENTIC_REPLAY profiling phase the auto-warmup is sized from.

    The agentic scenario lock (P2) stamps ``timing_mode=AGENTIC_REPLAY`` onto
    the profiling phase; mirror that here so ``_build_agentic_warmup_config``
    and ``_build_profiling_config`` route through the agentic path.
    """
    return _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": concurrency,
            "duration": duration,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
        }
    )


def test_warmup_config_uses_agentic_replay_when_top_level_is_agentic_replay() -> None:
    phase = _ar_profiling_phase()
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None
    assert warmup.timing_mode == TimingMode.AGENTIC_REPLAY
    assert warmup.phase == CreditPhase.WARMUP


def test_profiling_config_propagates_cap() -> None:
    phase = _ar_profiling_phase()
    profiling = _build_profiling_config(
        phase,
        default_cancellation=RequestCancellationConfig(),
        phase_index=0,
        profiling_index=0,
    )
    assert profiling.timing_mode == TimingMode.AGENTIC_REPLAY
    assert profiling.phase == CreditPhase.PROFILING


# =============================================================================
# Warmup phase termination via total_expected_requests
# =============================================================================
#
# ``credit_counter.is_final_credit`` requires either ``total_expected_requests``
# or ``expected_num_sessions`` to be non-None for the sending-complete stop
# condition to fire. ``_build_agentic_warmup_config`` sets
# ``total_expected_requests = phase.concurrency`` (the warmup burst size) so the
# warmup barrier releases after the burst lands.


def test_warmup_config_total_expected_requests_set() -> None:
    """Warmup config has a non-None ``total_expected_requests`` so the
    sending-complete stop condition can fire."""
    phase = _ar_profiling_phase(concurrency=10)
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None
    assert warmup.total_expected_requests is not None
    assert warmup.total_expected_requests == 10


def test_warmup_config_total_expected_requests_tracks_concurrency() -> None:
    """The count target matches ``concurrency`` (the burst size in the common
    case)."""
    for concurrency in (1, 7, 64):
        phase = _ar_profiling_phase(concurrency=concurrency)
        warmup = _build_agentic_warmup_config(phase)
        assert warmup is not None
        assert warmup.total_expected_requests == concurrency


def test_warmup_grace_defaults_to_infinity() -> None:
    """With no ``agentic_warmup_grace_period`` set, the warmup barrier waits
    indefinitely (inf) until every primed trajectory returns."""
    phase = _ar_profiling_phase()
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None
    assert warmup.grace_period_sec == float("inf")


def test_warmup_grace_uses_agentic_warmup_grace_period() -> None:
    """The agentic warmup barrier grace comes from
    ``agentic_warmup_grace_period`` (the ``--agentic-warmup-grace-period``
    knob), not from the profiling phase's own ``grace_period``."""
    phase = _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 10,
            "duration": 900,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
            "agentic_warmup_grace_period": 30.0,
        }
    )
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None
    assert warmup.grace_period_sec == 30.0


def test_warmup_grace_ignores_profiling_grace_period() -> None:
    """The profiling phase's own ``grace_period`` (the profiling tail) must NOT
    leak into the agentic warmup barrier. Only ``agentic_warmup_grace_period``
    feeds the warmup grace; absent it, the barrier stays infinite even when the
    profiling phase declares a finite grace."""
    phase = _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 10,
            "duration": 900,
            "grace_period": 45.0,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
        }
    )
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None
    assert warmup.grace_period_sec == float("inf")


def test_warmup_grace_zero_is_honored() -> None:
    """A zero grace is a real value (drain immediately), distinct from unset
    (wait forever)."""
    phase = _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 10,
            "duration": 900,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
            "agentic_warmup_grace_period": 0.0,
        }
    )
    warmup = _build_agentic_warmup_config(phase)
    assert warmup is not None
    assert warmup.grace_period_sec == 0.0


def test_cache_warmup_uses_strategy_controlled_stop() -> None:
    """With --agentic-cache-warmup-duration set, the warmup phase is
    strategy-terminated: the concurrency-sized request cap is dropped (the
    strategy emits ``mark_sending_complete`` when the duration elapses), the
    duration is threaded onto the credit-phase config, and the drain is bounded
    by max(benchmark grace period, min(cache_warmup_duration, 300s)) instead of
    the infinite snapshot-warmup barrier."""
    phase = _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 10,
            "duration": 900,
            "grace_period": 30.0,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
            "agentic_cache_warmup_duration": 600.0,
        }
    )

    warmup = _build_agentic_warmup_config(phase)

    assert warmup is not None
    assert warmup.total_expected_requests is None
    assert warmup.agentic_cache_warmup_duration_sec == 600.0
    assert warmup.grace_period_sec == 300.0


def test_cache_warmup_grace_uses_short_duration_without_benchmark_grace() -> None:
    phase = _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 10,
            "duration": 900,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
            "agentic_cache_warmup_duration": 2.0,
        }
    )

    warmup = _build_agentic_warmup_config(phase)

    assert warmup is not None
    assert warmup.grace_period_sec == 2.0


def test_cache_warmup_grace_keeps_larger_benchmark_grace() -> None:
    phase = _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 10,
            "duration": 900,
            "grace_period": 30.0,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
            "agentic_cache_warmup_duration": 2.0,
        }
    )

    warmup = _build_agentic_warmup_config(phase)

    assert warmup is not None
    assert warmup.grace_period_sec == 30.0


def test_cache_warmup_explicit_warmup_grace_overrides_benchmark_grace() -> None:
    phase = _PHASE_ADAPTER.validate_python(
        {
            "name": "profiling",
            "type": "concurrency",
            "concurrency": 10,
            "duration": 900,
            "grace_period": 30.0,
            "timing_mode": TimingMode.AGENTIC_REPLAY,
            "agentic_cache_warmup_duration": 600.0,
            "agentic_warmup_grace_period": 7.0,
        }
    )

    warmup = _build_agentic_warmup_config(phase)

    assert warmup is not None
    assert warmup.grace_period_sec == 7.0
