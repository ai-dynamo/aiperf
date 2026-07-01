# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import _build_profiling_config, _build_warmup_config


def _ar_user_config(
    concurrency: int = 10,
    duration: float = 900,
    cap: float = 60.0,
    benchmark_grace_period: float | None = None,
) -> MagicMock:
    cfg = MagicMock()
    cfg.timing_mode = TimingMode.AGENTIC_REPLAY
    cfg.loadgen.concurrency = concurrency
    cfg.loadgen.benchmark_duration = duration
    cfg.loadgen.inter_turn_delay_cap_seconds = cap
    cfg.loadgen.warmup_request_count = None
    cfg.loadgen.warmup_duration = None
    cfg.loadgen.agentic_cache_warmup_duration = None
    cfg.loadgen.warmup_num_sessions = None
    cfg.loadgen.warmup_concurrency = None
    cfg.loadgen.warmup_prefill_concurrency = None
    cfg.loadgen.warmup_arrival_pattern = None
    cfg.loadgen.warmup_request_rate = None
    cfg.loadgen.warmup_grace_period = None
    cfg.loadgen.warmup_concurrency_ramp_duration = None
    cfg.loadgen.warmup_prefill_concurrency_ramp_duration = None
    cfg.loadgen.warmup_request_rate_ramp_duration = None
    cfg.loadgen.request_count = None
    cfg.loadgen.request_rate = None
    cfg.loadgen.arrival_pattern = ArrivalPattern.CONCURRENCY_BURST
    cfg.loadgen.arrival_smoothness = None
    cfg.loadgen.concurrency_ramp_duration = None
    cfg.loadgen.prefill_concurrency = None
    cfg.loadgen.prefill_concurrency_ramp_duration = None
    cfg.loadgen.request_rate_ramp_duration = None
    cfg.loadgen.user_centric_rate = None
    cfg.loadgen.benchmark_grace_period = benchmark_grace_period
    cfg.loadgen.num_users = None
    cfg.input.conversation.num = None
    cfg.input.fixed_schedule_auto_offset = False
    cfg.input.fixed_schedule_start_offset = None
    cfg.input.fixed_schedule_end_offset = None
    return cfg


def test_warmup_config_uses_agentic_replay_when_top_level_is_agentic_replay() -> None:
    cfg = _ar_user_config()
    warmup = _build_warmup_config(cfg)
    assert warmup is not None
    assert warmup.timing_mode == TimingMode.AGENTIC_REPLAY
    assert warmup.phase == CreditPhase.WARMUP


def test_profiling_config_propagates_cap() -> None:
    cfg = _ar_user_config(cap=60.0)
    profiling = _build_profiling_config(cfg)
    assert profiling.timing_mode == TimingMode.AGENTIC_REPLAY
    assert profiling.phase == CreditPhase.PROFILING


# =============================================================================
# Warmup phase termination via total_expected_requests
# =============================================================================
#
# ``credit_counter.is_final_credit`` requires either ``total_expected_requests``
# or ``expected_num_sessions`` to be non-None for ``SendingCompleteStopCondition``
# to fire. ``_build_warmup_config`` sets ``total_expected_requests = loadgen.concurrency``
# (the warmup burst size) so the warmup barrier releases after the burst lands.


def test_warmup_config_total_expected_requests_set() -> None:
    """Warmup config has a non-None ``total_expected_requests`` so
    ``SendingCompleteStopCondition`` can fire."""
    cfg = _ar_user_config(concurrency=10)
    warmup = _build_warmup_config(cfg)
    assert warmup is not None
    assert warmup.total_expected_requests is not None
    assert warmup.total_expected_requests == 10


def test_warmup_config_total_expected_requests_tracks_concurrency() -> None:
    """The count target matches ``loadgen.concurrency`` (the cohort burst
    size in the common case)."""
    for concurrency in (1, 7, 64):
        cfg = _ar_user_config(concurrency=concurrency)
        warmup = _build_warmup_config(cfg)
        assert warmup is not None
        assert warmup.total_expected_requests == concurrency


def test_cache_warmup_uses_strategy_controlled_stop() -> None:
    cfg = _ar_user_config(concurrency=10, benchmark_grace_period=30.0)
    cfg.loadgen.agentic_cache_warmup_duration = 600.0

    warmup = _build_warmup_config(cfg)

    assert warmup is not None
    assert warmup.total_expected_requests is None
    assert warmup.agentic_cache_warmup_duration_sec == 600.0
    assert warmup.grace_period_sec == 300.0


def test_cache_warmup_grace_uses_short_duration_without_benchmark_grace() -> None:
    cfg = _ar_user_config(concurrency=10, benchmark_grace_period=None)
    cfg.loadgen.agentic_cache_warmup_duration = 2.0

    warmup = _build_warmup_config(cfg)

    assert warmup is not None
    assert warmup.grace_period_sec == 2.0


def test_cache_warmup_grace_keeps_larger_benchmark_grace() -> None:
    cfg = _ar_user_config(concurrency=10, benchmark_grace_period=30.0)
    cfg.loadgen.agentic_cache_warmup_duration = 2.0

    warmup = _build_warmup_config(cfg)

    assert warmup is not None
    assert warmup.grace_period_sec == 30.0


def test_cache_warmup_explicit_warmup_grace_overrides_benchmark_grace() -> None:
    cfg = _ar_user_config(concurrency=10, benchmark_grace_period=30.0)
    cfg.loadgen.agentic_cache_warmup_duration = 600.0
    cfg.loadgen.warmup_grace_period = 7.0

    warmup = _build_warmup_config(cfg)

    assert warmup is not None
    assert warmup.grace_period_sec == 7.0
