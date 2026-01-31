# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiperf.common.config import UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import ArrivalPattern, TimingMode, URLSelectionStrategy
from aiperf.timing.config import (
    CreditPhaseConfig,
    RequestCancellationConfig,
    TimingConfig,
)


def make_phase_config(**overrides) -> CreditPhaseConfig:
    defaults = {"phase": CreditPhase.PROFILING, "timing_mode": TimingMode.REQUEST_RATE}
    defaults.update(overrides)
    return CreditPhaseConfig(**defaults)


def make_user_config(**overrides) -> UserConfig:
    loadgen = MagicMock()
    loadgen.concurrency = overrides.get("concurrency", 10)
    loadgen.prefill_concurrency = overrides.get("prefill_concurrency")
    loadgen.request_rate = overrides.get("request_rate", 10.0)
    loadgen.user_centric_rate = overrides.get("user_centric_rate")
    loadgen.arrival_pattern = overrides.get("arrival_pattern", ArrivalPattern.POISSON)
    loadgen.request_count = overrides.get("request_count", 100)
    loadgen.num_users = overrides.get("num_users")
    loadgen.warmup_request_count = overrides.get("warmup_request_count")
    loadgen.warmup_duration = overrides.get("warmup_duration")
    loadgen.warmup_num_sessions = overrides.get("warmup_num_sessions")
    loadgen.warmup_concurrency = overrides.get("warmup_concurrency")
    loadgen.warmup_prefill_concurrency = overrides.get("warmup_prefill_concurrency")
    loadgen.warmup_request_rate = overrides.get("warmup_request_rate")
    loadgen.warmup_rate_mode = overrides.get("warmup_rate_mode")
    loadgen.warmup_arrival_pattern = overrides.get(
        "warmup_arrival_pattern", ArrivalPattern.CONSTANT
    )
    loadgen.warmup_concurrency_ramp_duration = overrides.get(
        "warmup_concurrency_ramp_duration"
    )
    loadgen.warmup_prefill_concurrency_ramp_duration = overrides.get(
        "warmup_prefill_concurrency_ramp_duration"
    )
    loadgen.warmup_request_rate_ramp_duration = overrides.get(
        "warmup_request_rate_ramp_duration"
    )
    loadgen.warmup_grace_period = overrides.get("warmup_grace_period")
    loadgen.benchmark_duration = overrides.get("benchmark_duration")
    loadgen.benchmark_grace_period = overrides.get("benchmark_grace_period", 30.0)
    loadgen.request_cancellation_rate = overrides.get("request_cancellation_rate")
    loadgen.request_cancellation_delay = overrides.get(
        "request_cancellation_delay", 0.0
    )
    loadgen.concurrency_ramp_duration = overrides.get("concurrency_ramp_duration")
    loadgen.prefill_concurrency_ramp_duration = overrides.get(
        "prefill_concurrency_ramp_duration"
    )
    loadgen.request_rate_ramp_duration = overrides.get("request_rate_ramp_duration")
    loadgen.arrival_smoothness = overrides.get("arrival_smoothness")
    input_config = MagicMock()
    input_config.random_seed = overrides.get("random_seed")
    input_config.fixed_schedule_auto_offset = overrides.get(
        "fixed_schedule_auto_offset", True
    )
    input_config.fixed_schedule_start_offset = overrides.get(
        "fixed_schedule_start_offset"
    )
    input_config.fixed_schedule_end_offset = overrides.get("fixed_schedule_end_offset")
    input_config.conversation = MagicMock()
    input_config.conversation.num = overrides.get("num_sessions")
    endpoint_config = MagicMock()
    endpoint_config.urls = overrides.get("urls", ["localhost:8000"])
    endpoint_config.url_selection_strategy = overrides.get(
        "url_selection_strategy", URLSelectionStrategy.ROUND_ROBIN
    )
    user_config = MagicMock(spec=UserConfig)
    user_config.timing_mode = overrides.get("timing_mode", TimingMode.REQUEST_RATE)
    user_config.loadgen = loadgen
    user_config.input = input_config
    user_config.endpoint = endpoint_config
    return user_config


class TestTimingConfig:
    def test_minimal_request_rate_config(self) -> None:
        cfg = TimingConfig(phase_configs=[make_phase_config()])
        assert len(cfg.phase_configs) == 1
        pc = cfg.phase_configs[0]
        assert pc.timing_mode == TimingMode.REQUEST_RATE
        assert pc.concurrency is None
        assert pc.request_rate is None

    def test_full_request_rate_config(self) -> None:
        pc = make_phase_config(
            concurrency=10,
            prefill_concurrency=5,
            request_rate=100.0,
            arrival_pattern=ArrivalPattern.CONSTANT,
            total_expected_requests=1000,
        )
        cfg = TimingConfig(phase_configs=[pc])
        p = cfg.phase_configs[0]
        assert (p.timing_mode, p.concurrency, p.prefill_concurrency) == (
            TimingMode.REQUEST_RATE,
            10,
            5,
        )
        assert (p.request_rate, p.arrival_pattern, p.total_expected_requests) == (
            100.0,
            ArrivalPattern.CONSTANT,
            1000,
        )

    def test_fixed_schedule_config(self) -> None:
        pc = make_phase_config(
            timing_mode=TimingMode.FIXED_SCHEDULE,
            auto_offset_timestamps=True,
            fixed_schedule_start_offset=1000,
            fixed_schedule_end_offset=5000,
        )
        cfg = TimingConfig(phase_configs=[pc])
        p = cfg.phase_configs[0]
        assert p.timing_mode == TimingMode.FIXED_SCHEDULE
        assert (
            p.auto_offset_timestamps,
            p.fixed_schedule_start_offset,
            p.fixed_schedule_end_offset,
        ) == (True, 1000, 5000)

    def test_user_centric_config(self) -> None:
        pc = make_phase_config(
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=10.0,
            concurrency=5,
            expected_num_sessions=100,
        )
        cfg = TimingConfig(phase_configs=[pc])
        p = cfg.phase_configs[0]
        assert (
            p.timing_mode,
            p.request_rate,
            p.concurrency,
            p.expected_num_sessions,
        ) == (TimingMode.USER_CENTRIC_RATE, 10.0, 5, 100)

    def test_cancellation_config(self) -> None:
        cfg = TimingConfig(
            phase_configs=[make_phase_config()],
            request_cancellation=RequestCancellationConfig(rate=50.0, delay=2.5),
        )
        assert (cfg.request_cancellation.rate, cfg.request_cancellation.delay) == (
            50.0,
            2.5,
        )

    def test_zero_values_allowed_for_ge0_fields(self) -> None:
        pc = make_phase_config(
            fixed_schedule_start_offset=0, fixed_schedule_end_offset=0
        )
        cfg = TimingConfig(
            phase_configs=[pc],
            request_cancellation=RequestCancellationConfig(rate=0.0, delay=0.0),
        )
        assert pc.fixed_schedule_start_offset == 0
        assert pc.fixed_schedule_end_offset == 0
        assert cfg.request_cancellation.rate == 0.0
        assert cfg.request_cancellation.delay == 0.0

    @pytest.mark.parametrize(
        "field,value",
        [("concurrency", 0), ("concurrency", -1), ("prefill_concurrency", 0), ("prefill_concurrency", -1)],
    )  # fmt: skip
    def test_ge1_fields_reject_zero_and_negative(self, field: str, value: int) -> None:
        with pytest.raises(ValidationError) as exc_info:
            make_phase_config(**{field: value})
        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == (field,)
        assert "greater than" in errors[0]["msg"]

    def test_config_is_frozen(self) -> None:
        cfg = TimingConfig(phase_configs=[make_phase_config()])
        with pytest.raises(ValidationError):
            cfg.request_cancellation = RequestCancellationConfig(rate=50.0)

    def test_phase_config_is_hashable(self) -> None:
        pc = make_phase_config()
        assert {pc: "value"}[pc] == "value"


class TestTimingConfigFromUserConfig:
    def test_maps_timing_mode(self) -> None:
        cfg = TimingConfig.from_user_config(
            make_user_config(timing_mode=TimingMode.FIXED_SCHEDULE)
        )
        profiling = next(
            pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING
        )
        assert profiling.timing_mode == TimingMode.FIXED_SCHEDULE

    def test_maps_loadgen_fields(self) -> None:
        cfg = TimingConfig.from_user_config(
            make_user_config(
                concurrency=8,
                prefill_concurrency=4,
                request_rate=50.0,
                request_count=500,
            )
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING)
        assert (
            p.concurrency,
            p.prefill_concurrency,
            p.request_rate,
            p.total_expected_requests,
        ) == (8, 4, 50.0, 500)

    def test_creates_warmup_when_configured(self) -> None:
        cfg = TimingConfig.from_user_config(make_user_config(warmup_request_count=25))
        phases = [pc.phase for pc in cfg.phase_configs]
        assert CreditPhase.WARMUP in phases
        assert cfg.phase_configs[0].phase == CreditPhase.WARMUP

    def test_no_warmup_when_not_configured(self) -> None:
        cfg = TimingConfig.from_user_config(make_user_config())
        phases = [pc.phase for pc in cfg.phase_configs]
        assert CreditPhase.WARMUP not in phases
        assert len(cfg.phase_configs) == 1

    def test_maps_fixed_schedule_fields(self) -> None:
        cfg = TimingConfig.from_user_config(
            make_user_config(
                timing_mode=TimingMode.FIXED_SCHEDULE,
                fixed_schedule_auto_offset=False,
                fixed_schedule_start_offset=2000,
                fixed_schedule_end_offset=8000,
            )
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING)
        assert (
            p.auto_offset_timestamps,
            p.fixed_schedule_start_offset,
            p.fixed_schedule_end_offset,
        ) == (False, 2000, 8000)

    def test_maps_cancellation_fields(self) -> None:
        cfg = TimingConfig.from_user_config(
            make_user_config(
                request_cancellation_rate=25.0, request_cancellation_delay=1.5
            )
        )
        assert (cfg.request_cancellation.rate, cfg.request_cancellation.delay) == (
            25.0,
            1.5,
        )

    def test_uses_user_centric_rate_when_request_rate_is_none(self) -> None:
        cfg = TimingConfig.from_user_config(
            make_user_config(request_rate=None, user_centric_rate=15.0)
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING)
        assert p.request_rate == 15.0

    def test_maps_num_sessions(self) -> None:
        cfg = TimingConfig.from_user_config(make_user_config(num_sessions=50))
        p = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING)
        assert p.expected_num_sessions == 50

    @pytest.mark.parametrize(
        "warmup_grace_period,expected",
        [(None, float("inf")), (15.0, 15.0), (0.0, 0.0)],
    )  # fmt: skip
    def test_warmup_grace_period(
        self, warmup_grace_period: float | None, expected: float
    ) -> None:
        cfg = TimingConfig.from_user_config(
            make_user_config(
                warmup_request_count=10, warmup_grace_period=warmup_grace_period
            )
        )
        warmup = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.WARMUP)
        assert warmup.grace_period_sec == expected
