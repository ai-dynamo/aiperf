# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TimingConfig validation and construction."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from aiperf.common.config import UserConfig
from aiperf.common.enums import ArrivalPattern, CreditPhase, TimingMode
from aiperf.timing.config import (
    CreditPhaseConfig,
    RequestCancellationConfig,
    TimingConfig,
)


def make_phase_config(**overrides) -> CreditPhaseConfig:
    """Create a CreditPhaseConfig with defaults and optional overrides."""
    defaults = {
        "phase": CreditPhase.PROFILING,
        "timing_mode": TimingMode.REQUEST_RATE,
    }
    defaults.update(overrides)
    return CreditPhaseConfig(**defaults)


def make_user_config(**overrides) -> UserConfig:
    """Create a UserConfig with defaults and optional overrides."""
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

    user_config = MagicMock(spec=UserConfig)
    user_config.timing_mode = overrides.get("timing_mode", TimingMode.REQUEST_RATE)
    user_config.loadgen = loadgen
    user_config.input = input_config

    return user_config


class TestTimingConfigValidConfigurations:
    """Tests for valid TimingConfig configurations."""

    def test_minimal_request_rate_config(self) -> None:
        """Minimal valid configuration for REQUEST_RATE mode."""
        phase_config = make_phase_config()
        config = TimingConfig(phase_configs=[phase_config])

        assert len(config.phase_configs) == 1
        assert config.phase_configs[0].timing_mode == TimingMode.REQUEST_RATE
        assert config.phase_configs[0].concurrency is None
        assert config.phase_configs[0].request_rate is None

    def test_full_request_rate_config(self) -> None:
        """Full configuration for REQUEST_RATE mode."""
        phase_config = make_phase_config(
            concurrency=10,
            prefill_concurrency=5,
            request_rate=100.0,
            arrival_pattern=ArrivalPattern.CONSTANT,
            total_expected_requests=1000,
        )
        config = TimingConfig(phase_configs=[phase_config], random_seed=42)

        pc = config.phase_configs[0]
        assert pc.timing_mode == TimingMode.REQUEST_RATE
        assert pc.concurrency == 10
        assert pc.prefill_concurrency == 5
        assert pc.request_rate == 100.0
        assert pc.arrival_pattern == ArrivalPattern.CONSTANT
        assert pc.total_expected_requests == 1000
        assert config.random_seed == 42

    def test_fixed_schedule_config(self) -> None:
        """Valid configuration for FIXED_SCHEDULE mode."""
        phase_config = make_phase_config(
            timing_mode=TimingMode.FIXED_SCHEDULE,
            auto_offset_timestamps=True,
            fixed_schedule_start_offset=1000,
            fixed_schedule_end_offset=5000,
        )
        config = TimingConfig(phase_configs=[phase_config])

        pc = config.phase_configs[0]
        assert pc.timing_mode == TimingMode.FIXED_SCHEDULE
        assert pc.auto_offset_timestamps is True
        assert pc.fixed_schedule_start_offset == 1000
        assert pc.fixed_schedule_end_offset == 5000

    def test_user_centric_config(self) -> None:
        """Valid configuration for USER_CENTRIC_RATE mode."""
        phase_config = make_phase_config(
            timing_mode=TimingMode.USER_CENTRIC_RATE,
            request_rate=10.0,
            concurrency=5,
            expected_num_sessions=100,
        )
        config = TimingConfig(phase_configs=[phase_config])

        pc = config.phase_configs[0]
        assert pc.timing_mode == TimingMode.USER_CENTRIC_RATE
        assert pc.request_rate == 10.0
        assert pc.concurrency == 5
        assert pc.expected_num_sessions == 100

    def test_cancellation_config(self) -> None:
        """Valid cancellation configuration."""
        phase_config = make_phase_config()
        config = TimingConfig(
            phase_configs=[phase_config],
            request_cancellation=RequestCancellationConfig(rate=50.0, delay=2.5),
        )

        assert config.request_cancellation.rate == 50.0
        assert config.request_cancellation.delay == 2.5

    def test_zero_values_allowed_for_ge0_fields(self) -> None:
        """Zero is valid for fields with ge=0 constraint."""
        phase_config = make_phase_config(
            fixed_schedule_start_offset=0,
            fixed_schedule_end_offset=0,
        )
        config = TimingConfig(
            phase_configs=[phase_config],
            random_seed=0,
            request_cancellation=RequestCancellationConfig(rate=0.0),
        )

        assert config.random_seed == 0
        assert config.request_cancellation.rate == 0.0


class TestTimingConfigValidationErrors:
    """Tests for TimingConfig validation errors."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("concurrency", 0),
            ("concurrency", -1),
            ("prefill_concurrency", 0),
            ("prefill_concurrency", -1),
        ],
    )  # fmt: skip
    def test_ge1_fields_reject_zero_and_negative(self, field: str, value: int) -> None:
        """Fields with ge=1 constraint reject zero and negative values."""
        with pytest.raises(ValidationError) as exc_info:
            make_phase_config(**{field: value})

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == (field,)
        assert "greater than" in errors[0]["msg"]


class TestTimingConfigFrozen:
    """Tests for TimingConfig frozen model behavior."""

    def test_config_is_frozen(self) -> None:
        """Config cannot be modified after creation."""
        phase_config = make_phase_config()
        config = TimingConfig(phase_configs=[phase_config])

        with pytest.raises(ValidationError):
            config.random_seed = 42

    def test_phase_config_is_hashable(self) -> None:
        """Frozen phase config can be used as dict key."""
        phase_config = make_phase_config()
        d = {phase_config: "value"}
        assert d[phase_config] == "value"


class TestTimingConfigFromUserConfig:
    """Tests for TimingConfig.from_user_config() class method."""

    def test_maps_timing_mode(self) -> None:
        """timing_mode is mapped from user_config to profiling phase config."""
        user_config = make_user_config(timing_mode=TimingMode.FIXED_SCHEDULE)
        config = TimingConfig.from_user_config(user_config)

        profiling = next(
            pc for pc in config.phase_configs if pc.phase == CreditPhase.PROFILING
        )
        assert profiling.timing_mode == TimingMode.FIXED_SCHEDULE

    def test_maps_loadgen_fields(self) -> None:
        """Loadgen fields are mapped to profiling phase config."""
        user_config = make_user_config(
            concurrency=8, prefill_concurrency=4, request_rate=50.0, request_count=500
        )
        config = TimingConfig.from_user_config(user_config)

        profiling = next(
            pc for pc in config.phase_configs if pc.phase == CreditPhase.PROFILING
        )
        assert profiling.concurrency == 8
        assert profiling.prefill_concurrency == 4
        assert profiling.request_rate == 50.0
        assert profiling.total_expected_requests == 500

    def test_creates_warmup_when_configured(self) -> None:
        """Creates warmup phase when warmup settings are provided."""
        user_config = make_user_config(warmup_request_count=25)
        config = TimingConfig.from_user_config(user_config)

        phases = [pc.phase for pc in config.phase_configs]
        assert CreditPhase.WARMUP in phases
        assert CreditPhase.PROFILING in phases
        assert config.phase_configs[0].phase == CreditPhase.WARMUP

    def test_no_warmup_when_not_configured(self) -> None:
        """No warmup phase when warmup settings are not provided."""
        user_config = make_user_config()
        config = TimingConfig.from_user_config(user_config)

        phases = [pc.phase for pc in config.phase_configs]
        assert CreditPhase.WARMUP not in phases
        assert len(config.phase_configs) == 1

    def test_maps_fixed_schedule_fields(self) -> None:
        """Fixed schedule fields are mapped to profiling phase config."""
        user_config = make_user_config(
            timing_mode=TimingMode.FIXED_SCHEDULE,
            fixed_schedule_auto_offset=False,
            fixed_schedule_start_offset=2000,
            fixed_schedule_end_offset=8000,
        )
        config = TimingConfig.from_user_config(user_config)

        profiling = next(
            pc for pc in config.phase_configs if pc.phase == CreditPhase.PROFILING
        )
        assert profiling.auto_offset_timestamps is False
        assert profiling.fixed_schedule_start_offset == 2000
        assert profiling.fixed_schedule_end_offset == 8000

    def test_maps_cancellation_fields(self) -> None:
        """Cancellation fields are mapped correctly to TimingConfig."""
        user_config = make_user_config(
            request_cancellation_rate=25.0, request_cancellation_delay=1.5
        )
        config = TimingConfig.from_user_config(user_config)

        assert config.request_cancellation.rate == 25.0
        assert config.request_cancellation.delay == 1.5

    def test_uses_user_centric_rate_when_request_rate_is_none(self) -> None:
        """user_centric_rate is used when request_rate is None."""
        user_config = make_user_config(request_rate=None, user_centric_rate=15.0)
        config = TimingConfig.from_user_config(user_config)

        profiling = next(
            pc for pc in config.phase_configs if pc.phase == CreditPhase.PROFILING
        )
        assert profiling.request_rate == 15.0

    def test_maps_num_sessions(self) -> None:
        """num_sessions is mapped from input.conversation.num."""
        user_config = make_user_config(num_sessions=50)
        config = TimingConfig.from_user_config(user_config)

        profiling = next(
            pc for pc in config.phase_configs if pc.phase == CreditPhase.PROFILING
        )
        assert profiling.expected_num_sessions == 50

    # fmt: skip
    @pytest.mark.parametrize(
        "warmup_grace_period,expected",
        [(None, float("inf")), (15.0, 15.0), (0.0, 0.0)],
    )
    def test_warmup_grace_period(
        self, warmup_grace_period: float | None, expected: float
    ) -> None:
        """Warmup grace period is mapped correctly from user config."""
        user_config = make_user_config(
            warmup_request_count=10, warmup_grace_period=warmup_grace_period
        )
        config = TimingConfig.from_user_config(user_config)

        warmup = next(
            pc for pc in config.phase_configs if pc.phase == CreditPhase.WARMUP
        )
        assert warmup.grace_period_sec == expected
