# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from aiperf.config import AIPerfConfig
from aiperf.plugin.enums import ArrivalPattern, TimingMode, URLSelectionStrategy
from aiperf.timing.config import (
    CreditPhaseConfig,
    RequestCancellationConfig,
    TimingConfig,
)

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"], "streaming": True},
    datasets={
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
)


def make_phase_config(**overrides) -> CreditPhaseConfig:
    defaults = {"phase": "profiling", "timing_mode": TimingMode.REQUEST_RATE}
    defaults.update(overrides)
    return CreditPhaseConfig(**defaults)


def make_config(phases: dict | None = None) -> AIPerfConfig:
    """Create an AIPerfConfig with the given phases."""
    if phases is None:
        phases = {
            "profiling": {
                "type": "poisson",
                "requests": 100,
                "rate": 10.0,
                "concurrency": 10,
            }
        }
    return AIPerfConfig(**_BASE, phases=phases)


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

    def test_config_is_frozen(self) -> None:
        cfg = TimingConfig(phase_configs=[make_phase_config()])
        with pytest.raises(ValidationError):
            cfg.request_cancellation = RequestCancellationConfig(rate=50.0)

    def test_phase_config_is_hashable(self) -> None:
        pc = make_phase_config()
        assert {pc: "value"}[pc] == "value"


class TestTimingConfigFromConfig:
    """Test TimingConfig.from_config builds CreditPhaseConfigs from AIPerfConfig load phases."""

    def test_maps_fixed_schedule_type(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={"profiling": {"type": "fixed_schedule", "requests": 100}}
            )
        )
        profiling = next(pc for pc in cfg.phase_configs if pc.phase == "profiling")
        assert profiling.timing_mode == TimingMode.FIXED_SCHEDULE

    def test_maps_poisson_type(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={"profiling": {"type": "poisson", "rate": 50.0, "requests": 500}}
            )
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == "profiling")
        assert p.timing_mode == TimingMode.REQUEST_RATE
        assert p.arrival_pattern == ArrivalPattern.POISSON

    def test_maps_phase_fields(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {
                        "type": "poisson",
                        "concurrency": 8,
                        "prefill_concurrency": 4,
                        "rate": 50.0,
                        "requests": 500,
                    }
                }
            )
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == "profiling")
        assert (
            p.concurrency,
            p.prefill_concurrency,
            p.request_rate,
            p.total_expected_requests,
        ) == (8, 4, 50.0, 500)

    def test_creates_warmup_from_excluded_phase(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "warmup": {
                        "type": "concurrency",
                        "concurrency": 1,
                        "requests": 25,
                        "exclude_from_results": True,
                    },
                    "profiling": {"type": "poisson", "rate": 10.0, "requests": 100},
                }
            )
        )
        phases = [pc.phase for pc in cfg.phase_configs]
        assert "warmup" in phases
        assert cfg.phase_configs[0].phase == "warmup"

    def test_no_warmup_when_no_excluded_phase(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={"profiling": {"type": "poisson", "rate": 10.0, "requests": 100}}
            )
        )
        phases = [pc.phase for pc in cfg.phase_configs]
        assert "warmup" not in phases
        assert len(cfg.phase_configs) == 1

    def test_maps_fixed_schedule_auto_offset_false(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {
                        "type": "fixed_schedule",
                        "requests": 100,
                        "auto_offset": False,
                    }
                }
            )
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == "profiling")
        assert p.auto_offset_timestamps is False

    def test_maps_fixed_schedule_end_offset(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {
                        "type": "fixed_schedule",
                        "requests": 100,
                        "end_offset": 8000,
                    }
                }
            )
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == "profiling")
        assert p.fixed_schedule_end_offset == 8000

    def test_maps_cancellation_from_phase(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {
                        "type": "poisson",
                        "rate": 10.0,
                        "requests": 100,
                        "cancellation": {"rate": 25.0, "delay": 1.5},
                    }
                }
            )
        )
        assert (cfg.request_cancellation.rate, cfg.request_cancellation.delay) == (
            25.0,
            1.5,
        )

    def test_maps_user_centric_type(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {
                        "type": "user_centric",
                        "rate": 15.0,
                        "users": 5,
                        "requests": 100,
                    }
                }
            )
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == "profiling")
        assert p.timing_mode == TimingMode.USER_CENTRIC_RATE
        assert p.request_rate == 15.0
        assert p.num_users == 5

    def test_maps_sessions(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {
                        "type": "poisson",
                        "rate": 10.0,
                        "sessions": 50,
                    }
                }
            )
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == "profiling")
        assert p.expected_num_sessions == 50

    @pytest.mark.parametrize(
        "grace_period,expected",
        [(None, float("inf")), (15.0, 15.0), (0.0, 0.0)],
    )  # fmt: skip
    def test_warmup_grace_period(
        self, grace_period: float | None, expected: float
    ) -> None:
        warmup_phase: dict = {
            "type": "concurrency",
            "concurrency": 1,
            "duration": 30,
            "exclude_from_results": True,
        }
        if grace_period is not None:
            warmup_phase["grace_period"] = grace_period
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "warmup": warmup_phase,
                    "profiling": {"type": "poisson", "rate": 10.0, "requests": 100},
                }
            )
        )
        warmup = next(pc for pc in cfg.phase_configs if pc.phase == "warmup")
        assert warmup.grace_period_sec == expected

    def test_maps_urls_from_endpoint(self) -> None:
        cfg = TimingConfig.from_config(make_config())
        assert cfg.urls == ["http://localhost:8000/v1/chat/completions"]
        assert cfg.url_selection_strategy == URLSelectionStrategy.ROUND_ROBIN

    def test_maps_concurrency_type(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {
                        "type": "concurrency",
                        "concurrency": 10,
                        "requests": 100,
                    }
                }
            )
        )
        p = cfg.phase_configs[0]
        assert p.timing_mode == TimingMode.REQUEST_RATE
        assert p.arrival_pattern == ArrivalPattern.CONCURRENCY_BURST
        assert p.concurrency == 10

    def test_maps_constant_type(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {"type": "constant", "rate": 10.0, "requests": 100}
                }
            )
        )
        p = cfg.phase_configs[0]
        assert p.arrival_pattern == ArrivalPattern.CONSTANT

    def test_maps_gamma_type_with_smoothness(self) -> None:
        cfg = TimingConfig.from_config(
            make_config(
                phases={
                    "profiling": {
                        "type": "gamma",
                        "rate": 10.0,
                        "requests": 100,
                        "smoothness": 2.5,
                    }
                }
            )
        )
        p = cfg.phase_configs[0]
        assert p.arrival_pattern == ArrivalPattern.GAMMA
        assert p.arrival_smoothness == 2.5
