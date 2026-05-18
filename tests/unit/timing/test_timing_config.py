# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import pytest
from pydantic import ValidationError

from aiperf.common.enums import CreditPhase
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import (
    CreditPhaseConfig,
    RequestCancellationConfig,
    TimingConfig,
)
from tests.unit.conftest import make_run_from_cli


def make_phase_config(**overrides) -> CreditPhaseConfig:
    defaults = {"phase": CreditPhase.PROFILING, "timing_mode": TimingMode.REQUEST_RATE}
    defaults.update(overrides)
    return CreditPhaseConfig(**defaults)


# Fields recognized by ``make_cfg`` overrides. Each is routed to either
# the loadgen or input section of a real v1 ``CLIConfig``; only fields the
# caller passes flow into ``model_fields_set`` so the v1 -> v2 resolver maps
# them faithfully.
_LOADGEN_FIELDS: frozenset[str] = frozenset(
    {
        "concurrency",
        "prefill_concurrency",
        "request_rate",
        "user_centric_rate",
        "arrival_pattern",
        "request_count",
        "num_users",
        "warmup_request_count",
        "warmup_duration",
        "warmup_num_sessions",
        "warmup_concurrency",
        "warmup_prefill_concurrency",
        "warmup_request_rate",
        "warmup_arrival_pattern",
        "warmup_concurrency_ramp_duration",
        "warmup_prefill_concurrency_ramp_duration",
        "warmup_request_rate_ramp_duration",
        "warmup_grace_period",
        "benchmark_duration",
        "benchmark_grace_period",
        "request_cancellation_rate",
        "request_cancellation_delay",
        "concurrency_ramp_duration",
        "prefill_concurrency_ramp_duration",
        "request_rate_ramp_duration",
        "arrival_smoothness",
    }
)
_INPUT_FIELDS: frozenset[str] = frozenset(
    {
        "fixed_schedule",
        "fixed_schedule_auto_offset",
        "fixed_schedule_start_offset",
        "fixed_schedule_end_offset",
    }
)


def make_cfg(**overrides) -> CLIConfig:
    """Build a real v1 ``CLIConfig`` whose ``model_fields_set`` reflects only
    the kwargs the caller passed. The v1 -> v2 resolver depends on
    ``model_fields_set`` to distinguish "user supplied" from "defaulted", so
    using a real config (not MagicMock) is required for fidelity through
    ``TimingConfig.from_run``.

    ``timing_mode=FIXED_SCHEDULE`` is rewritten to ``input.fixed_schedule=True``
    because v1 has no top-level ``timing_mode`` field — the resolver derives
    the timing mode from input/loadgen state.

    ``num_sessions`` is rewritten to ``conversation_num``.

    ``turn_mean`` is rewritten to ``conversation_turn_mean`` (USER_CENTRIC
    mode validates ``turn_mean >= 2``).

    ``streaming`` flag flows to the endpoint section (prefill_concurrency
    requires streaming=True at AIPerfConfig validation time).
    """
    loadgen_kwargs: dict[str, Any] = {}
    input_kwargs: dict[str, Any] = {}
    endpoint_kwargs: dict[str, Any] = {"model_names": ["test-model"]}

    if overrides.pop("timing_mode", None) == TimingMode.FIXED_SCHEDULE:
        input_kwargs["fixed_schedule"] = True

    if "num_sessions" in overrides:
        input_kwargs["conversation_num"] = overrides.pop("num_sessions")
    if "turn_mean" in overrides:
        input_kwargs["conversation_turn_mean"] = overrides.pop("turn_mean")

    if overrides.pop("streaming", False):
        endpoint_kwargs["streaming"] = True

    for key, value in overrides.items():
        if key in _LOADGEN_FIELDS:
            loadgen_kwargs[key] = value
        elif key in _INPUT_FIELDS:
            input_kwargs[key] = value
        else:
            raise KeyError(f"unknown make_cfg override: {key!r}")

    return CLIConfig(
        **endpoint_kwargs,
        **CLIConfig(**loadgen_kwargs).model_dump(exclude_unset=True),
        **input_kwargs,
    )


def _make_timing_config(**overrides) -> TimingConfig:
    """Convenience: build a CLIConfig with overrides, run it through the v1
    -> v2 resolver, and return the resulting ``TimingConfig``."""
    return TimingConfig.from_run(make_run_from_cli(make_cfg(**overrides)))


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


class TestTimingConfigFromCLIConfig:
    def test_maps_timing_mode(self) -> None:
        cfg = _make_timing_config(timing_mode=TimingMode.FIXED_SCHEDULE)
        profiling = next(
            pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING
        )
        assert profiling.timing_mode == TimingMode.FIXED_SCHEDULE

    def test_maps_loadgen_fields(self) -> None:
        cfg = _make_timing_config(
            concurrency=8,
            prefill_concurrency=4,
            request_rate=50.0,
            request_count=500,
            streaming=True,
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING)
        assert (
            p.concurrency,
            p.prefill_concurrency,
            p.request_rate,
            p.total_expected_requests,
        ) == (8, 4, 50.0, 500)

    def test_creates_warmup_when_configured(self) -> None:
        cfg = _make_timing_config(warmup_request_count=25)
        phases = [pc.phase for pc in cfg.phase_configs]
        assert CreditPhase.WARMUP in phases
        assert cfg.phase_configs[0].phase == CreditPhase.WARMUP

    def test_no_warmup_when_not_configured(self) -> None:
        cfg = _make_timing_config()
        phases = [pc.phase for pc in cfg.phase_configs]
        assert CreditPhase.WARMUP not in phases
        assert len(cfg.phase_configs) == 1

    def test_maps_fixed_schedule_fields(self) -> None:
        cfg = _make_timing_config(
            timing_mode=TimingMode.FIXED_SCHEDULE,
            fixed_schedule_auto_offset=False,
            fixed_schedule_start_offset=2000,
            fixed_schedule_end_offset=8000,
        )
        p = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING)
        assert (
            p.auto_offset_timestamps,
            p.fixed_schedule_start_offset,
            p.fixed_schedule_end_offset,
        ) == (False, 2000, 8000)

    def test_maps_cancellation_fields(self) -> None:
        cfg = _make_timing_config(
            request_cancellation_rate=25.0, request_cancellation_delay=1.5
        )
        assert (cfg.request_cancellation.rate, cfg.request_cancellation.delay) == (
            25.0,
            1.5,
        )

    def test_uses_user_centric_rate_when_request_rate_is_none(self) -> None:
        # USER_CENTRIC mode requires multi-turn sessions; the v1 -> v2 resolver
        # rejects USER_CENTRIC with the default turn mean of 1.
        cfg = _make_timing_config(user_centric_rate=15.0, num_users=4, turn_mean=2)
        p = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING)
        assert p.request_rate == 15.0

    def test_maps_num_sessions(self) -> None:
        cfg = _make_timing_config(num_sessions=50)
        p = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.PROFILING)
        assert p.expected_num_sessions == 50

    @pytest.mark.parametrize(
        "warmup_grace_period,expected",
        [(None, float("inf")), (15.0, 15.0), (0.0, 0.0)],
    )  # fmt: skip
    def test_warmup_grace_period(
        self, warmup_grace_period: float | None, expected: float
    ) -> None:
        # v2 phase validation requires ``duration`` whenever ``grace_period``
        # is set, so trigger a duration-bounded warmup. ``_build_warmup_config``
        # still defaults ``grace_period_sec`` to ``inf`` when phase.grace_period
        # is None, which preserves the original semantics.
        kwargs: dict[str, Any] = {"warmup_duration": 5.0}
        if warmup_grace_period is not None:
            kwargs["warmup_grace_period"] = warmup_grace_period
        cfg = _make_timing_config(**kwargs)
        warmup = next(pc for pc in cfg.phase_configs if pc.phase == CreditPhase.WARMUP)
        assert warmup.grace_period_sec == expected
