# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for phase config helpers in ``aiperf.config.phases``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.phases import (
    BasePhaseConfig,
    ConcurrencyPhase,
    ConstantPhase,
    FixedSchedulePhase,
    GammaPhase,
    PoissonPhase,
    UserCentricPhase,
    get_phase_rate,
)
from aiperf.config.sweep.adaptive import SLAFilter
from aiperf.plugin.enums import PhaseType


class TestGetPhaseRate:
    """get_phase_rate is the single accessor for the phase ``rate`` field.

    It must return the configured rate for genuine RatePhaseConfig subclasses
    and None for every other phase type -- gated by isinstance, not by
    attribute probing, so a future field rename fails fast in one place.
    """

    @pytest.mark.parametrize(
        ("phase", "expected"),
        [
            param(
                ConstantPhase(
                    name="profiling", type=PhaseType.CONSTANT, rate=5.0, requests=10
                ),
                5.0,
                id="constant",
            ),
            param(
                PoissonPhase(
                    name="profiling", type=PhaseType.POISSON, rate=2.5, requests=10
                ),
                2.5,
                id="poisson",
            ),
            param(
                GammaPhase(
                    name="profiling", type=PhaseType.GAMMA, rate=1.5, requests=10
                ),
                1.5,
                id="gamma",
            ),
            param(
                UserCentricPhase(
                    name="profiling",
                    type=PhaseType.USER_CENTRIC,
                    rate=8.0,
                    users=2,
                    requests=10,
                ),
                8.0,
                id="user_centric",
            ),
        ],
    )  # fmt: skip
    def test_get_phase_rate_rate_phase_returns_rate(
        self, phase: BasePhaseConfig, expected: float
    ) -> None:
        assert get_phase_rate(phase) == expected

    @pytest.mark.parametrize(
        "phase",
        [
            param(
                ConcurrencyPhase(
                    name="profiling",
                    type=PhaseType.CONCURRENCY,
                    concurrency=4,
                    requests=10,
                ),
                id="concurrency",
            ),
            param(
                FixedSchedulePhase(name="profiling", type=PhaseType.FIXED_SCHEDULE),
                id="fixed_schedule",
            ),
        ],
    )  # fmt: skip
    def test_get_phase_rate_non_rate_phase_returns_none(
        self, phase: BasePhaseConfig
    ) -> None:
        assert get_phase_rate(phase) is None

    def test_get_phase_rate_stray_rate_attr_on_non_rate_phase_returns_none(
        self,
    ) -> None:
        """A rate-shaped attribute on a non-rate phase must not leak through.

        The pre-helper ``getattr(phase, "rate", None)`` probes would have
        returned it; the isinstance gate must not.
        """
        phase = ConcurrencyPhase(
            name="profiling", type=PhaseType.CONCURRENCY, concurrency=4, requests=10
        )
        object.__setattr__(phase, "rate", 7.5)
        assert get_phase_rate(phase) is None


class TestAdaptiveScalePhaseValidation:
    def test_adaptive_scale_enabled_rejects_non_boolean_values(self) -> None:
        with pytest.raises(ValidationError, match="adaptive_scale.enabled"):
            ConcurrencyPhase(
                name="profiling",
                type=PhaseType.CONCURRENCY,
                concurrency=2,
                duration=10,
                adaptive_scale={
                    "enabled": 2,
                    "sustain_duration": 1,
                    "sla": {"request_latency": {"p95": {"le": 1000}}},
                },
            )

    def test_flat_adaptive_scale_rejects_non_boolean_values(self) -> None:
        with pytest.raises(ValidationError, match="adaptive_scale.enabled"):
            ConcurrencyPhase(
                name="profiling",
                type=PhaseType.CONCURRENCY,
                concurrency=2,
                duration=10,
                adaptive_scale=1,
                adaptive_sustain_duration=1,
                sla=[
                    SLAFilter(
                        metric_tag="request_latency",
                        stat="p95",
                        op="le",
                        threshold=1000,
                    )
                ],
            )

    def test_adaptive_scale_accepts_camel_case_nested_block(self) -> None:
        phase = ConcurrencyPhase(
            name="profiling",
            type=PhaseType.CONCURRENCY,
            concurrency=20,
            duration=10,
            adaptiveScale={
                "enabled": True,
                "control": {"variable": "concurrency", "min": 2, "max": 20},
                "assessmentPeriod": 5,
                "sustainDuration": 1,
                "sla": {"request_latency": {"p95": {"le": 1000}}},
            },
        )

        assert phase.adaptive_scale is True
        assert phase.adaptive_control_variable == "concurrency"
        assert phase.adaptive_control_min == 2
        assert phase.adaptive_control_max == 20
        assert phase.adaptive_assessment_period == 5
        assert phase.adaptive_sustain_duration == 1
        assert phase.sla == [
            SLAFilter(
                metric_tag="request_latency",
                stat="p95",
                op="le",
                threshold=1000,
            )
        ]

    @pytest.mark.parametrize(
        "adaptive_scale",
        [
            {"enabled": True, "typo": 1},
            {"enabled": True, "control": {"variable": "concurrency", "typo": 1}},
            {"enabled": True, "strategy": {"type": "ramp_until_fail", "typo": 1}},
        ],
    )
    def test_nested_adaptive_scale_rejects_unknown_keys(
        self, adaptive_scale: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError, match="unsupported field"):
            ConcurrencyPhase(
                name="profiling",
                type=PhaseType.CONCURRENCY,
                concurrency=20,
                duration=10,
                adaptive_scale=adaptive_scale,
                adaptive_sustain_duration=1,
                sla=[
                    SLAFilter(
                        metric_tag="request_latency",
                        stat="p95",
                        op="le",
                        threshold=1000,
                    )
                ],
            )

    def test_fixed_schedule_accepts_disabled_adaptive_scale_block(self) -> None:
        phase = FixedSchedulePhase(
            name="profiling",
            type=PhaseType.FIXED_SCHEDULE,
            duration=10,
            requests=1,
            adaptive_scale={"enabled": False},
        )

        assert phase.adaptive_scale is False

    def test_fixed_schedule_rejects_adaptive_scale(self) -> None:
        with pytest.raises(ValidationError, match="fixed_schedule"):
            FixedSchedulePhase(
                name="profiling",
                type=PhaseType.FIXED_SCHEDULE,
                duration=10,
                requests=1,
                adaptive_scale={
                    "enabled": True,
                    "sustain_duration": 1,
                    "control": {"min": 1, "max": 2},
                    "sla": {"request_latency": {"p95": {"le": 1000}}},
                },
            )

    def test_request_rate_adaptive_control_rejects_rate_series(self) -> None:
        with pytest.raises(ValidationError, match="rate_series"):
            PoissonPhase(
                name="profiling",
                type=PhaseType.POISSON,
                duration=10,
                requests=10,
                rate_series={
                    "points": [
                        {"time_s": 0, "qps": 5},
                        {"time_s": 10, "qps": 15},
                    ]
                },
                adaptive_scale={
                    "enabled": True,
                    "sustain_duration": 1,
                    "control": {
                        "variable": "request_rate",
                        "min": 1,
                        "max": 20,
                    },
                    "sla": {"request_latency": {"p95": {"le": 1000}}},
                },
            )
