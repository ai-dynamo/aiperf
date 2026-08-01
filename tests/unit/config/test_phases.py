# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for phase config helpers in ``aiperf.config.phases``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import UserCentricGapDistribution
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
        with pytest.raises(ValidationError, match=r"adaptive_scale\.enabled"):
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
        with pytest.raises(ValidationError, match=r"adaptive_scale\.enabled"):
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

    def test_adaptive_scale_field_takes_precedence_over_stale_alias(self) -> None:
        phase = ConcurrencyPhase(
            name="profiling",
            type=PhaseType.CONCURRENCY,
            concurrency=20,
            duration=10,
            adaptive_scale=False,
            adaptiveScale={
                "enabled": True,
                "control": {"variable": "concurrency", "min": 2, "max": 20},
                "sustainDuration": 1,
                "sla": {"request_latency": {"p95": {"le": 1000}}},
            },
        )

        assert phase.adaptive_scale is False

    @pytest.mark.parametrize(
        "adaptive_scale",
        [
            pytest.param({"enabled": True, "typo": 1}, id="top-level-unknown"),
            pytest.param(
                {"enabled": True, "control": {"variable": "concurrency", "typo": 1}},
                id="control-unknown",
            ),
            pytest.param(
                {"enabled": True, "strategy": {"type": "ramp_until_fail", "typo": 1}},
                id="strategy-unknown",
            ),
        ],
    )  # fmt: skip
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


def _make_user_centric_phase(**overrides) -> UserCentricPhase:
    kwargs = {
        "name": "profiling",
        "type": PhaseType.USER_CENTRIC,
        "rate": 10.0,
        "users": 5,
        "requests": 10,
    }
    kwargs.update(overrides)
    return UserCentricPhase(**kwargs)


class TestUserCentricGapDistribution:
    """Cross-field validation of gap_distribution / gap_median on UserCentricPhase."""

    def test_default_is_fixed_without_median(self) -> None:
        phase = _make_user_centric_phase()
        assert phase.gap_distribution == UserCentricGapDistribution.FIXED
        assert phase.gap_median is None

    @pytest.mark.parametrize(
        "distribution",
        [
            param(UserCentricGapDistribution.LOGNORMAL, id="lognormal"),
            param(UserCentricGapDistribution.WEIBULL, id="weibull"),
        ],
    )  # fmt: skip
    def test_valid_median_below_mean_accepted(
        self, distribution: UserCentricGapDistribution
    ) -> None:
        # mean gap = users / rate = 5 / 10.0 = 0.5s; median 0.3s < mean.
        phase = _make_user_centric_phase(gap_distribution=distribution, gap_median=0.3)
        assert phase.gap_distribution == distribution
        assert phase.gap_median == 0.3

    def test_median_with_fixed_distribution_rejected(self) -> None:
        with pytest.raises(ValueError, match="--user-centric-gap-median"):
            _make_user_centric_phase(gap_median=0.3)

    @pytest.mark.parametrize(
        "distribution",
        [
            param(UserCentricGapDistribution.LOGNORMAL, id="lognormal"),
            param(UserCentricGapDistribution.WEIBULL, id="weibull"),
        ],
    )  # fmt: skip
    def test_missing_median_for_sampled_distribution_rejected(
        self, distribution: UserCentricGapDistribution
    ) -> None:
        with pytest.raises(ValueError, match="requires --user-centric-gap-median"):
            _make_user_centric_phase(gap_distribution=distribution)

    @pytest.mark.parametrize(
        "median",
        [
            param(0.5, id="median_equals_mean"),
            param(1.5, id="median_above_mean"),
        ],
    )  # fmt: skip
    def test_median_not_below_mean_rejected_with_computed_mean(
        self, median: float
    ) -> None:
        # mean gap = users / rate = 5 / 10.0 = 0.5s.
        with pytest.raises(ValueError, match=r"0\.5"):
            _make_user_centric_phase(
                gap_distribution=UserCentricGapDistribution.LOGNORMAL,
                gap_median=median,
            )

    def test_non_positive_median_rejected(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            _make_user_centric_phase(
                gap_distribution=UserCentricGapDistribution.LOGNORMAL,
                gap_median=0.0,
            )


def _make_adaptive_users_phase(**overrides) -> UserCentricPhase:
    """User-centric phase scaling 'users' adaptively from control.min upward."""
    adaptive_scale = {
        "enabled": True,
        "sustain_duration": 1,
        "control": {"variable": "users", "min": 2},
        "sla": {"request_latency": {"p95": {"le": 1000}}},
    }
    adaptive_scale.update(overrides.pop("adaptive_scale", {}))
    return _make_user_centric_phase(
        rate=4.0,
        users=16,
        requests=None,
        duration=30,
        adaptive_scale=adaptive_scale,
        **overrides,
    )


class TestAdaptiveUsersGapMedianBound:
    """gap_median is bounded by the smallest mean turn gap the run can reach.

    ``users`` is the maximum user count; adaptive 'users' scaling starts the run
    at ``control.min``, where the mean turn gap is smallest.
    """

    @pytest.mark.parametrize(
        "median",
        [
            param(0.5, id="median_equals_control_min_gap"),
            param(1.0, id="median_between_control_min_and_users_gap"),
            param(3.9, id="median_just_below_users_gap"),
        ],
    )  # fmt: skip
    def test_median_above_control_min_gap_rejected(self, median: float) -> None:
        # control.min gap = 2 / 4.0 = 0.5s; users gap = 16 / 4.0 = 4.0s.
        with pytest.raises(ValueError, match=r"adaptive_scale\.control\.min \(2\)"):
            _make_adaptive_users_phase(
                gap_distribution=UserCentricGapDistribution.LOGNORMAL,
                gap_median=median,
            )

    def test_rejection_message_names_the_bound_and_its_value(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _make_adaptive_users_phase(
                gap_distribution=UserCentricGapDistribution.WEIBULL,
                gap_median=1.0,
            )
        message = str(excinfo.value)
        assert "adaptive_scale.control.min (2) / --user-centric-rate" in message
        assert "= 0.5 seconds" in message
        assert "starts the run at control.min" in message

    @pytest.mark.parametrize(
        "distribution",
        [
            param(UserCentricGapDistribution.LOGNORMAL, id="lognormal"),
            param(UserCentricGapDistribution.WEIBULL, id="weibull"),
        ],
    )  # fmt: skip
    def test_median_below_control_min_gap_accepted(
        self, distribution: UserCentricGapDistribution
    ) -> None:
        phase = _make_adaptive_users_phase(
            gap_distribution=distribution, gap_median=0.4
        )
        assert phase.gap_distribution == distribution
        assert phase.gap_median == 0.4

    def test_default_control_min_bounds_the_gap_at_one_user(self) -> None:
        # control.min defaults to 1, so the smallest gap is 1 / 4.0 = 0.25s.
        with pytest.raises(ValueError, match=r"adaptive_scale\.control\.min \(1\)"):
            _make_adaptive_users_phase(
                gap_distribution=UserCentricGapDistribution.LOGNORMAL,
                gap_median=0.4,
                adaptive_scale={"control": {"variable": "users"}},
            )

    def test_non_users_control_variable_keeps_the_num_users_bound(self) -> None:
        # control.min here bounds concurrency, not users, so the gap bound stays
        # at users / rate = 16 / 4.0 = 4.0s.
        phase = _make_adaptive_users_phase(
            gap_distribution=UserCentricGapDistribution.LOGNORMAL,
            gap_median=3.0,
            concurrency=32,
            adaptive_scale={
                "control": {"variable": "concurrency", "min": 2, "max": 32},
            },
        )
        assert phase.gap_median == 3.0

    def test_disabled_adaptive_scale_keeps_the_num_users_bound(self) -> None:
        phase = _make_adaptive_users_phase(
            gap_distribution=UserCentricGapDistribution.LOGNORMAL,
            gap_median=3.0,
            adaptive_scale={"enabled": False},
        )
        assert phase.gap_median == 3.0
