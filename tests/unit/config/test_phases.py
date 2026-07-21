# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for phase config helpers in ``aiperf.config.phases``."""

from __future__ import annotations

import pytest
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
