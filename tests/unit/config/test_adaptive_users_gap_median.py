# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin the config-time gap_median bound to the runtime solve.

``UserCentricPhase`` validates ``gap_median`` against the smallest mean turn gap
an adaptive-users run can reach; ``UserCentricStrategy`` is what actually
explodes when the median exceeds it, at phase start. These tests drive both
sides so the two bounds cannot drift apart -- something the pure config tests in
``test_phases.py`` structurally cannot check. The config-level coverage of the
bound itself lives there, in ``TestAdaptiveUsersGapMedianBound``.
"""

from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import CreditPhase, UserCentricGapDistribution
from aiperf.config.phases import UserCentricPhase
from aiperf.plugin.enums import PhaseType, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.user_centric_rate import UserCentricStrategy


def _make_adaptive_users_phase(**overrides) -> UserCentricPhase:
    """Phase with mean gap users / rate = 16 / 4 = 4.0s, floor 2 / 4 = 0.5s."""
    kwargs = {
        "name": "profiling",
        "type": PhaseType.USER_CENTRIC,
        "rate": 4.0,
        "users": 16,
        "duration": 30,
        "gap_distribution": UserCentricGapDistribution.LOGNORMAL,
        "gap_median": 0.4,
        "adaptive_scale": {
            "enabled": True,
            "sustain_duration": 1,
            "control": {"variable": "users", "min": 2},
            "sla": {"request_latency": {"p95": {"le": 1000}}},
        },
    }
    kwargs.update(overrides)
    return UserCentricPhase(**kwargs)


def _make_strategy(
    phase: UserCentricPhase, *, gap_median: float | None = None
) -> UserCentricStrategy:
    """Mirror the user-centric fields that ``_build_profiling_config`` copies.

    ``gap_median`` overrides the phase value so a median the config validator now
    rejects can still be handed to the runtime.
    """
    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.USER_CENTRIC_RATE,
        request_rate=phase.rate,
        num_users=phase.users,
        total_expected_requests=10,
        user_centric_gap_distribution=phase.gap_distribution,
        user_centric_gap_median=gap_median
        if gap_median is not None
        else phase.gap_median,
    )
    return UserCentricStrategy(
        config=cfg,
        conversation_source=MagicMock(),
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=MagicMock(),
        lifecycle=MagicMock(),
    )


class TestAdaptiveUsersGapMedianMatchesRuntime:
    def test_accepted_config_survives_the_runtime_solve_at_control_min(self) -> None:
        # AdaptiveScaleStrategy.setup_phase does _set_control(control.minimum),
        # which UsersControlBackend.set forwards to set_target_users.
        phase = _make_adaptive_users_phase()
        strategy = _make_strategy(phase)
        strategy.set_target_users(int(phase.adaptive_control_min))
        assert strategy._turn_gap == pytest.approx(0.5)
        # And the ramp all the way up to the users target stays solvable.
        strategy.set_target_users(phase.users)
        assert strategy._turn_gap == pytest.approx(4.0)

    def test_rejected_median_is_exactly_what_the_runtime_solve_refuses(self) -> None:
        # gap_median 1.0 sits between the floor gap (2 / 4 = 0.5s) and the target
        # gap (16 / 4 = 4.0s). The config now rejects it, and it still explodes
        # at phase start, so the tightened bound is not over-restriction.
        strategy = _make_strategy(_make_adaptive_users_phase(), gap_median=1.0)
        with pytest.raises(ValueError, match=r"0\.5"):
            strategy.set_target_users(2)
