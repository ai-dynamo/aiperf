# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: ``CreditCounter`` session accounting under agentic mid-trace resume."""

from __future__ import annotations

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter


def _cfg(
    reqs: int | None = None,
    sessions: int | None = None,
    dur: float | None = None,
) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.AGENTIC_REPLAY,
        total_expected_requests=reqs,
        expected_num_sessions=sessions,
        expected_duration_sec=dur,
    )


def _turn(
    *,
    idx: int,
    num: int,
    corr: str = "r1",
    depth: int = 0,
    counts: bool = True,
    session_start: bool = False,
) -> TurnToSend:
    return TurnToSend(
        conversation_id="trace",
        x_correlation_id=corr,
        turn_index=idx,
        num_turns=num,
        agent_depth=depth,
        counts_toward_phase_target=counts,
        is_session_start=session_start,
    )


class TestMidTraceRootSessionAccounting:
    """A root resumed mid-trace is counted as a session on its first credit."""

    def test_completed_sessions_never_exceeds_sent_sessions(self) -> None:
        c = CreditCounter(_cfg())

        c.increment_sent(_turn(idx=3, num=4, session_start=True))
        assert c.sent_sessions == 1, "the resume credit starts a session"

        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.completed_sessions == 1, "final turn bumps completed_sessions"

        assert c.completed_sessions <= c.sent_sessions

    def test_in_flight_sessions_never_negative(self) -> None:
        c = CreditCounter(_cfg())
        c.increment_sent(_turn(idx=2, num=4, session_start=True))
        c.increment_sent(_turn(idx=3, num=4))
        assert c.in_flight_sessions == 1
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.in_flight_sessions == 0
        assert c.in_flight_sessions >= 0

    def test_recycled_session_started_at_turn_zero_is_balanced(self) -> None:
        """Contrast: a recycled session (start_turn_index=0) is balanced."""
        c = CreditCounter(_cfg())
        c.increment_sent(_turn(idx=0, num=2))
        c.increment_sent(_turn(idx=1, num=2))
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.sent_sessions == 1
        assert c.completed_sessions == 1
        assert c.in_flight_sessions == 0


class TestGatedParentLaneSessionAccounting:
    """A gated parent admitted via ``acquire_lane_credit`` (turn 0 before t*, so"""

    def test_account_lane_session_keeps_in_flight_non_negative(self) -> None:
        c = CreditCounter(_cfg())
        c.account_lane_session(session_turns=2)
        assert c.sent_sessions == 1
        assert c.in_flight_sessions == 1

        c.increment_sent(_turn(idx=1, num=3))
        c.increment_sent(_turn(idx=2, num=3))
        assert c.sent_sessions == 1
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=True, cancelled=False)

        assert c.completed_sessions == 1
        assert c.completed_sessions <= c.sent_sessions
        assert c.in_flight_sessions == 0

    def test_total_session_turns_stays_consistent_with_root_sent(self) -> None:
        """account_lane_session adds the gated parent's remaining turns to"""
        c = CreditCounter(_cfg())
        c.account_lane_session(session_turns=2)
        assert c.total_session_turns == 2
        c.increment_sent(_turn(idx=1, num=3))
        c.increment_sent(_turn(idx=2, num=3))
        assert c.root_requests_sent == c.total_session_turns == 2
