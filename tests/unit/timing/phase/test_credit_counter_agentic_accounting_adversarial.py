# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression: ``CreditCounter`` session accounting under agentic mid-trace resume.

Agentic replay resumes a sampled trajectory mid-conversation:
``AgenticReplayStrategy._execute_profiling`` issues the first PROFILING credit
of every trajectory at ``turn_index = k_i + 1`` (>= 1), and snapshot lanes
resume their live root at ``state.next_turn_index`` (also > 0). The root's
turn 0 is therefore NEVER sent during the PROFILING phase.

The FIX: the strategy flags that first credit ``is_session_start=True`` (via
``SampledSession.build_turn_at_index``), so ``increment_sent`` bumps
``_sent_sessions`` for ``agent_depth == 0 and (turn_index == 0 or
is_session_start)`` and ``increment_returned`` bumps ``_completed_sessions`` on
the final turn -- keeping ``completed_sessions <= sent_sessions`` and
``in_flight_sessions >= 0``.

Before the fix, ``increment_sent`` gated only on ``turn_index == 0``, so a
resumed root *completed* a session it never *started*, breaking the invariant
and driving ``in_flight_sessions`` negative.

These are pure-counter assertions, no async wiring required.
"""

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
    """A root resumed mid-trace is counted as a session on its first credit.

    The strategy flags the resume credit ``is_session_start=True`` (via
    ``SampledSession.build_turn_at_index``), so the resumed root bumps
    ``sent_sessions`` even at ``turn_index > 0`` -- keeping
    ``completed_sessions <= sent_sessions`` and ``in_flight_sessions >= 0``.
    """

    def test_completed_sessions_never_exceeds_sent_sessions(self) -> None:
        c = CreditCounter(_cfg())

        # PROFILING resume of a 4-turn trace sampled at k_i=2: the first credit
        # in this phase is turn 3 (k_i+1) and carries is_session_start=True.
        c.increment_sent(_turn(idx=3, num=4, session_start=True))
        assert c.sent_sessions == 1, "the resume credit starts a session"

        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.completed_sessions == 1, "final turn bumps completed_sessions"

        assert c.completed_sessions <= c.sent_sessions

    def test_in_flight_sessions_never_negative(self) -> None:
        c = CreditCounter(_cfg())
        # 4-turn trace resumed at turn 2 (k_i=1 -> resume at 2), then turn 3.
        c.increment_sent(_turn(idx=2, num=4, session_start=True))
        c.increment_sent(_turn(idx=3, num=4))
        assert c.in_flight_sessions == 1  # session started, not yet finished
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.in_flight_sessions == 0
        assert c.in_flight_sessions >= 0

    def test_recycled_session_started_at_turn_zero_is_balanced(self) -> None:
        """Contrast: a recycled session (start_turn_index=0) is balanced.

        This is the path that DOES bump sent_sessions, so completed never
        exceeds sent. Demonstrates the asymmetry is specific to mid-trace
        resume, not multi-turn sessions in general.
        """
        c = CreditCounter(_cfg())
        c.increment_sent(_turn(idx=0, num=2))  # turn 0 -> sent_sessions=1
        c.increment_sent(_turn(idx=1, num=2))
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=True, cancelled=False)
        assert c.sent_sessions == 1
        assert c.completed_sessions == 1
        assert c.in_flight_sessions == 0


class TestGatedParentLaneSessionAccounting:
    """A gated parent admitted via ``acquire_lane_credit`` (turn 0 before t*, so
    it never dispatches a session-start root credit) must be counted in
    ``sent_sessions`` via :meth:`CreditCounter.account_lane_session`. Its join
    turn reaches a terminal turn that bumps ``completed_sessions``; without the
    symmetric ``sent`` bump ``in_flight_sessions`` goes negative.
    """

    def test_account_lane_session_keeps_in_flight_non_negative(self) -> None:
        c = CreditCounter(_cfg())
        # Gated parent of a 3-turn trace gated at turn 1: lane credit counts the
        # session up front (remaining turns = num_turns - gated_turn_index = 2).
        c.account_lane_session(session_turns=2)
        assert c.sent_sessions == 1
        assert c.in_flight_sessions == 1  # admitted, not yet completed

        # The join turn (turn 1, NOT a session start -> no second sent bump)
        # then the final turn (turn 2) dispatch and return.
        c.increment_sent(_turn(idx=1, num=3))
        c.increment_sent(_turn(idx=2, num=3))
        assert c.sent_sessions == 1  # join turns must NOT re-bump sent_sessions
        c.increment_returned(is_final_turn=False, cancelled=False)
        c.increment_returned(is_final_turn=True, cancelled=False)

        assert c.completed_sessions == 1
        assert c.completed_sessions <= c.sent_sessions
        assert c.in_flight_sessions == 0

    def test_total_session_turns_stays_consistent_with_root_sent(self) -> None:
        """account_lane_session adds the gated parent's remaining turns to
        total_session_turns so the can_send_any_turn turns-arm
        (root_requests_sent vs total_session_turns) is not biased early."""
        c = CreditCounter(_cfg())
        c.account_lane_session(session_turns=2)
        assert c.total_session_turns == 2
        # Both join turns dispatch -> root_requests_sent catches total exactly.
        c.increment_sent(_turn(idx=1, num=3))
        c.increment_sent(_turn(idx=2, num=3))
        assert c.root_requests_sent == c.total_session_turns == 2
