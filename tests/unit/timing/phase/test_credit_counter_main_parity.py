# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity + regression tests porting origin/main's CreditCounter fixes.

This branch (``cjq/agentx-v0.3``) forked before origin/main's
``ca590eec feat(dag): conversation DAG benchmarks (dag_jsonl)`` and
re-implemented DAG support its own way. In doing so it lost two fixes that
landed in main and should also hold here:

1. Error accounting: ``increment_returned(..., errored=True)`` bumps
   ``request_errors`` (the timing-layer counter the phase-complete
   ``errors=`` log reads). The branch had dropped the ``errored`` parameter,
   so fault-injected runs always logged ``errors=0`` while the aggregate JSON
   recorded the failures.

2. Root-only session-completion predicate: ``_root_requests_sent`` is used in
   the ``expected_num_sessions`` branch of ``is_final_credit`` instead of the
   global ``requests_sent``, so DAG children (which bump ``requests_sent`` but
   not ``sent_sessions``) cannot prematurely satisfy
   ``sent >= total_session_turns`` and exit the strategy loop before a
   multi-turn parent finishes dispatching.

The final test is the child-triggered version of the session-count invariant
that origin/main violates (1 root + 1 child -> completed=2, sent=1). It passes
HERE because this branch added the ``is_child`` guard to ``increment_returned``
-- so it locks that guard in against regression toward main's behavior.
"""

from __future__ import annotations

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter


def cfg(
    reqs: int | None = None,
    sessions: int | None = None,
    dur: float | None = None,
) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=reqs,
        expected_num_sessions=sessions,
        expected_duration_sec=dur,
    )


def turn(
    conv: str = "c1",
    idx: int = 0,
    num: int = 1,
    corr: str = "x1",
    *,
    depth: int = 0,
    parent: str | None = None,
) -> TurnToSend:
    return TurnToSend(
        conversation_id=conv,
        turn_index=idx,
        num_turns=num,
        x_correlation_id=corr,
        agent_depth=depth,
        parent_correlation_id=parent,
    )


class TestErrorAccountingPortedFromMain:
    """Ported from origin/main test_credit_counter.py (added in ca590eec)."""

    def test_increment_returned_errored_bumps_request_errors(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=False, errored=True)
        # Errored requests still count as "returned" (not "cancelled") to
        # preserve the all-returned invariant, but ALSO bump the error counter.
        assert c.requests_completed == 1
        assert c.requests_cancelled == 0
        assert c.request_errors == 1

    def test_increment_returned_cancelled_with_error_flag_does_not_double_count(
        self,
    ) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=True, errored=True)
        # Cancellation is the dominant signal for the cancelled path.
        assert c.requests_cancelled == 1
        assert c.requests_completed == 0
        assert c.request_errors == 0

    def test_increment_returned_errored_freezes_into_final_errors(self) -> None:
        c = CreditCounter(cfg())
        for _ in range(3):
            c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=False, errored=True)
        c.increment_returned(is_final_turn=False, cancelled=False, errored=True)
        c.increment_returned(is_final_turn=False, cancelled=False, errored=False)
        c.freeze_sent_counts()
        c.freeze_completed_counts()
        assert c.final_request_errors == 2

    def test_child_errored_return_still_bumps_request_errors(self) -> None:
        """request_errors is request-level, so a DAG child's errored return
        bumps it too (symmetric with requests_completed ticking for children).
        """
        c = CreditCounter(cfg())
        c.increment_sent(turn(depth=1, parent="root", corr="child"))
        c.increment_returned(
            is_final_turn=True, cancelled=False, errored=True, is_child=True
        )
        assert c.request_errors == 1
        assert c.requests_completed == 1
        # Session-level counters stay root-only for children.
        assert c.completed_sessions == 0


class TestRootOnlySessionPredicate:
    """``_root_requests_sent`` keeps DAG children from prematurely flipping
    ``is_final_credit`` on the ``expected_num_sessions`` path (origin/main fix)."""

    def test_child_wire_does_not_prematurely_satisfy_session_predicate(self) -> None:
        # One session expected, a 3-turn root. Children fire between root turns.
        c = CreditCounter(cfg(sessions=1))

        # Root turn 0 (session starts; total_session_turns -> 3).
        _, final0 = c.increment_sent(turn(idx=0, num=3, corr="root"))
        assert final0 is False

        # Two DAG children fire (reactive, off the phase target). With the
        # global-count predicate these would push requests_sent to 3 and
        # spuriously satisfy ``sent >= total_session_turns`` on the next root
        # turn. counts_toward_phase_target=False keeps children from flipping
        # is_final directly; _root_requests_sent keeps them out of the root
        # session predicate too.
        c.increment_sent(
            TurnToSend(
                conversation_id="child",
                x_correlation_id="c-a",
                turn_index=0,
                num_turns=1,
                agent_depth=1,
                parent_correlation_id="root",
                counts_toward_phase_target=False,
            )
        )
        c.increment_sent(
            TurnToSend(
                conversation_id="child",
                x_correlation_id="c-b",
                turn_index=0,
                num_turns=1,
                agent_depth=1,
                parent_correlation_id="root",
                counts_toward_phase_target=False,
            )
        )

        # Root turn 1: NOT final -- only 2 of 3 root turns sent, regardless of
        # the 2 child wires inflating requests_sent to 4.
        _, final1 = c.increment_sent(turn(idx=1, num=3, corr="root"))
        assert c.requests_sent == 4
        assert c.root_requests_sent == 2
        assert final1 is False, "child wires must not satisfy the session predicate"

        # Root turn 2: now the session's final root turn -> is_final.
        _, final2 = c.increment_sent(turn(idx=2, num=3, corr="root"))
        assert c.root_requests_sent == 3
        assert final2 is True


class TestChildSessionCountInvariantRegression:
    """Regression guard for the bug origin/main still has (this branch fixed).

    origin/main's ``increment_returned`` lacks the ``is_child`` guard, so a DAG
    child's final turn bumps ``completed_sessions`` without a matching
    ``sent_sessions`` -> ``completed_sessions > sent_sessions`` and
    ``in_flight_sessions < 0``. This branch's ``is_child`` guard prevents it;
    this test fails if that guard is ever removed (regressing toward main).
    """

    def test_dag_children_do_not_inflate_completed_sessions(self) -> None:
        c = CreditCounter(cfg())
        # 1 root (single turn) + 1 DAG child (single turn).
        c.increment_sent(turn(idx=0, num=1, corr="root"))
        c.increment_sent(turn(idx=0, num=1, corr="child", depth=1, parent="root"))
        # Root final return (real session completion).
        c.increment_returned(is_final_turn=True, cancelled=False, is_child=False)
        # Child final return -- must NOT bump completed_sessions.
        c.increment_returned(is_final_turn=True, cancelled=False, is_child=True)

        assert c.sent_sessions == 1
        assert c.completed_sessions == 1
        assert c.completed_sessions <= c.sent_sessions
        assert c.in_flight_sessions == 0

    def test_cancelled_child_does_not_inflate_cancelled_sessions(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=1, corr="root"))
        c.increment_sent(turn(idx=0, num=1, corr="child", depth=1, parent="root"))
        c.increment_returned(is_final_turn=True, cancelled=True, is_child=False)
        c.increment_returned(is_final_turn=True, cancelled=True, is_child=True)
        assert c.sent_sessions == 1
        assert c.cancelled_sessions == 1
        assert c.cancelled_sessions <= c.sent_sessions
        assert c.in_flight_sessions == 0
