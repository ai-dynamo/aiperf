# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for CreditCounter DAG accounting."""

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


class TestErrorAccounting:
    """Error returns update request-level accounting."""

    def test_increment_returned_errored_bumps_request_errors(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=False, errored=True)
        assert c.requests_completed == 1
        assert c.requests_cancelled == 0
        assert c.request_errors == 1

    def test_increment_returned_cancelled_with_error_flag_does_not_double_count(
        self,
    ) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn())
        c.increment_returned(is_final_turn=False, cancelled=True, errored=True)
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
        """request_errors is request-level, so a DAG child's errored return"""
        c = CreditCounter(cfg())
        c.increment_sent(turn(depth=1, parent="root", corr="child"))
        c.increment_returned(
            is_final_turn=True, cancelled=False, errored=True, is_child=True
        )
        assert c.request_errors == 1
        assert c.requests_completed == 1
        assert c.completed_sessions == 0


class TestRootOnlySessionPredicate:
    """``_root_requests_sent`` keeps DAG children from prematurely flipping"""

    def test_child_wire_does_not_prematurely_satisfy_session_predicate(self) -> None:
        c = CreditCounter(cfg(sessions=1))

        _, final0 = c.increment_sent(turn(idx=0, num=3, corr="root"))
        assert final0 is False

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

        _, final1 = c.increment_sent(turn(idx=1, num=3, corr="root"))
        assert c.requests_sent == 4
        assert c.root_requests_sent == 2
        assert final1 is False, "child wires must not satisfy the session predicate"

        _, final2 = c.increment_sent(turn(idx=2, num=3, corr="root"))
        assert c.root_requests_sent == 3
        assert final2 is True


class TestChildSessionCountInvariantRegression:
    """DAG child returns do not affect root-session counters."""

    def test_dag_children_do_not_inflate_completed_sessions(self) -> None:
        c = CreditCounter(cfg())
        c.increment_sent(turn(idx=0, num=1, corr="root"))
        c.increment_sent(turn(idx=0, num=1, corr="child", depth=1, parent="root"))
        c.increment_returned(is_final_turn=True, cancelled=False, is_child=False)
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
