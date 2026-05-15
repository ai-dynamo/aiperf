# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAG-aware ``is_final_credit`` flip in CreditCounter.

Pairs with RequestCountStopCondition.applies_to_dag_children=True
(P2T20) and request_rate._issue_child_continuation_or_release (P2T18)
to give --request-count literal wire-cap semantics for DAG fan-out.
"""

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter


def cfg(reqs: int | None = None) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=reqs,
        expected_num_sessions=None,
        expected_duration_sec=None,
    )


def root_turn() -> TurnToSend:
    return TurnToSend(
        conversation_id="c-root",
        x_correlation_id="x-root",
        turn_index=0,
        num_turns=1,
        agent_depth=0,
    )


def child_turn() -> TurnToSend:
    return TurnToSend(
        conversation_id="c-root",
        x_correlation_id="x-child",
        turn_index=0,
        num_turns=1,
        agent_depth=1,
        parent_correlation_id="x-root",
    )


class TestIsFinalCreditFlipForChildren:
    def test_child_below_cap_not_final(self) -> None:
        c = CreditCounter(cfg(reqs=10))
        # First send a root request — well below cap.
        _, root_final = c.increment_sent(root_turn())
        assert root_final is False
        # Child increment also below cap — not final.
        _, child_final = c.increment_sent(child_turn())
        assert child_final is False
        # Children DO bump _requests_sent (real wire traffic).
        assert c.requests_sent == 2

    def test_child_at_cap_marks_final(self) -> None:
        c = CreditCounter(cfg(reqs=2))
        # First send a root — requests_sent goes to 1.
        _, root_final = c.increment_sent(root_turn())
        assert root_final is False
        # Child crosses cap — must flip is_final_credit so the
        # strategy loop and phase runner unblock.
        _, child_final = c.increment_sent(child_turn())
        assert child_final is True
        assert c.requests_sent == 2

    def test_child_does_not_bump_session_counters(self) -> None:
        c = CreditCounter(cfg(reqs=10))
        c.increment_sent(root_turn())
        assert c.sent_sessions == 1
        # Child inherits the parent's session slot — does NOT bump
        # session counters even though it bumps requests_sent.
        c.increment_sent(child_turn())
        assert c.sent_sessions == 1
        assert c.total_session_turns == 1

    @pytest.mark.parametrize("reqs", [1, 5, 100])
    def test_child_only_run_flips_at_cap(self, reqs: int) -> None:
        # Edge: nothing prevents a counter from receiving back-to-back
        # children (issuer ordering is not the counter's concern).
        # Verify is_final flips exactly when crossing the cap.
        c = CreditCounter(cfg(reqs=reqs))
        for i in range(1, reqs + 1):
            _, is_final = c.increment_sent(child_turn())
            assert is_final is (i >= reqs)
