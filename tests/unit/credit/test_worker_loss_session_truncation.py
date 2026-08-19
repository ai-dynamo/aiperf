# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A session truncated by worker loss must still let the phase terminate.

When a sticky session's worker disappears and the credit forbids migration, the
router synthesizes a ``worker_unavailable:`` cancelled return and the session's
remaining turns are never issued. The phase plan booked that session's FULL turn
count when it started, so unless the plan is reconciled the deficit is permanent:
``root_requests_sent`` can never reach ``total_session_turns``, which means

- ``SessionCountStopCondition.can_send_any_turn`` is True forever, so
  ``RequestRateStrategy``'s main loop falls into its ``yield_to_event_loop``
  arm every iteration and burns a core for the life of the process, and
- ``freeze_sent_counts`` is never called, so ``final_requests_sent`` stays None
  and ``check_all_returned_or_cancelled`` is False forever -- with no
  ``--benchmark-duration`` the run never terminates at all.

The counterpart risk is a fix that fires the send-target predicate EARLY on a
normal run, so the untruncated path is pinned here too.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.credit_counter import CreditCounter
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import SessionCountStopCondition

WORKER_LOST = "worker_unavailable: worker stopped responding before next turn"


def cfg(sessions: int | None = None, reqs: int | None = None) -> CreditPhaseConfig:
    return CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        total_expected_requests=reqs,
        expected_num_sessions=sessions,
    )


def turn(corr: str, idx: int, num: int) -> TurnToSend:
    return TurnToSend(
        conversation_id="c1", x_correlation_id=corr, turn_index=idx, num_turns=num
    )


def credit(corr: str, idx: int, num: int) -> Credit:
    return Credit(
        id=idx,
        phase=CreditPhase.PROFILING,
        conversation_id="c1",
        x_correlation_id=corr,
        turn_index=idx,
        num_turns=num,
        issued_at_ns=0,
    )


class TestCounterPlanReconciliation:
    def test_truncated_session_lets_the_send_target_converge(self) -> None:
        """Two 3-turn sessions; the second dies after its first turn.

        Without reconciliation the plan stays at 6 turns against 4 root sends
        and the predicate can never be satisfied.
        """
        c = CreditCounter(cfg(sessions=2))
        c.increment_sent(turn("s1", 0, 3))
        c.increment_sent(turn("s1", 1, 3))
        _, final = c.increment_sent(turn("s1", 2, 3))
        assert final is False
        c.increment_sent(turn("s2", 0, 3))
        assert c.total_session_turns == 6 and c.root_requests_sent == 4

        # s2 dies on turn 0 of 3: turns 1 and 2 will never be issued.
        assert c.retire_unsent_session_turns(2) is True
        assert c.total_session_turns == 4
        assert SessionCountStopCondition(cfg(sessions=2), MagicMock(), c).can_send_any_turn() is False  # fmt: skip

    def test_truncation_mid_plan_does_not_signal_early(self) -> None:
        """A session dying while other sessions are still unstarted must not
        claim the plan is fully issued."""
        c = CreditCounter(cfg(sessions=3))
        c.increment_sent(turn("s1", 0, 3))
        assert c.retire_unsent_session_turns(2) is False
        assert c.total_session_turns == 1
        # Two sessions are still unstarted, so the phase must keep sending.
        cond = SessionCountStopCondition(cfg(sessions=3), MagicMock(), c)
        assert cond.can_send_any_turn() is True
        assert cond.can_start_new_session() is True

    def test_no_signal_when_predicate_already_satisfied(self) -> None:
        """The last root credit already flipped ``is_final_credit``; retiring a
        later truncation must not re-fire the one-shot sending-complete pair."""
        c = CreditCounter(cfg(sessions=1))
        _, final = c.increment_sent(turn("s1", 0, 1))
        assert final is True
        assert c.retire_unsent_session_turns(1) is False

    @pytest.mark.parametrize("unsent", [0, -1])
    def test_non_positive_retire_is_a_no_op(self, unsent: int) -> None:
        c = CreditCounter(cfg(sessions=1))
        c.increment_sent(turn("s1", 0, 2))
        assert c.retire_unsent_session_turns(unsent) is False
        assert c.total_session_turns == 2


class TestUntruncatedPathUnchanged:
    def test_is_final_credit_still_fires_only_on_the_last_planned_turn(self) -> None:
        """Guard against the reconciliation making the predicate fire early.

        Two 3-turn sessions: every credit but the very last must report
        ``is_final_credit`` False, and the plan must stay at 6 throughout.
        """
        c = CreditCounter(cfg(sessions=2))
        plan = [("s1", 0), ("s1", 1), ("s1", 2), ("s2", 0), ("s2", 1), ("s2", 2)]
        finals = []
        for corr, idx in plan:
            _, is_final = c.increment_sent(turn(corr, idx, 3))
            finals.append(is_final)
        assert finals == [False, False, False, False, False, True]
        assert c.total_session_turns == 6
        assert c.root_requests_sent == 6
        assert c.sent_sessions == 2


class TestCallbackHandlerReconciliation:
    """The handler is the site that observes the truncation and must both
    shrink the plan and emit the sending-complete signal ``CreditIssuer`` would
    otherwise have emitted on the final credit."""

    @pytest.fixture
    def wired(self):
        config = cfg(sessions=2)
        progress = PhaseProgressTracker(config)
        lifecycle = MagicMock()
        lifecycle.is_complete = False
        stop_checker = MagicMock()
        stop_checker.can_send_any_turn = MagicMock(return_value=True)
        strategy = MagicMock()
        strategy.handle_credit_return = AsyncMock()
        strategy.handle_session_ended = AsyncMock()
        del strategy.handle_credit_result
        del strategy.handle_first_token
        del strategy.observe_credit_return
        del strategy.enforce_system_idle_cap
        handler = CreditCallbackHandler(MagicMock())
        handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=progress,
            lifecycle=lifecycle,
            stop_checker=stop_checker,
            strategy=strategy,
        )
        return handler, progress, strategy

    @pytest.mark.asyncio
    async def test_worker_loss_truncation_completes_the_phase(self, wired) -> None:
        handler, progress, strategy = wired
        # Session 1 runs to completion (3 turns), session 2 dies on turn 0.
        for idx in range(3):
            progress.increment_sent(turn("s1", idx, 3))
        progress.increment_sent(turn("s2", 0, 3))

        lost = credit("s2", 0, 3)
        await handler.on_credit_return(
            "w-gone",
            CreditReturn(
                credit=lost, cancelled=True, error=WORKER_LOST, first_token_sent=False
            ),
        )

        # Plan shrank to what was actually issued...
        assert progress.counter.total_session_turns == 4
        # ...the strategy was told the session is over...
        strategy.handle_session_ended.assert_awaited_once()
        # ...sending was frozen and signalled (previously never happened)...
        assert progress.counter.final_requests_sent == 4
        assert progress.all_credits_sent_event.is_set()
        # ...and the completion predicate can now converge.
        for idx in range(3):
            progress.increment_returned(idx == 2, False)
        assert progress.check_all_returned_or_cancelled() is True
        assert (
            SessionCountStopCondition(
                cfg(sessions=2), MagicMock(), progress.counter
            ).can_send_any_turn()
            is False
        )

    @pytest.mark.asyncio
    async def test_normal_return_leaves_the_plan_alone(self, wired) -> None:
        """A non-truncating return must not touch the plan or signal anything."""
        handler, progress, _ = wired
        progress.increment_sent(turn("s1", 0, 3))
        await handler.on_credit_return(
            "w-1",
            CreditReturn(
                credit=credit("s1", 0, 3),
                cancelled=False,
                error=None,
                first_token_sent=True,
            ),
        )
        assert progress.counter.total_session_turns == 3
        assert progress.counter.final_requests_sent is None
        assert not progress.all_credits_sent_event.is_set()

    @pytest.mark.asyncio
    async def test_migratable_session_is_not_retired(self, wired) -> None:
        """``allow_worker_migration`` sessions continue on another worker, so
        their remaining turns are still part of the plan."""
        handler, progress, _ = wired
        progress.increment_sent(turn("s1", 0, 3))
        migratable = Credit(
            id=0,
            phase=CreditPhase.PROFILING,
            conversation_id="c1",
            x_correlation_id="s1",
            turn_index=0,
            num_turns=3,
            issued_at_ns=0,
            allow_worker_migration=True,
        )
        await handler.on_credit_return(
            "w-gone",
            CreditReturn(
                credit=migratable,
                cancelled=True,
                error=WORKER_LOST,
                first_token_sent=False,
            ),
        )
        assert progress.counter.total_session_turns == 3
        assert not progress.all_credits_sent_event.is_set()


@pytest.mark.asyncio
async def test_rate_loop_arm_stops_spinning_after_truncation() -> None:
    """The concrete hot-spin: ``RequestRateStrategy``'s loop falls to its
    ``yield_to_event_loop`` arm whenever it cannot start a session, has no
    queued continuation, and ``can_send_any_turn()`` is still True.
    """
    config = cfg(sessions=1)
    counter = CreditCounter(config)
    counter.increment_sent(turn("s1", 0, 5))
    cond = SessionCountStopCondition(config, MagicMock(), counter)

    # Session cap reached, plan says 4 turns still owed -> the spin arm.
    assert cond.can_start_new_session() is False
    assert cond.can_send_any_turn() is True

    counter.retire_unsent_session_turns(4)
    assert cond.can_send_any_turn() is False
    await asyncio.sleep(0)
