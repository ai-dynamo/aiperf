# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CreditCallbackHandler handling of ``CreditReturn.ended_on_truncation``.

A truncation-ended return must end its root conversation exactly like an
authored final turn: session counted complete, session slot released, no next
turn dispatched, and the phase able to finish sending when nothing is left.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker


def _credit(
    *,
    corr: str = "a",
    turn_index: int = 0,
    num_turns: int = 3,
    agent_depth: int = 0,
) -> Credit:
    return Credit(
        id=turn_index,
        phase=CreditPhase.PROFILING,
        conversation_id=f"conv-{corr}",
        x_correlation_id=corr,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
        agent_depth=agent_depth,
    )


def _send(progress: PhaseProgressTracker, credit: Credit) -> None:
    progress.increment_sent(
        TurnToSend(
            conversation_id=credit.conversation_id,
            x_correlation_id=credit.x_correlation_id,
            turn_index=credit.turn_index,
            num_turns=credit.num_turns,
            agent_depth=credit.agent_depth,
        )
    )


@pytest.fixture
def concurrency() -> MagicMock:
    return MagicMock()


@pytest.fixture
def strategy() -> MagicMock:
    return MagicMock(handle_credit_return=AsyncMock())


@pytest.fixture
def progress() -> PhaseProgressTracker:
    return PhaseProgressTracker(
        CreditPhaseConfig(
            phase=CreditPhase.PROFILING,
            timing_mode=TimingMode.REQUEST_RATE,
            expected_num_sessions=2,
        )
    )


@pytest.fixture
def handler(
    concurrency: MagicMock, strategy: MagicMock, progress: PhaseProgressTracker
) -> CreditCallbackHandler:
    handler = CreditCallbackHandler(concurrency)
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=progress,
        lifecycle=MagicMock(is_complete=False),
        stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=True)),
        strategy=strategy,
    )
    return handler


@pytest.mark.asyncio
class TestEndedOnTruncation:
    async def test_on_credit_return_truncated_nonfinal_root_ends_session(
        self,
        handler: CreditCallbackHandler,
        progress: PhaseProgressTracker,
        concurrency: MagicMock,
        strategy: MagicMock,
    ) -> None:
        credit = _credit(turn_index=0, num_turns=3)
        _send(progress, credit)

        await handler.on_credit_return(
            "w1", CreditReturn(credit=credit, ended_on_truncation=True)
        )

        assert progress.counter.completed_sessions == 1
        assert progress.counter.sessions_ended_on_truncation == 1
        assert progress.counter.turns_skipped_on_truncation == 2
        assert progress.counter.total_session_turns == 1
        concurrency.release_session_slot.assert_called_once()
        dispatched = strategy.handle_credit_return.await_args.args[0]
        assert dispatched.is_final_turn
        assert dispatched.turn_index == 0

    async def test_on_credit_return_flag_off_continues_conversation(
        self,
        handler: CreditCallbackHandler,
        progress: PhaseProgressTracker,
        concurrency: MagicMock,
        strategy: MagicMock,
    ) -> None:
        credit = _credit(turn_index=0, num_turns=3)
        _send(progress, credit)

        await handler.on_credit_return("w1", CreditReturn(credit=credit))

        assert progress.counter.completed_sessions == 0
        assert progress.counter.sessions_ended_on_truncation == 0
        concurrency.release_session_slot.assert_not_called()
        strategy.handle_credit_return.assert_awaited_once_with(credit, error=None)

    @pytest.mark.parametrize(
        "credit,cancelled",
        [
            param(_credit(turn_index=2, num_turns=3), False, id="final_turn"),
            param(_credit(turn_index=0, num_turns=3), True, id="cancelled"),
        ],
    )  # fmt: skip
    async def test_on_credit_return_flag_ignored_when_not_applicable(
        self,
        handler: CreditCallbackHandler,
        progress: PhaseProgressTracker,
        strategy: MagicMock,
        credit: Credit,
        cancelled: bool,
    ) -> None:
        _send(progress, credit)

        await handler.on_credit_return(
            "w1",
            CreditReturn(credit=credit, cancelled=cancelled, ended_on_truncation=True),
        )

        assert progress.counter.sessions_ended_on_truncation == 0
        if strategy.handle_credit_return.await_args is not None:
            assert strategy.handle_credit_return.await_args.args[0] == credit

    async def test_on_credit_return_last_pending_session_marks_sending_complete(
        self, handler: CreditCallbackHandler, progress: PhaseProgressTracker
    ) -> None:
        done = _credit(corr="a", turn_index=0, num_turns=1)
        truncated = _credit(corr="b", turn_index=0, num_turns=3)
        _send(progress, done)
        _send(progress, truncated)
        await handler.on_credit_return("w1", CreditReturn(credit=done))
        assert not progress.all_credits_sent_event.is_set()

        await handler.on_credit_return(
            "w1", CreditReturn(credit=truncated, ended_on_truncation=True)
        )

        assert progress.all_credits_sent_event.is_set()
        assert progress.counter.final_requests_sent == 2
        assert progress.all_credits_returned_event.is_set()

    async def test_on_credit_return_request_cap_reached_not_counted_as_skipped(
        self, concurrency: MagicMock, strategy: MagicMock
    ) -> None:
        progress = PhaseProgressTracker(
            CreditPhaseConfig(
                phase=CreditPhase.PROFILING,
                timing_mode=TimingMode.REQUEST_RATE,
                total_expected_requests=1,
            )
        )
        handler = CreditCallbackHandler(concurrency)
        handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=progress,
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=False)),
            strategy=strategy,
        )
        credit = _credit(turn_index=0, num_turns=3)
        _send(progress, credit)

        await handler.on_credit_return(
            "w1", CreditReturn(credit=credit, ended_on_truncation=True)
        )

        assert progress.counter.completed_sessions == 1
        assert progress.counter.sessions_ended_on_truncation == 0
        assert progress.counter.turns_skipped_on_truncation == 0
        concurrency.release_session_slot.assert_called_once()

    async def test_on_credit_return_unstarted_sessions_keep_sending_open(
        self, handler: CreditCallbackHandler, progress: PhaseProgressTracker
    ) -> None:
        truncated = _credit(corr="a", turn_index=0, num_turns=3)
        _send(progress, truncated)

        await handler.on_credit_return(
            "w1", CreditReturn(credit=truncated, ended_on_truncation=True)
        )

        assert not progress.all_credits_sent_event.is_set()
        assert progress.counter.final_requests_sent is None
