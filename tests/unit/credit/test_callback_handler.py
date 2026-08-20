# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CreditCallbackHandler.

Tests credit lifecycle callbacks from CreditRouter.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_concurrency():
    """Mock concurrency manager."""
    mock = MagicMock()
    mock.release_session_slot = MagicMock()
    mock.release_prefill_slot = MagicMock()
    return mock


@pytest.fixture
def mock_progress():
    """Mock progress tracker."""
    mock = MagicMock()
    mock.increment_returned = MagicMock(return_value=False)  # Not final return
    mock.increment_prefill_released = MagicMock()
    mock.all_credits_returned_event = asyncio.Event()
    mock.in_flight_sessions = 0
    return mock


@pytest.fixture
def mock_lifecycle():
    """Mock phase lifecycle."""
    mock = MagicMock()
    mock.is_complete = False
    return mock


@pytest.fixture
def mock_stop_checker():
    """Mock stop condition checker."""
    mock = MagicMock()
    mock.can_send_any_turn = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_strategy():
    """Mock timing strategy."""
    mock = MagicMock()
    mock.handle_credit_return = AsyncMock()
    return mock


@pytest.fixture
def callback_handler(mock_concurrency):
    """Create CreditCallbackHandler."""
    return CreditCallbackHandler(mock_concurrency)


@pytest.fixture
def mock_branch_orchestrator():
    """Mock BranchOrchestrator that records ``set_drain_observer`` calls."""
    mock = MagicMock()
    mock.set_drain_observer = MagicMock()
    return mock


@pytest.fixture
def registered_handler(
    callback_handler,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
):
    """Create CreditCallbackHandler with phase registered."""
    callback_handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    return callback_handler


def make_credit(
    credit_id: int = 1,
    conversation_id: str = "conv1",
    turn_index: int = 0,
    num_turns: int = 1,
    phase: CreditPhase = CreditPhase.PROFILING,
    phase_index: int | None = None,
    agent_depth: int = 0,
    allow_worker_migration: bool = False,
) -> Credit:
    """Create a Credit for testing."""
    return Credit(
        id=credit_id,
        phase=phase,
        phase_index=phase_index,
        conversation_id=conversation_id,
        x_correlation_id=f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
        agent_depth=agent_depth,
        allow_worker_migration=allow_worker_migration,
    )


def make_credit_return(
    credit: Credit,
    cancelled: bool = False,
    first_token_sent: bool = True,
    error: str | None = None,
) -> CreditReturn:
    """Create a CreditReturn for testing."""
    return CreditReturn(
        credit=credit,
        cancelled=cancelled,
        first_token_sent=first_token_sent,
        error=error,
    )


# =============================================================================
# Test: Phase Registration
# =============================================================================


class TestPhaseRegistration:
    """Tests for phase registration."""

    def test_register_phase_updates_handlers(self, callback_handler):
        """Register phase correctly updates handlers."""
        progress = MagicMock()
        progress.all_credits_returned_event = asyncio.Event()

        callback_handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=progress,
            lifecycle=MagicMock(),
            stop_checker=MagicMock(),
            strategy=MagicMock(),
        )

        assert CreditPhase.PROFILING in callback_handler._phase_handlers

    async def test_register_phase_same_kind_phases_uses_runtime_index(
        self,
        callback_handler,
        mock_concurrency,
        mock_lifecycle,
        mock_stop_checker,
        mock_strategy,
    ):
        """Two profiling phases must not overwrite each other's callbacks."""
        progress_0 = MagicMock()
        progress_0.all_credits_returned_event = asyncio.Event()
        progress_0.increment_returned.return_value = False
        progress_1 = MagicMock()
        progress_1.all_credits_returned_event = asyncio.Event()
        progress_1.increment_returned.return_value = False

        callback_handler.register_phase(
            phase=CreditPhase.PROFILING,
            phase_index=0,
            progress=progress_0,
            lifecycle=mock_lifecycle,
            stop_checker=mock_stop_checker,
            strategy=mock_strategy,
        )
        callback_handler.register_phase(
            phase=CreditPhase.PROFILING,
            phase_index=1,
            progress=progress_1,
            lifecycle=mock_lifecycle,
            stop_checker=mock_stop_checker,
            strategy=mock_strategy,
        )

        await callback_handler.on_credit_return(
            "worker-1", make_credit_return(make_credit(phase_index=1))
        )
        await callback_handler.on_first_token(
            FirstToken(
                credit_id=1,
                phase=CreditPhase.PROFILING,
                phase_index=1,
                ttft_ns=1000000,
            )
        )

        progress_0.increment_returned.assert_not_called()
        progress_0.increment_prefill_released.assert_not_called()
        progress_1.increment_returned.assert_called_once()
        progress_1.increment_prefill_released.assert_called_once()
        mock_concurrency.release_session_slot.assert_called_once_with(1)
        mock_concurrency.release_prefill_slot.assert_called_once_with(1)

    async def test_same_kind_phase_orchestrators_route_and_detach_independently(
        self,
        callback_handler,
        mock_concurrency,
        mock_lifecycle,
        mock_stop_checker,
    ):
        """A later named phase must not replace a seamless phase still draining."""
        orchestrators = [MagicMock(), MagicMock()]
        strategies = [MagicMock(), MagicMock()]
        progresses = [MagicMock(), MagicMock()]
        for phase_index in range(2):
            orchestrator = orchestrators[phase_index]
            orchestrator.intercept = AsyncMock(return_value=True)
            orchestrator.has_pending_branch_work.return_value = False
            orchestrator.get_branch_ids.return_value = []
            strategy = strategies[phase_index]
            strategy.handle_credit_return = AsyncMock()
            progress = progresses[phase_index]
            progress.all_credits_returned_event = asyncio.Event()
            progress.increment_returned.return_value = False
            progress.check_all_returned_or_cancelled.return_value = False
            callback_handler.register_phase(
                phase=CreditPhase.PROFILING,
                phase_index=phase_index,
                progress=progress,
                lifecycle=mock_lifecycle,
                stop_checker=mock_stop_checker,
                strategy=strategy,
            )
            callback_handler.set_branch_orchestrator(
                orchestrator,
                phase=CreditPhase.PROFILING,
                phase_index=phase_index,
            )

        phase_zero_credit = make_credit(phase_index=0, conversation_id="phase-zero")
        phase_one_credit = make_credit(phase_index=1, conversation_id="phase-one")
        await callback_handler.on_credit_return(
            "worker-0", make_credit_return(phase_zero_credit)
        )
        await callback_handler.on_credit_return(
            "worker-1", make_credit_return(phase_one_credit)
        )

        orchestrators[0].intercept.assert_awaited_once_with(phase_zero_credit)
        orchestrators[1].intercept.assert_awaited_once_with(phase_one_credit)

        callback_handler.set_branch_orchestrator(
            None,
            phase=CreditPhase.PROFILING,
            phase_index=0,
        )
        assert callback_handler._orchestrator_for(0, CreditPhase.PROFILING) is None
        assert (
            callback_handler._orchestrator_for(1, CreditPhase.PROFILING)
            is orchestrators[1]
        )
        orchestrators[0].set_drain_observer.assert_called_with(None)


# =============================================================================
# Test: Credit Return - Basic Flow
# =============================================================================


class TestCreditReturnBasicFlow:
    """Tests for basic credit return handling."""

    async def test_on_credit_return_increments_returned_count(
        self, registered_handler, mock_progress
    ):
        """Credit return should increment returned count."""
        credit = make_credit()
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_progress.increment_returned.assert_called_once_with(
            credit.is_final_turn,
            False,  # cancelled=False
            session_ended=True,
            session_cancelled=False,
            errored=False,
            is_child=False,  # agent_depth=0 root credit
            no_request=False,
        )

    async def test_on_credit_return_checks_global_idle_after_dispatch(
        self, registered_handler, mock_progress, mock_strategy
    ):
        """The strategy sees the final in-flight count after return dispatch."""
        mock_progress.in_flight = 0
        credit = make_credit(turn_index=0, num_turns=2)

        await registered_handler.on_credit_return(
            "worker-1", make_credit_return(credit)
        )

        mock_strategy.handle_credit_return.assert_awaited_once()
        mock_strategy.enforce_system_idle_cap.assert_called_once_with(0)

    async def test_on_credit_return_tracks_cancelled_status(
        self, registered_handler, mock_progress
    ):
        """Credit return should track cancelled status."""
        credit = make_credit()
        credit_return = make_credit_return(credit, cancelled=True)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_progress.increment_returned.assert_called_once_with(
            credit.is_final_turn,
            True,  # cancelled=True
            session_ended=True,
            session_cancelled=True,
            errored=False,
            is_child=False,  # agent_depth=0 root credit
            no_request=False,
        )

    async def test_on_credit_return_notifies_result_aware_strategy(
        self: "TestCreditReturnBasicFlow",
        callback_handler: CreditCallbackHandler,
        mock_progress: MagicMock,
        mock_lifecycle: MagicMock,
        mock_stop_checker: MagicMock,
        mock_strategy: MagicMock,
    ) -> None:
        """Strategies with a result hook should receive full return status."""
        mock_strategy.handle_credit_result = AsyncMock()
        callback_handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=mock_progress,
            lifecycle=mock_lifecycle,
            stop_checker=mock_stop_checker,
            strategy=mock_strategy,
        )
        credit = make_credit()
        credit_return = make_credit_return(
            credit, cancelled=True, error="worker failed"
        )

        await callback_handler.on_credit_return("worker-1", credit_return)

        mock_strategy.handle_credit_result.assert_awaited_once_with(credit_return)

    async def test_worker_loss_ends_non_migratable_root_session(
        self,
        registered_handler,
        mock_progress,
        mock_strategy,
        mock_concurrency,
    ) -> None:
        mock_strategy.handle_session_ended = AsyncMock()
        credit = make_credit(turn_index=0, num_turns=3)
        credit_return = make_credit_return(
            credit,
            cancelled=True,
            first_token_sent=False,
            error="worker_unavailable: worker lost mid-session",
        )

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_strategy.handle_credit_return.assert_not_awaited()
        mock_strategy.handle_session_ended.assert_awaited_once_with(credit)
        mock_progress.increment_returned.assert_called_once_with(
            False,
            True,
            session_ended=True,
            session_cancelled=True,
            errored=True,
            is_child=False,
            no_request=False,
        )
        mock_concurrency.release_session_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )

    async def test_worker_loss_continues_migratable_root_session(
        self, registered_handler, mock_strategy, mock_concurrency
    ) -> None:
        credit = make_credit(turn_index=0, num_turns=3, allow_worker_migration=True)
        credit_return = make_credit_return(
            credit,
            cancelled=True,
            first_token_sent=False,
            error="worker_unavailable: worker lost mid-session",
        )

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_strategy.handle_credit_return.assert_awaited_once_with(
            credit, error=credit_return.error
        )
        mock_concurrency.release_session_slot.assert_not_called()

    async def test_worker_loss_stops_nonfinal_dag_child_without_intercept(
        self,
        registered_handler,
        mock_strategy,
        mock_branch_orchestrator,
    ) -> None:
        mock_branch_orchestrator.on_child_stopped = AsyncMock()
        mock_branch_orchestrator.intercept = AsyncMock(return_value=False)
        mock_branch_orchestrator.has_pending_branch_work.return_value = True
        registered_handler.set_branch_orchestrator(
            mock_branch_orchestrator,
            phase=CreditPhase.PROFILING,
        )
        credit = make_credit(turn_index=0, num_turns=3, agent_depth=1)

        await registered_handler.on_credit_return(
            "worker-1",
            make_credit_return(
                credit,
                cancelled=True,
                first_token_sent=False,
                error="worker_unavailable: worker lost mid-session",
            ),
        )

        mock_branch_orchestrator.intercept.assert_not_awaited()
        mock_branch_orchestrator.on_child_stopped.assert_awaited_once_with(
            credit.x_correlation_id
        )
        mock_strategy.handle_credit_return.assert_not_awaited()

    async def test_result_hook_is_cached_at_phase_registration(
        self: "TestCreditReturnBasicFlow",
        registered_handler: CreditCallbackHandler,
        mock_strategy: MagicMock,
    ) -> None:
        """Credit returns should not rediscover optional hooks on the hot path."""
        late_hook = AsyncMock()
        mock_strategy.handle_credit_result = late_hook
        credit = make_credit()
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        late_hook.assert_not_awaited()

    async def test_on_credit_return_releases_session_slot_on_final_turn(
        self, registered_handler, mock_concurrency
    ):
        """Should release session slot when final turn returns."""
        credit = make_credit(turn_index=2, num_turns=3)  # Final turn
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_concurrency.release_session_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )

    async def test_on_credit_return_does_not_release_session_on_non_final_turn(
        self, registered_handler, mock_concurrency
    ):
        """Should NOT release session slot on non-final turn."""
        credit = make_credit(turn_index=0, num_turns=3)  # Not final
        credit_return = make_credit_return(credit)

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_concurrency.release_session_slot.assert_not_called()


# =============================================================================
# Test: Credit Return - TTFT Handling
# =============================================================================


class TestCreditReturnTTFTHandling:
    """Tests for TTFT-related handling in credit returns."""

    async def test_prefill_slot_released_only_when_ttft_not_sent(
        self, registered_handler, mock_progress, mock_concurrency
    ):
        """Prefill slot released when first_token_sent is False, not when True."""
        # No TTFT case
        credit_no_ttft = make_credit()
        credit_return_no_ttft = make_credit_return(
            credit_no_ttft, first_token_sent=False
        )
        await registered_handler.on_credit_return("worker-1", credit_return_no_ttft)

        mock_progress.increment_prefill_released.assert_called_once()
        mock_concurrency.release_prefill_slot.assert_called_once()

        # Reset mocks
        mock_progress.reset_mock()
        mock_concurrency.reset_mock()

        # With TTFT case
        credit_with_ttft = make_credit(credit_id=2)
        credit_return_with_ttft = make_credit_return(
            credit_with_ttft, first_token_sent=True
        )
        await registered_handler.on_credit_return("worker-1", credit_return_with_ttft)

        mock_progress.increment_prefill_released.assert_not_called()
        mock_concurrency.release_prefill_slot.assert_not_called()


# =============================================================================
# Test: Credit Return - Final Return Handling
# =============================================================================


class TestCreditReturnFinalHandling:
    """Tests for final return handling."""

    async def test_final_return_sets_event_and_releases_in_flight_slots(
        self, callback_handler, mock_concurrency
    ):
        """Final return sets event and releases in-flight session slots."""
        progress = MagicMock()
        progress.all_credits_returned_event = asyncio.Event()
        progress.increment_returned = MagicMock(return_value=True)  # Final return
        progress.increment_prefill_released = MagicMock()
        progress.in_flight_sessions = 2

        callback_handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=progress,
            lifecycle=MagicMock(is_complete=False),
            stop_checker=MagicMock(can_send_any_turn=MagicMock(return_value=False)),
            strategy=MagicMock(handle_credit_return=AsyncMock()),
        )

        credit = make_credit(turn_index=0, num_turns=1)  # Final turn
        credit_return = make_credit_return(credit)

        await callback_handler.on_credit_return("worker-1", credit_return)

        assert progress.all_credits_returned_event.is_set()
        # Should release 2 in-flight session slots + 1 for final turn
        assert mock_concurrency.release_session_slot.call_count == 3


# =============================================================================
# Test: Credit Return - Next Turn Dispatch
# =============================================================================


class TestNextTurnDispatch:
    """Tests for next turn dispatch via strategy."""

    async def test_dispatches_when_can_send_not_when_stopped(
        self, registered_handler, mock_strategy, mock_stop_checker
    ):
        """Dispatches to strategy when can_send_any_turn, skips when stopped."""
        # Can send case
        credit = make_credit(turn_index=0, num_turns=3)
        credit_return = make_credit_return(credit)
        await registered_handler.on_credit_return("worker-1", credit_return)
        mock_strategy.handle_credit_return.assert_called_once_with(credit, error=None)

        # Stop condition reached
        mock_strategy.reset_mock()
        mock_stop_checker.can_send_any_turn.return_value = False
        credit2 = make_credit(credit_id=2, turn_index=0, num_turns=3)
        credit_return2 = make_credit_return(credit2)
        await registered_handler.on_credit_return("worker-1", credit_return2)
        mock_strategy.handle_credit_return.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_open_tree_uses_registry_terminal_path(
    mock_concurrency,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
):
    """Accelerated warmup roots drain through SessionTreeRegistry."""
    registry = MagicMock()
    registry.has_tree.return_value = True
    handler = CreditCallbackHandler(mock_concurrency, session_tree_registry=registry)
    handler.register_phase(
        phase=CreditPhase.WARMUP,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    credit = make_credit(phase=CreditPhase.WARMUP)

    await handler.on_credit_return("worker-1", make_credit_return(credit))

    registry.on_root_terminal.assert_called_once_with(
        credit.effective_root_correlation_id
    )
    mock_concurrency.release_session_slot.assert_not_called()


@pytest.mark.asyncio
async def test_root_context_overflow_calls_registry_on_root_terminal(
    mock_concurrency,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
):
    """Non-final root context-overflow must clear root_pending via registry.

    Agentic replay early-returns under the tree registry expecting the
    callback handler to have already called ``on_root_terminal``; gating that
    call on ``is_final_turn`` alone leaks the session slot forever.
    """
    registry = MagicMock()
    registry.has_tree.return_value = True
    handler = CreditCallbackHandler(mock_concurrency, session_tree_registry=registry)
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    credit = make_credit(turn_index=1, num_turns=5)  # non-final
    await handler.on_credit_return(
        "worker-1",
        make_credit_return(
            credit,
            error="This model's maximum context length is 131072 tokens",
        ),
    )

    registry.on_root_terminal.assert_called_once_with(
        credit.effective_root_correlation_id
    )
    mock_strategy.handle_credit_return.assert_awaited_once()


@pytest.mark.asyncio
async def test_final_turn_context_overflow_still_runs_intercept(
    mock_concurrency,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
):
    """A FINAL-turn context-overflow is NOT overflow_terminal.

    overflow_terminal is scoped to non-final turns (a mid-trajectory death);
    on the authored final turn, intercept must still run so the turn's declared
    DAG children dispatch, and the tree is marked terminal because it is the
    final turn -- not because the error was an overflow.
    """
    registry = MagicMock()
    registry.has_tree.return_value = True
    handler = CreditCallbackHandler(mock_concurrency, session_tree_registry=registry)
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    orchestrator = MagicMock()
    orchestrator.intercept = AsyncMock(return_value=False)
    orchestrator.set_drain_observer = MagicMock()
    handler.set_branch_orchestrator(orchestrator)

    credit = make_credit(turn_index=4, num_turns=5)  # final turn
    await handler.on_credit_return(
        "worker-1",
        make_credit_return(
            credit,
            error="This model's maximum context length is 131072 tokens",
        ),
    )

    orchestrator.intercept.assert_awaited_once_with(credit)
    registry.on_root_terminal.assert_called_once_with(
        credit.effective_root_correlation_id
    )


@pytest.mark.asyncio
async def test_root_non_overflow_error_skips_registry_on_root_terminal(
    mock_concurrency,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
):
    """Generic mid-trajectory errors must not mark the root terminal."""
    registry = MagicMock()
    registry.has_tree.return_value = True
    handler = CreditCallbackHandler(mock_concurrency, session_tree_registry=registry)
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    credit = make_credit(turn_index=1, num_turns=5)
    await handler.on_credit_return(
        "worker-1",
        make_credit_return(credit, error="Internal server error: pool exhausted"),
    )

    registry.on_root_terminal.assert_not_called()
    mock_strategy.handle_credit_return.assert_awaited_once()


_OVERFLOW_ERROR = "This model's maximum context length is 131072 tokens"


@pytest.mark.asyncio
async def test_overflow_skips_intercept_and_runs_terminal_path(
    mock_concurrency,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
    mock_branch_orchestrator,
):
    """Overflow terminal must not honor gated-suspend early-return (R1).

    Even when ``intercept`` would return True (gated next turn), a
    context-overflow root return must still call ``on_root_terminal``,
    ``strategy.handle_credit_return`` (overflow recycle), and skip calling
    intercept entirely.
    """
    registry = MagicMock()
    registry.has_tree.return_value = True
    mock_branch_orchestrator.intercept = AsyncMock(return_value=True)
    mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
    handler = CreditCallbackHandler(
        mock_concurrency,
        branch_orchestrator=mock_branch_orchestrator,
        session_tree_registry=registry,
    )
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    credit = make_credit(turn_index=1, num_turns=5)  # non-final
    assert not credit.is_final_turn

    await handler.on_credit_return(
        "worker-1",
        make_credit_return(credit, error=_OVERFLOW_ERROR),
    )

    mock_branch_orchestrator.intercept.assert_not_awaited()
    registry.on_root_terminal.assert_called_once_with(
        credit.effective_root_correlation_id
    )
    mock_strategy.handle_credit_return.assert_awaited_once_with(
        credit, error=_OVERFLOW_ERROR
    )


@pytest.mark.asyncio
async def test_overflow_with_intercept_true_still_records_warmup_failure(
    mock_concurrency,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_branch_orchestrator,
):
    """Overflow must reach ``_handle_warmup_failure`` despite gated intercept (R4)."""
    registry = MagicMock()
    registry.has_tree.return_value = True
    mock_branch_orchestrator.intercept = AsyncMock(return_value=True)
    mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
    strategy = MagicMock()
    strategy.handle_credit_return = AsyncMock()
    strategy.record_warmup_failure = MagicMock()
    handler = CreditCallbackHandler(
        mock_concurrency,
        branch_orchestrator=mock_branch_orchestrator,
        session_tree_registry=registry,
    )
    handler.register_phase(
        phase=CreditPhase.WARMUP,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=strategy,
    )
    credit = make_credit(turn_index=0, num_turns=3, phase=CreditPhase.WARMUP)
    assert not credit.is_final_turn

    await handler.on_credit_return(
        "worker-1",
        make_credit_return(credit, error=_OVERFLOW_ERROR),
    )

    mock_branch_orchestrator.intercept.assert_not_awaited()
    strategy.record_warmup_failure.assert_called_once_with(credit.conversation_id)
    strategy.handle_credit_return.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_overflow_warmup_gated_intercept_still_records_warmup_failure(
    mock_concurrency,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_branch_orchestrator,
):
    """Non-overflow WARMUP + gated intercept must still record warmup failure.

    Accelerated cache-pressure warmup enables DAG intercept during WARMUP.
    A plain HTTP error on a gated root must not early-return past
    ``record_warmup_failure`` / live abort (overflow already skips intercept).
    """
    registry = MagicMock()
    registry.has_tree.return_value = True
    mock_branch_orchestrator.intercept = AsyncMock(return_value=True)
    mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
    strategy = MagicMock()
    strategy.handle_credit_return = AsyncMock()
    strategy.record_warmup_failure = MagicMock()
    strategy.wants_returns_after_sending_complete = False
    handler = CreditCallbackHandler(
        mock_concurrency,
        branch_orchestrator=mock_branch_orchestrator,
        session_tree_registry=registry,
    )
    handler.register_phase(
        phase=CreditPhase.WARMUP,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=strategy,
    )
    credit = make_credit(turn_index=0, num_turns=5, phase=CreditPhase.WARMUP)
    assert not credit.is_final_turn

    await handler.on_credit_return(
        "worker-1",
        make_credit_return(credit, error="Internal server error: pool exhausted"),
    )

    mock_branch_orchestrator.intercept.assert_awaited_once_with(credit)
    strategy.record_warmup_failure.assert_called_once_with(credit.conversation_id)
    strategy.handle_credit_return.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_overflow_gated_suspend_still_early_returns(
    mock_concurrency,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
    mock_branch_orchestrator,
):
    """Non-overflow gated suspend must still early-return past strategy dispatch."""
    registry = MagicMock()
    registry.has_tree.return_value = True
    mock_branch_orchestrator.intercept = AsyncMock(return_value=True)
    mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
    handler = CreditCallbackHandler(
        mock_concurrency,
        branch_orchestrator=mock_branch_orchestrator,
        session_tree_registry=registry,
    )
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    credit = make_credit(turn_index=1, num_turns=5)

    await handler.on_credit_return(
        "worker-1",
        make_credit_return(credit, error="Internal server error: pool exhausted"),
    )

    mock_branch_orchestrator.intercept.assert_awaited_once_with(credit)
    registry.on_root_terminal.assert_not_called()
    mock_strategy.handle_credit_return.assert_not_awaited()


@pytest.mark.asyncio
async def test_cache_warmup_handoff_allows_paused_dag_work(
    callback_handler,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
    mock_branch_orchestrator,
):
    """A drained accelerated-warmup phase completes on in_flight==0 even with
    the orchestrator still holding paused (handoff) branch work."""
    mock_progress.increment_returned = MagicMock(return_value=True)
    mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=True)
    mock_progress.in_flight = 0
    mock_lifecycle.is_sending_complete = True
    mock_strategy.allows_pending_branch_handoff_after_sending_complete = True
    mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=True)
    mock_branch_orchestrator.intercept = AsyncMock(return_value=True)
    callback_handler.set_branch_orchestrator(mock_branch_orchestrator)
    callback_handler.register_phase(
        phase=CreditPhase.WARMUP,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )

    credit = make_credit(
        phase=CreditPhase.WARMUP,
        turn_index=0,
        num_turns=2,
    )
    await callback_handler.on_credit_return("worker-1", make_credit_return(credit))

    assert mock_progress.all_credits_returned_event.is_set()
    mock_branch_orchestrator.has_pending_branch_work.assert_called_once_with()


# =============================================================================
# Test: Credit Return - Unregistered/Complete Phase
# =============================================================================


class TestUnregisteredAndCompletePhaseHandling:
    """Tests for handling credits from unregistered or complete phases."""

    async def test_ignores_unregistered_phase(self, callback_handler):
        """Silently ignores returns for unregistered phases."""
        credit = make_credit(phase=CreditPhase.WARMUP)
        credit_return = make_credit_return(credit)
        # Should not raise
        await callback_handler.on_credit_return("worker-1", credit_return)

    async def test_ignores_complete_phase(
        self, registered_handler, mock_lifecycle, mock_progress
    ):
        """Ignores late returns after phase is complete."""
        mock_lifecycle.is_complete = True
        credit = make_credit()
        credit_return = make_credit_return(credit)
        await registered_handler.on_credit_return("worker-1", credit_return)
        mock_progress.increment_returned.assert_not_called()


# =============================================================================
# Test: First Token (TTFT) Handling
# =============================================================================


class TestFirstTokenHandling:
    """Tests for TTFT event handling."""

    async def test_first_token_tracks_and_releases_prefill(
        self, registered_handler, mock_progress, mock_concurrency
    ):
        """TTFT tracks prefill release and releases slot."""
        first_token = FirstToken(
            credit_id=1,
            phase=CreditPhase.PROFILING,
            ttft_ns=1000000,
        )

        await registered_handler.on_first_token(first_token)

        mock_progress.increment_prefill_released.assert_called_once()
        mock_concurrency.release_prefill_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )

    async def test_first_token_notifies_strategy_hook(
        self,
        callback_handler,
        mock_progress,
        mock_lifecycle,
        mock_stop_checker,
        mock_strategy,
    ):
        """Strategies with a first-token hook should receive TTFT observations."""
        mock_strategy.handle_first_token = AsyncMock()
        callback_handler.register_phase(
            phase=CreditPhase.PROFILING,
            progress=mock_progress,
            lifecycle=mock_lifecycle,
            stop_checker=mock_stop_checker,
            strategy=mock_strategy,
        )
        first_token = FirstToken(
            credit_id=1,
            phase=CreditPhase.PROFILING,
            ttft_ns=1000000,
        )

        await callback_handler.on_first_token(first_token)

        mock_strategy.handle_first_token.assert_awaited_once_with(first_token)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.parametrize(
        "cancelled,first_token_sent",
        [(False, True), (True, False)],  # Sample: normal and cancelled-before-ttft
    )  # fmt: skip
    async def test_return_state_combinations(
        self,
        registered_handler,
        mock_progress,
        mock_concurrency,
        cancelled: bool,
        first_token_sent: bool,
    ):
        """Handles combinations of cancelled/first_token_sent correctly."""
        credit = make_credit()
        credit_return = make_credit_return(
            credit, cancelled=cancelled, first_token_sent=first_token_sent
        )

        await registered_handler.on_credit_return("worker-1", credit_return)

        mock_progress.increment_returned.assert_called_once_with(
            credit.is_final_turn,
            cancelled,
            session_ended=True,
            session_cancelled=cancelled,
            errored=False,
            is_child=False,
            no_request=False,
        )
        if not first_token_sent:
            mock_concurrency.release_prefill_slot.assert_called_once()
        else:
            mock_concurrency.release_prefill_slot.assert_not_called()


class TestDagWorkPending:
    """Pin the contract on ``_dag_work_pending``.

    ``intercept`` runs at every ``agent_depth``, so the branch-id lookup
    must run at every depth too — restricting it to ``agent_depth == 0``
    let nested grandchildren be truncated when the final outstanding
    credit at signal time happened to be a child whose own intercept was
    about to spawn more work.
    """

    def test_returns_true_when_pending_work_in_flight(
        self, callback_handler, mock_branch_orchestrator
    ):
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=True)
        callback_handler.set_branch_orchestrator(mock_branch_orchestrator)

        assert callback_handler._dag_work_pending(make_credit())

    def test_returns_true_for_root_credit_with_branch_ids(
        self, callback_handler, mock_branch_orchestrator
    ):
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
        mock_branch_orchestrator.get_branch_ids = MagicMock(return_value=["b0"])
        callback_handler.set_branch_orchestrator(mock_branch_orchestrator)

        assert callback_handler._dag_work_pending(make_credit(agent_depth=0))

    def test_returns_true_for_child_credit_with_branch_ids(
        self, callback_handler, mock_branch_orchestrator
    ):
        """Regression for the nested-DAG race: a child credit (agent_depth>0)
        whose own turn declares branches must defer the all-credits-returned
        event so ``intercept`` can spawn the grandchildren first.
        """
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
        mock_branch_orchestrator.get_branch_ids = MagicMock(return_value=["b1"])
        callback_handler.set_branch_orchestrator(mock_branch_orchestrator)

        assert callback_handler._dag_work_pending(make_credit(agent_depth=2))

    def test_returns_false_when_no_branch_ids_and_no_pending_work(
        self, callback_handler, mock_branch_orchestrator
    ):
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
        mock_branch_orchestrator.get_branch_ids = MagicMock(return_value=[])
        callback_handler.set_branch_orchestrator(mock_branch_orchestrator)

        assert not callback_handler._dag_work_pending(make_credit(agent_depth=1))


class TestDagWorkPendingAdversarial:
    """Hostile-input cases for ``_dag_work_pending``.

    ``_count_and_release`` reaches this helper inside the no-await counter
    section, so any exception or wrong answer here either deadlocks the
    phase (false-positive defer that never resolves) or truncates DAG
    work (false-negative signal that lets teardown win the race).
    """

    def test_returns_false_when_no_orchestrator_registered(self, callback_handler):
        """Plain non-DAG runs never attach an orchestrator. The predictor
        must short-circuit to False rather than dereferencing None — a
        crash here would propagate through ``_count_and_release`` and
        abort the credit-return callback for every credit."""
        assert callback_handler._branch_orchestrator is None
        assert not callback_handler._dag_work_pending(make_credit())

    def test_pending_work_dominates_empty_branch_ids_at_any_depth(
        self, callback_handler, mock_branch_orchestrator
    ):
        """``has_pending_branch_work=True`` is the in-flight signal. Even
        if the current credit's own turn declares no branches, other
        children are still draining — the event must defer."""
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=True)
        mock_branch_orchestrator.get_branch_ids = MagicMock(return_value=[])
        callback_handler.set_branch_orchestrator(mock_branch_orchestrator)

        assert callback_handler._dag_work_pending(make_credit(agent_depth=0))
        assert callback_handler._dag_work_pending(make_credit(agent_depth=4))

    def test_returns_false_when_get_branch_ids_raises(
        self, callback_handler, mock_branch_orchestrator
    ):
        """``get_branch_ids`` walks orchestrator state that may be missing
        for a credit issued on a transient session (e.g. a child whose
        metadata was already cleaned up). A raise here MUST become a
        False return, not a propagated exception — the credit-return
        callback must keep running for every credit."""
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
        mock_branch_orchestrator.get_branch_ids = MagicMock(
            side_effect=KeyError("missing conv")
        )
        callback_handler.set_branch_orchestrator(mock_branch_orchestrator)

        assert not callback_handler._dag_work_pending(make_credit(agent_depth=2))

    def test_returns_true_for_very_deep_credit_with_branch_ids(
        self, callback_handler, mock_branch_orchestrator
    ):
        """Depth has no semantic ceiling in the predictor — a credit at
        ``agent_depth=42`` whose own turn declares branches still defers
        signal. The old root-only guard would silently truncate this."""
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
        mock_branch_orchestrator.get_branch_ids = MagicMock(return_value=["deep"])
        callback_handler.set_branch_orchestrator(mock_branch_orchestrator)

        assert callback_handler._dag_work_pending(make_credit(agent_depth=42))

    def test_pending_work_short_circuits_before_get_branch_ids(
        self, callback_handler, mock_branch_orchestrator
    ):
        """When the orchestrator already has work in flight, the
        predictor must not bother walking ``get_branch_ids`` — that lookup
        can be expensive on hot paths. Wired by short-circuit ordering."""
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=True)
        mock_branch_orchestrator.get_branch_ids = MagicMock(
            side_effect=AssertionError("must not be called")
        )
        callback_handler.set_branch_orchestrator(mock_branch_orchestrator)

        assert callback_handler._dag_work_pending(make_credit(agent_depth=1))
        mock_branch_orchestrator.get_branch_ids.assert_not_called()


class TestDrainObserverWiring:
    """Regression for the concurrency>=2 race fixed in commit 7cd4180b7.

    The orchestrator's last drain step (``_handle_child_done`` decrement,
    ``dispatch_join_turn`` returning False under cap, all-children-rolled-
    back path) can land BETWEEN concurrent ``on_credit_return`` callbacks.
    Without the drain-observer hook, ``all_credits_returned_event`` is
    never set from the callback path and the phase runner blocks forever
    (or, post-`f6fb1ae29`, takes the slow drain-timeout path).

    These tests pin the wiring contract on
    ``CreditCallbackHandler.set_branch_orchestrator`` and the closure
    registered via ``BranchOrchestrator.set_drain_observer``.
    """

    def test_set_branch_orchestrator_registers_drain_observer(self, callback_handler):
        """Attaching an orchestrator must register a drain callback;
        detaching (set None) must clear it."""
        orchestrator = MagicMock()
        orchestrator.set_drain_observer = MagicMock()

        callback_handler.set_branch_orchestrator(orchestrator)
        orchestrator.set_drain_observer.assert_called_once()
        assert callable(orchestrator.set_drain_observer.call_args.args[0])

        callback_handler.set_branch_orchestrator(None)
        orchestrator.set_drain_observer.assert_called_with(None)

    def test_drain_observer_sets_event_when_predicate_satisfied(
        self, registered_handler, mock_progress, mock_branch_orchestrator
    ):
        """Race-closing path: callback fires AND counters say all returned
        AND orchestrator predicate clean -> event MUST set."""
        mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=True)
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
        assert not mock_progress.all_credits_returned_event.is_set()

        registered_handler.set_branch_orchestrator(mock_branch_orchestrator)
        callback = mock_branch_orchestrator.set_drain_observer.call_args.args[0]
        callback()

        assert mock_progress.all_credits_returned_event.is_set()

    def test_drain_observer_no_op_when_pending_work_remains(
        self, registered_handler, mock_progress, mock_branch_orchestrator
    ):
        """has_pending_branch_work=True must keep the event deferred —
        firing now would declare phase complete with children in flight."""
        mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=True)
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=True)

        registered_handler.set_branch_orchestrator(mock_branch_orchestrator)
        callback = mock_branch_orchestrator.set_drain_observer.call_args.args[0]
        callback()

        assert not mock_progress.all_credits_returned_event.is_set()

    def test_drain_observer_no_op_when_counters_disagree(
        self, registered_handler, mock_progress, mock_branch_orchestrator
    ):
        """check_all_returned_or_cancelled=False must keep the event
        deferred — sending isn't actually complete yet."""
        mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=False)
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)

        registered_handler.set_branch_orchestrator(mock_branch_orchestrator)
        callback = mock_branch_orchestrator.set_drain_observer.call_args.args[0]
        callback()

        assert not mock_progress.all_credits_returned_event.is_set()

    def test_drain_observer_skips_completed_phase_handlers(
        self,
        registered_handler,
        mock_progress,
        mock_lifecycle,
        mock_branch_orchestrator,
    ):
        """A phase whose lifecycle is already complete must be skipped —
        its event was already finalized by the normal end-of-phase path
        and re-setting from here would be racy."""
        mock_lifecycle.is_complete = True
        mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=True)
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)

        registered_handler.set_branch_orchestrator(mock_branch_orchestrator)
        callback = mock_branch_orchestrator.set_drain_observer.call_args.args[0]
        callback()

        assert not mock_progress.all_credits_returned_event.is_set()

    def test_drain_observer_idempotent_on_already_set_event(
        self, registered_handler, mock_progress, mock_branch_orchestrator
    ):
        """Multiple callback invocations after the event is already set
        must remain a no-op. The observer can fire several times in rapid
        succession (``_handle_child_done`` + ``_handle_child_errored_fail_fast``
        + ``_drain_vestigial_gates`` all call ``_notify_drain``)."""
        mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=True)
        mock_branch_orchestrator.has_pending_branch_work = MagicMock(return_value=False)
        mock_progress.all_credits_returned_event.set()

        registered_handler.set_branch_orchestrator(mock_branch_orchestrator)
        callback = mock_branch_orchestrator.set_drain_observer.call_args.args[0]
        callback()
        callback()
        callback()

        assert mock_progress.all_credits_returned_event.is_set()


# Test: WARMUP Terminal-Failure Accumulation + Live Early-Abort


class TestWarmupFailureRecording:
    """A terminal WARMUP root failure must be recorded via the strategy hook.

    A WARMUP credit primes turn k_i (the last request before t*); PROFILING
    resumes the same trajectory at k_i+1, so a warmed turn for a session active
    at t* is NEVER the trajectory's final turn (is_final_turn is False). The
    gate must therefore fire on a NON-final WARMUP root credit that returns with
    a terminal error/cancellation; gating it on is_final_turn made the whole
    safety mechanism dead.
    """

    @pytest.fixture
    def warmup_strategy(self):
        """Mock strategy exposing the record_warmup_failure hook."""
        mock = MagicMock()
        mock.handle_credit_return = AsyncMock()
        mock.record_warmup_failure = MagicMock()
        return mock

    @pytest.fixture
    def warmup_handler(
        self,
        callback_handler,
        mock_progress,
        mock_lifecycle,
        mock_stop_checker,
        warmup_strategy,
    ):
        callback_handler.register_phase(
            phase=CreditPhase.WARMUP,
            progress=mock_progress,
            lifecycle=mock_lifecycle,
            stop_checker=mock_stop_checker,
            strategy=warmup_strategy,
        )
        return callback_handler

    async def test_non_final_warmup_credit_error_records_failure(
        self, warmup_handler, warmup_strategy
    ):
        """A NON-final WARMUP root credit returning with an error MUST record a
        warmup failure (the gate must not require is_final_turn)."""
        credit = make_credit(turn_index=0, num_turns=3, phase=CreditPhase.WARMUP)
        assert not credit.is_final_turn  # the case the old gate silently dropped
        credit_return = CreditReturn(
            credit=credit, cancelled=False, first_token_sent=False, error="server 500"
        )

        await warmup_handler.on_credit_return("worker-1", credit_return)

        warmup_strategy.record_warmup_failure.assert_called_once_with(
            credit.conversation_id
        )

    async def test_non_final_warmup_credit_cancelled_records_failure(
        self, warmup_handler, warmup_strategy
    ):
        """Cancellation (not just error) on a non-final WARMUP credit also counts."""
        credit = make_credit(turn_index=1, num_turns=4, phase=CreditPhase.WARMUP)
        credit_return = make_credit_return(
            credit, cancelled=True, first_token_sent=False
        )

        await warmup_handler.on_credit_return("worker-1", credit_return)

        warmup_strategy.record_warmup_failure.assert_called_once_with(
            credit.conversation_id
        )

    async def test_successful_warmup_credit_does_not_record_failure(
        self, warmup_handler, warmup_strategy
    ):
        """A clean WARMUP return (no error, not cancelled) records nothing."""
        credit = make_credit(turn_index=0, num_turns=3, phase=CreditPhase.WARMUP)
        credit_return = make_credit_return(credit)

        await warmup_handler.on_credit_return("worker-1", credit_return)

        warmup_strategy.record_warmup_failure.assert_not_called()

    async def test_warmup_child_failure_does_not_record(
        self, warmup_handler, warmup_strategy
    ):
        """The gate is root-only (agent_depth == 0): a failed WARMUP child does
        not count toward trajectory warmup failure."""
        credit = make_credit(
            turn_index=0, num_turns=2, agent_depth=1, phase=CreditPhase.WARMUP
        )
        credit_return = CreditReturn(
            credit=credit, cancelled=False, first_token_sent=False, error="server 500"
        )

        await warmup_handler.on_credit_return("worker-1", credit_return)

        warmup_strategy.record_warmup_failure.assert_not_called()


class TestWarmupEarlyAbort:
    """Live early-abort: the first terminal WARMUP failure fires on_warmup_abort.

    A single terminal warmup failure means PROFILING must not start, so the
    handler broadcasts ProfileCancelCommand (via the injected callback) on the
    FIRST failure rather than waiting for the full warmup drain + teardown
    ``report_warmup_failures`` raise. The callback fires at most once per run.
    """

    @pytest.fixture
    def abort_cb(self):
        return AsyncMock()

    @pytest.fixture
    def warmup_strategy(self):
        mock = MagicMock()
        mock.handle_credit_return = AsyncMock()
        mock.record_warmup_failure = MagicMock()
        return mock

    @pytest.fixture
    def early_abort_handler(
        self,
        mock_concurrency,
        mock_progress,
        mock_lifecycle,
        mock_stop_checker,
        warmup_strategy,
        abort_cb,
    ):
        handler = CreditCallbackHandler(mock_concurrency, on_warmup_abort=abort_cb)
        handler.register_phase(
            phase=CreditPhase.WARMUP,
            progress=mock_progress,
            lifecycle=mock_lifecycle,
            stop_checker=mock_stop_checker,
            strategy=warmup_strategy,
        )
        return handler

    async def test_first_warmup_failure_fires_abort_once(
        self, early_abort_handler, abort_cb
    ):
        """First terminal warmup failure both records and fires the abort once."""
        credit = make_credit(turn_index=0, num_turns=3, phase=CreditPhase.WARMUP)
        credit_return = CreditReturn(
            credit=credit, cancelled=False, first_token_sent=False, error="server 500"
        )

        await early_abort_handler.on_credit_return("worker-1", credit_return)

        abort_cb.assert_awaited_once()

    async def test_subsequent_warmup_failures_do_not_refire_abort(
        self, early_abort_handler, abort_cb, warmup_strategy
    ):
        """Only the first failure fires the abort; later failures still record."""
        for idx in range(3):
            credit = make_credit(
                credit_id=idx,
                conversation_id=f"conv{idx}",
                turn_index=0,
                num_turns=3,
                phase=CreditPhase.WARMUP,
            )
            credit_return = CreditReturn(
                credit=credit,
                cancelled=False,
                first_token_sent=False,
                error="server 500",
            )
            await early_abort_handler.on_credit_return("worker-1", credit_return)

        abort_cb.assert_awaited_once()
        assert warmup_strategy.record_warmup_failure.call_count == 3

    async def test_successful_warmup_return_does_not_fire_abort(
        self, early_abort_handler, abort_cb
    ):
        """A clean warmup return never fires the abort."""
        credit = make_credit(turn_index=0, num_turns=3, phase=CreditPhase.WARMUP)
        credit_return = make_credit_return(credit)

        await early_abort_handler.on_credit_return("worker-1", credit_return)

        abort_cb.assert_not_awaited()

    async def test_publish_failure_resets_trigger_flag(
        self, mock_concurrency, mock_progress, mock_lifecycle, mock_stop_checker
    ):
        """If the abort broadcast raises, the flag resets so a later return retries
        and the teardown backstop can still fire."""
        failing_cb = AsyncMock(side_effect=RuntimeError("bus down"))
        strategy = MagicMock()
        strategy.handle_credit_return = AsyncMock()
        strategy.record_warmup_failure = MagicMock()
        handler = CreditCallbackHandler(mock_concurrency, on_warmup_abort=failing_cb)
        handler.register_phase(
            phase=CreditPhase.WARMUP,
            progress=mock_progress,
            lifecycle=mock_lifecycle,
            stop_checker=mock_stop_checker,
            strategy=strategy,
        )
        credit = make_credit(turn_index=0, num_turns=3, phase=CreditPhase.WARMUP)
        credit_return = CreditReturn(
            credit=credit, cancelled=False, first_token_sent=False, error="500"
        )

        await handler.on_credit_return("worker-1", credit_return)

        failing_cb.assert_awaited_once()
        strategy.record_warmup_failure.assert_called_once()
        assert handler._warmup_abort_triggered is False

    def test_on_warmup_abort_property(self, mock_concurrency, abort_cb):
        """The public property exposes the wired callback (None when unwired)."""
        assert CreditCallbackHandler(mock_concurrency).on_warmup_abort is None
        assert (
            CreditCallbackHandler(
                mock_concurrency, on_warmup_abort=abort_cb
            ).on_warmup_abort
            is abort_cb
        )

    async def test_unwired_handler_records_but_does_not_abort(
        self,
        warmup_strategy,
        mock_concurrency,
        mock_progress,
        mock_lifecycle,
        mock_stop_checker,
    ):
        """With no on_warmup_abort wired, the failure still records (the teardown
        backstop remains the only abort path)."""
        handler = CreditCallbackHandler(mock_concurrency)
        handler.register_phase(
            phase=CreditPhase.WARMUP,
            progress=mock_progress,
            lifecycle=mock_lifecycle,
            stop_checker=mock_stop_checker,
            strategy=warmup_strategy,
        )
        credit = make_credit(turn_index=0, num_turns=3, phase=CreditPhase.WARMUP)
        credit_return = CreditReturn(
            credit=credit, cancelled=False, first_token_sent=False, error="500"
        )

        await handler.on_credit_return("worker-1", credit_return)

        warmup_strategy.record_warmup_failure.assert_called_once()
        assert handler._warmup_abort_triggered is False


async def test_no_request_return_skips_prefill_counter(
    registered_handler, mock_progress
):
    """A no_request virtual credit must NOT bump the prefill-released counter
    (increment_sent skips its billable counter too), else the derived
    in_flight_prefills = sent - released underflows by one per coordinator turn.
    The concurrency slot is still released via the return path."""
    credit = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="c",
        x_correlation_id="corr-c",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        no_request=True,
    )
    credit_return = make_credit_return(credit, first_token_sent=False)
    await registered_handler.on_credit_return("worker-1", credit_return)
    mock_progress.increment_prefill_released.assert_not_called()
