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
) -> Credit:
    """Create a Credit for testing."""
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
    )


def make_credit_return(
    credit: Credit,
    cancelled: bool = False,
    first_token_sent: bool = True,
) -> CreditReturn:
    """Create a CreditReturn for testing."""
    return CreditReturn(
        credit=credit,
        cancelled=cancelled,
        first_token_sent=first_token_sent,
    )


# =============================================================================
# Test: Phase Registration
# =============================================================================


class TestPhaseRegistration:
    """Tests for phase registration and unregistration."""

    def test_register_and_unregister_phase(self, callback_handler):
        """Register and unregister phase correctly updates handlers."""
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

        callback_handler.unregister_phase(CreditPhase.PROFILING)
        assert CreditPhase.PROFILING not in callback_handler._phase_handlers


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
            errored=False,
            is_child=False,
        )

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
            errored=False,
            is_child=False,
        )

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
# Test: WARMUP terminal-failure accumulation
# =============================================================================


class TestWarmupFailureAccumulation:
    """Regression tests for the WARMUP terminal-failure gate in on_credit_return.

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
        credit = make_dag_credit(
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
            credit.is_final_turn, cancelled, errored=False, is_child=False
        )
        if not first_token_sent:
            mock_concurrency.release_prefill_slot.assert_called_once()
        else:
            mock_concurrency.release_prefill_slot.assert_not_called()


# =============================================================================
# Test: DAG (sub-agent) guards
# =============================================================================


def make_dag_credit(
    credit_id: int = 1,
    conversation_id: str = "conv-child",
    turn_index: int = 0,
    num_turns: int = 1,
    agent_depth: int = 1,
    parent_correlation_id: str = "parent-corr",
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    """Credit variant carrying DAG child fields."""
    return Credit(
        id=credit_id,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=f"child-corr-{credit_id}",
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
        agent_depth=agent_depth,
        parent_correlation_id=parent_correlation_id,
    )


@pytest.fixture
def mock_orchestrator():
    """Mock BranchOrchestrator with async hooks."""
    mock = MagicMock()
    mock.intercept = AsyncMock(return_value=False)
    mock.on_child_leaf_reached = AsyncMock()
    mock.on_child_errored = AsyncMock()
    mock.has_pending_branch_work = MagicMock(return_value=False)
    return mock


@pytest.fixture
def dag_handler(mock_concurrency, mock_orchestrator):
    """CreditCallbackHandler with a BranchOrchestrator wired in."""
    return CreditCallbackHandler(
        mock_concurrency, branch_orchestrator=mock_orchestrator
    )


@pytest.fixture
def registered_dag_handler(
    dag_handler,
    mock_progress,
    mock_lifecycle,
    mock_stop_checker,
    mock_strategy,
):
    dag_handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=mock_progress,
        lifecycle=mock_lifecycle,
        stop_checker=mock_stop_checker,
        strategy=mock_strategy,
    )
    return dag_handler


class TestDagCallbackGuards:
    """DAG-specific branches in ``on_credit_return``:

    1. ``release_session_slot`` must skip when ``agent_depth > 0`` —
       children inherit the root's slot and must not release a slot
       they never acquired.
    2. Strategy dispatch must still fire for children even when
       ``can_send_any_turn`` is False — phase-level stop conditions
       drive root sampling, not DAG continuation.
    3. ``all_credits_returned_event`` must defer when the orchestrator
       has pending branch work or the just-returned credit will spawn
       more children.
    4. Child final-turn returns must notify the orchestrator (leaf vs
       errored) so join counters decrement.
    """

    async def test_child_final_turn_does_not_release_session_slot(
        self, registered_dag_handler, mock_concurrency
    ):
        """agent_depth > 0 + is_final_turn → MUST NOT release_session_slot."""
        credit = make_dag_credit(turn_index=0, num_turns=1, agent_depth=1)
        credit_return = make_credit_return(credit)

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        mock_concurrency.release_session_slot.assert_not_called()

    async def test_root_final_turn_still_releases_session_slot(
        self, registered_dag_handler, mock_concurrency
    ):
        """Regression guard: the DAG guard must not leak into the root path."""
        credit = make_credit(turn_index=0, num_turns=1)  # agent_depth == 0
        credit_return = make_credit_return(credit)

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        mock_concurrency.release_session_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )

    async def test_child_dispatch_bypasses_can_send_any_turn_guard(
        self, registered_dag_handler, mock_stop_checker, mock_strategy
    ):
        """Children must continue even after phase sampling is complete."""
        mock_stop_checker.can_send_any_turn = MagicMock(return_value=False)
        credit = make_dag_credit(turn_index=0, num_turns=2, agent_depth=1)
        credit_return = make_credit_return(credit)

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        mock_strategy.handle_credit_return.assert_called_once_with(credit, error=None)

    async def test_root_dispatch_still_gated_by_can_send_any_turn(
        self, registered_dag_handler, mock_stop_checker, mock_strategy
    ):
        """Regression guard: root strategy dispatch stays gated."""
        mock_stop_checker.can_send_any_turn = MagicMock(return_value=False)
        credit = make_credit(turn_index=0, num_turns=2)  # agent_depth == 0
        credit_return = make_credit_return(credit)

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        mock_strategy.handle_credit_return.assert_not_called()

    async def test_all_credits_returned_deferred_when_orchestrator_has_pending_work(
        self, registered_dag_handler, mock_progress, mock_orchestrator
    ):
        """When the orchestrator has pending branch work at final return,
        all_credits_returned_event must NOT fire immediately."""
        mock_progress.increment_returned = MagicMock(return_value=True)
        mock_orchestrator.has_pending_branch_work = MagicMock(return_value=True)
        mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=True)

        credit = make_credit(turn_index=0, num_turns=1)
        credit_return = make_credit_return(credit)

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        # Event must stay unset — DAG is still draining.
        assert not mock_progress.all_credits_returned_event.is_set()

    async def test_all_credits_returned_fires_after_dag_drains(
        self, registered_dag_handler, mock_progress, mock_orchestrator
    ):
        """After intercept, if orchestrator reports no more pending work and
        progress confirms all returned, the event fires via the post-intercept
        re-check."""
        mock_progress.increment_returned = MagicMock(return_value=True)
        # First check: pending (defer). Second check (post-intercept): drained.
        mock_orchestrator.has_pending_branch_work = MagicMock(side_effect=[True, False])
        mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=True)

        credit = make_credit(turn_index=0, num_turns=1)
        credit_return = make_credit_return(credit)

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        assert mock_progress.all_credits_returned_event.is_set()

    async def test_cache_warmup_handoff_allows_paused_dag_work(
        self,
        dag_handler,
        mock_progress,
        mock_lifecycle,
        mock_stop_checker,
        mock_strategy,
        mock_orchestrator,
    ):
        mock_progress.increment_returned = MagicMock(return_value=True)
        mock_progress.check_all_returned_or_cancelled = MagicMock(return_value=True)
        mock_progress.in_flight = 0
        mock_lifecycle.is_sending_complete = True
        mock_strategy.allows_pending_branch_handoff_after_sending_complete = True
        mock_orchestrator.has_pending_branch_work = MagicMock(return_value=True)
        mock_orchestrator.intercept = AsyncMock(return_value=True)
        dag_handler.register_phase(
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
        await dag_handler.on_credit_return("worker-1", make_credit_return(credit))

        assert mock_progress.all_credits_returned_event.is_set()
        mock_orchestrator.has_pending_branch_work.assert_called_once_with()

    async def test_child_leaf_reached_called_on_child_final_turn(
        self, registered_dag_handler, mock_orchestrator
    ):
        """Successful child final-turn return → on_child_leaf_reached hook."""
        credit = make_dag_credit(turn_index=0, num_turns=1, agent_depth=1)
        credit_return = make_credit_return(credit)

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        mock_orchestrator.on_child_leaf_reached.assert_awaited_once_with(
            credit.x_correlation_id
        )
        mock_orchestrator.on_child_errored.assert_not_awaited()

    async def test_child_errored_called_when_credit_return_has_error(
        self, registered_dag_handler, mock_orchestrator
    ):
        """Errored child final turn → on_child_errored hook."""
        credit = make_dag_credit(turn_index=0, num_turns=1, agent_depth=1)
        credit_return = CreditReturn(
            credit=credit,
            cancelled=False,
            first_token_sent=False,
            error="server 500",
        )

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        mock_orchestrator.on_child_errored.assert_awaited_once_with(
            credit.x_correlation_id
        )
        mock_orchestrator.on_child_leaf_reached.assert_not_awaited()

    async def test_non_final_child_turn_does_not_fire_leaf_hook(
        self, registered_dag_handler, mock_orchestrator
    ):
        """Intermediate child turns shouldn't notify the orchestrator
        about leaf-reached — only the final turn does."""
        credit = make_dag_credit(turn_index=0, num_turns=3, agent_depth=1)
        credit_return = make_credit_return(credit)

        await registered_dag_handler.on_credit_return("worker-1", credit_return)

        mock_orchestrator.on_child_leaf_reached.assert_not_awaited()
        mock_orchestrator.on_child_errored.assert_not_awaited()
