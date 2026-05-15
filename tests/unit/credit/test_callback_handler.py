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
    agent_depth: int = 0,
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
        agent_depth=agent_depth,
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
        mock_strategy.handle_credit_return.assert_called_once_with(credit)

        # Stop condition reached
        mock_strategy.reset_mock()
        mock_stop_checker.can_send_any_turn.return_value = False
        credit2 = make_credit(credit_id=2, turn_index=0, num_turns=3)
        credit_return2 = make_credit_return(credit2)
        await registered_handler.on_credit_return("worker-1", credit_return2)
        mock_strategy.handle_credit_return.assert_not_called()


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
            credit.is_final_turn, cancelled, errored=False
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


class TestAbortObserverWiring:
    """``AIPERF_DAG_FAIL_FAST=true`` fires an abort observer from the
    orchestrator's fail-fast handler; the callback handler must cancel
    every active phase lifecycle so the strategy loop stops issuing new
    wire credits. Without this, only the parent of the errored child was
    aborted while unrelated roots kept firing — the budget ran out as
    if FAIL_FAST were disabled.
    """

    def test_set_branch_orchestrator_registers_abort_observer(self, callback_handler):
        """Attaching an orchestrator must register an abort callback;
        detaching (set None) must clear it."""
        orchestrator = MagicMock()
        orchestrator.set_drain_observer = MagicMock()
        orchestrator.set_abort_observer = MagicMock()

        callback_handler.set_branch_orchestrator(orchestrator)
        orchestrator.set_abort_observer.assert_called_once()
        assert callable(orchestrator.set_abort_observer.call_args.args[0])

        callback_handler.set_branch_orchestrator(None)
        orchestrator.set_abort_observer.assert_called_with(None)

    def test_abort_observer_cancels_lifecycle_and_signals_return_event(
        self,
        registered_handler,
        mock_progress,
        mock_lifecycle,
        mock_branch_orchestrator,
    ):
        """Fail-fast fires the abort observer; the callback handler must
        cancel the active phase's lifecycle (so LifecycleStopCondition
        gates further issuance) and set ``all_credits_returned_event`` so
        the phase runner unblocks rather than waiting for credits that
        will never be issued.
        """
        mock_lifecycle.is_complete = False
        mock_lifecycle.cancel = MagicMock()

        registered_handler.set_branch_orchestrator(mock_branch_orchestrator)
        callback = mock_branch_orchestrator.set_abort_observer.call_args.args[0]
        callback()

        mock_lifecycle.cancel.assert_called_once_with()
        assert mock_progress.all_credits_returned_event.is_set()

    def test_abort_observer_skips_completed_phase_handlers(
        self,
        registered_handler,
        mock_progress,
        mock_lifecycle,
        mock_branch_orchestrator,
    ):
        """A phase whose lifecycle is already complete must be skipped —
        re-cancelling it would be wrong (the phase has already finalized).
        """
        mock_lifecycle.is_complete = True
        mock_lifecycle.cancel = MagicMock()

        registered_handler.set_branch_orchestrator(mock_branch_orchestrator)
        callback = mock_branch_orchestrator.set_abort_observer.call_args.args[0]
        callback()

        mock_lifecycle.cancel.assert_not_called()
