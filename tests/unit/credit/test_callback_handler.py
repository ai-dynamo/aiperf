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
            credit.is_final_turn, cancelled
        )
        if not first_token_sent:
            mock_concurrency.release_prefill_slot.assert_called_once()
        else:
            mock_concurrency.release_prefill_slot.assert_not_called()
