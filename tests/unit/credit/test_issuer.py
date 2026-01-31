# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CreditIssuer.

Tests credit issuance with concurrency control and stop condition checking.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_stop_checker():
    """Mock stop condition checker that allows all by default."""
    mock = MagicMock()
    mock.can_send_any_turn = MagicMock(return_value=True)
    mock.can_start_new_session = MagicMock(return_value=True)
    return mock


@pytest.fixture
def mock_progress():
    """Mock progress tracker."""
    mock = MagicMock()
    mock.increment_sent = MagicMock(return_value=(1, False))  # (credit_index, is_final)
    mock.freeze_sent_counts = MagicMock()
    mock.all_credits_sent_event = asyncio.Event()
    return mock


@pytest.fixture
def mock_concurrency():
    """Mock concurrency manager."""
    mock = MagicMock()
    mock.acquire_session_slot = AsyncMock(return_value=True)
    mock.acquire_prefill_slot = AsyncMock(return_value=True)
    mock.release_session_slot = MagicMock()
    return mock


@pytest.fixture
def mock_router():
    """Mock credit router."""
    mock = MagicMock()
    mock.send_credit = AsyncMock()
    return mock


@pytest.fixture
def mock_cancellation():
    """Mock cancellation policy."""
    mock = MagicMock()
    mock.next_cancellation_delay_ns = MagicMock(return_value=None)
    return mock


@pytest.fixture
def mock_lifecycle():
    """Mock phase lifecycle."""
    mock = MagicMock()
    mock.time_left_in_seconds = MagicMock(return_value=None)
    mock.phase_start_ns = 0
    # CreditIssuer uses these to calculate issued_at_ns timestamps
    mock.started_at_ns = time.time_ns()
    mock.started_at_perf_ns = time.perf_counter_ns()
    return mock


@pytest.fixture
def credit_issuer(
    mock_stop_checker,
    mock_progress,
    mock_concurrency,
    mock_router,
    mock_cancellation,
    mock_lifecycle,
):
    """Create CreditIssuer with all mocked dependencies."""
    return CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=mock_stop_checker,
        progress=mock_progress,
        concurrency_manager=mock_concurrency,
        credit_router=mock_router,
        cancellation_policy=mock_cancellation,
        lifecycle=mock_lifecycle,
    )


def make_turn(
    conversation_id: str = "conv1",
    turn_index: int = 0,
    num_turns: int = 1,
) -> TurnToSend:
    """Create a TurnToSend for testing."""
    return TurnToSend(
        conversation_id=conversation_id,
        x_correlation_id=f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=num_turns,
    )


# =============================================================================
# Test: Basic Credit Issuance
# =============================================================================


class TestBasicCreditIssuance:
    """Tests for basic credit issuance flow."""

    async def test_issue_credit_first_turn_acquires_both_slots(
        self, credit_issuer, mock_concurrency, mock_router
    ):
        """First turn should acquire session slot AND prefill slot."""
        turn = make_turn(turn_index=0, num_turns=3)

        result = await credit_issuer.issue_credit(turn)

        assert result is True
        mock_concurrency.acquire_session_slot.assert_called_once()
        mock_concurrency.acquire_prefill_slot.assert_called_once()
        mock_router.send_credit.assert_called_once()

    async def test_issue_credit_subsequent_turn_acquires_only_prefill(
        self, credit_issuer, mock_concurrency, mock_router
    ):
        """Subsequent turns should only acquire prefill slot, not session slot."""
        turn = make_turn(turn_index=1, num_turns=3)  # Not first turn

        result = await credit_issuer.issue_credit(turn)

        assert result is True
        mock_concurrency.acquire_session_slot.assert_not_called()
        mock_concurrency.acquire_prefill_slot.assert_called_once()
        mock_router.send_credit.assert_called_once()

    async def test_issue_credit_creates_correct_credit_struct(
        self, credit_issuer, mock_router, mock_progress
    ):
        """Credit struct should have correct fields from turn."""
        mock_progress.increment_sent.return_value = (42, False)  # credit_index=42
        turn = make_turn(conversation_id="test-conv", turn_index=1, num_turns=5)

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.id == 42
        assert sent_credit.phase == CreditPhase.PROFILING
        assert sent_credit.conversation_id == "test-conv"
        assert sent_credit.x_correlation_id == "corr-test-conv"
        assert sent_credit.turn_index == 1
        assert sent_credit.num_turns == 5
        assert sent_credit.issued_at_ns > 0

    async def test_issue_credit_returns_true_when_more_credits_can_be_sent(
        self, credit_issuer, mock_progress
    ):
        """Should return True when not the final credit."""
        mock_progress.increment_sent.return_value = (1, False)  # Not final
        turn = make_turn()

        result = await credit_issuer.issue_credit(turn)

        assert result is True

    async def test_issue_credit_returns_false_when_final_credit(
        self, credit_issuer, mock_progress
    ):
        """Should return False when this is the final credit."""
        mock_progress.increment_sent.return_value = (10, True)  # Final credit
        turn = make_turn()

        result = await credit_issuer.issue_credit(turn)

        assert result is False


# =============================================================================
# Test: Slot Acquisition Failures
# =============================================================================


class TestSlotAcquisitionFailures:
    """Tests for when slot acquisition fails."""

    async def test_first_turn_returns_false_when_session_slot_fails(
        self, credit_issuer, mock_concurrency, mock_router
    ):
        """First turn should return False if session slot acquisition fails."""
        mock_concurrency.acquire_session_slot.return_value = False
        turn = make_turn(turn_index=0)

        result = await credit_issuer.issue_credit(turn)

        assert result is False
        mock_concurrency.acquire_prefill_slot.assert_not_called()
        mock_router.send_credit.assert_not_called()

    async def test_first_turn_releases_session_slot_when_prefill_fails(
        self, credit_issuer, mock_concurrency, mock_router
    ):
        """First turn should release session slot if prefill acquisition fails."""
        mock_concurrency.acquire_session_slot.return_value = True
        mock_concurrency.acquire_prefill_slot.return_value = False
        turn = make_turn(turn_index=0)

        result = await credit_issuer.issue_credit(turn)

        assert result is False
        mock_concurrency.release_session_slot.assert_called_once_with(
            CreditPhase.PROFILING
        )
        mock_router.send_credit.assert_not_called()

    async def test_subsequent_turn_returns_false_when_prefill_fails(
        self, credit_issuer, mock_concurrency, mock_router
    ):
        """Subsequent turn should return False if prefill acquisition fails."""
        mock_concurrency.acquire_prefill_slot.return_value = False
        turn = make_turn(turn_index=1)  # Not first turn

        result = await credit_issuer.issue_credit(turn)

        assert result is False
        mock_concurrency.acquire_session_slot.assert_not_called()
        mock_concurrency.release_session_slot.assert_not_called()
        mock_router.send_credit.assert_not_called()


# =============================================================================
# Test: Stop Condition Checking
# =============================================================================


class TestStopConditionChecking:
    """Tests for stop condition integration."""

    async def test_first_turn_uses_can_start_new_session_check(
        self, credit_issuer, mock_concurrency, mock_stop_checker
    ):
        """First turn should use can_start_new_session for stop check."""
        turn = make_turn(turn_index=0)

        await credit_issuer.issue_credit(turn)

        # Verify the correct check function was passed to acquire_session_slot
        call_args = mock_concurrency.acquire_session_slot.call_args
        check_fn = call_args[0][1]  # Second positional arg is the check function
        assert check_fn == mock_stop_checker.can_start_new_session

    async def test_subsequent_turn_uses_can_send_any_turn_check(
        self, credit_issuer, mock_concurrency, mock_stop_checker
    ):
        """Subsequent turn should use can_send_any_turn for stop check."""
        turn = make_turn(turn_index=1)

        await credit_issuer.issue_credit(turn)

        # Verify the correct check function was passed to acquire_prefill_slot
        call_args = mock_concurrency.acquire_prefill_slot.call_args
        check_fn = call_args[0][1]  # Second positional arg is the check function
        assert check_fn == mock_stop_checker.can_send_any_turn


# =============================================================================
# Test: Final Credit Handling
# =============================================================================


class TestFinalCreditHandling:
    """Tests for handling of final credits."""

    async def test_final_credit_freezes_sent_counts(self, credit_issuer, mock_progress):
        """Final credit should freeze sent counts."""
        mock_progress.increment_sent.return_value = (10, True)  # Final credit
        turn = make_turn()

        await credit_issuer.issue_credit(turn)

        mock_progress.freeze_sent_counts.assert_called_once()

    async def test_final_credit_sets_event(self, credit_issuer, mock_progress):
        """Final credit should set the all_credits_sent_event."""
        mock_progress.increment_sent.return_value = (10, True)  # Final credit
        turn = make_turn()

        await credit_issuer.issue_credit(turn)

        assert mock_progress.all_credits_sent_event.is_set()

    async def test_non_final_credit_does_not_freeze_or_set_event(
        self, credit_issuer, mock_progress
    ):
        """Non-final credit should not freeze counts or set event."""
        mock_progress.increment_sent.return_value = (5, False)  # Not final
        turn = make_turn()

        await credit_issuer.issue_credit(turn)

        mock_progress.freeze_sent_counts.assert_not_called()
        assert not mock_progress.all_credits_sent_event.is_set()


# =============================================================================
# Test: Cancellation Policy Integration
# =============================================================================


class TestCancellationPolicy:
    """Tests for cancellation policy integration."""

    async def test_credit_includes_cancellation_delay_when_set(
        self, credit_issuer, mock_router, mock_cancellation
    ):
        """Credit should include cancel_after_ns when cancellation is enabled."""
        mock_cancellation.next_cancellation_delay_ns.return_value = 5_000_000_000  # 5s
        turn = make_turn()

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.cancel_after_ns == 5_000_000_000

    async def test_credit_has_no_cancellation_when_disabled(
        self, credit_issuer, mock_router, mock_cancellation
    ):
        """Credit should have None cancel_after_ns when cancellation disabled."""
        mock_cancellation.next_cancellation_delay_ns.return_value = None
        turn = make_turn()

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.cancel_after_ns is None

    async def test_cancellation_policy_receives_turn_and_phase(
        self, credit_issuer, mock_cancellation
    ):
        """Cancellation policy should receive turn and phase."""
        turn = make_turn(conversation_id="test-conv")

        await credit_issuer.issue_credit(turn)

        mock_cancellation.next_cancellation_delay_ns.assert_called_once_with(
            turn, CreditPhase.PROFILING
        )


# =============================================================================
# Test: Atomic Credit Numbering
# =============================================================================


class TestAtomicCreditNumbering:
    """Tests for credit numbering via progress tracker."""

    async def test_credits_receive_sequential_ids(
        self,
        mock_stop_checker,
        mock_concurrency,
        mock_router,
        mock_cancellation,
        mock_lifecycle,
    ):
        """Each credit should receive a unique sequential ID."""
        progress = MagicMock()
        progress.all_credits_sent_event = asyncio.Event()
        call_count = [0]

        def increment_sent(turn):
            call_count[0] += 1
            return (call_count[0], call_count[0] >= 3)  # Final at 3rd call

        progress.increment_sent = increment_sent
        progress.freeze_sent_counts = MagicMock()

        issuer = CreditIssuer(
            phase=CreditPhase.PROFILING,
            stop_checker=mock_stop_checker,
            progress=progress,
            concurrency_manager=mock_concurrency,
            credit_router=mock_router,
            cancellation_policy=mock_cancellation,
            lifecycle=mock_lifecycle,
        )

        turns = [make_turn(f"conv{i}") for i in range(3)]
        for turn in turns:
            await issuer.issue_credit(turn)

        # Verify sequential IDs
        sent_credits = [
            call.kwargs["credit"] for call in mock_router.send_credit.call_args_list
        ]
        assert [c.id for c in sent_credits] == [1, 2, 3]


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_single_turn_conversation(self, credit_issuer, mock_router):
        """Single-turn conversation should work correctly."""
        turn = make_turn(turn_index=0, num_turns=1)

        result = await credit_issuer.issue_credit(turn)

        assert result is True
        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.turn_index == 0
        assert sent_credit.num_turns == 1

    async def test_warmup_phase(
        self,
        mock_stop_checker,
        mock_progress,
        mock_concurrency,
        mock_router,
        mock_cancellation,
        mock_lifecycle,
    ):
        """CreditIssuer should work with WARMUP phase."""
        issuer = CreditIssuer(
            phase=CreditPhase.WARMUP,
            stop_checker=mock_stop_checker,
            progress=mock_progress,
            concurrency_manager=mock_concurrency,
            credit_router=mock_router,
            cancellation_policy=mock_cancellation,
            lifecycle=mock_lifecycle,
        )
        turn = make_turn()

        await issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.phase == CreditPhase.WARMUP

    async def test_large_conversation_with_many_turns(self, credit_issuer, mock_router):
        """Should handle conversations with many turns."""
        turn = make_turn(turn_index=99, num_turns=100)  # Last turn of 100-turn conv

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.turn_index == 99
        assert sent_credit.num_turns == 100


# =============================================================================
# Test: Concurrency Slot Contract
# =============================================================================


class TestConcurrencySlotContract:
    """Tests verifying the concurrency slot acquisition contract."""

    @pytest.mark.parametrize(
        "turn_index,expects_session_acquire",
        [
            (0, True),   # First turn acquires session
            (1, False),  # Second turn doesn't
            (2, False),  # Third turn doesn't
            (9, False),  # 10th turn doesn't
        ],
    )  # fmt: skip
    async def test_session_slot_only_acquired_on_first_turn(
        self,
        credit_issuer,
        mock_concurrency,
        turn_index: int,
        expects_session_acquire: bool,
    ):
        """Session slot should only be acquired on first turn (turn_index=0)."""
        turn = make_turn(turn_index=turn_index, num_turns=10)

        await credit_issuer.issue_credit(turn)

        if expects_session_acquire:
            mock_concurrency.acquire_session_slot.assert_called_once()
        else:
            mock_concurrency.acquire_session_slot.assert_not_called()

    async def test_prefill_slot_acquired_on_every_turn(
        self, credit_issuer, mock_concurrency
    ):
        """Prefill slot should be acquired on every turn."""
        for turn_index in range(5):
            mock_concurrency.reset_mock()
            turn = make_turn(turn_index=turn_index, num_turns=5)

            await credit_issuer.issue_credit(turn)

            mock_concurrency.acquire_prefill_slot.assert_called_once()


# =============================================================================
# Test: Issued At Timestamp
# =============================================================================


class TestIssuedAtTimestamp:
    """Tests for credit timestamp accuracy."""

    async def test_issued_at_ns_is_recent(self, credit_issuer, mock_router):
        """Issued timestamp should be very recent (within 1 second)."""
        before = time.time_ns()
        turn = make_turn()

        await credit_issuer.issue_credit(turn)

        after = time.time_ns()
        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]

        assert before <= sent_credit.issued_at_ns <= after
        # Should be within 1 second
        assert (after - sent_credit.issued_at_ns) < 1_000_000_000


# =============================================================================
# Test: URL Selection Strategy Integration
# =============================================================================


class TestURLSelectionStrategy:
    """Tests for URL selection in multi-URL mode.

    When multiple --url endpoints are configured, the URL selection strategy
    (round-robin) should only be invoked on the first turn of a conversation.
    Subsequent turns get url_index=None and rely on the worker's session cache.
    """

    async def test_first_turn_gets_url_index_from_strategy(
        self,
        mock_stop_checker,
        mock_progress,
        mock_concurrency,
        mock_router,
        mock_cancellation,
        mock_lifecycle,
    ):
        """First turn should get url_index from URL selection strategy."""
        mock_url_strategy = MagicMock()
        mock_url_strategy.next_url_index.return_value = 2

        issuer = CreditIssuer(
            phase=CreditPhase.PROFILING,
            stop_checker=mock_stop_checker,
            progress=mock_progress,
            concurrency_manager=mock_concurrency,
            credit_router=mock_router,
            cancellation_policy=mock_cancellation,
            lifecycle=mock_lifecycle,
            url_selection_strategy=mock_url_strategy,
        )

        turn = make_turn(turn_index=0, num_turns=3)  # First turn
        await issuer.issue_credit(turn)

        # Strategy should be called for first turn
        mock_url_strategy.next_url_index.assert_called_once()
        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.url_index == 2

    async def test_subsequent_turns_get_none_url_index(
        self,
        mock_stop_checker,
        mock_progress,
        mock_concurrency,
        mock_router,
        mock_cancellation,
        mock_lifecycle,
    ):
        """Subsequent turns should get url_index=None (worker uses session cache)."""
        mock_url_strategy = MagicMock()
        mock_url_strategy.next_url_index.return_value = 5  # Should NOT be used

        issuer = CreditIssuer(
            phase=CreditPhase.PROFILING,
            stop_checker=mock_stop_checker,
            progress=mock_progress,
            concurrency_manager=mock_concurrency,
            credit_router=mock_router,
            cancellation_policy=mock_cancellation,
            lifecycle=mock_lifecycle,
            url_selection_strategy=mock_url_strategy,
        )

        turn = make_turn(turn_index=1, num_turns=3)  # NOT first turn
        await issuer.issue_credit(turn)

        # Strategy should NOT be called for subsequent turns
        mock_url_strategy.next_url_index.assert_not_called()
        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.url_index is None

    async def test_multi_turn_conversation_only_first_turn_advances_round_robin(
        self,
        mock_stop_checker,
        mock_progress,
        mock_concurrency,
        mock_router,
        mock_cancellation,
        mock_lifecycle,
    ):
        """Multi-turn conversation: only first turn should advance round-robin.

        This ensures all turns in a conversation hit the same backend server.
        The worker stores url_index from first turn in UserSession and uses
        that for all subsequent turns.
        """
        mock_url_strategy = MagicMock()
        call_count = [0]

        def next_url():
            idx = call_count[0]
            call_count[0] += 1
            return idx

        mock_url_strategy.next_url_index.side_effect = next_url

        issuer = CreditIssuer(
            phase=CreditPhase.PROFILING,
            stop_checker=mock_stop_checker,
            progress=mock_progress,
            concurrency_manager=mock_concurrency,
            credit_router=mock_router,
            cancellation_policy=mock_cancellation,
            lifecycle=mock_lifecycle,
            url_selection_strategy=mock_url_strategy,
        )

        # Simulate 3-turn conversation
        for turn_index in range(3):
            turn = make_turn(
                conversation_id="multi-turn-conv",
                turn_index=turn_index,
                num_turns=3,
            )
            await issuer.issue_credit(turn)

        # Round-robin should only advance once (for first turn)
        assert mock_url_strategy.next_url_index.call_count == 1

        # Check credits: first turn has url_index=0, others have None
        sent_credits = [
            call.kwargs["credit"] for call in mock_router.send_credit.call_args_list
        ]
        assert sent_credits[0].url_index == 0  # First turn gets index
        assert sent_credits[1].url_index is None  # Subsequent turns: None
        assert sent_credits[2].url_index is None

    async def test_no_url_strategy_means_none_url_index(
        self, credit_issuer, mock_router
    ):
        """Without URL strategy, all credits should have url_index=None."""
        turn = make_turn(turn_index=0, num_turns=1)

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.url_index is None
