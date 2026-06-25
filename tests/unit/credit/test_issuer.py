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
    mock.can_send_child_turn = MagicMock(return_value=True)
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
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
    counts_toward_phase_target: bool = True,
) -> TurnToSend:
    """Create a TurnToSend for testing."""
    return TurnToSend(
        conversation_id=conversation_id,
        x_correlation_id=f"corr-{conversation_id}",
        turn_index=turn_index,
        num_turns=num_turns,
        agent_depth=agent_depth,
        parent_correlation_id=parent_correlation_id,
        counts_toward_phase_target=counts_toward_phase_target,
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

    async def test_issue_credit_propagates_max_tokens_override(
        self, credit_issuer, mock_router
    ):
        turn = make_turn()
        turn = TurnToSend(
            conversation_id=turn.conversation_id,
            x_correlation_id=turn.x_correlation_id,
            turn_index=turn.turn_index,
            num_turns=turn.num_turns,
            max_tokens_override=1,
        )

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.max_tokens_override == 1

    async def test_runtime_max_tokens_override_covers_undecorated_turns(
        self, credit_issuer, mock_router
    ):
        credit_issuer.set_max_tokens_override(1)

        await credit_issuer.issue_credit(make_turn())

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.max_tokens_override == 1

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

    async def test_child_credit_uses_can_send_child_turn_check(
        self, credit_issuer, mock_concurrency, mock_stop_checker
    ):
        """DAG children must use ``can_send_child_turn`` — the narrow
        bypass that skips only ``is_sending_complete`` while still
        honoring cancellation, duration, and count limits.

        Children must use ``can_send_child_turn`` so user Ctrl-C, benchmark
        duration, and request-count limits still apply to DAG descendants.
        """
        turn = make_turn(turn_index=0, agent_depth=1, parent_correlation_id="parent-x")

        await credit_issuer.issue_credit(turn)

        call_args = mock_concurrency.acquire_prefill_slot.call_args
        check_fn = call_args[0][1]
        assert check_fn == mock_stop_checker.can_send_child_turn

    async def test_child_credit_blocked_when_can_send_child_turn_false(
        self, credit_issuer, mock_concurrency, mock_stop_checker
    ):
        """When ``can_send_child_turn`` returns False (cancellation /
        duration / count limit reached), prefill-slot acquisition is
        called with the gate — and the slot manager is responsible for
        declining. The issuer itself doesn't need to pre-check because
        the gate is passed into acquire_prefill_slot directly."""
        mock_stop_checker.can_send_child_turn = MagicMock(return_value=False)
        mock_concurrency.acquire_prefill_slot = AsyncMock(return_value=False)

        turn = make_turn(turn_index=0, agent_depth=1, parent_correlation_id="parent-x")

        result = await credit_issuer.issue_credit(turn)
        assert result is False


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


# =============================================================================
# Test: DAG fields propagation
# =============================================================================


class TestDagFieldsPropagation:
    """Tests for agent_depth / parent_correlation_id propagation through Credit."""

    async def test_credit_inherits_depth_and_parent_from_turn(
        self, credit_issuer, mock_router
    ):
        """Credit should carry agent_depth / parent_correlation_id from TurnToSend."""
        turn = TurnToSend(
            conversation_id="child-conv",
            x_correlation_id="child-xid",
            turn_index=0,
            num_turns=2,
            agent_depth=1,
            parent_correlation_id="parent-xid",
        )

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.agent_depth == 1
        assert sent_credit.parent_correlation_id == "parent-xid"
        assert sent_credit.counts_toward_phase_target is True

    async def test_credit_inherits_phase_target_membership_from_turn(
        self, credit_issuer, mock_router
    ):
        """Credit should carry explicit phase-target membership."""
        turn = make_turn(counts_toward_phase_target=False)

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.counts_toward_phase_target is False

    async def test_credit_default_depth_and_parent_when_unset(
        self, credit_issuer, mock_router
    ):
        """Credit should default to depth=0 / parent=None when TurnToSend does not set them."""
        turn = make_turn(turn_index=0, num_turns=1)

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.agent_depth == 0
        assert sent_credit.parent_correlation_id is None


# =============================================================================
# Test: Cache-bust fields propagation
# =============================================================================


class TestCacheBustFieldsPropagation:
    """Tests that cache_bust_marker / cache_bust_target propagate from TurnToSend
    to the issued Credit. Without this propagation the worker would always read
    None from credit.cache_bust_marker and the feature would never inject.
    """

    async def test_credit_inherits_cache_bust_fields_from_turn(
        self, credit_issuer, mock_router
    ):
        """Credit must carry cache_bust_marker / cache_bust_target from TurnToSend."""
        from aiperf.common.enums import CacheBustTarget

        turn = TurnToSend(
            conversation_id="conv1",
            x_correlation_id="corr-conv1",
            turn_index=0,
            num_turns=1,
            cache_bust_marker="\n\n[rid:test123abcde]",
            cache_bust_target=CacheBustTarget.SYSTEM_SUFFIX,
        )

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.cache_bust_marker == "\n\n[rid:test123abcde]"
        assert sent_credit.cache_bust_target == CacheBustTarget.SYSTEM_SUFFIX

    async def test_credit_default_cache_bust_fields_when_unset(
        self, credit_issuer, mock_router
    ):
        """Credit must default to marker=None / target=NONE when TurnToSend does not set them."""
        from aiperf.common.enums import CacheBustTarget

        turn = make_turn(turn_index=0, num_turns=1)

        await credit_issuer.issue_credit(turn)

        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.cache_bust_marker is None
        assert sent_credit.cache_bust_target == CacheBustTarget.NONE


# =============================================================================
# Test: dispatch_first_turn / dispatch_join_turn
# =============================================================================


class TestDispatchFirstTurn:
    """Tests for CreditIssuer.dispatch_first_turn."""

    async def test_dispatch_first_turn_issues_via_try_issue_credit(
        self, credit_issuer, mock_router, mock_progress
    ):
        """dispatch_first_turn should issue via try_issue_credit with depth/parent propagated."""
        from aiperf.common.models import ConversationMetadata, TurnMetadata
        from aiperf.timing.conversation_source import SampledSession

        metadata = ConversationMetadata(
            conversation_id="child-conv",
            turns=[TurnMetadata(timestamp_ms=0.0), TurnMetadata(timestamp_ms=1.0)],
        )
        session = SampledSession(
            conversation_id="child-conv",
            metadata=metadata,
            x_correlation_id="child-xid",
            agent_depth=1,
            parent_correlation_id="parent-xid",
        )

        result = await credit_issuer.dispatch_first_turn(session)

        assert result is True
        sent_credit = mock_router.send_credit.call_args.kwargs["credit"]
        assert sent_credit.conversation_id == "child-conv"
        assert sent_credit.x_correlation_id == "child-xid"
        assert sent_credit.turn_index == 0
        assert sent_credit.num_turns == 2
        assert sent_credit.agent_depth == 1
        assert sent_credit.parent_correlation_id == "parent-xid"
        assert sent_credit.counts_toward_phase_target is False

        counted_turn = mock_progress.increment_sent.call_args.args[0]
        assert counted_turn.counts_toward_phase_target is False

    async def test_dispatch_first_turn_bypasses_session_slot_for_subagent(
        self, credit_issuer, mock_concurrency, mock_router
    ):
        """dispatch_first_turn bypasses session-slot acquisition for DAG
        children (agent_depth > 0).

        Children inherit the root's session slot, so the issuer must never
        attempt to acquire a new one. The prefill slot is acquired through
        the blocking path so temporary saturation applies backpressure rather
        than dropping the child.
        """
        from aiperf.common.models import ConversationMetadata, TurnMetadata
        from aiperf.timing.conversation_source import SampledSession

        # Session slot path would fail; prefill slot is available.
        mock_concurrency.try_acquire_session_slot = MagicMock(return_value=False)
        mock_concurrency.acquire_prefill_slot = AsyncMock(return_value=True)

        metadata = ConversationMetadata(
            conversation_id="child-conv",
            turns=[TurnMetadata(timestamp_ms=0.0)],
        )
        session = SampledSession(
            conversation_id="child-conv",
            metadata=metadata,
            x_correlation_id="child-xid",
            agent_depth=1,
            parent_correlation_id="parent-xid",
        )

        result = await credit_issuer.dispatch_first_turn(session)

        assert result is True
        # Session-slot acquisition must NOT have been attempted: DAG children
        # inherit the parent's session slot rather than acquiring a new one.
        mock_concurrency.try_acquire_session_slot.assert_not_called()
        # Prefill slot was acquired through the blocking path.
        mock_concurrency.acquire_prefill_slot.assert_awaited_once()
        # The credit was sent to the router.
        mock_router.send_credit.assert_called_once()

    async def test_dispatch_first_turn_returns_false_when_prefill_wait_is_stopped(
        self, credit_issuer, mock_concurrency, mock_router
    ):
        """If a blocked prefill acquisition is stopped by phase conditions,
        the child is not issued and the caller receives the rollback signal.
        """
        from aiperf.common.models import ConversationMetadata, TurnMetadata
        from aiperf.timing.conversation_source import SampledSession

        mock_concurrency.try_acquire_session_slot = MagicMock(return_value=False)
        mock_concurrency.acquire_prefill_slot = AsyncMock(return_value=False)

        metadata = ConversationMetadata(
            conversation_id="child-conv",
            turns=[TurnMetadata(timestamp_ms=0.0)],
        )
        session = SampledSession(
            conversation_id="child-conv",
            metadata=metadata,
            x_correlation_id="child-xid",
            agent_depth=1,
            parent_correlation_id="parent-xid",
        )

        result = await credit_issuer.dispatch_first_turn(session)

        assert result is False
        # No credit was actually sent (slot acquisition failed).
        mock_router.send_credit.assert_not_called()

    async def test_parallel_children_wait_for_prefill_capacity_instead_of_dropping(
        self,
        credit_issuer,
        mock_router,
    ):
        """A one-slot prefill limit must queue the second sibling.

        The old non-blocking child path issued the first sibling and returned
        False for the second, causing the orchestrator to permanently remove
        it from the branch.
        """
        from aiperf.common.models import ConversationMetadata, TurnMetadata
        from aiperf.timing.concurrency import ConcurrencyManager
        from aiperf.timing.conversation_source import SampledSession

        concurrency = ConcurrencyManager()
        concurrency.configure_for_phase(
            CreditPhase.PROFILING,
            concurrency=None,
            prefill_concurrency=1,
        )
        credit_issuer._concurrency_manager = concurrency

        metadata = ConversationMetadata(
            conversation_id="child-template",
            turns=[TurnMetadata(timestamp_ms=0.0)],
        )
        sessions = [
            SampledSession(
                conversation_id=f"child-{i}",
                metadata=metadata,
                x_correlation_id=f"child-xid-{i}",
                agent_depth=1,
                parent_correlation_id="parent-xid",
            )
            for i in range(2)
        ]

        tasks = [
            asyncio.create_task(credit_issuer.dispatch_first_turn(session))
            for session in sessions
        ]
        for _ in range(10):
            await asyncio.sleep(0)
            if mock_router.send_credit.await_count == 1:
                break

        assert mock_router.send_credit.await_count == 1
        assert sum(task.done() for task in tasks) == 1

        concurrency.release_prefill_slot(CreditPhase.PROFILING)
        assert await asyncio.gather(*tasks) == [True, True]
        assert mock_router.send_credit.await_count == 2

        concurrency.release_prefill_slot(CreditPhase.PROFILING)


# =============================================================================
# Test: Lane credit (session slot held by a trajectory lane, not a credit)
# =============================================================================


class TestLaneCredit:
    """A trajectory lane can hold a session slot directly, with no root credit.

    Agentic replay dispatches one lane per ``--concurrency`` unit, but some
    lanes issue no slot-acquiring depth-0 root credit at PROFILING start: a
    rootless snapshot (root finished before t*, only background subagents
    remain) and a gated parent deferred on a child join. Such a lane must
    still hold exactly one session slot so it counts toward the configured
    concurrency, WITHOUT a subagent/sidecar acquiring one. ``acquire_lane_credit``
    / ``release_lane_credit`` are that path: a bare session-slot hold, no
    prefill slot and no credit on the wire.
    """

    async def test_acquire_lane_credit_acquires_only_a_session_slot(
        self, credit_issuer, mock_concurrency, mock_router
    ):
        """Acquires a session slot; no prefill slot, no credit sent on the wire."""
        acquired = await credit_issuer.acquire_lane_credit(
            "lane-root", root_pending=False
        )

        assert acquired is True
        mock_concurrency.acquire_session_slot.assert_called_once()
        mock_concurrency.acquire_prefill_slot.assert_not_called()
        mock_router.send_credit.assert_not_called()

    async def test_acquire_lane_credit_uses_can_start_new_session_gate(
        self, credit_issuer, mock_concurrency, mock_stop_checker
    ):
        """The slot acquisition is gated on can_start_new_session (a new lane)."""
        await credit_issuer.acquire_lane_credit("lane-root", root_pending=False)

        _, kwargs = mock_concurrency.acquire_session_slot.call_args
        args = mock_concurrency.acquire_session_slot.call_args.args
        passed = list(args) + list(kwargs.values())
        assert CreditPhase.PROFILING in passed
        assert mock_stop_checker.can_start_new_session in passed

    async def test_acquire_lane_credit_returns_false_when_no_slot(
        self, credit_issuer, mock_concurrency
    ):
        """When the session limiter is saturated, acquisition fails (False)."""
        mock_concurrency.acquire_session_slot = AsyncMock(return_value=False)

        assert (
            await credit_issuer.acquire_lane_credit("lane-root", root_pending=False)
            is False
        )

    def test_release_lane_credit_releases_one_session_slot(
        self, credit_issuer, mock_concurrency
    ):
        """Releasing a lane credit releases exactly one session slot."""
        credit_issuer.release_lane_credit()

        mock_concurrency.release_session_slot.assert_called_once()
