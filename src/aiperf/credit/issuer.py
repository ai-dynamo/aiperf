# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credit issuer for credit lifecycle management.

Handles credit issuance with concurrency control and stop condition checking.

Key responsibilities:
- Acquire concurrency slots (session + prefill)
- Check stop conditions after slot acquisition
- Atomic credit numbering via progress tracker
- Create and send Credit to router
- Signal completion when final credit is issued
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.request_cancellation import RequestCancellationSimulator


class CreditIssuer:
    """Issues credits with concurrency control and stop condition checking.

    Single point of contact for credit issuance operations:
    - Acquire concurrency slots (session on first turn, prefill on every turn)
    - Check stop conditions AFTER slot acquisition (prevents races)
    - Atomic credit numbering via progress tracker
    - Create and send Credit to router
    - Signal all_credits_sent_event when final credit is issued

    Concurrency contract:
    - Session slot: Acquired on first turn only
    - Prefill slot: Acquired on every turn
    - Slots are released on failure to maintain symmetry

    Used by timing strategies to issue credits without knowing about
    concurrency or routing internals.
    """

    def __init__(
        self,
        *,
        phase: CreditPhase,
        stop_checker: StopConditionChecker,
        progress: PhaseProgressTracker,
        concurrency_manager: ConcurrencyManager,
        credit_router: CreditRouterProtocol,
        cancellation_policy: RequestCancellationSimulator,
        lifecycle: PhaseLifecycle,
        url_selection_strategy: URLSelectionStrategyProtocol | None = None,
    ) -> None:
        """Initialize credit issuer.

        Args:
            phase: Phase enum (WARMUP or PROFILING).
            stop_checker: Evaluates stop conditions (can_send_any_turn, can_start_new_session).
            progress: Tracks credit progress (increment_sent, freeze_sent_counts).
            concurrency_manager: Manages concurrency slots (session + prefill).
            credit_router: Routes credits to workers.
            cancellation_policy: Determines cancellation delays.
            lifecycle: Phase lifecycle for timestamp data.
            url_selection_strategy: Optional URL selection strategy for multi-URL load
                balancing. If None, url_index will be None in credits.
        """
        self._phase = phase
        self._stop_checker = stop_checker
        self._progress = progress
        self._concurrency_manager = concurrency_manager
        self._credit_router = credit_router
        self._cancellation_policy = cancellation_policy
        self._lifecycle = lifecycle
        self._url_selection_strategy = url_selection_strategy

    def can_acquire_and_start_new_session(self) -> bool:
        """Check if a session slot can be acquired and a new session can be started."""
        return (
            self._concurrency_manager.session_slot_available(self._phase)
            and self._stop_checker.can_start_new_session()
        )

    async def issue_credit(self, turn: TurnToSend) -> bool:
        """Issue credit with full precondition checking.

        Acquires necessary concurrency slots, increments counters,
        creates Credit struct, and sends to router.

        Returns:
            True if more credits can be sent.
            False if this was the final credit or couldn't acquire slots.

        Note:
            For first turns (turn_index == 0), acquires session slot first.
            For all turns, acquires prefill slot.
            Slots are released automatically on failure.

        Flow:
            1. Acquire session slot (first turn only)
            2. Acquire prefill slot (all turns)
            3. Atomic numbering via increment_sent
            4. Calculate cancellation delay
            5. Create and send Credit
            6. If final credit: freeze counts + set event
        """
        is_first_turn = turn.turn_index == 0

        # Select appropriate check function based on turn type
        # - First turns need can_start_new_session (more restrictive - checks session quota)
        # - Subsequent turns use can_send_any_turn (less restrictive - allows finishing existing sessions)
        can_proceed_fn = (
            self._stop_checker.can_start_new_session
            if is_first_turn
            else self._stop_checker.can_send_any_turn
        )

        # Session concurrency: one slot per conversation, acquired on first turn only.
        # Controls how many multi-turn conversations can be active simultaneously.
        if is_first_turn:
            acquired = await self._concurrency_manager.acquire_session_slot(
                self._phase, self._stop_checker.can_start_new_session
            )
            if not acquired:
                return False

        # Prefill concurrency: one slot per request, released when TTFT arrives.
        # Limits concurrent prompt processing which is the GPU-intensive phase.
        acquired = await self._concurrency_manager.acquire_prefill_slot(
            self._phase, can_proceed_fn
        )
        if not acquired:
            # CRITICAL: Release session slot if we acquired it to maintain symmetry
            if is_first_turn:
                self._concurrency_manager.release_session_slot(self._phase)
            return False

        # Slots acquired - proceed with credit issuance
        return await self._issue_credit_internal(turn)

    async def try_issue_credit(self, turn: TurnToSend) -> bool | None:
        """Try to issue credit without blocking on concurrency slots.

        Non-blocking version of issue_credit for polling-based strategies.
        Returns immediately if slots aren't available.

        Args:
            turn: The turn to send.

        Returns:
            True: Credit issued, more credits can be sent.
            False: Credit issued but this was final, OR stop condition triggered.
            None: No slots available, credit NOT issued. Retry later.
        """
        is_first_turn = turn.turn_index == 0

        # Select appropriate check function based on turn type
        can_proceed_fn = (
            self._stop_checker.can_start_new_session
            if is_first_turn
            else self._stop_checker.can_send_any_turn
        )

        # Check stop condition FIRST - distinguishes False from None
        if not can_proceed_fn():
            return False

        if is_first_turn:
            acquired = self._concurrency_manager.try_acquire_session_slot(
                self._phase, can_proceed_fn
            )
            if not acquired:
                return None  # No slot - credit not issued

        acquired = self._concurrency_manager.try_acquire_prefill_slot(
            self._phase, can_proceed_fn
        )
        if not acquired:
            # CRITICAL: Release session slot if we acquired it to maintain symmetry
            if is_first_turn:
                self._concurrency_manager.release_session_slot(self._phase)
            return None  # No slot - credit not issued

        return await self._issue_credit_internal(turn)

    async def _issue_credit_internal(self, turn: TurnToSend) -> bool:
        """Issue credit after slots are acquired. Mark as final if this was the final credit.

        Returns:
            True if more credits can be sent, False if this was the final credit.
        """
        credit_index, is_final_credit = self._progress.increment_sent(turn)

        cancel_after_ns = self._cancellation_policy.next_cancellation_delay_ns(
            turn, self._phase
        )
        issued_at_ns = self._lifecycle.started_at_ns + (
            time.perf_counter_ns() - self._lifecycle.started_at_perf_ns
        )

        # Get URL index from strategy (for multi-URL load balancing)
        # Only advance the round-robin on the first turn of a conversation.
        # Subsequent turns will use the url_index stored in the worker's UserSession.
        is_first_turn = turn.turn_index == 0
        url_index = (
            self._url_selection_strategy.next_url_index()
            if self._url_selection_strategy and is_first_turn
            else None
        )

        credit = Credit(
            id=credit_index,
            phase=self._phase,
            conversation_id=turn.conversation_id,
            x_correlation_id=turn.x_correlation_id,
            turn_index=turn.turn_index,
            num_turns=turn.num_turns,
            issued_at_ns=issued_at_ns,
            cancel_after_ns=cancel_after_ns,
            url_index=url_index,
        )

        await self._credit_router.send_credit(credit=credit)
        if is_final_credit:
            self._progress.freeze_sent_counts()
            self._progress.all_credits_sent_event.set()

        return not is_final_credit
