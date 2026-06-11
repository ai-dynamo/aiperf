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

from msgspec.structs import replace as _struct_replace

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing.branch_orchestrator import PendingBranchJoin
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.conversation_source import SampledSession
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.request_cancellation import RequestCancellationSimulator


_logger = AIPerfLogger(__name__)


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
            For root first turns (turn_index == 0, agent_depth == 0), acquires
            a session slot first. Root continuations and all DAG-child turns
            (``agent_depth > 0``) inherit the root's session slot and skip
            session-slot acquisition. All turns acquire a prefill slot.
            Slots are released automatically on failure.

        Flow:
            1. Acquire session slot (root first turn only)
            2. Acquire prefill slot (all turns)
            3. Atomic numbering via increment_sent
            4. Calculate cancellation delay
            5. Create and send Credit
            6. If final credit: freeze counts + set event
        """
        # A session start is turn 0 OR an agentic mid-trace resume (flagged via
        # is_session_start, only emitted at a phase's initial dispatch).
        is_session_start = turn.turn_index == 0 or turn.is_session_start
        is_child = turn.agent_depth > 0

        # Select appropriate check function based on turn type.
        # - Root session starts need can_start_new_session (session-quota check).
        # - Root continuations use can_send_any_turn (finish existing sessions).
        # - DAG children use can_send_child_turn: bypasses only the
        #   ``is_sending_complete`` flag (root sampler done) while still
        #   honoring cancellation, duration timeout, and count limits.
        #   Children must progress past the root-sampler-done signal so
        #   the DAG can drain, but a user Ctrl-C or ``--benchmark-duration``
        #   elapse must still terminate children cleanly.
        if is_child:
            can_proceed_fn = self._stop_checker.can_send_child_turn
        else:
            can_proceed_fn = (
                self._stop_checker.can_start_new_session
                if is_session_start
                else self._stop_checker.can_send_any_turn
            )

        # Session concurrency: one slot per root conversation, acquired on its
        # first credit in the phase (turn 0, or a mid-trace resume). DAG
        # children inherit the root's slot and must not acquire their own —
        # fanout would otherwise consume the user's configured session budget.
        needs_session_slot = is_session_start and not is_child
        if needs_session_slot:
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
            if needs_session_slot:
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
        is_session_start = turn.turn_index == 0 or turn.is_session_start
        is_child = turn.agent_depth > 0

        # See issue_credit for the rationale on these three cases.
        if is_child:
            can_proceed_fn = self._stop_checker.can_send_child_turn
        else:
            can_proceed_fn = (
                self._stop_checker.can_start_new_session
                if is_session_start
                else self._stop_checker.can_send_any_turn
            )

        # Check stop condition FIRST - distinguishes False from None
        if not can_proceed_fn():
            return False

        needs_session_slot = is_session_start and not is_child
        if needs_session_slot:
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
            if needs_session_slot:
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

        # Get URL index from strategy (for multi-URL load balancing).
        # Only advance the round-robin when a session starts (turn 0 or a
        # mid-trace resume). Continuations reuse the url_index stored in the
        # worker's UserSession.
        is_session_start = turn.turn_index == 0 or turn.is_session_start
        url_index = (
            self._url_selection_strategy.next_url_index()
            if self._url_selection_strategy and is_session_start
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
            agent_depth=turn.agent_depth,
            parent_correlation_id=turn.parent_correlation_id,
            counts_toward_phase_target=turn.counts_toward_phase_target,
            has_forks=turn.has_forks,
            branch_mode=turn.branch_mode,
            cache_bust_marker=turn.cache_bust_marker,
            cache_bust_target=turn.cache_bust_target,
        )

        await self._credit_router.send_credit(credit=credit)
        if is_final_credit:
            self._progress.freeze_sent_counts()
            self._progress.all_credits_sent_event.set()

        return not is_final_credit

    async def dispatch_first_turn(self, sampled_session: SampledSession) -> bool:
        """Dispatch the first turn of a mid-run DAG child session.

        Thin wrapper around ``dispatch_child_turn`` that builds the
        first ``TurnToSend`` from the sampled session.

        Returns True if the credit was sent on the wire (orchestrator
        should expect a return), False otherwise (orchestrator should
        roll back its tracking via ``BranchOrchestrator.on_child_stopped``
        / per-child rollback).
        """
        return await self.dispatch_child_turn(sampled_session.build_first_turn())

    async def dispatch_child_turn(self, turn: TurnToSend) -> bool:
        """Dispatch a DAG child turn (first or continuation).

        Returns True if the credit was sent on the wire (caller should
        expect a return), False otherwise (caller should roll back its
        tracking via ``BranchOrchestrator.on_child_stopped``).

        We avoid the overloaded ``issue_credit`` / ``try_issue_credit``
        False (which conflates "gate refused, not issued" with "issued,
        was final credit") by inlining the child issuance path here:
        gate check, non-blocking prefill-slot acquisition, then
        ``_issue_credit_internal``. Children skip session-slot
        acquisition (they inherit the parent's slot). The dispatch is
        non-blocking on prefill (``try_acquire_prefill_slot``) — the
        orchestrator drains via ``on_child_stopped`` rather than
        waiting on a slot, matching the prior semantics.
        """
        can_proceed_fn = self._stop_checker.can_send_child_turn
        if not can_proceed_fn():
            return False
        # Children inherit the parent's session slot; only acquire
        # prefill (non-blocking, matches the orchestrator's rollback model).
        if not self._concurrency_manager.try_acquire_prefill_slot(
            self._phase, can_proceed_fn
        ):
            return False
        if turn.counts_toward_phase_target:
            turn = _struct_replace(turn, counts_toward_phase_target=False)
        await self._issue_credit_internal(turn)
        return True

    async def dispatch_join_turn(self, pending: PendingBranchJoin) -> bool:
        """Dispatch a parent's gated turn after all its children complete.

        The parent already holds a session slot (acquired at turn_index=0);
        the gated turn has turn_index > 0, so try_issue_credit's session-slot
        acquisition is naturally skipped (is_first_turn is False). Only a
        prefill slot is acquired here.

        Cache-bust propagation:
            The TurnToSend constructed here re-applies the parent's
            ``cache_bust_marker`` / ``cache_bust_target`` captured on the
            ``PendingBranchJoin`` at suspend time. Without this, turn k+1
            (the join turn) would dispatch with no marker while turns 0..k
            carried one, breaking per-session cache-bust uniqueness for
            multi-turn parents under DAG joins.

        Stop-condition interaction: when ``can_send_any_turn()`` returns
        False, try_issue_credit returns False without issuing and the
        orchestrator increments ``BranchStats.joins_suppressed``.

        Returns:
            True if the credit was issued, False if suppressed.
        """
        assert pending.gated_turn_index is not None, (
            "dispatch_join_turn called without a gated_turn_index"
        )
        turn = TurnToSend(
            conversation_id=pending.parent_conversation_id,
            x_correlation_id=pending.parent_x_correlation_id,
            turn_index=pending.gated_turn_index,
            num_turns=pending.parent_num_turns,
            agent_depth=pending.parent_agent_depth,
            parent_correlation_id=pending.parent_parent_correlation_id,
            # A nested parent (itself a DAG child, agent_depth > 0) resuming its
            # gated turn is reactive DAG work spawned after root sampling, so it
            # must NOT count toward the phase target (mirrors dispatch_child_turn
            # stripping the flag). Only a top-level parent's join turn is part of
            # the sampled root plan and counts.
            counts_toward_phase_target=pending.parent_agent_depth == 0,
            has_forks=pending.parent_has_forks_on_gated_turn,
            branch_mode=pending.parent_branch_mode,
            cache_bust_marker=pending.parent_cache_bust_marker,
            cache_bust_target=pending.parent_cache_bust_target,
        )
        result = await self.try_issue_credit(turn)
        return result is True
