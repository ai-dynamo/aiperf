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

import hashlib
import time
from typing import TYPE_CHECKING

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing._branch_orchestrator_state import PendingBranchJoin
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.conversation_source import SampledSession
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.request_cancellation import RequestCancellationSimulator
    from aiperf.timing.session_tree import SessionTreeRegistry


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
        session_tree_registry: SessionTreeRegistry | None = None,
        session_tree_registry_enabled: bool | None = None,
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
            session_tree_registry: Optional per-tree finality ledger (DAG
                datasets only). When engaged, a root session start opens a tree
                and the issuer stamps ``is_parent_final`` / ``is_tree_final`` on
                every emitted credit from live tree state. None on non-DAG paths
                (finality stays conservative ``(None, False)``).
            session_tree_registry_enabled: Explicit engage override. When None,
                the registry is engaged iff one was supplied (DAG runs engage it
                in every phase). Set True/False to force it on/off (tests).
        """
        self._phase = phase
        self._stop_checker = stop_checker
        self._progress = progress
        self._concurrency_manager = concurrency_manager
        self._credit_router = credit_router
        self._cancellation_policy = cancellation_policy
        self._lifecycle = lifecycle
        self._url_selection_strategy = url_selection_strategy
        # Per-TEMPLATE URL affinity for sticky graph credits, keyed by the
        # nonce-stripped template id and minted DETERMINISTICALLY from it
        # (`_stable_graph_url_index`) so the WARMUP priming issuer and the
        # PROFILING replay issuer -- separate per-phase objects -- land every
        # instance/recycle of one template on the same backend. Entries
        # persist for the issuer's lifetime BY DESIGN (a recycle after
        # end_graph_trace must reuse the primed backend); the issuer is
        # per-phase and the map is bounded by the corpus's template count.
        self._graph_url_affinity: dict[str, int] = {}
        self._session_tree_registry = (
            session_tree_registry
            if (
                session_tree_registry_enabled
                if session_tree_registry_enabled is not None
                else session_tree_registry is not None
            )
            else None
        )

    def _open_session_tree(self, turn: TurnToSend) -> None:
        """Open a session tree for a root session-start credit just admitted.

        No-op when tree accounting is not engaged (non-DAG). A root session
        start is always depth 0, so the tree root id is the root's own
        ``x_correlation_id``.
        """
        if self._session_tree_registry is not None:
            self._session_tree_registry.open_tree(
                turn.effective_root_correlation_id, self._phase, root_pending=True
            )

    def _finality_for_issue(self, turn: TurnToSend) -> tuple[bool | None, bool]:
        """Issue-time lineage finality from ``SessionTreeRegistry`` state.

        Conservative by spec: returns ``None``/``False`` whenever indeterminate
        (including the non-DAG path where no registry is engaged).
        """
        registry = self._session_tree_registry
        if registry is None:
            return None, False
        root_id = turn.effective_root_correlation_id
        is_root = turn.parent_correlation_id is None
        is_parent_final: bool | None = None
        if not is_root and turn.parent_correlation_id == root_id:
            # v1: parent finality is determinable only when the parent IS the
            # root (the registry tracks per-tree, not per-intermediate-node).
            is_parent_final = registry.root_terminal(root_id)
        is_tree_final = registry.is_last_tree_request(
            root_id,
            is_final_turn=turn.is_final_turn,
            is_root_credit=is_root,
            # Any-mode branch flag, NOT the FORK-only has_forks: a final turn
            # declaring SPAWN branches spawns descendants at return-intercept,
            # after this stamp, so it must never read as tree-final.
            has_branches=turn.has_branches,
        )
        return is_parent_final, is_tree_final

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
            self._open_session_tree(turn)

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
            self._open_session_tree(turn)

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

        # Get URL index from strategy (for multi-URL load balancing).
        # Sticky graph credits get per-template URL affinity minted
        # deterministically from the nonce-stripped template identity (NOT the
        # per-trajectory ``x_correlation_id``, which embeds a fresh uuid and
        # would let warmup priming and profiling replay land on
        # different backends); linear turns only advance the round-robin on
        # the first turn of a conversation -- subsequent turns use the url_index
        # stored in the worker's UserSession.
        if turn.trace_id is not None and self._url_selection_strategy:
            # Key on the nonce-stripped TEMPLATE id: instance ids carry fresh
            # nonces ({template}::{nonce}), and a nonce-bearing key would
            # silently re-shuffle URLs between warmup and profiling.
            affinity_key = turn.trace_id.split("::", 1)[0]
            if affinity_key in self._graph_url_affinity:
                url_index = self._graph_url_affinity[affinity_key]
            else:
                url_index = self._stable_graph_url_index(affinity_key)
                self._graph_url_affinity[affinity_key] = url_index
        else:
            is_first_turn = turn.turn_index == 0
            url_index = (
                self._url_selection_strategy.next_url_index()
                if self._url_selection_strategy and is_first_turn
                else None
            )

        is_parent_final, is_tree_final = self._finality_for_issue(turn)

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
            root_correlation_id=turn.root_correlation_id,
            has_forks=turn.has_forks,
            is_parent_final=is_parent_final,
            is_tree_final=is_tree_final,
            branch_mode=turn.branch_mode,
            trace_id=turn.trace_id,
            node_ordinal=turn.node_ordinal,
            phase_variant=turn.phase_variant,
            first_token_event=turn.first_token_event,
        )

        await self._credit_router.send_credit(credit=credit)
        if is_final_credit:
            self._progress.freeze_sent_counts()
            self._progress.all_credits_sent_event.set()

        return not is_final_credit

    async def issue_graph_credit(self, turn: TurnToSend) -> bool:
        """Issue a graph-IR credit, BYPASSING the linear session-slot lifecycle.

        Graph traces are a fan-out DAG: the executor fires nodes in dataflow-
        readiness order, not the linear ``turn_index==0`` -> ``num_turns-1``
        sequence the session-slot acquire/release arithmetic assumes. Engaging
        ``acquire_session_slot`` (turn0) here would either fail to acquire (the
        first-fired node is rarely ordinal 0) or leak the slot (the terminal
        node's ordinal rarely equals ``num_turns-1``), deadlocking the phase
        once every slot leaks. Trace-admission concurrency is owned by the
        ``GraphIRReplayStrategy`` instead.

        A prefill slot is STILL acquired per request (per-request prompt-
        processing back-pressure is orthogonal to session accounting). The
        ``can_send_any_turn`` stop gate is honored so cancellation /
        duration / request-count caps still apply.

        The matching release-side bypass lives in
        ``CreditCallbackHandler._release_slots_for_return`` (gated on
        ``credit.trace_id is not None``), so a graph credit never touches the
        session-slot counter on either side.

        Returns:
            True iff the credit was placed on the wire.
        """
        # Graph credits use the DAG-child stop gate, NOT ``can_send_any_turn``.
        # ``GraphIRReplayStrategy`` owns completion (a trace is DONE when its
        # executor drains), so the linear ``SessionCountStopCondition``
        # (``--num-conversations``) must NOT truncate a fan-out DAG mid-trace --
        # it opts out of DAG-child gating for exactly this reason. Cancellation,
        # duration, and ``--request-count`` still apply.
        if not self._stop_checker.can_send_dag_child_turn():
            return False
        acquired = await self._concurrency_manager.acquire_prefill_slot(
            self._phase, self._stop_checker.can_send_dag_child_turn
        )
        if not acquired:
            return False
        await self._issue_credit_internal(turn)
        return True

    def _stable_graph_url_index(self, affinity_key: str) -> int:
        """Deterministically map a graph template onto a URL index.

        The mapping must be a pure function of TEMPLATE identity, NOT of mint
        order: warmup priming and profiling replay run in SEPARATE per-phase
        issuers, and every instance/recycle of one template must land on the
        backend that primed its KV -- so the key (the nonce-stripped template
        id) hashes deterministically onto the URL list. Distribution across a corpus's templates is statistically
        uniform under the hash; a strategy exposing no ``urls`` falls back to
        its round-robin mint.
        """
        urls = getattr(self._url_selection_strategy, "urls", None)
        if not urls:
            return self._url_selection_strategy.next_url_index()
        digest = hashlib.sha256(affinity_key.encode()).digest()
        return int.from_bytes(digest[:8], "big") % len(urls)

    async def end_graph_trace(self, trace_id: str, phase_variant: str) -> None:
        """Close a graph instance's sticky lifecycle (router session only).

        Called ONCE per instance by ``GraphIRReplayStrategy`` at the
        adapter-reap points (all in-flight dispatches for the instance
        drained, or phase teardown for retained adapters). Forwards to the
        router, which closes the instance's sticky session and notifies the
        sticky worker (``GraphTraceEnd``). URL affinity is deliberately NOT
        evicted: it keys on the nonce-stripped template, and a recycle of the
        template must land on the backend that primed its KV. Idempotent end
        to end.
        """
        await self._credit_router.end_graph_trace(trace_id, phase_variant)

    def mark_graph_sending_complete(self) -> None:
        """Signal that a graph phase will issue no further credits.

        ``GraphIRReplayStrategy`` owns completion: a graph credit NEVER trips
        ``is_final_credit`` (``CreditCounter.increment_sent`` returns False for
        every ``trace_id``-carrying credit), so the issuer never auto-freezes
        the sent counts or sets ``all_credits_sent_event`` the way it does for
        linear phases (``_issue_credit_internal``). The strategy calls this once
        its executors have all drained -- i.e. every node-dispatch that will
        ever be issued has been issued -- to freeze the authoritative
        ``final_requests_sent`` (the per-node record count) and unblock the
        ``PhaseRunner``'s ``_wait_for_sending_complete``.

        Idempotent: setting an already-set ``asyncio.Event`` is a no-op, and
        re-freezing simply re-snapshots the (now stable) sent counters.
        """
        self._progress.freeze_sent_counts()
        self._progress.all_credits_sent_event.set()

    def graph_all_returned(self) -> bool:
        """True iff every issued graph credit has returned or cancelled.

        Reads the same ``check_all_returned_or_cancelled`` predicate the
        callback handler uses; meaningful only AFTER
        ``mark_graph_sending_complete`` has frozen ``final_requests_sent``.
        """
        return self._progress.check_all_returned_or_cancelled()

    def set_graph_all_returned_event(self) -> None:
        """Set the phase's all-credits-returned event for a graph phase.

        Used by ``GraphIRReplayStrategy`` only in the degenerate
        no-credits-issued case (an empty trace set), where no credit return
        will ever fire the event via the callback handler.
        """
        self._progress.all_credits_returned_event.set()

    # =========================================================================
    # DAG dispatch helpers (used by BranchOrchestrator)
    # =========================================================================

    async def dispatch_first_turn(self, child_session: SampledSession) -> bool:
        """Dispatch turn-0 of a freshly-spawned DAG child session.

        Children inherit the parent's session slot (no new session-slot
        acquisition); they still need a prefill slot per request. The cap
        gate applies — a refused dispatch returns False, which the
        orchestrator counts as ``children_truncated``.

        Args:
            child_session: A ``SampledSession`` produced by
                ``ConversationSource.start_branch_child`` /
                ``start_pre_session_child``.

        Returns:
            True if the credit was sent on the wire, False otherwise.
        """
        turn = child_session.build_first_turn()
        return await self._dispatch_dag_turn(turn)

    async def dispatch_child_turn(self, turn: TurnToSend) -> bool:
        """Dispatch a continuation turn of a DAG child session.

        Used by ``RequestRateStrategy._issue_child_continuation_or_release``
        for non-final child turns. Returns True iff the credit was actually
        placed on the wire (so the strategy can distinguish "dispatched" from
        "stop-blocked / refused at gate").

        Args:
            turn: The continuation turn to dispatch.

        Returns:
            True if the credit was sent on the wire, False otherwise.
        """
        return await self._dispatch_dag_turn(turn)

    async def dispatch_join_turn(self, pending: PendingBranchJoin) -> bool:
        """Dispatch a parent's gated turn after all children drained.

        Builds a ``TurnToSend`` from the ``PendingBranchJoin`` and sends it
        via the standard DAG dispatch path. Used by
        ``BranchOrchestrator._release_blocked_join``.

        Args:
            pending: The ``PendingBranchJoin`` whose gate is satisfied.

        Returns:
            True if the credit was sent on the wire, False if the cap
            blocked it (orchestrator tallies as ``joins_suppressed``).
        """
        if pending.gated_turn_index is None:
            return False
        turn = TurnToSend(
            conversation_id=pending.parent_conversation_id,
            x_correlation_id=pending.parent_x_correlation_id,
            turn_index=pending.gated_turn_index,
            num_turns=pending.parent_num_turns,
            agent_depth=pending.parent_agent_depth,
            parent_correlation_id=pending.parent_parent_correlation_id,
            root_correlation_id=pending.parent_root_correlation_id,
            has_forks=pending.parent_has_forks_on_gated_turn,
            has_branches=pending.parent_has_branches_on_gated_turn,
            branch_mode=pending.parent_branch_mode,
        )
        return await self._dispatch_dag_turn(turn)

    async def abort_session(self, x_correlation_id: str) -> None:
        """Abort an in-flight session (FORK/SPAWN parent or orphan).

        Currently a no-op: the credit-return slot-release path covers every
        reachable case under the v1 orchestrator. This method exists so the
        orchestrator's ``hasattr(self._issuer, "abort_session")`` guard
        resolves; under ``AIPERF_DAG_FAIL_FAST=true`` the orchestrator calls
        this when a child errors and the parent / orphan siblings must be
        torn down.

        If implemented in the future, the contract is:

        - Cancel any in-flight credit for ``x_correlation_id`` (via the router).
        - Release the session slot (do NOT release sibling sessions' slots).
        - Be idempotent -- orchestrator calls it once per session, but errors
          mid-tear-down may re-enter.
        - Be exception-safe -- orchestrator does not retry.
        """
        return None

    async def _dispatch_dag_turn(self, turn: TurnToSend) -> bool:
        """Send a DAG turn (child first/continuation, or parent join) on the
        wire. Bypasses session-slot acquisition (children share the root's
        slot) but still acquires a prefill slot and respects the
        ``can_send_dag_child_turn`` stop gate (``--request-count`` /
        duration / cancellation honored; ``--num-conversations`` bypassed
        for the dispatch since DAG offspring belong to their parent's
        session).

        Returns True iff the credit was actually placed on the wire.
        """
        if not self._stop_checker.can_send_dag_child_turn():
            return False
        acquired = await self._concurrency_manager.acquire_prefill_slot(
            self._phase, self._stop_checker.can_send_dag_child_turn
        )
        if not acquired:
            return False
        # _issue_credit_internal returns True when more credits can be sent
        # and False on the final credit. Either way the credit went out, so
        # we report True.
        await self._issue_credit_internal(turn)
        return True
