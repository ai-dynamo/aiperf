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

from msgspec.structs import replace as _struct_replace

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.phase import phase_runtime_key
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.graph.ids import template_id
from aiperf.timing.replay_dependencies import ReplayIssueGate
from aiperf.timing.strategies.cache_bust import (
    WARMUP_ISOLATION_MARKER,
    WARMUP_ISOLATION_TARGETS,
)
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing.branch_orchestrator import PendingBranchJoin
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.conversation_source import SampledSession
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator
    from aiperf.timing.request_cancellation import RequestCancellationSimulator
    from aiperf.timing.session_tree import SessionTreeRegistry


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
        phase_index: int | None = None,
        profiling_index: int | None = None,
        phase_name: str | None = None,
        phase_kind: str | None = None,
        stop_checker: StopConditionChecker,
        progress: PhaseProgressTracker,
        concurrency_manager: ConcurrencyManager,
        credit_router: CreditRouterProtocol,
        cancellation_policy: RequestCancellationSimulator,
        lifecycle: PhaseLifecycle,
        url_selection_strategy: URLSelectionStrategyProtocol | None = None,
        session_tree_registry: SessionTreeRegistry | None = None,
        session_tree_registry_enabled: bool | None = None,
        replay_barrier: ReplayBarrierCoordinator | None = None,
        cache_bust_target: CacheBustTarget = CacheBustTarget.NONE,
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
            session_tree_registry: Optional per-tree session-slot ledger (agentic
                replay only). When set and this is the PROFILING phase, a session
                slot acquired for a root (or a lane credit) opens a tree so the
                slot is held until the whole tree drains. None elsewhere (legacy
                per-root-credit release).
        """
        self._phase = phase
        self._phase_index = phase_index
        self._profiling_index = profiling_index
        self._phase_name = phase_name
        self._phase_kind = phase_kind
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
        # Tree accounting defaults to PROFILING-only (WARMUP keeps the legacy
        # in-flight teardown release), but ``session_tree_registry_enabled``
        # overrides that gate so accelerated agentic warmup -- which DOES open
        # trees and spawn descendants during WARMUP -- can engage it.
        self._session_tree_registry = (
            session_tree_registry
            if (
                session_tree_registry_enabled
                if session_tree_registry_enabled is not None
                else phase == CreditPhase.PROFILING
            )
            else None
        )
        self._cache_bust_target = cache_bust_target
        self._issuing_stopped = False
        self._max_tokens_override: int | None = None
        self.replay_gate = ReplayIssueGate(replay_barrier)

    def set_max_tokens_override(self, max_tokens: int | None) -> None:
        """Override generation length for every subsequently issued credit."""
        self._max_tokens_override = max_tokens

    def stop_issuing(self) -> None:
        """Refuse every subsequent root and child credit."""
        self._issuing_stopped = True

    def mark_sending_complete(self) -> None:
        """Wake the phase runner after strategy-controlled issuance ends.

        Stops further issuance and sets ``all_credits_sent_event`` WITHOUT
        freezing the phase's sent counts. The sole caller is the accelerated
        cache-pressure warmup drain (``AgenticReplayStrategy._finish_accelerated_warmup``),
        whose paused DAG branches are handed off to profiling. Completion for
        that phase must flow through the in-flight==0 handoff path
        (``allows_pending_branch_handoff_after_sending_complete``), never the
        frozen-count ``check_all_returned_or_cancelled`` path: with the count
        left unfrozen, ``_final_requests_sent`` stays None so the count-based
        completion check stays inert on a phase that still has work to hand off.
        Use :meth:`signal_sending_complete` instead when the count path MUST
        finalize the phase (e.g. a zero-warmup-credit lane that dispatches
        nothing and would otherwise block forever on the event).
        """
        self.stop_issuing()
        self._progress.all_credits_sent_event.set()

    @property
    def _phase_key(self) -> int | CreditPhase:
        return phase_runtime_key(self._phase, self._phase_index)

    def can_acquire_and_start_new_session(self) -> bool:
        """Check if a session slot can be acquired and a new session can be started."""
        return (
            self._concurrency_manager.session_slot_available(self._phase_key)
            and self._stop_checker.can_start_new_session()
        )

    async def acquire_lane_credit(
        self,
        root_correlation_id: str | None,
        *,
        root_pending: bool,
        session_turns: int = 0,
    ) -> bool:
        """Acquire a session slot held by a trajectory LANE, not by a credit.

        Agentic replay dispatches one lane per ``--concurrency`` unit, but
        some lanes issue no slot-acquiring depth-0 root credit at PROFILING
        start: a rootless snapshot (the root's turns are all before t*, so
        only its background subagents remain) and a parent gated on a child
        join (deferred until its children complete). Such a lane holds one
        session slot directly so it still counts toward the configured
        concurrency, while its subagents/sidecars acquire none (``issue_credit``
        skips slot acquisition for ``agent_depth > 0``). No prefill slot is
        taken and nothing is sent on the wire.

        The acquired slot becomes the session TREE's slot: it opens a tree in
        the ``SessionTreeRegistry`` keyed by ``root_correlation_id`` and is
        released by the registry when the whole tree drains (not via a separate
        ``release_lane_credit`` call).

        Args:
            root_correlation_id: the lane's session-tree root id (the snapshot's
                shared parent_corr).
            root_pending: True for a gated parent (a root credit will still run
                its join turn and reach a terminal turn); False for a truly
                rootless lane (no root credit ever -- drains on its background
                subagents alone).
            session_turns: the gated parent's remaining turn count (num_turns -
                gated_turn_index). Only meaningful when ``root_pending`` is True:
                the gated parent's terminal turn WILL bump ``completed_sessions``,
                so the session is counted in ``sent_sessions`` here to keep the
                accounting symmetric (otherwise ``in_flight_sessions`` goes
                negative). A rootless lane (root_pending=False) bumps neither.

        Returns:
            True if the slot was acquired and ``can_start_new_session``
            allowed it, False otherwise.
        """
        acquired = await self._concurrency_manager.acquire_session_slot(
            self._phase_key, self._stop_checker.can_start_new_session
        )
        if not acquired:
            return False
        if self._session_tree_registry is not None and root_correlation_id is not None:
            self._session_tree_registry.open_tree(
                root_correlation_id, self._phase_key, root_pending=root_pending
            )
        # Count the gated parent's session AFTER acquiring its slot (so it does
        # not gate its own admission via can_start_new_session). A rootless lane
        # reaches no terminal root turn, so it must NOT be counted here.
        if root_pending and session_turns > 0:
            self._progress.account_lane_session(session_turns)
        return True

    def _open_session_tree(self, turn: TurnToSend) -> None:
        """Open a session tree for a root session-start credit just admitted.

        The slot is then held until the whole tree (root + every descendant)
        drains. No-op when tree accounting is not engaged (non-PROFILING /
        non-agentic). A root session start is always depth 0, so the tree root
        id is the root's own ``x_correlation_id``.
        """
        if self._session_tree_registry is not None:
            self._session_tree_registry.open_tree(
                turn.effective_root_correlation_id, self._phase_key, root_pending=True
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

    def release_lane_credit(self) -> None:
        """Release a session slot directly (legacy / non-registry path).

        Retained for callers outside the ``SessionTreeRegistry`` flow; under the
        registry the lane credit's slot is released by the registry when the
        tree drains, so this is not called on that path.
        """
        self._concurrency_manager.release_session_slot(self._phase_key)

    def signal_sending_complete(self) -> None:
        """Mark the phase done sending and set the all-credits-sent event.

        Normally the progress tracker sets ``all_credits_sent_event`` when the
        sent count reaches ``total_expected_requests``. AGENTIC_REPLAY warmup
        sizes that cap to ``concurrency`` (an estimate) but may dispatch FEWER
        real warmup credits (e.g. a single-turn first trace whose t* precedes
        turn 0 yields zero warmup credits). Without an explicit signal the
        count path never reaches the cap and ``PhaseRunner._wait_for_sending_complete``
        blocks forever on the event. The strategy calls this once it has
        dispatched every warmup credit it will send (including none).
        """
        self._progress.freeze_sent_counts()
        self._progress.all_credits_sent_event.set()

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
            3. Open session tree (root first turn only; after both slots succeed)
            4. Atomic numbering via increment_sent
            5. Calculate cancellation delay
            6. Create and send Credit
            7. If final credit: freeze counts + set event
        """
        gate = getattr(self, "replay_gate", ReplayIssueGate(None))
        return await gate.submit(turn, lambda: self._issue_credit_ready(turn))

    async def _issue_credit_ready(self, turn: TurnToSend) -> bool:
        """Issue a turn whose recorded predecessor frontier is complete."""
        if self._issuing_stopped:
            return False

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
                self._phase_key, self._stop_checker.can_start_new_session
            )
            if not acquired:
                return False

        # Prefill concurrency: one slot per request, released when TTFT arrives.
        # Limits concurrent prompt processing which is the GPU-intensive phase.
        acquired = await self._concurrency_manager.acquire_prefill_slot(
            self._phase_key, can_proceed_fn
        )
        if not acquired:
            # CRITICAL: Release session slot if we acquired it to maintain symmetry.
            # Tree is not registered yet (deferred until both slots succeed), so
            # phase teardown release_all cannot double-release this slot.
            if needs_session_slot:
                self._concurrency_manager.release_session_slot(self._phase_key)
            return False

        # Both slots held: register the tree before issuing so drain/teardown
        # own the slot release. Must not run before prefill succeeds.
        if needs_session_slot:
            self._open_session_tree(turn)

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
                self._phase_key, can_proceed_fn
            )
            if not acquired:
                return None  # No slot - credit not issued

        acquired = self._concurrency_manager.try_acquire_prefill_slot(
            self._phase_key, can_proceed_fn
        )
        if not acquired:
            # CRITICAL: Release session slot if we acquired it to maintain symmetry.
            # Tree is not registered yet (deferred until both slots succeed), so
            # phase teardown release_all cannot double-release this slot.
            if needs_session_slot:
                self._concurrency_manager.release_session_slot(self._phase_key)
            return None  # No slot - credit not issued

        if needs_session_slot:
            self._open_session_tree(turn)

        return await self._issue_credit_internal(turn)

    async def _issue_credit_internal(self, turn: TurnToSend) -> bool:
        """Issue credit after slots are acquired. Mark as final if this was the final credit.

        Returns:
            True if more credits can be sent, False if this was the final credit.
        """
        if self._max_tokens_override is not None:
            turn = _struct_replace(turn, max_tokens_override=self._max_tokens_override)
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
        # different backends); linear turns only advance the round-robin when a
        # session starts (turn 0 or a mid-trace resume) -- continuations reuse
        # the url_index stored in the worker's UserSession.
        if turn.trace_id is not None and self._url_selection_strategy:
            # Key on the nonce-stripped TEMPLATE id: instance ids carry fresh
            # nonces ({template}::{nonce}), and a nonce-bearing key would
            # silently re-shuffle URLs between warmup and profiling.
            affinity_key = template_id(turn.trace_id)
            if affinity_key in self._graph_url_affinity:
                url_index = self._graph_url_affinity[affinity_key]
            else:
                url_index = self._stable_graph_url_index(
                    affinity_key, self._url_selection_strategy
                )
                self._graph_url_affinity[affinity_key] = url_index
        else:
            is_session_start = turn.turn_index == 0 or turn.is_session_start
            url_index = (
                self._url_selection_strategy.next_url_index()
                if self._url_selection_strategy and is_session_start
                else None
            )

        is_parent_final, is_tree_final = self._finality_for_issue(turn)

        cache_bust_marker = turn.cache_bust_marker
        cache_bust_target_for_credit = turn.cache_bust_target
        if self._cache_bust_target in WARMUP_ISOLATION_TARGETS:
            cache_bust_target_for_credit = self._cache_bust_target
            cache_bust_marker = (
                WARMUP_ISOLATION_MARKER if self._phase == CreditPhase.WARMUP else None
            )

        credit = Credit(
            id=credit_index,
            phase=self._phase,
            phase_index=self._phase_index,
            profiling_index=self._profiling_index,
            phase_name=self._phase_name,
            phase_kind=self._phase_kind,
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
            counts_toward_phase_target=turn.counts_toward_phase_target,
            has_forks=turn.has_forks,
            is_parent_final=is_parent_final,
            is_tree_final=is_tree_final,
            no_request=turn.no_request,
            branch_mode=turn.branch_mode,
            cache_bust_marker=cache_bust_marker,
            cache_bust_target=cache_bust_target_for_credit,
            max_tokens_override=turn.max_tokens_override,
            trace_id=turn.trace_id,
            node_ordinal=turn.node_ordinal,
            first_token_event=turn.first_token_event,
        )

        await self._credit_router.send_credit(credit=credit)
        replay_gate = getattr(self, "replay_gate", None)
        if replay_gate is not None:
            await replay_gate.observe_issued(credit)
        if is_final_credit:
            self._progress.freeze_sent_counts()
            self._progress.all_credits_sent_event.set()

        return not is_final_credit

    async def issue_graph_credit(self, turn: TurnToSend) -> bool:
        """Issue an agent-graph credit, BYPASSING the linear session-slot lifecycle.

        Graph traces are a fan-out DAG: the executor fires nodes in dataflow-
        readiness order, not the linear ``turn_index==0`` -> ``num_turns-1``
        sequence the session-slot acquire/release arithmetic assumes. Engaging
        ``acquire_session_slot`` (turn0) here would either fail to acquire (the
        first-fired node is rarely ordinal 0) or leak the slot (the terminal
        node's ordinal rarely equals ``num_turns-1``), deadlocking the phase
        once every slot leaks. Trace-admission concurrency is owned by the
        ``AgentGraphReplayStrategy`` instead.

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
        # ``AgentGraphReplayStrategy`` owns completion (a trace is DONE when its
        # executor drains), so the linear ``SessionCountStopCondition``
        # (``--num-conversations``) must NOT truncate a fan-out DAG mid-trace --
        # it opts out of DAG-child gating for exactly this reason. Cancellation,
        # duration, and ``--request-count`` still apply.
        if not self._stop_checker.can_send_dag_child_turn():
            return False
        acquired = await self._concurrency_manager.acquire_prefill_slot(
            self._phase_key, self._stop_checker.can_send_dag_child_turn
        )
        if not acquired:
            return False
        await self._issue_credit_internal(turn)
        return True

    def _stable_graph_url_index(
        self, affinity_key: str, strategy: URLSelectionStrategyProtocol
    ) -> int:
        """Deterministically map a graph template onto a URL index.

        The mapping must be a pure function of TEMPLATE identity, NOT of mint
        order: warmup priming and profiling replay run in SEPARATE per-phase
        issuers, and every instance/recycle of one template must land on the
        backend that primed its KV -- so the key (the nonce-stripped template
        id) hashes deterministically onto the URL list. Distribution across a
        corpus's templates is statistically uniform under the hash.

        Args:
            affinity_key: The nonce-stripped template id to map.
            strategy: The configured URL selection strategy (the caller gates on
                a configured strategy, so this is never None here).

        Raises:
            ValueError: If the strategy exposes an empty ``urls``. The built-in
                ``RoundRobinURLSampler`` rejects that at construction, but the
                ``url_selection_strategy`` plugin category is extensible, so a
                third-party strategy could reach here; there is no valid
                fallback (``next_url_index`` divides by ``len(urls)``).
        """
        urls = strategy.urls
        if not urls:
            raise ValueError(
                f"URL selection strategy {type(strategy).__name__} exposes an "
                "empty 'urls'; cannot map graph template "
                f"'{affinity_key}' onto a URL index."
            )
        digest = hashlib.sha256(affinity_key.encode()).digest()
        return int.from_bytes(digest[:8], "big") % len(urls)

    async def end_graph_trace(self, trace_id: str) -> None:
        """Close a graph instance's sticky lifecycle (router session only).

        Called ONCE per instance by ``AgentGraphReplayStrategy`` at the
        adapter-reap points (all in-flight dispatches for the instance
        drained, or phase teardown for retained adapters). Forwards to the
        router, which closes the instance's sticky session and notifies the
        sticky worker (``GraphTraceEnd``). URL affinity is deliberately NOT
        evicted: it keys on the nonce-stripped template, and a recycle of the
        template must land on the backend that primed its KV. Idempotent end
        to end.
        """
        await self._credit_router.end_graph_trace(trace_id)

    def mark_graph_sending_complete(self) -> None:
        """Signal that a graph phase will issue no further credits.

        ``AgentGraphReplayStrategy`` owns completion: a graph credit NEVER trips
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

        Used by ``AgentGraphReplayStrategy`` only in the degenerate
        no-credits-issued case (an empty trace set), where no credit return
        will ever fire the event via the callback handler.
        """
        self._progress.all_credits_returned_event.set()

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
        gate check, blocking prefill-slot acquisition, then
        ``_issue_credit_internal``. Children skip session-slot
        acquisition because they inherit the parent's slot.

        Prefill saturation is backpressure, not a reason to discard a child.
        This matters for fan-out: with a prefill limit of one, a non-blocking
        attempt would send the first sibling and permanently truncate every
        other sibling spawned in the same gather.
        """
        gate = getattr(self, "replay_gate", ReplayIssueGate(None))
        return await gate.submit(
            turn,
            lambda: self._dispatch_child_turn_ready(turn),
            child_refusal_cleanup=True,
        )

    async def _dispatch_child_turn_ready(self, turn: TurnToSend) -> bool:
        """Dispatch a child after its recorded predecessor frontier completes."""
        if self._issuing_stopped:
            return False
        can_proceed_fn = self._stop_checker.can_send_child_turn
        if not can_proceed_fn():
            return False
        # Children inherit the parent's session slot; wait for prefill
        # capacity so temporary saturation does not delete sibling branches.
        if not await self._concurrency_manager.acquire_prefill_slot(
            self._phase_key, can_proceed_fn
        ):
            return False
        if turn.counts_toward_phase_target:
            turn = _struct_replace(turn, counts_toward_phase_target=False)
        await self._issue_credit_internal(turn)
        return True

    async def dispatch_join_turn(self, pending: PendingBranchJoin) -> bool:
        """Dispatch a parent's gated turn after all its children complete.

        The parent already holds a session slot (acquired at turn_index=0);
        the gated turn has turn_index > 0, so ``issue_credit``'s session-slot
        acquisition is naturally skipped. Only a prefill slot is acquired
        here — via the blocking ``acquire_prefill_slot`` path inside
        ``issue_credit``, matching ``dispatch_child_turn``. Prefill
        saturation is backpressure, not a reason to permanently drop the
        join (the old non-replay ``try_issue_credit`` path treated ``None``
        as failure and suppressed the join with no retry).

        Cache-bust propagation:
            The TurnToSend constructed here re-applies the parent's
            ``cache_bust_marker`` / ``cache_bust_target`` captured on the
            ``PendingBranchJoin`` at suspend time. Without this, turn k+1
            (the join turn) would dispatch with no marker while turns 0..k
            carried one, breaking per-session cache-bust uniqueness for
            multi-turn parents under DAG joins.

        Stop-condition interaction: when the stop check refuses, the credit is
        NOT issued and the orchestrator increments ``BranchStats.joins_suppressed``.

        Returns:
            True IFF the credit was actually issued. Unlike ``issue_credit``
            (whose False conflates "refused, not issued" with "issued, was the
            phase's final credit"), this inlines the issuance so an issued-but-
            final join is reported as resumed, not suppressed -- mirroring
            ``dispatch_child_turn``.
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
            has_branches=pending.parent_has_branches_on_gated_turn,
            no_request=pending.parent_no_request_on_gated_turn,
            branch_mode=pending.parent_branch_mode,
            cache_bust_marker=pending.parent_cache_bust_marker,
            cache_bust_target=pending.parent_cache_bust_target,
        )
        gate = getattr(self, "replay_gate", ReplayIssueGate(None))
        return await gate.submit(turn, lambda: self._dispatch_join_turn_ready(turn))

    async def _dispatch_join_turn_ready(self, turn: TurnToSend) -> bool:
        """Issue a parent's gated (join) turn once its frontier is complete.

        Returns True IFF the credit was actually issued. A join turn is a parent
        continuation (``turn_index > 0``): it inherits the root's session slot
        (no session-slot acquisition) and only needs a prefill slot. The
        final-credit bookkeeping (freeze counts + done event) still runs inside
        ``_issue_credit_internal``; we simply do not let its "can-send-more"
        False leak out as a spurious suppression.
        """
        if self._issuing_stopped:
            return False
        # Nested (agent_depth > 0) join is reactive DAG work that must progress
        # past the root-sampler-done signal; a top-level join is a normal
        # continuation. Mirrors _issue_credit_ready's check selection.
        can_proceed_fn = (
            self._stop_checker.can_send_child_turn
            if turn.agent_depth > 0
            else self._stop_checker.can_send_any_turn
        )
        if not can_proceed_fn():
            return False
        if not await self._concurrency_manager.acquire_prefill_slot(
            self._phase_key, can_proceed_fn
        ):
            return False
        await self._issue_credit_internal(turn)
        return True

    async def abort_session(self, x_correlation_id: str) -> None:
        """Abort an in-flight session (FORK/SPAWN parent or orphan).

        Currently a no-op: the credit-return slot-release path covers every
        reachable case under the current orchestrator. This method exists so
        the orchestrator's ``hasattr(self._issuer, "abort_session")`` guard
        resolves; under ``AIPERF_DAG_FAIL_FAST=true`` the orchestrator calls
        this when a child errors and the parent / orphan siblings must be torn
        down.

        If implemented in the future, the contract is:

        - Cancel any in-flight credit for ``x_correlation_id`` (via the router).
        - Release the session slot (do NOT release sibling sessions' slots).
        - Be idempotent -- orchestrator calls it once per session, but errors
          mid-tear-down may re-enter.
        - Be exception-safe -- orchestrator does not retry.
        """
        return None
