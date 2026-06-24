# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAG branch orchestrator.

Intercepts parent-turn completion, dispatches child sessions (FORK or SPAWN
mode), tracks join completion, and releases per-parent state when the DAG
drains. See ``docs/benchmark-modes/dag.md`` for user-facing semantics.

Delayed joins (K>1)
-------------------
A parent may spawn children on turn T whose join fires on turn T+K for any
K>=1. The parent progresses turns T+1..T+K-1 normally while children execute
in parallel, and only suspends on the turn that immediately precedes the
gated turn. This matches the conflux author model and is validated at load
time by ``validate_for_orchestrator_v1``.

Dispatch offsets (SPAWN mode)
-----------------------------
A SPAWN child whose recorded first request starts after the branch spawn
(child turn-0 ``timestamp_ms`` past the branch ``start_timestamp_ms``)
dispatches via a delayed background task at that offset instead of firing
immediately, reproducing the recorded in-subagent timing (e.g. a weka
overflow stream whose first request landed minutes after the subagent
spawned). Join gates and descendant counts are registered before the sleep,
so gated parents wait for sleeping children; ``cleanup()`` cancels pending
sleepers. Datasets without timing (``--ignore-trace-delays``) carry None
timestamps and keep the immediate-dispatch behavior.

Sticky-routing locality (FORK mode)
-----------------------------------
FORK-mode children are routed to the parent's worker via the sticky router
(keyed by ``parent_correlation_id``). Because the parent's ``UserSession``
lives in the same worker's local memory, the child's
``UserSessionManager.create_and_store`` can clone ``turn_list`` directly
from the parent session with no cross-process plumbing. The orchestrator
bumps the parent's sticky refcount via
``StickyCreditRouter.register_child_routing`` before dispatching FORK-mode
children and releases it via ``release_child_routing`` when each child
terminates. SPAWN-mode children do not pin to the parent's worker and
therefore do not touch sticky refcounts.

Credit return flow
------------------
``CreditCallbackHandler.on_credit_return`` processing order::

    1. Atomic counting (progress.increment_returned)
    2. Track prefill release if TTFT never arrived
    3. Release concurrency slots (skipped for children: agent_depth > 0)
    4. DAG child-completion hook (on_child_leaf_reached / on_child_errored
       for final-turn child credits only)
    5. Signal all_credits_returned_event (deferred if DAG has pending work)
    6. intercept(credit): spawn branches declared on the completed turn and
       return True IFF the parent's NEXT turn is a gated turn with
       unsatisfied prereqs.
    7. Strategy dispatch if not intercepted (child bypass uses
       ``agent_depth > 0``)

Stop-condition interaction
--------------------------
Three coordinated guards achieve zero-overshoot, zero-deadlock around DAG
work that outlives the phase's root-sampling completion::

1. **Callback-handler child bypass** (step 7): credit returns carrying
   ``agent_depth > 0`` always reach ``handle_credit_return`` even after
   ``can_send_any_turn`` flips False. Without this, child final returns
   would be silently dropped, leaving parents stuck in ``_active_joins``.

2. **Completion-event deferral** (step 5): when a root's final return is
   about to trigger child dispatch (``_credit_will_dispatch_children``) or
   when the orchestrator still has ``has_pending_branch_work()``, the
   all-credits-returned event is held until the DAG drains.

3. **Session-slot bypass for children** (``CreditIssuer.issue_credit``):
   children with ``agent_depth > 0`` never acquire a session slot, so the
   callback handler's matching release is gated on ``agent_depth == 0``.
   The two sides are symmetric — see ``credit/issuer.py`` and
   ``credit/callback_handler.py``.

Cleanup
-------
``PhaseRunner`` calls ``cleanup()`` at every phase-exit path. Late credit
returns after cleanup find ``_cleaning_up=True`` and short-circuit without
dispatching new work. ``cleanup()`` logs final ``BranchStats`` and warns
about any leaked per-parent state — normally empty, non-empty indicates a
DAG that failed to drain (worker crash, protocol mismatch, bug).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

from aiperf.common.enums import (
    CacheBustTarget,
    ConversationBranchMode,
    CreditPhase,
    PrerequisiteKind,
)
from aiperf.common.environment import Environment
from aiperf.common.models.branch_stats import BranchStats

__all__ = [
    "BranchOrchestrator",
    "BranchStats",
    "ChildJoinEntry",
    "PendingBranchJoin",
    "PrereqState",
]

logger = logging.getLogger(__name__)


def _as_timestamp_ms(value) -> float | None:
    """Coerce a metadata timestamp to float ms; anything non-numeric is None."""
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None


def _turn0_timestamp_ms(meta) -> float | None:
    """First-turn timestamp of a conversation metadata, or None."""
    turns = getattr(meta, "turns", None)
    if not isinstance(turns, list | tuple) or not turns:
        return None
    return _as_timestamp_ms(getattr(turns[0], "timestamp_ms", None))


@dataclass
class PrereqState:
    """Per-prereq gate state (Phase 3).

    Tracks the number of expected child completions (``expected``) and the
    set of child correlation ids that have already reported (``completed``).
    The set form gives idempotent double-delivery protection; the counter
    form lets multiple spawn points contribute to the same ``prereq_key``
    (fan-in) without requiring the orchestrator to know every child
    correlation id at registration time.

    ``registered`` is False until the spawning turn actually fires and
    ``expected`` has been incremented for at least one child. Fan-in
    requires the gate to be seeded with every declared prereq_key at
    pending-join-creation time so a prereq that fires-and-completes before
    the sibling prereq registers doesn't prematurely satisfy the gate.
    """

    expected: int = 0
    completed: set[str] = field(default_factory=set)
    registered: bool = False

    @property
    def is_done(self) -> bool:
        """True once the prereq has been registered and every expected
        completion has landed. Unregistered prereqs are never done — even
        with expected==0 — because some future spawning turn will increment
        ``expected``.
        """
        return self.registered and len(self.completed) >= self.expected


@dataclass
class PendingBranchJoin:
    """Join state for a parent session awaiting outstanding children.

    Holds everything the credit issuer needs to build the parent's gated
    TurnToSend without re-entering the conversation source, so the orchestrator
    stays the single source of truth for join bookkeeping.

    Phase 3 uses ``outstanding: dict[prereq_key, PrereqState]`` where each
    ``PrereqState`` carries an ``expected`` counter and a ``completed`` set.
    A single gated turn may have multiple prereq keys (fan-in); all must be
    done for ``is_satisfied`` to be True.
    """

    parent_x_correlation_id: str
    parent_conversation_id: str
    parent_num_turns: int
    parent_agent_depth: int = 0
    parent_parent_correlation_id: str | None = None
    gated_turn_index: int | None = None
    outstanding: dict[str, PrereqState] = field(default_factory=dict)
    parent_branch_mode: ConversationBranchMode = ConversationBranchMode.FORK
    parent_has_forks_on_gated_turn: bool = False
    is_blocked: bool = False
    created_at_ns: int = field(default_factory=time.monotonic_ns)
    # Cache-bust state captured from the credit that suspends the parent so
    # the gated turn dispatched after children join carries the same marker
    # as turns 0..k-1 (otherwise the join turn would silently disable
    # cache-bust for that one turn).
    parent_cache_bust_marker: str | None = None
    parent_cache_bust_target: CacheBustTarget = CacheBustTarget.NONE

    @property
    def is_satisfied(self) -> bool:
        """True when every prereq's expected completions have all arrived."""
        return all(s.is_done for s in self.outstanding.values())

    @property
    def total_outstanding(self) -> int:
        """Total outstanding children across all prereqs (for diagnostics)."""
        return sum(
            max(0, s.expected - len(s.completed)) for s in self.outstanding.values()
        )


@dataclass(slots=True, frozen=True)
class ChildJoinEntry:
    """Tracks which parent pending-join a blocking child belongs to.

    ``prereq_key`` is ``None`` for background children (no gate); they still
    appear in ``_child_to_join`` so ``has_pending_branch_work`` and cleanup
    see them, but satisfying the entry skips gate bookkeeping.
    """

    parent_correlation_id: str
    gated_turn_index: int | None
    prereq_key: str | None


class BranchOrchestrator:
    """Handles DAG branch dispatch (FORK and SPAWN modes).

    See the module docstring for the credit-return flow, stop-condition
    guards, and cleanup semantics.
    """

    def __init__(
        self,
        conversation_source,
        credit_issuer,
        sticky_router=None,
        *,
        benchmark_id: str = "unknown",
        cache_bust_target: CacheBustTarget = CacheBustTarget.NONE,
        session_tree_registry=None,
        cache_bust_ledger=None,
    ) -> None:
        self._cs = conversation_source
        self._issuer = credit_issuer
        self._sticky_router = sticky_router
        self._benchmark_id = benchmark_id
        self._cache_bust_target = cache_bust_target
        # Shared CacheBustLedger (root_correlation_id -> marker). Descendants
        # resolve their tree-root's marker through it so the whole tree shares one
        # prefix-cache domain; None when no ledger is wired (e.g. unit tests with
        # cache-bust disabled).
        self._marker_ledger = cache_bust_ledger
        # Per-tree session-slot ledger (agentic replay only; None otherwise).
        # Every descendant this orchestrator spawns or snapshot-seeds is
        # registered against its tree's root_correlation_id so the tree's
        # session slot is held until the last descendant drains. Acquisition of
        # the slot happens in the credit issuer; the orchestrator only adjusts
        # the per-tree outstanding count.
        self._session_tree_registry = session_tree_registry
        # child x_correlation_id -> its tree's root_correlation_id, so the
        # terminal-completion / rollback paths can decrement the right tree.
        self._child_root: dict[str, str] = {}
        self._child_modes: dict[str, ConversationBranchMode] = {}
        # Two-level pending-join state: a "future" join is registered at
        # spawn time and promoted to "active" once the parent reaches the
        # turn immediately preceding the gated turn. Satisfying a join that
        # is still future-only pops it silently (no dispatch); satisfying
        # an active join dispatches the gated turn.
        self._future_joins: dict[str, dict[int, PendingBranchJoin]] = {}
        self._active_joins: dict[str, PendingBranchJoin] = {}
        self._child_to_join: dict[str, list[ChildJoinEntry]] = {}
        self._parent_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._descendant_counts: dict[str, int] = {}
        # Phase 2b: records (conv_id, branch_id) for branches that were
        # pre-dispatched via dispatch_pre_session_branches. The per-turn
        # spawn path in intercept skips branches that appear here so the
        # children are not dispatched a second time when the parent's
        # turn 0 credit returns.
        self._pre_dispatched_branches: set[tuple[str, str]] = set()
        self._fail_fast = Environment.DAG.FAIL_FAST
        self._cleaning_up: bool = False
        # SPAWN children whose recorded first request starts after the branch
        # spawn dispatch via delayed background tasks (see
        # _start_delayed_first_turn). cleanup() cancels pending sleepers.
        self._delayed_dispatch_tasks: set[asyncio.Task] = set()
        # Drain observer: sync callback fired after state mutations that may
        # drain has_pending_branch_work() to False. Wired by
        # CreditCallbackHandler.set_branch_orchestrator to re-evaluate the
        # deferred all-credits-returned signal when the last drain step
        # lands between concurrent on_credit_return callbacks (no further
        # return arrives to re-trigger the check). Without this hook the
        # phase runner's pre-wait short-circuit and drain-timeout backstop
        # are the only safety nets — both work, but the short-circuit only
        # catches the race when the runner is late, and the backstop costs
        # a drain timeout's worth of wall clock per occurrence. Closing the
        # race at the source eliminates both costs.
        self._drain_observer = None
        self.stats = BranchStats()
        # Pre-built index: (conv_id, spawning_turn_idx) -> list of
        # (branch_id, gated_turn_idx, prereq_key). Built once at init from
        # each turn's SPAWN_JOIN prerequisites; the mapping resolves a
        # declared branch back to the turn on which it was authored so
        # spawn-time code can register the future join directly.
        self._prereq_index: dict[tuple[str, int], list[tuple[str, int, str]]] = {}
        # Phase 3 fan-in seed: (conv_id, gated_turn_idx) -> set of all
        # prereq_keys that the gated turn needs. When a pending join is
        # created we pre-seed ``outstanding`` with an unregistered
        # PrereqState for every expected prereq so fan-in doesn't fire
        # early when one branch completes before another branch's spawning
        # turn has been reached.
        self._gated_turn_prereq_keys: dict[tuple[str, int], set[str]] = {}
        # (conv_id, gated_turn_idx, prereq_key) -> spawning turn index.
        # Snapshot seeding consults this to tell prereqs whose spawning turn
        # already fired before t* (their children either appear live in the
        # snapshot or completed entirely pre-t*) apart from prereqs whose
        # spawning turn will fire during replay.
        self._prereq_spawning_turn: dict[tuple[str, int, str], int] = {}
        # Defense-in-depth duplicate detection against future loaders that
        # bypass ``validate_for_orchestrator_v1``. A given
        # ``(branch_id, gated_turn_idx)`` tuple must not appear twice — that
        # would mean two identical prereq entries were authored.
        self._build_prereq_index()

    def _build_prereq_index(self) -> None:
        dataset_meta = getattr(self._cs, "dataset_metadata", None)
        conversations = getattr(dataset_meta, "conversations", None) or []
        for conv in conversations:
            # Resolve each SPAWN_JOIN prereq to the spawning turn that
            # declared the referenced branch_id.
            branch_declaration_turn: dict[str, int] = {}
            for turn_idx, turn in enumerate(conv.turns):
                for b_id in turn.branch_ids or []:
                    branch_declaration_turn.setdefault(b_id, turn_idx)
            for gated_idx, turn in enumerate(conv.turns):
                for prereq in turn.prerequisites:
                    if prereq.kind != PrerequisiteKind.SPAWN_JOIN:
                        continue
                    if prereq.branch_id is None:
                        continue
                    spawning_idx = branch_declaration_turn.get(prereq.branch_id)
                    if spawning_idx is None:
                        continue
                    prereq_key = f"SPAWN_JOIN:{prereq.branch_id}"
                    key = (conv.conversation_id, spawning_idx)
                    bucket = self._prereq_index.setdefault(key, [])
                    entry = (prereq.branch_id, gated_idx, prereq_key)
                    bucket.append(entry)
                    # Phase 3 fan-in seed: track every prereq_key feeding
                    # this (conv_id, gated_idx) so gate creation knows the
                    # full set of prereqs to wait for.
                    self._gated_turn_prereq_keys.setdefault(
                        (conv.conversation_id, gated_idx), set()
                    ).add(prereq_key)
                    self._prereq_spawning_turn[
                        (conv.conversation_id, gated_idx, prereq_key)
                    ] = spawning_idx

    def get_branch_ids(self, credit) -> list[str]:
        """Look up the completed turn's ``branch_ids`` from metadata.

        Public so the credit-callback handler can probe whether a returning
        credit will trigger DAG dispatch (used to defer phase-completion
        signalling).
        """
        meta = self._cs.get_metadata(credit.conversation_id)
        if credit.turn_index >= len(meta.turns):
            return []
        return list(meta.turns[credit.turn_index].branch_ids)

    def _marker_for_root(self, root_correlation_id: str | None) -> str | None:
        """Resolve the tree-root cache-bust marker for a spawned descendant.

        The marker is a property of the trajectory TREE (``root_correlation_id``):
        every descendant — subagents and flat agents at any depth — reuses the
        root's marker instead of minting its own, so the whole tree is one
        prefix-cache domain (a per-child marker would force the server to
        re-prefill any prefix the agents share). The root's marker was minted at
        trajectory setup (``AgenticReplayTiming._mint_marker_for_session``) and
        lives in the shared ledger keyed by ``root_correlation_id``. Returns None
        when cache-bust is disabled, the root has no marker, or no ledger is wired.
        """
        if self._cache_bust_target == CacheBustTarget.NONE or not root_correlation_id:
            return None
        if self._marker_ledger is None:
            return None
        return self._marker_ledger.session_marker.get(root_correlation_id)

    def _mint_child_marker(self, child_conversation_id: str) -> str | None:
        """Mint a marker for a DAG-authored pre-session (turn-0 background) child.

        Used only by ``dispatch_pre_session_branches``, where the spawning root
        session does not exist yet (``parent_correlation_id=None``), so the
        root's marker cannot be resolved from the ledger. Per-turn spawned
        descendants instead use ``_marker_for_root`` to share the tree marker.
        Returns None when cache-bust is disabled (target=NONE).
        """
        from aiperf.timing.strategies.cache_bust import build_cache_bust_marker

        if self._cache_bust_target == CacheBustTarget.NONE:
            return None
        return build_cache_bust_marker(
            self._benchmark_id,
            0,
            0,
            child_conversation_id,
            target=self._cache_bust_target,
        )

    async def dispatch_pre_session_branches(self) -> None:
        """Pre-dispatch background SPAWN children marked dispatch_timing='pre'.

        Called once by ``PhaseRunner.run`` before the strategy starts issuing
        root turn-0 credits. Fires each qualifying child with ``agent_depth=1``
        and ``parent_correlation_id=None`` — no real parent session exists
        yet. The per-turn spawn path (``_spawn_children_and_register_gates``)
        consults ``self._pre_dispatched_branches`` to skip these branches on
        the parent's turn-0 credit return so children are not dispatched
        twice.

        Validator (orchestrator_v1) guarantees the branches reaching this
        path are SPAWN mode, ``is_background=True``, attached to turn 0 of
        a root conversation.
        """
        if self._cleaning_up:
            return
        dataset_meta = getattr(self._cs, "dataset_metadata", None)
        if dataset_meta is None:
            return
        conversations = getattr(dataset_meta, "conversations", None) or []
        for conv in conversations:
            if getattr(conv, "agent_depth", 0) > 0 or not conv.turns:
                continue
            turn0_branch_ids = set(conv.turns[0].branch_ids or [])
            for branch in conv.branches:
                if getattr(branch, "dispatch_timing", "post") != "pre":
                    continue
                # Validator enforces this, but guard defensively so buggy
                # loaders can't silently skip the turn-0 attachment.
                if branch.branch_id not in turn0_branch_ids:
                    continue
                for child_cid in branch.child_conversation_ids:
                    try:
                        child_session = self._cs.start_pre_session_child(
                            child_cid,
                            cache_bust_marker=self._mint_child_marker(child_cid),
                            cache_bust_target=self._cache_bust_target,
                        )
                    except Exception:
                        logger.exception(
                            "start_pre_session_child failed for %s", child_cid
                        )
                        self.stats.children_errored += 1
                        continue
                    issued = await self._issuer.dispatch_first_turn(child_session)
                    if issued:
                        self.stats.children_spawned += 1
                    else:
                        # ``dispatch_first_turn`` -> ``dispatch_child_turn``
                        # only returns False when a stop condition refuses
                        # the child before or during prefill-slot acquisition.
                        # Exceptions are caught above. Tally as truncated,
                        # not errored.
                        self.stats.children_truncated += 1
                self._pre_dispatched_branches.add(
                    (conv.conversation_id, branch.branch_id)
                )

    def _register_tree_descendants(self, root_corr: str | None, n: int) -> None:
        """Add ``n`` descendants to their session tree's outstanding count.

        Mirrors the per-parent ``_descendant_counts`` bump but keyed on the
        tree's root so the tree's session slot is held until the WHOLE tree
        (root + every descendant at any depth) drains. No-op when tree
        accounting is not engaged."""
        if self._session_tree_registry is not None and root_corr is not None and n > 0:
            self._session_tree_registry.register_descendants(root_corr, n)

    def _tree_descendant_done(self, child_corr: str) -> None:
        """Account one descendant terminally finishing against its tree.

        Pops the child's recorded tree root and decrements that tree's
        outstanding count; the registry releases the tree's session slot (and
        recycles the freed lane) once the tree drains. Idempotent: a child with
        no recorded root (already accounted, or never tracked) is a no-op."""
        root_corr = self._child_root.pop(child_corr, None)
        if self._session_tree_registry is not None and root_corr is not None:
            self._session_tree_registry.on_descendant_done(root_corr)

    def _register_fork_routing(
        self, parent_corr: str, mode: ConversationBranchMode
    ) -> None:
        """Bump the parent's sticky routing refcount for a FORK child.

        FORK children sticky-route to the parent's worker; the refcount is
        balanced by ``_handle_child_done``'s ``release_child_routing`` on leaf.
        SPAWN children route freely and register no refcount.
        """
        if mode == ConversationBranchMode.FORK and self._sticky_router is not None:
            self._sticky_router.register_child_routing(parent_corr)

    def seed_snapshot(
        self,
        states,
        *,
        cache_bust_markers: dict[str, str | None] | None = None,
    ) -> None:
        """Seed join bookkeeping from an agentic replay wall-clock snapshot.

        Normal DAG state is discovered by observing a parent turn return and
        spawning children from that event. Snapshot replay starts after that
        event has already happened, so the strategy provides already-live
        child states and any gated parent state here.
        """
        if self._cleaning_up:
            return

        states_by_corr = {state.x_correlation_id: state for state in states}
        children_by_parent: dict[str, list] = defaultdict(list)
        for state in states:
            if state.agent_depth > 0 and state.parent_correlation_id is not None:
                children_by_parent[state.parent_correlation_id].append(state)

        for parent_corr, child_states in children_by_parent.items():
            parent_state = states_by_corr.get(parent_corr)
            parent_meta = None
            if parent_state is not None:
                parent_meta = self._cs.get_metadata(parent_state.conversation_id)

            tracked_children = 0
            for child_state in child_states:
                self._child_modes[child_state.x_correlation_id] = (
                    child_state.branch_mode
                )
                self._register_fork_routing(parent_corr, child_state.branch_mode)
                entries: list[ChildJoinEntry] = []
                if (
                    parent_state is not None
                    and parent_meta is not None
                    and child_state.join_target_turn_index is not None
                    and child_state.branch_id is not None
                ):
                    prereq_key = f"SPAWN_JOIN:{child_state.branch_id}"
                    pending = self._ensure_seeded_join(
                        parent_state=parent_state,
                        parent_meta=parent_meta,
                        gated_idx=child_state.join_target_turn_index,
                        cache_bust_marker=(cache_bust_markers or {}).get(
                            parent_state.root_correlation_id
                            or parent_state.x_correlation_id
                        ),
                    )
                    prereq_state = pending.outstanding.setdefault(
                        prereq_key, PrereqState()
                    )
                    prereq_state.expected += 1
                    prereq_state.registered = True
                    entries.append(
                        ChildJoinEntry(
                            parent_correlation_id=parent_corr,
                            gated_turn_index=child_state.join_target_turn_index,
                            prereq_key=prereq_key,
                        )
                    )
                else:
                    entries.append(
                        ChildJoinEntry(
                            parent_correlation_id=parent_corr,
                            gated_turn_index=None,
                            prereq_key=None,
                        )
                    )

                self._child_to_join[child_state.x_correlation_id] = entries
                self._child_root[child_state.x_correlation_id] = (
                    child_state.root_correlation_id or parent_corr
                )
                tracked_children += 1

            if tracked_children:
                self._descendant_counts[parent_corr] = (
                    self._descendant_counts.get(parent_corr, 0) + tracked_children
                )
                # All seeded children of one snapshot lane share the lane's tree
                # root (the snapshot's synthetic parent_corr), which is also the
                # id the lane credit opened the tree under.
                self._register_tree_descendants(parent_corr, tracked_children)
                self.stats.children_spawned += tracked_children

    def _ensure_seeded_join(
        self,
        *,
        parent_state,
        parent_meta,
        gated_idx: int,
        cache_bust_marker: str | None,
    ) -> PendingBranchJoin:
        parent_corr = parent_state.x_correlation_id
        active = self._active_joins.get(parent_corr)
        if active is not None and active.gated_turn_index == gated_idx:
            return active
        future = self._future_joins.get(parent_corr, {}).get(gated_idx)
        if future is not None:
            return future

        has_forks = False
        if 0 <= gated_idx < len(parent_meta.turns):
            has_forks = bool(getattr(parent_meta.turns[gated_idx], "has_forks", False))

        pending = PendingBranchJoin(
            parent_x_correlation_id=parent_corr,
            parent_conversation_id=parent_state.conversation_id,
            parent_num_turns=len(parent_meta.turns),
            parent_agent_depth=parent_state.agent_depth,
            parent_parent_correlation_id=parent_state.parent_correlation_id,
            gated_turn_index=gated_idx,
            parent_branch_mode=parent_state.branch_mode,
            parent_has_forks_on_gated_turn=has_forks,
            parent_cache_bust_marker=cache_bust_marker,
            parent_cache_bust_target=self._cache_bust_target,
        )
        for prereq_key in self._gated_turn_prereq_keys.get(
            (parent_state.conversation_id, gated_idx), set()
        ):
            state = PrereqState()
            spawning_idx = self._prereq_spawning_turn.get(
                (parent_state.conversation_id, gated_idx, prereq_key)
            )
            if spawning_idx is not None and spawning_idx < parent_state.next_turn_index:
                # The spawning turn fired before t* and will never replay.
                # Children still alive at t* re-register with expected
                # counts during this same seeding pass; a branch with no
                # live children completed entirely pre-t* and must seed as
                # satisfied, or the gate is permanently unsatisfiable and
                # the parent lane silently wedges for the whole phase.
                state.registered = True
            pending.outstanding[prereq_key] = state

        if (
            parent_state.waiting_on_children
            and parent_state.join_target_turn_index == gated_idx
        ):
            pending.is_blocked = True
            self._active_joins[parent_corr] = pending
            self.stats.parents_suspended += 1
        else:
            self._future_joins.setdefault(parent_corr, {})[gated_idx] = pending
        return pending

    async def intercept(self, credit) -> bool:
        """Intercept the credit-return path.

        Spawn any branches declared on the completed turn. Independently,
        check whether the parent's NEXT turn is a gated turn with
        unsatisfied prereqs; return True only in that case. Returning True
        suppresses the strategy's default next-turn dispatch.

        FORK-mode children are routed to the parent's worker via sticky routing
        (``parent_correlation_id`` keying); the worker seeds each child's
        ``UserSession.turn_list`` from the parent's local session.
        SPAWN-mode children route freely (no sticky pin).
        """
        if self._cleaning_up:
            return False

        # Warmup is one-shot per trajectory; strategy refuses to advance
        # child continuation turns. Spawning here leaks _descendant_counts
        # (children never reach is_final_turn) and wedges
        # all_credits_returned_event. DAG dispatch runs in PROFILING.
        if credit.phase == CreditPhase.WARMUP:
            return False

        # Child path: handled by the callback handler directly (child leaf /
        # error hooks). Child continuation turns dispatch via the strategy's
        # normal path and do not enter intercept with agent_depth > 0.
        if credit.agent_depth > 0:
            return False

        parent_corr = credit.x_correlation_id

        async with self._parent_locks[parent_corr]:
            branch_ids = self.get_branch_ids(credit)
            if branch_ids:
                await self._spawn_children_and_register_gates(credit, branch_ids)
            return self._maybe_suspend_parent(credit)

    async def _spawn_children_and_register_gates(
        self, credit, branch_ids: list[str]
    ) -> None:
        """Resolve branches, start children, and register future joins.

        Layout mirrors conflux's two-phase dispatch (register gates before
        dispatching) but retains weka's sticky-router and per-child
        rollback semantics for FORK-mode children.
        """
        parent_corr = credit.x_correlation_id
        parent_depth = credit.agent_depth
        parent_meta = self._cs.get_metadata(credit.conversation_id)
        branches_by_id = {b.branch_id: b for b in parent_meta.branches}

        # Index entries for (conversation_id, spawning_turn_idx). List is
        # empty if this turn's branches are all background / ungated. Phase
        # 3 multi-consumer: a branch may appear under multiple gate entries
        # — each (gated_idx, prereq_key) forms its own independent gate.
        prereq_entries = self._prereq_index.get(
            (credit.conversation_id, credit.turn_index), []
        )
        gate_for_branch: dict[str, list[tuple[int, str]]] = {}
        for branch_id, gated_idx, prereq_key in prereq_entries:
            gate_for_branch.setdefault(branch_id, []).append((gated_idx, prereq_key))

        all_children: list = []
        per_child_gates: dict[str, list[tuple[int, str]]] = {}
        per_child_branch_mode: dict[str, ConversationBranchMode] = {}
        dispatch_offset_by_corr: dict[str, float] = {}
        # Track gates we intended to create for a branch even when every
        # start_branch_child fails under that branch. We still must surface
        # a zero-outstanding gate so the parent doesn't hang.
        expected_gates: set[tuple[int, str]] = set()

        for b_id in branch_ids:
            branch = branches_by_id.get(b_id)
            if branch is None:
                continue
            # Phase 2b: branches already fired via dispatch_pre_session_branches
            # are recorded in _pre_dispatched_branches; skip them on the
            # parent's turn-0 return to avoid double-dispatch.
            if (credit.conversation_id, b_id) in self._pre_dispatched_branches:
                continue
            branch_gates = gate_for_branch.get(branch.branch_id, [])
            # Background branches never gate the parent even if the dataset
            # authored a spawning turn for them (the validator would have
            # rejected this, but defensive).
            if branch.is_background:
                branch_gates = []

            for gate in branch_gates:
                expected_gates.add(gate)

            # SPAWN children carry recorded dispatch offsets (child turn-0
            # timestamp relative to the branch spawn). FORK children continue
            # the parent context and always dispatch immediately.
            branch_start_ms = (
                self._branch_start_timestamp_ms(branch)
                if branch.mode == ConversationBranchMode.SPAWN
                else None
            )

            for child_conv_id in branch.child_conversation_ids:
                try:
                    child = self._cs.start_branch_child(
                        parent_correlation_id=parent_corr,
                        child_conversation_id=child_conv_id,
                        agent_depth=parent_depth + 1,
                        root_correlation_id=credit.effective_root_correlation_id,
                        branch_mode=branch.mode,
                        cache_bust_marker=self._marker_for_root(
                            credit.effective_root_correlation_id
                        ),
                        cache_bust_target=self._cache_bust_target,
                    )
                except Exception:
                    logger.exception("start_branch_child failed for %s", child_conv_id)
                    self.stats.children_errored += 1
                    continue

                child_corr = child.x_correlation_id
                self._child_root[child_corr] = credit.effective_root_correlation_id
                self._child_modes[child_corr] = branch.mode
                per_child_branch_mode[child_corr] = branch.mode
                per_child_gates[child_corr] = list(branch_gates)
                dispatch_offset_by_corr[child_corr] = self._child_dispatch_offset_ms(
                    branch_start_ms, child
                )
                all_children.append(child)

                # Only FORK-mode children sticky-route to the parent's worker.
                self._register_fork_routing(parent_corr, branch.mode)
                self.stats.children_spawned += 1

                # Register in _child_to_join (one entry per gate this child
                # contributes to) and bump each gate's expected counter.
                entries: list[ChildJoinEntry] = []
                if branch_gates:
                    for gated_idx, prereq_key in branch_gates:
                        pending = self._ensure_future_join(
                            credit, parent_meta, parent_corr, gated_idx
                        )
                        state = pending.outstanding.setdefault(
                            prereq_key, PrereqState()
                        )
                        state.expected += 1
                        state.registered = True
                        entries.append(
                            ChildJoinEntry(
                                parent_correlation_id=parent_corr,
                                gated_turn_index=gated_idx,
                                prereq_key=prereq_key,
                            )
                        )
                else:
                    # Background / no gate: still track for descendant
                    # accounting so the parent's root-slot release waits.
                    entries.append(
                        ChildJoinEntry(
                            parent_correlation_id=parent_corr,
                            gated_turn_index=None,
                            prereq_key=None,
                        )
                    )
                self._child_to_join[child_corr] = entries

        # Descendant-count accounting: track every successfully-started
        # child. The parent's own terminal-turn return is NOT reserved here
        # because ``_child_to_join`` already keeps ``has_pending_branch_work``
        # True until each child reports done; reserving an extra +1 with no
        # decrement path would leak ``_descendant_counts[parent] == 1``
        # forever (see test_background_spawn_child_outlives_parent).
        if all_children:
            self._descendant_counts.setdefault(parent_corr, 0)
            self._descendant_counts[parent_corr] += len(all_children)
            # Hold the tree's session slot until every one of these descendants
            # drains. The spawning parent in this path is the depth-0 root, so
            # the tree root is its effective_root_correlation_id.
            self._register_tree_descendants(
                credit.effective_root_correlation_id, len(all_children)
            )

        # If any expected gate had zero children actually register, still
        # create a future-join entry with an empty outstanding dict keyed
        # by the prereq so the drain-logic below sees it and fires.
        for gated_idx, prereq_key in expected_gates:
            pending = self._ensure_future_join(
                credit, parent_meta, parent_corr, gated_idx
            )
            state = pending.outstanding.setdefault(prereq_key, PrereqState())
            # The branch was declared even if zero children landed; mark
            # registered so the gate considers this prereq satisfied (0
            # expected, 0 completed, registered=True -> is_done).
            state.registered = True

        # Dispatch children. A SPAWN child whose recorded first request
        # starts after the branch spawn dispatches via a delayed background
        # task at that offset; everything else dispatches immediately.
        # try_issue_credit returning False/None rolls back per-child
        # bookkeeping (shared between both paths).
        immediate_children: list = []
        for child in all_children:
            offset_ms = dispatch_offset_by_corr.get(child.x_correlation_id, 0.0)
            if offset_ms > 0.0:
                self._start_delayed_first_turn(child, offset_ms, parent_corr)
            else:
                immediate_children.append(child)

        results = await asyncio.gather(
            *(self._dispatch_first_turn(child) for child in immediate_children),
            return_exceptions=True,
        )
        for child, result in zip(immediate_children, results, strict=True):
            if result is not True:
                self._rollback_failed_first_turn(child, result, parent_corr)
        await self._finalize_failed_dispatches(parent_corr)

    def _rollback_failed_first_turn(self, child, result, parent_corr: str) -> None:
        """Undo per-child bookkeeping for a turn-0 dispatch that didn't land.

        Shared by the immediate gather path and the delayed-dispatch tasks so
        both classify results identically (BaseException -> errored, False ->
        truncated, None -> silent no-op).
        """
        child_corr = child.x_correlation_id
        child_mode = self._child_modes.pop(child_corr, None)
        entries = self._child_to_join.pop(child_corr, [])
        for entry in entries:
            if entry.prereq_key is None:
                continue
            pending = self._get_join(
                parent_corr,
                entry.gated_turn_index,  # type: ignore[arg-type]
            )
            if pending is None:
                continue
            state = pending.outstanding.get(entry.prereq_key)
            if state is not None and state.expected > 0:
                # Rollback decrements ``expected`` without touching
                # ``completed``. The child never landed so it cannot
                # have reported, and discard-on-completed would be a
                # no-op. Clamp at >= len(completed) so an already-
                # delivered completion (unlikely but possible under
                # aggressive reordering) doesn't revert is_done.
                state.expected = max(len(state.completed), state.expected - 1)
        if (
            child_mode == ConversationBranchMode.FORK
            and self._sticky_router is not None
        ):
            self._sticky_router.release_child_routing(parent_corr)
        if parent_corr in self._descendant_counts:
            self._descendant_counts[parent_corr] -= 1
        # The child was counted into its tree at spawn time (register_descendants
        # over len(all_children)); a turn-0 dispatch that never landed must
        # decrement it too, or the tree's slot would never drain.
        self._tree_descendant_done(child_corr)
        # Three-way classification of non-True dispatch results:
        #   * BaseException -> genuine error (mirror commit 05d02720b
        #     which fixed the analogous bug in
        #     ``dispatch_pre_session_branches``).
        #   * False -> ``dispatch_child_turn`` stop-condition refusal;
        #     not an error.
        #   * None -> issuer suppressed silently; observable no-op.
        if isinstance(result, BaseException):
            logger.error(
                "dispatch_first_turn failed for child %s",
                child_corr,
                exc_info=result,
            )
            self.stats.children_errored += 1
        elif result is False:
            self.stats.children_truncated += 1
        elif result is None:
            pass
        else:
            logger.warning(
                "dispatch_first_turn returned unexpected value %r for child %s",
                result,
                child_corr,
            )
            self.stats.children_errored += 1
        self.stats.children_spawned -= 1

    async def _finalize_failed_dispatches(self, parent_corr: str) -> None:
        """Drain end-game after one or more turn-0 dispatch rollbacks.

        Pops vestigial gates, dispatches a satisfied gate that the parent is
        already suspended on, releases a fully-drained parent, and notifies the
        drain observer (no credit return follows a rollback to do it). Runs
        after the immediate gather settles and after each delayed dispatch
        settles; a no-op when nothing rolled back.
        """
        # If no children at all landed (all failed), pop gates that are now
        # zero-outstanding so the parent is not left suspended on a join
        # that can never fire via the child-leaf decrement path.
        gates_for_parent = self._future_joins.get(parent_corr, {})
        for gated_idx, pending in list(gates_for_parent.items()):
            # A gate may be vestigial (created this call and immediately
            # satisfied) if every child under every prereq rolled back.
            if pending.is_satisfied:
                # Pop silently regardless of position. With the gate gone,
                # _maybe_suspend_parent returns False and the strategy's
                # normal continuation dispatches the (now un-gated) turn as
                # an ordinary next turn - exactly once. Dispatching the
                # immediate-next gate here as well double-dispatched the
                # same turn: intercept still returned False, so the callback
                # handler fell through to handle_credit_return ->
                # _dispatch_next_turn for the identical turn_index.
                self._pop_future_join(parent_corr, gated_idx)
        # A gate already promoted into _active_joins (the parent suspended on
        # it in a prior intercept) is never in _future_joins, so the scan above
        # cannot see it. A rollback that empties such a gate (a delayed SPAWN
        # child refused after the parent suspended) leaves the satisfied active
        # gate with no child-leaf decrement to fire it -> the suspended parent
        # deadlocks until drain-timeout. Pop and dispatch it here so the parent
        # resumes; the parent stays suspended otherwise (intercept returned True
        # for the gate), so this is the only path that advances it.
        active = self._active_joins.get(parent_corr)
        drained_active = None
        if active is not None and active.is_satisfied:
            drained_active = self._active_joins.pop(parent_corr, None)
        # If no successful children AND no gated turns, release the
        # reserved parent state so the parent can drain.
        #
        # Sticky-router note: per-child rollback (the failure branch above)
        # already calls ``release_child_routing`` exactly once for each FORK
        # child whose ``register_child_routing`` was ever invoked, so no
        # additional deferred-eviction step is needed here. (Bug fix:
        # previous code released here unconditionally when any FORK child
        # was intended, racing the per-child rollback and double-
        # decrementing the parent's ref_count.)
        if (
            not any_child_tracked_for_parent(self._child_to_join, parent_corr)
            and not self._future_joins.get(parent_corr)
            and parent_corr in self._descendant_counts
            and self._descendant_counts[parent_corr] <= 0
        ):
            self._release_slot(parent_corr)
            del self._descendant_counts[parent_corr]
        # Dispatch the drained active gate's join turn. The gate was satisfied
        # with zero outstanding children (every child rolled back), so no
        # child-leaf decrement will ever fire it; without this the suspended
        # parent's gated turn is orphaned -> a hang. ``_release_slot`` above only
        # evicts the parent lock dict entry (the held lock is unaffected), so
        # dispatching after it is safe.
        if drained_active is not None:
            await self._release_blocked_join(drained_active)
        self._notify_drain()  # all-children-rolled-back path: no credit return follows

    def _branch_start_timestamp_ms(self, branch) -> float | None:
        """Branch spawn time in ms; falls back to min child turn-0 timestamp.

        Mirrors ``trajectory_source._branch_runtimes`` so snapshot seeding
        and live replay agree on when a branch enters the timeline. Returns
        None when no timing evidence exists (e.g. --ignore-trace-delays
        datasets), which disables offsets for the branch.
        """
        start = _as_timestamp_ms(getattr(branch, "start_timestamp_ms", None))
        if start is not None:
            return start
        child_starts: list[float] = []
        for child_id in branch.child_conversation_ids:
            try:
                meta = self._cs.get_metadata(child_id)
            except Exception:  # noqa: BLE001,S112 - missing metadata = no timing evidence
                continue
            ts = _turn0_timestamp_ms(meta)
            if ts is not None:
                child_starts.append(ts)
        if child_starts:
            return min(child_starts)
        return None

    @staticmethod
    def _child_dispatch_offset_ms(branch_start_ms: float | None, child) -> float:
        """Recorded offset of a child's first request from the branch spawn.

        Zero (immediate dispatch) when either timestamp is missing or the
        child's first request is at/before the spawn.
        """
        if branch_start_ms is None:
            return 0.0
        child_ts = _turn0_timestamp_ms(getattr(child, "metadata", None))
        if child_ts is None:
            return 0.0
        return max(0.0, child_ts - branch_start_ms)

    def _start_delayed_first_turn(
        self, child, offset_ms: float, parent_corr: str
    ) -> None:
        """Schedule a SPAWN child's turn-0 dispatch at its recorded offset.

        Join gates, descendant counts, and ``_child_to_join`` were registered
        at spawn time, so a gated parent keeps waiting while the child sleeps
        and ``has_pending_branch_work()`` stays True (the phase drain's
        existing timeout backstop bounds the wait if the run ends mid-sleep).
        ``cleanup()`` cancels pending sleepers and clears all bookkeeping
        itself, so a cancelled task performs no rollback of its own.
        """
        task = asyncio.create_task(
            self._dispatch_first_turn_after_offset(child, offset_ms, parent_corr)
        )
        self._delayed_dispatch_tasks.add(task)
        task.add_done_callback(self._delayed_dispatch_tasks.discard)
        self.stats.children_delayed += 1

    async def _sleep_offset_ms(self, offset_ms: float) -> None:
        """Sleep out a dispatch offset. Separate method so tests can gate it."""
        await asyncio.sleep(offset_ms / 1000.0)

    async def _dispatch_first_turn_after_offset(
        self, child, offset_ms: float, parent_corr: str
    ) -> None:
        """Delayed-dispatch task body: sleep, then dispatch + settle.

        A post-sleep stop-condition refusal (issuer returns False) rolls back
        exactly like an immediate refusal. Dispatch and settlement run under
        the parent lock, matching the intercept path's locking.
        """
        await self._sleep_offset_ms(offset_ms)
        if self._cleaning_up:
            return
        async with self._parent_locks[parent_corr]:
            try:
                result = await self._dispatch_first_turn(child)
            except Exception as exc:  # noqa: BLE001 - classified like the gather path
                result = exc
            if result is not True:
                self._rollback_failed_first_turn(child, result, parent_corr)
                await self._finalize_failed_dispatches(parent_corr)

    def _ensure_future_join(
        self,
        credit,
        parent_meta,
        parent_corr: str,
        gated_idx: int,
    ) -> PendingBranchJoin:
        """Return (creating if needed) the future join for this gated turn."""
        gates_for_parent = self._future_joins.setdefault(parent_corr, {})
        pending = gates_for_parent.get(gated_idx)
        if pending is None:
            has_forks = False
            if 0 <= gated_idx < len(parent_meta.turns):
                has_forks = bool(
                    getattr(parent_meta.turns[gated_idx], "has_forks", False)
                )
            pending = PendingBranchJoin(
                parent_x_correlation_id=parent_corr,
                parent_conversation_id=credit.conversation_id,
                parent_num_turns=len(parent_meta.turns),
                parent_agent_depth=credit.agent_depth,
                parent_parent_correlation_id=credit.parent_correlation_id,
                gated_turn_index=gated_idx,
                parent_branch_mode=getattr(
                    credit, "branch_mode", ConversationBranchMode.FORK
                ),
                parent_has_forks_on_gated_turn=has_forks,
                # Capture parent's cache-bust state from the suspending
                # credit so the join turn (k+1) inherits the same marker
                # as turns 0..k. The credit always has these fields
                # populated (defaults to None / CacheBustTarget.NONE when
                # the feature is disabled).
                parent_cache_bust_marker=getattr(credit, "cache_bust_marker", None),
                parent_cache_bust_target=getattr(
                    credit, "cache_bust_target", CacheBustTarget.NONE
                ),
            )
            # Phase 3 fan-in seed: pre-populate every prereq_key declared
            # by the gated turn with an unregistered PrereqState so the
            # gate cannot be is_satisfied until every contributing branch
            # has actually fired (registered=True) and reported all its
            # children.
            expected_keys = self._gated_turn_prereq_keys.get(
                (credit.conversation_id, gated_idx), set()
            )
            for prereq_key in expected_keys:
                pending.outstanding[prereq_key] = PrereqState()
            gates_for_parent[gated_idx] = pending
        return pending

    def _get_join(
        self, parent_corr: str, gated_idx: int | None
    ) -> PendingBranchJoin | None:
        """Look up the active or future join for a parent at a given gated turn."""
        if gated_idx is None:
            return None
        active = self._active_joins.get(parent_corr)
        if active is not None and active.gated_turn_index == gated_idx:
            return active
        return self._future_joins.get(parent_corr, {}).get(gated_idx)

    def _pop_future_join(
        self, parent_corr: str, gated_idx: int
    ) -> PendingBranchJoin | None:
        gates = self._future_joins.get(parent_corr)
        if gates is None:
            return None
        pending = gates.pop(gated_idx, None)
        if not gates:
            self._future_joins.pop(parent_corr, None)
        return pending

    def _iter_pending_joins(self) -> list[tuple[str, PendingBranchJoin]]:
        """Flatten active + future joins for cleanup/diagnostics."""
        out: list[tuple[str, PendingBranchJoin]] = list(self._active_joins.items())
        for parent_corr, gates in self._future_joins.items():
            for pending in gates.values():
                out.append((parent_corr, pending))
        return out

    def _maybe_suspend_parent(self, credit) -> bool:
        """Suspend the parent iff its NEXT turn is a gated turn.

        Returns True when the parent should NOT dispatch its next turn
        (strategy dispatch is suppressed). Children finishing before the
        parent arrives pop a "satisfied" future gate and return False.
        """
        parent_corr = credit.x_correlation_id
        next_idx = credit.turn_index + 1

        # Already blocked at this gate — treat as "still suspended".
        active = self._active_joins.get(parent_corr)
        if (
            active is not None
            and active.gated_turn_index == next_idx
            and not active.is_satisfied
        ):
            return True

        future = self._future_joins.get(parent_corr, {}).get(next_idx)
        if future is None:
            return False
        if future.is_satisfied:
            # Children already completed — no need to block.
            self._pop_future_join(parent_corr, next_idx)
            return False
        # Promote to active.
        future.is_blocked = True
        self._active_joins[parent_corr] = future
        # Remove from future layer; active and future for the same gate
        # would otherwise double-count in cleanup diagnostics.
        self._pop_future_join(parent_corr, next_idx)
        self.stats.parents_suspended += 1
        return True

    async def _satisfy_prerequisite(
        self,
        parent_corr: str,
        gated_idx: int | None,
        prereq_key: str | None,
        child_corr: str,
    ) -> PendingBranchJoin | None:
        """Mark one child as complete against a pending join's prereq.

        Returns the pending join iff it is fully satisfied AND the parent
        is already blocked on it (caller dispatches). If the gate becomes
        satisfied before the parent arrives, the future entry is popped
        and None is returned.
        """
        if gated_idx is None or prereq_key is None:
            return None
        pending = self._get_join(parent_corr, gated_idx)
        if pending is None:
            logger.warning(
                "satisfy_prerequisite: no join found for parent=%s gated_idx=%s",
                parent_corr,
                gated_idx,
            )
            return None
        outstanding = pending.outstanding.get(prereq_key)
        if outstanding is None:
            logger.warning(
                "satisfy_prerequisite: prereq_key=%s not registered on join for parent=%s",
                prereq_key,
                parent_corr,
            )
            return None
        # Idempotent double-delivery protection: re-delivery of the same
        # child_corr against the same prereq is a no-op.
        if child_corr in outstanding.completed:
            return None
        outstanding.completed.add(child_corr)
        if not pending.is_satisfied:
            return None
        if pending.is_blocked:
            return self._active_joins.pop(parent_corr, None)
        # Satisfied before the parent arrived — pop the future entry and
        # let the parent breeze through when it reaches the turn.
        self._pop_future_join(parent_corr, gated_idx)
        return None

    async def _release_blocked_join(self, pending: PendingBranchJoin) -> None:
        """Dispatch the parent's gated turn and update stats."""
        assert pending.gated_turn_index is not None, (
            "_release_blocked_join called without a gated_turn_index"
        )
        issued = await self._issuer.dispatch_join_turn(pending)
        if issued:
            self.stats.parents_resumed += 1
        else:
            self.stats.joins_suppressed += 1

    async def _dispatch_first_turn(self, child_sampled_session) -> bool:
        """Dispatch a child's turn-0 via the credit issuer.

        Returns True on successful dispatch, False when the issuer declined
        because a stop condition fired. Callers use this to roll back
        orchestrator bookkeeping when dispatch doesn't actually land a credit.
        """
        result = await self._issuer.dispatch_first_turn(child_sampled_session)
        return bool(result)

    async def on_child_leaf_reached(self, child_x_correlation_id: str) -> None:
        """Called when a child session reaches its final turn (or terminates early)."""
        if self._cleaning_up:
            return
        entries = self._child_to_join.get(child_x_correlation_id)
        if not entries:
            return
        self.stats.children_completed += 1
        await self._handle_child_done(child_x_correlation_id, entries)

    async def on_child_stopped(self, child_x_correlation_id: str) -> None:
        """Called when a child's continuation is blocked by a stop condition.

        The ``CreditCallbackHandler`` invokes this when a non-final child
        return arrives but ``can_send_child_turn`` is False — typically the
        ``--request-count`` cap has been reached. The child has already
        completed at least one turn (we're on its return path), but its
        remaining turns will not be issued. To prevent the parent's join
        from deadlocking, we treat the child as effectively done here:
        same cleanup as ``on_child_leaf_reached`` but tallied under
        ``children_truncated`` instead of ``children_completed`` so the
        observability stays accurate. Idempotent and safe under late or
        duplicate calls (children that have already drained are silently
        ignored).
        """
        if self._cleaning_up:
            return
        entries = self._child_to_join.get(child_x_correlation_id)
        if not entries:
            return
        self.stats.children_truncated += 1
        await self._handle_child_done(child_x_correlation_id, entries)

    async def _handle_child_done(
        self, child_corr: str, entries: list[ChildJoinEntry]
    ) -> None:
        """Shared bookkeeping: gate satisfaction + sticky release + descendant count.

        Phase 3: a single child may contribute to multiple gates when one
        branch is consumed by multiple gated turns. Every entry in
        ``entries`` advances its own gate; each fully-satisfied gate gets
        dispatched. Sticky release and descendant-count decrement fire
        exactly once per child regardless of gate count.
        """
        self._child_to_join.pop(child_corr, None)
        # Every entry shares the same parent_correlation_id by construction.
        parent = entries[0].parent_correlation_id
        child_mode = self._child_modes.pop(child_corr, None)
        if (
            child_mode == ConversationBranchMode.FORK
            and self._sticky_router is not None
        ):
            self._sticky_router.release_child_routing(parent)

        for entry in entries:
            pending = await self._satisfy_prerequisite(
                parent, entry.gated_turn_index, entry.prereq_key, child_corr
            )
            if pending is not None:
                await self._release_blocked_join(pending)

        # Descendant accounting — one decrement per child regardless of the
        # number of gates satisfied.
        if parent in self._descendant_counts:
            self._descendant_counts[parent] -= 1
            # If no active/future joins remain and count reached zero,
            # release the slot (mirrors prior behavior for the
            # no-join/no-child terminal path).
            if (
                self._descendant_counts[parent] <= 0
                and parent not in self._active_joins
                and parent not in self._future_joins
            ):
                self._release_slot(parent)
                del self._descendant_counts[parent]
        # Decrement the child's TREE outstanding count; the registry releases the
        # tree's session slot (and recycles its lane) once root + all descendants
        # have drained. Keyed on the tree root, not the direct parent, so a
        # subchild correctly holds the top root's slot.
        self._tree_descendant_done(child_corr)
        self._notify_drain()  # cap-suppressed joins finalize w/o credit return

    async def on_child_errored(self, child_x_correlation_id: str) -> None:
        """Called when a child session errors mid-branch.

        Under ``AIPERF_DAG_FAIL_FAST=true`` abort the parent and every
        orphan sibling; release sticky refcounts where FORK. Otherwise
        treat the error as leaf-reached for join accounting.
        """
        if self._cleaning_up:
            return
        entries = self._child_to_join.get(child_x_correlation_id)
        if not entries:
            return
        self.stats.children_errored += 1
        if self._fail_fast:
            await self._handle_child_errored_fail_fast(child_x_correlation_id, entries)
        else:
            await self._handle_child_done(child_x_correlation_id, entries)

    async def _handle_child_errored_fail_fast(
        self, child_corr: str, entries: list[ChildJoinEntry]
    ) -> None:
        parent = entries[0].parent_correlation_id
        errored_mode = self._child_modes.pop(child_corr, None)
        self._child_to_join.pop(child_corr, None)
        self._tree_descendant_done(child_corr)

        # Collect all tracked children for this parent as potential orphans.
        orphans = [
            cid
            for cid, ents in list(self._child_to_join.items())
            if ents and ents[0].parent_correlation_id == parent and cid != child_corr
        ]

        # Drop the parent's active/future joins — parent is going down.
        self._active_joins.pop(parent, None)
        self._future_joins.pop(parent, None)

        if (
            errored_mode == ConversationBranchMode.FORK
            and self._sticky_router is not None
        ):
            self._sticky_router.release_child_routing(parent)
        if hasattr(self._issuer, "abort_session"):
            await self._issuer.abort_session(parent)
        self.stats.parents_failed_due_to_child_error += 1

        for orphan in orphans:
            self._child_to_join.pop(orphan, None)
            self._tree_descendant_done(orphan)
            orphan_mode = self._child_modes.pop(orphan, None)
            if (
                orphan_mode == ConversationBranchMode.FORK
                and self._sticky_router is not None
            ):
                self._sticky_router.release_child_routing(parent)
            if hasattr(self._issuer, "abort_session"):
                await self._issuer.abort_session(orphan)

        self._descendant_counts.pop(parent, None)
        self._parent_locks.pop(parent, None)
        self._notify_drain()

    def _release_slot(self, parent_x_correlation_id: str) -> None:
        """Release per-parent orchestration state once the DAG has drained.

        Evicts the parent's lock so long-running benchmarks don't accumulate
        defaultdict entries for every completed root session. Strategy/credit-
        layer slot accounting is handled elsewhere.
        """
        self._parent_locks.pop(parent_x_correlation_id, None)

    def set_drain_observer(self, observer) -> None:
        """Register/detach the sync drain-observer callback. See ``__init__``."""
        self._drain_observer = observer

    def _notify_drain(self) -> None:
        """Fire the registered drain observer (no-op if unset)."""
        observer = self._drain_observer
        if observer is None:
            return
        try:
            observer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("drain observer raised: %s", exc)

    def has_pending_branch_work(self) -> bool:
        """Return True if any DAG-dispatched children are still outstanding."""
        if self._active_joins:
            return True
        if any(gates for gates in self._future_joins.values()):
            return True
        if self._child_to_join:
            return True
        if self._descendant_counts:
            return any(count > 0 for count in self._descendant_counts.values())
        return False

    def cleanup(self) -> None:
        """Log final stats and any leaked state, then clear tracking. Idempotent."""
        if self._cleaning_up:
            return
        self._cleaning_up = True
        self._drain_observer = None
        for task in self._delayed_dispatch_tasks:
            task.cancel()
        self._delayed_dispatch_tasks.clear()
        s = self.stats
        logger.info(
            "BranchOrchestrator stats: spawned=%d completed=%d errored=%d "
            "suspended=%d resumed=%d parents_failed_due_to_child_error=%d "
            "joins_suppressed=%d delayed=%d",
            s.children_spawned,
            s.children_completed,
            s.children_errored,
            s.parents_suspended,
            s.parents_resumed,
            s.parents_failed_due_to_child_error,
            s.joins_suppressed,
            s.children_delayed,
        )
        leaked = self._iter_pending_joins()
        if leaked or self._child_to_join or self._descendant_counts:
            logger.warning(
                "BranchOrchestrator leaked state at cleanup: "
                "%d active_joins, %d future_joins, %d tracked children, "
                "%d parents with descendants",
                len(self._active_joins),
                sum(len(g) for g in self._future_joins.values()),
                len(self._child_to_join),
                len(self._descendant_counts),
            )
            now_ns = time.monotonic_ns()
            for parent_corr, pending in leaked:
                age_ms = (now_ns - pending.created_at_ns) / 1_000_000
                logger.warning(
                    "Abandoned pending join for parent %s "
                    "(outstanding=%d, gated_turn_index=%s, age_ms=%.0f)",
                    parent_corr,
                    pending.total_outstanding,
                    pending.gated_turn_index,
                    age_ms,
                )
        self._active_joins.clear()
        self._future_joins.clear()
        self._child_to_join.clear()
        self._child_root.clear()
        self._child_modes.clear()
        self._descendant_counts.clear()
        self._parent_locks.clear()
        self._pre_dispatched_branches.clear()


def any_child_tracked_for_parent(
    child_to_join: dict[str, list[ChildJoinEntry]], parent_corr: str
) -> bool:
    """Return True if any child in ``child_to_join`` belongs to ``parent_corr``.

    Module-level helper (rather than method) because it is called from inside
    _spawn_children_and_register_gates to decide whether all children rolled
    back and no per-parent state should remain reserved.
    """
    return any(
        any(e.parent_correlation_id == parent_corr for e in ents)
        for ents in child_to_join.values()
    )
