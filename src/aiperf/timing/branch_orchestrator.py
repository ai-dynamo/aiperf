# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAG branch orchestrator. Intercepts parent-turn completion, dispatches
child sessions (FORK/SPAWN), tracks join completion, releases per-parent
state when the DAG drains. See ``docs/benchmark-modes/dag.md``.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.environment import Environment
from aiperf.common.models.branch_stats import BranchStats
from aiperf.timing._branch_orchestrator_drain import BranchOrchestratorDrainMixin
from aiperf.timing._branch_orchestrator_helpers import (
    any_child_tracked_for_parent,
    build_prereq_index,
)
from aiperf.timing._branch_orchestrator_logging import BranchOrchestratorLoggingMixin
from aiperf.timing._branch_orchestrator_spawn import BranchOrchestratorSpawnMixin
from aiperf.timing._branch_orchestrator_state import (
    ChildJoinEntry,
    PendingBranchJoin,
    PrereqState,
)

if TYPE_CHECKING:
    from aiperf.common.models.dataset_models import ConversationMetadata
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.sticky_router import StickyCreditRouter
    from aiperf.credit.structs import Credit
    from aiperf.timing.conversation_source import ConversationSource, SampledSession

__all__ = [
    "BranchOrchestrator",
    "BranchStats",
    "ChildJoinEntry",
    "PendingBranchJoin",
    "PrereqState",
    "any_child_tracked_for_parent",
]

logger = logging.getLogger(__name__)


class BranchOrchestrator(
    BranchOrchestratorDrainMixin,
    BranchOrchestratorLoggingMixin,
    BranchOrchestratorSpawnMixin,
):
    """DAG branch orchestrator: dispatches FORK/SPAWN children and gates parent joins.

    Single-instance-per-phase, owned by ``PhaseRunner``; constructed once
    the dataset is loaded and a ``CreditIssuer`` exists. Public lifecycle:

    1. ``await dispatch_pre_session_branches()`` -- ONCE before the strategy
       issues its first root-turn-0 credit. Idempotent.
    2. ``await intercept(credit)`` -- called from ``CreditCallbackHandler``
       on EVERY credit return. Returns True iff the strategy must suppress
       its next-turn dispatch (next turn gated by an unsatisfied SPAWN_JOIN).
       Side-effect: spawns branches declared on the completed turn.
    3. ``await on_child_leaf_reached(child_corr)`` /
       ``on_child_stopped(...)`` / ``on_child_errored(...)`` -- called by
       the worker when a child session terminates; drives gate satisfaction.
    4. ``has_pending_branch_work()`` -- strategy polls this to decide whether
       the phase can finalize.
    5. ``cleanup()`` -- idempotent; logs final ``BranchStats``. Call once
       at phase teardown.

    FORK children sticky-route via ``StickyCreditRouter.release_child_routing``
    paired with ``on_child_*``; under ``AIPERF_DAG_FAIL_FAST=true`` the first
    child error aborts the parent and every orphan sibling.
    See ``docs/benchmark-modes/dag.md``.
    """

    def __init__(
        self,
        conversation_source: ConversationSource,
        credit_issuer: CreditIssuer,
        sticky_router: StickyCreditRouter | None = None,
        *,
        benchmark_id: str = "unknown",
    ) -> None:
        self._cs = conversation_source
        self._issuer = credit_issuer
        self._sticky_router = sticky_router
        self._benchmark_id = benchmark_id
        self._child_modes: dict[str, ConversationBranchMode] = {}
        # Two-level pending-join state: a "future" join is registered at
        # spawn time and promoted to "active" once the parent reaches the
        # turn immediately preceding the gated turn.
        self._future_joins: dict[str, dict[int, PendingBranchJoin]] = {}
        self._active_joins: dict[str, PendingBranchJoin] = {}
        self._child_to_join: dict[str, list[ChildJoinEntry]] = {}
        self._parent_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._descendant_counts: dict[str, int] = {}
        # Phase 2b: records (conv_id, branch_id) for branches that were
        # pre-dispatched via dispatch_pre_session_branches. The per-turn
        # spawn path skips branches that appear here.
        self._pre_dispatched_branches: set[tuple[str, str]] = set()
        self._fail_fast = Environment.DAG.FAIL_FAST
        self._cleaning_up: bool = False
        self.stats = BranchStats()
        dataset_meta = getattr(self._cs, "dataset_metadata", None)
        conversations = getattr(dataset_meta, "conversations", None) or []
        # Pre-built indices (built once at init from each turn's
        # SPAWN_JOIN prerequisites): see helpers module.
        self._prereq_index, self._gated_turn_prereq_keys = build_prereq_index(
            conversations
        )

    def get_branch_ids(self, credit: Credit) -> list[str]:
        """Look up the completed turn's ``branch_ids`` from metadata.

        Public so the credit-callback handler can probe whether a returning
        credit will trigger DAG dispatch (used to defer phase-completion
        signalling).
        """
        meta = self._cs.get_metadata(credit.conversation_id)
        if credit.turn_index >= len(meta.turns):
            return []
        return list(meta.turns[credit.turn_index].branch_ids)

    async def dispatch_pre_session_branches(self) -> None:
        """Pre-dispatch background SPAWN children marked dispatch_timing='pre'.

        Called once by ``PhaseRunner.run`` before the strategy starts issuing
        root turn-0 credits. The per-turn spawn path consults
        ``_pre_dispatched_branches`` to skip these branches on the parent's
        turn-0 credit return so children are not dispatched twice. The
        validator restricts this path to SPAWN-mode ``dispatch_timing='pre'``
        branches attached to turn 0 of a root conversation.
        """
        if self._cleaning_up:
            return
        dataset_meta = getattr(self._cs, "dataset_metadata", None)
        if dataset_meta is None:
            return
        conversations = getattr(dataset_meta, "conversations", None) or []
        for conv in conversations:
            # Filter primarily on ``is_root`` so SPAWN-mode children
            # (``is_root=False`` but ``agent_depth=0`` by sampler semantics)
            # are skipped. ``agent_depth > 0`` stays as a defensive belt for
            # programmatic bypass that would otherwise dispatch a nested
            # child's pre branch as if it were a root.
            is_root = getattr(conv, "is_root", True)
            agent_depth = getattr(conv, "agent_depth", 0)
            if not is_root or agent_depth > 0 or not conv.turns:
                continue
            turn0_branch_ids = set(conv.turns[0].branch_ids or [])
            for branch in conv.branches:
                if getattr(branch, "dispatch_timing", "post") != "pre":
                    continue
                if branch.branch_id not in turn0_branch_ids:
                    continue
                await self._fire_pre_session_children(branch)
                self._pre_dispatched_branches.add(
                    (conv.conversation_id, branch.branch_id)
                )

    async def intercept(self, credit: Credit) -> bool:
        """Credit-callback hook: dispatch DAG branches and gate parent joins.

        Called by ``CreditCallbackHandler`` on every credit return. Two things
        happen independently:

        1. Side-effect: any FORK/SPAWN branches declared on the completed
           turn are spawned (FORK sticky-pinned, SPAWN with explicit
           ``join_at`` registered as a future-join).
        2. Returns ``True`` iff the parent's NEXT turn is gated by an
           unsatisfied SPAWN_JOIN -- the strategy MUST then suppress its
           own default next-turn dispatch.

        Per-session locking via ``_parent_locks[credit.x_correlation_id]``
        serializes intercepts within a session; intercepts at different
        correlation_ids run concurrently. Safe at any ``agent_depth``.
        """
        if self._cleaning_up:
            return False
        parent_corr = credit.x_correlation_id
        async with self._parent_locks[parent_corr]:
            branch_ids = self.get_branch_ids(credit)
            if branch_ids:
                await self._spawn_children_and_register_gates(credit, branch_ids)
            return self._maybe_suspend_parent(credit)

    def _ensure_future_join(
        self,
        credit: Credit,
        parent_meta: ConversationMetadata,
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
            )
            # Phase 3 fan-in seed: pre-populate every prereq_key declared
            # by the gated turn with an unregistered PrereqState so the
            # gate cannot be is_satisfied until every contributing branch
            # has actually fired and reported all its children.
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

    def _maybe_suspend_parent(self, credit: Credit) -> bool:
        """Suspend the parent iff its NEXT turn is a gated turn.

        Returns True when the parent should NOT dispatch its next turn
        (strategy dispatch is suppressed). Children finishing before the
        parent arrives pop a "satisfied" future gate and return False.
        """
        parent_corr = credit.x_correlation_id
        next_idx = credit.turn_index + 1

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
        is already blocked on it. If the gate becomes satisfied before the
        parent arrives, the future entry is popped and None is returned.
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
        # Idempotent double-delivery protection.
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

    async def _dispatch_first_turn(self, child_sampled_session: SampledSession) -> bool:
        """Dispatch a child's turn-0 via the credit issuer."""
        result = await self._issuer.dispatch_first_turn(child_sampled_session)
        return bool(result)

    async def on_child_leaf_reached(self, child_x_correlation_id: str) -> None:
        """Called when a child session reaches its final turn."""
        if self._cleaning_up:
            return
        entries = self._child_to_join.get(child_x_correlation_id)
        if not entries:
            return
        self.stats.children_completed += 1
        await self._handle_child_done(child_x_correlation_id, entries)

    async def on_child_stopped(self, child_x_correlation_id: str) -> None:
        """Called when a child's continuation is blocked by a stop condition.

        Treated as effectively done so the parent's join can drain, but
        tallied under ``children_truncated`` instead of
        ``children_completed`` so observability stays accurate.
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
        """Shared bookkeeping: gate satisfaction + sticky release + descendant count."""
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
            if (
                self._descendant_counts[parent] <= 0
                and parent not in self._active_joins
                and parent not in self._future_joins
            ):
                self._release_slot(parent)
                del self._descendant_counts[parent]
        # Drain hook: cap-suppressed joins finalize without a downstream
        # credit return, so re-check completion here.
        self._notify_drain()

    async def on_child_errored(self, child_x_correlation_id: str) -> None:
        """Called when a child session errors mid-branch.

        Under ``AIPERF_DAG_FAIL_FAST=true`` abort the parent and every
        orphan sibling. Otherwise treat the error as leaf-reached for join
        accounting.
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
        # Honor the docs' "abort the whole run on first DAG child error"
        # contract: tell the phase-side handler to cancel every active
        # phase lifecycle so the strategy loop stops issuing new wire
        # credits. Without this, only the parent of the errored child
        # was aborted while other unrelated parents kept firing — the
        # wire-request budget ran out as if FAIL_FAST were disabled.
        self._notify_abort()

    def _release_slot(self, parent_x_correlation_id: str) -> None:
        """Release per-parent orchestration state once the DAG has drained."""
        self._parent_locks.pop(parent_x_correlation_id, None)

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

    def snapshot_branch_stats(self) -> BranchStats:
        """Independent copy of the BranchStats counters (for publication)."""
        return self.stats.model_copy()

    def cleanup(self) -> None:
        """Log final stats and any leaked state, then clear tracking. Idempotent."""
        if self._cleaning_up:
            return
        self._cleaning_up = True
        self._drain_observer = None
        self._abort_observer = None
        self._log_stats()
        self._log_leaks()
        self._active_joins.clear()
        self._future_joins.clear()
        self._child_to_join.clear()
        self._child_modes.clear()
        self._descendant_counts.clear()
        self._parent_locks.clear()
        self._pre_dispatched_branches.clear()
