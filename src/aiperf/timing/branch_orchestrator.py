# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DAG branch orchestrator.

Intercepts parent-turn completion, dispatches child sessions (FORK mode),
and releases per-parent state when the DAG drains. See
``docs/benchmark-modes/dag.md`` for user-facing semantics.

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
    6. intercept(credit): if the completed turn declared branches, spawn
       children and suppress strategy dispatch
    7. Strategy dispatch if not intercepted (child bypass uses
       ``agent_depth > 0``)

Stop-condition interaction
--------------------------
Three coordinated guards achieve zero-overshoot, zero-deadlock around DAG
work that outlives the phase's root-sampling completion::

1. **Callback-handler child bypass** (step 7): credit returns carrying
   ``agent_depth > 0`` always reach ``handle_credit_return`` even after
   ``can_send_any_turn`` flips False. Without this, child final returns
   would be silently dropped, leaving parents stuck in ``_pending_joins``.

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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.environment import Environment
from aiperf.common.models.branch_stats import BranchStats
from aiperf.credit.structs import Credit

if TYPE_CHECKING:
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.sticky_router import StickyCreditRouter
    from aiperf.timing.conversation_source import ConversationSource, SampledSession

__all__ = ["BranchOrchestrator", "BranchStats", "PendingBranchJoin"]

logger = logging.getLogger(__name__)


@dataclass
class PendingBranchJoin:
    """Tracking state for outstanding children of a parent session."""

    parent_x_correlation_id: str
    outstanding_children: set[str] = field(default_factory=set)


class BranchOrchestrator:
    """Handles DAG branch dispatch (FORK mode).

    See the module docstring for the credit-return flow, stop-condition
    guards, and cleanup semantics.
    """

    def __init__(
        self,
        conversation_source: ConversationSource,
        credit_issuer: CreditIssuer,
        sticky_router: StickyCreditRouter | None = None,
    ) -> None:
        self._cs = conversation_source
        self._issuer = credit_issuer
        self._sticky_router = sticky_router
        self._child_to_parent: dict[str, str] = {}
        self._child_modes: dict[str, ConversationBranchMode] = {}
        self._pending_joins: dict[str, PendingBranchJoin] = {}
        self._parent_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._descendant_counts: dict[str, int] = {}
        self._fail_fast = Environment.DAG.FAIL_FAST
        self._cleaning_up: bool = False
        self.stats = BranchStats()

    def _get_branch_ids(self, credit: Credit) -> list[str]:
        """Look up the completed turn's ``branch_ids`` from metadata."""
        meta = self._cs.get_metadata(credit.conversation_id)
        if credit.turn_index >= len(meta.turns):
            return []
        return list(meta.turns[credit.turn_index].branch_ids)

    async def intercept(self, credit: Credit) -> bool:
        """Intercept the credit-return path.

        If the completed turn triggers branches, start children and return True
        to suppress the strategy's default next-turn dispatch. Otherwise return
        False.

        FORK-mode children are routed to the parent's worker via sticky routing
        (``parent_correlation_id`` keying); the worker seeds each child's
        ``UserSession.turn_list`` from the parent's local session.
        SPAWN-mode children route freely (no sticky pin).
        """
        if self._cleaning_up:
            return False
        branch_ids = self._get_branch_ids(credit)
        if not branch_ids:
            return False

        parent_corr = credit.x_correlation_id
        parent_meta = self._cs.get_metadata(credit.conversation_id)
        branches_by_id = {s.branch_id: s for s in parent_meta.branches}

        async with self._parent_locks[parent_corr]:
            all_children: list = []
            all_child_ids: set[str] = set()
            any_fork_intended = False
            fork_registrations = 0
            for b_id in branch_ids:
                branch = branches_by_id[b_id]
                is_fork = branch.mode == ConversationBranchMode.FORK
                if is_fork:
                    any_fork_intended = True
                for child_conv_id in branch.child_conversation_ids:
                    # Static depth is precomputed by the dag_jsonl loader's
                    # topology BFS (``_compute_depths``) and stamped on
                    # ``ConversationMetadata.agent_depth``. No runtime
                    # ``parent_depth + 1`` arithmetic — the loader is
                    # authoritative for the FORK-only single-parent
                    # topology, and runtime is just a lookup.
                    child_meta = self._cs.get_metadata(child_conv_id)
                    try:
                        child = self._cs.start_branch_child(
                            parent_correlation_id=parent_corr,
                            child_conversation_id=child_conv_id,
                            agent_depth=child_meta.agent_depth,
                            branch_mode=branch.mode,
                        )
                    except Exception:
                        logger.exception(
                            "start_branch_child failed for %s", child_conv_id
                        )
                        self.stats.children_errored += 1
                        continue

                    self._child_to_parent[child.x_correlation_id] = parent_corr
                    self._child_modes[child.x_correlation_id] = branch.mode
                    all_child_ids.add(child.x_correlation_id)
                    all_children.append(child)
                    # Only FORK-mode children sticky-route to the parent's
                    # worker; SPAWN-mode children do not register a refcount.
                    if is_fork and self._sticky_router is not None:
                        self._sticky_router.register_child_routing(parent_corr)
                        fork_registrations += 1
                    self.stats.children_spawned += 1

            self._pending_joins[parent_corr] = PendingBranchJoin(
                parent_x_correlation_id=parent_corr,
                outstanding_children=all_child_ids,
            )
            self._descendant_counts[parent_corr] = 1 + len(all_child_ids)

            results = await asyncio.gather(
                *(self._dispatch_first_turn(child) for child in all_children),
                return_exceptions=True,
            )
            # Clean up state for children whose first-turn dispatch failed
            # (returned falsy or raised). ``dispatch_first_turn`` returns
            # True iff the credit was actually sent on the wire; False
            # means the gate refused (no credit to wait for) and we must
            # roll back our own bookkeeping to avoid leaking the
            # pending-join entry and sticky refcount.
            for child, result in zip(all_children, results, strict=True):
                if result is True:
                    continue
                child_corr = child.x_correlation_id
                child_mode = self._child_modes.pop(child_corr, None)
                pending = self._pending_joins.get(parent_corr)
                if pending is not None:
                    pending.outstanding_children.discard(child_corr)
                self._child_to_parent.pop(child_corr, None)
                # Mirror the mode-gated registration above: only FORK-mode
                # children ever incremented the sticky refcount.
                if (
                    child_mode == ConversationBranchMode.FORK
                    and self._sticky_router is not None
                ):
                    self._sticky_router.release_child_routing(parent_corr)
                    fork_registrations -= 1
                if parent_corr in self._descendant_counts:
                    self._descendant_counts[parent_corr] -= 1
                self.stats.children_errored += 1
                self.stats.children_spawned -= 1

            # If every dispatch failed and there's no join, mirror the no-join
            # terminal path: release the parent slot and drop the pending join.
            pending = self._pending_joins.get(parent_corr)
            if pending is not None and not pending.outstanding_children:
                del self._pending_joins[parent_corr]
                if parent_corr in self._descendant_counts:
                    # Drop the root's reserved count along with the pending join.
                    self._descendant_counts[parent_corr] -= 1
                    if self._descendant_counts[parent_corr] <= 0:
                        self._release_slot(parent_corr)
                        del self._descendant_counts[parent_corr]
                # The sticky router deferred eviction of the parent entry
                # because this turn declared FORK branches; if no FORK child
                # is live to release the refcount later, drop it now so the
                # entry and its active_sessions count do not leak.
                if (
                    any_fork_intended
                    and fork_registrations <= 0
                    and self._sticky_router is not None
                ):
                    self._sticky_router.release_child_routing(parent_corr)

        return True

    async def _dispatch_first_turn(self, child_sampled_session: SampledSession) -> bool:
        """Dispatch a child's turn-0 via the credit issuer.

        Returns True on successful dispatch, False when the issuer declined
        (e.g. slots saturated). Callers use this to roll back orchestrator
        bookkeeping when dispatch doesn't actually land a credit.
        """
        result = await self._issuer.dispatch_first_turn(child_sampled_session)
        return bool(result)

    async def on_child_leaf_reached(self, child_x_correlation_id: str) -> None:
        """Called when a child session reaches its final turn (or terminates early).

        Decrements join counters, dispatches the parent's join turn when all
        outstanding children complete, and releases the parent's slot when all
        descendants drain.
        """
        if self._cleaning_up:
            return
        if child_x_correlation_id not in self._child_to_parent:
            return
        self.stats.children_completed += 1
        await self._decrement_for_done_child(child_x_correlation_id)

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
        if child_x_correlation_id not in self._child_to_parent:
            return
        self.stats.children_truncated += 1
        await self._decrement_for_done_child(child_x_correlation_id)

    async def _decrement_for_done_child(self, child_x_correlation_id: str) -> None:
        """Shared logic: remove child from parent tracking and update join."""
        parent = self._child_to_parent.pop(child_x_correlation_id, None)
        if parent is None:
            return
        child_mode = self._child_modes.pop(child_x_correlation_id, None)
        # Only FORK-mode children took a sticky refcount at dispatch; releasing
        # must match. SPAWN-mode children never pinned to the parent's worker.
        if (
            child_mode == ConversationBranchMode.FORK
            and self._sticky_router is not None
        ):
            self._sticky_router.release_child_routing(parent)
        await self._on_descendant_done(parent, child_x_correlation_id)

    async def on_child_errored(self, child_x_correlation_id: str) -> None:
        """Called when a child session errors mid-branch.

        Under ``AIPERF_DAG_FAIL_FAST=true`` drop the pending join, release the
        errored child's sticky refcount (FORK mode only), and abort every
        orphaned sibling (releasing their sticky refcount where FORK, and,
        where supported, aborting their issuer sessions). Otherwise treat the
        error as leaf-reached for join counting.

        ``children_errored`` is only incremented when the child was still
        tracked in ``_child_to_parent``; late/duplicate errored notifications
        for orphans already aborted by fail-fast are silently ignored so the
        stats counter stays accurate.
        """
        if self._cleaning_up:
            return
        # Peek, do not pop: the fail-fast branch below may still need to
        # identify this child, and the else branch's
        # _decrement_for_done_child does the pop.
        if child_x_correlation_id not in self._child_to_parent:
            return
        self.stats.children_errored += 1
        if self._fail_fast:
            parent = self._child_to_parent.pop(child_x_correlation_id, None)
            if parent is None:
                return
            errored_mode = self._child_modes.pop(child_x_correlation_id, None)
            pending = self._pending_joins.pop(parent, None)
            # Release the errored child's sticky refcount iff FORK-registered.
            if (
                errored_mode == ConversationBranchMode.FORK
                and self._sticky_router is not None
            ):
                self._sticky_router.release_child_routing(parent)
            if hasattr(self._issuer, "abort_session"):
                await self._issuer.abort_session(parent)
            self.stats.parents_failed_due_to_child_error += 1
            orphans = (
                pending.outstanding_children - {child_x_correlation_id}
                if pending
                else set()
            )
            # Orphans: release their sticky refcount too (FORK-mode only) and
            # abort their sessions.
            for orphan in orphans:
                self._child_to_parent.pop(orphan, None)
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
        else:
            await self._decrement_for_done_child(child_x_correlation_id)

    async def _on_descendant_done(self, parent: str, descendant_x_corr: str) -> None:
        if parent in self._descendant_counts:
            self._descendant_counts[parent] -= 1

        pending = self._pending_joins.get(parent)
        if pending is not None:
            pending.outstanding_children.discard(descendant_x_corr)
            if not pending.outstanding_children:
                # Parent's own terminal turn is already complete, so drop
                # root's reserved count along with the pending join.
                if parent in self._descendant_counts:
                    self._descendant_counts[parent] -= 1
                del self._pending_joins[parent]

        if parent in self._descendant_counts and self._descendant_counts[parent] <= 0:
            self._release_slot(parent)
            del self._descendant_counts[parent]

    def _release_slot(self, parent_x_correlation_id: str) -> None:
        """Release per-parent orchestration state once the DAG has drained.

        Evicts the parent's lock so long-running benchmarks don't accumulate
        defaultdict entries for every completed root session. Strategy/credit-
        layer slot accounting is handled elsewhere.
        """
        self._parent_locks.pop(parent_x_correlation_id, None)

    def has_pending_branch_work(self) -> bool:
        """Return True if any DAG-dispatched children are still outstanding.

        Strategies check this before exiting their main dispatch loop so that
        children (and grandchildren) can continue to dispatch continuation
        turns even after the phase has been marked ``is_sending_complete`` for
        root sampling. Returns False when the DAG has fully drained.
        """
        if self._pending_joins:
            return True
        if self._descendant_counts:
            return any(count > 0 for count in self._descendant_counts.values())
        return bool(self._child_to_parent)

    def cleanup(self) -> None:
        """Log final stats and any leaked state, then clear tracking. Idempotent.

        Called by ``PhaseRunner`` at every phase-exit path (normal completion,
        error, cancellation). Setting ``_cleaning_up`` short-circuits
        ``intercept`` / ``on_child_leaf_reached`` / ``on_child_errored`` for
        late-arriving credit returns so they do not dispatch new work after
        the phase has closed.

        Any residual entries in ``_pending_joins``, ``_child_to_parent``, or
        ``_descendant_counts`` at this point indicate a DAG that failed to
        drain — typically a worker crash or a protocol mismatch on a child
        session. The parent correlation id and outstanding count are logged
        to aid diagnosis; no automatic recovery is attempted.
        """
        if self._cleaning_up:
            return
        self._cleaning_up = True
        s = self.stats
        logger.info(
            "BranchOrchestrator stats: spawned=%d completed=%d errored=%d "
            "truncated=%d suspended=%d resumed=%d parents_failed_due_to_child_error=%d",
            s.children_spawned,
            s.children_completed,
            s.children_errored,
            s.children_truncated,
            s.parents_suspended,
            s.parents_resumed,
            s.parents_failed_due_to_child_error,
        )
        if self._pending_joins or self._child_to_parent or self._descendant_counts:
            logger.warning(
                "BranchOrchestrator leaked state at cleanup: "
                "%d pending_joins, %d tracked children, %d parents with descendants",
                len(self._pending_joins),
                len(self._child_to_parent),
                len(self._descendant_counts),
            )
            for parent_corr, pending in self._pending_joins.items():
                logger.warning(
                    "Abandoned pending join for parent %s (outstanding=%d)",
                    parent_corr,
                    len(pending.outstanding_children),
                )
        self._pending_joins.clear()
        self._child_to_parent.clear()
        self._child_modes.clear()
        self._descendant_counts.clear()
        self._parent_locks.clear()
