# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spawn-path mixin for ``BranchOrchestrator``.

The branch-spawn pipeline (``_spawn_children_and_register_gates`` plus its
substeps) lives here so the main orchestrator file stays within the
ergonomics file-size cap. The ``BranchOrchestrator`` class inherits from
``BranchOrchestratorSpawnMixin`` to compose the methods.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aiperf.common.enums import ConversationBranchMode
from aiperf.timing._branch_orchestrator_helpers import (
    any_child_tracked_for_parent,
    gate_for_branch_map,
)
from aiperf.timing._branch_orchestrator_state import (
    ChildJoinEntry,
    PendingBranchJoin,
    PrereqState,
)

if TYPE_CHECKING:
    from aiperf.common.models.branch import ConversationBranchInfo
    from aiperf.common.models.dataset_models import ConversationMetadata
    from aiperf.credit.structs import Credit
    from aiperf.timing.conversation_source import SampledSession

logger = logging.getLogger(__name__)


class BranchOrchestratorSpawnMixin:
    """Spawn pipeline + child-dispatch rollback. See ``BranchOrchestrator``.

    Note: "spawn" here means dispatching a child conversation in DAG mode
    (``ConversationBranchMode.SPAWN``) -- unrelated to ``SpawnWorkersCommand``,
    which spawns worker subprocesses.
    """

    async def _fire_pre_session_children(self, branch: ConversationBranchInfo) -> None:
        """Issue turn-0 for every child of a pre-session SPAWN branch."""
        for child_cid in branch.child_conversation_ids:
            try:
                child_session = self._cs.start_pre_session_child(child_cid)
            except Exception:
                logger.exception("start_pre_session_child failed for %s", child_cid)
                self.stats.children_errored += 1
                continue
            issued = await self._issuer.dispatch_first_turn(child_session)
            if issued:
                self.stats.children_spawned += 1
            else:
                # ``dispatch_first_turn`` only returns False under
                # stop-condition refusal (cap reached). Tally as truncated.
                self.stats.children_truncated += 1

    async def _spawn_children_and_register_gates(
        self, credit: Credit, branch_ids: list[str]
    ) -> None:
        """Resolve branches, start children, and register future joins."""
        parent_corr = credit.x_correlation_id
        parent_meta = self._cs.get_metadata(credit.conversation_id)
        branches_by_id = {b.branch_id: b for b in parent_meta.branches}

        prereq_entries = self._prereq_index.get(
            (credit.conversation_id, credit.turn_index), []
        )
        gate_for_branch = gate_for_branch_map(prereq_entries)

        all_children, per_child_branch_mode, expected_gates = self._start_children(
            credit,
            parent_meta,
            branch_ids=branch_ids,
            branches_by_id=branches_by_id,
            gate_for_branch=gate_for_branch,
        )

        # Track every successfully-started child for descendant accounting.
        if all_children:
            self._descendant_counts.setdefault(parent_corr, 0)
            self._descendant_counts[parent_corr] += len(all_children)

        # If any expected gate had zero children actually register, still
        # create a future-join entry so the drain logic sees it and fires.
        for gated_idx, prereq_key in expected_gates:
            pending = self._ensure_future_join(
                credit, parent_meta, parent_corr, gated_idx
            )
            state = pending.outstanding.setdefault(prereq_key, PrereqState())
            state.registered = True

        results = await asyncio.gather(
            *(self._dispatch_first_turn(child) for child in all_children),
            return_exceptions=True,
        )
        for child, result in zip(all_children, results, strict=True):
            if result is True:
                continue
            self._rollback_failed_child(
                child, result, parent_corr, per_child_branch_mode
            )

        # Drain any vestigial gates created in this call that are now
        # zero-outstanding (every child rolled back).
        await self._drain_vestigial_gates(parent_corr)

    def _start_children(
        self,
        credit: Credit,
        parent_meta: ConversationMetadata,
        *,
        branch_ids: list[str],
        branches_by_id: dict[str, ConversationBranchInfo],
        gate_for_branch: dict[str, list[tuple[int, str]]],
    ) -> tuple[
        list[SampledSession], dict[str, ConversationBranchMode], set[tuple[int, str]]
    ]:
        """Start children for each branch and register their per-gate
        bookkeeping. Returns ``(all_children, per_child_branch_mode,
        expected_gates)``.
        """
        parent_corr = credit.x_correlation_id
        parent_depth = credit.agent_depth
        all_children: list[SampledSession] = []
        per_child_branch_mode: dict[str, ConversationBranchMode] = {}
        expected_gates: set[tuple[int, str]] = set()

        for b_id in branch_ids:
            branch = branches_by_id.get(b_id)
            if branch is None:
                continue
            # Skip branches already fired via dispatch_pre_session_branches.
            if (credit.conversation_id, b_id) in self._pre_dispatched_branches:
                continue
            branch_gates = gate_for_branch.get(branch.branch_id, [])
            # Pre-session (background) branches never gate the parent.
            if getattr(branch, "dispatch_timing", "post") == "pre":
                branch_gates = []
            for gate in branch_gates:
                expected_gates.add(gate)
            for child_conv_id in branch.child_conversation_ids:
                started = self._start_one_child(
                    credit,
                    parent_meta,
                    parent_corr=parent_corr,
                    parent_depth=parent_depth,
                    branch=branch,
                    branch_gates=branch_gates,
                    child_conv_id=child_conv_id,
                )
                if started is None:
                    continue
                child, child_corr = started
                per_child_branch_mode[child_corr] = branch.mode
                all_children.append(child)
        return all_children, per_child_branch_mode, expected_gates

    def _start_one_child(
        self,
        credit: Credit,
        parent_meta: ConversationMetadata,
        *,
        parent_corr: str,
        parent_depth: int,
        branch: ConversationBranchInfo,
        branch_gates: list[tuple[int, str]],
        child_conv_id: str,
    ) -> tuple[SampledSession, str] | None:
        """Start one child session and register all its bookkeeping.

        Returns ``(child, child_corr)`` on success, ``None`` if start failed.
        """
        try:
            child = self._cs.start_branch_child(
                parent_correlation_id=parent_corr,
                child_conversation_id=child_conv_id,
                agent_depth=parent_depth + 1,
                branch_mode=branch.mode,
            )
        except Exception:
            logger.exception("start_branch_child failed for %s", child_conv_id)
            self.stats.children_errored += 1
            return None
        child_corr = child.x_correlation_id
        self._child_modes[child_corr] = branch.mode
        # FORK-mode children sticky-route to the parent's worker; SPAWN-mode
        # children do not register a refcount.
        if (
            branch.mode == ConversationBranchMode.FORK
            and self._sticky_router is not None
        ):
            self._sticky_router.register_child_routing(parent_corr)
        self.stats.children_spawned += 1
        # Register in _child_to_join (one entry per gate this child
        # contributes to) and bump each gate's expected counter.
        entries: list[ChildJoinEntry] = []
        if branch_gates:
            for gated_idx, prereq_key in branch_gates:
                pending = self._ensure_future_join(
                    credit, parent_meta, parent_corr, gated_idx
                )
                state = pending.outstanding.setdefault(prereq_key, PrereqState())
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
            # Background / no gate: still track for descendant accounting.
            entries.append(
                ChildJoinEntry(
                    parent_correlation_id=parent_corr,
                    gated_turn_index=None,
                    prereq_key=None,
                )
            )
        self._child_to_join[child_corr] = entries
        return child, child_corr

    def _rollback_failed_child(
        self,
        child: SampledSession,
        result: object,
        parent_corr: str,
        per_child_branch_mode: dict[str, ConversationBranchMode],
    ) -> None:
        """Undo bookkeeping for a child whose dispatch_first_turn didn't return True."""
        child_corr = child.x_correlation_id
        child_mode = per_child_branch_mode.get(child_corr)
        self._child_modes.pop(child_corr, None)
        entries = self._child_to_join.pop(child_corr, [])
        for entry in entries:
            if entry.prereq_key is None:
                continue
            pending = self._get_join(parent_corr, entry.gated_turn_index)
            if pending is None:
                continue
            state = pending.outstanding.get(entry.prereq_key)
            if state is not None and state.expected > 0:
                # Rollback decrements ``expected`` without touching
                # ``completed``. Clamp at >= len(completed) so an already-
                # delivered completion doesn't revert is_done.
                state.expected = max(len(state.completed), state.expected - 1)
        if (
            child_mode == ConversationBranchMode.FORK
            and self._sticky_router is not None
        ):
            self._sticky_router.release_child_routing(parent_corr)
        if parent_corr in self._descendant_counts:
            self._descendant_counts[parent_corr] -= 1
        # Three-way classification of non-True gather results:
        #   * BaseException -> genuine error.
        #   * False -> stop-condition refusal; tally as truncated.
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

    async def _drain_vestigial_gates(self, parent_corr: str) -> None:
        """Drain any zero-outstanding gates created in this spawn cycle.

        If no children at all landed for some declared gate (every child
        rolled back), the gate is now is_satisfied — dispatch the gated
        turn immediately to avoid hanging the parent.
        """
        gates_for_parent = self._future_joins.get(parent_corr, {})
        drained_gates: list[PendingBranchJoin] = []
        for gated_idx, pending in list(gates_for_parent.items()):
            if pending.is_satisfied:
                drained_gates.append(pending)
                self._pop_future_join(parent_corr, gated_idx)
        # Sticky-router note: per-child rollback already calls
        # ``release_child_routing`` exactly once for each FORK child whose
        # ``register_child_routing`` was ever invoked, so no additional
        # deferred-eviction step is needed here.
        if (
            not any_child_tracked_for_parent(self._child_to_join, parent_corr)
            and not self._future_joins.get(parent_corr)
            and parent_corr in self._descendant_counts
            and self._descendant_counts[parent_corr] <= 0
        ):
            self._release_slot(parent_corr)
            del self._descendant_counts[parent_corr]
        for pending in drained_gates:
            await self._release_blocked_join(pending)
        # Drain hook: covers the all-children-rolled-back path where every
        # spawn was refused at the cap gate and no credit return will
        # follow. Without this, the phase blocks on
        # all_credits_returned_event.
        self._notify_drain()
