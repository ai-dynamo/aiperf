# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 unit tests for :class:`BranchOrchestrator` and :class:`CreditIssuer`.

Covers Phase 0 adjacent bug fixes (still valid under Phase 1's revised
data model):

- ``dispatch_join_turn`` propagates ``parent_branch_mode`` and
  ``parent_has_forks_on_gated_turn`` from :class:`PendingBranchJoin` instead
  of hardcoding FORK.
- ``BranchOrchestrator.intercept`` dispatches the gated join turn immediately
  when every ``start_branch_child`` call fails (no children landed), instead
  of registering a dead pending join that hangs the parent.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator


def _mk_conv(
    cid: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo],
) -> ConversationMetadata:
    return ConversationMetadata(conversation_id=cid, turns=turns, branches=branches)


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: next(
        c for c in conversations if c.conversation_id == cid
    )
    return cs


# ============================================================
# 0.1. dispatch_join_turn propagates SPAWN parent mode
# ============================================================

# NOTE: test_dispatch_join_turn_preserves_spawn_parent_mode and
# test_dispatch_join_turn_preserves_has_forks_on_gated_turn pruned here -
# they exercise CreditIssuer.dispatch_join_turn which is wired in P2T18.
# When restoring, also re-add imports for TurnToSend (aiperf.credit.structs)
# and PendingBranchJoin (aiperf.timing.branch_orchestrator).


# ============================================================
# 0.3. intercept with all-children-failed + gate must not hang
# ============================================================


@pytest.mark.asyncio
async def test_intercept_all_children_failed_with_gate_does_not_hang():
    """When every ``start_branch_child`` raises on a parent turn whose next
    turn is gated, the future join has zero outstanding children and would
    never fire via the child-leaf decrement path. The orchestrator pops the
    vacuously-satisfied gate SILENTLY and returns False; the strategy's normal
    continuation (callback handler -> handle_credit_return -> _dispatch_next_turn)
    then dispatches the now-ungated turn exactly once. The orchestrator must NOT
    dispatch the join turn here for an immediate-next gate -- intercept returns
    False, so doing so would double-dispatch the same turn_index."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["a", "b"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv = _mk_conv(
        "conv",
        [
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        [branch],
    )
    cs = _mk_source([conv])
    cs.start_branch_child = MagicMock(side_effect=RuntimeError("boom"))

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=False)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    credit = MagicMock(
        x_correlation_id="root-corr",
        conversation_id="conv",
        turn_index=0,
        agent_depth=0,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )

    # No children landed; the gate is vacuously satisfied at zero outstanding
    # and popped SILENTLY. Parent's next turn is turn 1; intercept returns False
    # (no surviving unsatisfied gate at next_turn_index), so the strategy's
    # normal continuation dispatches turn 1 -- exactly once.
    result = await orch.intercept(credit)
    assert result is False

    # The orchestrator must NOT dispatch the immediate-next gate here: intercept
    # returned False, so the callback handler's normal continuation advances
    # turn 1. Dispatching here too would double-dispatch the same turn_index.
    assert issuer.dispatch_join_turn.await_count == 0

    # No leaked per-parent state (the gate popped, the slot released on drain).
    assert "root-corr" not in orch._active_joins
    assert "root-corr" not in orch._future_joins
    assert "root-corr" not in orch._descendant_counts
    # parents_resumed counts orchestrator-driven resumes (_release_blocked_join);
    # here the parent resumes via the strategy's normal continuation, so it is
    # not counted (matches v1, which also pops this gate silently).
    assert orch.stats.parents_resumed == 0
    assert orch.stats.children_errored == 2
    assert orch.stats.children_spawned == 0
