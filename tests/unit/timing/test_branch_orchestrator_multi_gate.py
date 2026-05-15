# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2 unit tests: multi-gated branches per spawning turn.

Covers the Phase 2 semantics:

- A single spawning turn may declare multiple gated branches, each with a
  distinct ``gated_turn_index``. The parent suspends separately at each.
- Mixing one background branch with one blocking branch on the same
  spawning turn works without either affecting the other's bookkeeping.
- Partial dispatch failure on one branch rolls back that branch's state
  without corrupting the other branch's pending join.
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


def _mk_credit(conv_id: str, corr_id: str, turn_index: int):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=0,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )


def _k1_k2_k4_metadata() -> list[ConversationMetadata]:
    """Parent with 5 turns; turn 0 spawns three branches gating T+1, T+2, T+4."""
    branch_b1 = ConversationBranchInfo(
        branch_id="root:0:b1",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_b2 = ConversationBranchInfo(
        branch_id="root:0:b2",
        child_conversation_ids=["c2"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_b3 = ConversationBranchInfo(
        branch_id="root:0:b3",
        child_conversation_ids=["c3"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0:b1", "root:0:b2", "root:0:b3"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:b1"
                    )
                ]
            ),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:b2"
                    )
                ]
            ),
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:b3"
                    )
                ]
            ),
        ],
        [branch_b1, branch_b2, branch_b3],
    )
    c1 = _mk_conv("c1", [TurnMetadata()], [])
    c2 = _mk_conv("c2", [TurnMetadata()], [])
    c3 = _mk_conv("c3", [TurnMetadata()], [])
    return [root, c1, c2, c3]


@pytest.mark.asyncio
async def test_multi_gated_branches_per_turn_k1_k2_k3():
    """Turn 0 spawns 3 branches gating at T=1, T=2, T=4. Parent suspends
    separately at each gated turn and resumes when its corresponding child
    completes. Independent pending joins exist per branch."""
    cs = _mk_source(_k1_k2_k4_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0 return: spawns all three children; next turn (T=1) is gated by b1.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    # All three future gates should exist under parent "corr-root"
    # BEFORE promotion strips the T=1 future into active_joins.
    # After promotion: _active_joins has T=1; _future_joins has T=2 and T=4.
    assert orch._active_joins["corr-root"].gated_turn_index == 1
    gate_indices = set(orch._future_joins["corr-root"].keys())
    assert gate_indices == {2, 4}

    # Child c1 completes -> parent resumes for turn 1.
    await orch.on_child_leaf_reached("corr-c1")
    assert issuer.dispatch_join_turn.await_count == 1
    assert orch.stats.parents_resumed == 1
    # After the T=1 gate fires, T=2 and T=4 are still future.
    assert "corr-root" not in orch._active_joins
    assert set(orch._future_joins["corr-root"].keys()) == {2, 4}

    # Turn 1 return: next turn (T=2) is gated by b2.
    assert await orch.intercept(_mk_credit("root", "corr-root", 1)) is True
    assert orch._active_joins["corr-root"].gated_turn_index == 2
    assert set(orch._future_joins["corr-root"].keys()) == {4}

    # Child c2 completes -> parent resumes for turn 2.
    await orch.on_child_leaf_reached("corr-c2")
    assert issuer.dispatch_join_turn.await_count == 2

    # Turn 2 return: next turn T=3 is NOT gated.
    assert await orch.intercept(_mk_credit("root", "corr-root", 2)) is False
    # Turn 3 return: next turn T=4 IS gated by b3.
    assert await orch.intercept(_mk_credit("root", "corr-root", 3)) is True
    assert orch._active_joins["corr-root"].gated_turn_index == 4

    # Child c3 completes -> parent resumes for turn 4.
    await orch.on_child_leaf_reached("corr-c3")
    assert issuer.dispatch_join_turn.await_count == 3
    assert orch.stats.parents_suspended == 3
    assert orch.stats.parents_resumed == 3


@pytest.mark.asyncio
async def test_multi_branch_one_background_one_blocking():
    """One branch is background (fire-and-forget, no gate), the other is
    blocking with a gate at T+1. Parent suspends only for the blocking branch;
    the background child's termination must not interfere with gate state."""
    branch_blocking = ConversationBranchInfo(
        branch_id="root:0:block",
        child_conversation_ids=["c_block"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_bg = ConversationBranchInfo(
        branch_id="root:0:bg",
        child_conversation_ids=["c_bg"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0:block", "root:0:bg"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:block"
                    )
                ]
            ),
        ],
        [branch_blocking, branch_bg],
    )
    c_block = _mk_conv("c_block", [TurnMetadata()], [])
    c_bg = _mk_conv("c_bg", [TurnMetadata()], [])
    cs = _mk_source([root, c_block, c_bg])

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0 return: spawns both children; T=1 is gated by block branch only.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    # Blocking branch promoted to active; no future joins remain (bg ungated).
    active = orch._active_joins["corr-root"]
    assert active.gated_turn_index == 1
    assert orch._future_joins.get("corr-root", {}) == {}

    # Background child completes — must not advance the gate.
    await orch.on_child_leaf_reached("corr-c_bg")
    issuer.dispatch_join_turn.assert_not_called()
    # Gate still active, still unsatisfied.
    assert "corr-root" in orch._active_joins

    # Blocking child completes -> parent resumes.
    await orch.on_child_leaf_reached("corr-c_block")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_suspended == 1
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_multi_branch_rollback_partial_dispatch_failure():
    """One branch's dispatch_first_turn raises; the other branch's gate
    state must be preserved. The failing branch's gate is drained (zero
    outstanding); the surviving branch's gate still blocks the parent."""
    cs = _mk_source(_k1_k2_k4_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()

    # c1 dispatch succeeds; c2 dispatch fails (returns False); c3 succeeds.
    async def _dispatch(session):
        return session.x_correlation_id != "corr-c2"

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0 return: spawns three children; c2 fails to dispatch.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True

    # b1 gate (T=1) was promoted to active.
    assert orch._active_joins["corr-root"].gated_turn_index == 1

    # b3 gate (T=4) still future.
    assert 4 in orch._future_joins.get("corr-root", {})

    # b2 gate (T=2) — c2 was the only child; dispatch rolled back; the gate
    # is now zero-outstanding. _spawn_children_and_register_gates detects
    # the drained gate and dispatches it immediately (Phase 0 hang-fix
    # semantics preserved). T=2 must be gone from _future_joins.
    root_futures = orch._future_joins.get("corr-root", {})
    assert 2 not in root_futures, (
        "b2 gate should have drained after its sole child's dispatch failed"
    )
    # The drained gate fired dispatch_join_turn immediately; resumed count
    # includes b2's forced dispatch.
    assert issuer.dispatch_join_turn.await_count >= 1
    resumed_after_rollback = orch.stats.parents_resumed

    # Surviving branches: c1 still dispatched; c3 still registered.
    assert "corr-c1" in orch._child_to_join
    assert "corr-c3" in orch._child_to_join
    assert "corr-c2" not in orch._child_to_join

    # c1 completes -> parent resumes for turn 1 (b1 gate satisfied).
    await orch.on_child_leaf_reached("corr-c1")
    assert orch.stats.parents_resumed == resumed_after_rollback + 1

    # Parent progresses through T=1, T=2, T=3; b3's gate still future at T=4.
    assert await orch.intercept(_mk_credit("root", "corr-root", 1)) is False
    assert await orch.intercept(_mk_credit("root", "corr-root", 2)) is False
    # Turn 3 return: next is T=4 gated.
    assert await orch.intercept(_mk_credit("root", "corr-root", 3)) is True
    assert orch._active_joins["corr-root"].gated_turn_index == 4

    # c3 completes -> parent resumes for turn 4.
    await orch.on_child_leaf_reached("corr-c3")
    assert orch.stats.parents_resumed == resumed_after_rollback + 2
