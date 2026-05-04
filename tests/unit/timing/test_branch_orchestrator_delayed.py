# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 unit tests for delayed joins in :class:`BranchOrchestrator`.

Covers the delayed-join semantics:

- K>1 delayed joins: parent runs turns [spawn+1 .. gate-1] without suspension
  and suspends only when it's about to dispatch the gated turn.
- Children finishing before the parent arrives pop the future gate and the
  parent breezes through with no suspension.
- K=1 (legacy) behavior still works under the new architecture.
- Stop conditions during the gap propagate to ``joins_suppressed``.
- Fail-fast aborts parent + orphan siblings mid-gap.
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


def _k5_metadata() -> list[ConversationMetadata]:
    """Parent conv with 6 turns: spawn on turn 0, gate on turn 5 (K=5)."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c0", "c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(),
            TurnMetadata(),
            TurnMetadata(),
            TurnMetadata(),
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
    c0 = _mk_conv("c0", [TurnMetadata()], [])
    c1 = _mk_conv("c1", [TurnMetadata()], [])
    return [root, c0, c1]


@pytest.mark.asyncio
async def test_delayed_join_k5_parent_progresses():
    """Spawn at T=0, gate at T=5. Parent returns from turns 0..3 without
    suspension; only turn 4's return (which would dispatch turn 5) triggers
    suspension."""
    cs = _mk_source(_k5_metadata())

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

    # Turn 0 return: spawns children; next turn is 1 (not gated) -> False.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is False
    assert "corr-root" in orch._future_joins
    assert 5 in orch._future_joins["corr-root"]
    assert orch.stats.parents_suspended == 0

    # Turns 1..3 return: no spawns, not next-to-gate, intercept returns False.
    for t in range(1, 4):
        assert await orch.intercept(_mk_credit("root", "corr-root", t)) is False
    assert orch.stats.parents_suspended == 0

    # Turn 4 return: NEXT turn = 5 = gated -> suspend.
    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is True
    assert "corr-root" in orch._active_joins
    assert orch.stats.parents_suspended == 1

    # Children complete -> join fires.
    await orch.on_child_leaf_reached("corr-c0")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-c1")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_delayed_join_children_finish_before_parent_arrives():
    """Children complete before the parent returns from turn 4. When the
    parent reaches turn 4's return (about to dispatch turn 5), the future
    gate is already satisfied -> popped -> intercept returns False."""
    cs = _mk_source(_k5_metadata())

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

    # Turn 0 spawns.
    await orch.intercept(_mk_credit("root", "corr-root", 0))

    # Both children complete before parent returns from turn 4.
    await orch.on_child_leaf_reached("corr-c0")
    await orch.on_child_leaf_reached("corr-c1")

    # Parent now returns from turn 4 -> gate already satisfied -> no suspension.
    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is False
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins
    assert orch.stats.parents_suspended == 0
    # Join never dispatched (children finished on their own path, parent
    # breezes through naturally into turn 5).
    issuer.dispatch_join_turn.assert_not_called()


@pytest.mark.asyncio
async def test_delayed_join_k1_regression_via_new_architecture():
    """K=1 auto-desugared case: spawn on turn 0, gate on turn 1. Parent's
    turn 0 return finds next_idx=1 as gated -> suspends immediately."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c0"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
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
    c0 = _mk_conv("c0", [TurnMetadata()], [])
    cs = _mk_source([root, c0])

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

    # Turn 0 return: spawns child + next turn is 1 (gated) -> True.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    assert orch.stats.parents_suspended == 1

    # Child finishes -> join fires.
    await orch.on_child_leaf_reached("corr-c0")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_delayed_join_stop_condition_fires_during_gap_suppresses_join():
    """If the issuer reports ``dispatch_join_turn`` returned False (stop
    fired), the orchestrator increments ``joins_suppressed`` instead of
    ``parents_resumed``."""
    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    # Stop condition suppresses dispatch_join_turn.
    issuer.dispatch_join_turn = AsyncMock(return_value=False)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 4))  # suspend

    await orch.on_child_leaf_reached("corr-c0")
    await orch.on_child_leaf_reached("corr-c1")

    assert orch.stats.joins_suppressed == 1
    assert orch.stats.parents_resumed == 0


@pytest.mark.asyncio
async def test_delayed_join_fail_fast_aborts_siblings_mid_gap(monkeypatch):
    """With ``AIPERF_DAG_FAIL_FAST=true`` and a child erroring during the
    gap, the parent and every orphan sibling are aborted immediately."""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)

    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock()
    issuer.abort_session = AsyncMock()

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Parent spawns on turn 0 and moves into gap (does NOT suspend yet).
    await orch.intercept(_mk_credit("root", "corr-root", 0))

    # Mid-gap, child c0 errors. Parent + orphan sibling aborted.
    await orch.on_child_errored("corr-c0")
    assert orch.stats.parents_failed_due_to_child_error == 1
    issuer.abort_session.assert_any_await("corr-root")
    issuer.abort_session.assert_any_await("corr-c1")
    assert "corr-root" not in orch._future_joins
    assert "corr-root" not in orch._active_joins


@pytest.mark.asyncio
async def test_delayed_join_multiple_branches_different_k_values_accepted_phase2():
    """Phase 2: declaring two gated branches on the same spawning turn with
    distinct gated_turn_index values is now accepted. The runtime is
    exercised in tests/unit/timing/test_branch_orchestrator_multi_gate.py;
    here we just assert the validator no longer rejects the shape."""
    from aiperf.common.validators.orchestrator_v1 import (
        validate_for_orchestrator_v1,
    )

    branch_a = ConversationBranchInfo(
        branch_id="r:0a",
        child_conversation_ids=["ca"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_b = ConversationBranchInfo(
        branch_id="r:0b",
        child_conversation_ids=["cb"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv = _mk_conv(
        "r",
        [
            TurnMetadata(branch_ids=["r:0a", "r:0b"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0a")
                ]
            ),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0b")
                ]
            ),
        ],
        [branch_a, branch_b],
    )
    ca = _mk_conv("ca", [TurnMetadata()], [])
    cb = _mk_conv("cb", [TurnMetadata()], [])
    md = DatasetMetadata(
        conversations=[conv, ca, cb],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)
