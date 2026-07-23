# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial probes for the DAG ``BranchOrchestrator``."""

from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import (
    BranchOrchestrator,
    ChildJoinEntry,
    PendingBranchJoin,
    PrereqState,
)
from aiperf.timing.trajectory_source import ConversationState
from tests.unit.timing._shared_helpers import _mk_issuer


def _mk_conv(
    cid: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo],
    agent_depth: int = 0,
) -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=cid,
        turns=turns,
        branches=branches,
        agent_depth=agent_depth,
    )


def _mk_source(conversations: list[ConversationMetadata], *, unique_children=False):
    """Build a MagicMock conversation source."""
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: next(
        c for c in conversations if c.conversation_id == cid
    )
    counter = itertools.count()

    def _start_branch(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        suffix = f"-{next(counter)}" if unique_children else ""
        s.x_correlation_id = f"corr-{child_conversation_id}{suffix}"
        s.conversation_id = child_conversation_id
        return s

    cs.start_branch_child = MagicMock(side_effect=_start_branch)
    return cs


def _mk_credit(conv_id: str, corr_id: str, turn_index: int, agent_depth: int = 0):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=agent_depth,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )


def _delayed_join_root() -> ConversationMetadata:
    """Root: branch spawns on turn 0, join gated on turn 3 (K=3, delayed)."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    return _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"]),
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


def _seed_fork_conv():
    """Parent (3 turns) + one FORK child gating turn 2, for seed_snapshot."""
    branch_id = "b0"
    parent_meta = ConversationMetadata(
        conversation_id="parent",
        turns=[
            TurnMetadata(),
            TurnMetadata(branch_ids=[branch_id], has_forks=True),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id=branch_id
                    )
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id=branch_id,
                child_conversation_ids=["child"],
                mode=ConversationBranchMode.FORK,
                start_timestamp_ms=13000.0,
            )
        ],
    )
    child_meta = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(), TurnMetadata()],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="parent",
    )

    class _Source:
        dataset_metadata = DatasetMetadata(
            conversations=[parent_meta, child_meta],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        def get_metadata(self, conversation_id):
            return {"parent": parent_meta, "child": child_meta}[conversation_id]

    return _Source(), branch_id


@pytest.mark.asyncio
async def test_delayed_join_all_children_raise_does_not_dispatch_gate_early():
    """The branch spawns on turn 0; the join is gated on turn 3. Every"""
    cs = _mk_source([_delayed_join_root(), _mk_conv("c1", [TurnMetadata()], [])])
    cs.start_branch_child = MagicMock(side_effect=RuntimeError("start failed"))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert suspended is False
    issuer.dispatch_join_turn.assert_not_called()


@pytest.mark.asyncio
async def test_delayed_join_all_children_refused_does_not_dispatch_gate_early():
    """Identical ordering invariant to test 1, exercised through the"""
    cs = _mk_source([_delayed_join_root(), _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    issuer.dispatch_first_turn = AsyncMock(return_value=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert suspended is False
    issuer.dispatch_join_turn.assert_not_called()


@pytest.mark.asyncio
async def test_seed_snapshot_fork_child_sticky_release_is_balanced():
    """A snapshot-replayed FORK child must have balanced sticky refcount"""
    source, branch_id = _seed_fork_conv()
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=source, credit_issuer=issuer, sticky_router=sticky
    )
    states = (
        ConversationState(
            conversation_id="parent",
            x_correlation_id="parent-corr",
            next_turn_index=2,
            waiting_on_children=True,
            join_target_turn_index=2,
        ),
        ConversationState(
            conversation_id="child",
            x_correlation_id="child-corr",
            next_turn_index=1,
            agent_depth=1,
            parent_correlation_id="parent-corr",
            join_target_turn_index=2,
            branch_id=branch_id,
            branch_mode=ConversationBranchMode.FORK,
        ),
    )
    orch.seed_snapshot(states)

    await orch.on_child_leaf_reached("child-corr")

    assert (
        sticky.release_child_routing.call_count
        == sticky.register_child_routing.call_count
    )


@pytest.mark.asyncio
async def test_duplicate_spawning_turn_credit_double_dispatches_children():
    """The orchestrator has no per-turn idempotency guard: delivering the"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
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
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])], unique_children=True)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 0))

    assert cs.start_branch_child.call_count == 2
    assert issuer.dispatch_first_turn.await_count == 2
    assert len(orch._child_to_join) == 2


@pytest.mark.asyncio
async def test_background_branch_all_children_refused_drains_clean():
    """A background (ungated) branch whose children are ALL refused dispatch"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1", "c2"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()],
        [branch],
    )
    cs = _mk_source(
        [
            root,
            _mk_conv("c1", [TurnMetadata()], []),
            _mk_conv("c2", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    issuer.dispatch_first_turn = AsyncMock(return_value=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert suspended is False
    assert orch.stats.children_spawned == 0
    assert orch.stats.children_truncated == 2
    assert orch.stats.children_errored == 0
    assert orch._descendant_counts == {}
    assert orch._child_to_join == {}
    assert orch.has_pending_branch_work() is False


@pytest.mark.asyncio
async def test_drain_observer_fires_when_last_ungated_child_drains():
    """The drain observer is the race-closing hook the callback handler relies"""
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    observed: list[bool] = []
    orch.set_drain_observer(lambda: observed.append(orch.has_pending_branch_work()))

    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="parent", gated_turn_index=None, prereq_key=None
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.SPAWN}
    orch._descendant_counts["parent"] = 1

    await orch.on_child_leaf_reached("cA")

    assert observed, "drain observer must fire on the draining completion"
    assert observed[-1] is False
    assert orch.has_pending_branch_work() is False


@pytest.mark.asyncio
async def test_leaf_then_error_double_delivery_counts_child_once(force_fail_fast):
    """A worker that delivers BOTH a leaf and an error for the same child must"""
    force_fail_fast(True)
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=issuer, sticky_router=sticky
    )
    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=1, completed=set(), registered=True
    )
    pending.is_blocked = True
    orch._active_joins["p"] = pending
    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=1, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.SPAWN}
    orch._descendant_counts["p"] = 1

    await orch.on_child_leaf_reached("cA")
    await orch.on_child_errored("cA")

    assert orch.stats.children_completed == 1
    assert orch.stats.children_errored == 0
    issuer.abort_session.assert_not_awaited()
    assert orch.stats.parents_failed_due_to_child_error == 0
    assert issuer.dispatch_join_turn.await_count == 1


@pytest.mark.asyncio
async def test_stopped_then_error_double_delivery_counts_child_once():
    """A child first cap-stopped (truncated) then later erroring must be"""
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=MagicMock(), credit_issuer=issuer)
    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=None, prereq_key=None
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.SPAWN}
    orch._descendant_counts["p"] = 1

    await orch.on_child_stopped("cA")
    await orch.on_child_errored("cA")

    assert orch.stats.children_truncated == 1
    assert orch.stats.children_errored == 0


@pytest.mark.asyncio
async def test_seed_snapshot_orphan_child_without_parent_state_drains_clean():
    """When a snapshot includes a child whose parent state is absent (parent"""
    child_meta = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(), TurnMetadata()],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="parent",
    )

    class _Source:
        dataset_metadata = DatasetMetadata(
            conversations=[child_meta],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        def get_metadata(self, conversation_id):
            return {"child": child_meta}[conversation_id]

    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=_Source(), credit_issuer=issuer)
    states = (
        ConversationState(
            conversation_id="child",
            x_correlation_id="child-corr",
            next_turn_index=1,
            agent_depth=1,
            parent_correlation_id="parent-corr",
            join_target_turn_index=1,
            branch_id="b0",
            branch_mode=ConversationBranchMode.SPAWN,
        ),
    )
    orch.seed_snapshot(states)

    entries = orch._child_to_join["child-corr"]
    assert len(entries) == 1
    assert entries[0].prereq_key is None
    assert orch._descendant_counts["parent-corr"] == 1
    assert orch.has_pending_branch_work() is True

    await orch.on_child_leaf_reached("child-corr")

    issuer.dispatch_join_turn.assert_not_called()
    assert "parent-corr" not in orch._descendant_counts
    assert orch.has_pending_branch_work() is False


@pytest.mark.asyncio
async def test_over_completed_prereq_total_outstanding_clamped_non_negative():
    """If more distinct children report against a prereq than its ``expected``"""
    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=1, completed={"cA", "cB"}, registered=True
    )

    assert pending.total_outstanding == 0
    assert pending.outstanding["SPAWN_JOIN:b"].is_done is True
    assert pending.is_satisfied is True


@pytest.mark.asyncio
async def test_cleanup_mid_drain_then_late_child_is_noop():
    """Cleanup is called while children are still tracked (DAG abandoned, e.g."""
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=MagicMock(), credit_issuer=issuer)
    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=2, completed=set(), registered=True
    )
    orch._active_joins["p"] = pending
    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=2, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.SPAWN}
    orch._descendant_counts["p"] = 2

    orch.cleanup()
    assert orch.has_pending_branch_work() is False

    await orch.on_child_leaf_reached("cA")
    issuer.dispatch_join_turn.assert_not_called()
    assert orch.stats.children_completed == 0


@pytest.mark.asyncio
async def test_non_fail_fast_error_on_sole_child_fires_all_gates_once(
    force_fail_fast,
):
    """A single SPAWN child feeds three gated turns (1, 2, 3). The child errors"""
    force_fail_fast(False)
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
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
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
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
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert suspended is True
    assert orch._active_joins["corr-root"].gated_turn_index == 1
    assert set(orch._future_joins["corr-root"].keys()) == {2, 3}

    await orch.on_child_errored("corr-c1")

    assert orch.stats.children_errored == 1
    assert issuer.dispatch_join_turn.await_count == 1
    assert "corr-root" not in orch._active_joins
    assert orch._future_joins.get("corr-root", {}) == {}
