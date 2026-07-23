# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 unit tests: fan-in (multi-prereq per gated turn)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.timing.branch_orchestrator import (
    BranchOrchestrator,
    PendingBranchJoin,
    PrereqState,
)
from tests.unit.timing._shared_helpers import _mk_conv, _mk_issuer, _mk_source


def _mk_credit(conv_id: str, corr_id: str, turn_index: int):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=0,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )


def _fan_in_metadata() -> list[ConversationMetadata]:
    """Parent has 6 turns. Turn 0 spawns branch_A (2 children); turn 2 spawns"""
    branch_a = ConversationBranchInfo(
        branch_id="root:0:A",
        child_conversation_ids=["a1", "a2"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_b = ConversationBranchInfo(
        branch_id="root:2:B",
        child_conversation_ids=["b1", "b2", "b3"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0:A"]),
            TurnMetadata(),
            TurnMetadata(branch_ids=["root:2:B"]),
            TurnMetadata(),
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:A"
                    ),
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:2:B"
                    ),
                ]
            ),
        ],
        [branch_a, branch_b],
    )
    children = [
        _mk_conv(cid, [TurnMetadata()], []) for cid in ("a1", "a2", "b1", "b2", "b3")
    ]
    return [root, *children]


def _mk_start(cs):
    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)


@pytest.mark.asyncio
async def test_fan_in_two_spawn_points_single_gate():
    """Turn 0 spawns A (2 children); turn 2 spawns B (3 children); turn 5"""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is False
    pending_5 = orch._future_joins["corr-root"][5]
    a_state = pending_5.outstanding["SPAWN_JOIN:root:0:A"]
    assert a_state.expected == 2
    assert a_state.registered is True
    b_state = pending_5.outstanding["SPAWN_JOIN:root:2:B"]
    assert b_state.expected == 0
    assert b_state.registered is False
    assert not pending_5.is_satisfied

    assert await orch.intercept(_mk_credit("root", "corr-root", 1)) is False

    assert await orch.intercept(_mk_credit("root", "corr-root", 2)) is False
    pending_5 = orch._future_joins["corr-root"][5]
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].expected == 2
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].expected == 3

    assert await orch.intercept(_mk_credit("root", "corr-root", 3)) is False
    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is True
    assert orch._active_joins["corr-root"].gated_turn_index == 5
    issuer.dispatch_join_turn.assert_not_called()

    await orch.on_child_leaf_reached("corr-a1")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-a2")
    issuer.dispatch_join_turn.assert_not_called()

    await orch.on_child_leaf_reached("corr-b1")
    await orch.on_child_leaf_reached("corr-b2")
    issuer.dispatch_join_turn.assert_not_called()

    await orch.on_child_leaf_reached("corr-b3")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert "corr-root" not in orch._active_joins
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_fan_in_partial_satisfy_then_full_satisfy():
    """All A children complete before parent suspends; B still has one child"""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-a1")
    await orch.on_child_leaf_reached("corr-a2")
    pending_5 = orch._future_joins["corr-root"][5]
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].is_done

    await orch.intercept(_mk_credit("root", "corr-root", 1))
    await orch.intercept(_mk_credit("root", "corr-root", 2))
    pending_5 = orch._future_joins["corr-root"][5]
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].is_done
    assert not pending_5.outstanding["SPAWN_JOIN:root:2:B"].is_done

    await orch.intercept(_mk_credit("root", "corr-root", 3))
    await orch.on_child_leaf_reached("corr-b1")
    await orch.on_child_leaf_reached("corr-b2")
    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is True
    issuer.dispatch_join_turn.assert_not_called()

    await orch.on_child_leaf_reached("corr-b3")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_fan_in_three_way_with_fork_and_spawn_mixed():
    """Mix FORK and SPAWN: turn 0 spawns FORK branch F (2 children); turn 1"""
    branch_f = ConversationBranchInfo(
        branch_id="root:0:F",
        child_conversation_ids=["f1", "f2"],
        mode=ConversationBranchMode.FORK,
    )
    branch_s = ConversationBranchInfo(
        branch_id="root:1:S",
        child_conversation_ids=["s1", "s2"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0:F"], has_forks=True),
            TurnMetadata(branch_ids=["root:1:S"]),
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:F"
                    ),
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:1:S"
                    ),
                ]
            ),
        ],
        [branch_f, branch_s],
    )
    children = [_mk_conv(cid, [TurnMetadata()], []) for cid in ("f1", "f2", "s1", "s2")]
    cs = _mk_source([root, *children])
    _mk_start(cs)
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert sticky.register_child_routing.call_count == 2
    await orch.intercept(_mk_credit("root", "corr-root", 1))
    assert sticky.register_child_routing.call_count == 2

    assert await orch.intercept(_mk_credit("root", "corr-root", 2)) is True

    for cid in ("f1", "f2"):
        await orch.on_child_leaf_reached(f"corr-{cid}")
    issuer.dispatch_join_turn.assert_not_called()
    assert sticky.release_child_routing.call_count == 2

    for cid in ("s1", "s2"):
        await orch.on_child_leaf_reached(f"corr-{cid}")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert sticky.release_child_routing.call_count == 2


@pytest.mark.asyncio
async def test_fan_in_idempotent_on_double_delivery():
    """Calling _satisfy_prerequisite twice for the same child_corr on the"""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 1))
    await orch.intercept(_mk_credit("root", "corr-root", 2))
    await orch.intercept(_mk_credit("root", "corr-root", 3))
    await orch.intercept(_mk_credit("root", "corr-root", 4))
    assert "corr-root" in orch._active_joins

    await orch.on_child_leaf_reached("corr-a1")
    await orch.on_child_leaf_reached("corr-a2")
    await orch.on_child_leaf_reached("corr-b1")
    result = await orch._satisfy_prerequisite(
        "corr-root", 5, "SPAWN_JOIN:root:2:B", "corr-b1"
    )
    assert result is None, "duplicate delivery must return None"
    state = orch._active_joins["corr-root"].outstanding["SPAWN_JOIN:root:2:B"]
    assert len(state.completed) == 1
    issuer.dispatch_join_turn.assert_not_called()

    await orch.on_child_leaf_reached("corr-b2")
    await orch.on_child_leaf_reached("corr-b3")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_fan_in_under_fail_fast_cascades_correctly(monkeypatch):
    """AIPERF_DAG_FAIL_FAST=true: one B child errors. Parent + every orphan"""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    assert orch._fail_fast is True

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 1))
    await orch.intercept(_mk_credit("root", "corr-root", 2))
    assert {f"corr-{c}" for c in ("a1", "a2", "b1", "b2", "b3")} <= set(
        orch._child_to_join.keys()
    )

    await orch.on_child_errored("corr-b2")
    issuer.abort_session.assert_any_await("corr-root")
    aborted = {call.args[0] for call in issuer.abort_session.await_args_list}
    assert {"corr-a1", "corr-a2", "corr-b1", "corr-b3"} <= aborted
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins
    assert orch.stats.parents_failed_due_to_child_error == 1


@pytest.mark.asyncio
async def test_fan_in_rollback_decrements_expected_not_completed():
    """A partial dispatch failure for one branch feeding a fan-in gate"""
    cs = _mk_source(_fan_in_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)
    issuer = MagicMock()

    async def _dispatch(session):
        return session.x_correlation_id != "corr-b2"

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 2))

    pending_5 = orch._future_joins["corr-root"][5]
    a_state = pending_5.outstanding["SPAWN_JOIN:root:0:A"]
    assert a_state.expected == 2
    assert a_state.completed == set()
    b_state = pending_5.outstanding["SPAWN_JOIN:root:2:B"]
    assert b_state.expected == 2
    assert b_state.completed == set()

    assert "corr-b2" not in orch._child_to_join
    assert "corr-b1" in orch._child_to_join
    assert "corr-b3" in orch._child_to_join


@pytest.mark.asyncio
async def test_fan_in_same_turn_gates_dont_collide_across_branches():
    """Different branches contribute to the same ``gated_turn_index``."""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    pending_5 = orch._future_joins["corr-root"][5]
    assert set(pending_5.outstanding) == {
        "SPAWN_JOIN:root:0:A",
        "SPAWN_JOIN:root:2:B",
    }
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].expected == 2
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].registered is True
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].expected == 0
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].registered is False

    await orch.intercept(_mk_credit("root", "corr-root", 2))
    pending_5 = orch._future_joins["corr-root"][5]
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].expected == 2
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].expected == 3
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].registered is True


@pytest.mark.asyncio
async def test_is_satisfied_empty_gate_is_true():
    """A PendingBranchJoin with no prereqs is trivially satisfied (vacuous"""
    p = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    assert p.is_satisfied


@pytest.mark.asyncio
async def test_prereq_state_is_done_semantics():
    """PrereqState.is_done: registered AND len(completed) >= expected."""
    s = PrereqState(expected=3, completed=set(), registered=True)
    assert not s.is_done
    s.completed.add("a")
    s.completed.add("b")
    assert not s.is_done
    s.completed.add("c")
    assert s.is_done
    s.completed.add("d")
    assert s.is_done
    unreg = PrereqState(expected=0, registered=False)
    assert not unreg.is_done


@pytest.mark.asyncio
async def test_fan_in_multi_consumer_same_branch_multiple_gates():
    """Phase 3: a single branch feeding prereqs on two different gated"""
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
        ],
        [branch],
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    entries = orch._prereq_index[("root", 0)]
    gated_idxs = {gated_idx for _, gated_idx, _ in entries}
    assert gated_idxs == {1, 2}

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    assert orch._active_joins["corr-root"].gated_turn_index == 1
    assert 2 in orch._future_joins["corr-root"]


def test_pending_branch_join_outstanding_is_prereq_state_shape():
    """Shape regression: PendingBranchJoin.outstanding values are"""
    p = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    p.outstanding["SPAWN_JOIN:b"] = PrereqState(expected=1, registered=True)
    assert isinstance(p.outstanding["SPAWN_JOIN:b"], PrereqState)


@pytest.mark.asyncio
async def test_fan_in_child_to_join_entry_points_at_single_gate_per_child():
    """A child that contributes to a fan-in gate has ONE ChildJoinEntry"""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 2))

    assert isinstance(orch._child_to_join["corr-a1"], list)
    assert len(orch._child_to_join["corr-a1"]) == 1
    assert orch._child_to_join["corr-a1"][0].prereq_key == "SPAWN_JOIN:root:0:A"
    assert orch._child_to_join["corr-a1"][0].gated_turn_index == 5
    assert orch._child_to_join["corr-b1"][0].prereq_key == "SPAWN_JOIN:root:2:B"
    assert orch._child_to_join["corr-b1"][0].gated_turn_index == 5


@pytest.mark.asyncio
async def test_snapshot_annotations_preserves_multi_consumer_gate_memberships():
    """Handoff snapshot must keep every gate a multi-consumer child feeds,"""
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
        ],
        [branch],
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    assert len(orch._child_to_join["corr-c1"]) == 2

    blocked, children = orch.snapshot_annotations()
    assert children["corr-c1"] == [("root:0", 1), ("root:0", 2)]
    assert blocked["corr-root"] == 1


@pytest.mark.asyncio
async def test_seed_snapshot_re_registers_all_multi_gate_memberships():
    """seed_snapshot must restore every join_gate_memberships entry so a"""
    from aiperf.timing.trajectory_source import ConversationState

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
        ],
        [branch],
    )
    child = _mk_conv("c1", [TurnMetadata(), TurnMetadata()], [])
    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    orch.seed_snapshot(
        (
            ConversationState(
                conversation_id="root",
                x_correlation_id="corr-root",
                next_turn_index=1,
                waiting_on_children=True,
                join_target_turn_index=1,
                root_correlation_id="corr-root",
            ),
            ConversationState(
                conversation_id="c1",
                x_correlation_id="corr-c1",
                next_turn_index=0,
                agent_depth=1,
                parent_correlation_id="corr-root",
                root_correlation_id="corr-root",
                branch_id="root:0",
                join_target_turn_index=1,
                join_gate_memberships=(("root:0", 1), ("root:0", 2)),
                branch_mode=ConversationBranchMode.SPAWN,
            ),
        )
    )

    entries = orch._child_to_join["corr-c1"]
    assert [(e.prereq_key, e.gated_turn_index) for e in entries] == [
        ("SPAWN_JOIN:root:0", 1),
        ("SPAWN_JOIN:root:0", 2),
    ]
    assert orch._active_joins["corr-root"].gated_turn_index == 1
    assert 2 in orch._future_joins["corr-root"]
    assert (
        orch._active_joins["corr-root"].outstanding["SPAWN_JOIN:root:0"].expected == 1
    )
    assert (
        orch._future_joins["corr-root"][2].outstanding["SPAWN_JOIN:root:0"].expected
        == 1
    )
