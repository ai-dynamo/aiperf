# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for the BranchOrchestrator state machine."""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import (
    CacheBustTarget,
    ConversationBranchMode,
    PrerequisiteKind,
)
from aiperf.common.environment import Environment
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
from tests.unit.timing._shared_helpers import _mk_issuer


def _mk_conv(
    cid: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo],
    agent_depth: int = 0,
    is_root: bool = True,
) -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=cid,
        turns=turns,
        branches=branches,
        agent_depth=agent_depth,
        is_root=is_root,
    )


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: next(
        c for c in conversations if c.conversation_id == cid
    )

    def _start_branch(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        s.conversation_id = child_conversation_id
        return s

    cs.start_branch_child = MagicMock(side_effect=_start_branch)

    def _start_pre(child_cid, **kwargs):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_cid}"
        s.conversation_id = child_cid
        s.agent_depth = 1
        s.parent_correlation_id = None
        return s

    cs.start_pre_session_child = MagicMock(side_effect=_start_pre)
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


def _fan_in_metadata() -> list[ConversationMetadata]:
    """Reused: turn 0 spawns A (2 children); turn 2 spawns B (3 children); turn 5 gates on both."""
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


@pytest.mark.asyncio
async def test_race_children_complete_before_parent_arrives_pops_silently():
    """All children complete first; parent then arrives at the gated turn."""
    cs = _mk_source(_fan_in_metadata())
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 2))

    for cid in ("a1", "a2", "b1", "b2", "b3"):
        await orch.on_child_leaf_reached(f"corr-{cid}")

    pending_5 = orch._future_joins.get("corr-root", {}).get(5)
    if pending_5 is not None:
        assert pending_5.is_satisfied

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 4))
    assert suspended is False
    issuer.dispatch_join_turn.assert_not_called()
    assert "corr-root" not in orch._active_joins
    assert orch._future_joins.get("corr-root", {}).get(5) is None


@pytest.mark.asyncio
async def test_race_parent_arrives_first_then_last_child_releases():
    """Parent arrives first -> suspended. Last child completes ->"""
    cs = _mk_source(_fan_in_metadata())
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    for t in range(5):
        await orch.intercept(_mk_credit("root", "corr-root", t))
    assert orch._active_joins["corr-root"].gated_turn_index == 5
    issuer.dispatch_join_turn.assert_not_called()

    for cid in ("a1", "a2", "b1", "b2"):
        await orch.on_child_leaf_reached(f"corr-{cid}")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-b3")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_concurrent_intercepts_on_same_parent_serialize():
    """Two ``asyncio.gather``-driven intercept calls on the same parent_corr"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()],
        [branch],
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()

    enter_first = asyncio.Event()
    release_first = asyncio.Event()
    seen_in_progress: list[str] = []

    async def _slow_dispatch(child):
        seen_in_progress.append(f"start-{child.x_correlation_id}")
        if not enter_first.is_set():
            enter_first.set()
            await release_first.wait()
        seen_in_progress.append(f"done-{child.x_correlation_id}")
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_slow_dispatch)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credit = _mk_credit("root", "corr-root", 0)
    t1 = asyncio.create_task(orch.intercept(credit))
    await enter_first.wait()
    t2 = asyncio.create_task(orch.intercept(credit))
    for _ in range(5):
        await asyncio.sleep(0)
    assert seen_in_progress == ["start-corr-c1"]

    release_first.set()
    await asyncio.gather(t1, t2)

    assert seen_in_progress[0] == "start-corr-c1"
    assert seen_in_progress[1] == "done-corr-c1"
    assert seen_in_progress[2] == "start-corr-c1"
    assert seen_in_progress[3] == "done-corr-c1"


@pytest.mark.asyncio
async def test_satisfy_prerequisite_idempotent_under_repeated_delivery():
    """Calling ``_satisfy_prerequisite`` 5x with the same child_corr advances"""
    cs = _mk_source(_fan_in_metadata())
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    for t in range(5):
        await orch.intercept(_mk_credit("root", "corr-root", t))
    assert "corr-root" in orch._active_joins

    state = orch._active_joins["corr-root"].outstanding["SPAWN_JOIN:root:0:A"]
    assert state.expected == 2
    assert len(state.completed) == 0

    for _ in range(5):
        result = await orch._satisfy_prerequisite(
            "corr-root", 5, "SPAWN_JOIN:root:0:A", "corr-a1"
        )
        assert result is None

    assert state.completed == {"corr-a1"}
    assert len(state.completed) == 1
    issuer.dispatch_join_turn.assert_not_called()


@pytest.mark.asyncio
async def test_vacuous_gate_trap_does_not_fire_before_second_branch_registers():
    """Branch_A registers 2 children at spawning turn T=0 and ALL complete"""
    cs = _mk_source(_fan_in_metadata())
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-a1")
    await orch.on_child_leaf_reached("corr-a2")

    pending_5 = orch._future_joins["corr-root"][5]
    a_state = pending_5.outstanding["SPAWN_JOIN:root:0:A"]
    b_state = pending_5.outstanding["SPAWN_JOIN:root:2:B"]
    assert a_state.is_done
    assert not b_state.registered
    assert not pending_5.is_satisfied

    await orch.intercept(_mk_credit("root", "corr-root", 1))
    await orch.intercept(_mk_credit("root", "corr-root", 2))
    await orch.intercept(_mk_credit("root", "corr-root", 3))
    suspended = await orch.intercept(_mk_credit("root", "corr-root", 4))
    assert suspended is True
    issuer.dispatch_join_turn.assert_not_called()

    for cid in ("b1", "b2", "b3"):
        await orch.on_child_leaf_reached(f"corr-{cid}")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_cleanup_during_fail_fast_cascade_no_exception(monkeypatch):
    """Trigger fail-fast then call cleanup; verify no exception, full clear,"""
    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)
    cs = _mk_source(_fan_in_metadata())
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    assert orch._fail_fast is True

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 2))

    await orch.on_child_errored("corr-b2")

    orch.cleanup()
    assert orch._cleaning_up is True
    assert orch._active_joins == {}
    assert orch._future_joins == {}
    assert orch._child_to_join == {}
    assert orch._descendant_counts == {}
    assert orch._pre_dispatched_branches == set()

    orch.cleanup()


def test_cleanup_clears_pre_dispatched_and_logs_leak(caplog):
    cs = _mk_source([])
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())

    pending = PendingBranchJoin(
        parent_x_correlation_id="ghost-parent",
        parent_conversation_id="ghost-conv",
        parent_num_turns=10,
        gated_turn_index=7,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=2, completed=set(), registered=True
    )
    orch._active_joins["ghost-parent"] = pending
    orch._future_joins["ghost-parent"] = {
        9: PendingBranchJoin(
            parent_x_correlation_id="ghost-parent",
            parent_conversation_id="ghost-conv",
            parent_num_turns=10,
            gated_turn_index=9,
        )
    }
    orch._child_to_join["ghost-child"] = [
        ChildJoinEntry(
            parent_correlation_id="ghost-parent",
            gated_turn_index=7,
            prereq_key="SPAWN_JOIN:b",
        )
    ]
    orch._descendant_counts["ghost-parent"] = 3
    orch._pre_dispatched_branches.add(("conv-x", "branch-y"))

    with caplog.at_level(logging.WARNING, logger="aiperf.timing.branch_orchestrator"):
        orch.cleanup()

    leak_warnings = [r for r in caplog.records if "leaked state" in r.getMessage()]
    assert len(leak_warnings) == 1
    abandoned = [
        r for r in caplog.records if "Abandoned pending join" in r.getMessage()
    ]
    assert len(abandoned) >= 2

    assert orch._active_joins == {}
    assert orch._future_joins == {}
    assert orch._child_to_join == {}
    assert orch._descendant_counts == {}
    assert orch._pre_dispatched_branches == set()


def test_has_pending_branch_work_truth_table():
    cs = _mk_source([])
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())

    assert orch.has_pending_branch_work() is False

    orch._active_joins["p"] = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    assert orch.has_pending_branch_work() is True
    orch._active_joins.clear()

    orch._future_joins["p"] = {}
    assert orch.has_pending_branch_work() is False
    orch._future_joins["p"][3] = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=4,
        gated_turn_index=3,
    )
    assert orch.has_pending_branch_work() is True
    orch._future_joins.clear()

    orch._descendant_counts["p"] = 1
    assert orch.has_pending_branch_work() is True
    orch._descendant_counts["p"] = 0
    assert orch.has_pending_branch_work() is False
    orch._descendant_counts.clear()

    orch._child_to_join["c1"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=1, prereq_key="SPAWN_JOIN:b"
        )
    ]
    assert orch.has_pending_branch_work() is True
    orch._child_to_join.clear()

    orch._descendant_counts["p"] = 5
    orch._child_to_join["c1"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=1, prereq_key="SPAWN_JOIN:b"
        )
    ]
    assert orch.has_pending_branch_work() is True


@pytest.mark.asyncio
async def test_k0_self_gate_does_not_infinite_loop():
    """Validator rejects K=0 (gated_turn_idx == spawning_idx) but a buggy"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(
                branch_ids=["root:0"],
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ],
            ),
            TurnMetadata(),
        ],
        [branch],
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    entries = orch._prereq_index.get(("root", 0), [])
    assert any(g == 0 for _, g, _ in entries)

    result = await asyncio.wait_for(
        orch.intercept(_mk_credit("root", "corr-root", 0)),
        timeout=2.0,
    )
    assert result is False
    assert 0 in orch._future_joins.get("corr-root", {})


@pytest.mark.asyncio
async def test_branch_with_empty_children_list_is_graceful():
    """A branch declared with empty children. Validator may or may not"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=[],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()],
        [branch],
    )
    cs = _mk_source([root])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    result = await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert result is False
    assert orch.stats.children_spawned == 0
    assert orch.stats.children_errored == 0
    assert orch._child_to_join == {}
    assert orch._active_joins == {}


def test_duplicate_branch_id_on_same_turn_tolerated_at_orchestrator_layer():
    """The orchestrator no longer asserts on duplicate ``(branch_id,"""
    branch = ConversationBranchInfo(
        branch_id="dup",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["dup"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="dup"),
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="dup"),
                ]
            ),
        ],
        [branch],
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())


@pytest.mark.asyncio
async def test_gated_turn_past_num_turns_does_not_misroute():
    """Parent has 3 turns; prereq targets turn 5. Bypass validator. Verify"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"]),
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
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    orch._prereq_index[("root", 0)] = [("root:0", 5, "SPAWN_JOIN:root:0")]
    orch._gated_turn_prereq_keys[("root", 5)] = {"SPAWN_JOIN:root:0"}

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert 5 in orch._future_joins["corr-root"]

    for t in range(3):
        suspended = await orch.intercept(_mk_credit("root", "corr-root", t))
        assert suspended is False
    assert 5 in orch._future_joins.get("corr-root", {})


@pytest.mark.asyncio
async def test_pre_session_branch_missing_child_logs_and_counts_errored():
    """``start_pre_session_child`` raises (conv_id not in dataset). The"""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["does_not_exist"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()],
        [pre_branch],
    )
    cs = _mk_source([root])

    cs.start_pre_session_child = MagicMock(side_effect=KeyError("does_not_exist"))

    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    assert orch.stats.children_errored == 1
    assert orch.stats.children_spawned == 0
    assert ("root", "root:pre") in orch._pre_dispatched_branches


@pytest.mark.asyncio
async def test_pre_session_dispatch_skips_non_root_conversation():
    """Validator rejects pre on non-root, but bypass: construct"""
    pre_branch = ConversationBranchInfo(
        branch_id="sub:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    sub = _mk_conv(
        "sub",
        [TurnMetadata(branch_ids=["sub:pre"]), TurnMetadata()],
        [pre_branch],
        agent_depth=1,
    )
    early = _mk_conv("early", [TurnMetadata()], [])
    cs = _mk_source([sub, early])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    cs.start_pre_session_child.assert_not_called()
    assert orch.stats.children_spawned == 0
    assert ("sub", "sub:pre") not in orch._pre_dispatched_branches


@pytest.mark.asyncio
async def test_massive_fan_in_100_prereqs_one_gate_fires_exactly_once():
    """100 distinct branches, each spawning 1 child on its own turn, all"""
    N = 100
    branches = [
        ConversationBranchInfo(
            branch_id=f"root:{i}:b",
            child_conversation_ids=[f"c{i}"],
            mode=ConversationBranchMode.SPAWN,
        )
        for i in range(N)
    ]
    spawn_turns = [TurnMetadata(branch_ids=[f"root:{i}:b"]) for i in range(N)]
    gated_turn = TurnMetadata(
        prerequisites=[
            TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id=f"root:{i}:b")
            for i in range(N)
        ]
    )
    root = _mk_conv("root", [*spawn_turns, gated_turn], branches)
    children = [_mk_conv(f"c{i}", [TurnMetadata()], []) for i in range(N)]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    for i in range(N):
        await orch.intercept(_mk_credit("root", "corr-root", i))

    suspended = await orch.intercept(_mk_credit("root", "corr-root", N - 1))
    assert orch._active_joins["corr-root"].gated_turn_index == N
    assert suspended is True

    for i in range(N):
        await orch.on_child_leaf_reached(f"corr-c{i}")

    issuer.dispatch_join_turn.assert_awaited_once()
    assert "corr-root" not in orch._active_joins
    state = orch.stats
    assert state.children_completed == N
    assert state.parents_resumed == 1


@pytest.mark.asyncio
async def test_massive_fan_out_1000_children_no_pathology():
    """One branch with 1000 children; gate at T+1. Verify counter math"""
    N = 1000
    children_ids = [f"c{i}" for i in range(N)]
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=children_ids,
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
    children = [_mk_conv(cid, [TurnMetadata()], []) for cid in children_ids]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    start = time.monotonic()
    suspended = await orch.intercept(_mk_credit("root", "corr-root", 0))
    spawn_time = time.monotonic() - start
    assert suspended is True
    assert spawn_time < 10.0, f"spawning 1000 children took {spawn_time:.2f}s"

    state = orch._active_joins["corr-root"].outstanding["SPAWN_JOIN:root:0"]
    assert state.expected == N

    start = time.monotonic()
    for cid in children_ids:
        await orch.on_child_leaf_reached(f"corr-{cid}")
    completion_time = time.monotonic() - start
    assert completion_time < 10.0, (
        f"completing 1000 children took {completion_time:.2f}s"
    )

    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_multi_consumer_single_branch_three_gates_all_advance():
    """Branch on turn 0 referenced by SPAWN_JOIN on turns 1, 2, 3."""
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

    entries = orch._child_to_join["corr-c1"]
    assert len(entries) == 3
    gated_idxs = {e.gated_turn_index for e in entries}
    assert gated_idxs == {1, 2, 3}

    await orch.on_child_leaf_reached("corr-c1")
    assert issuer.dispatch_join_turn.await_count == 1
    assert "corr-root" not in orch._active_joins
    assert orch._future_joins.get("corr-root", {}) == {}

    assert await orch.intercept(_mk_credit("root", "corr-root", 1)) is False
    assert await orch.intercept(_mk_credit("root", "corr-root", 2)) is False


@pytest.mark.asyncio
async def test_multi_consumer_fail_fast_aborts_parent_and_drops_all_gates(
    monkeypatch,
):
    """Phase 3: same branch feeds 3 gates; child errors with fail-fast."""
    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1", "c2"],
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
    cs = _mk_source(
        [
            root,
            _mk_conv("c1", [TurnMetadata()], []),
            _mk_conv("c2", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert "corr-root" in orch._active_joins
    assert set(orch._future_joins["corr-root"].keys()) == {2, 3}

    await orch.on_child_errored("corr-c1")
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins
    aborted = {call.args[0] for call in issuer.abort_session.await_args_list}
    assert "corr-root" in aborted
    assert "corr-c2" in aborted
    assert orch.stats.parents_failed_due_to_child_error == 1


@pytest.mark.asyncio
async def test_stop_condition_during_delayed_join_increments_joins_suppressed():
    """When the strategy declines to dispatch the gated turn (issuer returns"""
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
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    issuer.dispatch_join_turn = AsyncMock(return_value=False)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert orch._active_joins["corr-root"].gated_turn_index == 1

    await orch.on_child_leaf_reached("corr-c1")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 0
    assert orch.stats.joins_suppressed == 1


@pytest.mark.asyncio
async def test_fail_fast_cascade_drops_all_future_gates(monkeypatch):
    """Two SPAWNs from turn 0 each registering at different gates (T=2 and"""
    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)
    branch_a = ConversationBranchInfo(
        branch_id="root:0:A",
        child_conversation_ids=["a1"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_b = ConversationBranchInfo(
        branch_id="root:0:B",
        child_conversation_ids=["b1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0:A", "root:0:B"]),
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:A"
                    )
                ]
            ),
            TurnMetadata(),
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:B"
                    )
                ]
            ),
        ],
        [branch_a, branch_b],
    )
    cs = _mk_source(
        [
            root,
            _mk_conv("a1", [TurnMetadata()], []),
            _mk_conv("b1", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspended_0 = await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert suspended_0 is False
    assert set(orch._future_joins["corr-root"].keys()) == {2, 5}

    suspended_1 = await orch.intercept(_mk_credit("root", "corr-root", 1))
    assert suspended_1 is True
    assert orch._active_joins["corr-root"].gated_turn_index == 2
    assert 5 in orch._future_joins["corr-root"]

    await orch.on_child_errored("corr-a1")
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins
    aborted = {call.args[0] for call in issuer.abort_session.await_args_list}
    assert {"corr-root", "corr-b1"} <= aborted


@pytest.mark.asyncio
async def test_same_parent_corr_for_two_conversations_state_does_not_clobber():
    """Two distinct conversations sharing the same parent_correlation_id is"""
    branch_x = ConversationBranchInfo(
        branch_id="X:0",
        child_conversation_ids=["xc"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_y = ConversationBranchInfo(
        branch_id="Y:0",
        child_conversation_ids=["yc"],
        mode=ConversationBranchMode.SPAWN,
    )
    convx = _mk_conv(
        "convX",
        [
            TurnMetadata(branch_ids=["X:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="X:0")
                ]
            ),
        ],
        [branch_x],
    )
    convy = _mk_conv(
        "convY",
        [
            TurnMetadata(branch_ids=["Y:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="Y:0")
                ]
            ),
        ],
        [branch_y],
    )
    cs = _mk_source(
        [
            convx,
            convy,
            _mk_conv("xc", [TurnMetadata()], []),
            _mk_conv("yc", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    SHARED = "shared-corr"
    await orch.intercept(_mk_credit("convX", SHARED, 0))
    assert orch._active_joins[SHARED].parent_conversation_id == "convX"

    await orch.intercept(_mk_credit("convY", SHARED, 0))
    assert "corr-xc" in orch._child_to_join or "corr-yc" in orch._child_to_join


@pytest.mark.asyncio
async def test_cleanup_mid_intercept_no_deadlock():
    """One task is mid-intercept holding the parent lock. Another task is"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()],
        [branch],
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()

    block_event = asyncio.Event()

    async def _slow_dispatch(child):
        await block_event.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_slow_dispatch)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    credit = _mk_credit("root", "corr-root", 0)
    t1 = asyncio.create_task(orch.intercept(credit))
    for _ in range(5):
        await asyncio.sleep(0)

    orch.cleanup()
    assert orch._cleaning_up is True

    block_event.set()
    await asyncio.wait_for(t1, timeout=2.0)

    result2 = await orch.intercept(credit)
    assert result2 is False


@pytest.mark.asyncio
async def test_satisfy_prerequisite_orphan_child_logs_warn_no_exception(caplog):
    """``_satisfy_prerequisite`` for a prereq_key not in pending.outstanding"""
    cs = _mk_source(_fan_in_metadata())
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    with caplog.at_level(logging.WARNING, logger="aiperf.timing.branch_orchestrator"):
        result = await orch._satisfy_prerequisite(
            "corr-root", 5, "SPAWN_JOIN:does:not:exist", "ghost-child"
        )
    assert result is None
    assert any("not registered on join" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_satisfy_prerequisite_unknown_parent_logs_warn_no_exception(caplog):
    """``_satisfy_prerequisite`` for a parent_corr with no join must log a"""
    cs = _mk_source([])
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())
    with caplog.at_level(logging.WARNING, logger="aiperf.timing.branch_orchestrator"):
        result = await orch._satisfy_prerequisite(
            "no-such-parent", 1, "SPAWN_JOIN:b", "ghost"
        )
    assert result is None
    assert any("no join found" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_mixed_fork_spawn_fan_in_partial_completion_releases_fork_sticky():
    """Branch A is FORK (2 children), branch B is SPAWN (2 children); both"""
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
    cs = _mk_source(
        [
            root,
            *[_mk_conv(c, [TurnMetadata()], []) for c in ("f1", "f2", "s1", "s2")],
        ]
    )
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert sticky.register_child_routing.call_count == 2
    await orch.intercept(_mk_credit("root", "corr-root", 1))
    assert sticky.register_child_routing.call_count == 2

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 2))
    assert suspended is True

    await orch.on_child_leaf_reached("corr-f1")
    await orch.on_child_leaf_reached("corr-f2")
    issuer.dispatch_join_turn.assert_not_called()
    assert sticky.release_child_routing.call_count == 2

    await orch.on_child_leaf_reached("corr-s1")
    await orch.on_child_leaf_reached("corr-s2")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert sticky.release_child_routing.call_count == 2


@pytest.mark.asyncio
async def test_pre_session_child_runs_its_own_second_level_dag():
    """A pre-session SPAWN child is itself a conversation with its own"""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["middle"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()],
        [pre_branch],
    )
    middle_branch = ConversationBranchInfo(
        branch_id="middle:0",
        child_conversation_ids=["leaf"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    middle = _mk_conv(
        "middle",
        [TurnMetadata(branch_ids=["middle:0"]), TurnMetadata()],
        [middle_branch],
        agent_depth=1,
    )
    leaf = _mk_conv("leaf", [TurnMetadata()], [], agent_depth=2)
    cs = _mk_source([root, middle, leaf])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    cs.start_pre_session_child.assert_called_once_with(
        "middle", cache_bust_marker=None, cache_bust_target=CacheBustTarget.NONE
    )
    assert issuer.dispatch_first_turn.await_count == 1

    pre_credit = MagicMock(
        x_correlation_id="corr-middle",
        conversation_id="middle",
        turn_index=0,
        agent_depth=1,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.SPAWN,
    )
    await orch.intercept(pre_credit)
    leaf_calls = [
        call
        for call in cs.start_branch_child.call_args_list
        if call.kwargs.get("child_conversation_id") == "leaf"
    ]
    assert len(leaf_calls) == 1, (
        f"second-level branch from a pre-session child must dispatch its "
        f"grand-child via intercept; got {len(leaf_calls)} calls"
    )
    assert leaf_calls[0].kwargs["agent_depth"] == 2
    assert leaf_calls[0].kwargs["parent_correlation_id"] == "corr-middle"


@pytest.mark.asyncio
async def test_pre_session_skips_when_both_belts_fail_simultaneously():
    """Both ``is_root=False`` AND ``agent_depth>0`` at once must skip."""
    pre_branch = ConversationBranchInfo(
        branch_id="bad:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    bad = _mk_conv(
        "bad",
        [TurnMetadata(branch_ids=["bad:pre"]), TurnMetadata()],
        [pre_branch],
        agent_depth=3,
        is_root=False,
    )
    early = _mk_conv("early", [TurnMetadata()], [])
    cs = _mk_source([bad, early])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()

    cs.start_pre_session_child.assert_not_called()
    issuer.dispatch_first_turn.assert_not_called()
    assert orch.stats.children_spawned == 0


@pytest.mark.asyncio
async def test_pre_session_dispatch_all_non_root_dataset_is_noop():
    """A dataset entirely composed of non-root conversations (e.g. an"""
    pre_branch = ConversationBranchInfo(
        branch_id="c1:pre",
        child_conversation_ids=["target"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    c1 = _mk_conv(
        "c1",
        [TurnMetadata(branch_ids=["c1:pre"]), TurnMetadata()],
        [pre_branch],
        is_root=False,
    )
    c2 = _mk_conv("c2", [TurnMetadata()], [], is_root=False, agent_depth=2)
    target = _mk_conv("target", [TurnMetadata()], [], is_root=False)
    cs = _mk_source([c1, c2, target])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()

    cs.start_pre_session_child.assert_not_called()
    issuer.dispatch_first_turn.assert_not_called()
    assert not orch._pre_dispatched_branches


@pytest.mark.asyncio
async def test_pre_session_mixed_roots_only_root_pre_fires():
    """A dataset mixing one root (with a pre branch) and several non-root"""
    root_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["child_a"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    rogue_branch = ConversationBranchInfo(
        branch_id="rogue:pre",
        child_conversation_ids=["child_b"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()],
        [root_branch],
    )
    rogue = _mk_conv(
        "rogue",
        [TurnMetadata(branch_ids=["rogue:pre"]), TurnMetadata()],
        [rogue_branch],
        is_root=False,
    )
    child_a = _mk_conv("child_a", [TurnMetadata()], [], is_root=False)
    child_b = _mk_conv("child_b", [TurnMetadata()], [], is_root=False)
    cs = _mk_source([root, rogue, child_a, child_b])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()

    cs.start_pre_session_child.assert_called_once_with(
        "child_a", cache_bust_marker=None, cache_bust_target=CacheBustTarget.NONE
    )
    cs.start_pre_session_child.assert_called_once()
    assert ("root", "root:pre") in orch._pre_dispatched_branches
    assert ("rogue", "rogue:pre") not in orch._pre_dispatched_branches
