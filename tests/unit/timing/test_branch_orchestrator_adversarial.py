# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for :class:`BranchOrchestrator`."""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
)
from aiperf.common.models import (
    ConversationBranchInfo,
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
from tests.unit.timing._shared_helpers import _mk_conv, _mk_source


def test_orchestrator_index_empty_on_empty_dataset_metadata():
    """Empty DatasetMetadata.conversations -> empty _prereq_index."""
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=[],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())
    assert orch._prereq_index == {}


def test_orchestrator_index_ignores_branches_not_consumed_by_any_prereq():
    """A declared branch with no SPAWN_JOIN prereq consuming it is absent"""
    branch = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.FORK,
    )
    conv = _mk_conv(
        "r",
        [TurnMetadata(branch_ids=["r:0"]), TurnMetadata()],
        [branch],
    )
    cs = _mk_source([conv])
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())
    assert orch._prereq_index == {}


def test_orchestrator_index_keys_by_conv_id_plus_spawning_turn_no_cross_collision():
    """Two conversations may each declare a branch called 'b:0'; both"""
    b1 = ConversationBranchInfo(
        branch_id="b:0",
        child_conversation_ids=["x"],
        mode=ConversationBranchMode.SPAWN,
    )
    b2 = ConversationBranchInfo(
        branch_id="b:0",
        child_conversation_ids=["y"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv1 = _mk_conv(
        "conv-A",
        [
            TurnMetadata(branch_ids=["b:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b:0")
                ]
            ),
        ],
        [b1],
    )
    conv2 = _mk_conv(
        "conv-B",
        [
            TurnMetadata(branch_ids=["b:0"]),
            TurnMetadata(),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b:0")
                ]
            ),
        ],
        [b2],
    )
    cs = _mk_source([conv1, conv2])
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())
    conv_a_entries = orch._prereq_index.get(("conv-A", 0), [])
    assert [(b, g) for b, g, _ in conv_a_entries] == [("b:0", 1)]
    conv_b_entries = orch._prereq_index.get(("conv-B", 0), [])
    assert [(b, g) for b, g, _ in conv_b_entries] == [("b:0", 2)]


@pytest.mark.asyncio
async def test_intercept_spawns_without_gate_when_branch_has_no_consumer_prereq():
    """When a turn declares a branch but no later turn has a SPAWN_JOIN"""
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a"],
            mode=ConversationBranchMode.FORK,
        ),
    ]
    parent_meta.turns = [MagicMock(branch_ids=["root:0"])]
    cs.get_metadata = MagicMock(return_value=parent_meta)
    cs.start_branch_child = MagicMock(
        side_effect=lambda **kw: MagicMock(
            x_correlation_id=f"child-{kw['child_conversation_id']}"
        )
    )

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    assert orch._prereq_index == {}
    credit = MagicMock(
        x_correlation_id="root",
        conversation_id="c",
        turn_index=0,
        agent_depth=0,
        parent_correlation_id=None,
    )
    assert await orch.intercept(credit) is False
    assert cs.start_branch_child.call_count == 1
    assert orch._active_joins == {}
    assert orch._future_joins == {}


@pytest.mark.asyncio
async def test_intercept_concurrent_on_same_parent_corr_serializes_via_parent_lock():
    """Two concurrent intercept() calls for the same parent_corr must be"""
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a"],
            mode=ConversationBranchMode.FORK,
        ),
    ]
    parent_meta.turns = [MagicMock(branch_ids=["root:0"])]
    cs.get_metadata = MagicMock(return_value=parent_meta)

    enter_event = asyncio.Event()
    release_event = asyncio.Event()
    call_counter = {"n": 0}

    def _fake_child(**kw):
        call_counter["n"] += 1
        return MagicMock(x_correlation_id=f"child-{call_counter['n']}")

    cs.start_branch_child = MagicMock(side_effect=_fake_child)

    issuer = MagicMock()

    order: list[str] = []

    async def _dispatch_first(child):
        if not enter_event.is_set():
            enter_event.set()
            order.append(f"first-enter-{child.x_correlation_id}")
            await release_event.wait()
            order.append(f"first-exit-{child.x_correlation_id}")
        else:
            order.append(f"second-{child.x_correlation_id}")
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch_first)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credit = MagicMock(
        x_correlation_id="root",
        conversation_id="c",
        turn_index=0,
        agent_depth=0,
        parent_correlation_id=None,
    )

    task1 = asyncio.create_task(orch.intercept(credit))
    await enter_event.wait()

    task2 = asyncio.create_task(orch.intercept(credit))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert order == ["first-enter-child-1"]

    release_event.set()
    await asyncio.gather(task1, task2)

    assert order[0].startswith("first-enter-")
    assert order[1].startswith("first-exit-")
    assert order[2].startswith("second-")


@pytest.mark.asyncio
async def test_intercept_short_circuits_when_cleaning_up():
    cs = MagicMock()
    cs.start_branch_child = MagicMock()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())
    orch._cleaning_up = True
    credit = MagicMock(
        x_correlation_id="root",
        conversation_id="c",
        turn_index=0,
        agent_depth=0,
        parent_correlation_id=None,
    )
    assert await orch.intercept(credit) is False
    cs.start_branch_child.assert_not_called()


@pytest.mark.asyncio
async def test_start_branch_child_raise_rolls_back_sticky_refcount_unchanged():
    """When ``start_branch_child`` raises, no partial bookkeeping remains."""
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a"],
            mode=ConversationBranchMode.FORK,
        ),
    ]
    parent_meta.turns = [MagicMock(branch_ids=["root:0"])]
    cs.get_metadata = MagicMock(return_value=parent_meta)

    cs.start_branch_child = MagicMock(side_effect=RuntimeError("boom"))

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock()

    sticky_router = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky_router
    )
    baseline_descendant_counts = dict(orch._descendant_counts)

    credit = MagicMock(
        x_correlation_id="root",
        conversation_id="c",
        turn_index=0,
        agent_depth=0,
        parent_correlation_id=None,
    )
    assert await orch.intercept(credit) is False

    assert orch.stats.children_errored == 1
    assert orch.stats.children_spawned == 0
    sticky_router.register_child_routing.assert_not_called()
    sticky_router.evict_unclaimed_sticky.assert_called_once_with("root")
    assert orch._descendant_counts == baseline_descendant_counts
    assert orch._child_to_join == {}


@pytest.mark.asyncio
async def test_on_child_leaf_reached_unknown_parent_corr_logs_and_noops():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    await orch.on_child_leaf_reached("no-such-child")
    assert orch.stats.children_completed == 0
    assert orch._active_joins == {}


@pytest.mark.asyncio
async def test_on_child_errored_fail_fast_env_terminates(monkeypatch):
    """With ``AIPERF_DAG_FAIL_FAST=true`` set BEFORE construction, the"""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)

    issuer = MagicMock()
    issuer.abort_session = AsyncMock()
    sticky_router = MagicMock()

    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=issuer,
        sticky_router=sticky_router,
    )
    assert orch._fail_fast is True

    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=1, completed=set(), registered=True
    )
    pending.is_blocked = True
    orch._active_joins["p"] = pending
    orch._child_to_join["c1"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=2, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_modes = {"c1": ConversationBranchMode.FORK}
    orch._descendant_counts["p"] = 2

    await orch.on_child_errored("c1")
    assert orch.stats.parents_failed_due_to_child_error == 1
    assert "p" not in orch._active_joins
    issuer.abort_session.assert_any_await("p")


@pytest.mark.asyncio
async def test_on_child_errored_non_fail_fast_continues(monkeypatch):
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", False)

    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock()
    issuer.abort_session = AsyncMock()
    sticky_router = MagicMock()

    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=issuer,
        sticky_router=sticky_router,
    )
    assert orch._fail_fast is False

    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=2, completed=set(), registered=True
    )
    pending.is_blocked = True
    orch._active_joins["p"] = pending
    orch._child_to_join["c1"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=2, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_to_join["c2"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=2, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_modes = {
        "c1": ConversationBranchMode.FORK,
        "c2": ConversationBranchMode.FORK,
    }
    orch._descendant_counts["p"] = 3

    await orch.on_child_errored("c1")
    issuer.abort_session.assert_not_called()
    assert "p" in orch._active_joins
    state = orch._active_joins["p"].outstanding["SPAWN_JOIN:b"]
    assert state.expected == 2
    assert state.completed == {"c1"}
    assert orch.stats.parents_failed_due_to_child_error == 0


@pytest.mark.asyncio
async def test_gate_closes_only_after_all_hundred_children_complete():
    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=MagicMock(), credit_issuer=issuer)

    child_ids = {f"c{i}" for i in range(100)}
    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=len(child_ids), completed=set(), registered=True
    )
    pending.is_blocked = True
    orch._active_joins["p"] = pending
    for cid in child_ids:
        orch._child_to_join[cid] = [
            ChildJoinEntry(
                parent_correlation_id="p",
                gated_turn_index=1,
                prereq_key="SPAWN_JOIN:b",
            )
        ]
    orch._child_modes = {cid: ConversationBranchMode.SPAWN for cid in child_ids}
    orch._descendant_counts["p"] = 1 + 100

    ordered = sorted(child_ids, key=lambda s: int(s[1:]))
    for idx, cid in enumerate(ordered):
        await orch.on_child_leaf_reached(cid)
        if idx < 99:
            assert issuer.dispatch_join_turn.await_count == 0, (
                f"dispatch_join_turn fired early at child #{idx}"
            )
    assert issuer.dispatch_join_turn.await_count == 1
    assert "p" not in orch._active_joins


@pytest.mark.asyncio
async def test_intercept_gather_exception_in_one_child_does_not_block_siblings():
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a", "b", "c"],
            mode=ConversationBranchMode.FORK,
        ),
    ]
    parent_meta.turns = [MagicMock(branch_ids=["root:0"])]
    cs.get_metadata = MagicMock(return_value=parent_meta)

    def _fake_child(**kw):
        cid = kw["child_conversation_id"]
        if cid == "b":
            raise RuntimeError("start failed for b")
        return MagicMock(x_correlation_id=f"child-{cid}")

    cs.start_branch_child = MagicMock(side_effect=_fake_child)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credit = MagicMock(
        x_correlation_id="root",
        conversation_id="conv",
        turn_index=0,
        agent_depth=0,
        parent_correlation_id=None,
    )
    assert await orch.intercept(credit) is False

    assert orch.stats.children_spawned == 2
    assert orch.stats.children_errored == 1
    assert "child-a" in orch._child_to_join
    assert "child-c" in orch._child_to_join
    assert "child-b" not in orch._child_to_join
    assert issuer.dispatch_first_turn.await_count == 2


def test_cleanup_with_pending_joins_logs_leak_warning(caplog):
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    pending = PendingBranchJoin(
        parent_x_correlation_id="leaky",
        parent_conversation_id="conv",
        parent_num_turns=4,
        gated_turn_index=3,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=1, completed=set(), registered=True
    )
    orch._active_joins["leaky"] = pending
    with caplog.at_level(logging.WARNING, logger="aiperf.timing.branch_orchestrator"):
        orch.cleanup()

    leak_records = [r for r in caplog.records if "leaked state" in r.getMessage()]
    assert len(leak_records) == 1
    abandoned_records = [
        r for r in caplog.records if "Abandoned pending join" in r.getMessage()
    ]
    assert abandoned_records, "expected per-parent abandoned-join warning"
    assert "leaky" in abandoned_records[0].getMessage()


@pytest.mark.asyncio
async def test_intercept_reentry_for_same_parent_after_join_starts_new_gate():
    """After one intercept cycle for parent P completes, a second"""
    cs = MagicMock()

    parent_meta = MagicMock()
    first_branch = MagicMock(
        branch_id="p:0",
        child_conversation_ids=["a"],
        mode=ConversationBranchMode.SPAWN,
    )
    second_branch = MagicMock(
        branch_id="p:1",
        child_conversation_ids=["b"],
        mode=ConversationBranchMode.SPAWN,
    )
    parent_meta.branches = [first_branch, second_branch]
    parent_meta.turns = [
        MagicMock(branch_ids=["p:0"]),
        MagicMock(branch_ids=["p:1"]),
    ]
    cs.get_metadata = MagicMock(return_value=parent_meta)

    cs.start_branch_child = MagicMock(
        side_effect=lambda **kw: MagicMock(
            x_correlation_id=f"child-{kw['child_conversation_id']}"
        )
    )

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credit0 = MagicMock(
        x_correlation_id="parent-P",
        conversation_id="conv-P",
        turn_index=0,
        agent_depth=0,
        parent_correlation_id=None,
    )
    assert await orch.intercept(credit0) is False
    assert "child-a" in orch._child_to_join
    await orch.on_child_leaf_reached("child-a")
    assert "child-a" not in orch._child_to_join

    credit1 = MagicMock(
        x_correlation_id="parent-P",
        conversation_id="conv-P",
        turn_index=1,
        agent_depth=0,
        parent_correlation_id=None,
    )
    assert await orch.intercept(credit1) is False
    assert "child-b" in orch._child_to_join
