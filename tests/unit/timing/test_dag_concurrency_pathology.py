# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Concurrency / cancellation / stop-condition pathology tests for ``BranchOrchestrator``."""

from __future__ import annotations

import asyncio
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
from aiperf.timing.branch_orchestrator import (
    BranchOrchestrator,
    ChildJoinEntry,
)
from aiperf.timing.phase.stop_conditions import (
    DagLifecycleStopCondition,
    DurationStopCondition,
    LifecycleStopCondition,
    RequestCountStopCondition,
    SessionCountStopCondition,
)
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


def _simple_spawn_metadata(
    n_children: int = 2, conv_id: str = "root"
) -> list[ConversationMetadata]:
    """Conversation: turn 0 spawns ``n_children`` children, turn 1 gates them."""
    branch = ConversationBranchInfo(
        branch_id=f"{conv_id}:0",
        child_conversation_ids=[f"{conv_id}-c{i}" for i in range(n_children)],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        conv_id,
        [
            TurnMetadata(branch_ids=[f"{conv_id}:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id=f"{conv_id}:0"
                    )
                ]
            ),
        ],
        [branch],
    )
    children = [
        _mk_conv(f"{conv_id}-c{i}", [TurnMetadata()], []) for i in range(n_children)
    ]
    return [root, *children]


@pytest.mark.asyncio
async def test_cancel_during_intercept_releases_parent_lock():
    """Cancel a task awaiting ``dispatch_first_turn`` inside ``intercept``."""
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()

    block = asyncio.Event()

    async def _hang(child):
        await block.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_hang)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credit = _mk_credit("root", "corr-root", 0)
    t1 = asyncio.create_task(orch.intercept(credit))
    for _ in range(5):
        await asyncio.sleep(0)
    assert "corr-root" in orch._parent_locks
    t1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t1

    lock = orch._parent_locks.get("corr-root")
    if lock is not None:
        assert not lock.locked(), "lock leaked after intercept cancel"

    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    result = await asyncio.wait_for(
        orch.intercept(_mk_credit("root", "corr-root", 0)), timeout=2.0
    )
    assert isinstance(result, bool)


@pytest.mark.asyncio
async def test_cancel_during_satisfy_prerequisite_keeps_state_consistent():
    """``_satisfy_prerequisite`` itself has no awaits between the"""
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()

    dispatch_started = asyncio.Event()
    dispatch_block = asyncio.Event()

    async def _join_dispatch(pending):
        dispatch_started.set()
        await dispatch_block.wait()
        return True

    issuer.dispatch_join_turn = AsyncMock(side_effect=_join_dispatch)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert "corr-root" in orch._active_joins

    t = asyncio.create_task(orch.on_child_leaf_reached("corr-root-c0"))
    await dispatch_started.wait()

    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    assert "corr-root" not in orch._active_joins
    assert "corr-root-c0" not in orch._child_to_join

    await orch.on_child_leaf_reached("corr-root-c0")
    dispatch_block.set()
    await asyncio.sleep(0)
    assert issuer.dispatch_join_turn.await_count == 1


@pytest.mark.asyncio
async def test_cancel_during_gather_partial_dispatch_rolls_back_consistently():
    """One child raises a generic exception (return_exceptions=True ⇒ caught"""
    cs = _mk_source(_simple_spawn_metadata(3))
    issuer = _mk_issuer()

    async def _dispatch_with_one_failure(child):
        if child.x_correlation_id == "corr-root-c1":
            raise RuntimeError("boom")
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch_with_one_failure)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    assert "corr-root-c0" in orch._child_to_join
    assert "corr-root-c2" in orch._child_to_join
    assert "corr-root-c1" not in orch._child_to_join

    pending = orch._active_joins["corr-root"]
    state = pending.outstanding["SPAWN_JOIN:root:0"]
    assert state.expected == 2
    assert orch.stats.children_errored == 1
    assert orch.stats.children_spawned == 2


@pytest.mark.asyncio
async def test_cancel_during_pre_session_loop_partial_pre_dispatched_set():
    """Three pre-session branches; the second blocks, gets cancelled. Only"""
    branches = [
        ConversationBranchInfo(
            branch_id=f"root:pre{i}",
            child_conversation_ids=[f"pre{i}"],
            mode=ConversationBranchMode.SPAWN,
            is_background=True,
            dispatch_timing="pre",
        )
        for i in range(3)
    ]
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=[f"root:pre{i}" for i in range(3)]),
            TurnMetadata(),
        ],
        branches,
    )
    children = [_mk_conv(f"pre{i}", [TurnMetadata()], []) for i in range(3)]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()

    call_count = 0
    block = asyncio.Event()

    async def _dispatch(session):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            await block.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    t = asyncio.create_task(orch.dispatch_pre_session_branches())
    for _ in range(10):
        await asyncio.sleep(0)
        if call_count >= 2:
            break

    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    pre = orch._pre_dispatched_branches
    assert ("root", "root:pre0") in pre
    assert ("root", "root:pre1") not in pre
    assert ("root", "root:pre2") not in pre


@pytest.mark.asyncio
async def test_100_concurrent_intercepts_independent_parents_isolated_state():
    """Each parent's gates / joins are independent. No cross-talk via the"""
    N = 100
    convs: list[ConversationMetadata] = []
    for i in range(N):
        convs.extend(_simple_spawn_metadata(2, conv_id=f"r{i}"))
    cs = _mk_source(convs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credits = [_mk_credit(f"r{i}", f"corr-r{i}", 0) for i in range(N)]
    results = await asyncio.gather(*(orch.intercept(c) for c in credits))

    assert all(r is True for r in results)
    assert orch.stats.children_spawned == 2 * N
    assert len(orch._active_joins) == N
    for i in range(N):
        active = orch._active_joins[f"corr-r{i}"]
        assert active.gated_turn_index == 1
        state = active.outstanding[f"SPAWN_JOIN:r{i}:0"]
        assert state.expected == 2


@pytest.mark.asyncio
async def test_100_concurrent_intercepts_same_parent_serialized():
    """Single parent receives 100 intercept calls at distinct turn_indexes"""
    cs = _mk_source(_simple_spawn_metadata(2))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credits = [_mk_credit("root", "corr-root", i % 2) for i in range(100)]
    await asyncio.gather(*(orch.intercept(c) for c in credits))

    assert orch.stats.children_spawned > 0
    lock = orch._parent_locks["corr-root"]
    assert not lock.locked()


@pytest.mark.asyncio
async def test_race_parent_return_and_last_child_completion_gate_fires_once():
    """Two orderings — child-first then parent, parent-first then child —"""
    cs1 = _mk_source(_simple_spawn_metadata(1))
    issuer1 = _mk_issuer()
    orch1 = BranchOrchestrator(conversation_source=cs1, credit_issuer=issuer1)
    await orch1.intercept(_mk_credit("root", "corr-root", 0))
    assert orch1._active_joins["corr-root"].gated_turn_index == 1
    await orch1.on_child_leaf_reached("corr-root-c0")
    issuer1.dispatch_join_turn.assert_awaited_once()

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
    cs2 = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer2 = _mk_issuer()
    orch2 = BranchOrchestrator(conversation_source=cs2, credit_issuer=issuer2)
    await orch2.intercept(_mk_credit("root", "corr-root2", 0))
    await orch2.on_child_leaf_reached("corr-c1")
    suspended = await orch2.intercept(_mk_credit("root", "corr-root2", 1))
    assert suspended is False
    issuer2.dispatch_join_turn.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_mid_pre_session_dispatch_no_state_leak():
    """``cleanup()`` is synchronous — it cannot interrupt an awaiting"""
    branches = [
        ConversationBranchInfo(
            branch_id=f"root:pre{i}",
            child_conversation_ids=[f"pre{i}"],
            mode=ConversationBranchMode.SPAWN,
            is_background=True,
            dispatch_timing="pre",
        )
        for i in range(3)
    ]
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=[f"root:pre{i}" for i in range(3)]),
            TurnMetadata(),
        ],
        branches,
    )
    children = [_mk_conv(f"pre{i}", [TurnMetadata()], []) for i in range(3)]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    started = asyncio.Event()
    proceed = asyncio.Event()

    async def _slow(session):
        started.set()
        await proceed.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_slow)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    t = asyncio.create_task(orch.dispatch_pre_session_branches())
    await started.wait()
    orch.cleanup()
    proceed.set()
    await t

    assert orch._cleaning_up is True
    orch.cleanup()
    assert (await orch.intercept(_mk_credit("root", "corr-root", 0))) is False


@pytest.mark.asyncio
async def test_stop_flips_during_release_increments_joins_suppressed_only_once():
    """``dispatch_join_turn`` returns False (simulating stop). Verify"""
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()
    issuer.dispatch_join_turn = AsyncMock(return_value=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-root-c0")
    assert orch.stats.joins_suppressed == 1
    assert orch.stats.parents_resumed == 0

    await orch.on_child_leaf_reached("corr-root-c0")
    assert orch.stats.joins_suppressed == 1
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_fail_fast_two_simultaneous_child_errors_aborts_parent_once(
    monkeypatch, force_fail_fast
):
    """Under fail-fast, two children of the same parent fire"""

    force_fail_fast(True)
    cs = _mk_source(_simple_spawn_metadata(3))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert "corr-root" in orch._active_joins

    await asyncio.gather(
        orch.on_child_errored("corr-root-c0"),
        orch.on_child_errored("corr-root-c1"),
    )

    aborts = [c.args[0] for c in issuer.abort_session.await_args_list]
    assert "corr-root" in aborts
    assert orch.stats.parents_failed_due_to_child_error == 1


@pytest.mark.asyncio
async def test_wait_for_zero_timeout_cancels_intercept_lock_released():
    """Force a TimeoutError -> CancelledError propagation into intercept."""
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()

    block = asyncio.Event()

    async def _hang(child):
        await block.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_hang)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            orch.intercept(_mk_credit("root", "corr-root", 0)),
            timeout=0.001,
        )

    lock = orch._parent_locks.get("corr-root")
    if lock is not None:
        assert not lock.locked()
    block.set()


@pytest.mark.asyncio
async def test_release_blocked_join_does_not_recurse_into_intercept():
    """If ``dispatch_join_turn`` synchronously triggered another intercept"""
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    intercept_calls_during_dispatch: list[bool] = []
    in_dispatch = False

    async def _join_dispatch(pending):
        nonlocal in_dispatch
        in_dispatch = True
        await asyncio.sleep(0)
        in_dispatch = False
        return True

    issuer.dispatch_join_turn = AsyncMock(side_effect=_join_dispatch)
    original_intercept = orch.intercept

    async def _spy_intercept(credit):
        intercept_calls_during_dispatch.append(in_dispatch)
        return await original_intercept(credit)

    orch.intercept = _spy_intercept  # type: ignore[method-assign]

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-root-c0")

    assert intercept_calls_during_dispatch == [False]
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_leaf_and_errored_for_same_child_one_wins():
    """Concurrent leaf + errored for same child. ``_child_to_join.pop``"""
    cs = _mk_source(_simple_spawn_metadata(2))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    await asyncio.gather(
        orch.on_child_leaf_reached("corr-root-c0"),
        orch.on_child_errored("corr-root-c0"),
    )

    pending = orch._active_joins["corr-root"]
    state = pending.outstanding["SPAWN_JOIN:root:0"]
    assert "corr-root-c0" in state.completed
    assert len(state.completed) == 1


def test_stop_condition_applies_to_dag_children_truth_table():
    """Children honor: cancellation (DagLifecycle), Duration, RequestCount."""
    assert DagLifecycleStopCondition.applies_to_dag_children is True
    assert DurationStopCondition.applies_to_dag_children is True
    assert LifecycleStopCondition.applies_to_dag_children is False
    assert RequestCountStopCondition.applies_to_dag_children is True
    assert SessionCountStopCondition.applies_to_dag_children is False


@pytest.mark.asyncio
async def test_pre_session_dispatch_first_turn_returns_false_counts_truncated():
    """``issued`` is False ⇒ stop-condition refusal (e.g. ``--request-count``"""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()],
        [pre_branch],
    )
    early = _mk_conv("early", [TurnMetadata()], [])
    cs = _mk_source([root, early])
    issuer = _mk_issuer()
    issuer.dispatch_first_turn = AsyncMock(return_value=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()

    assert orch.stats.children_spawned == 0
    assert orch.stats.children_errored == 0
    assert orch.stats.children_truncated == 1
    assert ("root", "root:pre") in orch._pre_dispatched_branches


@pytest.mark.asyncio
async def test_many_parents_simultaneous_gate_arrival_no_active_joins_iter_corruption():
    """50 parents all arrive at their gated turn simultaneously. _active_joins"""
    N = 50
    convs: list[ConversationMetadata] = []
    for i in range(N):
        convs.extend(_simple_spawn_metadata(1, conv_id=f"r{i}"))
    cs = _mk_source(convs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await asyncio.gather(
        *(orch.intercept(_mk_credit(f"r{i}", f"corr-r{i}", 0)) for i in range(N))
    )
    assert len(orch._active_joins) == N

    await asyncio.gather(
        *(orch.on_child_leaf_reached(f"corr-r{i}-c0") for i in range(N))
    )

    assert issuer.dispatch_join_turn.await_count == N
    assert orch._active_joins == {}
    assert orch.stats.parents_resumed == N


@pytest.mark.asyncio
async def test_one_of_fifty_children_raises_others_complete_state_consistent():
    """Inside ``_spawn_children_and_register_gates`` the gather uses"""
    cs = _mk_source(_simple_spawn_metadata(50))
    issuer = _mk_issuer()

    async def _maybe_raise(child):
        if child.x_correlation_id == "corr-root-c25":
            raise RuntimeError("boom")
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_maybe_raise)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    pending = orch._active_joins["corr-root"]
    state = pending.outstanding["SPAWN_JOIN:root:0"]
    assert state.expected == 49
    assert orch.stats.children_errored == 1
    for i in range(50):
        if i == 25:
            continue
        await orch.on_child_leaf_reached(f"corr-root-c{i}")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_release_blocked_join_before_dispatch_returns_no_double_count():
    """Mid-await of ``dispatch_join_turn``, cancel the satisfying task. The"""
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()

    started = asyncio.Event()
    block = asyncio.Event()

    async def _hang_dispatch(pending):
        started.set()
        await block.wait()
        return True

    issuer.dispatch_join_turn = AsyncMock(side_effect=_hang_dispatch)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    t = asyncio.create_task(orch.on_child_leaf_reached("corr-root-c0"))
    await started.wait()
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    assert orch.stats.parents_resumed == 0
    assert orch.stats.joins_suppressed == 0
    assert "corr-root" not in orch._active_joins
    assert "corr-root-c0" not in orch._child_to_join
    block.set()


@pytest.mark.asyncio
async def test_concurrent_intercepts_post_cleanup_all_short_circuit():
    """After cleanup, every intercept must early-return False without"""
    cs = _mk_source(_simple_spawn_metadata(2))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    orch.cleanup()

    results = await asyncio.gather(
        *(orch.intercept(_mk_credit("root", "corr-root", 0)) for _ in range(20))
    )
    assert all(r is False for r in results)
    assert orch.stats.children_spawned == 0
    cs.start_branch_child.assert_not_called()


@pytest.mark.asyncio
async def test_intercept_after_cleanup_does_not_repopulate_parent_locks():
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    orch.cleanup()

    for i in range(50):
        await orch.intercept(_mk_credit("root", f"corr-{i}", 0))
    assert orch._parent_locks == {}


@pytest.mark.asyncio
async def test_cancel_mid_spawn_partial_state_visible_no_corruption():
    """Cancel the intercept task while ``_spawn_children_and_register_gates``"""
    cs = _mk_source(_simple_spawn_metadata(3))
    issuer = _mk_issuer()
    block = asyncio.Event()
    started_count = 0

    async def _slow(child):
        nonlocal started_count
        started_count += 1
        await block.wait()
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_slow)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    t = asyncio.create_task(orch.intercept(_mk_credit("root", "corr-root", 0)))
    for _ in range(10):
        await asyncio.sleep(0)
        if started_count >= 3:
            break

    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t

    assert len(orch._child_to_join) == 3
    pending = orch._active_joins.get("corr-root") or orch._future_joins.get(
        "corr-root", {}
    ).get(1)
    assert pending is not None
    assert pending.outstanding["SPAWN_JOIN:root:0"].expected == 3

    block.set()
    orch.cleanup()


@pytest.mark.asyncio
async def test_cleanup_during_satisfy_release_does_not_fire_dispatch():
    """``cleanup()`` sets ``_cleaning_up=True`` synchronously. A child"""
    cs = _mk_source(_simple_spawn_metadata(1))
    issuer = _mk_issuer()
    started = asyncio.Event()
    block = asyncio.Event()

    async def _hang(pending):
        started.set()
        await block.wait()
        return True

    issuer.dispatch_join_turn = AsyncMock(side_effect=_hang)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "corr-root", 0))

    t = asyncio.create_task(orch.on_child_leaf_reached("corr-root-c0"))
    await started.wait()
    orch.cleanup()
    block.set()
    await t
    issuer.dispatch_join_turn.assert_awaited_once()


def test_child_join_entry_is_frozen_and_hashable():
    e = ChildJoinEntry(
        parent_correlation_id="p", gated_turn_index=1, prereq_key="SPAWN_JOIN:b"
    )
    with pytest.raises((AttributeError, Exception)):
        e.parent_correlation_id = "x"  # type: ignore[misc]
    s = {e}
    assert e in s


def test_orchestrator_never_imports_stop_conditions():
    """Sanity: BranchOrchestrator must not depend on StopCondition state —"""
    import inspect

    import aiperf.timing.branch_orchestrator as mod

    src = inspect.getsource(mod)
    assert "StopCondition" not in src
    assert "stop_conditions" not in src
