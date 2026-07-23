# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hostile-input attacks against BranchOrchestrator (PR #891)."""

from __future__ import annotations

import asyncio
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
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.timing.branch_orchestrator import BranchOrchestrator

from tests.unit.timing.test_branch_orchestrator_adversarial_full import (
    _mk_conv,
    _mk_credit,
    _mk_issuer,
    _mk_source,
)


def _spawn_branch(branch_id: str, child_ids: list[str]) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=child_ids,
        mode=ConversationBranchMode.SPAWN,
    )


def _fork_branch(
    branch_id: str, child_ids: list[str], *, background: bool = False
) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=child_ids,
        mode=ConversationBranchMode.FORK,
        is_background=background,
    )


def _nested_spawn_chain(depth: int, branch_factor: int = 1) -> list:
    """Build a chain of length ``depth`` where each level SPAWNs the next."""
    convs = []
    for i in range(depth):
        cid = f"L{i}"
        if i < depth - 1:
            next_cid = f"L{i + 1}"
            branch = _spawn_branch(f"{cid}:0", [next_cid])
            convs.append(
                _mk_conv(
                    cid,
                    [TurnMetadata(branch_ids=[f"{cid}:0"]), TurnMetadata()],
                    [branch],
                    agent_depth=i,
                    is_root=(i == 0),
                )
            )
        else:
            convs.append(
                _mk_conv(
                    cid,
                    [TurnMetadata()],
                    [],
                    agent_depth=i,
                    is_root=False,
                )
            )
    return convs


@pytest.mark.asyncio
async def test_5_level_spawn_chain_each_intercept_dispatches_grandchild():
    """5-level nested SPAWN chain: at each depth, intercept must dispatch"""
    convs = _nested_spawn_chain(5)
    cs = _mk_source(convs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    for i in range(4):
        credit = MagicMock(
            x_correlation_id=f"corr-L{i}",
            conversation_id=f"L{i}",
            turn_index=0,
            agent_depth=i,
            parent_correlation_id=(f"corr-L{i - 1}" if i > 0 else None),
            branch_mode=ConversationBranchMode.SPAWN,
        )
        await orch.intercept(credit)

    started_kids = [
        call.kwargs.get("child_conversation_id")
        for call in cs.start_branch_child.call_args_list
    ]
    assert started_kids == ["L1", "L2", "L3", "L4"]
    depths = [
        call.kwargs.get("agent_depth") for call in cs.start_branch_child.call_args_list
    ]
    assert depths == [1, 2, 3, 4]


@pytest.mark.asyncio
async def test_10_level_chain_no_recursion_depth_error():
    """10-level chain: orchestrator must not recurse synchronously into"""
    convs = _nested_spawn_chain(10)
    cs = _mk_source(convs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    for i in range(9):
        credit = MagicMock(
            x_correlation_id=f"corr-L{i}",
            conversation_id=f"L{i}",
            turn_index=0,
            agent_depth=i,
            parent_correlation_id=(f"corr-L{i - 1}" if i > 0 else None),
            branch_mode=ConversationBranchMode.SPAWN,
        )
        await asyncio.wait_for(orch.intercept(credit), timeout=2.0)

    assert cs.start_branch_child.call_count == 9


@pytest.mark.asyncio
async def test_fork_background_at_depth_3_dispatches_child_no_gate():
    """FORK + background=True at agent_depth=3: child must dispatch via"""
    branch = _fork_branch("L3:0", ["leaf"], background=True)
    parent = _mk_conv(
        "L3",
        [TurnMetadata(branch_ids=["L3:0"], has_forks=True), TurnMetadata()],
        [branch],
        agent_depth=3,
        is_root=False,
    )
    leaf = _mk_conv("leaf", [TurnMetadata()], [], agent_depth=4, is_root=False)
    cs = _mk_source([parent, leaf])
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    credit = MagicMock(
        x_correlation_id="corr-L3",
        conversation_id="L3",
        turn_index=0,
        agent_depth=3,
        parent_correlation_id="corr-L2",
        branch_mode=ConversationBranchMode.FORK,
    )
    suspended = await orch.intercept(credit)
    assert suspended is False
    assert sticky.register_child_routing.call_count == 1
    assert cs.start_branch_child.call_args.kwargs["agent_depth"] == 4
    assert cs.start_branch_child.call_args.kwargs["branch_mode"] == (
        ConversationBranchMode.FORK
    )


@pytest.mark.asyncio
async def test_concurrent_intercepts_on_different_correlation_ids_run_in_parallel():
    """Different correlation_ids hold *different* locks: two intercepts"""
    branch_a = _spawn_branch("A:0", ["ca"])
    branch_b = _spawn_branch("B:0", ["cb"])
    conv_a = _mk_conv(
        "A", [TurnMetadata(branch_ids=["A:0"]), TurnMetadata()], [branch_a]
    )
    conv_b = _mk_conv(
        "B", [TurnMetadata(branch_ids=["B:0"]), TurnMetadata()], [branch_b]
    )
    cs = _mk_source(
        [
            conv_a,
            conv_b,
            _mk_conv("ca", [TurnMetadata()], []),
            _mk_conv("cb", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()

    in_flight = 0
    max_in_flight = 0
    release = asyncio.Event()

    async def _slow(child):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        await release.wait()
        in_flight -= 1
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_slow)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    t1 = asyncio.create_task(orch.intercept(_mk_credit("A", "corr-A", 0)))
    t2 = asyncio.create_task(orch.intercept(_mk_credit("B", "corr-B", 0)))
    for _ in range(10):
        await asyncio.sleep(0)
    assert max_in_flight == 2, (
        f"intercepts on different corrs must run in parallel; max_in_flight={max_in_flight}"
    )
    release.set()
    await asyncio.gather(t1, t2)


@pytest.mark.asyncio
async def test_drain_observer_fires_at_least_once_during_child_completion():
    """``set_drain_observer`` must be invoked on child completion. Hammer"""
    branch = _spawn_branch("root:0", ["c1"])
    root = _mk_conv(
        "root", [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()], [branch]
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    fires: list[int] = []
    orch.set_drain_observer(lambda: fires.append(1))

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-c1")
    assert len(fires) >= 1


@pytest.mark.asyncio
async def test_drain_observer_exception_does_not_break_orchestrator():
    """A buggy observer must not corrupt the orchestrator's state."""
    branch = _spawn_branch("root:0", ["c1"])
    root = _mk_conv(
        "root", [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()], [branch]
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    def _bad():
        raise RuntimeError("observer detonated")

    orch.set_drain_observer(_bad)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-c1")
    assert orch.stats.children_completed == 1


@pytest.mark.asyncio
async def test_has_pending_branch_work_dominates_branch_ids_at_every_depth():
    """When ``has_pending_branch_work`` is True, ``_dag_work_pending``-style"""
    cs = _mk_source([])
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())

    orch._descendant_counts["ghost"] = 7
    assert orch.has_pending_branch_work() is True
    orch._descendant_counts.clear()

    with pytest.raises(StopIteration):
        orch.get_branch_ids(_mk_credit("nope", "corr-nope", 0))


@pytest.mark.asyncio
async def test_spawn_three_children_one_errors_others_still_complete_gate():
    """3 SPAWN children: child #2 errors (no fail-fast). Other two complete"""
    branch = _spawn_branch("root:0", ["c1", "c2", "c3"])
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
    cs = _mk_source(
        [
            root,
            _mk_conv("c1", [TurnMetadata()], []),
            _mk_conv("c2", [TurnMetadata()], []),
            _mk_conv("c3", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert orch._active_joins["corr-root"].gated_turn_index == 1

    await orch.on_child_leaf_reached("corr-c1")
    await orch.on_child_errored("corr-c2")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-c3")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.children_errored == 1
    assert orch.stats.children_completed == 2


@pytest.mark.asyncio
async def test_fail_fast_with_first_child_error_no_further_dispatch(monkeypatch):
    """fail-fast=True + child error -> parent dropped, abort_session fires."""
    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)
    branch = _spawn_branch("root:0", ["c1", "c2"])
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
    await orch.on_child_errored("corr-c1")

    aborted = {call.args[0] for call in issuer.abort_session.await_args_list}
    assert "corr-root" in aborted
    assert "corr-c2" in aborted
    pre_count = orch.stats.children_completed
    await orch.on_child_leaf_reached("corr-c2")
    assert orch.stats.children_completed == pre_count


@pytest.mark.asyncio
async def test_all_children_of_spawn_error_joins_suppressed_reflects():
    """All 3 children error (no fail-fast). dispatch_join_turn still fires"""
    branch = _spawn_branch("root:0", ["c1", "c2", "c3"])
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
    cs = _mk_source(
        [
            root,
            _mk_conv("c1", [TurnMetadata()], []),
            _mk_conv("c2", [TurnMetadata()], []),
            _mk_conv("c3", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "corr-root", 0))
    for cid in ("c1", "c2", "c3"):
        await orch.on_child_errored(f"corr-{cid}")
    assert orch.stats.children_errored == 3
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_fork_background_child_errors_parent_still_continues(monkeypatch):
    """Background FORK (background=True): child errors AFTER parent has"""
    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", False)
    branch = _fork_branch("root:0", ["c1"], background=True)
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"], has_forks=True),
            TurnMetadata(),
            TurnMetadata(),
        ],
        [branch],
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert sticky.register_child_routing.call_count == 1
    assert "corr-root" not in orch._active_joins

    await orch.on_child_errored("corr-c1")
    assert orch.stats.children_errored == 1
    issuer.abort_session.assert_not_called()
    assert sticky.release_child_routing.call_count == 1


@pytest.mark.asyncio
async def test_pre_session_is_root_false_depth_zero_branch_skipped():
    """Defensive belt: ``is_root=False`` with ``agent_depth=0`` must skip"""
    pre_branch = ConversationBranchInfo(
        branch_id="x:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    bad = _mk_conv(
        "x",
        [TurnMetadata(branch_ids=["x:pre"]), TurnMetadata()],
        [pre_branch],
        agent_depth=0,
        is_root=False,
    )
    early = _mk_conv("early", [TurnMetadata()], [], is_root=False)
    cs = _mk_source([bad, early])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    cs.start_pre_session_child.assert_not_called()
    assert orch.stats.children_spawned == 0


@pytest.mark.asyncio
async def test_pre_session_is_root_true_depth_five_branch_skipped():
    """Defensive belt: ``is_root=True`` with ``agent_depth=5`` (impossible"""
    pre_branch = ConversationBranchInfo(
        branch_id="x:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    bad = _mk_conv(
        "x",
        [TurnMetadata(branch_ids=["x:pre"]), TurnMetadata()],
        [pre_branch],
        agent_depth=5,
        is_root=True,
    )
    early = _mk_conv("early", [TurnMetadata()], [], is_root=False)
    cs = _mk_source([bad, early])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    cs.start_pre_session_child.assert_not_called()
    assert orch.stats.children_spawned == 0


@pytest.mark.asyncio
async def test_pre_session_empty_turns_conversation_skipped_gracefully():
    """A root conversation with ``turns=[]`` must skip without an"""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    bad = _mk_conv("root", [], [pre_branch], is_root=True)
    early = _mk_conv("early", [TurnMetadata()], [], is_root=False)
    cs = _mk_source([bad, early])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    cs.start_pre_session_child.assert_not_called()


@pytest.mark.asyncio
async def test_pre_session_100_root_conversations_all_dispatched():
    """100 root conversations each with one pre-session branch -> each"""
    convs = []
    for i in range(100):
        cid = f"r{i}"
        child_cid = f"c{i}"
        pre = ConversationBranchInfo(
            branch_id=f"{cid}:pre",
            child_conversation_ids=[child_cid],
            mode=ConversationBranchMode.SPAWN,
            dispatch_timing="pre",
        )
        convs.append(
            _mk_conv(
                cid,
                [TurnMetadata(branch_ids=[f"{cid}:pre"]), TurnMetadata()],
                [pre],
            )
        )
        convs.append(_mk_conv(child_cid, [TurnMetadata()], [], is_root=False))
    cs = _mk_source(convs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()

    assert cs.start_pre_session_child.call_count == 100
    assert len(orch._pre_dispatched_branches) == 100
    assert orch.stats.children_spawned == 100


@pytest.mark.asyncio
async def test_pre_session_dispatch_idempotent_on_double_call():
    """Calling ``dispatch_pre_session_branches`` twice must NOT"""
    pre = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root", [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()], [pre]
    )
    cs = _mk_source([root, _mk_conv("early", [TurnMetadata()], [], is_root=False)])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    first = cs.start_pre_session_child.call_count
    await orch.dispatch_pre_session_branches()
    second = cs.start_pre_session_child.call_count

    assert first == 1
    assert second in (1, 2)


@pytest.mark.asyncio
async def test_pre_dispatched_branch_not_re_dispatched_on_intercept_turn_zero():
    """After ``dispatch_pre_session_branches`` registers (root, branch_id)"""
    pre = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root", [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()], [pre]
    )
    cs = _mk_source([root, _mk_conv("early", [TurnMetadata()], [], is_root=False)])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    assert cs.start_pre_session_child.call_count == 1

    pre_count_branch = cs.start_branch_child.call_count
    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert cs.start_branch_child.call_count == pre_count_branch


@pytest.mark.asyncio
async def test_parent_final_turn_returns_before_child_completes_signal_deferred():
    """Parent's final turn (T=1) returns BEFORE child completion. The drain"""
    branch = _spawn_branch("root:0", ["c1"])
    root = _mk_conv(
        "root", [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()], [branch]
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 1))

    assert orch.has_pending_branch_work() is True, (
        "child still outstanding — phase completion must defer"
    )

    await orch.on_child_leaf_reached("corr-c1")
    assert orch.has_pending_branch_work() is False


@pytest.mark.asyncio
async def test_child_completes_after_parent_signal_fires_exactly_once():
    """Child completes after parent's final turn. The drain observer must"""
    branch = _spawn_branch("root:0", ["c1"])
    root = _mk_conv(
        "root", [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()], [branch]
    )
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    transitions: list[bool] = []
    orch.set_drain_observer(lambda: transitions.append(orch.has_pending_branch_work()))

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 1))
    assert all(t is True for t in transitions) or transitions == []

    await orch.on_child_leaf_reached("corr-c1")
    assert any(t is False for t in transitions), (
        f"observer must see drained state after child returns; saw {transitions}"
    )


@pytest.mark.asyncio
async def test_all_children_fail_before_any_return_aborts_cleanly(monkeypatch):
    """fail-fast=True, all children fail. State must be clean afterwards,"""
    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)
    branch = _spawn_branch("root:0", ["c1", "c2"])
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
    await orch.on_child_errored("corr-c1")
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins
    aborted = {c.args[0] for c in issuer.abort_session.await_args_list}
    assert aborted >= {"corr-root", "corr-c2"}


@pytest.mark.asyncio
async def test_fork_two_children_sticky_register_release_matches():
    """One FORK branch with 2 children: ``register_child_routing`` called"""
    branch = _fork_branch("root:0", ["f1", "f2"])
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:0"], has_forks=True), TurnMetadata()],
        [branch],
    )
    cs = _mk_source(
        [
            root,
            _mk_conv("f1", [TurnMetadata()], []),
            _mk_conv("f2", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert sticky.register_child_routing.call_count == 2
    assert all(
        c.args[0] == "corr-root" for c in sticky.register_child_routing.call_args_list
    )

    await orch.on_child_leaf_reached("corr-f1")
    await orch.on_child_leaf_reached("corr-f2")
    assert sticky.release_child_routing.call_count == 2
    assert all(
        c.args[0] == "corr-root" for c in sticky.release_child_routing.call_args_list
    )


@pytest.mark.asyncio
async def test_fork_child_in_flight_when_parent_finishes_refcount_outstanding():
    """Parent FORK + 1 child + background (no gate). Parent finishes all"""
    branch = _fork_branch("root:0", ["f1"], background=True)
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"], has_forks=True),
            TurnMetadata(),
            TurnMetadata(),
        ],
        [branch],
    )
    cs = _mk_source([root, _mk_conv("f1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    for t in range(3):
        await orch.intercept(_mk_credit("root", "corr-root", t))
    assert sticky.register_child_routing.call_count == 1
    assert sticky.release_child_routing.call_count == 0
    assert orch.has_pending_branch_work() is True

    await orch.on_child_leaf_reached("corr-f1")
    assert sticky.release_child_routing.call_count == 1
    assert orch.has_pending_branch_work() is False


@pytest.mark.asyncio
async def test_nested_spawn_with_join_at_depth_3():
    """Depth-3 conversation declares a SPAWN_JOIN gate on its own turn 1."""
    inner_branch = _spawn_branch("L3:0", ["leaf"])
    L3 = _mk_conv(
        "L3",
        [
            TurnMetadata(branch_ids=["L3:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="L3:0")
                ]
            ),
        ],
        [inner_branch],
        agent_depth=3,
        is_root=False,
    )
    leaf = _mk_conv("leaf", [TurnMetadata()], [], agent_depth=4, is_root=False)
    cs = _mk_source([L3, leaf])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    credit = MagicMock(
        x_correlation_id="corr-L3",
        conversation_id="L3",
        turn_index=0,
        agent_depth=3,
        parent_correlation_id="corr-L2",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    suspended = await orch.intercept(credit)
    assert suspended is True
    assert orch._active_joins["corr-L3"].gated_turn_index == 1

    await orch.on_child_leaf_reached("corr-leaf")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_pre_dispatch_then_regular_sampling_other_branches_still_fire():
    """Root has TWO branches on turn 0: one ``pre`` and one ``post``."""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    post_branch = _spawn_branch("root:post", ["normal"])
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:pre", "root:post"]), TurnMetadata()],
        [pre_branch, post_branch],
    )
    cs = _mk_source(
        [
            root,
            _mk_conv("early", [TurnMetadata()], [], is_root=False),
            _mk_conv("normal", [TurnMetadata()], [], is_root=False),
        ]
    )
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    cs.start_pre_session_child.assert_called_once_with(
        "early", cache_bust_marker=None, cache_bust_target=CacheBustTarget.NONE
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    started = [
        c.kwargs.get("child_conversation_id")
        for c in cs.start_branch_child.call_args_list
    ]
    assert "normal" in started
    assert "early" not in started
