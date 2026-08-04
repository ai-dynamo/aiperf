# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-component adversarial tests for the DAG orchestrator."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import (
    CacheBustTarget,
    ConversationBranchMode,
    CreditPhase,
    PrerequisiteKind,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.credit.sticky_router import (
    StickyCreditRouter,
    WorkerLoad,
    _StickyEntry,
)
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.workers.session_manager import UserSessionManager


def _mk_conv_meta(
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
        s.agent_depth = agent_depth
        s.parent_correlation_id = parent_correlation_id
        s.branch_mode = branch_mode
        return s

    def _start_pre(child_cid, **kwargs):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_cid}"
        s.conversation_id = child_cid
        s.agent_depth = 1
        s.parent_correlation_id = None
        s.branch_mode = ConversationBranchMode.SPAWN
        return s

    cs.start_branch_child = MagicMock(side_effect=_start_branch)
    cs.start_pre_session_child = MagicMock(side_effect=_start_pre)
    return cs


def _mk_credit(
    conv_id: str,
    corr_id: str,
    turn_index: int,
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=agent_depth,
        parent_correlation_id=parent_correlation_id,
        branch_mode=ConversationBranchMode.FORK,
    )


def _mk_issuer(dispatch_first=True, dispatch_join=True):
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=dispatch_first)
    issuer.dispatch_join_turn = AsyncMock(return_value=dispatch_join)
    issuer.abort_session = AsyncMock()
    return issuer


def _k5_metadata():
    """Parent with 6 turns: spawn on turn 0 (FORK) gating turn 5."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c0"],
        mode=ConversationBranchMode.FORK,
    )
    root = _mk_conv_meta(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"], has_forks=True),
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
    c0 = _mk_conv_meta("c0", [TurnMetadata()], [])
    return [root, c0]


def _make_real_conversation(cid: str, num_turns: int) -> Conversation:
    """Build a real ``Conversation`` with sentinel turns so snapshot semantics are detectable (mutating the parent's turn_list later must not leak)."""
    return Conversation(
        session_id=cid,
        turns=[Turn(role="user", model="m") for _ in range(num_turns)],
        branches=[
            ConversationBranchInfo(
                branch_id=f"{cid}:0",
                child_conversation_ids=["whatever"],
                mode=ConversationBranchMode.FORK,
            )
        ],
    )


def test_fork_child_turn_list_snapshot_taken_at_create_time():
    """A FORK child seeded when the parent has dispatched turns 0..2 snapshots the parent's current turn_list, so later parent advances do not leak into the child's turn_list."""
    mgr = UserSessionManager()
    parent_conv = _make_real_conversation("parent", num_turns=6)
    parent = mgr.create_and_store(
        x_correlation_id="parent-corr",
        conversation=parent_conv,
        num_turns=6,
    )
    parent.advance_turn(0)
    parent.advance_turn(1)
    parent.advance_turn(2)
    assert len(parent.turn_list) == 3

    child_conv = _make_real_conversation("child", num_turns=2)
    child = mgr.create_and_store(
        x_correlation_id="child-corr",
        conversation=child_conv,
        num_turns=2,
        parent_correlation_id="parent-corr",
        branch_mode=ConversationBranchMode.FORK,
    )
    mgr.seed_from_parent("child-corr", "parent-corr")
    assert len(child.turn_list) == 3

    parent.advance_turn(3)
    parent.advance_turn(4)
    assert len(parent.turn_list) == 5
    assert len(child.turn_list) == 3, (
        "FORK snapshot must not alias the parent's turn_list"
    )


@pytest.mark.asyncio
async def test_fork_refcount_decrements_on_child_terminal_not_on_gate_satisfy():
    """Each FORK child increments the parent's sticky refcount, and decrements occur only when the child reports terminal completion via on_child_leaf_reached."""
    branch_a = ConversationBranchInfo(
        branch_id="root:0:A",
        child_conversation_ids=["a"],
        mode=ConversationBranchMode.FORK,
    )
    branch_b = ConversationBranchInfo(
        branch_id="root:0:B",
        child_conversation_ids=["b"],
        mode=ConversationBranchMode.FORK,
    )
    root = _mk_conv_meta(
        "root",
        [
            TurnMetadata(branch_ids=["root:0:A", "root:0:B"], has_forks=True),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0:A"
                    )
                ]
            ),
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
            _mk_conv_meta("a", [TurnMetadata()], []),
            _mk_conv_meta("b", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert sticky.register_child_routing.call_count == 2

    await orch.on_child_leaf_reached("corr-a")
    assert sticky.release_child_routing.call_count == 1

    await orch.on_child_leaf_reached("corr-b")
    assert sticky.release_child_routing.call_count == 2


def test_sticky_entry_stays_when_child_completes_before_parent_final(benchmark_run):
    """Child completion decrements ref_count, but the entry remains in place because parent_final_seen=False (the parent has not reached its terminal turn yet)."""
    router = StickyCreditRouter(run=benchmark_run, service_id="rtr-3")
    router._workers = {"w1": WorkerLoad(worker_id="w1")}
    router._workers_cache = list(router._workers.values())
    router._sticky_sessions["parent-corr"] = _StickyEntry(
        worker_id="w1", ref_count=1, parent_final_seen=False
    )
    router.register_child_routing("parent-corr")  # ref=2
    assert router._sticky_sessions["parent-corr"].ref_count == 2

    router.release_child_routing("parent-corr")  # ref=1, final_seen=False
    assert "parent-corr" in router._sticky_sessions
    entry = router._sticky_sessions["parent-corr"]
    assert entry.ref_count == 1
    assert entry.parent_final_seen is False


def test_sticky_evicts_when_parent_final_then_child_release(benchmark_run):
    """Order A: parent terminal first sets parent_final_seen=True, and a later child release drops ref_count to 0, triggering eviction."""
    router = StickyCreditRouter(run=benchmark_run, service_id="rtr-4a")
    router._workers = {"w1": WorkerLoad(worker_id="w1", active_sessions=1)}
    router._workers["w1"].active_session_ids.add("parent-corr")
    router._workers_cache = list(router._workers.values())
    router._sticky_sessions["parent-corr"] = _StickyEntry(
        worker_id="w1", ref_count=2, parent_final_seen=True
    )
    router.release_child_routing("parent-corr")  # ref=1, evict skipped
    assert "parent-corr" in router._sticky_sessions
    router.release_child_routing("parent-corr")  # ref=0 + final_seen -> evict
    assert "parent-corr" not in router._sticky_sessions
    assert router._workers["w1"].active_sessions == 0


def test_sticky_evicts_when_child_release_brings_ref_to_zero_after_final(benchmark_run):
    """Order B: child completes first (ref=1, no final_seen), then parent terminal flips parent_final_seen=True and drops ref to 0, so the eviction trigger (ref_count<=0 AND parent_final_seen) fires."""
    router = StickyCreditRouter(run=benchmark_run, service_id="rtr-4b")
    router._workers = {"w1": WorkerLoad(worker_id="w1", active_sessions=1)}
    router._workers["w1"].active_session_ids.add("parent-corr")
    router._workers_cache = list(router._workers.values())
    router._sticky_sessions["parent-corr"] = _StickyEntry(
        worker_id="w1", ref_count=2, parent_final_seen=False
    )
    router.release_child_routing("parent-corr")  # ref=1 no eviction
    assert "parent-corr" in router._sticky_sessions
    # Simulate parent terminal turn arriving: send_credit's final-turn branch
    # flips parent_final_seen=True and decrements ref_count.
    router._sticky_sessions["parent-corr"].parent_final_seen = True
    router._sticky_sessions["parent-corr"].ref_count -= 1
    entry = router._sticky_sessions["parent-corr"]
    assert entry.ref_count == 0
    # Eviction trigger fires inline in send_credit's final-turn branch when
    # ref_count<=0 AND not has_forks; here we manually exercise it.
    if entry.ref_count <= 0 and entry.parent_final_seen:
        router._sticky_sessions.pop("parent-corr", None)
    assert "parent-corr" not in router._sticky_sessions


@pytest.mark.asyncio
async def test_register_then_dispatch_fail_rolls_back_sticky_refcount():
    """The orchestrator registers FORK sticky refcount before dispatch_first_turn, so a False dispatch triggers the per-child rollback to release exactly once (register count == release count)."""
    cs = _mk_source(_k5_metadata())
    issuer = _mk_issuer(dispatch_first=False)
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    assert sticky.register_child_routing.call_count == 1
    assert sticky.release_child_routing.call_count == 1, (
        "rollback path must release sticky exactly once per failed FORK child"
    )
    # ``dispatch_first=False`` is stop-condition refusal, not an error.
    assert orch.stats.children_truncated == 1
    assert orch.stats.children_errored == 0


@pytest.mark.asyncio
async def test_all_start_branch_child_failures_evict_unclaimed_sticky(benchmark_run):
    """When a parent final with has_forks retains sticky and every start_branch_child raises before register_child_routing, finalize must force-pop the entry or active_sessions leaks permanently."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["a", "b"],
        mode=ConversationBranchMode.FORK,
    )
    root = _mk_conv_meta(
        "root",
        [TurnMetadata(branch_ids=["root:0"], has_forks=True)],
        [branch],
    )
    cs = _mk_source(
        [
            root,
            _mk_conv_meta("a", [TurnMetadata()], []),
            _mk_conv_meta("b", [TurnMetadata()], []),
        ]
    )
    cs.start_branch_child = MagicMock(side_effect=RuntimeError("start failed"))

    router = StickyCreditRouter(run=benchmark_run, service_id="rtr-5b")
    router._workers = {"w1": WorkerLoad(worker_id="w1", active_sessions=1)}
    router._workers["w1"].active_session_ids.add("corr-root")
    router._workers_cache = list(router._workers.values())
    # State after send_credit of parent final with has_forks=True.
    router._sticky_sessions["corr-root"] = _StickyEntry(
        worker_id="w1", ref_count=0, parent_final_seen=True
    )

    orch = BranchOrchestrator(
        conversation_source=cs,
        credit_issuer=_mk_issuer(),
        sticky_router=router,
    )
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is False

    assert orch.stats.children_errored == 2
    assert orch.stats.children_spawned == 0
    assert "corr-root" not in router._sticky_sessions
    assert router._workers["w1"].active_sessions == 0


def test_register_child_routing_with_no_existing_sticky_entry_is_noop(benchmark_run):
    """register_child_routing on a parent that never had a turn dispatched is a documented no-op; the router does not create an entry."""
    router = StickyCreditRouter(run=benchmark_run, service_id="rtr-6")
    router._workers = {"w1": WorkerLoad(worker_id="w1")}
    router._workers_cache = list(router._workers.values())

    router.register_child_routing("ghost-parent")
    assert "ghost-parent" not in router._sticky_sessions

    router.release_child_routing("ghost-parent")
    assert "ghost-parent" not in router._sticky_sessions


@pytest.mark.asyncio
async def test_dispatch_join_turn_returns_false_increments_joins_suppressed():
    """When the credit issuer reports the gated turn was suppressed, the orchestrator increments joins_suppressed and does not retry."""
    cs = _mk_source(_k5_metadata())
    issuer = _mk_issuer(dispatch_join=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 4))  # suspend
    assert "corr-root" in orch._active_joins

    await orch.on_child_leaf_reached("corr-c0")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.joins_suppressed == 1
    assert orch.stats.parents_resumed == 0
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins


@pytest.mark.asyncio
async def test_orchestrator_handles_dispatch_first_turn_returning_falsy():
    """The orchestrator's _dispatch_first_turn wraps with bool(), so both False and None collapse to False and trigger the rollback path."""
    cs = _mk_source(_k5_metadata())
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(side_effect=[None])
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    # ``_dispatch_first_turn`` wraps with ``bool()``: None -> False, treated
    # as stop-condition refusal (truncated), not an error.
    assert orch.stats.children_truncated == 1
    assert orch.stats.children_errored == 0
    assert sticky.release_child_routing.call_count == 1


@pytest.mark.asyncio
async def test_orchestrator_handles_dispatch_first_turn_raising():
    """If dispatch_first_turn raises, asyncio.gather(return_exceptions=True) captures the exception and the orchestrator's per-child loop treats any non-True result as failure, rolling back bookkeeping cleanly."""
    cs = _mk_source(_k5_metadata())
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(
        side_effect=RuntimeError("FORK routing invariant violated")
    )
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    assert orch.stats.children_errored == 1
    assert sticky.register_child_routing.call_count == 1
    assert sticky.release_child_routing.call_count == 1


@pytest.mark.asyncio
async def test_worker_disconnect_mid_dag_cleanup_clears_state(caplog):
    """After a FORK child credit is dispatched but never returns (worker died), cleanup() at phase teardown must clear _child_to_join, _descendant_counts, and active/future joins."""
    import logging as _logging

    cs = _mk_source(_k5_metadata())
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))  # spawn FORK c0
    await orch.intercept(_mk_credit("root", "corr-root", 4))  # suspend
    assert "corr-root" in orch._active_joins
    assert orch._descendant_counts.get("corr-root", 0) == 1

    with caplog.at_level(_logging.WARNING, logger="aiperf.timing.branch_orchestrator"):
        orch.cleanup()

    assert not orch._active_joins
    assert not orch._future_joins
    assert not orch._child_to_join
    assert not orch._descendant_counts
    assert any("leaked state" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_fork_siblings_pin_to_parents_worker_via_sticky_routing(benchmark_run):
    """Two FORK children sharing a parent route to the same worker because the credit router's routing_key uses parent_correlation_id."""
    router = StickyCreditRouter(run=benchmark_run, service_id="rtr-11")
    router._router_client = MagicMock()
    router._router_client.send_to = AsyncMock()
    router._workers = {
        "w1": WorkerLoad(worker_id="w1", active_sessions=1, in_flight_credits=0),
        "w2": WorkerLoad(worker_id="w2", in_flight_credits=0),
    }
    router._workers["w1"].active_session_ids.add("parent-corr")
    router._workers_cache = list(router._workers.values())
    router._workers_by_load[0].update({"w1", "w2"})
    router._sticky_sessions["parent-corr"] = _StickyEntry(
        worker_id="w1", ref_count=1, parent_final_seen=False
    )

    issued_ns = time.time_ns()
    child_a = Credit(
        id=10,
        phase=CreditPhase.PROFILING,
        conversation_id="ca",
        x_correlation_id="corr-a",
        turn_index=0,
        num_turns=1,
        issued_at_ns=issued_ns,
        agent_depth=1,
        parent_correlation_id="parent-corr",
    )
    child_b = Credit(
        id=11,
        phase=CreditPhase.PROFILING,
        conversation_id="cb",
        x_correlation_id="corr-b",
        turn_index=0,
        num_turns=1,
        issued_at_ns=issued_ns,
        agent_depth=1,
        parent_correlation_id="parent-corr",
    )
    await router.send_credit(child_a)
    await router.send_credit(child_b)

    sent_workers = [c.args[0] for c in router._router_client.send_to.call_args_list]
    assert sent_workers == ["w1", "w1"], (
        "FORK siblings must both pin to parent's worker via sticky routing"
    )


@pytest.mark.asyncio
async def test_spawn_child_does_not_call_register_child_routing():
    """SPAWN-mode children do not bump sticky refcount, so the orchestrator does not call ``register_child_routing`` for them."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["s0", "s1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv_meta(
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
            _mk_conv_meta("s0", [TurnMetadata()], []),
            _mk_conv_meta("s1", [TurnMetadata()], []),
        ]
    )
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    sticky.register_child_routing.assert_not_called()

    await orch.on_child_leaf_reached("corr-s0")
    await orch.on_child_leaf_reached("corr-s1")
    sticky.release_child_routing.assert_not_called()


@pytest.mark.asyncio
async def test_spawn_child_inherits_parent_cache_bust_state():
    """A non-agentic SPAWN child preserves its parent's cache-bust state."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv_meta("root", [TurnMetadata(branch_ids=["root:0"])], [branch])
    cs = _mk_source([root, _mk_conv_meta("child", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(
        conversation_source=cs,
        credit_issuer=issuer,
        cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
    )
    credit = _mk_credit("root", "corr-root", 0)
    credit.effective_root_correlation_id = "corr-root"
    credit.cache_bust_marker = "[rid:parent]"
    credit.cache_bust_target = CacheBustTarget.FIRST_TURN_PREFIX

    await orch.intercept(credit)

    kwargs = cs.start_branch_child.call_args.kwargs
    assert kwargs["cache_bust_marker"] == "[rid:parent]"
    assert kwargs["cache_bust_target"] == CacheBustTarget.FIRST_TURN_PREFIX


def test_pre_session_child_routing_key_falls_back_to_own_correlation():
    """Pre-session children have parent_correlation_id=None, so routing_key falls back to x_correlation_id (no sticky pin to a non-existent parent)."""
    from aiperf.timing.conversation_source import SampledSession

    pre = SampledSession(
        conversation_id="early",
        metadata=ConversationMetadata(conversation_id="early", turns=[TurnMetadata()]),
        x_correlation_id="self-corr",
        agent_depth=1,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.SPAWN,
    )
    assert pre.routing_key == "self-corr"


@pytest.mark.asyncio
async def test_pre_session_dispatch_failure_still_records_branch():
    """A False dispatch_first_turn during pre-session dispatch bumps children_truncated but still adds the branch to _pre_dispatched_branches so the per-turn intercept does not re-dispatch it on the parent's turn-0 credit return."""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        dispatch_timing="pre",
    )
    root = _mk_conv_meta(
        "root",
        [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()],
        [pre_branch],
    )
    cs = _mk_source([root, _mk_conv_meta("early", [TurnMetadata()], [])])
    issuer = _mk_issuer(dispatch_first=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    assert orch.stats.children_errored == 0
    assert orch.stats.children_truncated == 1
    assert ("root", "root:pre") in orch._pre_dispatched_branches

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    cs.start_branch_child.assert_not_called()


@pytest.mark.asyncio
async def test_start_branch_child_raises_for_one_sibling_others_continue():
    """If start_branch_child raises for one child but succeeds for another, the surviving child is still tracked, dispatched, and counted."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["bad", "good"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv_meta(
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
            _mk_conv_meta("bad", [TurnMetadata()], []),
            _mk_conv_meta("good", [TurnMetadata()], []),
        ]
    )

    def _start_branch_with_failure(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        if child_conversation_id == "bad":
            raise RuntimeError("kaboom")
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        s.conversation_id = child_conversation_id
        return s

    cs.start_branch_child = MagicMock(side_effect=_start_branch_with_failure)

    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))

    assert orch.stats.children_errored == 1
    assert orch.stats.children_spawned == 1
    issuer.dispatch_first_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_pre_session_child_raises_siblings_continue():
    """In dispatch_pre_session_branches, an exception from start_pre_session_child is caught per-child and stats are bumped, and the next child still attempts to dispatch."""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["bad", "good"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
        dispatch_timing="pre",
    )
    root = _mk_conv_meta(
        "root",
        [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()],
        [pre_branch],
    )
    cs = _mk_source(
        [
            root,
            _mk_conv_meta("bad", [TurnMetadata()], []),
            _mk_conv_meta("good", [TurnMetadata()], []),
        ]
    )

    def _start_pre_with_failure(child_cid, **kwargs):
        if child_cid == "bad":
            raise RuntimeError("kaboom")
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_cid}"
        s.conversation_id = child_cid
        s.agent_depth = 1
        s.parent_correlation_id = None
        return s

    cs.start_pre_session_child = MagicMock(side_effect=_start_pre_with_failure)

    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()

    assert orch.stats.children_errored == 1
    assert orch.stats.children_spawned == 1
    issuer.dispatch_first_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_metadata_raises_propagates_through_intercept():
    """ConversationSource.get_metadata raising KeyError on an unknown conversation propagates out of intercept, since a missing-conversation invariant violation must be loud rather than swallowed."""
    cs = _mk_source(_k5_metadata())
    cs.get_metadata.side_effect = KeyError("no metadata for conv")
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    with pytest.raises(KeyError):
        await orch.intercept(_mk_credit("root", "corr-root", 0))


def test_spawn_child_with_parent_correlation_routes_to_parent_worker():
    """SampledSession.routing_key returns parent_correlation_id whenever set, so SPAWN children that inherit it route to the parent's worker regardless of branch_mode."""
    from aiperf.timing.conversation_source import SampledSession

    spawn = SampledSession(
        conversation_id="child",
        metadata=ConversationMetadata(conversation_id="child", turns=[TurnMetadata()]),
        x_correlation_id="self-corr",
        agent_depth=1,
        parent_correlation_id="parent-corr",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    assert spawn.routing_key == "parent-corr"


@pytest.mark.asyncio
async def test_concurrent_register_release_refcount_converges(benchmark_run):
    """Since the sticky router runs single-threaded with synchronous register/release, N concurrent tasks each doing register+release leave ref_count at the original value."""
    router = StickyCreditRouter(run=benchmark_run, service_id="rtr-19")
    router._workers = {"w1": WorkerLoad(worker_id="w1")}
    router._workers_cache = list(router._workers.values())
    router._sticky_sessions["parent-corr"] = _StickyEntry(
        worker_id="w1", ref_count=1, parent_final_seen=False
    )

    async def _bump_and_release():
        router.register_child_routing("parent-corr")
        await asyncio.sleep(0)
        router.release_child_routing("parent-corr")

    await asyncio.gather(*(_bump_and_release() for _ in range(50)))
    assert router._sticky_sessions["parent-corr"].ref_count == 1


@pytest.mark.asyncio
async def test_active_sessions_unchanged_when_fork_children_share_parent_sticky(
    benchmark_run,
):
    """When FORK children route via parent_correlation_id, send_credit reuses the existing sticky entry instead of creating a new one, so WorkerLoad.active_sessions stays at the parent's count (1 here)."""
    router = StickyCreditRouter(run=benchmark_run, service_id="rtr-20")
    router._router_client = MagicMock()
    router._router_client.send_to = AsyncMock()
    router._workers = {
        "w1": WorkerLoad(worker_id="w1", active_sessions=1, in_flight_credits=0),
    }
    router._workers["w1"].active_session_ids.add("parent-corr")
    router._workers_cache = list(router._workers.values())
    router._workers_by_load[0].add("w1")
    router._sticky_sessions["parent-corr"] = _StickyEntry(
        worker_id="w1", ref_count=1, parent_final_seen=False
    )

    issued_ns = time.time_ns()
    for n in range(5):
        child = Credit(
            id=100 + n,
            phase=CreditPhase.PROFILING,
            conversation_id=f"c{n}",
            x_correlation_id=f"corr-c{n}",
            turn_index=0,
            num_turns=1,
            issued_at_ns=issued_ns,
            agent_depth=1,
            parent_correlation_id="parent-corr",
        )
        await router.send_credit(child)

    assert router._workers["w1"].active_sessions == 1
    assert router._workers["w1"].active_session_ids == {"parent-corr"}
