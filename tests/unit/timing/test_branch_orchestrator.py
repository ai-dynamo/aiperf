# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BranchOrchestrator skeleton + sticky-routing refcount hooks."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.timing.branch_orchestrator import (
    BranchOrchestrator,
    ChildJoinEntry,
    PendingBranchJoin,
    PrereqState,
)


@pytest.mark.asyncio
async def test_intercept_no_spawn_returns_false():
    cs = MagicMock()
    cs.get_metadata = MagicMock(
        return_value=MagicMock(turns=[MagicMock(branch_ids=[])])
    )
    issuer = MagicMock()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    credit = MagicMock(
        x_correlation_id="root", conversation_id="c", turn_index=0, agent_depth=0
    )
    assert await orch.intercept(credit) is False


@pytest.mark.asyncio
async def test_intercept_with_spawn_dispatches_children_and_registers_sticky():
    """Phase 1 semantics: intercept returns False after a pure-spawn with no
    gate on the very next turn (the parent may continue running)."""
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a", "b"],
            dispatch_timing="post",
            mode=ConversationBranchMode.FORK,
        ),
    ]
    parent_meta.turns = [MagicMock(branch_ids=["root:0"])]
    cs.get_metadata = MagicMock(return_value=parent_meta)

    def _fake_child(
        *,
        parent_correlation_id,
        child_conversation_id,
        agent_depth,
        branch_mode=None,
        **kwargs,
    ):
        return MagicMock(x_correlation_id=f"child-{child_conversation_id}")

    cs.start_branch_child = MagicMock(side_effect=_fake_child)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)

    sticky_router = MagicMock()
    sticky_router.register_child_routing = MagicMock()

    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky_router
    )
    credit = MagicMock(
        x_correlation_id="root", conversation_id="c", turn_index=0, agent_depth=0
    )

    # No SPAWN_JOIN prereq set -> no gate -> intercept returns False.
    assert await orch.intercept(credit) is False
    assert cs.start_branch_child.call_count == 2
    assert issuer.dispatch_first_turn.await_count == 2
    assert orch.stats.children_spawned == 2
    # Sticky-routing refcount bumped once per spawned child.
    assert sticky_router.register_child_routing.call_count == 2
    sticky_router.register_child_routing.assert_called_with("root")


@pytest.mark.asyncio
async def test_intercept_uses_get_metadata():
    """ConversationSource must expose ``get_metadata``; the orchestrator calls
    it directly."""

    class _FakeSource:
        def __init__(self, meta):
            self._meta = meta

        def get_metadata(self, conversation_id):
            return self._meta

    parent_meta = MagicMock()
    parent_meta.turns = [MagicMock(branch_ids=[])]
    parent_meta.branches = []
    source = _FakeSource(parent_meta)
    orch = BranchOrchestrator(conversation_source=source, credit_issuer=MagicMock())
    credit = MagicMock(
        x_correlation_id="root", conversation_id="c", turn_index=0, agent_depth=0
    )
    assert await orch.intercept(credit) is False


@pytest.mark.asyncio
async def test_dispatch_first_turn_raises_when_issuer_lacks_method():
    orch = BranchOrchestrator(conversation_source=MagicMock(), credit_issuer=object())
    with pytest.raises(AttributeError):
        await orch._dispatch_first_turn(MagicMock())


def _mk_pending_for_parent(
    parent_corr: str,
    *,
    gated_turn_index: int,
    prereq_key: str,
    outstanding: set[str],
    num_turns: int = 2,
) -> PendingBranchJoin:
    p = PendingBranchJoin(
        parent_x_correlation_id=parent_corr,
        parent_conversation_id="c",
        parent_num_turns=num_turns,
        gated_turn_index=gated_turn_index,
    )
    # Phase 3: outstanding values are PrereqState with an expected counter
    # and completed set. Pre-register expected==len(outstanding); the
    # provided child_corr ids remain outstanding (none are in completed).
    p.outstanding[prereq_key] = PrereqState(
        expected=len(outstanding), completed=set(), registered=True
    )
    return p


@pytest.mark.asyncio
async def test_child_leaf_decrements_and_triggers_join_when_all_done():
    cs = MagicMock()
    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    sticky_router = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky_router
    )
    pending = _mk_pending_for_parent(
        "parent",
        gated_turn_index=1,
        prereq_key="SPAWN_JOIN:b",
        outstanding={"cA", "cB"},
    )
    pending.is_blocked = True
    orch._active_joins["parent"] = pending
    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="parent",
            gated_turn_index=1,
            prereq_key="SPAWN_JOIN:b",
        )
    ]
    orch._child_to_join["cB"] = [
        ChildJoinEntry(
            parent_correlation_id="parent",
            gated_turn_index=1,
            prereq_key="SPAWN_JOIN:b",
        )
    ]
    orch._child_modes = {
        "cA": ConversationBranchMode.FORK,
        "cB": ConversationBranchMode.FORK,
    }
    orch._descendant_counts["parent"] = 3  # root + 2 children

    await orch.on_child_leaf_reached("cA")
    assert issuer.dispatch_join_turn.await_count == 0
    # Phase 3 counter form: cA reported, cB still outstanding (expected=2,
    # completed={"cA"}).
    state = orch._active_joins["parent"].outstanding["SPAWN_JOIN:b"]
    assert state.expected == 2
    assert state.completed == {"cA"}
    assert sticky_router.release_child_routing.call_count == 1

    await orch.on_child_leaf_reached("cB")
    assert issuer.dispatch_join_turn.await_count == 1
    awaited_pending = issuer.dispatch_join_turn.await_args.args[0]
    assert awaited_pending.parent_x_correlation_id == "parent"
    assert awaited_pending.gated_turn_index == 1
    assert "parent" not in orch._active_joins
    assert orch.stats.parents_resumed == 1
    assert sticky_router.release_child_routing.call_count == 2
    sticky_router.release_child_routing.assert_called_with("parent")


@pytest.mark.asyncio
async def test_no_join_case_releases_slot_when_descendants_drain():
    """Background / no-gate children still participate in descendant count
    accounting; the parent's slot is released once every tracked descendant
    reports done."""
    cs = MagicMock()
    issuer = MagicMock()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    released: list[str] = []
    orch._release_slot = lambda p: released.append(p)

    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="parent", gated_turn_index=None, prereq_key=None
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.FORK}
    orch._descendant_counts["parent"] = 2  # root terminal + 1 child

    await orch.on_child_leaf_reached("cA")
    # Without a gated_turn_index, nothing to dispatch; descendant count
    # drops to 1 (root still pending). The slot releases when the count
    # hits zero — here root hasn't reported yet, so the release fires only
    # after both hit zero. Simulate root terminal done:
    orch._descendant_counts["parent"] -= 1
    # Trigger a second decrement via a dummy child path (we only want to
    # assert the pure descendant-count arithmetic here).
    assert "parent" in orch._descendant_counts
    # When count reaches 0 the orchestrator releases the slot via
    # _handle_child_done. Simulate via on_child_leaf_reached with a fresh
    # entry:
    orch._child_to_join["cB"] = [
        ChildJoinEntry(
            parent_correlation_id="parent", gated_turn_index=None, prereq_key=None
        )
    ]
    orch._descendant_counts["parent"] = 1  # only one tracked descendant left
    await orch.on_child_leaf_reached("cB")
    assert released == ["parent"]


@pytest.mark.asyncio
async def test_leaf_for_unknown_child_is_noop():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    await orch.on_child_leaf_reached("unknown")
    assert orch.stats.children_completed == 0


@pytest.mark.asyncio
async def test_branch_orchestrator_child_stopped_decrements_pending_join():
    """on_child_stopped: when a child's continuation is cap-blocked, the
    parent's pending join must still drain so the join turn fires; the
    child is tallied under children_truncated, not children_completed."""
    cs = MagicMock()
    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    sticky_router = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky_router
    )
    pending = _mk_pending_for_parent(
        "parent",
        gated_turn_index=1,
        prereq_key="SPAWN_JOIN:b",
        outstanding={"cA"},
    )
    pending.is_blocked = True
    orch._active_joins["parent"] = pending
    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="parent",
            gated_turn_index=1,
            prereq_key="SPAWN_JOIN:b",
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.FORK}
    orch._descendant_counts["parent"] = 2  # root + 1 child

    await orch.on_child_stopped("cA")

    assert orch.stats.children_truncated == 1
    assert orch.stats.children_completed == 0
    # Pending join drained: parent removed and join turn dispatched.
    assert "parent" not in orch._active_joins
    assert issuer.dispatch_join_turn.await_count == 1
    # FORK sticky refcount released.
    sticky_router.release_child_routing.assert_called_once_with("parent")


@pytest.mark.asyncio
async def test_child_stopped_for_unknown_child_is_noop():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    await orch.on_child_stopped("unknown")
    assert orch.stats.children_truncated == 0


@pytest.mark.asyncio
async def test_dispatch_join_turn_raises_when_issuer_lacks_method():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock(spec=[])
    )
    pending = PendingBranchJoin(
        parent_x_correlation_id="parent",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    with pytest.raises(AttributeError):
        await orch._release_blocked_join(pending)


@pytest.mark.asyncio
async def test_child_error_decrements_join_when_not_fail_fast(monkeypatch):
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", False)

    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    sticky_router = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=issuer,
        sticky_router=sticky_router,
    )
    pending = _mk_pending_for_parent(
        "p",
        gated_turn_index=2,
        prereq_key="SPAWN_JOIN:b",
        outstanding={"c1"},
        num_turns=3,
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
    assert orch.stats.children_errored == 1
    assert issuer.dispatch_join_turn.await_count == 1
    sticky_router.release_child_routing.assert_called_once_with("p")


@pytest.mark.asyncio
async def test_child_error_fail_fast_aborts_parent(monkeypatch):
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)

    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock()
    issuer.abort_session = AsyncMock()
    sticky_router = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=issuer,
        sticky_router=sticky_router,
    )
    pending = _mk_pending_for_parent(
        "p",
        gated_turn_index=2,
        prereq_key="SPAWN_JOIN:b",
        outstanding={"c1", "c2"},
        num_turns=3,
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
    issuer.dispatch_join_turn.assert_not_awaited()
    assert orch.stats.parents_failed_due_to_child_error == 1
    assert "p" not in orch._active_joins
    assert "p" not in orch._descendant_counts
    assert "c2" not in orch._child_to_join
    # Refcount released for the errored child plus its orphan sibling.
    assert sticky_router.release_child_routing.call_count == 2
    # abort_session awaited for the parent and the orphan sibling.
    assert issuer.abort_session.await_count == 2
    awaited_targets = {call.args[0] for call in issuer.abort_session.await_args_list}
    assert awaited_targets == {"p", "c2"}


@pytest.mark.asyncio
async def test_child_error_fail_fast_fires_abort_observer(monkeypatch):
    """Under FAIL_FAST, the orchestrator must fire its abort observer
    after parent + orphan tear-down so the phase-side handler can cancel
    every active phase lifecycle. Without this, the strategy loop keeps
    issuing wire credits for unrelated roots and the docs' "abort the
    whole run on first DAG child error" contract is violated.
    """
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)
    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=MagicMock(
            dispatch_join_turn=AsyncMock(), abort_session=AsyncMock()
        ),
    )
    pending = _mk_pending_for_parent(
        "p",
        gated_turn_index=1,
        prereq_key="SPAWN_JOIN:b",
        outstanding={"c1"},
        num_turns=2,
    )
    pending.is_blocked = True
    orch._active_joins["p"] = pending
    orch._child_to_join["c1"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=1, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_modes = {"c1": ConversationBranchMode.SPAWN}
    orch._descendant_counts["p"] = 1

    abort_observer = MagicMock()
    orch.set_abort_observer(abort_observer)

    await orch.on_child_errored("c1")

    abort_observer.assert_called_once_with()
    assert orch.stats.parents_failed_due_to_child_error == 1


@pytest.mark.asyncio
async def test_child_error_non_fail_fast_does_not_fire_abort_observer(monkeypatch):
    """Default (FAIL_FAST=False) behavior: an errored child is treated as
    leaf-reached, NOT a whole-run abort. The abort observer must stay
    silent so unrelated parents keep running.
    """
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", False)
    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=MagicMock(dispatch_join_turn=AsyncMock(return_value=True)),
    )
    pending = _mk_pending_for_parent(
        "p",
        gated_turn_index=1,
        prereq_key="SPAWN_JOIN:b",
        outstanding={"c1"},
        num_turns=2,
    )
    orch._active_joins["p"] = pending
    orch._child_to_join["c1"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=1, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_modes = {"c1": ConversationBranchMode.SPAWN}
    orch._descendant_counts["p"] = 1

    abort_observer = MagicMock()
    orch.set_abort_observer(abort_observer)

    await orch.on_child_errored("c1")

    abort_observer.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_clears_abort_observer():
    """``cleanup`` must clear ``_abort_observer`` alongside
    ``_drain_observer`` so a torn-down orchestrator does not leak
    references to phase-side handlers across phase boundaries.
    """
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch.set_drain_observer(MagicMock())
    orch.set_abort_observer(MagicMock())
    orch.cleanup()
    assert orch._drain_observer is None
    assert orch._abort_observer is None


@pytest.mark.asyncio
async def test_dispatch_failure_rolls_back_bookkeeping():
    """When _dispatch_first_turn returns False (e.g. slots saturated), the
    orchestrator must undo its children_spawned / sticky-refcount /
    descendant-count / _child_to_join bookkeeping for the failed child."""
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a", "b"],
            dispatch_timing="post",
            mode=ConversationBranchMode.FORK,
        ),
    ]
    parent_meta.turns = [MagicMock(branch_ids=["root:0"])]
    cs.get_metadata = MagicMock(return_value=parent_meta)

    def _fake_child(
        *,
        parent_correlation_id,
        child_conversation_id,
        agent_depth,
        branch_mode=None,
        **kwargs,
    ):
        return MagicMock(x_correlation_id=f"child-{child_conversation_id}")

    cs.start_branch_child = MagicMock(side_effect=_fake_child)

    issuer = MagicMock()

    # First dispatch succeeds (True), second fails (False -- slots saturated).
    async def _dispatch(session):
        return session.x_correlation_id == "child-a"

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch)

    sticky_router = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky_router
    )
    credit = MagicMock(
        x_correlation_id="root", conversation_id="c", turn_index=0, agent_depth=0
    )

    # No gate -> intercept returns False. Only the successful child stays tracked.
    assert await orch.intercept(credit) is False
    assert orch.stats.children_spawned == 1
    # ``dispatch_first_turn`` returning False is stop-condition refusal
    # (slots saturated), not an error — tally as truncated.
    assert orch.stats.children_truncated == 1
    assert orch.stats.children_errored == 0
    assert "child-a" in orch._child_to_join
    assert "child-b" not in orch._child_to_join
    # register_child_routing fired for both children; release fired for the one
    # that failed to dispatch.
    assert sticky_router.register_child_routing.call_count == 2
    assert sticky_router.release_child_routing.call_count == 1


@pytest.mark.asyncio
async def test_child_error_for_unknown_child_is_noop():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    await orch.on_child_errored("unknown")
    assert orch.stats.children_errored == 0


@pytest.mark.asyncio
async def test_spawn_mode_branch_does_not_register_sticky_routing():
    """SPAWN-mode children must NOT increment the parent's sticky refcount
    (they do not inherit the parent's worker)."""
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["spawn-a"],
            dispatch_timing="post",
            mode=ConversationBranchMode.SPAWN,
        ),
    ]
    parent_meta.turns = [MagicMock(branch_ids=["root:0"])]
    cs.get_metadata = MagicMock(return_value=parent_meta)

    def _fake_child(
        *,
        parent_correlation_id,
        child_conversation_id,
        agent_depth,
        branch_mode,
        **kwargs,
    ):
        assert branch_mode == ConversationBranchMode.SPAWN
        return MagicMock(x_correlation_id=f"child-{child_conversation_id}")

    cs.start_branch_child = MagicMock(side_effect=_fake_child)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)

    sticky_router = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky_router
    )
    credit = MagicMock(
        x_correlation_id="root", conversation_id="c", turn_index=0, agent_depth=0
    )

    # No gate -> intercept returns False; children still spawn.
    assert await orch.intercept(credit) is False
    assert orch.stats.children_spawned == 1
    # Sticky refcount untouched for SPAWN-mode children.
    assert sticky_router.register_child_routing.call_count == 0

    # Leaf-reached must also NOT release anything because register didn't fire.
    await orch.on_child_leaf_reached("child-spawn-a")
    assert sticky_router.release_child_routing.call_count == 0


def test_has_pending_branch_work_empty_orchestrator():
    """Fresh orchestrator has no pending state."""
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    assert orch.has_pending_branch_work() is False


def test_has_pending_branch_work_with_active_join():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch._active_joins["p"] = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=1,
        gated_turn_index=None,
    )
    assert orch.has_pending_branch_work() is True


def test_has_pending_branch_work_with_descendant_count():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch._descendant_counts["p"] = 2
    assert orch.has_pending_branch_work() is True


def test_has_pending_branch_work_zeroed_descendant_count_is_false():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch._descendant_counts["p"] = 0
    assert orch.has_pending_branch_work() is False


def test_has_pending_branch_work_bare_child_tracking():
    """Child-to-join entries alone keep has_pending True — a child
    still in flight (not yet evicted) counts as outstanding work."""
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch._child_to_join["c"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=None, prereq_key=None
        )
    ]
    assert orch.has_pending_branch_work() is True


def test_cleanup_is_idempotent():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch.cleanup()
    # Second call is a no-op; must not raise.
    orch.cleanup()
    assert orch._cleaning_up is True


def test_cleanup_emits_leak_warning_when_state_nonempty(caplog):
    """Any residual active/future joins at cleanup time means the DAG failed
    to drain — cleanup logs a warning so diagnosis has a breadcrumb."""
    import logging

    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    pending = PendingBranchJoin(
        parent_x_correlation_id="leaky-parent",
        parent_conversation_id="conv-leaky",
        parent_num_turns=6,
        gated_turn_index=5,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=2, completed=set(), registered=True
    )
    orch._active_joins["leaky-parent"] = pending
    orch._child_to_join["child-a"] = [
        ChildJoinEntry(
            parent_correlation_id="leaky-parent",
            gated_turn_index=5,
            prereq_key="SPAWN_JOIN:b",
        )
    ]
    orch._descendant_counts["leaky-parent"] = 2

    with caplog.at_level(logging.WARNING, logger="aiperf.timing.branch_orchestrator"):
        orch.cleanup()

    leak_messages = [r for r in caplog.records if "leaked state" in r.getMessage()]
    assert len(leak_messages) == 1, "cleanup must warn about leaked state once"

    abandoned_joins = [
        r for r in caplog.records if "Abandoned pending join" in r.getMessage()
    ]
    assert len(abandoned_joins) == 1
    assert "leaky-parent" in abandoned_joins[0].getMessage()

    # State is cleared even on the warning path so subsequent access is clean.
    assert orch._active_joins == {}
    assert orch._future_joins == {}
    assert orch._child_to_join == {}
    assert orch._descendant_counts == {}


@pytest.mark.asyncio
async def test_intercept_short_circuits_when_cleaning_up():
    """Late credit returns after cleanup must not dispatch new work."""
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch.cleanup()
    credit = MagicMock(
        x_correlation_id="root", conversation_id="c", turn_index=0, agent_depth=0
    )
    assert await orch.intercept(credit) is False


@pytest.mark.asyncio
async def test_on_child_leaf_reached_short_circuits_when_cleaning_up():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch._child_to_join["c"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=None, prereq_key=None
        )
    ]
    orch.cleanup()
    # State snapshotted by cleanup was cleared, but the method must
    # also guard against re-entrancy with a direct early-return.
    orch._child_to_join["c"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=None, prereq_key=None
        )
    ]
    await orch.on_child_leaf_reached("c")
    # children_completed should NOT increment during teardown.
    assert orch.stats.children_completed == 0
