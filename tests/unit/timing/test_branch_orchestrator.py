# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BranchOrchestrator skeleton + sticky-routing refcount hooks."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.timing.branch_orchestrator import BranchOrchestrator


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
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a", "b"],
            is_background=False,
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
    ):
        return MagicMock(x_correlation_id=f"child-{child_conversation_id}")

    cs.start_branch_child = MagicMock(side_effect=_fake_child)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock()

    sticky_router = MagicMock()
    sticky_router.register_child_routing = MagicMock()

    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky_router
    )
    credit = MagicMock(
        x_correlation_id="root", conversation_id="c", turn_index=0, agent_depth=0
    )

    assert await orch.intercept(credit) is True
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


@pytest.mark.asyncio
async def test_no_join_case_releases_slot_when_descendants_drain():
    from aiperf.timing.branch_orchestrator import PendingBranchJoin

    cs = MagicMock()
    issuer = MagicMock()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    released: list[str] = []
    orch._release_slot = lambda p: released.append(p)

    orch._pending_joins["parent"] = PendingBranchJoin("parent", {"cA"})
    orch._child_to_parent = {"cA": "parent"}
    orch._child_modes = {"cA": ConversationBranchMode.FORK}
    orch._descendant_counts["parent"] = 2  # root terminal + 1 child

    await orch.on_child_leaf_reached("cA")
    assert "parent" not in orch._pending_joins
    assert released == ["parent"]


@pytest.mark.asyncio
async def test_leaf_for_unknown_child_is_noop():
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    await orch.on_child_leaf_reached("unknown")
    assert orch.stats.children_completed == 0


@pytest.mark.asyncio
async def test_child_error_fail_fast_aborts_parent(monkeypatch):
    from aiperf.common.environment import Environment
    from aiperf.timing.branch_orchestrator import PendingBranchJoin

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)

    issuer = MagicMock()
    issuer.abort_session = AsyncMock()
    sticky_router = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=issuer,
        sticky_router=sticky_router,
    )
    orch._pending_joins["p"] = PendingBranchJoin("p", {"c1", "c2"})
    orch._child_to_parent = {"c1": "p", "c2": "p"}
    orch._child_modes = {
        "c1": ConversationBranchMode.FORK,
        "c2": ConversationBranchMode.FORK,
    }
    orch._descendant_counts["p"] = 3

    await orch.on_child_errored("c1")
    assert orch.stats.parents_failed_due_to_child_error == 1
    assert "p" not in orch._pending_joins
    assert "p" not in orch._descendant_counts
    assert "c2" not in orch._child_to_parent
    # Refcount released for the errored child plus its orphan sibling.
    assert sticky_router.release_child_routing.call_count == 2
    # abort_session awaited for the parent and the orphan sibling.
    assert issuer.abort_session.await_count == 2
    awaited_targets = {call.args[0] for call in issuer.abort_session.await_args_list}
    assert awaited_targets == {"p", "c2"}


@pytest.mark.asyncio
async def test_dispatch_failure_rolls_back_bookkeeping():
    """When _dispatch_first_turn returns False (e.g. slots saturated), the
    orchestrator must undo its children_spawned / sticky-refcount /
    descendant-count / _child_to_parent / _pending_joins bookkeeping for the
    failed child."""
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a", "b"],
            is_background=False,
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

    assert await orch.intercept(credit) is True
    # Only the successful child stays tracked.
    assert orch.stats.children_spawned == 1
    assert orch.stats.children_errored == 1
    assert "child-a" in orch._child_to_parent
    assert "child-b" not in orch._child_to_parent
    # register_child_routing fired for both children; release fired for the one
    # that failed to dispatch.
    assert sticky_router.register_child_routing.call_count == 2
    assert sticky_router.release_child_routing.call_count == 1
    # pending join tracks only the surviving child.
    assert orch._pending_joins["root"].outstanding_children == {"child-a"}


@pytest.mark.asyncio
async def test_child_error_for_unknown_child_is_noop():
    """Late or spurious ``on_child_errored`` for a child the orchestrator is
    no longer tracking must not crash, and must not inflate stats. Fail-fast
    orphan cascades pop siblings from ``_child_to_parent`` before the
    siblings' own credits return; counting those subsequent notifications
    would double-count ``children_errored``."""
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
            is_background=False,
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

    assert await orch.intercept(credit) is True
    assert orch.stats.children_spawned == 1
    # Sticky refcount untouched for SPAWN-mode children.
    assert sticky_router.register_child_routing.call_count == 0

    # Leaf-reached must also NOT release anything because register didn't fire.
    await orch.on_child_leaf_reached("child-spawn-a")
    assert sticky_router.release_child_routing.call_count == 0


# ============================================================
# has_pending_branch_work / cleanup coverage
# ============================================================


def test_has_pending_branch_work_empty_orchestrator():
    """Fresh orchestrator has no pending state."""
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    assert orch.has_pending_branch_work() is False


def test_has_pending_branch_work_with_pending_join():
    from aiperf.timing.branch_orchestrator import PendingBranchJoin

    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch._pending_joins["p"] = PendingBranchJoin(
        parent_x_correlation_id="p",
        outstanding_children={"c"},
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
    """Child-to-parent entries alone keep has_pending True — a child
    still in flight (not yet evicted) counts as outstanding work."""
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch._child_to_parent["c"] = "p"
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
    """Any residual _pending_joins / _child_to_parent / _descendant_counts
    at cleanup time means the DAG failed to drain — cleanup logs a
    warning so diagnosis has a breadcrumb."""
    import logging

    from aiperf.timing.branch_orchestrator import PendingBranchJoin

    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    orch._pending_joins["leaky-parent"] = PendingBranchJoin(
        parent_x_correlation_id="leaky-parent",
        outstanding_children={"child-a", "child-b"},
    )
    orch._child_to_parent["child-a"] = "leaky-parent"
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
    assert orch._pending_joins == {}
    assert orch._child_to_parent == {}
    assert orch._descendant_counts == {}


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
    orch._child_to_parent["c"] = "p"
    orch.cleanup()
    # State snapshotted by cleanup was cleared, but the method must
    # also guard against re-entrancy with a direct early-return.
    orch._child_to_parent["c"] = "p"
    await orch.on_child_leaf_reached("c")
    # children_completed should NOT increment during teardown.
    assert orch.stats.children_completed == 0
