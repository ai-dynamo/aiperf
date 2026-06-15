# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BranchOrchestrator skeleton + sticky-routing refcount hooks."""

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
    PendingBranchJoin,
    PrereqState,
)
from aiperf.timing.trajectory_source import ConversationState


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
async def test_seed_snapshot_registers_active_join_and_releases_parent():
    branch_id = "b0"
    parent_meta = ConversationMetadata(
        conversation_id="parent",
        turns=[
            TurnMetadata(timestamp_ms=0.0),
            TurnMetadata(timestamp_ms=12000.0, branch_ids=[branch_id]),
            TurnMetadata(
                timestamp_ms=20000.0,
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN,
                        branch_id=branch_id,
                    )
                ],
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id=branch_id,
                child_conversation_ids=["child"],
                mode=ConversationBranchMode.SPAWN,
                start_timestamp_ms=13000.0,
            )
        ],
    )
    child_meta = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(timestamp_ms=13000.0), TurnMetadata(timestamp_ms=14000.0)],
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
            return {
                "parent": parent_meta,
                "child": child_meta,
            }[conversation_id]

    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=_Source(), credit_issuer=issuer)
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
            branch_mode=ConversationBranchMode.SPAWN,
        ),
    )

    orch.seed_snapshot(states, cache_bust_markers={"parent-corr": "marker"})

    assert orch.has_pending_branch_work() is True
    assert "parent-corr" in orch._active_joins
    pending = orch._active_joins["parent-corr"]
    assert pending.parent_cache_bust_marker == "marker"
    assert pending.outstanding[f"SPAWN_JOIN:{branch_id}"].expected == 1

    await orch.on_child_leaf_reached("child-corr")

    issuer.dispatch_join_turn.assert_awaited_once()
    released = issuer.dispatch_join_turn.await_args.args[0]
    assert released.parent_x_correlation_id == "parent-corr"
    assert released.gated_turn_index == 2


def _two_branch_join_meta() -> tuple[
    ConversationMetadata, dict[str, ConversationMetadata]
]:
    """Parent with branches A (spawned turn 1) and B (spawned turn 2), both
    gating turn 3. Children are one-turn SPAWN conversations."""
    parent_meta = ConversationMetadata(
        conversation_id="parent",
        turns=[
            TurnMetadata(timestamp_ms=0.0),
            TurnMetadata(timestamp_ms=10_000.0, branch_ids=["A"]),
            TurnMetadata(timestamp_ms=20_000.0, branch_ids=["B"]),
            TurnMetadata(
                timestamp_ms=30_000.0,
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="A"),
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="B"),
                ],
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id="A",
                child_conversation_ids=["a_child"],
                mode=ConversationBranchMode.SPAWN,
                start_timestamp_ms=11_000.0,
            ),
            ConversationBranchInfo(
                branch_id="B",
                child_conversation_ids=["b_child"],
                mode=ConversationBranchMode.SPAWN,
                start_timestamp_ms=21_000.0,
            ),
        ],
    )
    children = {
        cid: ConversationMetadata(
            conversation_id=cid,
            turns=[TurnMetadata(timestamp_ms=ts)],
            is_root=False,
            agent_depth=1,
            parent_conversation_id="parent",
        )
        for cid, ts in [("a_child", 11_000.0), ("b_child", 21_000.0)]
    }
    return parent_meta, children


def _source_for(parent_meta, children):
    class _Source:
        dataset_metadata = DatasetMetadata(
            conversations=[parent_meta, *children.values()],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        def get_metadata(self, conversation_id):
            return {"parent": parent_meta, **children}[conversation_id]

    return _Source()


@pytest.mark.asyncio
async def test_seed_snapshot_prereq_completed_before_t_star_does_not_wedge_gate():
    """A prereq group that fully completed before t* must not wedge the join.

    Two branches share the turn-3 gate. t* falls between their completions:
    branch A's child finished pre-t* (absent from the snapshot) while branch
    B's child is live. A's spawning turn fired before the parent's resume
    position and can never re-fire during replay, so its prereq key must be
    seeded as satisfied - otherwise the gate is permanently unsatisfiable and
    the parent lane silently wedges for the entire phase.
    """
    parent_meta, children = _two_branch_join_meta()
    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(
        conversation_source=_source_for(parent_meta, children), credit_issuer=issuer
    )
    states = (
        ConversationState(
            conversation_id="parent",
            x_correlation_id="parent-corr",
            next_turn_index=3,
            waiting_on_children=True,
            join_target_turn_index=3,
        ),
        ConversationState(
            conversation_id="b_child",
            x_correlation_id="b-corr",
            next_turn_index=0,
            agent_depth=1,
            parent_correlation_id="parent-corr",
            join_target_turn_index=3,
            branch_id="B",
            branch_mode=ConversationBranchMode.SPAWN,
        ),
    )
    orch.seed_snapshot(states)

    pending = orch._active_joins["parent-corr"]
    assert pending.outstanding["SPAWN_JOIN:A"].is_done, (
        "Branch A completed before t* but its prereq was seeded unsatisfiable"
    )
    assert not pending.is_satisfied  # B's live child is still outstanding.

    await orch.on_child_leaf_reached("b-corr")

    issuer.dispatch_join_turn.assert_awaited_once()
    released = issuer.dispatch_join_turn.await_args.args[0]
    assert released.parent_x_correlation_id == "parent-corr"
    assert released.gated_turn_index == 3
    assert "parent-corr" not in orch._active_joins


@pytest.mark.asyncio
async def test_seed_snapshot_keeps_unfired_future_prereq_unregistered():
    """A prereq whose spawning turn replays after t* must stay unregistered.

    The parent resumes at turn 2 (B's spawning turn), so branch B will fire
    during replay: seeding it as satisfied would release the gate before B's
    children even spawn. Only branches whose spawning turn fired before the
    parent's resume position may be auto-satisfied.
    """
    parent_meta, children = _two_branch_join_meta()
    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(
        conversation_source=_source_for(parent_meta, children), credit_issuer=issuer
    )
    states = (
        ConversationState(
            conversation_id="parent",
            x_correlation_id="parent-corr",
            next_turn_index=2,
        ),
        ConversationState(
            conversation_id="a_child",
            x_correlation_id="a-corr",
            next_turn_index=0,
            agent_depth=1,
            parent_correlation_id="parent-corr",
            join_target_turn_index=3,
            branch_id="A",
            branch_mode=ConversationBranchMode.SPAWN,
        ),
    )
    orch.seed_snapshot(states)

    pending = orch._future_joins["parent-corr"][3]
    assert pending.outstanding["SPAWN_JOIN:A"].expected == 1
    assert not pending.outstanding["SPAWN_JOIN:B"].registered

    await orch.on_child_leaf_reached("a-corr")

    # A is done but B has not even spawned yet: the gate must hold.
    issuer.dispatch_join_turn.assert_not_awaited()
    assert orch._future_joins["parent-corr"][3] is pending


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
async def test_child_error_decrements_join_when_not_fail_fast(
    monkeypatch, force_fail_fast
):
    force_fail_fast(False)

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
async def test_child_error_fail_fast_aborts_parent(monkeypatch, force_fail_fast):
    force_fail_fast(True)

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
async def test_dispatch_failure_rolls_back_bookkeeping():
    """When _dispatch_first_turn returns False due to a stop condition, the
    orchestrator must undo its children_spawned / sticky-refcount /
    descendant-count / _child_to_join bookkeeping for the failed child."""
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
        **kwargs,
    ):
        return MagicMock(x_correlation_id=f"child-{child_conversation_id}")

    cs.start_branch_child = MagicMock(side_effect=_fake_child)

    issuer = MagicMock()

    # First dispatch succeeds; the second is refused by the issuer.
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
    # ``dispatch_first_turn`` returning False is stop-condition refusal,
    # not an error — tally as truncated.
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


def test_marker_for_root_resolves_tree_marker_from_ledger():
    """A spawned descendant reuses its tree root's cache-bust marker (keyed by
    root_correlation_id) rather than minting its own per-child marker."""
    from aiperf.common.enums import CacheBustTarget
    from aiperf.timing.trajectory_source import CacheBustLedger

    ledger = CacheBustLedger()
    ledger.session_marker["ROOT"] = "[rid:deadbeefcafe]\n\n"
    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=MagicMock(),
        benchmark_id="b",
        cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
        cache_bust_ledger=ledger,
    )
    assert orch._marker_for_root("ROOT") == "[rid:deadbeefcafe]\n\n"
    assert orch._marker_for_root("UNKNOWN-ROOT") is None
    assert orch._marker_for_root(None) is None


def test_marker_for_root_none_when_cache_bust_disabled():
    from aiperf.common.enums import CacheBustTarget
    from aiperf.timing.trajectory_source import CacheBustLedger

    ledger = CacheBustLedger()
    ledger.session_marker["ROOT"] = "[rid:abc]"
    orch = BranchOrchestrator(
        conversation_source=MagicMock(),
        credit_issuer=MagicMock(),
        cache_bust_target=CacheBustTarget.NONE,
        cache_bust_ledger=ledger,
    )
    assert orch._marker_for_root("ROOT") is None
