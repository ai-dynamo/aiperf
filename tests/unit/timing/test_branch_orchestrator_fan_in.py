# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 3 unit tests: fan-in (multi-prereq per gated turn).

Covers the Phase 3 semantics:

- A single gated parent turn may declare prerequisites on multiple different
  branches spawned from different parent turns. The gate only fires once ALL
  prereqs are satisfied.
- The gate is idempotent under double-delivery: the same child_corr reporting
  twice against the same prereq does not advance the counter twice.
- Rollback on dispatch failure decrements ``expected`` without touching the
  ``completed`` set. When ``expected == 0`` for every prereq, the gate fires
  immediately.
- Fail-fast cascades across orphan siblings of every contributing branch.
- FORK + SPAWN mixed branches can feed one gate with sticky refcounts
  tracked correctly per branch.
"""

from __future__ import annotations

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
    PendingBranchJoin,
    PrereqState,
)


def _mk_conv(
    cid: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo],
) -> ConversationMetadata:
    return ConversationMetadata(conversation_id=cid, turns=turns, branches=branches)


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: next(
        c for c in conversations if c.conversation_id == cid
    )
    return cs


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
    """Parent has 6 turns. Turn 0 spawns branch_A (2 children); turn 2 spawns
    branch_B (3 children). Turn 5 is gated on BOTH branches."""
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


def _mk_issuer():
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    issuer.abort_session = AsyncMock()
    return issuer


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
    """Turn 0 spawns A (2 children); turn 2 spawns B (3 children); turn 5
    gated on both. Parent progresses 0->1->2->3->4 normally (spawning A then
    B along the way) and only suspends at turn 5. All 5 children must
    complete before turn 5 fires."""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0 return: spawns A; next turn (T=1) is ungated; no suspension.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is False
    # Gate for turn 5 is now future. Both prereq_keys are pre-seeded (A and
    # B) because the gated turn declares both — but B is not yet registered.
    # A has expected=2 and registered=True after spawning.
    pending_5 = orch._future_joins["corr-root"][5]
    a_state = pending_5.outstanding["SPAWN_JOIN:root:0:A"]
    assert a_state.expected == 2
    assert a_state.registered is True
    b_state = pending_5.outstanding["SPAWN_JOIN:root:2:B"]
    assert b_state.expected == 0
    assert b_state.registered is False
    # Gate is NOT yet satisfied because B is unregistered.
    assert not pending_5.is_satisfied

    # Turn 1 return: no spawn, no gate on turn 2.
    assert await orch.intercept(_mk_credit("root", "corr-root", 1)) is False

    # Turn 2 return: spawns B; next turn (T=3) is ungated.
    assert await orch.intercept(_mk_credit("root", "corr-root", 2)) is False
    # Gate for turn 5 now has both prereqs registered.
    pending_5 = orch._future_joins["corr-root"][5]
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].expected == 2
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].expected == 3

    # Turn 3 return: no gate on turn 4.
    assert await orch.intercept(_mk_credit("root", "corr-root", 3)) is False
    # Turn 4 return: NEXT turn is T=5 which IS gated -> suspend.
    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is True
    assert orch._active_joins["corr-root"].gated_turn_index == 5
    # None of the children have completed yet; gate should NOT fire.
    issuer.dispatch_join_turn.assert_not_called()

    # Complete all A children; gate still waits on B.
    await orch.on_child_leaf_reached("corr-a1")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-a2")
    issuer.dispatch_join_turn.assert_not_called()

    # Complete two of three B children; gate still waits.
    await orch.on_child_leaf_reached("corr-b1")
    await orch.on_child_leaf_reached("corr-b2")
    issuer.dispatch_join_turn.assert_not_called()

    # Final B child completes -> gate fires.
    await orch.on_child_leaf_reached("corr-b3")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert "corr-root" not in orch._active_joins
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_fan_in_partial_satisfy_then_full_satisfy():
    """All A children complete before parent suspends; B still has one child
    outstanding when the parent reaches turn 5. Gate must stay active."""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))  # spawn A
    # A finishes before the parent progresses further.
    await orch.on_child_leaf_reached("corr-a1")
    await orch.on_child_leaf_reached("corr-a2")
    # Gate is still future; A's prereq is done but B hasn't been registered.
    pending_5 = orch._future_joins["corr-root"][5]
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].is_done

    await orch.intercept(_mk_credit("root", "corr-root", 1))
    await orch.intercept(_mk_credit("root", "corr-root", 2))  # spawn B
    # Both prereqs now registered; A is done, B is outstanding.
    pending_5 = orch._future_joins["corr-root"][5]
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].is_done
    assert not pending_5.outstanding["SPAWN_JOIN:root:2:B"].is_done

    await orch.intercept(_mk_credit("root", "corr-root", 3))
    # Two of three B children complete before suspension.
    await orch.on_child_leaf_reached("corr-b1")
    await orch.on_child_leaf_reached("corr-b2")
    # Parent suspends at T=5.
    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is True
    issuer.dispatch_join_turn.assert_not_called()

    # Final B child completes -> gate fires.
    await orch.on_child_leaf_reached("corr-b3")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_fan_in_three_way_with_fork_and_spawn_mixed():
    """Mix FORK and SPAWN: turn 0 spawns FORK branch F (2 children); turn 1
    spawns SPAWN branch S (2 children); turn 3 gated on both. Sticky
    refcounts registered only for FORK children."""
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

    # Turn 0: spawn F (FORK); 2 sticky refcounts registered.
    await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert sticky.register_child_routing.call_count == 2
    # Turn 1: spawn S (SPAWN); no sticky registration.
    await orch.intercept(_mk_credit("root", "corr-root", 1))
    assert sticky.register_child_routing.call_count == 2

    # Turn 2 return -> T=3 is gated; parent suspends.
    assert await orch.intercept(_mk_credit("root", "corr-root", 2)) is True

    # Complete all children; FORK releases refcounts per-child.
    for cid in ("f1", "f2"):
        await orch.on_child_leaf_reached(f"corr-{cid}")
    # F prereq done, S still outstanding -> gate waits.
    issuer.dispatch_join_turn.assert_not_called()
    assert sticky.release_child_routing.call_count == 2

    for cid in ("s1", "s2"):
        await orch.on_child_leaf_reached(f"corr-{cid}")
    issuer.dispatch_join_turn.assert_awaited_once()
    # SPAWN children never triggered sticky release.
    assert sticky.release_child_routing.call_count == 2


@pytest.mark.asyncio
async def test_fan_in_idempotent_on_double_delivery():
    """Calling _satisfy_prerequisite twice for the same child_corr on the
    same prereq must not advance the counter twice. The gate must fire
    only after every child actually completes."""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Fast-forward to suspension at turn 5.
    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 1))
    await orch.intercept(_mk_credit("root", "corr-root", 2))
    await orch.intercept(_mk_credit("root", "corr-root", 3))
    await orch.intercept(_mk_credit("root", "corr-root", 4))
    assert "corr-root" in orch._active_joins

    # All A children report; then b1 reports THREE times. Gate must not fire.
    await orch.on_child_leaf_reached("corr-a1")
    await orch.on_child_leaf_reached("corr-a2")
    await orch.on_child_leaf_reached("corr-b1")
    # Re-deliver b1 completion directly through _satisfy_prerequisite.
    result = await orch._satisfy_prerequisite(
        "corr-root", 5, "SPAWN_JOIN:root:2:B", "corr-b1"
    )
    assert result is None, "duplicate delivery must return None"
    # Still only 1 B child completed.
    state = orch._active_joins["corr-root"].outstanding["SPAWN_JOIN:root:2:B"]
    assert len(state.completed) == 1
    issuer.dispatch_join_turn.assert_not_called()

    # Complete the remaining B children; gate fires once.
    await orch.on_child_leaf_reached("corr-b2")
    await orch.on_child_leaf_reached("corr-b3")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_fan_in_under_fail_fast_cascades_correctly(monkeypatch):
    """AIPERF_DAG_FAIL_FAST=true: one B child errors. Parent + every orphan
    in BOTH A and B is aborted; both branches' gate state is dropped."""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    assert orch._fail_fast is True

    await orch.intercept(_mk_credit("root", "corr-root", 0))  # spawn A
    await orch.intercept(_mk_credit("root", "corr-root", 1))
    await orch.intercept(_mk_credit("root", "corr-root", 2))  # spawn B
    # At this point all 5 children are tracked.
    assert {f"corr-{c}" for c in ("a1", "a2", "b1", "b2", "b3")} <= set(
        orch._child_to_join.keys()
    )

    # b2 errors. Fail-fast path aborts parent + every orphan.
    await orch.on_child_errored("corr-b2")
    # Parent aborted.
    issuer.abort_session.assert_any_await("corr-root")
    # Every orphan sibling (a1, a2, b1, b3) aborted.
    aborted = {call.args[0] for call in issuer.abort_session.await_args_list}
    assert {"corr-a1", "corr-a2", "corr-b1", "corr-b3"} <= aborted
    # Parent's join state cleared from both active AND future maps.
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins
    # Stats.
    assert orch.stats.parents_failed_due_to_child_error == 1


@pytest.mark.asyncio
async def test_fan_in_rollback_decrements_expected_not_completed():
    """A partial dispatch failure for one branch feeding a fan-in gate
    decrements that prereq's ``expected`` count without touching
    ``completed``. Other branches' prereq state is untouched."""
    cs = _mk_source(_fan_in_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)
    issuer = MagicMock()

    # A children dispatch successfully; b2 dispatch fails (returns False).
    async def _dispatch(session):
        return session.x_correlation_id != "corr-b2"

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))  # spawn A
    await orch.intercept(_mk_credit("root", "corr-root", 2))  # spawn B (b2 rolls back)

    pending_5 = orch._future_joins["corr-root"][5]
    # A prereq still expects 2 children.
    a_state = pending_5.outstanding["SPAWN_JOIN:root:0:A"]
    assert a_state.expected == 2
    assert a_state.completed == set()
    # B prereq initially expected 3; b2 rolled back -> 2.
    b_state = pending_5.outstanding["SPAWN_JOIN:root:2:B"]
    assert b_state.expected == 2
    assert b_state.completed == set()

    # b2 is NOT in child_to_join (rolled back).
    assert "corr-b2" not in orch._child_to_join
    assert "corr-b1" in orch._child_to_join
    assert "corr-b3" in orch._child_to_join


@pytest.mark.asyncio
async def test_fan_in_same_turn_gates_dont_collide_across_branches():
    """Different branches contribute to the same ``gated_turn_index``.
    Each branch gets its own ``prereq_key`` entry; they do not clobber each
    other's expected counter on registration. Pre-seed marks BOTH keys
    present from gate creation; registered flips True per branch-spawn."""
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))  # spawn A
    pending_5 = orch._future_joins["corr-root"][5]
    # Both keys pre-seeded; A registered, B not yet.
    assert set(pending_5.outstanding) == {
        "SPAWN_JOIN:root:0:A",
        "SPAWN_JOIN:root:2:B",
    }
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].expected == 2
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].registered is True
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].expected == 0
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].registered is False

    await orch.intercept(_mk_credit("root", "corr-root", 2))  # spawn B
    pending_5 = orch._future_joins["corr-root"][5]
    assert pending_5.outstanding["SPAWN_JOIN:root:0:A"].expected == 2
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].expected == 3
    assert pending_5.outstanding["SPAWN_JOIN:root:2:B"].registered is True


@pytest.mark.asyncio
async def test_is_satisfied_empty_gate_is_true():
    """A PendingBranchJoin with no prereqs is trivially satisfied (vacuous
    truth: ``all(...)`` over empty iterable)."""
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
    # Over-delivery (defensive) keeps is_done True.
    s.completed.add("d")
    assert s.is_done
    # Unregistered prereqs (even with expected==0) are NOT done — a future
    # spawning turn may increment expected.
    unreg = PrereqState(expected=0, registered=False)
    assert not unreg.is_done


@pytest.mark.asyncio
async def test_fan_in_multi_consumer_same_branch_multiple_gates():
    """Phase 3: a single branch feeding prereqs on two different gated
    turns. Each gate installs an independent PendingBranchJoin entry."""
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

    # Prereq index should have entries for BOTH gated turns keyed by
    # (conv_id, spawning_turn_idx=0).
    entries = orch._prereq_index[("root", 0)]
    gated_idxs = {gated_idx for _, gated_idx, _ in entries}
    assert gated_idxs == {1, 2}

    # Turn 0 return: spawn creates future joins for turn 1 AND turn 2. The
    # single child c1 is registered under both gates.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    # Active join is the nearest gated turn (T=1).
    assert orch._active_joins["corr-root"].gated_turn_index == 1
    # Future join for T=2 still present.
    assert 2 in orch._future_joins["corr-root"]


def test_pending_branch_join_outstanding_is_prereq_state_shape():
    """Shape regression: PendingBranchJoin.outstanding values are
    PrereqState instances (Phase 3 counter+set form)."""
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
    """A child that contributes to a fan-in gate has ONE ChildJoinEntry
    pointing at its (gated_turn_idx, prereq_key). Fan-in is achieved by
    multiple prereq entries on the same gate, not by multiple child entries.
    """
    cs = _mk_source(_fan_in_metadata())
    _mk_start(cs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))  # A
    await orch.intercept(_mk_credit("root", "corr-root", 2))  # B

    assert isinstance(orch._child_to_join["corr-a1"], list)
    assert len(orch._child_to_join["corr-a1"]) == 1
    assert orch._child_to_join["corr-a1"][0].prereq_key == "SPAWN_JOIN:root:0:A"
    assert orch._child_to_join["corr-a1"][0].gated_turn_index == 5
    assert orch._child_to_join["corr-b1"][0].prereq_key == "SPAWN_JOIN:root:2:B"
    assert orch._child_to_join["corr-b1"][0].gated_turn_index == 5
