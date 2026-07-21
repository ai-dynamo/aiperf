# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial probes for the DAG ``BranchOrchestrator``.

These tests target accounting and ordering anomalies NOT covered by
``test_branch_orchestrator.py`` or ``test_branch_orchestrator_adversarial_full.py``:

Confirmed bugs (xfail strict):
  * Delayed-join (K>1) gates that become zero-outstanding because EVERY child
    failed to start dispatch the gated turn IMMEDIATELY on the spawning turn's
    return, skipping the parent's intervening turns (turn-ordering violation).
    Probed via both the start_branch_child-raises path and the
    dispatch-refused (gather-False) path.
  * A FORK child seeded through ``seed_snapshot`` never has its sticky refcount
    incremented (seed does not touch the router), yet ``_handle_child_done``
    releases it once on leaf — a net sticky refcount underflow.

Passing characterizations (surprising-but-current behavior):
  * Duplicate intercept of the same spawning-turn credit double-dispatches the
    branch's children (no per-turn idempotency guard).
  * Full rollback of a background branch (all children refused) leaves zero
    leaked accounting state.
  * ``_notify_drain`` fires the observer on the last child's drain, and
    ``has_pending_branch_work`` flips False exactly then.
  * leaf-then-error and stopped-then-error double delivery for one child is
    counted exactly once (the first hook wins; the second is a no-op).
  * Seeding a child whose parent state is absent tracks it as an ungated
    descendant and drains cleanly on leaf.
  * An over-completed prereq (more distinct children report than expected)
    never reports a negative ``total_outstanding``.
"""

from __future__ import annotations

import itertools
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
)
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

# ---------------------------------------------------------------------------
# Shared helpers (mirror the style of test_branch_orchestrator_adversarial_full)
# ---------------------------------------------------------------------------


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


def _mk_source(conversations: list[ConversationMetadata], *, unique_children=False):
    """Build a MagicMock conversation source.

    ``unique_children`` mints a fresh correlation id on every
    ``start_branch_child`` call (mirrors the real ConversationSource's uuid4),
    so duplicate dispatch produces two distinct tracked children rather than
    silently colliding on one id.
    """
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: next(
        c for c in conversations if c.conversation_id == cid
    )
    counter = itertools.count()

    def _start_branch(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        suffix = f"-{next(counter)}" if unique_children else ""
        s.x_correlation_id = f"corr-{child_conversation_id}{suffix}"
        s.conversation_id = child_conversation_id
        return s

    cs.start_branch_child = MagicMock(side_effect=_start_branch)
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


def _mk_issuer():
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    issuer.abort_session = AsyncMock()
    return issuer


def _delayed_join_root() -> ConversationMetadata:
    """Root: branch spawns on turn 0, join gated on turn 3 (K=3, delayed)."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    return _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"]),
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


def _seed_fork_conv():
    """Parent (3 turns) + one FORK child gating turn 2, for seed_snapshot."""
    branch_id = "b0"
    parent_meta = ConversationMetadata(
        conversation_id="parent",
        turns=[
            TurnMetadata(),
            TurnMetadata(branch_ids=[branch_id], has_forks=True),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id=branch_id
                    )
                ]
            ),
        ],
        branches=[
            ConversationBranchInfo(
                branch_id=branch_id,
                child_conversation_ids=["child"],
                mode=ConversationBranchMode.FORK,
                start_timestamp_ms=13000.0,
            )
        ],
    )
    child_meta = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(), TurnMetadata()],
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
            return {"parent": parent_meta, "child": child_meta}[conversation_id]

    return _Source(), branch_id


# ---------------------------------------------------------------------------
# 1. CONFIRMED BUG: all-children-fail-to-start on a delayed (K>1) gate
#    dispatches the gated turn out of order.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delayed_join_all_children_raise_does_not_dispatch_gate_early():
    """The branch spawns on turn 0; the join is gated on turn 3. Every
    ``start_branch_child`` raises, so the gate becomes vacuously satisfied.

    Correct behavior: the parent has NOT reached the turn before the gate, so
    no join turn should fire yet — the gate must be popped silently and the
    parent should advance turns 1, 2 normally before reaching turn 3. The
    orchestrator instead dispatches turn 3 immediately, out of order.
    """
    cs = _mk_source([_delayed_join_root(), _mk_conv("c1", [TurnMetadata()], [])])
    cs.start_branch_child = MagicMock(side_effect=RuntimeError("start failed"))
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 0))
    # Parent's next turn (idx 1) is NOT the gate; it must not suspend.
    assert suspended is False
    # INVARIANT: the gated turn (idx 3) must not be dispatched before the
    # parent reaches it.
    issuer.dispatch_join_turn.assert_not_called()


# ---------------------------------------------------------------------------
# 2. CONFIRMED BUG: same as (1) but via the dispatch-refused (gather-False)
#    rollback path rather than an exception.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delayed_join_all_children_refused_does_not_dispatch_gate_early():
    """Identical ordering invariant to test 1, exercised through the
    rollback (gather returns False) branch instead of the exception branch."""
    cs = _mk_source([_delayed_join_root(), _mk_conv("c1", [TurnMetadata()], [])])
    issuer = _mk_issuer()
    issuer.dispatch_first_turn = AsyncMock(return_value=False)  # refused
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert suspended is False
    issuer.dispatch_join_turn.assert_not_called()


# ---------------------------------------------------------------------------
# 3. CONFIRMED BUG: seed_snapshot FORK child -> sticky refcount underflow.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_snapshot_fork_child_sticky_release_is_balanced():
    """A snapshot-replayed FORK child must have balanced sticky refcount
    operations: register count == release count. seed_snapshot omits the
    register, so the leaf release underflows by one."""
    source, branch_id = _seed_fork_conv()
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=source, credit_issuer=issuer, sticky_router=sticky
    )
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
            branch_mode=ConversationBranchMode.FORK,
        ),
    )
    orch.seed_snapshot(states)

    await orch.on_child_leaf_reached("child-corr")

    # INVARIANT: balanced refcount accounting for the FORK child.
    assert (
        sticky.release_child_routing.call_count
        == sticky.register_child_routing.call_count
    )


# ---------------------------------------------------------------------------
# 4. CHARACTERIZATION: duplicate spawning-turn credit double-dispatches.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_spawning_turn_credit_double_dispatches_children():
    """The orchestrator has no per-turn idempotency guard: delivering the
    SAME spawning-turn credit twice spawns the branch's children twice. With
    realistic fresh-uuid children each dispatch is a distinct session, so the
    branch fans out 2x. Documented as current behavior (upstream is expected
    to deliver each credit return exactly once)."""
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
    cs = _mk_source([root, _mk_conv("c1", [TurnMetadata()], [])], unique_children=True)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 0))

    # Two distinct child dispatches for one logical branch (no de-dup).
    assert cs.start_branch_child.call_count == 2
    assert issuer.dispatch_first_turn.await_count == 2
    assert len(orch._child_to_join) == 2


# ---------------------------------------------------------------------------
# 5. CHARACTERIZATION: full rollback of a background branch leaves no leak.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_branch_all_children_refused_drains_clean():
    """A background (ungated) branch whose children are ALL refused dispatch
    must fully roll back: children_spawned back to 0, children_truncated
    tallied, and no leaked descendant/child-tracking state."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1", "c2"],
        mode=ConversationBranchMode.SPAWN,
        is_background=True,
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:0"]), TurnMetadata()],
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
    issuer.dispatch_first_turn = AsyncMock(return_value=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspended = await orch.intercept(_mk_credit("root", "corr-root", 0))
    assert suspended is False
    assert orch.stats.children_spawned == 0
    assert orch.stats.children_truncated == 2
    assert orch.stats.children_errored == 0
    assert orch._descendant_counts == {}
    assert orch._child_to_join == {}
    assert orch.has_pending_branch_work() is False


# ---------------------------------------------------------------------------
# 6. CHARACTERIZATION: _notify_drain fires on the last child's drain.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_observer_fires_when_last_ungated_child_drains():
    """The drain observer is the race-closing hook the callback handler relies
    on. Verify it fires on the child-completion that empties the orchestrator,
    and that ``has_pending_branch_work`` is False at that moment (so the
    deferred all-credits-returned event can latch)."""
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=MagicMock()
    )
    observed: list[bool] = []
    orch.set_drain_observer(lambda: observed.append(orch.has_pending_branch_work()))

    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="parent", gated_turn_index=None, prereq_key=None
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.SPAWN}
    orch._descendant_counts["parent"] = 1  # only the one ungated child

    await orch.on_child_leaf_reached("cA")

    # Observer was invoked, and by the time it ran the orchestrator had drained.
    assert observed, "drain observer must fire on the draining completion"
    assert observed[-1] is False
    assert orch.has_pending_branch_work() is False


# ---------------------------------------------------------------------------
# 7. CHARACTERIZATION: leaf-then-error double delivery counts once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leaf_then_error_double_delivery_counts_child_once(force_fail_fast):
    """A worker that delivers BOTH a leaf and an error for the same child must
    not double-count or re-run cleanup. The first hook pops _child_to_join; the
    second finds nothing and is a no-op. So children_completed==1 and
    children_errored==0, and no fail-fast cascade fires on the stale second
    delivery."""
    force_fail_fast(True)
    issuer = _mk_issuer()
    sticky = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=MagicMock(), credit_issuer=issuer, sticky_router=sticky
    )
    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=1, completed=set(), registered=True
    )
    pending.is_blocked = True
    orch._active_joins["p"] = pending
    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=1, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.SPAWN}
    orch._descendant_counts["p"] = 1

    await orch.on_child_leaf_reached("cA")
    await orch.on_child_errored("cA")  # stale duplicate

    assert orch.stats.children_completed == 1
    assert orch.stats.children_errored == 0
    # No fail-fast abort on the stale second delivery.
    issuer.abort_session.assert_not_awaited()
    assert orch.stats.parents_failed_due_to_child_error == 0
    # Join fired exactly once on the first (leaf) delivery.
    assert issuer.dispatch_join_turn.await_count == 1


# ---------------------------------------------------------------------------
# 8. CHARACTERIZATION: stopped-then-error double delivery counts once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stopped_then_error_double_delivery_counts_child_once():
    """A child first cap-stopped (truncated) then later erroring must be
    counted once: children_truncated==1, children_errored==0 (the error hook
    finds an already-drained child)."""
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=MagicMock(), credit_issuer=issuer)
    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=None, prereq_key=None
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.SPAWN}
    orch._descendant_counts["p"] = 1

    await orch.on_child_stopped("cA")
    await orch.on_child_errored("cA")  # stale

    assert orch.stats.children_truncated == 1
    assert orch.stats.children_errored == 0


# ---------------------------------------------------------------------------
# 9. CHARACTERIZATION: seeded child with no parent state is ungated + drains.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_seed_snapshot_orphan_child_without_parent_state_drains_clean():
    """When a snapshot includes a child whose parent state is absent (parent
    already finished before t*), the child is tracked as an ungated descendant
    (gated_turn_index=None) and decrements cleanly on leaf with no join
    dispatch and no leaked descendant count."""
    child_meta = ConversationMetadata(
        conversation_id="child",
        turns=[TurnMetadata(), TurnMetadata()],
        is_root=False,
        agent_depth=1,
        parent_conversation_id="parent",
    )

    class _Source:
        dataset_metadata = DatasetMetadata(
            conversations=[child_meta],
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        def get_metadata(self, conversation_id):
            return {"child": child_meta}[conversation_id]

    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=_Source(), credit_issuer=issuer)
    states = (
        # No state for "parent-corr" at all.
        ConversationState(
            conversation_id="child",
            x_correlation_id="child-corr",
            next_turn_index=1,
            agent_depth=1,
            parent_correlation_id="parent-corr",
            join_target_turn_index=1,
            branch_id="b0",
            branch_mode=ConversationBranchMode.SPAWN,
        ),
    )
    orch.seed_snapshot(states)

    # Tracked as ungated (no parent_state -> prereq_key None).
    entries = orch._child_to_join["child-corr"]
    assert len(entries) == 1
    assert entries[0].prereq_key is None
    assert orch._descendant_counts["parent-corr"] == 1
    assert orch.has_pending_branch_work() is True

    await orch.on_child_leaf_reached("child-corr")

    issuer.dispatch_join_turn.assert_not_called()
    assert "parent-corr" not in orch._descendant_counts
    assert orch.has_pending_branch_work() is False


# ---------------------------------------------------------------------------
# 10. CHARACTERIZATION: over-completed prereq never reports negative
#     total_outstanding.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_over_completed_prereq_total_outstanding_clamped_non_negative():
    """If more distinct children report against a prereq than its ``expected``
    counter (rollback shrank ``expected`` after a sibling already completed),
    ``total_outstanding`` must clamp at 0, never go negative, and the gate is
    considered done. Probes the max(0, ...) guard in
    ``PendingBranchJoin.total_outstanding`` and the >= comparison in
    ``PrereqState.is_done``."""
    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=2,
        gated_turn_index=1,
    )
    # expected=1 but two distinct children landed (over-completed).
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=1, completed={"cA", "cB"}, registered=True
    )

    assert pending.total_outstanding == 0  # clamped, not -1
    assert pending.outstanding["SPAWN_JOIN:b"].is_done is True
    assert pending.is_satisfied is True


# ---------------------------------------------------------------------------
# 11. CHARACTERIZATION: cleanup mid-drain leaves has_pending_branch_work False
#     and a late child completion is a no-op.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleanup_mid_drain_then_late_child_is_noop():
    """Cleanup is called while children are still tracked (DAG abandoned, e.g.
    worker crash / cancellation). The orchestrator clears state, reports no
    pending work, and a late child-leaf delivery after cleanup is a silent
    no-op (does not resurrect tracking or bump stats)."""
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=MagicMock(), credit_issuer=issuer)
    pending = PendingBranchJoin(
        parent_x_correlation_id="p",
        parent_conversation_id="c",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    pending.outstanding["SPAWN_JOIN:b"] = PrereqState(
        expected=2, completed=set(), registered=True
    )
    orch._active_joins["p"] = pending
    orch._child_to_join["cA"] = [
        ChildJoinEntry(
            parent_correlation_id="p", gated_turn_index=2, prereq_key="SPAWN_JOIN:b"
        )
    ]
    orch._child_modes = {"cA": ConversationBranchMode.SPAWN}
    orch._descendant_counts["p"] = 2

    orch.cleanup()
    assert orch.has_pending_branch_work() is False

    await orch.on_child_leaf_reached("cA")
    issuer.dispatch_join_turn.assert_not_called()
    assert orch.stats.children_completed == 0


# ---------------------------------------------------------------------------
# 12. CHARACTERIZATION: non-fail-fast child error on a multi-consumer branch
#     fires every satisfied gate but counts the error once.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_fail_fast_error_on_sole_child_fires_all_gates_once(
    force_fail_fast,
):
    """A single SPAWN child feeds three gated turns (1, 2, 3). The child errors
    (non-fail-fast), which is treated as leaf-reached: the nearest gate (active
    at turn 1) dispatches, the future gates at 2/3 are popped as satisfied, and
    children_errored increments exactly once."""
    force_fail_fast(False)
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

    await orch.on_child_errored("corr-c1")

    assert orch.stats.children_errored == 1
    # Active gate fired once; the future gates were popped (satisfied early).
    assert issuer.dispatch_join_turn.await_count == 1
    assert "corr-root" not in orch._active_joins
    assert orch._future_joins.get("corr-root", {}) == {}
