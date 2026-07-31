# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 1 unit tests for delayed joins in :class:`BranchOrchestrator`.

Covers the delayed-join semantics:

- K>1 delayed joins: parent runs turns [spawn+1 .. gate-1] without suspension
  and suspends only when it's about to dispatch the gated turn.
- Children finishing before the parent arrives pop the future gate and the
  parent breezes through with no suspension.
- K=1 (legacy) behavior still works under the new architecture.
- Stop conditions during the gap propagate to ``joins_suppressed``.
- Fail-fast aborts parent + orphan siblings mid-gap.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase, PrerequisiteKind
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.trajectory_source import ConversationState
from tests.unit.timing._shared_helpers import _mk_conv, _mk_source


def _mk_credit(conv_id: str, corr_id: str, turn_index: int):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=0,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )


def _k5_metadata() -> list[ConversationMetadata]:
    """Parent conv with 6 turns: spawn on turn 0, gate on turn 5 (K=5)."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c0", "c1"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:0"]),
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
    c0 = _mk_conv("c0", [TurnMetadata()], [])
    c1 = _mk_conv("c1", [TurnMetadata()], [])
    return [root, c0, c1]


def _timed_k1_metadata(
    *,
    delay_ms: float | None = 198_046.0,
) -> list[ConversationMetadata]:
    """Parent and child matching the two-readiness join shape from #1231.

    Times are normalized by subtracting the source root start (110.893s):
    the previous parent ends at 31.042s, the child ends at 47.658s, and the
    gated parent is due at 229.088s. Thus the parent's ordinary end-to-start
    delay is 198.046s and its source-aligned post-child residual is 181.430s.
    """
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(
                timestamp_ms=0.0,
                api_time_ms=31_042.0,
                branch_ids=[branch.branch_id],
            ),
            TurnMetadata(
                timestamp_ms=229_088.0,
                delay_ms=delay_ms,
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN,
                        branch_id=branch.branch_id,
                    )
                ],
            ),
        ],
        [branch],
    )
    child = _mk_conv(
        "child",
        [TurnMetadata(timestamp_ms=21_181.0, api_time_ms=26_477.0)],
        [],
    )
    return [root, child]


def _timed_join_orchestrator(
    *,
    delay_ms: float | None = 198_046.0,
    scheduler: MagicMock | LoopScheduler | None = None,
    allow_accelerated_warmup: bool = False,
) -> tuple[BranchOrchestrator, MagicMock, MagicMock]:
    cs = _mk_source(_timed_k1_metadata(delay_ms=delay_ms))
    child_session = MagicMock(
        x_correlation_id="corr-child",
        metadata=cs.get_metadata("child"),
        effective_root_correlation_id="corr-root",
    )
    cs.start_branch_child.return_value = child_session
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    scheduler = scheduler or MagicMock()
    orchestrator = BranchOrchestrator(
        conversation_source=cs,
        credit_issuer=issuer,
        allow_accelerated_warmup=allow_accelerated_warmup,
        scheduler=scheduler,
    )
    return orchestrator, issuer, scheduler


@pytest.mark.asyncio
async def test_join_fast_child_waits_for_recorded_parent_deadline() -> None:
    """A fast child cannot erase the parent's recorded residual think time."""
    orch, issuer, scheduler = _timed_join_orchestrator()

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    delay_s, deadline_coro = scheduler.schedule_later.call_args.args
    assert delay_s == pytest.approx(198.046)
    metadata = orch._cs.get_metadata("root")
    child = orch._cs.get_metadata("child")
    source_child_end_ms = child.turns[0].timestamp_ms + child.turns[0].api_time_ms
    assert metadata.turns[1].timestamp_ms - source_child_end_ms == pytest.approx(
        181_430.0
    )

    await orch.on_child_leaf_reached("corr-child")
    issuer.dispatch_join_turn.assert_not_awaited()
    assert orch._active_joins["corr-root"].is_satisfied

    await deadline_coro
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_join_slow_child_releases_at_child_completion_without_extra_delay() -> (
    None
):
    """A child finishing after the deadline releases immediately, not delay later."""
    orch, issuer, scheduler = _timed_join_orchestrator()

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    deadline_coro = scheduler.schedule_later.call_args.args[1]
    await deadline_coro
    issuer.dispatch_join_turn.assert_not_awaited()

    await orch.on_child_leaf_reached("corr-child")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_accelerated_warmup_compresses_join_time_but_keeps_child_gate() -> None:
    """Zero-idle warmup removes replay time, never the subagent dependency."""
    orch, issuer, scheduler = _timed_join_orchestrator(allow_accelerated_warmup=True)
    orch.start_accelerated_warmup()

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    scheduler.schedule_later.assert_not_called()
    pending = orch._active_joins["corr-root"]
    assert pending.replay_deadline_elapsed
    assert not pending.is_satisfied
    issuer.dispatch_join_turn.assert_not_awaited()

    await orch.on_child_leaf_reached("corr-child")

    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_join_derives_deadline_from_timestamps_when_delay_is_absent() -> None:
    """Timestamp/api_time metadata retains the same end-to-start invariant."""
    orch, _, scheduler = _timed_join_orchestrator(delay_ms=None)

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    delay_s, deadline_coro = scheduler.schedule_later.call_args.args
    assert delay_s == pytest.approx(198.046)
    deadline_coro.close()


@pytest.mark.asyncio
async def test_join_deadline_uses_shared_scheduler_idle_cap() -> None:
    """The system-idle cap may advance time but never bypass the child gate."""
    scheduler = LoopScheduler()
    orch, issuer, _ = _timed_join_orchestrator(
        delay_ms=60_000.0,
        scheduler=scheduler,
    )

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    assert scheduler.pending_count == 1
    assert scheduler.cap_pending_delay(0.0) == pytest.approx(60.0, abs=0.02)
    for _ in range(3):
        await asyncio.sleep(0)
    assert scheduler.pending_count == 0
    issuer.dispatch_join_turn.assert_not_awaited()

    await orch.on_child_leaf_reached("corr-child")
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_phase_stop_drains_join_whose_child_already_finished() -> None:
    """Cancelling a future deadline cannot strand a satisfied active join."""
    orch, issuer, _ = _timed_join_orchestrator()
    issuer.dispatch_join_turn.return_value = False

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    await orch.on_child_leaf_reached("corr-child")
    assert orch.has_pending_branch_work()

    await orch.expire_replay_deadlines()

    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.joins_suppressed == 1
    assert not orch.has_pending_branch_work()


@pytest.mark.asyncio
async def test_phase_stop_then_child_completion_drains_join() -> None:
    """A child returning after phase stop still closes the gated parent."""
    orch, issuer, _ = _timed_join_orchestrator()
    issuer.dispatch_join_turn.return_value = False

    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    await orch.expire_replay_deadlines()
    issuer.dispatch_join_turn.assert_not_awaited()

    await orch.on_child_leaf_reached("corr-child")

    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.joins_suppressed == 1
    assert not orch.has_pending_branch_work()


@pytest.mark.asyncio
async def test_snapshot_join_uses_absolute_replay_offset_and_child_gate() -> None:
    """A parent already blocked at t* keeps its profiling-relative deadline."""
    cs = _mk_source(_timed_k1_metadata())
    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    scheduler = MagicMock()
    orch = BranchOrchestrator(
        conversation_source=cs,
        credit_issuer=issuer,
        scheduler=scheduler,
    )
    parent = ConversationState(
        conversation_id="root",
        x_correlation_id="corr-root",
        next_turn_index=1,
        waiting_on_children=True,
        join_target_turn_index=1,
    )
    child = ConversationState(
        conversation_id="child",
        x_correlation_id="corr-child",
        next_turn_index=0,
        agent_depth=1,
        parent_correlation_id="corr-root",
        root_correlation_id="corr-root",
        join_target_turn_index=1,
        branch_id="root:0",
        branch_mode=ConversationBranchMode.SPAWN,
    )

    orch.seed_snapshot(
        (parent, child),
        join_release_delays_ms={"corr-root": 181_430.0},
    )
    delay_s, deadline_coro = scheduler.schedule_later.call_args.args
    assert delay_s == pytest.approx(181.430)

    await orch.on_child_leaf_reached("corr-child")
    issuer.dispatch_join_turn.assert_not_awaited()
    await deadline_coro
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_delayed_join_k5_parent_progresses():
    """Spawn at T=0, gate at T=5. Parent returns from turns 0..3 without
    suspension; only turn 4's return (which would dispatch turn 5) triggers
    suspension."""
    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0 return: spawns children; next turn is 1 (not gated) -> False.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is False
    assert "corr-root" in orch._future_joins
    assert 5 in orch._future_joins["corr-root"]
    assert orch.stats.parents_suspended == 0

    # Turns 1..3 return: no spawns, not next-to-gate, intercept returns False.
    for t in range(1, 4):
        assert await orch.intercept(_mk_credit("root", "corr-root", t)) is False
    assert orch.stats.parents_suspended == 0

    # Turn 4 return: NEXT turn = 5 = gated -> suspend.
    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is True
    assert "corr-root" in orch._active_joins
    assert orch.stats.parents_suspended == 1

    # Children complete -> join fires.
    await orch.on_child_leaf_reached("corr-c0")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-c1")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_breeze_through_applies_between_round_think_time():
    """Regression: when children drain before the parent reaches a request-free
    spine gate, the orchestrator must still apply the gate's authored
    between-round think-time before dispatching the gated turn -- rather than
    letting the strategy breeze through with no wait."""
    metadata = _k5_metadata()
    # Make the gated turn (index 5) a request-free spine gate with a think-time.
    gate_turn = metadata[0].turns[5]
    gate_turn.no_request = True
    gate_turn.delay_ms = 400.0

    cs = _mk_source(metadata)

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    slept: list[float] = []

    async def _capture(seconds: float) -> None:
        slept.append(seconds)

    orch._sleep_think_ms = _capture

    # Turn 0 spawns; both children finish before the parent nears the gate.
    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.on_child_leaf_reached("corr-c0")
    await orch.on_child_leaf_reached("corr-c1")

    # Parent walks turns 1..3 (no gate), then turn 4's return dispatches the
    # satisfied turn-5 gate -- honoring its 400 ms authored think-time.
    for t in range(1, 4):
        await orch.intercept(_mk_credit("root", "corr-root", t))
    suspended = await orch.intercept(_mk_credit("root", "corr-root", 4))

    assert suspended is True
    issuer.dispatch_join_turn.assert_awaited_once()
    assert slept == [0.4]  # 400 ms authored think-time honored on the breeze path


@pytest.mark.asyncio
async def test_delayed_join_children_finish_before_parent_arrives():
    """Children complete before the parent returns from turn 4. This is a
    NORMAL DAG gate (no request-free think-time), so a gate satisfied before
    the parent arrives is popped and the parent breezes through the strategy
    path -> intercept returns False and no dispatch_join_turn fires. (Spine
    gates, which carry think-time, take the retain-and-dispatch path instead.)"""
    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0 spawns.
    await orch.intercept(_mk_credit("root", "corr-root", 0))

    # Both children complete before parent returns from turn 4.
    await orch.on_child_leaf_reached("corr-c0")
    await orch.on_child_leaf_reached("corr-c1")

    # Parent now returns from turn 4 -> gate already satisfied -> no suspension.
    assert await orch.intercept(_mk_credit("root", "corr-root", 4)) is False
    assert "corr-root" not in orch._active_joins
    assert "corr-root" not in orch._future_joins
    assert orch.stats.parents_suspended == 0
    # Join never dispatched (children finished on their own path, parent
    # breezes through naturally into turn 5).
    issuer.dispatch_join_turn.assert_not_called()


@pytest.mark.asyncio
async def test_delayed_join_k1_regression_via_new_architecture():
    """K=1 auto-desugared case: spawn on turn 0, gate on turn 1. Parent's
    turn 0 return finds next_idx=1 as gated -> suspends immediately."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c0"],
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
    c0 = _mk_conv("c0", [TurnMetadata()], [])
    cs = _mk_source([root, c0])

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0 return: spawns child + next turn is 1 (gated) -> True.
    assert await orch.intercept(_mk_credit("root", "corr-root", 0)) is True
    assert orch.stats.parents_suspended == 1

    # Child finishes -> join fires.
    await orch.on_child_leaf_reached("corr-c0")
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1


@pytest.mark.asyncio
async def test_overlap_branch_dispatches_from_parent_start_not_return() -> None:
    branch = ConversationBranchInfo(
        branch_id="root:overlap",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
        start_timestamp_ms=0.0,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(
                timestamp_ms=0.0,
                api_time_ms=5000.0,
                branch_ids=[branch.branch_id],
            ),
            TurnMetadata(
                timestamp_ms=6000.0,
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN,
                        branch_id=branch.branch_id,
                    )
                ],
            ),
        ],
        [branch],
    )
    root.replay_scope_id = "root"
    child = _mk_conv("child", [TurnMetadata(timestamp_ms=0.0)], [])
    cs = _mk_source([root, child])
    child_session = MagicMock(
        x_correlation_id="corr-child",
        metadata=child,
        effective_root_correlation_id="corr-root",
    )
    cs.start_branch_child.return_value = child_session
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    credit = _mk_credit("root", "corr-root", 0)
    credit.phase = CreditPhase.PROFILING
    credit.effective_root_correlation_id = "corr-root"

    await orch.on_credit_issued(credit)

    issuer.dispatch_first_turn.assert_awaited_once_with(child_session)
    assert await orch.intercept(credit) is True
    issuer.dispatch_first_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_overlap_dispatch_skips_fork_branches() -> None:
    """FORK branches must not dispatch at credit-issue even when their
    start_timestamp overlaps the parent turn; they sticky-clone parent
    context that is only complete after the declaring turn returns.
    """
    branch = ConversationBranchInfo(
        branch_id="root:fork-overlap",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.FORK,
        start_timestamp_ms=0.0,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(
                timestamp_ms=0.0,
                api_time_ms=5000.0,
                branch_ids=[branch.branch_id],
            ),
            TurnMetadata(),
        ],
        [branch],
    )
    root.replay_scope_id = "root"
    child = _mk_conv("child", [TurnMetadata(timestamp_ms=0.0)], [])
    cs = _mk_source([root, child])
    cs.start_branch_child = MagicMock()
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    credit = _mk_credit("root", "corr-root", 0)
    credit.phase = CreditPhase.PROFILING
    credit.effective_root_correlation_id = "corr-root"

    await orch.on_credit_issued(credit)

    cs.start_branch_child.assert_not_called()
    issuer.dispatch_first_turn.assert_not_awaited()
    assert orch._overlap_dispatched_branches == set()


@pytest.mark.asyncio
async def test_delayed_join_stop_condition_fires_during_gap_suppresses_join():
    """If the issuer reports ``dispatch_join_turn`` returned False (stop
    fired), the orchestrator increments ``joins_suppressed`` instead of
    ``parents_resumed``."""
    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    # Stop condition suppresses dispatch_join_turn.
    issuer.dispatch_join_turn = AsyncMock(return_value=False)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "corr-root", 0))
    await orch.intercept(_mk_credit("root", "corr-root", 4))  # suspend

    await orch.on_child_leaf_reached("corr-c0")
    await orch.on_child_leaf_reached("corr-c1")

    assert orch.stats.joins_suppressed == 1
    assert orch.stats.parents_resumed == 0


@pytest.mark.asyncio
async def test_delayed_join_fail_fast_aborts_siblings_mid_gap(monkeypatch):
    """With ``AIPERF_DAG_FAIL_FAST=true`` and a child erroring during the
    gap, the parent and every orphan sibling are aborted immediately."""
    from aiperf.common.environment import Environment

    monkeypatch.setattr(Environment.DAG, "FAIL_FAST", True)

    cs = _mk_source(_k5_metadata())

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_branch_child = MagicMock(side_effect=_start)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock()
    issuer.abort_session = AsyncMock()

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Parent spawns on turn 0 and moves into gap (does NOT suspend yet).
    await orch.intercept(_mk_credit("root", "corr-root", 0))

    # Mid-gap, child c0 errors. Parent + orphan sibling aborted.
    await orch.on_child_errored("corr-c0")
    assert orch.stats.parents_failed_due_to_child_error == 1
    issuer.abort_session.assert_any_await("corr-root")
    issuer.abort_session.assert_any_await("corr-c1")
    assert "corr-root" not in orch._future_joins
    assert "corr-root" not in orch._active_joins


@pytest.mark.asyncio
async def test_delayed_join_multiple_branches_different_k_values_accepted_phase2():
    """Phase 2: declaring two gated branches on the same spawning turn with
    distinct gated_turn_index values is now accepted. The runtime is
    exercised in tests/unit/timing/test_branch_orchestrator_multi_gate.py;
    here we just assert the validator no longer rejects the shape."""
    from aiperf.common.validators.orchestrator_v1 import (
        validate_for_orchestrator_v1,
    )

    branch_a = ConversationBranchInfo(
        branch_id="r:0a",
        child_conversation_ids=["ca"],
        mode=ConversationBranchMode.SPAWN,
    )
    branch_b = ConversationBranchInfo(
        branch_id="r:0b",
        child_conversation_ids=["cb"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv = _mk_conv(
        "r",
        [
            TurnMetadata(branch_ids=["r:0a", "r:0b"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0a")
                ]
            ),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0b")
                ]
            ),
        ],
        [branch_a, branch_b],
    )
    ca = _mk_conv("ca", [TurnMetadata()], [])
    cb = _mk_conv("cb", [TurnMetadata()], [])
    md = DatasetMetadata(
        conversations=[conv, ca, cb],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    validate_for_orchestrator_v1(md)
