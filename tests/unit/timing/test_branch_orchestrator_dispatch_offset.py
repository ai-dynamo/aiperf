# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recorded dispatch offsets for SPAWN children in :class:`BranchOrchestrator`.

A SPAWN child whose first request was recorded after the branch spawn
(child turn-0 ``timestamp_ms`` past the branch ``start_timestamp_ms``) must
dispatch at that offset via a delayed background task, not immediately at
spawn. Covers:

- offset computation from branch start vs child turn-0 timestamps
- fallback branch start = min child turn-0 timestamp
- immediate dispatch when timestamps are absent (legacy / --ignore-trace-delays)
- SPAWN_JOIN gates keep waiting for a still-sleeping child
- cleanup() cancels pending sleepers without dispatching
- post-sleep stop-condition refusal rolls back like an immediate refusal
- mixed branches dispatch immediate children during intercept
"""

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
from aiperf.timing.branch_orchestrator import BranchOrchestrator


async def _tick(n: int = 3) -> None:
    """Yield to the event loop so spawned dispatch tasks can progress."""
    for _ in range(n):
        await asyncio.sleep(0)


class _SleepGate:
    """Replaces ``BranchOrchestrator._sleep_offset_ms`` for determinism.

    Records every requested offset and holds all sleepers until released,
    so tests can assert the not-yet-dispatched state without real time.
    """

    def __init__(self) -> None:
        self.offsets: list[float] = []
        self._release = asyncio.Event()

    async def __call__(self, offset_ms: float) -> None:
        self.offsets.append(offset_ms)
        await self._release.wait()

    def release(self) -> None:
        self._release.set()


def _mk_conv(
    cid: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo] | None = None,
) -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=cid, turns=turns, branches=branches or []
    )


def _mk_harness(
    conversations: list[ConversationMetadata],
    *,
    dispatch_result: bool = True,
):
    """(orchestrator, conversation_source, issuer) with real metadata models."""
    by_id = {c.conversation_id: c for c in conversations}
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: by_id[cid]

    def _fake_child(*, child_conversation_id, **kwargs):
        return MagicMock(
            x_correlation_id=f"corr-{child_conversation_id}",
            metadata=by_id[child_conversation_id],
        )

    cs.start_branch_child = MagicMock(side_effect=_fake_child)

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=dispatch_result)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    return orch, cs, issuer


def _mk_credit(conv_id: str, corr_id: str, turn_index: int):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=0,
        parent_correlation_id=None,
        phase=None,
    )


def _spawn_branch(
    branch_id: str,
    child_ids: list[str],
    *,
    start_timestamp_ms: float | None,
    is_background: bool = True,
) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=child_ids,
        mode=ConversationBranchMode.SPAWN,
        is_background=is_background,
        start_timestamp_ms=start_timestamp_ms,
    )


def _parent_conv(
    branches: list[ConversationBranchInfo],
    *,
    gated_turn: int | None = None,
    branch_id: str = "b0",
) -> ConversationMetadata:
    turns = [
        TurnMetadata(timestamp_ms=0.0, branch_ids=[b.branch_id for b in branches]),
        TurnMetadata(timestamp_ms=10_000.0),
        TurnMetadata(timestamp_ms=20_000.0),
    ]
    if gated_turn is not None:
        turns[gated_turn].prerequisites = [
            TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id=branch_id)
        ]
    return _mk_conv("parent", turns, branches)


def _child_conv(cid: str, turn0_ts_ms: float | None) -> ConversationMetadata:
    return _mk_conv(cid, [TurnMetadata(timestamp_ms=turn0_ts_ms)])


@pytest.mark.asyncio
async def test_offset_child_dispatches_after_sleep_not_during_intercept():
    """A child recorded 4s after the spawn sleeps 4000ms before dispatching."""
    parent = _parent_conv([_spawn_branch("b0", ["kid"], start_timestamp_ms=1_000.0)])
    orch, _, issuer = _mk_harness([parent, _child_conv("kid", 5_000.0)])
    gate = _SleepGate()
    orch._sleep_offset_ms = gate

    await orch.intercept(_mk_credit("parent", "P", 0))
    await _tick()

    issuer.dispatch_first_turn.assert_not_awaited()
    assert gate.offsets == [4_000.0]
    assert orch.stats.children_spawned == 1
    assert orch.stats.children_delayed == 1

    gate.release()
    await _tick()
    assert issuer.dispatch_first_turn.await_count == 1


@pytest.mark.asyncio
async def test_missing_timestamps_keep_immediate_dispatch():
    """None timestamps (e.g. --ignore-trace-delays) mean offset 0: dispatch
    happens during intercept exactly as before the offset mechanism."""
    parent = _parent_conv([_spawn_branch("b0", ["kid"], start_timestamp_ms=None)])
    orch, _, issuer = _mk_harness([parent, _child_conv("kid", None)])
    gate = _SleepGate()
    orch._sleep_offset_ms = gate

    await orch.intercept(_mk_credit("parent", "P", 0))

    assert issuer.dispatch_first_turn.await_count == 1
    assert gate.offsets == []
    assert orch.stats.children_delayed == 0


@pytest.mark.asyncio
async def test_branch_start_falls_back_to_min_child_turn0():
    """Without start_timestamp_ms the spawn boundary is the earliest child
    turn-0 timestamp: that child is immediate, the later sibling is offset
    by the difference (mirrors trajectory_source._branch_runtimes)."""
    parent = _parent_conv(
        [_spawn_branch("b0", ["early", "late"], start_timestamp_ms=None)]
    )
    orch, _, issuer = _mk_harness(
        [parent, _child_conv("early", 1_000.0), _child_conv("late", 5_000.0)]
    )
    gate = _SleepGate()
    orch._sleep_offset_ms = gate

    await orch.intercept(_mk_credit("parent", "P", 0))

    # "early" dispatched during intercept; "late" sleeping out 4000ms.
    assert issuer.dispatch_first_turn.await_count == 1
    assert gate.offsets == [4_000.0]

    gate.release()
    await _tick()
    assert issuer.dispatch_first_turn.await_count == 2


@pytest.mark.asyncio
async def test_spawn_join_gate_waits_for_sleeping_child():
    """A SPAWN_JOIN gate registered at spawn time holds while the child
    sleeps out its offset; the join only satisfies after the child
    dispatches and reaches its leaf."""
    branch = _spawn_branch("b0", ["kid"], start_timestamp_ms=0.0, is_background=False)
    parent = _parent_conv([branch], gated_turn=2)
    orch, _, issuer = _mk_harness([parent, _child_conv("kid", 7_000.0)])
    gate = _SleepGate()
    orch._sleep_offset_ms = gate

    await orch.intercept(_mk_credit("parent", "P", 0))
    await _tick()

    pending = orch._future_joins["P"][2]
    state = pending.outstanding["SPAWN_JOIN:b0"]
    assert state.expected == 1
    assert not pending.is_satisfied
    issuer.dispatch_first_turn.assert_not_awaited()

    gate.release()
    await _tick()
    assert issuer.dispatch_first_turn.await_count == 1

    await orch.on_child_leaf_reached("corr-kid")
    assert pending.is_satisfied


@pytest.mark.asyncio
async def test_cleanup_cancels_sleeping_dispatch():
    """cleanup() during the sleep cancels the task; the child never
    dispatches and the task set drains."""
    parent = _parent_conv([_spawn_branch("b0", ["kid"], start_timestamp_ms=0.0)])
    orch, _, issuer = _mk_harness([parent, _child_conv("kid", 60_000.0)])
    gate = _SleepGate()
    orch._sleep_offset_ms = gate

    await orch.intercept(_mk_credit("parent", "P", 0))
    assert len(orch._delayed_dispatch_tasks) == 1

    orch.cleanup()
    await _tick()

    issuer.dispatch_first_turn.assert_not_awaited()
    assert orch._delayed_dispatch_tasks == set()


@pytest.mark.asyncio
async def test_post_sleep_refusal_rolls_back_like_immediate_refusal():
    """An issuer refusal (False) after the sleep rolls the child back:
    truncated stat, join tracking released, descendant count drained."""
    parent = _parent_conv([_spawn_branch("b0", ["kid"], start_timestamp_ms=0.0)])
    orch, _, issuer = _mk_harness(
        [parent, _child_conv("kid", 5_000.0)], dispatch_result=False
    )
    gate = _SleepGate()
    orch._sleep_offset_ms = gate

    await orch.intercept(_mk_credit("parent", "P", 0))
    await _tick()
    assert "corr-kid" in orch._child_to_join

    gate.release()
    await _tick()

    assert orch.stats.children_truncated == 1
    assert orch.stats.children_spawned == 0
    assert "corr-kid" not in orch._child_to_join
    assert "P" not in orch._descendant_counts


@pytest.mark.asyncio
async def test_mixed_branch_immediate_child_unaffected_by_delayed_sibling():
    """One child at the spawn boundary and one recorded later: the first
    dispatches during intercept, the second after its offset."""
    parent = _parent_conv(
        [_spawn_branch("b0", ["now", "later"], start_timestamp_ms=2_000.0)]
    )
    orch, _, issuer = _mk_harness(
        [parent, _child_conv("now", 2_000.0), _child_conv("later", 9_000.0)]
    )
    gate = _SleepGate()
    orch._sleep_offset_ms = gate

    await orch.intercept(_mk_credit("parent", "P", 0))

    assert issuer.dispatch_first_turn.await_count == 1
    dispatched = issuer.dispatch_first_turn.await_args.args[0]
    assert dispatched.x_correlation_id == "corr-now"
    assert gate.offsets == [7_000.0]
    assert orch.stats.children_delayed == 1

    gate.release()
    await _tick()
    assert issuer.dispatch_first_turn.await_count == 2


@pytest.mark.asyncio
async def test_delayed_spawn_rollback_after_parent_suspended_dispatches_active_gate():
    """A delayed SPAWN child whose dispatch is refused AFTER the parent already
    suspended must still fire the now-satisfied join.

    Sequence: turn-0 spawns a delayed child gated on turn 1; the parent's
    ``_maybe_suspend_parent`` promotes that gate into ``_active_joins`` (the
    child is still sleeping, so the gate is unsatisfied). When the sleep
    releases and the issuer refuses the child, rollback empties the gate's only
    prereq -> the gate is satisfied but sits in ``_active_joins`` where the
    future-join drain scan never looks. The suspended parent would otherwise
    deadlock until drain-timeout; the join turn must dispatch instead."""
    parent = _parent_conv(
        [_spawn_branch("b0", ["kid"], start_timestamp_ms=0.0, is_background=False)],
        gated_turn=1,
    )
    orch, _, issuer = _mk_harness(
        [parent, _child_conv("kid", 5_000.0)], dispatch_result=False
    )
    gate = _SleepGate()
    orch._sleep_offset_ms = gate

    # Turn-0 return: child scheduled delayed, gate at turn 1 promoted to active.
    assert await orch.intercept(_mk_credit("parent", "P", 0)) is True
    await _tick()
    assert "P" in orch._active_joins
    assert orch._active_joins["P"].gated_turn_index == 1
    assert not orch._active_joins["P"].is_satisfied
    issuer.dispatch_join_turn.assert_not_awaited()

    # Sleep releases -> dispatch refused -> rollback empties the gate.
    gate.release()
    await _tick()

    # The satisfied active gate is dispatched; the parent is no longer stuck.
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1
    assert "P" not in orch._active_joins
    assert orch.stats.children_truncated == 1
