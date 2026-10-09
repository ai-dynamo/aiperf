# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stopped phase must not wait out a join's recorded timestamp.

Under fixed schedule a gated turn is held until its recorded timestamp so a
trace with subagents does not replay faster than it was recorded. That hold is
only legal while the phase is still sending. ``PhaseRunner`` calls
``expire_replay_deadlines`` once the duration cutoff fires, and the issuer then
refuses the stopped-phase turn -- so holding for a target far in the future
buys nothing and stretches the observation window that throughput is computed
over. A parent recorded at t=20s under ``--benchmark-duration 4`` would keep
the run alive to t=20s and divide its requests by that stalled window.
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
from aiperf.timing.branch_orchestrator import BranchOrchestrator, PendingBranchJoin

RECORDED_TARGET_SEC = 20.0


def _orchestrator() -> BranchOrchestrator:
    """One parent whose turn 1 is gated on a SPAWN_JOIN, recorded at t=20s."""
    branch = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv = ConversationMetadata(
        conversation_id="r",
        turns=[
            TurnMetadata(branch_ids=["r:0"], timestamp_ms=0),
            TurnMetadata(
                timestamp_ms=int(RECORDED_TARGET_SEC * 1000),
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")
                ],
            ),
        ],
        branches=[branch],
    )
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=[conv],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: conv

    issuer = MagicMock()
    issuer.dispatch_join_turn = AsyncMock(return_value=False)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    # The recorded target sits RECORDED_TARGET_SEC into the run, far beyond a
    # duration cutoff that has already fired.
    orch.set_schedule_target_resolver(lambda ms: _now() + ms / 1000.0)
    return orch


def _now() -> float:
    import time

    return time.perf_counter()


def _blocked_join(orch: BranchOrchestrator) -> PendingBranchJoin:
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-1",
        parent_conversation_id="r",
        parent_num_turns=2,
        parent_agent_depth=0,
        parent_parent_correlation_id=None,
        gated_turn_index=1,
    )
    pending.is_blocked = True
    orch._active_joins["corr-1"] = pending
    return pending


@pytest.fixture
def recorded_holds(monkeypatch):
    """Capture every recorded-timestamp hold the orchestrator performs."""
    holds: list[float] = []

    async def _spy(self, seconds: float) -> None:
        holds.append(seconds)

    monkeypatch.setattr(BranchOrchestrator, "_sleep_until_schedule_target", _spy)
    return holds


@pytest.mark.asyncio
async def test_duration_cutoff_does_not_wait_for_the_recorded_target(
    recorded_holds,
) -> None:
    orch = _orchestrator()
    _blocked_join(orch)

    await orch.expire_replay_deadlines()

    assert not [h for h in recorded_holds if h > 0], (
        f"a stopped phase held for {recorded_holds} seconds waiting on a "
        f"recorded timestamp it can no longer dispatch at"
    )


@pytest.mark.asyncio
async def test_normal_release_still_honours_the_recorded_target(
    recorded_holds,
) -> None:
    """The hold this PR exists to add must survive the stop-path fix."""
    orch = _orchestrator()
    pending = _blocked_join(orch)

    await orch._release_blocked_join(pending)

    assert [h for h in recorded_holds if h > 0], (
        "the recorded-timestamp hold was dropped on the normal release path; "
        "gated turns would fire as soon as their children finish"
    )


@pytest.mark.asyncio
async def test_fixed_schedule_without_the_public_hook_fails_loudly() -> None:
    """A rename must not silently drop the recorded-timestamp hold.

    The resolver is looked up by name, so a strategy that renamed it would
    otherwise keep running with gated turns firing as soon as their children
    finish -- the exact compression this mode exists to prevent, invisible in
    the results.
    """
    from aiperf.plugin.enums import TimingMode
    from aiperf.timing.phase.runner import PhaseRunner

    runner = MagicMock(spec=PhaseRunner)
    runner._branch_orchestrator = MagicMock()
    runner._config = MagicMock(timing_mode=TimingMode.FIXED_SCHEDULE)

    strategy = MagicMock(spec=[])  # exposes no attributes at all

    with pytest.raises(AttributeError, match="schedule_target_perf_sec"):
        PhaseRunner._wire_join_schedule_target(runner, strategy)

    runner._branch_orchestrator.set_schedule_target_resolver.assert_not_called()


@pytest.mark.asyncio
async def test_cutoff_releases_a_join_already_holding_for_its_target() -> None:
    """The hold is interrupted, not merely skipped.

    When a join's children finish *before* the cutoff, the normal deadline path
    has already popped it from ``_active_joins`` and is sleeping toward its
    recorded target. ``expire_replay_deadlines`` iterates that dict, so it
    cannot reach the sleeper -- a fix that only skips the hold for joins still
    in the dict leaves this path stalling the run for the full target.

    Runs the real wait, so it fails by timing out if the hold is uninterruptible.
    """
    orch = _orchestrator()
    pending = _blocked_join(orch)
    pending.replay_deadline_armed = True

    holding = asyncio.create_task(orch._on_join_replay_deadline("corr-1", 1))
    # Let the task reach the hold and drop out of _active_joins.
    for _ in range(3):
        await asyncio.sleep(0)
    assert "corr-1" not in orch._active_joins, (
        "precondition: the sleeper must already be out of the dict"
    )

    await orch.expire_replay_deadlines()

    await asyncio.wait_for(holding, timeout=2)
    assert orch._issuer.dispatch_join_turn.await_count == 1
