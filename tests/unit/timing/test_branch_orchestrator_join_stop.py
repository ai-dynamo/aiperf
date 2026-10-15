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

    async def _spy(self, seconds: float, still_sendable) -> None:
        holds.append(seconds)

    monkeypatch.setattr(BranchOrchestrator, "_interruptible_sleep", _spy)
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


@pytest.mark.asyncio
async def test_recorded_hold_returns_when_the_gated_turn_becomes_unsendable() -> None:
    """A refused join must not keep holding for its recorded timestamp.

    The sending cutoff is not the only way a gated turn stops being sendable:
    ``--request-count`` / ``--num-conversations`` close the issuer's gate while
    the phase is still scheduling, so ``_schedule_stopped`` is never set. A
    hold that watches only that latch then sleeps to a recorded instant for a
    turn the issuer will refuse on wake, stretching the window throughput is
    divided by for a request that is never sent.

    Runs the real wait, so it fails by timing out if the gate is ignored.
    """
    orch = _orchestrator()
    pending = _blocked_join(orch)
    sendable = True
    orch._issuer.can_dispatch_join = lambda depth: sendable
    orch._issuer.can_dispatch_child_turn = lambda: sendable

    holding = asyncio.create_task(orch._await_recorded_join_target(pending))
    for _ in range(3):
        await asyncio.sleep(0)
    assert not holding.done(), (
        "precondition: the hold must still be waiting on its recorded target"
    )

    sendable = False
    orch._wake_think_time_sleepers()

    await asyncio.wait_for(holding, timeout=2)


@pytest.mark.asyncio
async def test_recorded_hold_keeps_waiting_while_the_turn_is_still_sendable() -> None:
    """The gate must not collapse the hold into a no-op.

    A wake that is not accompanied by a closed gate (any other sleeper being
    woken) must leave the recorded target intact -- otherwise the compression
    this hold exists to prevent returns, and silently.
    """
    orch = _orchestrator()
    pending = _blocked_join(orch)
    orch._issuer.can_dispatch_join = lambda depth: True

    holding = asyncio.create_task(orch._await_recorded_join_target(pending))
    for _ in range(3):
        await asyncio.sleep(0)

    orch._wake_think_time_sleepers()
    for _ in range(5):
        await asyncio.sleep(0)

    assert not holding.done(), (
        "the recorded-timestamp hold was dropped on a wake even though the "
        "gated turn was still sendable"
    )
    holding.cancel()


@pytest.mark.asyncio
async def test_cleanup_releases_a_join_holding_for_its_recorded_target() -> None:
    """Teardown without a sending cutoff must still end the hold.

    A phase torn down by a cancel or a failure never reaches
    ``expire_replay_deadlines``, so ``cleanup()`` is the only thing that fires.
    If it does not end the hold, shutdown waits out the full recorded target.
    """
    orch = _orchestrator()
    pending = _blocked_join(orch)
    orch._issuer.can_dispatch_join = lambda depth: True

    holding = asyncio.create_task(orch._await_recorded_join_target(pending))
    for _ in range(3):
        await asyncio.sleep(0)
    assert not holding.done()

    orch.cleanup()

    await asyncio.wait_for(holding, timeout=2)


def _armed(orch: BranchOrchestrator) -> tuple[MagicMock, PendingBranchJoin]:
    """Give the orchestrator a scheduler and a blocked join ready to arm."""
    scheduler = MagicMock()
    orch._scheduler = scheduler
    return scheduler, _blocked_join(orch)


@pytest.mark.asyncio
async def test_a_recorded_join_is_armed_on_its_absolute_target() -> None:
    """The deadline is an instant on the recorded timeline, not a stopwatch.

    ``delay_ms`` is an end-to-start delta, so arming relative to now inherits
    however late the previous turn's response landed and the error tracks
    server latency -- the replay stretches the trace it exists to reproduce.
    Re-deriving a relative delay here would resample ``perf_counter`` and hand
    back the drift the absolute target exists to remove, so the target must be
    passed through as-is.
    """
    orch = _orchestrator()
    scheduler, pending = _armed(orch)
    expected = orch._recorded_join_target_perf_sec(pending)
    assert expected is not None

    orch._arm_join_replay_deadline(pending, delay_ms=1.0)

    scheduler.schedule_later.assert_not_called()
    scheduler.schedule_at_perf_sec.assert_called_once()
    assert scheduler.schedule_at_perf_sec.call_args.args[0] == pytest.approx(
        expected, abs=0.5
    )
    assert pending.replay_deadline_armed
    assert not pending.replay_deadline_elapsed


@pytest.mark.asyncio
async def test_a_join_off_the_recorded_timeline_keeps_the_relative_delay() -> None:
    """Without a resolver the join is reactive, which is its documented mode.

    Routing every join through the absolute path would silently re-time every
    non-fixed-schedule DAG benchmark against timestamps it never recorded.
    """
    orch = _orchestrator()
    orch.set_schedule_target_resolver(None)
    scheduler, pending = _armed(orch)

    orch._arm_join_replay_deadline(pending, delay_ms=1000.0)

    scheduler.schedule_at_perf_sec.assert_not_called()
    scheduler.schedule_later.assert_called_once()
    assert scheduler.schedule_later.call_args.args[0] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_a_target_already_in_the_past_arms_nothing_and_elapses() -> None:
    """Children overran the recorded window: dispatch now, do not compound drift."""
    orch = _orchestrator()
    orch.set_schedule_target_resolver(lambda ms: _now() - 100.0)
    scheduler, pending = _armed(orch)

    orch._arm_join_replay_deadline(pending, delay_ms=1000.0)

    scheduler.schedule_at_perf_sec.assert_not_called()
    scheduler.schedule_later.assert_not_called()
    assert pending.replay_deadline_elapsed


@pytest.mark.asyncio
async def test_an_accelerated_warmup_ignores_the_recorded_timeline() -> None:
    """Warmup exists to finish fast; replaying its recorded pacing defeats it."""
    orch = _orchestrator()
    orch._accelerated_warmup_started = True
    scheduler, pending = _armed(orch)

    orch._arm_join_replay_deadline(pending, delay_ms=5000.0)

    scheduler.schedule_at_perf_sec.assert_not_called()
    assert pending.replay_deadline_elapsed


@pytest.mark.asyncio
async def test_a_turn_index_past_the_recorded_turns_has_no_target() -> None:
    """A gated index the metadata does not reach must not raise or invent a time."""
    orch = _orchestrator()
    pending = _blocked_join(orch)
    pending.gated_turn_index = 99

    assert orch._recorded_join_target_perf_sec(pending) is None


@pytest.mark.asyncio
async def test_a_turn_without_a_recorded_timestamp_has_no_target() -> None:
    """Fixed schedule does not imply every turn carries a timestamp.

    A turn with none stays reactive rather than being pinned to t=0, which
    would fire it at the start of the run.
    """
    orch = _orchestrator()
    pending = _blocked_join(orch)
    orch._cs.get_metadata("r").turns[1].timestamp_ms = None

    assert orch._recorded_join_target_perf_sec(pending) is None


@pytest.mark.asyncio
async def test_an_unknown_parent_conversation_has_no_target() -> None:
    """A metadata lookup that fails degrades to reactive, not to a crash."""
    orch = _orchestrator()
    pending = _blocked_join(orch)
    orch._cs.get_metadata.side_effect = KeyError("r")

    assert orch._recorded_join_target_perf_sec(pending) is None


@pytest.mark.asyncio
async def test_an_ungated_join_has_no_target() -> None:
    """Without a gated turn index there is no turn whose timestamp to honour."""
    orch = _orchestrator()
    pending = _blocked_join(orch)
    pending.gated_turn_index = None

    assert orch._recorded_join_target_perf_sec(pending) is None


@pytest.mark.asyncio
async def test_a_target_already_reached_does_not_hold_the_release() -> None:
    """``_await_recorded_join_target`` is a lower bound, never an upper one."""
    orch = _orchestrator()
    pending = _blocked_join(orch)
    orch.set_schedule_target_resolver(lambda ms: _now() - 100.0)

    await asyncio.wait_for(orch._await_recorded_join_target(pending), timeout=2)


@pytest.mark.asyncio
async def test_fixed_schedule_wires_the_strategy_hook_onto_the_orchestrator() -> None:
    """The hook must actually reach the orchestrator, not just exist.

    ``test_fixed_schedule_without_the_public_hook_fails_loudly`` pins the
    failure path; without this, a wiring that raised correctly but never
    called ``set_schedule_target_resolver`` would pass both.
    """
    from aiperf.plugin.enums import TimingMode
    from aiperf.timing.phase.runner import PhaseRunner

    runner = MagicMock(spec=PhaseRunner)
    runner._branch_orchestrator = MagicMock()
    runner._config = MagicMock(timing_mode=TimingMode.FIXED_SCHEDULE)

    strategy = MagicMock(spec=["schedule_target_perf_sec"])

    PhaseRunner._wire_join_schedule_target(runner, strategy)

    runner._branch_orchestrator.set_schedule_target_resolver.assert_called_once_with(
        strategy.schedule_target_perf_sec
    )


@pytest.mark.asyncio
async def test_a_non_fixed_schedule_phase_clears_any_stale_resolver() -> None:
    """Phases share one orchestrator, so a stale resolver must be cleared.

    A fixed-schedule warmup followed by a request-rate profiling phase would
    otherwise keep honouring warmup timestamps on the profiling phase's joins.
    """
    from aiperf.plugin.enums import TimingMode
    from aiperf.timing.phase.runner import PhaseRunner

    runner = MagicMock(spec=PhaseRunner)
    runner._branch_orchestrator = MagicMock()
    runner._config = MagicMock(timing_mode=TimingMode.REQUEST_RATE)

    PhaseRunner._wire_join_schedule_target(runner, MagicMock(spec=[]))

    runner._branch_orchestrator.set_schedule_target_resolver.assert_called_once_with(
        None
    )


@pytest.mark.asyncio
async def test_wiring_is_a_no_op_without_a_branch_orchestrator() -> None:
    """A non-DAG run has no orchestrator; wiring must not fault on it."""
    from aiperf.plugin.enums import TimingMode
    from aiperf.timing.phase.runner import PhaseRunner

    runner = MagicMock(spec=PhaseRunner)
    runner._branch_orchestrator = None
    runner._config = MagicMock(timing_mode=TimingMode.FIXED_SCHEDULE)

    PhaseRunner._wire_join_schedule_target(runner, MagicMock(spec=[]))
