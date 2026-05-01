# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for the FIFO recycle queue in AgenticReplayStrategy.

Covers spec section 8.4.3:
    1. Single trace, concurrency=1: recycle reuses the just-finished trace.
    2. Pool=1, concurrency=2: second consumer waits without deadlock.
    3. Burst of 10 completions in one tick: order preserved.
    4. Push-back races concurrent pop: asyncio.Queue order preserved.
    5. Double-recycle programmer error: debug-build assertion guard.
    6. Cooldown after DurationStopCondition: no new sessions begin.
    7. Pool=750, concurrency=100: every trace replayed; deterministic order.
    8. Trajectory with N_i=1 (warmup-only): immediate recycle at PROFILING.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    Trajectory,
    TrajectorySource,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_dataset(num_traces: int, turns_per_trace: int) -> DatasetMetadata:
    """Build a deterministic dataset of `num_traces` conversations of fixed length."""
    convs = []
    for i in range(num_traces):
        turns = [
            TurnMetadata(timestamp_ms=None, delay_ms=None)
            for _ in range(turns_per_trace)
        ]
        convs.append(ConversationMetadata(conversation_id=f"trace_{i}", turns=turns))
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def _build_real_trajectory_source(
    *,
    dataset: DatasetMetadata,
    trajectories: list[Trajectory],
) -> TrajectorySource:
    """Construct a TrajectorySource with a deterministic trajectory."""
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = dataset
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in dataset.conversations}
    src._random_seed = 0
    src._target_size = len(trajectories)
    src.trajectories = list(trajectories)
    return src


def _make_strategy(
    *,
    phase: CreditPhase,
    trajectories: list[Trajectory],
    dataset: DatasetMetadata,
    issuer: AsyncMock | None = None,
    scheduler: MagicMock | None = None,
    stop_checker: MagicMock | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock]:
    src = _build_real_trajectory_source(dataset=dataset, trajectories=trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = max(1, len(trajectories))
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    stop_checker = stop_checker if stop_checker is not None else MagicMock()
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    return strategy, issuer, stop_checker


def _make_credit(
    *,
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
    x_correlation_id: str = "xcorr",
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.FORK,
    )


# =============================================================================
# Test 1: Single trace, concurrency=1 -> immediate self-recycle
# =============================================================================


@pytest.mark.asyncio
async def test_single_trace_concurrency_one_recycles_self():
    """Pool of 1 trace == trajectory. After finishing, the same trace is re-served."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=1, turns_per_trace=3)
    issued: list[tuple[str, int]] = []

    async def capture(turn):
        issued.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    assert strategy._recycle_queue is not None
    assert strategy._recycle_queue.qsize() == 0  # trajectory consumes the only trace

    # Register the in-flight session's lane (normally done by _execute_profiling).
    strategy._correlation_to_lane["xcorr"] = 0

    # Final turn (last index = 2 of num_turns=3)
    final = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=3)
    await strategy.handle_credit_return(final)

    # The just-finished trace must be re-served at turn 0.
    assert issued == [("trace_0", 0)]
    # Queue is back to empty: pushed then immediately popped.
    assert strategy._recycle_queue.qsize() == 0


# =============================================================================
# Test 2: Pool=1, concurrency=2 -> second consumer waits, no deadlock
# =============================================================================


@pytest.mark.asyncio
async def test_pool_one_concurrency_two_no_deadlock():
    """Two trajectories but only one queued trace -> second consumer's recycle
    just reuses the queued slot. No deadlock; both consumers progress.

    Models a real run with two parallel sessions where the recycle queue at
    PROFILING start has exactly one entry. After both sessions finish, both
    push their trace_id and both pop the FIFO head. No blocking await on get().
    """
    # Two trajectories, three traces total -> queue at PROFILING setup has
    # exactly one trace (trace_2) in it.
    trajectory = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    ds = _make_dataset(num_traces=3, turns_per_trace=2)
    issued: list[str] = []

    async def capture(turn):
        issued.append(turn.conversation_id)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    assert strategy._recycle_queue.qsize() == 1  # only trace_2 queued

    # Register lane bookkeeping for both in-flight sessions (normally seeded by
    # _execute_profiling). handle_credit_return's recycle path requires
    # finished_correlation_id to be in _correlation_to_lane.
    strategy._correlation_to_lane["xcorr_a"] = 0
    strategy._correlation_to_lane["xcorr_b"] = 1

    # Two parallel consumers complete. We use asyncio.gather to drive them
    # concurrently within the same event-loop tick. asyncio.Queue is non-blocking
    # for both put_nowait and get_nowait so neither call blocks.
    final_a = _make_credit(
        conversation_id="trace_0",
        turn_index=1,
        num_turns=2,
        x_correlation_id="xcorr_a",
    )
    final_b = _make_credit(
        conversation_id="trace_1",
        turn_index=1,
        num_turns=2,
        x_correlation_id="xcorr_b",
    )
    await asyncio.wait_for(
        asyncio.gather(
            strategy.handle_credit_return(final_a),
            strategy.handle_credit_return(final_b),
        ),
        timeout=2.0,
    )

    # Both consumers fired exactly one new credit.
    assert len(issued) == 2
    # Sequence: gather schedules tasks, both run within ticks.
    #   call A: push trace_0 -> queue=[trace_2, trace_0]; pop -> trace_2; queue=[trace_0]; serves trace_2
    #   call B: push trace_1 -> queue=[trace_0, trace_1]; pop -> trace_0; queue=[trace_1]; serves trace_0
    # End state: served=[trace_2, trace_0], queue=[trace_1].
    assert issued == ["trace_2", "trace_0"]
    remaining: list[str] = []
    while not strategy._recycle_queue.empty():
        remaining.append(strategy._recycle_queue.get_nowait())
    assert remaining == ["trace_1"]


# =============================================================================
# Test 3: Burst of 10 completions within one tick -> order preserved
# =============================================================================


@pytest.mark.asyncio
async def test_burst_of_ten_completions_preserves_completion_order():
    """10 sessions complete sequentially within the same loop tick.

    Each handle_credit_return call pushes-then-pops, so after all 10 fire the
    queue tail order matches the completion order.
    """
    # 12 traces, 10 trajectories -> queue starts with 2 traces (trace_10, trace_11).
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(10)
    ]
    ds = _make_dataset(num_traces=12, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    assert strategy._recycle_queue.qsize() == 2

    # Register lane bookkeeping for the 10 in-flight sessions.
    for i in range(10):
        strategy._correlation_to_lane[f"xcorr_{i}"] = i

    # Fire 10 completions in completion order: trace_0..trace_9 finish in order.
    for i in range(10):
        await strategy.handle_credit_return(
            _make_credit(
                conversation_id=f"trace_{i}",
                turn_index=1,
                num_turns=2,
                x_correlation_id=f"xcorr_{i}",
            )
        )

    # Each call pushes the finished trace, then pops the head.
    # After 10 calls: queue tail = the completion order (last 2 are still there
    # because head pops always served the leading 10 entries).
    # Sequence: queue=[t10, t11]
    #  push t0 -> [t10, t11, t0], pop -> [t11, t0], served t10
    #  push t1 -> [t11, t0, t1], pop -> [t0, t1], served t11
    #  push t2 -> [t0, t1, t2], pop -> [t1, t2], served t0
    #  ...
    # Final queue after 10 pushes/pops = [t8, t9].
    remaining = []
    while not strategy._recycle_queue.empty():
        remaining.append(strategy._recycle_queue.get_nowait())
    assert remaining == ["trace_8", "trace_9"]


# =============================================================================
# Test 4: Push-back races concurrent pop -> no lost or duplicated trace_ids
# =============================================================================


@pytest.mark.asyncio
async def test_concurrent_recycle_no_lost_or_duplicated_trace_ids():
    """Drive 50 completions concurrently via asyncio.gather; verify the conservation law.

    Invariant: the multiset of all trace_ids ever observed (queue contents at
    end + dispatched-as-new-session during the burst) equals the multiset of
    all trace_ids that ever entered the system (initial queue + completed).
    """
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(50)
    ]
    ds = _make_dataset(num_traces=70, turns_per_trace=2)  # 20 in queue at start
    served: list[str] = []

    async def capture(turn):
        served.append(turn.conversation_id)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    initial_queue = list(strategy._recycle_queue._queue)  # snapshot
    assert len(initial_queue) == 20

    # Register lane bookkeeping for the 50 in-flight sessions.
    for i in range(50):
        strategy._correlation_to_lane[f"xcorr_{i}"] = i

    finals = [
        _make_credit(
            conversation_id=f"trace_{i}",
            turn_index=1,
            num_turns=2,
            x_correlation_id=f"xcorr_{i}",
        )
        for i in range(50)
    ]
    await asyncio.gather(*(strategy.handle_credit_return(c) for c in finals))

    final_queue: list[str] = []
    while not strategy._recycle_queue.empty():
        final_queue.append(strategy._recycle_queue.get_nowait())

    # Conservation: served + final_queue == initial_queue + completed_trace_ids.
    completed = [c.conversation_id for c in finals]
    assert sorted(served + final_queue) == sorted(initial_queue + completed)

    # No duplicates anywhere in served (each completion drives one fresh dispatch).
    assert len(served) == 50


# =============================================================================
# Test 5: Double-recycle programmer error -> debug-build assertion
# =============================================================================


@pytest.mark.asyncio
async def test_double_recycle_same_trace_raises():
    """Calling handle_credit_return twice for the same final turn must raise.

    This is a programmer-error guard: the recycle queue's invariant is that a
    given trace_id is in-flight (either dispatched or queued) exactly once at
    any moment. Pushing the same trace_id twice in the same recycle cycle
    breaks that invariant.

    The guard is unconditional (was previously gated on ``__debug__``, which
    ``python -O`` strips, silently allowing the duplicate-final-turn corruption
    to escape into production).
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=3, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    # Queue starts with [trace_1, trace_2].
    # We keep trace_0 in-flight by *not* popping it after the first push.
    # Manually manipulate state to simulate the "trace_0 just got pushed but
    # somehow was reported finished again before it was popped" pathology.
    strategy._in_flight_recycled.add("trace_0")
    # Register the in-flight session's lane bookkeeping so we get past the
    # missing-correlation guard and reach the double-recycle assertion.
    strategy._correlation_to_lane["xcorr"] = 0

    final = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=2)
    with pytest.raises(RuntimeError, match="Double recycle"):
        await strategy.handle_credit_return(final)


# =============================================================================
# Test 6: Recycle during PROFILING-end cooldown -> no new sessions
# =============================================================================


@pytest.mark.asyncio
async def test_recycle_during_cooldown_does_not_start_new_sessions():
    """When DurationStopCondition has fired, in-flight credit returns must not
    spawn fresh sessions: cooldown is for finishing, not starting.

    Verifies the strategy honors stop_checker.can_start_new_session() in its
    recycle-spawn path. The finished trace_id IS still re-enqueued (cooldown
    gates *starting*, not preserving recycle FIFO state) but no fresh session
    is dispatched.
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=5, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    stop_checker = MagicMock()
    stop_checker.can_start_new_session.return_value = False  # post-stop
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
        stop_checker=stop_checker,
    )
    await strategy.setup_phase()
    initial_size = strategy._recycle_queue.qsize()
    assert initial_size == 4  # trace_1..trace_4

    # Register the in-flight session's lane bookkeeping.
    strategy._correlation_to_lane["xcorr"] = 0

    # Final turn arrives during cooldown.
    final = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=2)
    await strategy.handle_credit_return(final)

    # No new credit issued (cooldown gates spawning a fresh session).
    assert issuer.issue_credit.await_count == 0
    # Queue grew by 1: the finished trace_id was re-enqueued before the
    # cooldown gate so the recycle pool isn't permanently lossy across
    # cooldown boundaries.
    assert strategy._recycle_queue.qsize() == initial_size + 1
    tail = list(strategy._recycle_queue._queue)
    assert tail[-1] == "trace_0"


# =============================================================================
# Test 7: Pool=750, concurrency=100 -> every trace replayed; deterministic order
# =============================================================================


@pytest.mark.asyncio
async def test_large_pool_every_trace_replayed_deterministic_order():
    """750 traces, 100 trajectories, run for several recycle generations.

    Every non-trajectory trace must be served at least once. Trajectory traces also
    get recycled once their initial session ends. Order is deterministic given
    the trajectory layout because asyncio.Queue FIFO + sequential completion.
    """
    num_traces = 750
    trajectory_count = 100
    turns_per_trace = 2
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0)
        for i in range(trajectory_count)
    ]
    ds = _make_dataset(num_traces=num_traces, turns_per_trace=turns_per_trace)
    served: list[str] = []
    served_correlation_ids: list[str] = []

    async def capture(turn):
        served.append(turn.conversation_id)
        served_correlation_ids.append(turn.x_correlation_id)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    assert strategy._recycle_queue.qsize() == num_traces - trajectory_count  # 650

    # Snapshot initial queue order: it's insertion order over conversations
    # minus trajectory_ids -> trace_100, trace_101, ..., trace_749.
    initial_queue = list(strategy._recycle_queue._queue)
    assert initial_queue[0] == "trace_100"
    assert initial_queue[-1] == "trace_749"

    # Drive recycle generations realistically: each completed session must
    # have first been dispatched. The trajectory is initially "in flight" (its
    # k_i+1 dispatches happened in execute_phase, here we just simulate them).
    # We use a deque of (trace_id, correlation_id) for in-flight sessions; each
    # iteration finishes the head and the recycle path appends the just-
    # dispatched session's (trace_id, correlation_id) to the tail.
    from collections import deque

    # Seed the trajectory's correlation_ids: handle_credit_return now requires
    # finished_correlation_id to be present in _correlation_to_lane. Mimic
    # _execute_profiling's bookkeeping for the initial trajectory cohort.
    in_flight: deque[tuple[str, str]] = deque()
    for lane in range(trajectory_count):
        corr = f"xcorr_traj_{lane}"
        strategy._correlation_to_lane[corr] = lane
        in_flight.append((f"trace_{lane}", corr))

    total_completions = 1500
    for _ in range(total_completions):
        finishing_trace, finishing_corr = in_flight.popleft()
        # Snapshot len(served) BEFORE the call to know what trace_id was dispatched.
        before = len(served)
        await strategy.handle_credit_return(
            _make_credit(
                conversation_id=finishing_trace,
                turn_index=turns_per_trace - 1,
                num_turns=turns_per_trace,
                x_correlation_id=finishing_corr,
            )
        )
        # The recycle path always dispatches exactly one fresh session here
        # (queue is non-empty and credit_issuer is mocked truthy).
        assert len(served) == before + 1
        in_flight.append((served[-1], served_correlation_ids[-1]))

    # Every non-trajectory trace must have been served at least once.
    served_set = set(served)
    for i in range(trajectory_count, num_traces):
        assert f"trace_{i}" in served_set, f"trace_{i} never replayed"

    # Determinism: the first 650 served must equal the initial queue order
    # (because the very first 650 completions only pop from the initial queue
    # and push completed trajectory ids that are still behind those 650).
    assert served[: num_traces - trajectory_count] == initial_queue


# =============================================================================
# Test 8: Trajectory with N_i=1 (warmup-only) -> immediate recycle
# =============================================================================


@pytest.mark.asyncio
async def test_trajectory_with_one_turn_recycles_immediately_at_profiling_start():
    """Trajectory's trace has exactly one turn (k_i = 0 = last turn).

    PROFILING setup must not wait for a steady-state turn that never comes;
    the strategy must invoke the recycle path during _execute_profiling().
    """
    trajectory = [
        # trace_0 has 1 turn; k_i=0 is also the last turn.
        Trajectory(conversation_id="trace_0", start_turn_index=0),
    ]
    # Mixed-length dataset: trace_0 has 1 turn, trace_1+trace_2 have 3 turns.
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[TurnMetadata(timestamp_ms=None, delay_ms=None)],
            ),
            ConversationMetadata(
                conversation_id="trace_1",
                turns=[
                    TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(3)
                ],
            ),
            ConversationMetadata(
                conversation_id="trace_2",
                turns=[
                    TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(3)
                ],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    issued: list[tuple[str, int]] = []

    async def capture(turn):
        issued.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    # Strategy should have recycled trace_0 immediately, NOT issued at k_i+1=1.
    # The fresh session pulled from the recycle queue (trace_1 was head) must
    # be at turn 0.
    assert len(issued) == 1
    assert issued[0] == ("trace_1", 0)

    # trace_0 is now in the recycle queue tail (it got pushed after the pop).
    remaining = []
    while not strategy._recycle_queue.empty():
        remaining.append(strategy._recycle_queue.get_nowait())
    assert remaining == ["trace_2", "trace_0"]


# =============================================================================
# Test 9: Missing finished_correlation_id in _correlation_to_lane logs warning
# =============================================================================


@pytest.mark.asyncio
async def test_recycle_missing_correlation_id_logs_warning(caplog):
    """When _spawn_from_recycle_or_id is called with a finished_correlation_id
    that isn't tracked in _correlation_to_lane (per-session bookkeeping
    invariant violated upstream), the strategy logs a warning and falls back
    to lane 0 so the recycle still progresses (silent skip would wedge the
    queue head and break the test contract that recycle is unconditional on
    final-turn return).
    """
    import logging

    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()

    # Deliberately do NOT seed _correlation_to_lane for the finished id.
    strategy._correlation_to_lane.clear()

    with caplog.at_level(logging.WARNING, logger="AgenticReplayTiming"):
        await strategy._spawn_from_recycle_or_id(
            "trace_0",
            finished_correlation_id="xcorr_unknown",
        )

    invariant_msgs = [
        r.getMessage()
        for r in caplog.records
        if "bookkeeping invariant" in r.getMessage()
    ]
    assert invariant_msgs, (
        f"Expected bookkeeping-invariant warning; got: "
        f"{[r.getMessage() for r in caplog.records]}"
    )
    assert any("xcorr_unknown" in m for m in invariant_msgs)

    # The fallback path issues a fresh credit (lane 0) so recycle progresses.
    assert issuer.issue_credit.await_count == 1
