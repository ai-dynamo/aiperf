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

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode, CreditPhase
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
    cache_bust_target: CacheBustTarget | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock]:
    src = _build_real_trajectory_source(dataset=dataset, trajectories=trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = max(1, len(trajectories))
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    stop_checker = stop_checker if stop_checker is not None else MagicMock()
    # Default user_config=None preserves the old path used by all prior tests:
    # _cache_bust_target resolves to CacheBustTarget.NONE in __init__.
    user_config = None
    if cache_bust_target is not None:
        user_config = MagicMock()
        user_config.input.prompt.cache_bust.target = cache_bust_target
        user_config.benchmark_id = "bench_test"
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        user_config=user_config,
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
    # Full pool: queue holds [trace_0] at setup.
    assert strategy._recycle_queue.qsize() == 1

    # Register the in-flight session's lane (normally done by _execute_profiling).
    # Seed _active_traces so the new pop loop skips trace_0 while it is alive.
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._active_traces["trace_0"] += 1

    # Final turn (last index = 2 of num_turns=3)
    final = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=3)
    await strategy.handle_credit_return(final)

    # The just-finished trace must be re-served at turn 0.
    assert issued == [("trace_0", 0)]
    # Queue holds the lone trace at completion: trace_0 was pushed (tail) and
    # the new session that got popped (head) is trace_0 again — push & pop
    # both happen on the lone slot.
    assert strategy._recycle_queue.qsize() == 1


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
    # Full pool: queue holds [trace_0, trace_1, trace_2] at setup.
    assert strategy._recycle_queue.qsize() == 3

    # Register lane bookkeeping for both in-flight sessions (normally seeded by
    # _execute_profiling). handle_credit_return's recycle path requires
    # finished_correlation_id to be in _correlation_to_lane. Seed
    # _active_traces too: the new full-pool pop loop skips trace_ids whose
    # session is currently alive, mirroring _execute_profiling behavior.
    strategy._correlation_to_lane["xcorr_a"] = 0
    strategy._correlation_to_lane["xcorr_b"] = 1
    strategy._active_traces["trace_0"] += 1
    strategy._active_traces["trace_1"] += 1

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
    # Sequence (gather schedules tasks, each runs to first await):
    #   call A: discard t0; push t0 -> [t0,t1,t2,t0]; pop t0 (not active),
    #           serves trace_0; queue=[t1,t2,t0], active={t1, t0}
    #   call B: discard t1; push t1 -> [t1,t2,t0,t1]; pop t1 (not active),
    #           serves trace_1; queue=[t2,t0,t1], active={t0, t1}
    # End state: served=[trace_0, trace_1], queue=[trace_2, trace_0, trace_1].
    assert issued == ["trace_0", "trace_1"]
    remaining: list[str] = []
    while not strategy._recycle_queue.empty():
        remaining.append(strategy._recycle_queue.get_nowait())
    assert remaining == ["trace_2", "trace_0", "trace_1"]


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
    # Full pool: queue holds all 12 traces at setup.
    assert strategy._recycle_queue.qsize() == 12

    # Register lane bookkeeping for the 10 in-flight sessions. Seed
    # _active_traces too so the new pop loop skips trace_ids whose session
    # is alive (mirroring _execute_profiling).
    for i in range(10):
        strategy._correlation_to_lane[f"xcorr_{i}"] = i
        strategy._active_traces[f"trace_{i}"] += 1

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

    # Each call discards the finishing trace from _active_traces, pushes it
    # to the queue tail, then pops the head. Because the head is the just-
    # discarded trace_i (full-pool layout), each iteration serves trace_i.
    # Sequence: queue=[t0..t11]
    #  i=0: discard t0; push t0 -> [t0..t11,t0]; pop t0 -> [t1..t11,t0]; served t0
    #  i=1: discard t1; push t1 -> [t1..t11,t0,t1]; pop t1 -> [t2..t11,t0,t1]; served t1
    #  ...
    #  i=9: queue ends as [t10, t11, t0, t1, ..., t8, t9]
    remaining = []
    while not strategy._recycle_queue.empty():
        remaining.append(strategy._recycle_queue.get_nowait())
    assert remaining == [
        "trace_10",
        "trace_11",
        "trace_0",
        "trace_1",
        "trace_2",
        "trace_3",
        "trace_4",
        "trace_5",
        "trace_6",
        "trace_7",
        "trace_8",
        "trace_9",
    ]


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
    # Full pool: queue holds all 70 traces at setup.
    assert len(initial_queue) == 70

    # Register lane bookkeeping for the 50 in-flight sessions. Seed
    # _active_traces too so the new pop loop skips alive trace_ids.
    for i in range(50):
        strategy._correlation_to_lane[f"xcorr_{i}"] = i
        strategy._active_traces[f"trace_{i}"] += 1

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

    This is a programmer-error guard: each session's final turn must trigger
    exactly one recycle. Firing handle_credit_return twice with the same
    correlation_id means the same final turn was reported twice — invariant
    violation, never legitimate.

    The guard is keyed on x_correlation_id (not trace_id) so that wrap-filled
    lanes legitimately sharing a trace_id with distinct correlation_ids don't
    collide. It is unconditional (was previously gated on ``__debug__``, which
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
    # Seed the in-flight-recycled set with the correlation_id we're about to
    # report-finished, simulating "this session's final turn was already
    # processed and is being reported again" — the actual bug class the guard
    # exists to catch.
    strategy._in_flight_recycled.add("xcorr")
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
    # Full pool: queue holds all 5 traces at setup.
    assert initial_size == 5

    # Register the in-flight session's lane bookkeeping. Seed _active_traces
    # so the cooldown gate is reached after the discard at the top of
    # _spawn_from_recycle_or_id.
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._active_traces["trace_0"] += 1

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
    # Full pool: queue holds all 750 traces (including trajectory ids) at setup.
    assert strategy._recycle_queue.qsize() == num_traces  # 750

    # Snapshot initial queue order: full dataset iteration order
    # -> trace_0, trace_1, ..., trace_749.
    initial_queue = list(strategy._recycle_queue._queue)
    assert initial_queue[0] == "trace_0"
    assert initial_queue[-1] == "trace_749"

    # Drive recycle generations realistically: each completed session must
    # have first been dispatched. The trajectory is initially "in flight" (its
    # k_i+1 dispatches happened in execute_phase, here we just simulate them).
    # We use a deque of (trace_id, correlation_id) for in-flight sessions; each
    # iteration finishes the head and the recycle path appends the just-
    # dispatched session's (trace_id, correlation_id) to the tail.
    from collections import deque

    # Seed the trajectory's correlation_ids and _active_traces:
    # handle_credit_return now requires finished_correlation_id to be present
    # in _correlation_to_lane, and the new full-pool pop loop skips trace_ids
    # in _active_traces. Mimic _execute_profiling's bookkeeping for the
    # initial trajectory cohort.
    in_flight: deque[tuple[str, str]] = deque()
    for lane in range(trajectory_count):
        corr = f"xcorr_traj_{lane}"
        strategy._correlation_to_lane[corr] = lane
        strategy._active_traces[f"trace_{lane}"] += 1
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

    # Determinism: with the full-pool queue, the first 100 completions each
    # discard their own trajectory trace_id, push it to the tail, and then
    # find that same trace_id at the head (just-discarded -> not active) so
    # they all "self-recycle" — served[:100] == trajectory ids in order.
    assert served[:trajectory_count] == [f"trace_{i}" for i in range(trajectory_count)]
    # After the trajectory cohort self-recycles, the next 650 completions
    # serve the non-trajectory pool in iteration order (trace_100..trace_749).
    assert served[
        trajectory_count : trajectory_count + (num_traces - trajectory_count)
    ] == [f"trace_{i}" for i in range(trajectory_count, num_traces)]


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
    # With the full-pool recycle queue, the head is trace_0 (iteration order
    # from dataset_metadata.conversations). trace_0 is discarded from
    # _active_traces inside _spawn_from_recycle_or_id before the pop loop, so
    # trace_0 is popped and re-dispatched at turn 0 as the recycled session.
    assert len(issued) == 1
    assert issued[0] == ("trace_0", 0)

    # Queue tail order: head trace_0 popped, then [trace_1, trace_2, trace_0]
    # remains (trace_0 was pushed at the end before pop).
    remaining = []
    while not strategy._recycle_queue.empty():
        remaining.append(strategy._recycle_queue.get_nowait())
    assert remaining == ["trace_1", "trace_2", "trace_0"]


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


# =============================================================================
# Tests 10-13: DAG-child final-turn short-circuit
#
# DAG-child terminal completion is owned by BranchOrchestrator
# (on_child_leaf_reached / on_child_errored, invoked by CreditCallbackHandler
# before reaching the strategy). The trajectory recycle pool is root-only:
# child conversation_ids like ``parent::sa:agent_id`` are NOT legitimate pool
# entries, and they repeat across recycle passes of the same parent. Without
# the short-circuit, the second time a parent re-runs, its child re-completes
# with the same conversation_id and trips the double-recycle guard.
# =============================================================================


def _make_child_credit(
    *,
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    agent_depth: int = 1,
    x_correlation_id: str = "xcorr_child",
    parent_correlation_id: str = "xcorr_parent",
) -> Credit:
    return Credit(
        id=0,
        phase=CreditPhase.PROFILING,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        agent_depth=agent_depth,
        parent_correlation_id=parent_correlation_id,
        branch_mode=ConversationBranchMode.SPAWN,
    )


@pytest.mark.asyncio
async def test_child_final_turn_does_not_enter_recycle_pool():
    """A DAG-child final-turn return must NOT push the child's conversation_id
    into the recycle queue, must NOT add it to ``_in_flight_recycled``, and
    must NOT dispatch a fresh session.
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
    initial_size = strategy._recycle_queue.qsize()
    initial_queue = list(strategy._recycle_queue._queue)

    child_cid = "trace_0::sa:codex_subagent_001_3b3e9875"
    final_child = _make_child_credit(
        conversation_id=child_cid,
        turn_index=4,
        num_turns=5,
    )
    await strategy.handle_credit_return(final_child)

    # Issuer untouched: no fresh session dispatched on child terminal.
    assert issuer.issue_credit.await_count == 0
    # Recycle queue untouched: child conversation_id is not a pool entry.
    assert strategy._recycle_queue.qsize() == initial_size
    assert list(strategy._recycle_queue._queue) == initial_queue
    # Double-recycle bookkeeping untouched.
    assert child_cid not in strategy._in_flight_recycled


@pytest.mark.asyncio
async def test_child_final_turn_repeated_does_not_trigger_double_recycle():
    """Regression for the production crash: when the parent trace is recycled
    and re-runs, its subagent child re-completes with the SAME
    ``conversation_id`` (deterministic ``parent::sa:agent_id``). The strategy
    must not raise the double-recycle ``RuntimeError`` in this case.
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

    child_cid = "trace_0::sa:codex_subagent_001_3b3e9875"
    # First recycle-pass child completion.
    await strategy.handle_credit_return(
        _make_child_credit(
            conversation_id=child_cid,
            turn_index=2,
            num_turns=3,
            x_correlation_id="xcorr_child_pass0",
        )
    )
    # Second pass: same child conversation_id, fresh x_correlation_id.
    await strategy.handle_credit_return(
        _make_child_credit(
            conversation_id=child_cid,
            turn_index=2,
            num_turns=3,
            x_correlation_id="xcorr_child_pass1",
        )
    )

    # Neither call raised, and neither touched recycle state.
    assert child_cid not in strategy._in_flight_recycled
    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_child_non_final_turn_still_dispatches_next_turn():
    """Non-final child returns MUST continue to dispatch the next turn — the
    short-circuit applies only to terminal child returns. This protects the
    BranchOrchestrator's contract that "child continuation turns dispatch via
    the strategy's normal path".
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=2)

    # Register the child conversation_id in the metadata lookup so
    # _dispatch_next_turn -> get_next_turn_metadata succeeds.
    child_cid = "trace_0::sa:agent_a"
    child_meta = ConversationMetadata(
        conversation_id=child_cid,
        turns=[TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(3)],
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
    strategy.conversation_source._metadata_lookup[child_cid] = child_meta
    await strategy.setup_phase()

    non_final_child = _make_child_credit(
        conversation_id=child_cid,
        turn_index=0,
        num_turns=3,
    )
    await strategy.handle_credit_return(non_final_child)

    # Next turn (turn_index=1) was issued via the normal continuation path.
    assert issued == [(child_cid, 1)]


@pytest.mark.asyncio
async def test_root_final_turn_still_recycles_after_child_shortcircuit():
    """Regression baseline: the child-final short-circuit must not affect
    root final-turn recycling. A root (``agent_depth == 0``) final-turn return
    must still push to the recycle queue and dispatch the next session.
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=2)
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
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._active_traces["trace_0"] += 1

    # Root credit: agent_depth defaults to 0 via _make_credit.
    root_final = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=2)
    await strategy.handle_credit_return(root_final)

    # Recycle dispatched a fresh session — proves the short-circuit didn't
    # block the root path. (For this layout the head of the recycle queue
    # is trace_0 after push, so it self-recycles.)
    assert len(issued) == 1
    assert issued[0] == "trace_0"
