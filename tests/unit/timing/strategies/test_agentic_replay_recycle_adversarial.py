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
from aiperf.dataset.dataset_samplers import SequentialSampler
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
    _roots = [
        c.conversation_id
        for c in src._dataset_metadata.conversations
        if getattr(c, "is_root", True)
    ]
    src._dataset_sampler = SequentialSampler(_roots) if _roots else MagicMock()
    src._pool_size = len(_roots)
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

    # Register the in-flight session's lane (normally done by _execute_profiling).
    strategy._correlation_to_lane["xcorr"] = 0

    # Final turn (last index = 2 of num_turns=3)
    final = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=3)
    await strategy.handle_credit_return(final)

    # The sampler's lone root is re-served at turn 0.
    assert issued == [("trace_0", 0)]


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

    # Both consumers fired exactly one new credit; no deadlock (wait_for above
    # would have timed out). Each recycle draws the next root from the
    # sequential sampler (index 0, 1) -> trace_0 then trace_1.
    assert len(issued) == 2
    assert issued == ["trace_0", "trace_1"]


# =============================================================================
# Test 3: Burst of 10 completions within one tick -> order preserved
# =============================================================================


@pytest.mark.asyncio
async def test_burst_of_ten_completions_recycle_in_sampler_order():
    """10 sessions complete sequentially within the same loop tick.

    Each completion draws the next root from the sequential sampler, so the
    10 recycled sessions are trace_0..trace_9 in sampler order (the sampler
    starts at index 0 for this manually-built source).
    """
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(10)
    ]
    ds = _make_dataset(num_traces=12, turns_per_trace=2)
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

    # Register lane bookkeeping for the 10 in-flight sessions.
    for i in range(10):
        strategy._correlation_to_lane[f"xcorr_{i}"] = i

    # Fire 10 completions in order: trace_0..trace_9 finish.
    for i in range(10):
        await strategy.handle_credit_return(
            _make_credit(
                conversation_id=f"trace_{i}",
                turn_index=1,
                num_turns=2,
                x_correlation_id=f"xcorr_{i}",
            )
        )

    # Round-robin over the 12-root pool, starting at index 0.
    assert served == [f"trace_{i}" for i in range(10)]


# =============================================================================
# Test 4: Push-back races concurrent pop -> no lost or duplicated trace_ids
# =============================================================================


@pytest.mark.asyncio
async def test_concurrent_recycle_serves_distinct_roots_from_pool():
    """Drive 50 completions concurrently via asyncio.gather.

    Each completion draws one fresh root from the sequential sampler. With a
    70-root pool and 50 completions (< one full pass), the served roots are
    distinct and all belong to the root pool -- nothing is lost or duplicated.
    """
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(50)
    ]
    ds = _make_dataset(num_traces=70, turns_per_trace=2)
    root_ids = {c.conversation_id for c in ds.conversations}
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

    # One fresh dispatch per completion, all from the root pool, all distinct
    # (50 < 70 roots, so no wrap and no repeats).
    assert len(served) == 50
    assert set(served) <= root_ids
    assert len(set(served)) == 50


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
    recycle-spawn path -- no fresh session is dispatched during cooldown.
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

    # Register the in-flight session's lane bookkeeping.
    strategy._correlation_to_lane["xcorr"] = 0

    # Final turn arrives during cooldown.
    final = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=2)
    await strategy.handle_credit_return(final)

    # No new credit issued (cooldown gates spawning a fresh session).
    assert issuer.issue_credit.await_count == 0


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

    # Drive recycle generations realistically: each completed session must
    # have first been dispatched. The trajectory is initially "in flight" (its
    # k_i+1 dispatches happened in execute_phase, here we just simulate them).
    # We use a deque of (trace_id, correlation_id) for in-flight sessions; each
    # iteration finishes the head and the recycle path appends the just-
    # dispatched session's (trace_id, correlation_id) to the tail.
    from collections import deque

    # Seed the trajectory's correlation_ids: handle_credit_return requires
    # finished_correlation_id to be present in _correlation_to_lane.
    # _active_traces was already pre-registered by setup_phase.
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

    # Every trace must have been served at least once (1500 completions over a
    # 750-root pool -> two full round-robin passes).
    served_set = set(served)
    for i in range(num_traces):
        assert f"trace_{i}" in served_set, f"trace_{i} never replayed"

    # Determinism: the sequential sampler (index 0 for this manually-built
    # source) yields the roots in dataset order, so the first full pass is
    # trace_0..trace_749 and it then wraps.
    assert served[:num_traces] == [f"trace_{i}" for i in range(num_traces)]
    assert served[num_traces : 2 * num_traces] == [
        f"trace_{i}" for i in range(num_traces)
    ]


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

    # Strategy should have recycled immediately, NOT issued at k_i+1=1. The
    # recycle draws the next root from the sequential sampler (index 0 ->
    # trace_0), re-dispatched at turn 0.
    assert len(issued) == 1
    assert issued[0] == ("trace_0", 0)


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
async def test_child_final_turn_does_not_recycle():
    """A DAG-child final-turn return must NOT dispatch a fresh session and must
    NOT add the child's conversation_id to ``_in_flight_recycled``.
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
    final_child = _make_child_credit(
        conversation_id=child_cid,
        turn_index=4,
        num_turns=5,
    )
    await strategy.handle_credit_return(final_child)

    # Issuer untouched: no fresh session dispatched on child terminal.
    assert issuer.issue_credit.await_count == 0
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
    # Child continuations now route through the chokepoint
    # (dispatch_child_turn -> clean True-iff-on-wire) rather than the
    # overloaded issue_credit, so a cap refusal can be drained.
    issuer.dispatch_child_turn.side_effect = capture
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

    # Next turn (turn_index=1) was issued via the child-continuation chokepoint.
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

    # Root credit: agent_depth defaults to 0 via _make_credit.
    root_final = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=2)
    await strategy.handle_credit_return(root_final)

    # Recycle dispatched a fresh session — proves the short-circuit didn't
    # block the root path. (For this layout the head of the recycle queue
    # is trace_0 after push, so it self-recycles.)
    assert len(issued) == 1
    assert issued[0] == "trace_0"
