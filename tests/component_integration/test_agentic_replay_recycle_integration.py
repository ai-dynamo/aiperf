# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for AgenticReplayStrategy PROFILING-phase recycle
queue and per-turn delay-scheduling behavior.

Targets ``AgenticReplayStrategy._spawn_from_recycle_or_id`` (FIFO recycle with
cooldown gate) and ``_dispatch_next_turn`` (scheduler routing on positive
``delay_ms`` versus immediate dispatch on zero / None) at the strategy level.

Inter-turn-delay-cap clamping happens upstream in the loader; these tests
build synthetic ``ConversationMetadata`` directly with chosen ``delay_ms``
values to pin the strategy-level routing in isolation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.structs import Credit
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.component_integration


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class _DispatchLog:
    """Records each direct ``issue_credit`` call by (conversation_id, turn_index)."""

    entries: list[tuple[str, int]] = field(default_factory=list)


def _make_dataset_with_delays(
    num_traces: int,
    turn_delays_ms: list[int | float | None],
) -> DatasetMetadata:
    """Build a DatasetMetadata where every conversation has the same per-turn
    delay schedule.

    ``turn_delays_ms[i]`` is assigned to ``TurnMetadata.delay_ms`` for turn ``i``
    of every conversation; the conversation length equals ``len(turn_delays_ms)``.
    """
    convs: list[ConversationMetadata] = []
    for i in range(num_traces):
        turns = [
            TurnMetadata(timestamp_ms=None, delay_ms=delay) for delay in turn_delays_ms
        ]
        convs.append(ConversationMetadata(conversation_id=f"trace_{i}", turns=turns))
    return DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _make_recording_issuer(log: _DispatchLog) -> AsyncMock:
    """Build an AsyncMock credit issuer that records each direct dispatch."""
    issuer = AsyncMock()

    async def _issue(turn) -> bool:
        log.entries.append((turn.conversation_id, turn.turn_index))
        return True

    issuer.issue_credit.side_effect = _issue
    return issuer


def _make_stop_checker(allow_new_sessions: bool = True) -> MagicMock:
    sc = MagicMock()
    sc.can_start_new_session.return_value = allow_new_sessions
    return sc


def _make_credit(
    *,
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    x_correlation_id: str | None = None,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id
        if x_correlation_id is not None
        else uuid.uuid4().hex,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.FORK,
    )


def _build_source(
    *,
    num_traces: int,
    turn_delays_ms: list[int | float | None],
    concurrency: int,
    seed: int = 12345,
    force_k_zero: bool = True,
) -> TrajectorySource:
    """Build a TrajectorySource. With ``force_k_zero=True`` (default), every
    trajectory's ``start_turn_index`` is overridden to 0 so the strategy's
    resume-at-k_i+1 path lands on a deterministic turn for tests that pin
    delay-routing or recycle ordering.
    """
    from aiperf.timing.trajectory_source import Trajectory

    dataset = _make_dataset_with_delays(num_traces, turn_delays_ms)
    # Production wrapping sampler: recycle reuses this same sampler (continuing
    # from where the build loop left off) and wraps round-robin over the root
    # pool, so recycle order is predictable and never raises StopIteration.
    sampler = SequentialSampler([c.conversation_id for c in dataset.conversations])
    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=concurrency,
        random_seed=seed,
    )
    if force_k_zero:
        source.trajectories = [
            Trajectory(conversation_id=t.conversation_id, start_turn_index=0)
            for t in source.trajectories
        ]
    return source


def _build_profiling_strategy(
    *,
    source: TrajectorySource,
    issuer: AsyncMock,
    scheduler: MagicMock | None = None,
    stop_checker: MagicMock | None = None,
) -> AgenticReplayStrategy:
    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = len(source.trajectories)
    return AgenticReplayStrategy(
        config=cfg,
        conversation_source=source,
        scheduler=scheduler if scheduler is not None else MagicMock(),
        stop_checker=stop_checker if stop_checker is not None else _make_stop_checker(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )


def _final_credit_for(source: TrajectorySource, conversation_id: str) -> Credit:
    """Build a final-turn Credit for the given conversation (by metadata length)."""
    n = len(source._metadata_lookup[conversation_id].turns)
    return _make_credit(
        conversation_id=conversation_id,
        turn_index=n - 1,
        num_turns=n,
    )


# =============================================================================
# Test 1: multi-round recycle follows the dataset sampler's rotation order
# =============================================================================


@pytest.mark.asyncio
async def test_multi_round_recycle_preserves_fifo_order() -> None:
    """Push-then-pop FIFO semantics hold across many rounds under the
    full-pool initial recycle queue.

    Each final-turn credit return produces exactly one fresh ``issue_credit``
    call at ``turn_index == 0``, and the recycled conversation_id follows the
    dataset sampler's round-robin rotation. The build consumed one root per
    lane up front (the ``trajectory_ids``), so recycle continues the same
    rotation from there. With ``concurrency=2, pool=4``, drive ~12 final-turn
    returns and assert each fresh dispatch matches a parallel SequentialSampler
    replaying that rotation; over enough rounds every root appears.
    """
    source = _build_source(
        num_traces=4,
        turn_delays_ms=[None, None, None],
        concurrency=2,
    )
    assert len(source.trajectories) == 2
    trajectory_ids = [t.conversation_id for t in source.trajectories]
    all_trace_ids = [c.conversation_id for c in source.dataset_metadata.conversations]

    log = _DispatchLog()
    issuer = _make_recording_issuer(log)
    strategy = _build_profiling_strategy(source=source, issuer=issuer)

    await strategy.setup_phase()

    # Parallel sampler that replays the production round-robin: the build loop
    # consumed one root per trajectory built, so advance the predictor past
    # those before predicting recycle draws.
    predictor = SequentialSampler(all_trace_ids)
    for _ in range(len(trajectory_ids)):
        predictor.next_conversation_id()

    await strategy.execute_phase()
    # Each trajectory resumes at k_i + 1 = 1 (n=3, k_i=0). Both trajectory
    # sessions are now active at turn 1. We track which trace_ids are
    # currently in flight and at which turn so we can finalize them in
    # a controlled order.
    in_flight: dict[str, int] = {}
    for cid, idx in log.entries:
        in_flight[cid] = idx  # latest dispatched turn for each trace
    assert set(in_flight.keys()) == set(trajectory_ids)

    rounds = 0
    max_rounds = 12
    while rounds < max_rounds and in_flight:
        rounds += 1
        # Pick the in-flight session with the highest turn (closest to
        # final) to drive next; ties broken by trace_id to keep ordering
        # deterministic. This rotates through active lanes rather than
        # repeatedly finalizing the same trace_id.
        cid = sorted(in_flight.keys(), key=lambda k: (-in_flight[k], k))[0]
        idx = in_flight[cid]
        n = len(source._metadata_lookup[cid].turns)

        # Step to final via non-final returns (delay_ms=None -> direct
        # dispatch). Each non-final return dispatches the next turn.
        while idx < n - 1:
            step_credit = _make_credit(conversation_id=cid, turn_index=idx, num_turns=n)
            pre_step_len = len(log.entries)
            await strategy.handle_credit_return(step_credit)
            assert len(log.entries) == pre_step_len + 1
            assert log.entries[-1] == (cid, idx + 1)
            idx += 1
        in_flight[cid] = idx

        # Now cid is at its final turn. Recycle draws the next root from the
        # shared dataset sampler; predict it by advancing the parallel sampler.
        predicted = predictor.next_conversation_id()

        pre_len = len(log.entries)
        final_credit = _make_credit(conversation_id=cid, turn_index=n - 1, num_turns=n)
        await strategy.handle_credit_return(final_credit)
        # cid's session is done; remove from in_flight.
        del in_flight[cid]

        post_len = len(log.entries)
        assert post_len == pre_len + 1, (
            f"round {rounds}: final-turn return must trigger one fresh dispatch"
        )
        fresh_cid, fresh_idx = log.entries[-1]
        assert fresh_idx == 0, "fresh recycle dispatch must start at turn 0"
        assert fresh_cid == predicted, (
            f"round {rounds}: recycle must follow the sampler rotation -- "
            f"expected {predicted!r}, got {fresh_cid!r}"
        )
        # The freshly dispatched session is now in flight at turn 0.
        in_flight[fresh_cid] = fresh_idx

    # Sanity: every trace_id in the dataset must have been touched at least once.
    seen = {cid for cid, _ in log.entries}
    for tid in all_trace_ids:
        assert tid in seen, f"trace_id {tid!r} never dispatched over {rounds} rounds"


# =============================================================================
# Test 2: zero delay -> immediate dispatch, no scheduler call
# =============================================================================


@pytest.mark.asyncio
async def test_dispatch_next_turn_with_zero_delay_dispatches_immediately() -> None:
    """``delay_ms = 0`` must route through the direct ``issue_credit`` await."""
    source = _build_source(num_traces=1, turn_delays_ms=[0, 0, 0], concurrency=1)
    assert len(source.trajectories) == 1
    trajectory = source.trajectories[0]

    log = _DispatchLog()
    issuer = _make_recording_issuer(log)
    scheduler = MagicMock()
    strategy = _build_profiling_strategy(
        source=source, issuer=issuer, scheduler=scheduler
    )

    await strategy.setup_phase()
    await strategy.execute_phase()
    pre_len = len(log.entries)
    assert pre_len >= 1, "execute_phase must dispatch the resume turn (k_i + 1)"
    last_cid, last_idx = log.entries[-1]
    n = len(source._metadata_lookup[last_cid].turns)
    assert last_idx < n - 1, "test setup expects a non-final resume index"

    non_final_credit = _make_credit(
        conversation_id=last_cid, turn_index=last_idx, num_turns=n
    )
    await strategy.handle_credit_return(non_final_credit)

    assert len(log.entries) == pre_len + 1, (
        "zero-delay non-final return must immediately issue the next turn"
    )
    assert log.entries[-1] == (last_cid, last_idx + 1)
    assert scheduler.schedule_later.call_count == 0, (
        "zero-delay path must NOT route through scheduler.schedule_later"
    )
    # Silence trajectory unused warning.
    assert trajectory.conversation_id == last_cid


# =============================================================================
# Test 3: positive delay -> scheduler.schedule_later, no direct dispatch
# =============================================================================


@pytest.mark.asyncio
async def test_dispatch_next_turn_with_positive_delay_routes_through_scheduler() -> (
    None
):
    """``delay_ms > 0`` must route through ``scheduler.schedule_later`` with the
    correct seconds and a coroutine; no direct ``issue_credit`` await for that
    turn."""
    # Schedule: [None, 2500, None]. With k_i forced to 0, execute_phase
    # resumes at turn 1 (delay_ms=2500 lives ON turn 1, but execute_phase
    # issues directly without honoring turn 1's own delay -- delay_ms gates
    # the *transition* into the next turn from _dispatch_next_turn).
    #
    # To pin the scheduler path, we send a non-final return for turn 0, which
    # triggers _dispatch_next_turn for the *next* turn (turn 1) whose
    # delay_ms=2500 routes through scheduler.schedule_later.
    source = _build_source(
        num_traces=1,
        turn_delays_ms=[None, 2500, None],
        concurrency=1,
    )
    trajectory_id = source.trajectories[0].conversation_id

    log = _DispatchLog()
    issuer = _make_recording_issuer(log)
    scheduler = MagicMock()
    strategy = _build_profiling_strategy(
        source=source, issuer=issuer, scheduler=scheduler
    )

    await strategy.setup_phase()
    await strategy.execute_phase()
    # execute_phase dispatched turn 1 directly (resume = k_i + 1 = 1).
    pre_len = len(log.entries)
    assert pre_len == 1
    assert log.entries[-1] == (trajectory_id, 1)
    assert scheduler.schedule_later.call_count == 0

    # Send non-final return for turn 0; the strategy looks up turn 1's
    # delay_ms=2500 and routes through scheduler.schedule_later.
    non_final = _make_credit(
        conversation_id=trajectory_id,
        turn_index=0,
        num_turns=3,
    )
    await strategy.handle_credit_return(non_final)

    # No new direct dispatch (the scheduler-bound coroutine has not been
    # awaited).
    assert len(log.entries) == pre_len, (
        "positive-delay next turn must NOT be issued directly via issue_credit"
    )
    assert scheduler.schedule_later.call_count == 1, (
        "positive-delay next turn must route through scheduler.schedule_later"
    )

    call_args = scheduler.schedule_later.call_args
    seconds, coro = call_args.args
    assert seconds == pytest.approx(2.5), (
        f"scheduler delay must be 2500ms / 1000 = 2.5s, got {seconds}"
    )
    assert hasattr(coro, "close"), "scheduler arg must be a coroutine-like object"
    coro.close()  # avoid "coroutine was never awaited" warning


# =============================================================================
# Test 4: None delay -> immediate dispatch, no scheduler call
# =============================================================================


@pytest.mark.asyncio
async def test_dispatch_next_turn_with_none_delay_dispatches_immediately() -> None:
    """``delay_ms = None`` must route through the direct ``issue_credit`` await."""
    source = _build_source(
        num_traces=1, turn_delays_ms=[None, None, None], concurrency=1
    )
    log = _DispatchLog()
    issuer = _make_recording_issuer(log)
    scheduler = MagicMock()
    strategy = _build_profiling_strategy(
        source=source, issuer=issuer, scheduler=scheduler
    )

    await strategy.setup_phase()
    await strategy.execute_phase()
    pre_len = len(log.entries)
    last_cid, last_idx = log.entries[-1]
    n = len(source._metadata_lookup[last_cid].turns)
    assert last_idx < n - 1

    non_final = _make_credit(conversation_id=last_cid, turn_index=last_idx, num_turns=n)
    await strategy.handle_credit_return(non_final)

    assert len(log.entries) == pre_len + 1, (
        "None-delay non-final return must immediately issue the next turn"
    )
    assert log.entries[-1] == (last_cid, last_idx + 1)
    assert scheduler.schedule_later.call_count == 0


# =============================================================================
# Test 5: burst final-turn returns recycle in input order
# =============================================================================


@pytest.mark.asyncio
async def test_burst_final_turn_returns_recycle_in_input_order() -> None:
    """Three final-turn returns -> 3 fresh turn-0 dispatches following the
    sampler rotation (NOT the finishing ids).

    With concurrency=3 over a 6-trace pool, the build consumed ``trace_0,
    trace_1, trace_2`` for the three lanes; recycle reuses the SAME sampler, so
    it continues the round-robin from there. Finishing the three lanes in turn
    yields ``trace_3, trace_4, trace_5`` -- recycle follows the sampler
    rotation, not the finishing order.
    """
    source = _build_source(num_traces=6, turn_delays_ms=[None, None], concurrency=3)
    assert len(source.trajectories) == 3
    trajectory_ids = [t.conversation_id for t in source.trajectories]
    assert trajectory_ids == ["trace_0", "trace_1", "trace_2"], (
        "sequential sampler must yield trace_0..trace_2 as trajectories"
    )
    all_trace_ids = [c.conversation_id for c in source.dataset_metadata.conversations]

    log = _DispatchLog()
    issuer = _make_recording_issuer(log)
    strategy = _build_profiling_strategy(source=source, issuer=issuer)

    await strategy.setup_phase()

    # Parallel sampler replaying the production rotation, advanced past the ids
    # consumed by the build (one per lane).
    predictor = SequentialSampler(all_trace_ids)
    for _ in range(len(trajectory_ids)):
        predictor.next_conversation_id()
    expected_recycled = [predictor.next_conversation_id() for _ in trajectory_ids]
    assert expected_recycled == ["trace_3", "trace_4", "trace_5"]

    await strategy.execute_phase()
    pre_burst_len = len(log.entries)

    # Drive final-turn returns for the three lanes in order. Each recycle draws
    # the next root from the shared sampler.
    for finishing_id in trajectory_ids:
        pre_step = len(log.entries)
        await strategy.handle_credit_return(_final_credit_for(source, finishing_id))
        assert len(log.entries) == pre_step + 1, (
            f"final-turn return for {finishing_id!r} must produce one fresh dispatch"
        )

    new_dispatches = log.entries[pre_burst_len:]
    assert [cid for cid, _ in new_dispatches] == expected_recycled, (
        f"recycle order must follow the sampler rotation {expected_recycled}, "
        f"got {[cid for cid, _ in new_dispatches]}"
    )
    assert all(idx == 0 for _, idx in new_dispatches), (
        "every recycled dispatch must start at turn 0"
    )


# =============================================================================
# Test 6: cooldown flips mid-burst -> remaining final-turn returns are no-op
# =============================================================================


@pytest.mark.asyncio
async def test_cooldown_flips_mid_burst_blocks_remaining_spawns() -> None:
    """Once ``stop_checker.can_start_new_session`` returns False, subsequent
    final-turn returns must NOT produce any new dispatches."""
    source = _build_source(num_traces=8, turn_delays_ms=[None, None], concurrency=4)
    assert len(source.trajectories) == 4
    trajectory_ids = [t.conversation_id for t in source.trajectories]

    log = _DispatchLog()
    issuer = _make_recording_issuer(log)
    stop_checker = _make_stop_checker(allow_new_sessions=True)
    strategy = _build_profiling_strategy(
        source=source, issuer=issuer, stop_checker=stop_checker
    )

    await strategy.setup_phase()
    await strategy.execute_phase()
    after_execute_len = len(log.entries)
    # All 4 trajectories resume at k_i + 1 (turns_per_trace=2 means k_max=1
    # so k_i in {0, 1}; if k_i=1, resume_index=2 which equals n=2 so the
    # strategy recycles immediately rather than dispatching at k+1).
    # In either case execute_phase produces exactly 4 dispatches.
    assert after_execute_len == 4

    # Pre-cooldown: drive 2 final-turn returns -> 2 new dispatches.
    await strategy.handle_credit_return(_final_credit_for(source, trajectory_ids[0]))
    await strategy.handle_credit_return(_final_credit_for(source, trajectory_ids[1]))
    after_pre_cooldown = len(log.entries)
    assert after_pre_cooldown == after_execute_len + 2

    # Flip cooldown.
    stop_checker.can_start_new_session.return_value = False

    # Post-cooldown: 2 more final-turn returns -> 0 new dispatches.
    await strategy.handle_credit_return(_final_credit_for(source, trajectory_ids[2]))
    await strategy.handle_credit_return(_final_credit_for(source, trajectory_ids[3]))
    final_len = len(log.entries)
    assert final_len == after_pre_cooldown, (
        "post-cooldown final-turn returns must NOT spawn new sessions; "
        f"saw {final_len - after_pre_cooldown} extra dispatches"
    )

    # Total exact accounting: 4 initial + 2 pre-cooldown spawns + 0 post-cooldown.
    assert final_len == 4 + 2


# =============================================================================
# Test 7: cooldown gate -> no spawn, no exception
# =============================================================================


@pytest.mark.asyncio
async def test_empty_recycle_queue_with_cooldown_no_spawn_no_exception() -> None:
    """With cooldown active, a final-turn ``handle_credit_return`` must be a
    clean no-op: the cooldown gate short-circuits recycle before the sampler is
    even consulted."""
    source = _build_source(num_traces=2, turn_delays_ms=[None, None], concurrency=1)
    assert len(source.trajectories) == 1
    trajectory_id = source.trajectories[0].conversation_id  # trace_0

    log = _DispatchLog()
    issuer = _make_recording_issuer(log)
    stop_checker = _make_stop_checker(allow_new_sessions=True)
    strategy = _build_profiling_strategy(
        source=source, issuer=issuer, stop_checker=stop_checker
    )

    await strategy.setup_phase()
    await strategy.execute_phase()
    after_execute = len(log.entries)

    # Flip cooldown so _dispatch_recycled_on_lane short-circuits at the gate.
    stop_checker.can_start_new_session.return_value = False

    # No exception raised, no new dispatch on the final-turn return.
    await strategy.handle_credit_return(_final_credit_for(source, trajectory_id))
    assert len(log.entries) == after_execute, (
        "cooldown gate must short-circuit: no fresh dispatch when "
        "can_start_new_session is False"
    )
