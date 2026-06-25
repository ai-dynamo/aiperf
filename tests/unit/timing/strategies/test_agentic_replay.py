# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AgenticReplayStrategy.

Tests the phase-aware trajectory dispatch (WARMUP) and resume-at-k+1 + recycle
(PROFILING) behaviors specified in agentx-mvp Spec §4.2.
"""

from __future__ import annotations

import asyncio
import logging
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.credit.structs import Credit
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    ConversationState,
    Trajectory,
    TrajectorySnapshot,
    TrajectorySource,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_dataset(num_traces: int, turns_per_trace: int) -> DatasetMetadata:
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
    num_traces: int,
    turns_per_trace: int,
    trajectories: list[Trajectory],
    dataset: DatasetMetadata | None = None,
) -> TrajectorySource:
    """Build a real TrajectorySource with deterministic trajectories.

    We construct the source via __new__ + manual init so we control the
    trajectories exactly (avoid randomization in tests).
    """
    ds = dataset if dataset is not None else _make_dataset(num_traces, turns_per_trace)

    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    # Recycle draws roots from this sampler; use a real SequentialSampler over
    # the root pool so recycle exercises the production round-robin path.
    root_ids = [
        c.conversation_id for c in ds.conversations if getattr(c, "is_root", True)
    ]
    src._dataset_sampler = SequentialSampler(root_ids) if root_ids else MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src._random_seed = 0
    src._target_size = len(trajectories)
    src._pool_size = len(root_ids)
    src.trajectories = list(trajectories)
    return src


def _make_strategy(
    *,
    phase: CreditPhase,
    trajectories: list[Trajectory],
    num_traces: int = 5,
    turns_per_trace: int = 4,
    issuer: AsyncMock | None = None,
    scheduler: MagicMock | None = None,
    user_config: object | None = None,
    dataset: DatasetMetadata | None = None,
    cache_warmup_duration: float | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock, TrajectorySource]:
    src = _build_real_trajectory_source(
        num_traces, turns_per_trace, trajectories, dataset=dataset
    )
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(trajectories)
    cfg.agentic_cache_warmup_duration_sec = cache_warmup_duration
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        user_config=user_config,
    )
    return strategy, issuer, scheduler, src


def _make_credit(
    *,
    conversation_id: str,
    x_correlation_id: str = "xcorr",
    turn_index: int,
    num_turns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK,
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        agent_depth=agent_depth,
        parent_correlation_id=parent_correlation_id,
        branch_mode=branch_mode,
    )


# =============================================================================
# Constructor validation
# =============================================================================


def test_constructor_rejects_unknown_phase():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    src = _build_real_trajectory_source(1, 2, trajectories)
    cfg = MagicMock()
    cfg.phase = "unknown"
    cfg.concurrency = 1
    with pytest.raises(ValueError):
        AgenticReplayStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=AsyncMock(),
            lifecycle=MagicMock(),
        )


def test_constructor_rejects_non_trajectory_source():
    """ConversationSource that is not a TrajectorySource is rejected."""
    cfg = MagicMock()
    cfg.phase = CreditPhase.WARMUP
    cfg.concurrency = 1
    plain_src = MagicMock()  # not a TrajectorySource instance
    with pytest.raises(TypeError):
        AgenticReplayStrategy(
            config=cfg,
            conversation_source=plain_src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=AsyncMock(),
            lifecycle=MagicMock(),
        )


def test_constructor_accepts_warmup_and_profiling():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    for phase in (CreditPhase.WARMUP, CreditPhase.PROFILING):
        strategy, *_ = _make_strategy(phase=phase, trajectories=trajectories)
        assert strategy.config.phase == phase


# =============================================================================
# WARMUP phase
# =============================================================================


def test_warmup_only_overrides_max_tokens():
    trajectory = Trajectory(conversation_id="trace_0", start_turn_index=1)
    warmup, _, _, warmup_source = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=[trajectory]
    )
    profiling, _, _, profiling_source = _make_strategy(
        phase=CreditPhase.PROFILING, trajectories=[trajectory]
    )

    warmup_session = warmup_source.session_for(trajectory)
    profiling_session = profiling_source.session_for(trajectory)

    assert warmup._build_turn_for_session(warmup_session, 0).max_tokens_override == 1
    assert (
        profiling._build_turn_for_session(profiling_session, 1).max_tokens_override
        is None
    )


@pytest.mark.asyncio
async def test_warmup_dispatches_one_credit_per_trajectory():
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert issuer.issue_credit.await_count == 3


@pytest.mark.asyncio
async def test_cache_warmup_starts_after_baseline_and_removes_idle_delay():
    trajectory = Trajectory(conversation_id="trace_0", start_turn_index=1)
    strategy, issuer, scheduler, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=[trajectory],
        cache_warmup_duration=600.0,
    )

    await strategy.execute_phase()
    baseline = issuer.issue_credit.await_args_list[0].args[0]
    assert baseline.turn_index == 1

    await strategy.handle_credit_return(
        _make_credit(
            conversation_id="trace_0",
            x_correlation_id=baseline.x_correlation_id,
            turn_index=1,
            num_turns=4,
            phase=CreditPhase.WARMUP,
        )
    )

    pressure = issuer.issue_credit.await_args_list[1].args[0]
    assert pressure.turn_index == 2
    assert pressure.max_tokens_override == 1
    assert pressure.is_session_start is False
    issuer.set_max_tokens_override.assert_called_once_with(1)
    scheduler.schedule_later.assert_called_once()
    assert scheduler.schedule_later.call_args.args[0] == 600.0


@pytest.mark.asyncio
async def test_cache_warmup_cutoff_stops_issuer_and_persists_next_turn():
    trajectory = Trajectory(conversation_id="trace_0", start_turn_index=1)
    strategy, issuer, _, source = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=[trajectory],
        cache_warmup_duration=10.0,
    )
    issuer.mark_sending_complete = MagicMock()

    await strategy.execute_phase()
    baseline = issuer.issue_credit.await_args_list[0].args[0]
    await strategy.handle_credit_return(
        _make_credit(
            conversation_id="trace_0",
            x_correlation_id=baseline.x_correlation_id,
            turn_index=1,
            num_turns=4,
            phase=CreditPhase.WARMUP,
        )
    )
    returned = _make_credit(
        conversation_id="trace_0",
        x_correlation_id=baseline.x_correlation_id,
        turn_index=2,
        num_turns=4,
        phase=CreditPhase.WARMUP,
    )
    strategy.observe_credit_return(returned)

    await strategy._finish_accelerated_warmup()
    await strategy.finalize_phase()

    issuer.mark_sending_complete.assert_called_once_with()
    snapshot = source.trajectories[0].snapshot
    assert snapshot is not None
    assert len(snapshot.states) == 1
    assert snapshot.states[0].next_turn_index == 3
    assert snapshot.states[0].x_correlation_id == baseline.x_correlation_id


@pytest.mark.asyncio
async def test_warmup_dispatch_uses_start_turn_index():
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=1),
        Trajectory(conversation_id="trace_2", start_turn_index=2),
    ]
    issued_turn_indices: list[int] = []

    async def capture(turn):
        issued_turn_indices.append(turn.turn_index)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert sorted(issued_turn_indices) == [0, 1, 2]


@pytest.mark.asyncio
async def test_warmup_warms_every_active_session_including_gated_parent():
    """Every session mid-flight at t* is warmed at its turn n-1: the
    mid-flight subagent (turn 0) AND the gated parent (turn 1, priming its
    join turn). Both count toward the warmup barrier."""
    parent_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="parent",
        next_turn_index=2,
        agent_depth=0,
        waiting_on_children=True,
        join_target_turn_index=2,
    )
    child_state = ConversationState(
        conversation_id="trace_1",
        x_correlation_id="child",
        next_turn_index=1,  # mid-flight: turn 0 < t*, turn 1 >= t*
        agent_depth=1,
        parent_correlation_id="parent",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    trajectories = [
        Trajectory(
            conversation_id="trace_0",
            start_turn_index=2,
            snapshot=TrajectorySnapshot(
                t_star_ms=0.0,
                states=(parent_state, child_state),
            ),
        )
    ]
    issued = []

    async def capture(turn):
        issued.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, src = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        issuer=issuer,
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    # Gated parent warmed at turn 1 (n-1); mid-flight child at turn 0 (n-1).
    by_cid = {t.conversation_id: t for t in issued}
    assert set(by_cid) == {"trace_0", "trace_1"}
    assert by_cid["trace_0"].turn_index == 1  # gated parent's last-before-t*
    assert by_cid["trace_1"].turn_index == 0  # mid-flight child's last-before-t*
    assert by_cid["trace_1"].agent_depth == 1
    assert all(t.counts_toward_phase_target for t in issued)
    assert src.warmup_credit_count == 2


@pytest.mark.asyncio
async def test_warmup_spreads_globally_aligned_on_t_star_by_default():
    """By default (spread), warmup aligns GLOBALLY across trajectories so every
    trajectory's t* lands at the same instant.

    Three trajectories, each one mid-flight root whose warmup request fired a
    different lead before its own t*: 5s, 15s, 10s. The furthest-before-t*
    (15s) fires immediately at warmup-time 0, the 10s one 5s later, the 5s one
    10s later -- dispatch offset = max_lead(15s) - lead. Total spread = 10s
    (15s - 5s). This is the exact example from the design discussion.
    """

    # (conversation_id, x_corr, t*, warm_ts) -> lead = t* - warm_ts.
    lanes = [
        ("t_a", "r_a", 5_000.0, 0.0),  # lead 5s
        ("t_b", "r_b", 15_000.0, 0.0),  # lead 15s (furthest before t*)
        ("t_c", "r_c", 10_000.0, 0.0),  # lead 10s
    ]
    convs = [
        ConversationMetadata(
            conversation_id=cid,
            turns=[
                TurnMetadata(timestamp_ms=warm_ts),  # turn 0: warmup (before t*)
                TurnMetadata(timestamp_ms=t_star + 1_000.0),  # turn 1: profiled
            ],
        )
        for cid, _, t_star, warm_ts in lanes
    ]
    ds = DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    trajectories = [
        Trajectory(
            conversation_id=cid,
            start_turn_index=1,
            snapshot=TrajectorySnapshot(
                t_star_ms=t_star,
                states=(
                    ConversationState(
                        conversation_id=cid,
                        x_correlation_id=xc,
                        next_turn_index=1,  # mid-flight: warmup turn = 0
                    ),
                ),
            ),
        )
        for cid, xc, t_star, _ in lanes
    ]
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src.trajectories = trajectories

    issued: list[str] = []

    async def capture(turn):
        issued.append(turn.x_correlation_id)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    scheduled: list[tuple[float, object]] = []
    scheduler = MagicMock()
    scheduler.schedule_later.side_effect = lambda delay, coro: scheduled.append(
        (delay, coro)
    )
    lifecycle = MagicMock()
    lifecycle.is_sending_complete = False

    cfg = MagicMock()
    cfg.phase = CreditPhase.WARMUP
    cfg.concurrency = 3
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    # The furthest-before-t* request (r_b, lead 15s) fires immediately at t=0.
    assert issued == ["r_b"]
    # r_c (lead 10s) at 5s; r_a (lead 5s) at 10s. Total spread = 10s.
    delay_by_corr = {}
    for delay, coro in scheduled:
        await coro  # drains -> appends to issued
        delay_by_corr[issued[-1]] = delay
    assert delay_by_corr["r_c"] == pytest.approx(5.0)
    assert delay_by_corr["r_a"] == pytest.approx(10.0)
    assert set(issued) == {"r_a", "r_b", "r_c"}

    # Spread path must NOT mark sending complete early (would refuse the
    # scheduled dispatches); the count path drives completion.
    lifecycle.mark_sending_complete.assert_not_called()


@pytest.mark.asyncio
async def test_warmup_lead_clamped_to_idle_gap_cap():
    """A per-conversation idle far exceeding the idle-gap cap is clamped so the
    warmup spread stays bounded by the cap, not the raw multi-hour idle.

    Two lanes: one warmed ~10s before its t*, one idle ~3h before its t*. With
    a 60s cap, the 3h lead clamps to 60s -> spread = 60 - 10 = 50s, not ~3h.
    """
    lanes = [
        ("t_near", "near", 10_000.0, 0.0),  # lead 10s
        ("t_idle", "idle", 10_800_000.0, 0.0),  # lead 3h -> clamps to cap
    ]
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id=cid,
                turns=[
                    TurnMetadata(timestamp_ms=warm_ts),
                    TurnMetadata(timestamp_ms=t_star + 1_000.0),
                ],
            )
            for cid, _, t_star, warm_ts in lanes
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    trajectories = [
        Trajectory(
            conversation_id=cid,
            start_turn_index=1,
            snapshot=TrajectorySnapshot(
                t_star_ms=t_star,
                states=(
                    ConversationState(
                        conversation_id=cid, x_correlation_id=xc, next_turn_index=1
                    ),
                ),
            ),
        )
        for cid, xc, t_star, _ in lanes
    ]
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src.trajectories = trajectories

    issued: list[str] = []

    async def capture(turn):
        issued.append(turn.x_correlation_id)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    scheduled: list[float] = []
    scheduler = MagicMock()
    scheduler.schedule_later.side_effect = lambda d, c: scheduled.append(d)
    lifecycle = MagicMock()
    lifecycle.is_sending_complete = False

    cfg = MagicMock()
    cfg.phase = CreditPhase.WARMUP
    cfg.concurrency = 2
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )
    # Idle-gap cap of 60s (what the agentx scenario sets).
    strategy._phase_offset_cap_ms = 60_000.0

    await strategy.setup_phase()
    await strategy.execute_phase()

    # idle lane's lead clamped 10800s -> 60s, so it is furthest-before-t* and
    # fires at 0; near lane (lead 10s) fires (60 - 10) = 50s later.
    assert issued == ["idle"]
    assert scheduled == [pytest.approx(50.0)]


@pytest.mark.asyncio
async def test_warmup_skips_pending_start_child():
    """A child whose recorded first request is after t* is not warmed.

    The server had not seen the stream at the snapshot instant, so warming
    its turn 0 would both fire it early and let the profiling continuation
    advance past a single-turn child entirely. ``warmup_credit_count`` must
    agree with the dispatch loop (the warmup barrier re-anchors to it).
    """
    root_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="root",
        next_turn_index=1,
        agent_depth=0,
    )
    pending_child = ConversationState(
        conversation_id="trace_0::sa:a",
        x_correlation_id="kid",
        next_turn_index=0,
        next_dispatch_offset_ms=76_000.0,
        agent_depth=1,
        parent_correlation_id="root",
        branch_id="b0",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    trajectories = [
        Trajectory(
            conversation_id="trace_0",
            start_turn_index=1,
            snapshot=TrajectorySnapshot(
                t_star_ms=10_000.0,
                states=(root_state, pending_child),
            ),
        )
    ]
    issued = []

    async def capture(turn):
        issued.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, src = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        issuer=issuer,
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    assert [t.conversation_id for t in issued] == ["trace_0"]
    assert src.warmup_credit_count == 1


@pytest.mark.asyncio
async def test_warmup_handle_credit_return_is_noop():
    """In WARMUP, handle_credit_return must not dispatch follow-up turns."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories, turns_per_trace=4
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    credit = _make_credit(
        conversation_id="trace_0",
        turn_index=0,
        num_turns=4,
        phase=CreditPhase.WARMUP,
    )
    await strategy.handle_credit_return(credit)

    assert issuer.issue_credit.await_count == 0


def test_report_warmup_failures_raises_when_failures_present():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, *_ = _make_strategy(phase=CreditPhase.WARMUP, trajectories=trajectories)
    strategy.record_warmup_failure("trace_0")
    strategy.record_warmup_failure("trace_3")
    with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
        strategy.report_warmup_failures()
    assert exc_info.value.failed_trace_ids == ["trace_0", "trace_3"]


def test_report_warmup_failures_silent_when_no_failures():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, *_ = _make_strategy(phase=CreditPhase.WARMUP, trajectories=trajectories)
    strategy.report_warmup_failures()  # must not raise


# =============================================================================
# PROFILING phase: setup_phase + execute_phase
# =============================================================================


@pytest.mark.asyncio
async def test_profiling_recycle_cycles_full_root_pool_in_sampler_order():
    """PROFILING recycle draws roots from the dataset sampler, so it cycles
    through the FULL root pool (sequential -> dataset order) then wraps -- every
    root is reused equally instead of a strategy-side queue favoring some."""
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_2", start_turn_index=1),
    ]
    issued: list[str] = []

    async def capture(turn):
        issued.append(turn.conversation_id)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=5,  # trace_0..trace_4
        issuer=issuer,
    )
    strategy.stop_checker.can_start_new_session.return_value = True
    await strategy.setup_phase()

    for _ in range(10):
        await strategy._dispatch_recycled_on_lane(0)

    # Two full round-robin passes over the 5 roots, in dataset order.
    assert issued == ["trace_0", "trace_1", "trace_2", "trace_3", "trace_4"] * 2


@pytest.mark.asyncio
async def test_profiling_phase_resumes_trajectory_at_k_plus_one():
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),  # resume at 1
        Trajectory(conversation_id="trace_1", start_turn_index=2),  # resume at 3
    ]
    captured: list[tuple[str, int]] = []

    async def capture(turn):
        captured.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        turns_per_trace=5,
        issuer=issuer,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert sorted(captured) == [("trace_0", 1), ("trace_1", 3)]


@pytest.mark.asyncio
async def test_profiling_snapshot_dispatches_inflight_child_and_seeds_join():
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=12000.0),
                    TurnMetadata(timestamp_ms=20000.0),
                ],
            ),
            ConversationMetadata(
                conversation_id="trace_0::sa:0",
                turns=[
                    TurnMetadata(timestamp_ms=13000.0),
                    TurnMetadata(timestamp_ms=14000.0),
                    TurnMetadata(timestamp_ms=16000.0, delay_ms=2000.0),
                ],
                is_root=False,
                agent_depth=1,
                parent_conversation_id="trace_0",
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    parent_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="parent",
        next_turn_index=2,
        agent_depth=0,
        waiting_on_children=True,
        join_target_turn_index=2,
    )
    child_state = ConversationState(
        conversation_id="trace_0::sa:0",
        x_correlation_id="child",
        next_turn_index=1,
        next_dispatch_offset_ms=500.0,
        agent_depth=1,
        parent_correlation_id="parent",
        join_target_turn_index=2,
        branch_id="b0",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=2,
        snapshot=TrajectorySnapshot(
            t_star_ms=13500.0,
            states=(parent_state, child_state),
        ),
    )
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src.trajectories = [trajectory]

    issued: list[tuple[str, int, int, str, str | None]] = []

    async def capture(turn):
        issued.append(
            (
                turn.conversation_id,
                turn.turn_index,
                turn.agent_depth,
                turn.x_correlation_id,
                turn.parent_correlation_id,
            )
        )
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    scheduler = MagicMock()
    scheduler.schedule_later.side_effect = lambda _delay, coro: asyncio.create_task(
        coro
    )
    branch_orchestrator = MagicMock()

    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        branch_orchestrator=branch_orchestrator,
    )
    strategy._burst_phase_starts = True  # exercise the burst (T0-normalize) path

    await strategy.setup_phase()
    await strategy.execute_phase()

    branch_orchestrator.seed_snapshot.assert_called_once()
    # Child is the only dispatchable stream -> it anchors T0 and fires
    # immediately (no schedule), profiling its own next_turn_index = 1
    # (turn 0 was warmed during WARMUP).
    scheduler.schedule_later.assert_not_called()
    assert issued == [
        (
            "trace_0::sa:0",
            1,
            1,
            "child",
            branch_orchestrator.seed_snapshot.call_args.args[0][
                1
            ].parent_correlation_id,
        )
    ]
    seeded_states = branch_orchestrator.seed_snapshot.call_args.args[0]
    assert seeded_states[0].x_correlation_id == "parent"
    assert seeded_states[1].x_correlation_id == "child"
    assert seeded_states[0].waiting_on_children is True
    assert seeded_states[1].parent_correlation_id == seeded_states[0].x_correlation_id


@pytest.mark.asyncio
async def test_profiling_burst_normalizes_offsets_first_request_fires_at_zero():
    """With --burst-phase-starts, profiling anchors the trajectory's earliest
    post-t* request at time 0 and preserves every other request's recorded
    relative offset.

    Two pending-start subagent chains spawn after t* under a gated parent.
    The earlier one (offset 20s from t*) fires immediately; the later one
    (offset 95s) fires 75s into profiling -- the recorded gap between them.
    The gated parent is not dispatched (its join fires when children drain).
    """
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=12_000.0),
                    TurnMetadata(timestamp_ms=200_000.0),
                ],
            ),
            ConversationMetadata(
                conversation_id="trace_0::sa:a:fa:000",
                turns=[TurnMetadata(timestamp_ms=33_000.0)],
                is_root=False,
                agent_depth=1,
                parent_conversation_id="trace_0",
            ),
            ConversationMetadata(
                conversation_id="trace_0::sa:a:fa:001",
                turns=[TurnMetadata(timestamp_ms=108_000.0)],
                is_root=False,
                agent_depth=1,
                parent_conversation_id="trace_0",
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    parent_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="parent",
        next_turn_index=2,
        agent_depth=0,
        waiting_on_children=True,
        join_target_turn_index=2,
    )
    # t* = 13_000: child A first request 20s out, child B 95s out.
    child_a = ConversationState(
        conversation_id="trace_0::sa:a:fa:000",
        x_correlation_id="kid-a",
        next_turn_index=0,
        next_dispatch_offset_ms=20_000.0,
        agent_depth=1,
        parent_correlation_id="parent",
        join_target_turn_index=2,
        branch_id="b0",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    child_b = ConversationState(
        conversation_id="trace_0::sa:a:fa:001",
        x_correlation_id="kid-b",
        next_turn_index=0,
        next_dispatch_offset_ms=95_000.0,
        agent_depth=1,
        parent_correlation_id="parent",
        join_target_turn_index=2,
        branch_id="b0",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=2,
        snapshot=TrajectorySnapshot(
            t_star_ms=13_000.0,
            states=(parent_state, child_a, child_b),
        ),
    )
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src.trajectories = [trajectory]

    issued: list[tuple[str, int]] = []

    async def capture(turn):
        issued.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    scheduled: list[tuple[float, object]] = []

    def fake_schedule_later(delay, coro):
        scheduled.append((delay, coro))

    scheduler = MagicMock()
    scheduler.schedule_later.side_effect = fake_schedule_later
    branch_orchestrator = MagicMock()

    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        branch_orchestrator=branch_orchestrator,
    )
    strategy._burst_phase_starts = True  # exercise the burst (T0-normalize) path

    await strategy.setup_phase()
    await strategy.execute_phase()

    # Earliest child (T0 anchor) fires immediately; parent gated (not sent).
    assert issued == [("trace_0::sa:a:fa:000", 0)]
    # Later child scheduled 75s in (95s - 20s recorded gap).
    assert [delay for delay, _ in scheduled] == [pytest.approx(75.0)]
    for _, coro in scheduled:
        await coro
    assert issued == [("trace_0::sa:a:fa:000", 0), ("trace_0::sa:a:fa:001", 0)]

    # All three states seeded; parent stays gated until both children drain.
    seeded_states = branch_orchestrator.seed_snapshot.call_args.args[0]
    assert [s.x_correlation_id for s in seeded_states] == ["parent", "kid-a", "kid-b"]
    assert seeded_states[0].waiting_on_children is True


@pytest.mark.asyncio
async def test_profiling_idle_trajectory_caps_leading_idle_preserving_subagent_spacing():
    """A trajectory idle at t* (every stream's first request far past t*) caps
    only the LEADING idle (t* -> earliest stream) and shifts every stream left
    by the same amount, preserving recorded subagent spacing.

    Regression for the per-stream ``min(offset, cap)`` clamp, which collapsed
    every idle subagent onto t=cap. Three children at offsets 100s/130s/220s
    with a 60s cap: leading idle 100s -> 60s (shift 40s), so they fire at
    60s/90s/180s -- the recorded 30s and 90s gaps survive, not 60s/60s/60s.
    """
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=12_000.0),
                    TurnMetadata(timestamp_ms=300_000.0),
                ],
            ),
            *(
                ConversationMetadata(
                    conversation_id=cid,
                    turns=[TurnMetadata(timestamp_ms=offset + 13_000.0)],
                    is_root=False,
                    agent_depth=1,
                    parent_conversation_id="trace_0",
                )
                for cid, offset in (
                    ("trace_0::sa:a:fa:000", 100_000.0),
                    ("trace_0::sa:a:fa:001", 130_000.0),
                    ("trace_0::sa:a:fa:002", 220_000.0),
                )
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    parent_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="parent",
        next_turn_index=2,
        agent_depth=0,
        waiting_on_children=True,
        join_target_turn_index=2,
    )
    children = [
        ConversationState(
            conversation_id=cid,
            x_correlation_id=xc,
            next_turn_index=0,
            next_dispatch_offset_ms=offset,
            agent_depth=1,
            parent_correlation_id="parent",
            join_target_turn_index=2,
            branch_id="b0",
            branch_mode=ConversationBranchMode.SPAWN,
        )
        for xc, cid, offset in (
            ("kid-a", "trace_0::sa:a:fa:000", 100_000.0),
            ("kid-b", "trace_0::sa:a:fa:001", 130_000.0),
            ("kid-c", "trace_0::sa:a:fa:002", 220_000.0),
        )
    ]
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=2,
        snapshot=TrajectorySnapshot(
            t_star_ms=13_000.0,
            states=(parent_state, *children),
        ),
    )
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src.trajectories = [trajectory]

    issued: list[str] = []

    async def capture(turn):
        issued.append(turn.conversation_id)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    scheduled: list[tuple[float, object]] = []
    scheduler = MagicMock()
    scheduler.schedule_later.side_effect = lambda delay, coro: scheduled.append(
        (delay, coro)
    )

    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        branch_orchestrator=MagicMock(),
    )
    strategy._phase_offset_cap_ms = 60_000.0  # agentx idle-gap cap of 60s

    await strategy.setup_phase()
    await strategy.execute_phase()

    # Leading idle (t* -> earliest child, 100s) capped to 60s via a 40s uniform
    # shift; subagents keep their recorded 30s and 90s spacing -- NOT collapsed
    # to a single 60s/60s/60s instant.
    assert issued == []
    assert [delay for delay, _ in scheduled] == [
        pytest.approx(60.0),
        pytest.approx(90.0),
        pytest.approx(180.0),
    ]
    for _, coro in scheduled:
        await coro
    assert issued == [
        "trace_0::sa:a:fa:000",
        "trace_0::sa:a:fa:001",
        "trace_0::sa:a:fa:002",
    ]


def test_profiling_spread_reports_first_request_per_trajectory_not_all_streams():
    """The logged PROFILING spread is the ramp-in window of each trajectory's
    FIRST request, not the full span of every stream. A late subagent within a
    trajectory must not inflate it.

    Two trajectories, each a root firing early plus a subagent ~83 min out. The
    first-request spread is 10s (root offsets 0 and 10s); the late subagents are
    excluded -- summing them would report ~8000s instead.
    """

    def _traj(cid: str, offsets: list[float]) -> Trajectory:
        states = tuple(
            ConversationState(
                conversation_id=f"{cid}:s{i}",
                x_correlation_id=f"{cid}-{i}",
                next_turn_index=0,
                next_dispatch_offset_ms=off,
                waiting_on_children=False,
            )
            for i, off in enumerate(offsets)
        )
        return Trajectory(
            conversation_id=cid,
            start_turn_index=0,
            snapshot=TrajectorySnapshot(t_star_ms=0.0, states=states),
        )

    src = TrajectorySource.__new__(TrajectorySource)
    src.trajectories = [
        _traj("t0", [0.0, 5_000_000.0]),
        _traj("t1", [10_000.0, 8_000_000.0]),
    ]
    strategy = AgenticReplayStrategy.__new__(AgenticReplayStrategy)
    strategy.conversation_source = src
    strategy._burst_phase_starts = False
    strategy._phase_offset_cap_ms = 60_000.0

    assert strategy._profiling_spread_seconds() == pytest.approx(10.0)

    # An idle-at-t* trajectory's first request is capped to the idle-gap cap,
    # so the spread stays bounded by the cap rather than the raw leading idle.
    src.trajectories.append(_traj("t2", [200_000.0, 9_000_000.0]))
    assert strategy._profiling_spread_seconds() == pytest.approx(60.0)


@pytest.mark.asyncio
async def test_profiling_preserve_start_gap_delays_first_request_by_default():
    """By default (spread), a trajectory's first post-t* request waits out its
    recorded offset from t* instead of firing at 0.

    A lone root resuming 8s after t* is scheduled 8s out -- the leading idle
    gap is preserved. (--burst-phase-starts would collapse it to fire at 0.)
    """
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=18_000.0),
                ],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # t* = 10_000: turn 0 < t* (warmed), turn 1 (18s) is 8s out from t*.
    root_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="root",
        next_turn_index=1,
        next_dispatch_offset_ms=8_000.0,
    )
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=1,
        snapshot=TrajectorySnapshot(
            t_star_ms=10_000.0,
            states=(root_state,),
        ),
    )
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src.trajectories = [trajectory]

    issued: list[tuple[str, int]] = []

    async def capture(turn):
        issued.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    scheduled: list[tuple[float, object]] = []
    scheduler = MagicMock()
    scheduler.schedule_later.side_effect = lambda delay, coro: scheduled.append(
        (delay, coro)
    )

    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    # Leading gap preserved: nothing fires inline; turn 1 scheduled 8s out.
    assert issued == []
    assert [delay for delay, _ in scheduled] == [pytest.approx(8.0)]
    for _, coro in scheduled:
        await coro
    assert issued == [("trace_0", 1)]


@pytest.mark.asyncio
async def test_profiling_gated_parent_not_dispatched_child_profiles():
    """A parent gated on a child join at t* is not dispatched in PROFILING;
    its join is seeded and the blocking child profiles its remaining turns.

    The parent's gated turn fires later via the orchestrator when the child
    drains (no child completes during WARMUP under the new model, so the
    parent stays gated through the warmup barrier). Covers both a mid-flight
    child (warmed at n-1, profiles from n) and the gate staying registered.
    """
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=12_000.0),
                    TurnMetadata(timestamp_ms=20_000.0),
                ],
            ),
            ConversationMetadata(
                conversation_id="trace_0::sa:0",
                turns=[
                    TurnMetadata(timestamp_ms=13_000.0),
                    TurnMetadata(timestamp_ms=14_000.0),
                ],
                is_root=False,
                agent_depth=1,
                parent_conversation_id="trace_0",
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    parent_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="parent",
        next_turn_index=2,
        waiting_on_children=True,
        join_target_turn_index=2,
    )
    child_state = ConversationState(
        conversation_id="trace_0::sa:0",
        x_correlation_id="child",
        next_turn_index=1,  # turn 0 < t*, turn 1 >= t*
        next_dispatch_offset_ms=500.0,
        agent_depth=1,
        parent_correlation_id="parent",
        join_target_turn_index=2,
        branch_id="b0",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=2,
        snapshot=TrajectorySnapshot(
            t_star_ms=13_500.0,
            states=(parent_state, child_state),
        ),
    )
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src.trajectories = [trajectory]

    issued: list[tuple[str, int]] = []

    async def capture(turn):
        issued.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    branch_orchestrator = MagicMock()

    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        branch_orchestrator=branch_orchestrator,
    )
    strategy._burst_phase_starts = True  # exercise the burst (T0-normalize) path

    await strategy.setup_phase()
    await strategy.execute_phase()

    # Only the child dispatches (its own next_turn_index = 1); parent gated.
    assert issued == [("trace_0::sa:0", 1)]
    seeded_states = branch_orchestrator.seed_snapshot.call_args.args[0]
    by_corr = {s.x_correlation_id: s for s in seeded_states}
    assert by_corr["parent"].waiting_on_children is True
    assert by_corr["parent"].join_target_turn_index == 2
    assert by_corr["child"].next_turn_index == 1


@pytest.mark.asyncio
async def test_profiling_single_turn_root_profiles_its_own_turn_zero():
    """A single-turn root sampled at t* == its turn-0 timestamp (n == 0) has
    nothing to warm, so PROFILING measures its own turn 0 rather than
    warming-and-discarding it and recycling.

    The dispatched credit carries the snapshot's own x_correlation_id (the
    session continues from the snapshot, not a fresh recycle); recycling
    happens later on the turn's completion via handle_credit_return.
    """
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[TurnMetadata(timestamp_ms=0.0)],
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=0,
        snapshot=TrajectorySnapshot(
            t_star_ms=0.0,
            states=(
                ConversationState(
                    conversation_id="trace_0",
                    x_correlation_id="snap-root",
                    next_turn_index=0,
                ),
            ),
        ),
    )
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src.trajectories = [trajectory]
    issued = []

    async def capture(turn):
        issued.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    assert len(issued) == 1
    assert issued[0].conversation_id == "trace_0"
    assert issued[0].turn_index == 0
    assert issued[0].x_correlation_id == "snap-root"


@pytest.mark.asyncio
async def test_single_turn_root_snapshot_dispatches_are_concurrent_not_serial():
    """Regression: profiling startup must burst at t=0, not serialize.

    Commit f47bd5537e once introduced an awaited per-trajectory step in the
    startup dispatch loop; with N trajectories that blocked the Kth dispatch
    until the (K-1)th completed, trickling 256 sessions in over ~54 s instead
    of bursting at t=0 on a real cluster. The per-lane gather must keep all N
    first dispatches concurrent.
    """
    N = 3
    # N single-turn traces sampled at t* == turn-0 ts (n == 0): each profiles
    # its own turn 0 immediately (nothing to warm), so all N first dispatches
    # should reach the issuer concurrently via the per-lane gather.
    trajectories = [
        Trajectory(
            conversation_id=f"trace_{i}",
            start_turn_index=0,
            snapshot=TrajectorySnapshot(
                t_star_ms=0.0,
                states=(
                    ConversationState(
                        conversation_id=f"trace_{i}",
                        x_correlation_id=f"warmed-{i}",
                        next_turn_index=0,
                    ),
                ),
            ),
        )
        for i in range(N)
    ]

    # Gate that blocks each recycled credit until we release it. in_flight
    # counts issue_credit calls simultaneously blocked at the gate - if
    # serial only 1 is ever in-flight, if concurrent all N are.
    gate = asyncio.Event()
    in_flight = 0
    captured: list = []

    async def gated_issue_credit(turn):
        nonlocal in_flight
        in_flight += 1
        captured.append(turn)
        await gate.wait()
        in_flight -= 1
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = gated_issue_credit
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=N,
        turns_per_trace=1,
        issuer=issuer,
    )

    await strategy.setup_phase()
    task = asyncio.create_task(strategy.execute_phase())

    # Pump the event loop enough times for all concurrent dispatches to reach
    # the gate. With the serial bug only 1 will be in-flight; with the fix
    # all N will be blocked at gate.wait() simultaneously.
    for _ in range(N + 4):
        await asyncio.sleep(0)

    concurrent_at_gate = in_flight
    gate.set()
    await asyncio.wait_for(task, timeout=5.0)

    assert concurrent_at_gate == N, (
        f"Expected {N} recycled credits in-flight simultaneously at profiling "
        f"startup, but only {concurrent_at_gate} were. Terminal-root recycles "
        "appear to be dispatched serially, blocking the startup loop."
    )
    # Exactly one recycled turn-0 session per lane, each a distinct session -
    # N-at-the-gate must not be satisfiable by the wrong N credits.
    assert issuer.issue_credit.await_count == N
    assert all(turn.turn_index == 0 for turn in captured)
    assert len({turn.x_correlation_id for turn in captured}) == N


@pytest.mark.asyncio
async def test_plain_trajectory_resumes_are_concurrent_not_serial():
    """The k_i+1 resume path (timestamp-less trajectories) must burst at t=0.

    Companion to the terminal-root regression test: a refactor that
    re-serializes only the snapshot-less resume dispatch would otherwise
    ship green.
    """
    N = 3
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(N)
    ]
    gate = asyncio.Event()
    in_flight = 0

    async def gated_issue_credit(turn):
        nonlocal in_flight
        in_flight += 1
        await gate.wait()
        in_flight -= 1
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = gated_issue_credit
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=N,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    task = asyncio.create_task(strategy.execute_phase())
    for _ in range(N + 4):
        await asyncio.sleep(0)

    concurrent_at_gate = in_flight
    gate.set()
    await asyncio.wait_for(task, timeout=5.0)

    assert concurrent_at_gate == N, (
        f"Expected {N} resume credits in-flight simultaneously at profiling "
        f"startup, but only {concurrent_at_gate} were."
    )


@pytest.mark.asyncio
async def test_continuing_session_keeps_warmup_marker_across_phase_boundary():
    """A continued session's cache-bust marker must not rotate at the boundary.

    Under wrap-fill, two lanes share trace_X and both are mid-flight at t*:
    lane 0's root is gated on a child join, lane 1's root is ready. Both are
    warmed now (gated parents included), each minting its own marker keyed by
    x_correlation_id. Positional re-minting (by dispatch order/count rather
    than identity) would hand the continuing session the wrong digest and make
    its warmed KV prefix unreachable for the measured turns. The marker minted
    in WARMUP must be reused verbatim for the same x_correlation_id in
    PROFILING, and distinct sessions must get distinct markers.
    """
    ds = _make_dataset(num_traces=1, turns_per_trace=3)
    trajectories = [
        Trajectory(
            conversation_id="trace_0",
            start_turn_index=0,
            snapshot=TrajectorySnapshot(
                t_star_ms=0.0,
                states=(
                    ConversationState(
                        conversation_id="trace_0",
                        x_correlation_id="A-root",
                        next_turn_index=2,
                        waiting_on_children=True,
                        join_target_turn_index=2,
                    ),
                ),
            ),
        ),
        Trajectory(
            conversation_id="trace_0",
            start_turn_index=1,
            snapshot=TrajectorySnapshot(
                t_star_ms=0.0,
                states=(
                    ConversationState(
                        conversation_id="trace_0",
                        x_correlation_id="B-root",
                        # Mid-flight at t*: warmed at turn 0, profiles turn 1
                        # with the same marker (continuity across the boundary).
                        next_turn_index=1,
                    ),
                ),
            ),
        ),
    ]
    src = _build_real_trajectory_source(1, 3, trajectories, dataset=ds)
    user_config = MagicMock()
    user_config.input.prompt.cache_bust.target = CacheBustTarget.FIRST_TURN_PREFIX
    user_config.benchmark_id = "bench"

    def _strategy_for(phase: CreditPhase, issuer: AsyncMock) -> AgenticReplayStrategy:
        cfg = MagicMock()
        cfg.phase = phase
        return AgenticReplayStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=issuer,
            lifecycle=MagicMock(),
            user_config=user_config,
        )

    warmup_issuer = AsyncMock()
    warmup = _strategy_for(CreditPhase.WARMUP, warmup_issuer)
    await warmup.setup_phase()
    await warmup.execute_phase()
    warmup_turns = [c.args[0] for c in warmup_issuer.issue_credit.await_args_list]
    warmup_marker = warmup._session_marker["B-root"]
    assert warmup_marker is not None
    # Both lanes warmed (gated A-root at its turn n-1, ready B-root at turn 0),
    # each carrying its own marker keyed by x_correlation_id.
    by_corr = {t.x_correlation_id: t for t in warmup_turns}
    assert set(by_corr) == {"A-root", "B-root"}
    assert by_corr["B-root"].cache_bust_marker == warmup_marker
    assert by_corr["A-root"].cache_bust_marker != warmup_marker

    profiling_issuer = AsyncMock()
    profiling = _strategy_for(CreditPhase.PROFILING, profiling_issuer)
    await profiling.setup_phase()
    await profiling.execute_phase()
    profiling_turns = {
        t.conversation_id: t
        for c in profiling_issuer.issue_credit.await_args_list
        for t in [c.args[0]]
        if t.turn_index == 1
    }
    continuing = profiling_turns["trace_0"]
    assert continuing.cache_bust_marker == warmup_marker, (
        "Continuing session's marker rotated at the WARMUP->PROFILING "
        "boundary - the warmed KV prefix is unreachable for measured turns"
    )
    # The unblocked lane-0 parent is a distinct session and must NOT share
    # the continuing session's digest.
    assert profiling._session_marker["A-root"] != warmup_marker


@pytest.mark.asyncio
async def test_startup_dispatch_snapshot_root_and_resume():
    """PROFILING execute dispatches each lane's initial credit: a snapshot
    root at turn 0 and a plain trajectory resumed at k_i+1. No recycle fires
    during execute_phase (recycle is driven by credit returns)."""
    conversations = [
        ConversationMetadata(
            conversation_id="trace_0",
            turns=[TurnMetadata(), TurnMetadata()],
        ),
        ConversationMetadata(
            conversation_id="trace_1",
            turns=[TurnMetadata()],
        ),
    ]
    ds = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    trajectories = [
        Trajectory(
            conversation_id="trace_1",
            start_turn_index=0,
            snapshot=TrajectorySnapshot(
                t_star_ms=0.0,
                states=(
                    ConversationState(
                        conversation_id="trace_1",
                        x_correlation_id="warmed-1",
                        next_turn_index=0,
                    ),
                ),
            ),
        ),
        Trajectory(conversation_id="trace_0", start_turn_index=0),
    ]
    captured: list[tuple[str, int]] = []

    async def capture(turn):
        captured.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        issuer=issuer,
        dataset=ds,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    # Lane 0 dispatches its snapshot root trace_1 at turn 0; lane 1 resumes
    # trace_0 at k_i+1. No recycle during execute_phase.
    assert sorted(captured) == [("trace_0", 1), ("trace_1", 0)]


@pytest.mark.asyncio
async def test_profiling_dispatch_error_waits_for_siblings_and_reraises():
    """One lane's dispatch failure must not detach the sibling dispatches.

    execute_phase must keep ownership of every sibling lane until it settles,
    then re-raise the failure. A bare gather would return the exception while
    the remaining lanes keep issuing credits into a failing phase unsupervised.
    """
    N = 3
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(N)
    ]
    gate = asyncio.Event()
    completed: list[str] = []

    async def gated_issue_credit(turn):
        if turn.conversation_id == "trace_1":
            raise RuntimeError("lane boom")
        await gate.wait()
        completed.append(turn.conversation_id)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = gated_issue_credit
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=N,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    task = asyncio.create_task(strategy.execute_phase())
    for _ in range(N + 4):
        await asyncio.sleep(0)

    # Lane 1 has already raised, but lanes 0 and 2 are still blocked at the
    # gate: phase execution must still be in flight rather than finished
    # with an exception while its siblings run detached.
    assert not task.done(), (
        "execute_phase finished while sibling dispatches were still in "
        "flight - one lane's failure detached the remaining lanes"
    )

    gate.set()
    with pytest.raises(RuntimeError, match="lane boom"):
        await asyncio.wait_for(task, timeout=5.0)
    assert sorted(completed) == ["trace_0", "trace_2"]


@pytest.mark.asyncio
async def test_profiling_skips_trajectory_at_last_turn_and_recycles():
    """If k_i is already the last turn, k_i+1 is out of range. Recycle immediately."""
    trajectories = [
        Trajectory(
            conversation_id="trace_0", start_turn_index=3
        ),  # turns_per_trace=4 -> last index
    ]
    captured: list[tuple[str, int]] = []

    async def capture(turn):
        captured.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=3,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    # No resume issued for the trajectory; instead a recycle session at turn 0.
    assert all(idx == 0 for _, idx in captured)
    assert len(captured) == 1
    # With the full-pool recycle queue, the head is "trace_0" (iteration
    # order from dataset_metadata.conversations). The trajectory's session
    # is discarded from _active_traces inside _spawn_from_recycle_or_id
    # before the pop loop runs, so trace_0 is popped and dispatched at
    # turn 0 as the first recycled session.
    assert captured[0][0] == "trace_0"


# =============================================================================
# PROFILING handle_credit_return: continuation + recycle
# =============================================================================


@pytest.mark.asyncio
async def test_handle_credit_return_dispatches_next_turn_when_not_final():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    credit = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=4)
    await strategy.handle_credit_return(credit)

    assert issuer.issue_credit.await_count == 1
    issued_turn = issuer.issue_credit.await_args.args[0]
    assert issued_turn.turn_index == 2
    assert issued_turn.conversation_id == "trace_0"


@pytest.mark.asyncio
async def test_handle_credit_return_honors_delay_ms_via_scheduler():
    """When next turn has delay_ms, dispatch is scheduled, not immediate."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    scheduler = MagicMock()

    # Build a dataset where turn_index=2 has a delay_ms.
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                    TurnMetadata(timestamp_ms=None, delay_ms=500),
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                ],
            )
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src._random_seed = 0
    src._target_size = 1
    src.trajectories = list(trajectories)

    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()
    scheduler.schedule_later.reset_mock()

    credit = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=4)
    await strategy.handle_credit_return(credit)

    # No direct issue; one scheduled dispatch with delay 0.5s.
    assert issuer.issue_credit.await_count == 0
    assert scheduler.schedule_later.call_count == 1
    delay_arg = scheduler.schedule_later.call_args.args[0]
    assert delay_arg == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_handle_credit_return_recycles_on_final_turn():
    """Last turn of a session -> lane recycles into the next root from the
    sampler (sequential over trace_0..trace_2 -> trace_0 first), at turn 0."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issued_sessions: list[tuple[str, int]] = []

    async def capture(turn):
        issued_sessions.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=3,
        turns_per_trace=4,
        issuer=issuer,
    )
    strategy.stop_checker.can_start_new_session.return_value = True
    await strategy.setup_phase()

    # Register the in-flight session's lane bookkeeping (normally done by
    # _execute_profiling); the recycle path requires finished_correlation_id
    # to be in _correlation_to_lane.
    strategy._correlation_to_lane["xcorr"] = 0

    issuer.issue_credit.reset_mock()
    issued_sessions.clear()

    # trace_0 finishes its last turn (index 3 of 4) -> recycle into the next
    # sampler root at turn 0.
    final_credit = _make_credit(conversation_id="trace_0", turn_index=3, num_turns=4)
    await strategy.handle_credit_return(final_credit)

    assert issued_sessions == [("trace_0", 0)]


@pytest.mark.asyncio
async def test_handle_credit_return_does_not_recycle_final_child_turn():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=2,
        turns_per_trace=3,
        issuer=issuer,
    )
    await strategy.setup_phase()
    strategy._correlation_to_lane["child-corr"] = 0
    strategy._session_marker["child-corr"] = None
    issuer.issue_credit.reset_mock()

    final_child_credit = _make_credit(
        conversation_id="trace_0::sa:0",
        x_correlation_id="child-corr",
        turn_index=1,
        num_turns=2,
        agent_depth=1,
        parent_correlation_id="parent-corr",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    await strategy.handle_credit_return(final_child_credit)

    assert issuer.issue_credit.await_count == 0
    assert "child-corr" not in strategy._correlation_to_lane
    assert "child-corr" not in strategy._session_marker


@pytest.mark.asyncio
async def test_handle_credit_return_reuses_sole_root_when_pool_is_one():
    """Single-root dataset: the sampler only ever yields trace_0, so recycle
    reuses it (at turn 0)."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issued_sessions: list[tuple[str, int]] = []

    async def capture(turn):
        issued_sessions.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=1,  # single-trace dataset
        turns_per_trace=3,
        issuer=issuer,
    )
    strategy.stop_checker.can_start_new_session.return_value = True
    await strategy.setup_phase()

    # Register the in-flight session's lane (normally done by _execute_profiling).
    strategy._correlation_to_lane["xcorr"] = 0

    issuer.issue_credit.reset_mock()
    issued_sessions.clear()

    final_credit = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=3)
    await strategy.handle_credit_return(final_credit)

    assert issued_sessions == [("trace_0", 0)]


@pytest.mark.asyncio
async def test_spawn_from_recycle_prunes_marker_dicts_on_stop_checker_reject():
    """Early-return paths in _spawn_from_recycle_or_id must still prune marker/lane dicts.

    Regression: previously the pop only happened on the success path, so a finished
    session whose recycle attempt hit any early return (stop-checker reject, queue
    empty without a put because _recycle_queue is None, missing metadata) would
    leak its entry into _session_marker and _correlation_to_lane for the rest of
    the phase.
    """
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=2,
        turns_per_trace=3,
        issuer=issuer,
    )
    await strategy.setup_phase()

    # Simulate an in-flight session for trace_0: lane assigned, marker minted.
    finished_corr_id = "xcorr-finished"
    strategy._correlation_to_lane[finished_corr_id] = 0
    strategy._session_marker[finished_corr_id] = None

    # Force the stop_checker early-return path - cooldown reached, no new sessions.
    strategy.stop_checker.can_start_new_session = MagicMock(return_value=False)

    issuer.issue_credit.reset_mock()
    await strategy._spawn_from_recycle_or_id(
        "trace_0", finished_correlation_id=finished_corr_id
    )

    # Early return must still have pruned both bookkeeping dicts.
    assert finished_corr_id not in strategy._session_marker
    assert finished_corr_id not in strategy._correlation_to_lane
    # No new credit issued because of the early-return.
    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_handle_credit_return_warmup_phase_is_noop_for_final_turn():
    """In WARMUP, even a final-turn credit return must not trigger recycle."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=2,
        issuer=issuer,
    )
    await strategy.setup_phase()

    issuer.issue_credit.reset_mock()
    final_credit = _make_credit(
        conversation_id="trace_0",
        turn_index=1,
        num_turns=2,
        phase=CreditPhase.WARMUP,
    )
    await strategy.handle_credit_return(final_credit)
    assert issuer.issue_credit.await_count == 0


# =============================================================================
# Warmup signals sending-complete after dispatch
# =============================================================================
#
# Belt-and-suspenders alongside total_expected_requests=loadgen.concurrency:
# _execute_warmup must call lifecycle.mark_sending_complete() AFTER the
# cohort dispatch loop. Without it, when pool_size < concurrency the count
# target is never reached and the cohort barrier holds forever. Must be
# called exactly once, after all credits are issued.


@pytest.mark.asyncio
async def test_warmup_marks_sending_complete():
    """``_execute_warmup`` signals sending-complete once after dispatching
    all trajectory credits.

    ``mark_sending_complete`` is a guarded fallback now that PhaseRunner re-anchors
    ``total_expected_requests`` to the actual trajectory count: when the count-based
    path wins the race, the strategy's call is skipped via the
    ``is_sending_complete`` guard. Force the guard to evaluate ``False`` so this
    legacy behavioral assertion still applies.
    """
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories
    )
    # The strategy's belt-and-suspenders mark only fires in burst mode; in the
    # default spread mode the count path / runner finalize sending instead.
    strategy._burst_phase_starts = True
    strategy.lifecycle.is_sending_complete = False
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert strategy.lifecycle.mark_sending_complete.call_count == 1
    # Sanity: dispatch happened for each trajectory.
    assert issuer.issue_credit.await_count == 3


@pytest.mark.asyncio
async def test_warmup_marks_sending_complete_after_dispatch():
    """``mark_sending_complete`` is called AFTER all credits are issued,
    not before — otherwise ``SendingCompleteStopCondition`` can fire
    mid-dispatch."""
    call_order: list[str] = []

    async def record_issue(_turn) -> bool:
        call_order.append("issue_credit")
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = record_issue

    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories, issuer=issuer
    )
    # Belt-and-suspenders mark only fires in burst mode (see sibling test).
    strategy._burst_phase_starts = True
    strategy.lifecycle.is_sending_complete = False

    def record_mark() -> None:
        call_order.append("mark_sending_complete")

    strategy.lifecycle.mark_sending_complete.side_effect = record_mark

    await strategy.setup_phase()
    await strategy.execute_phase()

    assert call_order == [
        "issue_credit",
        "issue_credit",
        "issue_credit",
        "mark_sending_complete",
    ]


@pytest.mark.asyncio
async def test_warmup_skips_mark_sending_complete_when_already_complete():
    """When ``CreditCounter.is_final_credit`` already fired (and PhaseRunner's
    ``CreditIssuer`` already advanced the lifecycle into SENDING_COMPLETE),
    the strategy must NOT re-call ``mark_sending_complete``. Without this
    guard the strategy double-transitions the state machine -> ValueError.

    This is the regression guard for the warmup-hang fix: PhaseRunner now
    re-anchors ``total_expected_requests`` to the actual trajectory count,
    so the count-based path is the primary signal and the strategy's call
    becomes a guarded fallback.
    """
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectories
    )
    # Burst mode is where the strategy would otherwise call mark; the guard
    # must still suppress it when the count path already completed sending.
    strategy._burst_phase_starts = True
    # Simulate the count-based path having already won the race.
    strategy.lifecycle.is_sending_complete = True

    await strategy.setup_phase()
    await strategy.execute_phase()

    strategy.lifecycle.mark_sending_complete.assert_not_called()


# =============================================================================
# Cache-bust marker minting (Task 5)
# =============================================================================
#
# Per spec §4.5, AgenticReplayStrategy mints one marker per session keyed by
# x_correlation_id, reuses it across the warmup k_i / profile k_i+1 boundary,
# and rotates it on recycle (recycle_pass increments). Lane (trajectory_index)
# is stable per slot so marker digests change only across recycle passes.

_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


def _make_user_config(
    *, target: CacheBustTarget, benchmark_id: str = "bench-fixed"
) -> SimpleNamespace:
    """Lightweight stand-in for UserConfig; only the two attributes the
    strategy reads are exposed (avoids spinning up real Pydantic config)."""
    return SimpleNamespace(
        input=SimpleNamespace(
            prompt=SimpleNamespace(cache_bust=SimpleNamespace(target=target))
        ),
        benchmark_id=benchmark_id,
    )


def _extract_rid(marker: str | None) -> str | None:
    if marker is None:
        return None
    m = _RID_RE.search(marker)
    return m.group(0) if m else None


@pytest.mark.asyncio
async def test_warmup_session_marker_reused_in_profile_resume():
    """Trajectory's warmup turn k_i and profile turn k_i+1 share the same
    marker (recycle_pass=0, same lane index, same benchmark_id, same
    trace_id; phase deliberately NOT in the digest tuple per spec
    warmup-coherence requirement)."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=2)]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)

    # WARMUP phase mints first.
    issuer = AsyncMock()
    warmup_turns: list = []

    async def capture_warmup(turn):
        warmup_turns.append(turn)
        return True

    issuer.issue_credit.side_effect = capture_warmup

    strategy_w, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=5,
        issuer=issuer,
        user_config=user_config,
    )
    await strategy_w.setup_phase()
    await strategy_w.execute_phase()

    # PROFILING phase (constructed fresh like PhaseRunner does).
    issuer2 = AsyncMock()
    profile_turns: list = []

    async def capture_profile(turn):
        profile_turns.append(turn)
        return True

    issuer2.issue_credit.side_effect = capture_profile

    strategy_p, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        turns_per_trace=5,
        issuer=issuer2,
        user_config=user_config,
    )
    await strategy_p.setup_phase()
    await strategy_p.execute_phase()

    assert len(warmup_turns) == 1
    assert len(profile_turns) == 1
    warmup_rid = _extract_rid(warmup_turns[0].cache_bust_marker)
    profile_rid = _extract_rid(profile_turns[0].cache_bust_marker)
    assert warmup_rid is not None
    assert warmup_rid == profile_rid, (
        "Spec requires warmup-coherence: the digest tuple "
        "(benchmark_id, recycle_pass, trajectory_index, trace_id) is "
        "phase-agnostic, so WARMUP turn k_i and PROFILING turn k_i+1 must "
        "render the same marker so warmup KV-cache work transfers to profile."
    )
    assert warmup_turns[0].cache_bust_target == CacheBustTarget.SYSTEM_PREFIX
    assert profile_turns[0].cache_bust_target == CacheBustTarget.SYSTEM_PREFIX


@pytest.mark.asyncio
async def test_recycle_increments_pass_and_rotates_marker():
    """Spawn for traceA, finish, recycle traceA — markers differ because
    recycle_pass increments. Single-trace dataset so the just-finished
    trace_id is reused immediately on recycle."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)
    issued_turns: list = []

    async def capture(turn):
        issued_turns.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=1,  # forces queue empty -> reuse on recycle
        turns_per_trace=2,
        issuer=issuer,
        user_config=user_config,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert len(issued_turns) == 1
    initial_marker = issued_turns[0].cache_bust_marker
    initial_rid = _extract_rid(initial_marker)
    initial_xcorr = issued_turns[0].x_correlation_id

    # Final-turn credit return triggers recycle of trace_0.
    final_credit = _make_credit(
        conversation_id="trace_0",
        x_correlation_id=initial_xcorr,
        turn_index=1,
        num_turns=2,
    )
    await strategy.handle_credit_return(final_credit)

    assert len(issued_turns) == 2
    recycled_rid = _extract_rid(issued_turns[1].cache_bust_marker)
    assert recycled_rid is not None
    assert recycled_rid != initial_rid


@pytest.mark.asyncio
async def test_two_trajectories_same_starting_trace_get_distinct_markers():
    """Two trajectories at lane 0 and lane 1 mint different markers because
    trajectory_index differs. (TrajectorySource itself rejects duplicate
    trace_ids in trajectories, so we model 'same trace' as recycle reuse:
    trajectory[0] starts on trace_x; later trajectory[1]'s recycle pulls
    trace_x. We assert markers differ via the lane component instead.)"""
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)
    issued_turns: list = []

    async def capture(turn):
        issued_turns.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=3,
        issuer=issuer,
        user_config=user_config,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert len(issued_turns) == 2
    rid0 = _extract_rid(issued_turns[0].cache_bust_marker)
    rid1 = _extract_rid(issued_turns[1].cache_bust_marker)
    assert rid0 is not None
    assert rid1 is not None
    # Different lane (trajectory_index) -> different digest, even with same
    # benchmark_id and same recycle_pass=0.
    assert rid0 != rid1


@pytest.mark.asyncio
async def test_target_none_emits_no_marker():
    """With target=NONE (or no user_config plumbed), cache_bust_marker is
    None and cache_bust_target is NONE on every issued turn."""
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(2)
    ]
    user_config = _make_user_config(target=CacheBustTarget.NONE)
    issued_turns: list = []

    async def capture(turn):
        issued_turns.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        turns_per_trace=3,
        issuer=issuer,
        user_config=user_config,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert len(issued_turns) == 2
    for turn in issued_turns:
        assert turn.cache_bust_marker is None
        assert turn.cache_bust_target == CacheBustTarget.NONE


@pytest.mark.asyncio
async def test_two_traces_at_same_pass_and_lane_get_distinct_markers():
    """Two different trace_ids landing on the same (recycle_pass, lane) tuple
    must mint distinct markers. Regression bar for the collision-free fix:
    the marker tuple now includes ``trace_id`` so cross-trace collisions on
    the same (pass, lane) are eliminated by construction.

    Setup: single-lane (concurrency=1) PROFILING run starting on trace_A.
    When trace_A finishes its only profile turn, the empty recycle queue
    forces FIFO reuse — but we seed a second trajectory by directly
    inspecting the strategy's marker state via the per-session minting path.
    Cleaner: drive two sessions on lane 0 explicitly via the mint helper and
    assert the digests differ. ``recycle_pass`` is per-trace_id so both
    start at 0; ``trajectory_index`` is fixed at 0; only trace_id differs.
    """
    trajectories = [Trajectory(conversation_id="trace_A", start_turn_index=0)]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=2,
        turns_per_trace=2,
        user_config=user_config,
    )
    # Mint markers for two distinct trace_ids both at lane 0, both at
    # recycle_pass=0 (their first incarnation).
    marker_a = strategy._mint_marker_for_session(
        root_correlation_id="xcorr_a", conversation_id="trace_A", trajectory_index=0
    )
    marker_b = strategy._mint_marker_for_session(
        root_correlation_id="xcorr_b", conversation_id="trace_B", trajectory_index=0
    )
    rid_a = _extract_rid(marker_a)
    rid_b = _extract_rid(marker_b)
    assert rid_a is not None
    assert rid_b is not None
    assert rid_a != rid_b, (
        "Two distinct traces at the same (recycle_pass=0, lane=0) must "
        "produce distinct markers — collision-free uniqueness depends on "
        "trace_id being part of the digest tuple."
    )


# =============================================================================
# Signature lock: _spawn_from_recycle_or_id requires finished_correlation_id
# =============================================================================


def test_spawn_from_recycle_or_id_requires_finished_correlation_id() -> None:
    """``finished_correlation_id`` must be a required keyword-only parameter
    so the lane bookkeeping pop has a valid key on every code path."""
    import inspect

    sig = inspect.signature(AgenticReplayStrategy._spawn_from_recycle_or_id)
    param = sig.parameters["finished_correlation_id"]
    assert param.default is inspect.Parameter.empty
    assert param.kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.asyncio
async def test_spawn_from_recycle_or_id_pops_lane_and_marker_for_correlation() -> None:
    """The finished session's lane and marker entries are popped from the
    bookkeeping dicts so memory stays bounded by live concurrency."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    user_config = _make_user_config(target=CacheBustTarget.SYSTEM_PREFIX)
    strategy, *_ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=2,
        user_config=user_config,
    )

    correlation_id = "xcorr-finished"
    strategy._correlation_to_lane[correlation_id] = 7
    strategy._session_marker[correlation_id] = "[rid:abc123]"

    # Force early-return after the unconditional cleanup pop, isolating the
    # bookkeeping behavior from the spawn path.
    strategy.stop_checker.can_start_new_session = MagicMock(return_value=False)
    strategy._recycle_queue = asyncio.Queue()

    await strategy._spawn_from_recycle_or_id(
        "trace_0", finished_correlation_id=correlation_id
    )

    assert correlation_id not in strategy._correlation_to_lane
    assert correlation_id not in strategy._session_marker


# =============================================================================
# PROFILING phase: rootless lanes (root finished before t*) hold a lane credit
# =============================================================================


def _children_dataset() -> DatasetMetadata:
    """Dataset with one root trace and two background subagent conversations."""
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=1000.0),
                ],
                is_root=True,
            ),
            ConversationMetadata(
                conversation_id="trace_0::fa:0",
                turns=[TurnMetadata(timestamp_ms=2000.0)],
                is_root=False,
                agent_depth=1,
            ),
            ConversationMetadata(
                conversation_id="trace_0::aux:0",
                turns=[TurnMetadata(timestamp_ms=2000.0)],
                is_root=False,
                agent_depth=1,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _rootless_trajectory() -> Trajectory:
    """A snapshot whose root finished before t*: NO root state, only the
    still-active background ::fa:/::aux: children remain (the rootless case
    from ``_snapshot_for`` when ``root_next_idx is None``)."""
    child_a = ConversationState(
        conversation_id="trace_0::fa:0",
        x_correlation_id="kid_a",
        next_turn_index=0,
        next_dispatch_offset_ms=0.0,
        agent_depth=1,
        parent_correlation_id="root_corr",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    child_b = ConversationState(
        conversation_id="trace_0::aux:0",
        x_correlation_id="kid_b",
        next_turn_index=0,
        next_dispatch_offset_ms=0.0,
        agent_depth=1,
        parent_correlation_id="root_corr",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    return Trajectory(
        conversation_id="trace_0",
        start_turn_index=0,  # sentinel default; no root state in the snapshot
        snapshot=TrajectorySnapshot(t_star_ms=5000.0, states=(child_a, child_b)),
    )


@pytest.mark.asyncio
async def test_rootless_snapshot_acquires_exactly_one_lane_credit():
    """A rootless lane holds one session credit though it dispatches no root.

    The root's turns are all before t*, so PROFILING dispatches only the
    background children. The lane must still acquire exactly one session slot
    (so it counts toward --concurrency); the children acquire none of their
    own (they are agent_depth > 0).
    """
    dispatched: list = []

    async def capture(turn):
        dispatched.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    issuer.acquire_lane_credit = AsyncMock(return_value=True)

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[_rootless_trajectory()],
        issuer=issuer,
        dataset=_children_dataset(),
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert issuer.acquire_lane_credit.await_count == 1
    # Both background children were dispatched, none acquiring its own slot.
    assert {t.conversation_id for t in dispatched} == {
        "trace_0::fa:0",
        "trace_0::aux:0",
    }
    assert all(t.agent_depth == 1 for t in dispatched)


@pytest.mark.asyncio
async def test_rooted_snapshot_acquires_no_lane_credit():
    """A normal rooted lane gets its slot via the root credit, not a lane credit."""
    root_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="root_corr",
        next_turn_index=1,
        next_dispatch_offset_ms=0.0,
        agent_depth=0,
    )
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=1,
        snapshot=TrajectorySnapshot(t_star_ms=5000.0, states=(root_state,)),
    )

    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    issuer.acquire_lane_credit = AsyncMock(return_value=True)

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[trajectory],
        issuer=issuer,
        dataset=_children_dataset(),
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert issuer.acquire_lane_credit.await_count == 0


@pytest.mark.asyncio
async def test_rootless_lane_recycles_into_fresh_root_when_children_drain():
    """When a rootless lane's last background child finishes, the lane releases
    its credit and recycles into a fresh root (turn 0) on the same lane.

    Until the last child drains, the lane credit is held (the lane is still
    doing background work). On the final child's terminal return the credit is
    released exactly once and one fresh depth-0 root is dispatched from the
    recycle pool, so the lane keeps contributing load instead of going dark.
    """
    issued: list = []

    async def capture(turn):
        issued.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    issuer.acquire_lane_credit = AsyncMock(return_value=True)
    issuer.release_lane_credit = MagicMock()

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[_rootless_trajectory()],
        issuer=issuer,
        dataset=_children_dataset(),
    )
    strategy.stop_checker.can_start_new_session.return_value = True
    await strategy.setup_phase()
    await strategy.execute_phase()
    n_after_dispatch = len(issued)
    assert issuer.acquire_lane_credit.await_count == 1

    # First of two background children finishes: lane credit still held.
    await strategy.handle_credit_return(
        _make_credit(
            conversation_id="trace_0::fa:0",
            x_correlation_id="kid_a",
            turn_index=0,
            num_turns=1,
            agent_depth=1,
            parent_correlation_id="root_corr",
        )
    )
    assert issuer.release_lane_credit.call_count == 0
    assert len(issued) == n_after_dispatch  # no recycle dispatch yet

    # Last child finishes: release the lane credit and spawn a fresh root.
    await strategy.handle_credit_return(
        _make_credit(
            conversation_id="trace_0::aux:0",
            x_correlation_id="kid_b",
            turn_index=0,
            num_turns=1,
            agent_depth=1,
            parent_correlation_id="root_corr",
        )
    )
    assert issuer.release_lane_credit.call_count == 1
    new = issued[n_after_dispatch:]
    assert len(new) == 1, f"expected one fresh root dispatch, got {new}"
    assert new[0].turn_index == 0
    assert new[0].agent_depth == 0
    assert new[0].conversation_id == "trace_0"


@pytest.mark.asyncio
async def test_gated_parent_lane_acquires_a_lane_credit():
    """A snapshot-resumed gated parent (waiting on a child join) holds a lane
    credit, because it dispatches no root credit at PROFILING start.

    The gated root is excluded from the dispatchable set, so without this it
    holds NO session slot -- yet its join turn later completes and the depth-0
    final-turn path releases a slot, over-releasing the session limiter and
    admitting sessions above --concurrency. It must NOT be tracked as rootless:
    the parent resumes and recycles through the normal depth-0 final-turn path,
    so a child draining must not release/recycle the lane here.
    """
    root_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="root_corr",
        next_turn_index=2,
        agent_depth=0,
        waiting_on_children=True,
        join_target_turn_index=2,
    )
    child = ConversationState(
        conversation_id="trace_0::sa:a",
        x_correlation_id="kid",
        next_turn_index=0,
        next_dispatch_offset_ms=0.0,
        agent_depth=1,
        parent_correlation_id="root_corr",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=2,
        snapshot=TrajectorySnapshot(t_star_ms=5000.0, states=(root_state, child)),
    )
    dataset = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=1000.0),
                    TurnMetadata(timestamp_ms=2000.0),
                ],
                is_root=True,
            ),
            ConversationMetadata(
                conversation_id="trace_0::sa:a",
                turns=[TurnMetadata(timestamp_ms=2000.0)],
                is_root=False,
                agent_depth=1,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )

    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    issuer.acquire_lane_credit = AsyncMock(return_value=True)

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[trajectory],
        issuer=issuer,
        dataset=dataset,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    assert issuer.acquire_lane_credit.await_count == 1
    assert 0 not in strategy._rootless_lane_outstanding


@pytest.mark.asyncio
async def test_profiling_setup_logs_rootless_lane_count(caplog):
    """PROFILING setup surfaces how many sampled lanes are rootless (root
    finished before t*) so an under-target run is diagnosable from the log."""
    rooted_state = ConversationState(
        conversation_id="trace_1",
        x_correlation_id="r1",
        next_turn_index=1,
        agent_depth=0,
    )
    rooted = Trajectory(
        conversation_id="trace_1",
        start_turn_index=1,
        snapshot=TrajectorySnapshot(t_star_ms=5000.0, states=(rooted_state,)),
    )
    dataset = DatasetMetadata(
        conversations=[
            *_children_dataset().conversations,
            ConversationMetadata(
                conversation_id="trace_1",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=1000.0),
                ],
                is_root=True,
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[_rootless_trajectory(), rooted],
        dataset=dataset,
    )

    with caplog.at_level(logging.INFO, logger="AgenticReplayTiming"):
        await strategy.setup_phase()

    msgs = [r.getMessage() for r in caplog.records]
    assert any("rootless" in m and "1" in m for m in msgs), msgs
