# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for AgenticReplayStrategy phase-branching."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.credit.structs import Credit
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import TimingMode
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    Trajectory,
    TrajectorySource,
)
from tests.unit.timing.strategies._shared_helpers import _make_credit, _make_dataset


def _build_real_trajectory_source(
    num_traces: int,
    turns_per_trace: int,
    trajectories: list[Trajectory],
) -> TrajectorySource:
    ds = _make_dataset(num_traces, turns_per_trace)
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    _roots = [
        c.conversation_id
        for c in src._dataset_metadata.conversations
        if getattr(c, "is_root", True)
    ]
    src._dataset_sampler = SequentialSampler(_roots) if _roots else MagicMock()
    src._pool_size = len(_roots)
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src._random_seed = 0
    src._target_size = len(trajectories)
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
    timing_mode=None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock, TrajectorySource]:
    src = _build_real_trajectory_source(num_traces, turns_per_trace, trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.timing_mode = (
        timing_mode if timing_mode is not None else TimingMode.AGENTIC_REPLAY
    )
    cfg.concurrency = max(1, len(trajectories))
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    return strategy, issuer, scheduler, src


def test_warmup_phase_with_non_agentic_timing_mode_pins_current_behavior():
    """Test 1: ``config.phase = WARMUP`` with ``config.timing_mode != AGENTIC_REPLAY``."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    src = _build_real_trajectory_source(1, 2, trajectory)
    cfg = MagicMock()
    cfg.phase = CreditPhase.WARMUP
    cfg.timing_mode = TimingMode.REQUEST_RATE
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=AsyncMock(),
        lifecycle=MagicMock(),
    )
    assert strategy.config.timing_mode == TimingMode.REQUEST_RATE
    assert strategy.config.phase == CreditPhase.WARMUP


@pytest.mark.asyncio
async def test_warmup_empty_trajectories_emits_no_credits():
    """Test 2a: Empty trajectory during WARMUP -> strategy issues zero credits."""
    strategy, issuer, _, _ = _make_strategy(phase=CreditPhase.WARMUP, trajectories=[])
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_profiling_empty_trajectories_aborts_setup_with_clear_error():
    """Test 2b: PROFILING phase with empty trajectory raises a clear error."""
    strategy, _, _, _ = _make_strategy(phase=CreditPhase.PROFILING, trajectories=[])
    with pytest.raises(RuntimeError) as exc_info:
        await strategy.setup_phase()
    msg = str(exc_info.value)
    assert "trajectory" in msg.lower()
    assert "empty" in msg.lower() or "warmup" in msg.lower()


@pytest.mark.asyncio
async def test_warmup_terminal_failure_blocks_profiling():
    """Test 3: ``record_warmup_failure`` accumulates; ``report_warmup_failures``"""
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    issuer.issue_credit.reset_mock()

    strategy.record_warmup_failure("trace_0")
    strategy.record_warmup_failure("trace_2")

    failed_credit = _make_credit(
        conversation_id="trace_0",
        turn_index=0,
        num_turns=3,
        phase=CreditPhase.WARMUP,
    )
    await strategy.handle_credit_return(failed_credit)
    assert issuer.issue_credit.await_count == 0

    with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
        strategy.report_warmup_failures()
    assert exc_info.value.failed_trace_ids == ["trace_0", "trace_2"]


@pytest.mark.asyncio
async def test_warmup_no_embedded_wallclock_abort():
    """Test 4: Strategy MUST NOT enforce its own wall-clock timeout."""
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(2)
    ]
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert issuer.issue_credit.await_count == 2
    assert not hasattr(strategy, "_warmup_deadline")
    assert not hasattr(strategy, "_warmup_aborted")


@pytest.mark.asyncio
async def test_profiling_without_preceding_warmup_does_not_self_enforce():
    """Test 5: PROFILING with a populated trajectory but no recorded WARMUP completion"""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, issuer, _, src = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        num_traces=3,
        turns_per_trace=4,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    assert issuer.issue_credit.await_count == 1
    issued = issuer.issue_credit.await_args.args[0]
    assert issued.turn_index == 1
    assert issued.conversation_id == "trace_0"


@pytest.mark.asyncio
async def test_profiling_credit_return_after_stop_dispatches_next_turn():
    """Test 6: When ``DurationStopCondition`` has fired, an in-flight trajectory"""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        num_traces=3,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    strategy.lifecycle.is_sending_complete = True

    in_flight_return = _make_credit(
        conversation_id="trace_0", turn_index=1, num_turns=4
    )
    await strategy.handle_credit_return(in_flight_return)
    assert issuer.issue_credit.await_count == 1
    next_turn = issuer.issue_credit.await_args.args[0]
    assert next_turn.turn_index == 2
    assert next_turn.conversation_id == "trace_0"


@pytest.mark.asyncio
async def test_warmup_credit_return_does_not_self_spawn_subagents():
    """Test 7: When a trajectory warmup turn ``k_i`` happens to be a turn flagged for"""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    issuer = AsyncMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectory,
        turns_per_trace=4,
        issuer=issuer,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    spawning_credit = Credit(
        id=0,
        phase=CreditPhase.WARMUP,
        conversation_id="trace_0",
        x_correlation_id="xcorr",
        turn_index=0,
        num_turns=4,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.SPAWN,
        has_forks=True,
    )
    await strategy.handle_credit_return(spawning_credit)
    assert issuer.issue_credit.await_count == 0


def test_strategy_constructed_multiple_times_within_one_phase_is_independent():
    """Test 8: PhaseRunner is contractually expected to construct the strategy"""
    trajectory = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=1),
    ]
    src = _build_real_trajectory_source(3, 4, trajectory)

    def _build():
        cfg = MagicMock()
        cfg.phase = CreditPhase.PROFILING
        cfg.timing_mode = TimingMode.AGENTIC_REPLAY
        cfg.concurrency = 2
        return AgenticReplayStrategy(
            config=cfg,
            conversation_source=src,
            scheduler=MagicMock(),
            stop_checker=MagicMock(),
            credit_issuer=AsyncMock(),
            lifecycle=MagicMock(),
        )

    s1 = _build()
    s2 = _build()

    assert s1 is not s2
    assert s1.conversation_source is s2.conversation_source
    s1.record_warmup_failure("trace_0")
    assert s1._failed_warmup_traces == ["trace_0"]
    assert s2._failed_warmup_traces == []
    s1._in_flight_recycled.add("x")
    assert "x" not in s2._in_flight_recycled


@pytest.mark.asyncio
async def test_strategy_setup_twice_within_one_phase_is_idempotent():
    """Calling ``setup_phase`` twice on the same instance MUST be safe (no"""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        num_traces=3,
        turns_per_trace=4,
    )
    await strategy.setup_phase()
    await strategy.setup_phase()

    assert strategy.conversation_source.next_recycle_conversation_id() is not None


@pytest.mark.asyncio
async def test_warmup_execute_does_not_emit_per_credit_long_warmup_log(caplog):
    """The strategy's WARMUP execute path must not emit a long-warmup INFO log"""
    trajectory = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(5)
    ]
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory
    )
    with caplog.at_level(logging.INFO, logger="AgenticReplayTiming"):
        await strategy.setup_phase()
        await strategy.execute_phase()
    long_warmup_logs = [
        r
        for r in caplog.records
        if "5 minutes" in r.getMessage() or "exceeded" in r.getMessage().lower()
    ]
    assert long_warmup_logs == []
