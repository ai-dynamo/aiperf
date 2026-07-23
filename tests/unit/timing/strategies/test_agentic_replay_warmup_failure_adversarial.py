# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for AgenticReplayStrategy warmup-failure accumulation and dispatch routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    Trajectory,
)
from tests.unit.timing.strategies._shared_helpers import (
    _build_real_trajectory_source,
    _make_credit,
    _make_dataset,
)

# Helpers (duplicated from sibling adversarial tests for self-containment)


def _make_strategy(
    *,
    phase: CreditPhase,
    trajectories: list[Trajectory],
    dataset: DatasetMetadata,
    issuer: AsyncMock | None = None,
    scheduler: MagicMock | None = None,
    stop_checker: MagicMock | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock, MagicMock]:
    src = _build_real_trajectory_source(dataset=dataset, trajectories=trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = max(1, len(trajectories))
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    if stop_checker is None:
        stop_checker = MagicMock()
        stop_checker.can_start_new_session.return_value = True
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    return strategy, issuer, scheduler, stop_checker


# Test 1: record_warmup_failure preserves call order including duplicates


def test_record_warmup_failure_accumulates_in_call_order() -> None:
    """Duplicates and order matter: report_warmup_failures must emit them as recorded."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=1, turns_per_trace=2)
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory, dataset=ds
    )

    strategy.record_warmup_failure("a")
    strategy.record_warmup_failure("b")
    strategy.record_warmup_failure("a")

    assert strategy._failed_warmup_traces == ["a", "b", "a"]


# Test 2: report_warmup_failures with no failures is a noop


def test_report_warmup_failures_empty_is_noop() -> None:
    """Fresh strategy: report_warmup_failures returns None and does not raise."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=1, turns_per_trace=2)
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory, dataset=ds
    )

    result = strategy.report_warmup_failures()
    assert result is None


# Test 3: report_warmup_failures raises with the recorded ids in order


def test_report_warmup_failures_raises_with_failed_trace_ids() -> None:
    """The raised TrajectoryWarmupFailedError carries failed_trace_ids in record order."""
    trajectory = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    ds = _make_dataset(num_traces=2, turns_per_trace=2)
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=trajectory, dataset=ds
    )

    strategy.record_warmup_failure("trace_1")
    strategy.record_warmup_failure("trace_0")

    with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
        strategy.report_warmup_failures()
    assert exc_info.value.failed_trace_ids == ["trace_1", "trace_0"]


# Test 4: WARMUP handle_credit_return is a strategy-level no-op


@pytest.mark.asyncio
async def test_warmup_handle_credit_return_is_noop() -> None:
    """A returning WARMUP credit must not provoke any new issue or schedule."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=1, turns_per_trace=3)
    issuer = AsyncMock()
    scheduler = MagicMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
        scheduler=scheduler,
    )
    await strategy.setup_phase()

    credit = _make_credit(
        conversation_id="trace_0",
        turn_index=0,
        num_turns=3,
        phase=CreditPhase.WARMUP,
    )
    await strategy.handle_credit_return(credit)

    assert issuer.issue_credit.await_count == 0
    scheduler.schedule_later.assert_not_called()


# Test 5: PROFILING credit return during cooldown does not spawn or push


@pytest.mark.asyncio
async def test_profiling_handle_credit_return_during_cooldown_no_spawn() -> None:
    """Cooldown gates the fresh-dispatch step: an in-flight credit returning
    after the stop condition has fired must not start a new session.

    ``_dispatch_recycled_on_lane`` checks ``can_start_new_session`` before
    drawing the next root from the sampler, so no fresh credit is issued.
    """
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=4, turns_per_trace=2)
    issuer = AsyncMock()
    stop_checker = MagicMock()
    stop_checker.can_start_new_session.return_value = False
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
        stop_checker=stop_checker,
    )
    await strategy.setup_phase()
    strategy._correlation_to_lane["xcorr"] = 0

    final = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=2)
    await strategy.handle_credit_return(final)

    # Cooldown gates the fresh spawn: no new credit issued.
    assert issuer.issue_credit.await_count == 0


# Test 6: _dispatch_next_turn with delay_ms=0 issues immediately


@pytest.mark.asyncio
async def test_dispatch_next_turn_with_zero_delay_issues_immediately() -> None:
    """A non-final turn with delay_ms=0 bypasses the scheduler."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=4)
    issuer = AsyncMock()
    scheduler = MagicMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
        scheduler=scheduler,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    strategy.conversation_source.get_next_turn_metadata = MagicMock(
        return_value=TurnMetadata(timestamp_ms=None, delay_ms=0)
    )

    credit = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=4)
    await strategy.handle_credit_return(credit)

    assert issuer.issue_credit.await_count == 1
    scheduler.schedule_later.assert_not_called()


# Test 7: _dispatch_next_turn with positive delay routes through scheduler


@pytest.mark.asyncio
async def test_dispatch_next_turn_with_positive_delay_routes_through_scheduler() -> (
    None
):
    """delay_ms=1500 -> scheduler.schedule_later(1.5, coro); no direct issue."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=4)
    issuer = AsyncMock()
    scheduler = MagicMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
        scheduler=scheduler,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    strategy.conversation_source.get_next_turn_metadata = MagicMock(
        return_value=TurnMetadata(timestamp_ms=None, delay_ms=1500)
    )

    credit = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=4)
    try:
        await strategy.handle_credit_return(credit)
    finally:
        # Production hands a coroutine to scheduler.schedule_later but the
        # MagicMock never awaits it; close it to avoid the "coroutine was
        # never awaited" RuntimeWarning on test teardown.
        if scheduler.schedule_later.call_args is not None:
            coro_arg = scheduler.schedule_later.call_args.args[1]
            if hasattr(coro_arg, "close"):
                coro_arg.close()

    scheduler.schedule_later.assert_called_once()
    delay_arg, coro_arg = scheduler.schedule_later.call_args.args
    assert delay_arg == 1.5
    # Second arg is the issue_credit(turn) coroutine handed to the scheduler.
    assert hasattr(coro_arg, "send") and hasattr(coro_arg, "throw")
    # issue_credit was NOT awaited directly by the strategy - the scheduler
    # owns the coroutine now.
    assert issuer.issue_credit.await_count == 0


# Test 8: _dispatch_next_turn with delay_ms=None issues immediately


@pytest.mark.asyncio
async def test_dispatch_next_turn_with_none_delay_issues_immediately() -> None:
    """delay_ms=None is treated as zero - immediate dispatch, no scheduler."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=4)
    issuer = AsyncMock()
    scheduler = MagicMock()
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
        scheduler=scheduler,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    strategy.conversation_source.get_next_turn_metadata = MagicMock(
        return_value=TurnMetadata(timestamp_ms=None, delay_ms=None)
    )

    credit = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=4)
    await strategy.handle_credit_return(credit)

    assert issuer.issue_credit.await_count == 1
    scheduler.schedule_later.assert_not_called()


# Test 10: PROFILING setup with empty trajectories raises with the canonical message


@pytest.mark.asyncio
async def test_profiling_setup_raises_when_trajectories_empty() -> None:
    """Empty trajectories at PROFILING setup is a degraded WARMUP signal."""
    ds = _make_dataset(num_traces=3, turns_per_trace=2)
    src = _build_real_trajectory_source(dataset=ds, trajectories=[])
    src.trajectories = []  # belt-and-suspenders explicit
    cfg = MagicMock()
    cfg.phase = CreditPhase.PROFILING
    cfg.concurrency = 1
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=AsyncMock(),
        lifecycle=MagicMock(),
    )
    with pytest.raises(RuntimeError) as exc_info:
        await strategy.setup_phase()
    assert "WARMUP must complete" in str(exc_info.value)


# G17: empty-warmup unblock (signal_sending_complete when nothing precedes t*)


@pytest.mark.asyncio
async def test_warmup_signals_complete_when_no_request_precedes_t_star() -> None:
    """When every lane's first request is at/after t* (``warmup_turn_index`` is
    None for all states), ``_execute_warmup`` prepares zero credits. The count
    path that normally drives completion is triggered by credit dispatch, so
    with no credits the warmup barrier (sized to concurrency) would hang. The
    strategy must call ``credit_issuer.signal_sending_complete()`` instead.

    Regression guard for commit 14f7b0e40 (G17): re-introducing the deadlock
    leaves no automated detection otherwise (grep ``signal_sending_complete``
    over tests/ was previously empty).
    """
    dataset = _make_dataset(num_traces=1, turns_per_trace=2)
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP, trajectories=[], dataset=dataset
    )
    # One lane whose snapshot states all start at/after t* -> nothing to warm.
    src = MagicMock()
    src.warmup_credit_count = 0
    traj = MagicMock()
    traj.snapshot = MagicMock()
    traj.snapshot.t_star_ms = 1000.0
    state = MagicMock()
    state.warmup_turn_index = None
    traj.snapshot.states = [state]
    src.trajectories = [traj]
    strategy.conversation_source = src
    strategy._burst_phase_starts = False  # spread mode

    await strategy._execute_warmup()

    issuer.signal_sending_complete.assert_called_once()
    issuer.issue_credit.assert_not_called()
