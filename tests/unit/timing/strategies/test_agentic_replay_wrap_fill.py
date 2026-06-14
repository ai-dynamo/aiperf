# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AgenticReplayStrategy with wrap-filled (shared-trace) lanes.

Covers invariants relaxed when ``len(distinct trace_ids) < concurrency``:

1. ``_active_traces`` is a multiset; ``_pop_next_eligible_trace`` skips only
   when every lane for a trace is busy.
2. ``_lanes_per_trace`` reflects wrap-fill distribution.
3. Old "any lane busy" semantics preserved when every trajectory has a
   distinct trace_id (every lanes_per_trace value == 1).
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.timing.trajectory_source import Trajectory
from tests.unit.timing.strategies.test_agentic_replay_recycle_adversarial import (
    _make_dataset,
    _make_strategy,
)


@pytest.mark.asyncio
async def test_lanes_per_trace_reflects_wrap_fill_distribution():
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    ds = _make_dataset(num_traces=2, turns_per_trace=4)
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=AsyncMock(),
    )
    assert strategy._lanes_per_trace == Counter({"trace_0": 2, "trace_1": 1})


@pytest.mark.asyncio
async def test_double_recycle_guard_keys_on_correlation_id():
    """Two lanes share trace_0. Lane A and lane B independently complete
    final turns with DISTINCT correlation_ids. Neither should trip the
    double-recycle RuntimeError.
    """
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=3, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    strategy._correlation_to_lane["xcorr_a"] = 0
    strategy._correlation_to_lane["xcorr_b"] = 1

    final_a = MagicMock()
    final_a.conversation_id = "trace_0"
    final_a.x_correlation_id = "xcorr_a"
    final_a.turn_index = 1
    final_a.num_turns = 2
    final_a.agent_depth = 0
    final_a.phase = CreditPhase.PROFILING

    final_b = MagicMock()
    final_b.conversation_id = "trace_0"
    final_b.x_correlation_id = "xcorr_b"
    final_b.turn_index = 1
    final_b.num_turns = 2
    final_b.agent_depth = 0
    final_b.phase = CreditPhase.PROFILING

    await strategy.handle_credit_return(final_a)
    await strategy.handle_credit_return(final_b)


@pytest.mark.asyncio
async def test_double_recycle_guard_still_fires_on_repeated_correlation_id():
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    strategy._correlation_to_lane["xcorr_a"] = 0

    final = MagicMock()
    final.conversation_id = "trace_0"
    final.x_correlation_id = "xcorr_a"
    final.turn_index = 1
    final.num_turns = 2
    final.agent_depth = 0
    final.phase = CreditPhase.PROFILING

    await strategy.handle_credit_return(final)
    with pytest.raises(RuntimeError, match="Double recycle"):
        await strategy.handle_credit_return(final)


@pytest.mark.asyncio
async def test_recycle_guard_window_is_bounded_fifo():
    """The double-recycle guard set is FIFO-bounded by RECYCLE_GUARD_MAX_WINDOW.

    Without a bound it retains one entry per recycled session for the entire
    PROFILING phase (a memory leak on long, high-throughput runs). Once the
    window is full the oldest recycled correlation_id is evicted; a duplicate
    still within the window must still raise.
    """
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=8, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    strategy._recycle_guard_max_window = 2

    for i in range(4):
        corr = f"xcorr_{i}"
        strategy._correlation_to_lane[corr] = 0
        await strategy._spawn_from_recycle_or_id(
            "trace_0", finished_correlation_id=corr
        )

    # Bounded to the window: oldest two evicted, newest two retained.
    assert len(strategy._in_flight_recycled) == 2
    assert "xcorr_0" not in strategy._in_flight_recycled
    assert "xcorr_1" not in strategy._in_flight_recycled
    assert "xcorr_2" in strategy._in_flight_recycled
    assert "xcorr_3" in strategy._in_flight_recycled

    # A duplicate still within the window still raises (guard preserved).
    strategy._correlation_to_lane["xcorr_3"] = 0
    with pytest.raises(RuntimeError, match="Double recycle"):
        await strategy._spawn_from_recycle_or_id(
            "trace_0", finished_correlation_id="xcorr_3"
        )


@pytest.mark.asyncio
async def test_warning_emitted_when_wrap_fill_and_cache_bust_none(caplog):
    import logging

    from aiperf.common.enums import CacheBustTarget

    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=1, turns_per_trace=4)
    with caplog.at_level(logging.WARNING, logger="AgenticReplayTiming"):
        _make_strategy(
            phase=CreditPhase.PROFILING,
            trajectories=trajectories,
            dataset=ds,
            issuer=AsyncMock(),
            cache_bust_target=CacheBustTarget.NONE,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("cache_bust" in m.lower() and "identical" in m.lower() for m in msgs), (
        msgs
    )


@pytest.mark.asyncio
async def test_no_warning_when_wrap_fill_and_cache_bust_set(caplog):
    import logging

    from aiperf.common.enums import CacheBustTarget

    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=1, turns_per_trace=4)
    with caplog.at_level(logging.WARNING, logger="AgenticReplayTiming"):
        _make_strategy(
            phase=CreditPhase.PROFILING,
            trajectories=trajectories,
            dataset=ds,
            issuer=AsyncMock(),
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("identical" in m.lower() for m in msgs), msgs


@pytest.mark.asyncio
async def test_no_warning_when_no_wrap_fill_and_cache_bust_none(caplog):
    """Warning is about wrap-fill creating identical traffic, not about
    cache-bust being off in general.
    """
    import logging

    from aiperf.common.enums import CacheBustTarget

    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    ds = _make_dataset(num_traces=2, turns_per_trace=4)
    with caplog.at_level(logging.WARNING, logger="AgenticReplayTiming"):
        _make_strategy(
            phase=CreditPhase.PROFILING,
            trajectories=trajectories,
            dataset=ds,
            issuer=AsyncMock(),
            cache_bust_target=CacheBustTarget.NONE,
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("identical" in m.lower() for m in msgs), msgs
