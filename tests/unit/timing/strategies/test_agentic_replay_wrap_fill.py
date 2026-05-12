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
from unittest.mock import AsyncMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.timing.trajectory_source import Trajectory
from tests.unit.timing.strategies.test_agentic_replay_recycle_adversarial import (
    _make_dataset,
    _make_strategy,
)


@pytest.mark.asyncio
async def test_active_traces_uses_counter_for_shared_lanes():
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=1, turns_per_trace=4)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.execute_phase()
    assert isinstance(strategy._active_traces, Counter)
    assert strategy._active_traces["trace_0"] == 2


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
async def test_pop_eligible_skips_only_when_all_lanes_busy():
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]
    ds = _make_dataset(num_traces=1, turns_per_trace=4)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=issuer,
    )
    await strategy.setup_phase()
    strategy._active_traces["trace_0"] = 2
    # All 2 lanes busy: pop returns None.
    assert strategy._pop_next_eligible_trace() is None
    # Lane 0 finishes — decrement.
    strategy._active_traces["trace_0"] -= 1
    # Now one lane free; same trace eligible.
    assert strategy._pop_next_eligible_trace() == "trace_0"


@pytest.mark.asyncio
async def test_pop_eligible_old_behavior_preserved_when_no_duplicates():
    trajectories = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_1", start_turn_index=0),
    ]
    ds = _make_dataset(num_traces=3, turns_per_trace=4)
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        dataset=ds,
        issuer=AsyncMock(),
    )
    await strategy.setup_phase()
    strategy._active_traces["trace_0"] = 1
    popped = strategy._pop_next_eligible_trace()
    # trace_0 capped (1/1) — skip and pop another.
    assert popped in {"trace_1", "trace_2"}
