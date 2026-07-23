# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for the FIFO recycle queue in AgenticReplayStrategy."""

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
)
from tests.unit.timing.strategies._shared_helpers import (
    _build_real_trajectory_source,
    _make_credit,
    _make_dataset,
)


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
    run = None
    if cache_bust_target is not None:
        run = _make_run(target=cache_bust_target, benchmark_id="bench_test")
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        run=run,
    )
    return strategy, issuer, stop_checker


def _make_run(*, target: CacheBustTarget, benchmark_id: str = "bench_test"):
    """Build a v2 ``BenchmarkRun`` exposing the values the strategy reads."""
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "type": "completions",
                "urls": ["http://localhost:8000/v1"],
                "streaming": False,
            },
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "prompts": {"cache_bust": {"target": target}},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 1,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id=benchmark_id,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )


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

    strategy._correlation_to_lane["xcorr"] = 0

    final = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=3)
    await strategy.handle_credit_return(final)

    assert issued == [("trace_0", 0)]


@pytest.mark.asyncio
async def test_pool_one_concurrency_two_no_deadlock():
    """Two trajectories but only one queued trace -> second consumer's recycle"""
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

    strategy._correlation_to_lane["xcorr_a"] = 0
    strategy._correlation_to_lane["xcorr_b"] = 1

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

    assert len(issued) == 2
    assert issued == ["trace_0", "trace_1"]


@pytest.mark.asyncio
async def test_burst_of_ten_completions_recycle_in_sampler_order():
    """10 sessions complete sequentially within the same loop tick."""
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

    for i in range(10):
        strategy._correlation_to_lane[f"xcorr_{i}"] = i

    for i in range(10):
        await strategy.handle_credit_return(
            _make_credit(
                conversation_id=f"trace_{i}",
                turn_index=1,
                num_turns=2,
                x_correlation_id=f"xcorr_{i}",
            )
        )

    assert served == [f"trace_{i}" for i in range(10)]


@pytest.mark.asyncio
async def test_concurrent_recycle_serves_distinct_roots_from_pool():
    """Drive 50 completions concurrently via asyncio.gather."""
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

    assert len(served) == 50
    assert set(served) <= root_ids
    assert len(set(served)) == 50


@pytest.mark.asyncio
async def test_double_recycle_same_trace_raises():
    """Calling handle_credit_return twice for the same final turn must raise."""
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
    strategy._in_flight_recycled.add("xcorr")
    strategy._correlation_to_lane["xcorr"] = 0

    final = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=2)
    with pytest.raises(RuntimeError, match="Double recycle"):
        await strategy.handle_credit_return(final)


@pytest.mark.asyncio
async def test_recycle_during_cooldown_does_not_start_new_sessions():
    """When DurationStopCondition has fired, in-flight credit returns must not"""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=5, turns_per_trace=2)
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    stop_checker = MagicMock()
    stop_checker.can_start_new_session.return_value = False
    strategy, _, _ = _make_strategy(
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

    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_large_pool_every_trace_replayed_deterministic_order():
    """750 traces, 100 trajectories, run for several recycle generations."""
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

    from collections import deque

    in_flight: deque[tuple[str, str]] = deque()
    for lane in range(trajectory_count):
        corr = f"xcorr_traj_{lane}"
        strategy._correlation_to_lane[corr] = lane
        in_flight.append((f"trace_{lane}", corr))

    total_completions = 1500
    for _ in range(total_completions):
        finishing_trace, finishing_corr = in_flight.popleft()
        before = len(served)
        await strategy.handle_credit_return(
            _make_credit(
                conversation_id=finishing_trace,
                turn_index=turns_per_trace - 1,
                num_turns=turns_per_trace,
                x_correlation_id=finishing_corr,
            )
        )
        assert len(served) == before + 1
        in_flight.append((served[-1], served_correlation_ids[-1]))

    served_set = set(served)
    for i in range(num_traces):
        assert f"trace_{i}" in served_set, f"trace_{i} never replayed"

    assert served[:num_traces] == [f"trace_{i}" for i in range(num_traces)]
    assert served[num_traces : 2 * num_traces] == [
        f"trace_{i}" for i in range(num_traces)
    ]


@pytest.mark.asyncio
async def test_trajectory_with_one_turn_recycles_immediately_at_profiling_start():
    """Trajectory's trace has exactly one turn (k_i = 0 = last turn)."""
    trajectory = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
    ]
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

    assert len(issued) == 1
    assert issued[0] == ("trace_0", 0)


@pytest.mark.asyncio
async def test_recycle_missing_correlation_id_logs_warning(caplog):
    """When _spawn_from_recycle_or_id is called with a finished_correlation_id"""
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

    assert issuer.issue_credit.await_count == 1


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
    """A DAG-child final-turn return must NOT dispatch a fresh session and must"""
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

    assert issuer.issue_credit.await_count == 0
    assert child_cid not in strategy._in_flight_recycled


@pytest.mark.asyncio
async def test_child_final_turn_repeated_does_not_trigger_double_recycle():
    """Regression for the production crash: when the parent trace is recycled"""
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
    await strategy.handle_credit_return(
        _make_child_credit(
            conversation_id=child_cid,
            turn_index=2,
            num_turns=3,
            x_correlation_id="xcorr_child_pass0",
        )
    )
    await strategy.handle_credit_return(
        _make_child_credit(
            conversation_id=child_cid,
            turn_index=2,
            num_turns=3,
            x_correlation_id="xcorr_child_pass1",
        )
    )

    assert child_cid not in strategy._in_flight_recycled
    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_child_non_final_turn_still_dispatches_next_turn():
    """Non-final child returns MUST continue to dispatch the next turn — the"""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=2)

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

    assert issued == [(child_cid, 1)]


@pytest.mark.asyncio
async def test_root_final_turn_still_recycles_after_child_shortcircuit():
    """Regression baseline: the child-final short-circuit must not affect"""
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

    root_final = _make_credit(conversation_id="trace_0", turn_index=1, num_turns=2)
    await strategy.handle_credit_return(root_final)

    assert len(issued) == 1
    assert issued[0] == "trace_0"
