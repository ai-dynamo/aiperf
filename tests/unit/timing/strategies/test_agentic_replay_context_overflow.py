# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-overflow short-circuit tests for AgenticReplayStrategy."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    DatasetMetadata,
)
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


@pytest.mark.asyncio
async def test_mid_trajectory_context_overflow_recycles_trace():
    """Non-final turn with context-overflow error → recycle to next trace."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=5)
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
    # Seed the lane mapping; _active_traces was pre-registered by
    # setup_phase. The finishing trace is discarded from _active_traces at
    # the top of _spawn_from_recycle_or_id before the pop loop runs.
    strategy._correlation_to_lane["xcorr"] = 0
    strategy._root_to_lane["xcorr"] = 0

    # Mid-trajectory turn (index 2 of 5) errors with context-overflow.
    mid = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    await strategy.handle_credit_return(
        mid, error="This model's maximum context length is 131072 tokens"
    )

    # Should NOT have dispatched turn 3 of trace_0 — overflow short-circuit
    # terminates the trajectory mid-flight rather than continuing.
    assert ("trace_0", 3) not in issued, (
        f"trajectory should not advance after overflow; got issued={issued}"
    )
    # With the full-pool recycle queue, the head is trace_0 (iteration order
    # from dataset_metadata.conversations). After the discard-at-top removes
    # trace_0 from _active_traces, the pop loop pulls trace_0 and spawns a
    # fresh session for it at turn 0. This is the spec-correct recycle —
    # the trajectory's own trace_id is back in the rotation pool.
    assert ("trace_0", 0) in issued, (
        f"recycle should have spawned a fresh session at turn 0; got issued={issued}"
    )
    # Finished root mapping pruned; recycled session may re-seed under a new id.
    assert "xcorr" not in strategy._root_to_lane


@pytest.mark.asyncio
async def test_tree_drained_prunes_root_to_lane():
    """Registry drain callback must pop the finished root from ``_root_to_lane``."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=3)
    issuer = AsyncMock()
    issuer.issue_credit = AsyncMock(return_value=True)
    issuer.replay_gate = MagicMock()
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )
    # Engage registry path so setup_phase wires the drain callback.
    strategy._session_tree_registry = MagicMock()
    await strategy.setup_phase()
    strategy._correlation_to_lane["root-corr"] = 0
    strategy._root_to_lane["root-corr"] = 0
    strategy._session_marker["root-corr"] = "marker"

    strategy._on_tree_drained("root-corr", CreditPhase.PROFILING)

    assert "root-corr" not in strategy._root_to_lane
    assert "root-corr" not in strategy._correlation_to_lane
    assert "root-corr" not in strategy._session_marker
    issuer.replay_gate.close_root.assert_called_once_with("root-corr")
    strategy.scheduler.schedule_later.assert_called_once()


@pytest.mark.asyncio
async def test_non_overflow_error_does_not_recycle():
    """Non-context-overflow errors (e.g. 500s) should NOT short-circuit."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=5)
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

    # Mid-trajectory turn errors with a transient 500.
    mid = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    await strategy.handle_credit_return(
        mid, error="Internal server error: pool exhausted"
    )

    # Should dispatch turn 3 of trace_0, NOT recycle.
    assert ("trace_0", 3) in issued, (
        f"trajectory should advance on non-overflow error; got issued={issued}"
    )
    assert ("trace_1", 0) not in issued, (
        f"recycle should not fire on generic errors; got issued={issued}"
    )


@pytest.mark.asyncio
async def test_final_turn_overflow_recycles_normally():
    """Final-turn overflow takes the same recycle path as any final-turn return."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=3)
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
    # Seed the lane mapping; _active_traces was pre-registered by
    # setup_phase (the finishing trace is discarded at the top of
    # _spawn_from_recycle_or_id before the pop loop runs).
    strategy._correlation_to_lane["xcorr"] = 0

    final = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=3)
    await strategy.handle_credit_return(
        final, error="context_length_exceeded: prompt too long"
    )

    # Final-turn return always recycles, independent of error status. With the
    # full-pool recycle queue, head=trace_0; after the top-of-function discard
    # removes trace_0 from _active_traces, the pop loop spawns a fresh session
    # for trace_0 at turn 0.
    assert ("trace_0", 0) in issued, (
        f"final-turn return should recycle; got issued={issued}"
    )


@pytest.mark.asyncio
async def test_overflow_error_during_warmup_is_noop():
    """WARMUP returns are no-ops at the strategy level even with overflow."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=5)
    issued: list[tuple[str, int]] = []

    async def capture(turn):
        issued.append((turn.conversation_id, turn.turn_index))
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectory,
        dataset=ds,
        issuer=issuer,
    )

    mid = _make_credit(
        conversation_id="trace_0",
        turn_index=2,
        num_turns=5,
        phase=CreditPhase.WARMUP,
    )
    await strategy.handle_credit_return(
        mid, error="This model's maximum context length is 131072 tokens"
    )

    # WARMUP is a no-op — no recycle, no dispatch.
    assert issued == []


@pytest.mark.asyncio
async def test_no_error_falls_through_to_next_turn():
    """Default error=None path must still dispatch the next turn unchanged."""
    trajectory = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    ds = _make_dataset(num_traces=2, turns_per_trace=5)
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

    mid = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    await strategy.handle_credit_return(mid)  # no error kwarg

    assert ("trace_0", 3) in issued
    assert ("trace_1", 0) not in issued
