# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for the AgenticReplayStrategy warmup-failure attribution path.

Pins the contract for the WARMUP failure path of the ``agentic_replay`` timing
mode (spec §4.2 / §8.4.7):
  - ``record_warmup_failure(trace_id)`` accumulates per-trajectory terminal
    failures into ``_failed_warmup_traces``.
  - ``report_warmup_failures()`` raises ``TrajectoryWarmupFailedError`` with
    exactly the recorded trace_ids when any are present, else no-op.
  - PROFILING ``setup_phase`` raises ``RuntimeError`` only after the source
    has been explicitly cleaned; ``report_warmup_failures`` itself does NOT
    auto-clean the trajectory list.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.component_integration


# =============================================================================
# Helpers
# =============================================================================


class _SequentialSampler:
    """Deterministic sampler over a fixed conversation_id list."""

    def __init__(self, conversation_ids: list[str]) -> None:
        self._ids = list(conversation_ids)
        self._idx = 0

    def next_conversation_id(self) -> str:
        if self._idx >= len(self._ids):
            raise StopIteration
        cid = self._ids[self._idx]
        self._idx += 1
        return cid


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


def _make_real_source(
    num_traces: int,
    turns_per_trace: int,
    *,
    concurrency: int,
    seed: int,
) -> TrajectorySource:
    ds = _make_dataset(num_traces, turns_per_trace)
    sampler = _SequentialSampler([c.conversation_id for c in ds.conversations])
    return TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=sampler,
        concurrency=concurrency,
        random_seed=seed,
    )


def _make_recording_issuer() -> AsyncMock:
    """Build an AsyncMock credit_issuer; pulls dispatched turns via await_args_list."""
    issuer = AsyncMock()
    issuer.issue_credit.return_value = True
    return issuer


def _build_strategy(
    *,
    phase: CreditPhase,
    source: TrajectorySource,
    issuer: AsyncMock,
) -> AgenticReplayStrategy:
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(source.trajectories)
    return AgenticReplayStrategy(
        config=cfg,
        conversation_source=source,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )


def _captured_warmup_pairs(issuer: AsyncMock) -> list[tuple[str, str]]:
    """Return ``[(x_correlation_id, conversation_id), ...]`` from issued turns."""
    pairs: list[tuple[str, str]] = []
    for call in issuer.issue_credit.await_args_list:
        turn = call.args[0]
        pairs.append((turn.x_correlation_id, turn.conversation_id))
    return pairs


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
async def test_partial_warmup_failure_three_of_four_raises_with_only_failed_ids() -> (
    None
):
    """3/4 trajectories fail terminally: error lists exactly those 3, in record order."""
    source = _make_real_source(
        num_traces=4, turns_per_trace=5, concurrency=4, seed=12345
    )
    assert len(source.trajectories) == 4

    issuer = _make_recording_issuer()
    strategy = _build_strategy(phase=CreditPhase.WARMUP, source=source, issuer=issuer)

    await strategy.setup_phase()
    await strategy.execute_phase()

    assert len(_captured_warmup_pairs(issuer)) == 4

    failed_ids = [t.conversation_id for t in source.trajectories[:3]]
    survivor_id = source.trajectories[3].conversation_id

    for trace_id in failed_ids:
        strategy.record_warmup_failure(trace_id)

    with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
        strategy.report_warmup_failures()

    assert exc_info.value.failed_trace_ids == failed_ids, (
        "Error must carry exactly the recorded trace_ids in the order recorded"
    )
    assert survivor_id not in exc_info.value.failed_trace_ids


@pytest.mark.asyncio
async def test_total_warmup_failure_all_four_raises_with_all_ids() -> None:
    """All 4 trajectories fail: error lists all 4 conversation_ids."""
    source = _make_real_source(
        num_traces=4, turns_per_trace=5, concurrency=4, seed=12345
    )
    assert len(source.trajectories) == 4

    issuer = _make_recording_issuer()
    strategy = _build_strategy(phase=CreditPhase.WARMUP, source=source, issuer=issuer)

    await strategy.setup_phase()
    await strategy.execute_phase()

    all_ids = [t.conversation_id for t in source.trajectories]
    for trace_id in all_ids:
        strategy.record_warmup_failure(trace_id)

    with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
        strategy.report_warmup_failures()

    assert exc_info.value.failed_trace_ids == all_ids
    assert len(exc_info.value.failed_trace_ids) == 4


@pytest.mark.asyncio
async def test_warmup_failure_blocks_profiling_setup() -> None:
    """report_warmup_failures does NOT auto-clean trajectories.

    Pins that cleanup is the caller's responsibility: after a raise, building
    a PROFILING strategy from the SAME source still succeeds at setup_phase
    because the trajectories list is still populated. Production prevents this
    via PhaseRunner stopping before PROFILING — but the source itself is
    untouched.
    """
    source = _make_real_source(
        num_traces=4, turns_per_trace=5, concurrency=4, seed=12345
    )

    warmup_issuer = _make_recording_issuer()
    warmup_strategy = _build_strategy(
        phase=CreditPhase.WARMUP, source=source, issuer=warmup_issuer
    )
    await warmup_strategy.setup_phase()
    await warmup_strategy.execute_phase()

    for trajectory in source.trajectories:
        warmup_strategy.record_warmup_failure(trajectory.conversation_id)

    with pytest.raises(TrajectoryWarmupFailedError):
        warmup_strategy.report_warmup_failures()

    # Source not auto-cleaned.
    assert len(source.trajectories) == 4

    profiling_issuer = _make_recording_issuer()
    profiling_strategy = _build_strategy(
        phase=CreditPhase.PROFILING, source=source, issuer=profiling_issuer
    )
    # Must not raise: trajectories list still populated.
    await profiling_strategy.setup_phase()


@pytest.mark.asyncio
async def test_report_warmup_failures_with_no_failures_is_noop() -> None:
    """No record_warmup_failure calls -> report_warmup_failures returns None silently."""
    source = _make_real_source(
        num_traces=4, turns_per_trace=5, concurrency=4, seed=12345
    )

    issuer = _make_recording_issuer()
    strategy = _build_strategy(phase=CreditPhase.WARMUP, source=source, issuer=issuer)

    await strategy.setup_phase()
    await strategy.execute_phase()

    # No record_warmup_failure calls at all.
    result = strategy.report_warmup_failures()
    assert result is None
    assert strategy._failed_warmup_traces == []


@pytest.mark.asyncio
async def test_report_warmup_failures_can_be_called_after_record_then_clear() -> None:
    """_failed_warmup_traces is the sole state; clearing it makes report a no-op.

    Pins that report_warmup_failures has no internal "already raised" flag —
    behavior is purely a function of the current contents of
    ``_failed_warmup_traces``.
    """
    source = _make_real_source(
        num_traces=4, turns_per_trace=5, concurrency=4, seed=12345
    )

    issuer = _make_recording_issuer()
    strategy = _build_strategy(phase=CreditPhase.WARMUP, source=source, issuer=issuer)

    await strategy.setup_phase()
    await strategy.execute_phase()

    failed_ids = [t.conversation_id for t in source.trajectories[:2]]
    for trace_id in failed_ids:
        strategy.record_warmup_failure(trace_id)

    with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
        strategy.report_warmup_failures()
    assert exc_info.value.failed_trace_ids == failed_ids

    # Direct mutation: clearing the list returns the strategy to a no-op state.
    strategy._failed_warmup_traces.clear()

    # Second call: must NOT raise.
    result = strategy.report_warmup_failures()
    assert result is None
    assert strategy._failed_warmup_traces == []
