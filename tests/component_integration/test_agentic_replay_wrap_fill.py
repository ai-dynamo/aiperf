# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration E2E: agentic_replay with pool < concurrency.

Validates the full warmup -> profiling -> recycle loop when the trajectory
pool is smaller than --concurrency (wrap-fill activated). Asserts:

1. Strategy construction succeeds (no InsufficientTrajectoriesError - that
   class is gone in this branch).
2. Warmup dispatches one credit per LANE (not per distinct trace).
3. Each lane's cache-bust marker is unique even when lanes share a
   trace_id, because the marker digest includes lane_index.
4. Profiling completes without raising the double-recycle RuntimeError
   that previously fired when two lanes finished the same trace_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import (
    CacheBustTarget,
    ConversationBranchMode,
    CreditPhase,
)
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.component_integration


# =============================================================================
# Test-local helpers
# =============================================================================
#
# Duplicated (not imported) from the pool_concurrency integration test on
# purpose: component-integration files keep their helpers local to avoid
# collection-order coupling. The shapes are intentionally similar.


@dataclass
class _DispatchLog:
    """Capture every credit issued through the strategy for ordering checks."""

    entries: list[tuple[CreditPhase, str, int]] = field(default_factory=list)
    """List of (phase, conversation_id, turn_index) per dispatched credit."""


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
    """Synthetic DatasetMetadata with uniform turn counts and no inter-turn delays."""
    convs: list[ConversationMetadata] = []
    for i in range(num_traces):
        turns = [
            TurnMetadata(timestamp_ms=None, delay_ms=None)
            for _ in range(turns_per_trace)
        ]
        convs.append(ConversationMetadata(conversation_id=f"trace_{i}", turns=turns))
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def _make_recording_issuer(
    log: _DispatchLog, current_phase: list[CreditPhase]
) -> AsyncMock:
    issuer = AsyncMock()

    async def _issue(turn) -> bool:
        log.entries.append((current_phase[0], turn.conversation_id, turn.turn_index))
        return True

    issuer.issue_credit.side_effect = _issue
    return issuer


def _make_stop_checker(allow_new_sessions: bool = True) -> MagicMock:
    sc = MagicMock()
    sc.can_start_new_session.return_value = allow_new_sessions
    return sc


def _build_strategy(
    *,
    phase: CreditPhase,
    source: TrajectorySource,
    issuer: AsyncMock,
    cache_bust_target: CacheBustTarget,
    benchmark_id: str = "bench_e2e",
    stop_checker: MagicMock | None = None,
) -> AgenticReplayStrategy:
    """Build an AgenticReplayStrategy with a MagicMock user_config wired up.

    Mirrors the unit-test `_make_strategy` pattern from Task 6 so the
    cache-bust target plumbs through to `_cache_bust_target` and
    `_session_marker` digests are produced for the marker-uniqueness
    assertion.
    """
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(source.trajectories)
    user_config = MagicMock()
    user_config.input.prompt.cache_bust.target = cache_bust_target
    user_config.benchmark_id = benchmark_id
    return AgenticReplayStrategy(
        config=cfg,
        conversation_source=source,
        scheduler=MagicMock(),
        stop_checker=stop_checker if stop_checker is not None else _make_stop_checker(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        user_config=user_config,
    )


def _make_final_credit(
    *,
    conversation_id: str,
    x_correlation_id: str,
    num_turns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    """Build a final-turn Credit (turn_index == num_turns - 1)."""
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=num_turns - 1,
        num_turns=num_turns,
        issued_at_ns=0,
        agent_depth=0,
        branch_mode=ConversationBranchMode.FORK,
    )


# =============================================================================
# E2E test: pool=1, concurrency=4 -> wrap-fill activates, full loop completes
# =============================================================================


@pytest.mark.asyncio
async def test_pool_one_concurrency_four_wrap_fill_e2e() -> None:
    """1-trace pool, 4-way concurrency: wrap-fill kicks in.

    The four lanes all run ``trace_0`` with decorrelated ``start_turn_index``
    values, distinct per-lane cache-bust markers, and the profiling recycle
    loop completes without tripping the double-recycle guard.
    """
    dataset = _make_dataset(num_traces=1, turns_per_trace=6)
    sampler = _SequentialSampler([c.conversation_id for c in dataset.conversations])

    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=4,
        random_seed=42,
    )

    # 1. Wrap-fill construction contract.
    assert len(source.trajectories) == 4
    assert all(t.conversation_id == "trace_0" for t in source.trajectories)
    distinct_k = {t.start_turn_index for t in source.trajectories}
    assert len(distinct_k) >= 2, (
        f"wrap-fill must decorrelate k_i across lanes sharing trace_0; "
        f"got start_turn_index values={sorted(distinct_k)!r}"
    )

    # 2. Warmup: one credit per LANE (not per distinct trace).
    warmup_log = _DispatchLog()
    current_phase = [CreditPhase.WARMUP]
    warmup_issuer = _make_recording_issuer(warmup_log, current_phase)
    warmup = _build_strategy(
        phase=CreditPhase.WARMUP,
        source=source,
        issuer=warmup_issuer,
        cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
    )
    await warmup.execute_phase()

    assert warmup_issuer.issue_credit.await_count == 4, (
        f"warmup must dispatch one credit per lane (4), not per distinct "
        f"trace (1); got await_count={warmup_issuer.issue_credit.await_count}"
    )

    # 3. Per-lane cache-bust markers are unique.
    markers = list(warmup._session_marker.values())
    assert len(markers) == 4, f"expected 4 session markers, got {markers!r}"
    assert all(m is not None for m in markers), (
        f"every lane must have a non-None marker when cache_bust.target != NONE; "
        f"got {markers!r}"
    )
    assert len(set(markers)) == 4, (
        f"per-lane markers must be byte-distinct (digest salts with lane_index); "
        f"got {markers!r}"
    )

    # 4. Profiling: setup + execute_phase + simulate 4 final-turn returns.
    # Build a FRESH strategy for PROFILING (PhaseRunner does the same).
    profiling_log = _DispatchLog()
    current_phase_p = [CreditPhase.PROFILING]
    profiling_issuer = _make_recording_issuer(profiling_log, current_phase_p)
    profiling = _build_strategy(
        phase=CreditPhase.PROFILING,
        source=source,
        issuer=profiling_issuer,
        cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
    )
    await profiling.setup_phase()
    await profiling.execute_phase()

    # After PROFILING execute_phase, each lane has one in-flight session.
    assert profiling_issuer.issue_credit.await_count == 4
    assert profiling._active_traces["trace_0"] == 4

    # Snapshot the 4 active correlation_ids (one per lane).
    initial_correlations = list(profiling._correlation_to_lane.keys())
    assert len(initial_correlations) == 4

    # 5. Simulate each lane's final turn returning. The strategy should
    # recycle each one into a fresh session (queue head is the just-finished
    # trace_id, only entry in the pool). No double-recycle guard trip.
    pre_recycle_dispatches = profiling_issuer.issue_credit.await_count
    for xcorr in initial_correlations:
        final = _make_final_credit(
            conversation_id="trace_0",
            x_correlation_id=xcorr,
            num_turns=6,
        )
        # The await must not raise (would fire if the trace-id-keyed
        # guard were still in place, or if Counter bookkeeping went
        # negative).
        await profiling.handle_credit_return(final)

    # 6. Strategy continued dispatching: each finished lane recycled into a
    # fresh session. Exactly 4 new dispatches (one per recycled lane).
    post_recycle_dispatches = profiling_issuer.issue_credit.await_count
    assert post_recycle_dispatches > pre_recycle_dispatches, (
        f"recycle must dispatch fresh sessions for each finished lane; "
        f"pre={pre_recycle_dispatches}, post={post_recycle_dispatches}"
    )
    assert post_recycle_dispatches == pre_recycle_dispatches + 4, (
        f"expected exactly 4 new dispatches (one per recycled lane); "
        f"pre={pre_recycle_dispatches}, post={post_recycle_dispatches}"
    )

    # Steady-state: 4 lanes still active on trace_0 (the only trace in the pool).
    assert profiling._active_traces["trace_0"] == 4
    # 4 fresh correlation_ids replaced the originals (1-to-1 lane reuse).
    assert len(profiling._correlation_to_lane) == 4
    assert set(profiling._correlation_to_lane.keys()).isdisjoint(initial_correlations)
