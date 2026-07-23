# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic_replay cross-phase state continuity adversarial tests."""

from __future__ import annotations

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
from aiperf.timing.trajectory_source import (
    TrajectorySource,
)


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
    """Build a real TrajectorySource with deterministic sampling."""
    ds = _make_dataset(num_traces, turns_per_trace)
    sampler = SequentialSampler([c.conversation_id for c in ds.conversations])
    return TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=sampler,
        concurrency=concurrency,
        random_seed=seed,
    )


def _make_strategy(
    *,
    phase: CreditPhase,
    source: TrajectorySource,
    issuer: AsyncMock | None = None,
    scheduler: MagicMock | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock]:
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(source.trajectories)
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=source,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )
    return strategy, issuer, scheduler


def _make_credit(
    *,
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    x_correlation_id: str = "xcorr",
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.FORK,
    )


def _capture_dispatched_turns(
    issuer: AsyncMock,
) -> list[tuple[str, int, str]]:
    """Materialize all (conversation_id, turn_index, x_correlation_id) triples"""
    out: list[tuple[str, int, str]] = []
    for call in issuer.issue_credit.await_args_list:
        turn = call.args[0]
        out.append((turn.conversation_id, turn.turn_index, turn.x_correlation_id))
    return out


@pytest.mark.component_integration
class TestTrajectoryKSurvivesPhaseBoundary:
    """Same source, two strategies, identical k_i values; PROFILING resumes at"""

    @pytest.mark.asyncio
    async def test_trajectory_k_observable_identically_in_both_phases(self):
        source = _make_real_source(
            num_traces=8, turns_per_trace=10, concurrency=4, seed=12345
        )
        trajectories_before_warmup = [
            (trajectory.conversation_id, trajectory.start_turn_index)
            for trajectory in source.trajectories
        ]

        warmup_strategy, warmup_issuer, _ = _make_strategy(
            phase=CreditPhase.WARMUP, source=source
        )
        await warmup_strategy.setup_phase()
        await warmup_strategy.execute_phase()

        warmup_dispatched = {
            (cid, idx) for cid, idx, _ in _capture_dispatched_turns(warmup_issuer)
        }
        warmup_correlations = {
            cid: xcorr for cid, _, xcorr in _capture_dispatched_turns(warmup_issuer)
        }
        assert warmup_dispatched == set(trajectories_before_warmup), (
            "WARMUP must dispatch each trajectory at exactly its sampled k_i"
        )

        trajectories_after_warmup = [
            (trajectory.conversation_id, trajectory.start_turn_index)
            for trajectory in source.trajectories
        ]
        assert trajectories_after_warmup == trajectories_before_warmup

        profiling_strategy, profiling_issuer, _ = _make_strategy(
            phase=CreditPhase.PROFILING, source=source
        )
        await profiling_strategy.setup_phase()
        await profiling_strategy.execute_phase()

        profiling_indices = {
            (cid, idx) for cid, idx, _ in _capture_dispatched_turns(profiling_issuer)
        }
        profiling_correlations = {
            cid: xcorr for cid, _, xcorr in _capture_dispatched_turns(profiling_issuer)
        }
        expected = {(cid, k + 1) for cid, k in trajectories_before_warmup}
        assert profiling_indices == expected, (
            "PROFILING must resume each trajectory at k_i + 1 (k_i unchanged)"
        )
        assert profiling_correlations == warmup_correlations, (
            "PROFILING continuation must preserve each warmed trajectory's "
            "x_correlation_id"
        )


@pytest.mark.component_integration
class TestWarmupGraceExceedsEstimate:
    """A slow server forces WARMUP to run longer than the initial duration"""

    @pytest.mark.asyncio
    async def test_profiling_starts_cleanly_after_extended_warmup(self):
        source = _make_real_source(
            num_traces=6, turns_per_trace=8, concurrency=3, seed=777
        )
        snapshot = list(source.trajectories)

        warmup_strategy, warmup_issuer, _ = _make_strategy(
            phase=CreditPhase.WARMUP, source=source
        )
        await warmup_strategy.setup_phase()
        await warmup_strategy.execute_phase()

        for trajectory in source.trajectories:
            ret = _make_credit(
                conversation_id=trajectory.conversation_id,
                turn_index=trajectory.start_turn_index,
                num_turns=10,
                phase=CreditPhase.WARMUP,
            )
            await warmup_strategy.handle_credit_return(ret)

        warmup_dispatched_after = _capture_dispatched_turns(warmup_issuer)
        assert len(warmup_dispatched_after) == len(snapshot), (
            "WARMUP must not issue follow-up credits even after extended runtime"
        )

        warmup_strategy.report_warmup_failures()

        assert source.trajectories == snapshot

        profiling_strategy, profiling_issuer, _ = _make_strategy(
            phase=CreditPhase.PROFILING, source=source
        )
        await profiling_strategy.setup_phase()
        await profiling_strategy.execute_phase()

        resumed = {
            (cid, idx) for cid, idx, _ in _capture_dispatched_turns(profiling_issuer)
        }
        assert resumed == {
            (m.conversation_id, m.start_turn_index + 1) for m in snapshot
        }


@pytest.mark.component_integration
class TestMultiMachineDeterminism:
    """Same dataset + same seed -> same trajectory, same k_i values, and same"""

    @pytest.mark.asyncio
    async def test_two_independent_sources_yield_identical_trajectories_and_recycle_order(
        self,
    ):
        seed = 13_579
        source_a = _make_real_source(
            num_traces=12, turns_per_trace=10, concurrency=5, seed=seed
        )
        source_b = _make_real_source(
            num_traces=12, turns_per_trace=10, concurrency=5, seed=seed
        )

        trajectories_a = [
            (m.conversation_id, m.start_turn_index) for m in source_a.trajectories
        ]
        trajectories_b = [
            (m.conversation_id, m.start_turn_index) for m in source_b.trajectories
        ]
        assert trajectories_a == trajectories_b
        assert len(trajectories_a) == 5

        async def _capture_recycle_order(
            source: TrajectorySource,
        ) -> tuple[list[str], list[int]]:
            recycled: list[tuple[str, int]] = []

            async def capture(turn):
                recycled.append((turn.conversation_id, turn.turn_index))
                return True

            issuer = AsyncMock()
            issuer.issue_credit.side_effect = capture
            strat, _, _ = _make_strategy(
                phase=CreditPhase.PROFILING, source=source, issuer=issuer
            )
            await strat.setup_phase()
            await strat.execute_phase()

            lane_to_corr = {
                lane: corr for corr, lane in strat._correlation_to_lane.items()
            }
            recycled.clear()
            for lane in range(len(source.trajectories)):
                corr = lane_to_corr[lane]
                cid = source.trajectories[lane].conversation_id
                n = len(source._metadata_lookup[cid].turns)
                await strat.handle_credit_return(
                    _make_credit(
                        conversation_id=cid,
                        turn_index=n - 1,
                        num_turns=n,
                        x_correlation_id=corr,
                    )
                )
            return [c for c, _ in recycled], [idx for _, idx in recycled]

        order_a, turns_a = await _capture_recycle_order(source_a)
        order_b, turns_b = await _capture_recycle_order(source_b)

        assert order_a == order_b, (
            "Two independent runs with the same dataset + seed must produce "
            "identical recycle dispatch order"
        )
        assert len(order_a) == 5, "one recycle per finalized lane"
        assert all(idx == 0 for idx in turns_a), (
            "every recycled dispatch must start at turn 0"
        )

    @pytest.mark.asyncio
    async def test_different_seeds_produce_distinguishable_trajectories(self):
        """Sanity check: determinism is seed-driven, not constant. Without"""
        source_a = _make_real_source(
            num_traces=5, turns_per_trace=20, concurrency=5, seed=1
        )
        source_b = _make_real_source(
            num_traces=5, turns_per_trace=20, concurrency=5, seed=999_999
        )
        trajectories_a = [
            (m.conversation_id, m.start_turn_index) for m in source_a.trajectories
        ]
        trajectories_b = [
            (m.conversation_id, m.start_turn_index) for m in source_b.trajectories
        ]

        ids_a = [cid for cid, _ in trajectories_a]
        ids_b = [cid for cid, _ in trajectories_b]
        assert ids_a == ids_b, "sampler is sequential -- id order should match"
        assert trajectories_a != trajectories_b, (
            "Different seeds must yield distinguishable k_i assignments "
            "(otherwise the determinism test above is vacuous)"
        )
