# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic_replay cross-phase state continuity adversarial tests.

Each test exercises the WARMUP -> PROFILING boundary by sharing a single
``TrajectorySource`` between two freshly-constructed ``AgenticReplayStrategy``
instances (one per phase), mirroring how the TimingManager / PhaseRunner wires
the two phases in production: the source is constructed ONCE at TimingManager
scope, and a fresh ``AgenticReplayStrategy`` is built per phase but reads from
the same source.

These tests stay at the strategy + source level (rather than spinning up a
full CLI run) because the invariants under test are about *state survival*
across the two-strategies/one-source seam. End-to-end CLI coverage of the
agentic_replay scenario lives elsewhere.

V2 PORT NOTE: built against ``aiperf.config``'s v2 surface. The source is
constructed through the PUBLIC ``TrajectorySource(...)`` constructor (not
``__new__``) so trajectory selection and per-trajectory ``k_i`` sampling run
through the production code path, with a deterministic ``SequentialSampler``
for reproducibility. The cross-phase legs covered here -- k_i + correlation-id
survival across the boundary, extended-warmup-grace clean resume, and
multi-source recycle determinism -- are NOT exercised by the unit-level
``test_agentic_replay_phase_adversarial.py`` /
``test_agentic_replay_recycle_adversarial.py`` (which use ``__new__`` sources
and a single strategy instance).
"""

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


def _make_real_source(
    num_traces: int,
    turns_per_trace: int,
    *,
    concurrency: int,
    seed: int,
) -> TrajectorySource:
    """Build a real TrajectorySource with deterministic sampling.

    Uses the public constructor (not __new__) so trajectory selection runs through
    the production code path; ``SequentialSampler`` provides reproducibility
    without leaning on dataset_sampler RNG state.
    """
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
    """Materialize all (conversation_id, turn_index, x_correlation_id) triples
    that were issued through the credit_issuer mock."""
    out: list[tuple[str, int, str]] = []
    for call in issuer.issue_credit.await_args_list:
        turn = call.args[0]
        out.append((turn.conversation_id, turn.turn_index, turn.x_correlation_id))
    return out


# =============================================================================
# Test 1: k_i + x_correlation_id survive the WARMUP -> PROFILING boundary
# =============================================================================


@pytest.mark.component_integration
class TestTrajectoryKSurvivesPhaseBoundary:
    """Same source, two strategies, identical k_i values; PROFILING resumes at
    k_i + 1 and preserves each warmed trajectory's x_correlation_id."""

    @pytest.mark.asyncio
    async def test_trajectory_k_observable_identically_in_both_phases(self):
        source = _make_real_source(
            num_traces=8, turns_per_trace=10, concurrency=4, seed=12345
        )
        trajectories_before_warmup = [
            (trajectory.conversation_id, trajectory.start_turn_index)
            for trajectory in source.trajectories
        ]

        # WARMUP phase -- observe what gets dispatched (each trajectory at k_i).
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

        # Trajectory list itself is unchanged after WARMUP execute.
        trajectories_after_warmup = [
            (trajectory.conversation_id, trajectory.start_turn_index)
            for trajectory in source.trajectories
        ]
        assert trajectories_after_warmup == trajectories_before_warmup

        # PROFILING phase -- same source, fresh strategy. Must resume each
        # trajectory at k_i + 1, proving k_i is still observable.
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


# =============================================================================
# Test 2: WARMUP grace-period extends beyond duration estimate
# =============================================================================


@pytest.mark.component_integration
class TestWarmupGraceExceedsEstimate:
    """A slow server forces WARMUP to run longer than the initial duration
    estimate; PROFILING must still start cleanly with the same trajectory
    state."""

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

        # Simulate a slow server: many credit returns flow through, none are
        # final, none trigger recycle (WARMUP recycle is a no-op anyway).
        # PhaseRunner's grace-period logic is the actual time-extender; from
        # the strategy's perspective the only requirement is "no state
        # change". Verify by issuing several no-op credit returns.
        for trajectory in source.trajectories:
            ret = _make_credit(
                conversation_id=trajectory.conversation_id,
                turn_index=trajectory.start_turn_index,
                num_turns=10,
                phase=CreditPhase.WARMUP,
            )
            await warmup_strategy.handle_credit_return(ret)

        # No follow-up credits issued by WARMUP regardless of how long it ran.
        warmup_dispatched_after = _capture_dispatched_turns(warmup_issuer)
        assert len(warmup_dispatched_after) == len(snapshot), (
            "WARMUP must not issue follow-up credits even after extended runtime"
        )

        # No terminal failures recorded -- report_warmup_failures must be silent.
        warmup_strategy.report_warmup_failures()  # must not raise

        # Trajectory is unchanged.
        assert source.trajectories == snapshot

        # PROFILING phase starts cleanly: setup + execute both succeed.
        profiling_strategy, profiling_issuer, _ = _make_strategy(
            phase=CreditPhase.PROFILING, source=source
        )
        await profiling_strategy.setup_phase()
        await profiling_strategy.execute_phase()

        # Each trajectory resumed at k_i + 1.
        resumed = {
            (cid, idx) for cid, idx, _ in _capture_dispatched_turns(profiling_issuer)
        }
        assert resumed == {
            (m.conversation_id, m.start_turn_index + 1) for m in snapshot
        }


# =============================================================================
# Test 3: Multi-machine determinism -- same dataset + seed -> identical state
# =============================================================================


@pytest.mark.component_integration
class TestMultiMachineDeterminism:
    """Same dataset + same seed -> same trajectory, same k_i values, and same
    recycle order across two independent PROFILING sources."""

    @pytest.mark.asyncio
    async def test_two_independent_sources_yield_identical_trajectories_and_recycle_order(
        self,
    ):
        seed = 13_579
        # Build two independent sources with byte-identical inputs.
        source_a = _make_real_source(
            num_traces=12, turns_per_trace=10, concurrency=5, seed=seed
        )
        source_b = _make_real_source(
            num_traces=12, turns_per_trace=10, concurrency=5, seed=seed
        )

        # Same trajectory assignment + same k_i per member.
        trajectories_a = [
            (m.conversation_id, m.start_turn_index) for m in source_a.trajectories
        ]
        trajectories_b = [
            (m.conversation_id, m.start_turn_index) for m in source_b.trajectories
        ]
        assert trajectories_a == trajectories_b
        assert len(trajectories_a) == 5

        # Same recycle order: drive identical final-turn-return sequences
        # through both PROFILING strategies and compare the recycled dispatch
        # ids. Recycle draws from each source's own deterministic
        # SequentialSampler (at the same position after the build), so the two
        # sequences must be byte-identical and each fresh dispatch starts at 0.
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

            # Finalize each lane in lane order; each final-turn return recycles
            # the lane into the next sampler root.
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
        """Sanity check: determinism is seed-driven, not constant. Without
        this check, ``test_two_independent_sources_yield_identical_trajectories_and_recycle_order``
        would also pass for a buggy implementation that always returns the
        same trajectory regardless of seed."""
        # Use a turn count where a seed difference will yield different k_i
        # for at least one trace (with k_max=floor(0.75*20) capped at n-2=18,
        # many possible values per trace, 5 traces -> overwhelmingly different
        # k_i sets).
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

        # Same conversation_ids (deterministic sequential sampler), but k_i
        # values differ for at least one trace.
        ids_a = [cid for cid, _ in trajectories_a]
        ids_b = [cid for cid, _ in trajectories_b]
        assert ids_a == ids_b, "sampler is sequential -- id order should match"
        assert trajectories_a != trajectories_b, (
            "Different seeds must yield distinguishable k_i assignments "
            "(otherwise the determinism test above is vacuous)"
        )
