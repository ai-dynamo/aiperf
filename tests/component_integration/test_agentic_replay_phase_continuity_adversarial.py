# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic_replay cross-phase state continuity adversarial tests.

Spec §8.4.7. Each test exercises the WARMUP -> PROFILING boundary by sharing
a single ``TrajectorySource`` between two freshly-constructed
``AgenticReplayStrategy`` instances (one per phase), mirroring how
``PhaseRunner`` wires the two phases in production.

These tests stay at the strategy + source level (rather than spinning up a
full CLI run) because the invariants under test are about *state survival*:
``TrajectorySource`` is constructed once at TimingManager scope, and
``AgenticReplayStrategy`` is constructed fresh per phase but reads from the
same source. End-to-end CLI coverage of the agentic_replay scenario lives in
the e2e test (``test_agentic_replay_e2e.py``).
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
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
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
    the production code path; ``_SequentialSampler`` provides reproducibility
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
# Test 1: k_i survives the WARMUP -> PROFILING boundary
# =============================================================================


@pytest.mark.component_integration
class TestTrajectoryKSurvivesPhaseBoundary:
    """Spec §8.4.7 test 1: same source, two strategies, identical k_i values."""

    @pytest.mark.asyncio
    async def test_trajectory_k_observable_identically_in_both_phases(self):
        source = _make_real_source(
            num_traces=8, turns_per_trace=10, concurrency=4, seed=12345
        )
        trajectories_before_warmup = [
            (trajectory.conversation_id, trajectory.start_turn_index)
            for trajectory in source.trajectories
        ]

        # WARMUP phase — observe what gets dispatched (each trajectory at k_i).
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

        # PROFILING phase — same source, fresh strategy. Must resume each
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
    """Spec §8.4.7 test 2: a slow server forces WARMUP to run longer than
    the initial duration estimate; PROFILING must still start cleanly with
    the same trajectory state."""

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

        # No terminal failures recorded — report_warmup_failures must be silent.
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
# Test 3: WARMUP aborts mid-trajectory -> PROFILING does not start, source cleans up
# =============================================================================


@pytest.mark.component_integration
class TestWarmupAbortMidTrajectoriesCleansUp:
    """Spec §8.4.7 test 3: a terminal warmup credit failure aborts the run.
    PROFILING never runs; the warmup-failure surface is observable
    (no leaked queue handles, no orphan credits)."""

    @pytest.mark.asyncio
    async def test_warmup_terminal_failure_blocks_profiling_and_cleans_source(self):
        source = _make_real_source(
            num_traces=5, turns_per_trace=6, concurrency=3, seed=42
        )
        original_trajectories = list(source.trajectories)
        assert len(original_trajectories) == 3

        warmup_strategy, _, _ = _make_strategy(phase=CreditPhase.WARMUP, source=source)
        await warmup_strategy.setup_phase()
        await warmup_strategy.execute_phase()

        # Simulate a single trajectory failing terminally.
        failed = original_trajectories[1]
        warmup_strategy.record_warmup_failure(failed.conversation_id)

        with pytest.raises(TrajectoryWarmupFailedError) as exc_info:
            warmup_strategy.report_warmup_failures()
        assert failed.conversation_id in exc_info.value.failed_trace_ids

        # Manually clear trajectories to simulate the PhaseRunner abort path
        # that drops trajectory state when WARMUP fails.
        source.trajectories = []

        # If a PROFILING strategy were ever (incorrectly) constructed after a
        # WARMUP abort, AgenticReplayStrategy.setup_phase must refuse to start
        # with an empty trajectory — no orphan credit dispatch, clear failure.
        leaked_strategy, leaked_issuer, _ = _make_strategy(
            phase=CreditPhase.PROFILING, source=source
        )
        with pytest.raises(RuntimeError, match="trajectories empty"):
            await leaked_strategy.setup_phase()
        assert leaked_issuer.issue_credit.await_count == 0, (
            "Empty-trajectory PROFILING must not issue credits (no orphan dispatch)"
        )


# =============================================================================
# Test 4: Recycled trajectory trace plays again at start_turn_index=0, not k_i
# =============================================================================


@pytest.mark.component_integration
class TestRecycledTrajectoryTracePlaysFromTurnZero:
    """Spec §8.4.7 test 4: when a trajectory trace finishes during PROFILING,
    its lane recycles into the next root drawn from the dataset sampler. The
    fresh play starts at turn 0 — not at k_i."""

    @pytest.mark.asyncio
    async def test_finished_trajectory_trace_recycled_to_tail_and_replayed_from_zero(
        self,
    ):
        # 2-trace pool, concurrency=1: the build consumed trace_0 for the lone
        # lane; recycle reuses the same sampler, so the next draw is trace_1.
        source = _make_real_source(
            num_traces=2, turns_per_trace=4, concurrency=1, seed=99
        )
        assert len(source.trajectories) == 1
        trajectory = source.trajectories[0]

        # Predict the recycled id via a parallel sampler advanced past the ids
        # consumed at build time (one per lane).
        all_ids = [c.conversation_id for c in source.dataset_metadata.conversations]
        predictor = SequentialSampler(all_ids)
        for _ in range(len(source.trajectories)):
            predictor.next_conversation_id()
        expected_recycled = predictor.next_conversation_id()

        captured: list[tuple[str, int, str]] = []

        async def capture(turn):
            captured.append(
                (turn.conversation_id, turn.turn_index, turn.x_correlation_id)
            )
            return True

        issuer = AsyncMock()
        issuer.issue_credit.side_effect = capture

        profiling_strategy, _, _ = _make_strategy(
            phase=CreditPhase.PROFILING, source=source, issuer=issuer
        )
        await profiling_strategy.setup_phase()

        # _execute_profiling registers the trajectory's correlation_id; we need
        # to dispatch the WARMUP-then-PROFILING resume path so the lane map is
        # populated before we send a final-turn credit return. Resume happens
        # inside execute_phase, which we deliberately invoke here so the
        # strategy mints the correlation_id we'll then echo back as final.
        await profiling_strategy.execute_phase()
        lane_to_correlation = {
            lane: cid for cid, lane in profiling_strategy._correlation_to_lane.items()
        }
        # Trajectory is at lane 0 (only trajectory).
        trajectory_xcorr = lane_to_correlation[0]

        # Trajectory finishes its last turn (final_turn=3 of 4).
        captured.clear()
        final_credit = _make_credit(
            conversation_id=trajectory.conversation_id,
            turn_index=3,
            num_turns=4,
            x_correlation_id=trajectory_xcorr,
        )
        await profiling_strategy.handle_credit_return(final_credit)

        # Exactly one new credit issued: the recycled root from the sampler
        # rotation, started at turn 0.
        assert len(captured) == 1
        recycled_cid, recycled_turn, _ = captured[0]
        assert recycled_cid == expected_recycled, (
            "Recycle must follow the sampler rotation (next root after the "
            f"build draw); expected {expected_recycled!r}, got {recycled_cid!r}"
        )
        assert recycled_turn == 0, (
            "Recycled session must start at turn 0, NOT at the original k_i"
        )

    @pytest.mark.asyncio
    async def test_same_trace_id_replays_at_turn_zero_when_picked_by_other_slot(self):
        """Drives two recycle cycles and asserts each fresh dispatch starts at
        turn_index=0 — byte-exact same trace, starting at turn 0 rather than at
        k_i — and that the recycled ids follow the dataset sampler's rotation.

        2-trace pool, concurrency=1: the build consumed trace_0 for the lone
        lane, so recycle reuses the same sampler from there: cycle 1 draws
        trace_1, cycle 2 wraps back to trace_0. Both fresh dispatches start at
        turn 0."""
        source = _make_real_source(
            num_traces=2, turns_per_trace=3, concurrency=1, seed=2024
        )
        trajectory = source.trajectories[0]

        # Predict the two recycle draws via a parallel sampler advanced past the
        # build draw (one per lane).
        all_ids = [c.conversation_id for c in source.dataset_metadata.conversations]
        predictor = SequentialSampler(all_ids)
        for _ in range(len(source.trajectories)):
            predictor.next_conversation_id()
        expected_cycle1 = predictor.next_conversation_id()
        expected_cycle2 = predictor.next_conversation_id()

        captured: list[tuple[str, int]] = []

        async def capture(turn):
            captured.append((turn.conversation_id, turn.turn_index))
            return True

        issuer = AsyncMock()
        issuer.issue_credit.side_effect = capture
        strategy, _, _ = _make_strategy(
            phase=CreditPhase.PROFILING, source=source, issuer=issuer
        )
        await strategy.setup_phase()
        await strategy.execute_phase()
        # After execute_phase, the trajectory's correlation_id is registered
        # at lane 0 (only trajectory).
        lane_to_correlation = {
            lane: cid for cid, lane in strategy._correlation_to_lane.items()
        }
        trajectory_xcorr = lane_to_correlation[0]

        # Cycle 1: trajectory finishes -> recycle draws the next sampler root.
        final_credit_trajectory = _make_credit(
            conversation_id=trajectory.conversation_id,
            turn_index=2,
            num_turns=3,
            x_correlation_id=trajectory_xcorr,
        )
        captured.clear()
        await strategy.handle_credit_return(final_credit_trajectory)
        assert captured == [(expected_cycle1, 0)], (
            "Cycle 1 recycle must follow the sampler rotation and start at "
            "turn 0, not at the original k_i"
        )

        # The recycled session was just registered at lane 0.
        lane_to_correlation = {
            lane: cid for cid, lane in strategy._correlation_to_lane.items()
        }
        replay_xcorr = lane_to_correlation[0]

        # Cycle 2: the recycled session finishes -> recycle draws the next
        # sampler root (the rotation wraps), again at turn 0.
        final_credit_replay = _make_credit(
            conversation_id=expected_cycle1,
            turn_index=2,
            num_turns=3,
            x_correlation_id=replay_xcorr,
        )
        captured.clear()
        await strategy.handle_credit_return(final_credit_replay)

        assert captured == [(expected_cycle2, 0)], (
            "Cycle 2 recycle must follow the sampler rotation (wraps) and the "
            "fresh play must start at turn 0, not at the original k_i"
        )


# =============================================================================
# Test 5: Multi-machine determinism — same dataset + seed -> identical state
# =============================================================================


@pytest.mark.component_integration
class TestMultiMachineDeterminism:
    """Spec §8.4.7 test 5: same dataset + same seed -> same trajectory, same
    k_i values, and same recycle order across two independent runs."""

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
        async def _capture_recycle_order(source: TrajectorySource) -> list[str]:
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
        # for at least one trace (with k_max=floor(0.7*20)=14, 15 possible
        # values per trace, 5 traces -> overwhelmingly different k_i sets).
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
        assert ids_a == ids_b, "sampler is sequential — id order should match"
        assert trajectories_a != trajectories_b, (
            "Different seeds must yield distinguishable k_i assignments "
            "(otherwise the determinism test above is vacuous)"
        )
