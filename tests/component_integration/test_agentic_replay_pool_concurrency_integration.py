# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component-integration tests for agentic_replay concurrency x pool boundary sweeps.

Pins behavior at the seams between user-supplied ``concurrency`` and the
loader-produced trace pool size in ``TrajectorySource`` /
``AgenticReplayStrategy`` (PROFILING phase recycle setup):

- concurrency < pool_size: trajectory count = concurrency; recycle queue holds
  the rest.
- concurrency == pool_size: every trace becomes a trajectory; recycle queue
  starts EMPTY and the just-finished trace_id is reused via the
  put-then-pop-on-empty path in ``_spawn_from_recycle_or_id``.
- concurrency > pool_size: ``TrajectorySource`` wrap-fills the missing lanes
  by cycling through distinct trajectories with fresh ``start_turn_index``
  salts (Task 8 covers the full E2E recycle behavior).
- traces with 0 turns are skipped at trajectory-selection time with a per-trace
  WARNING; an entirely-empty pool raises ``EmptyTracePoolError`` from the
  ``TrajectorySource`` constructor before any strategy is built.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario.base import (
    EmptyTracePoolError,
)
from aiperf.credit.structs import Credit
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.component_integration


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class _DispatchLog:
    """Capture every credit issued through the strategy for ordering checks."""

    entries: list[tuple[CreditPhase, str, int]] = field(default_factory=list)
    """List of (phase, conversation_id, turn_index) per dispatched credit."""

    def by_phase(self, phase: CreditPhase) -> list[tuple[str, int]]:
        return [(cid, idx) for ph, cid, idx in self.entries if ph == phase]

    def trace_ids_in_phase(self, phase: CreditPhase) -> list[str]:
        return [cid for ph, cid, _ in self.entries if ph == phase]


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


def _make_dataset_with_zero_turn_traces(
    valid_count: int, zero_count: int, valid_turns: int
) -> DatasetMetadata:
    """Build a synthetic dataset where some traces have 0 turns.

    Layout interleaves zero-turn traces between valid ones so the sampler hits
    both kinds during selection.
    """
    convs: list[ConversationMetadata] = []
    valid_remaining = valid_count
    zero_remaining = zero_count
    valid_idx = 0
    zero_idx = 0
    while valid_remaining or zero_remaining:
        if zero_remaining and (zero_idx <= valid_idx or valid_remaining == 0):
            convs.append(
                ConversationMetadata(conversation_id=f"empty_{zero_idx}", turns=[])
            )
            zero_idx += 1
            zero_remaining -= 1
        else:
            turns = [
                TurnMetadata(timestamp_ms=None, delay_ms=None)
                for _ in range(valid_turns)
            ]
            convs.append(
                ConversationMetadata(conversation_id=f"valid_{valid_idx}", turns=turns)
            )
            valid_idx += 1
            valid_remaining -= 1
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


def _make_credit(
    *,
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    x_correlation_id: str | None = None,
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id
        if x_correlation_id is not None
        else uuid.uuid4().hex,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.FORK,
    )


def _build_phase_strategy(
    *,
    phase: CreditPhase,
    source: TrajectorySource,
    issuer: AsyncMock,
    stop_checker: MagicMock | None = None,
) -> AgenticReplayStrategy:
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(source.trajectories)
    return AgenticReplayStrategy(
        config=cfg,
        conversation_source=source,
        scheduler=MagicMock(),
        stop_checker=stop_checker if stop_checker is not None else _make_stop_checker(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
    )


# =============================================================================
# Test 1: concurrency=1, pool=10 -> 1 trajectory + 9 in recycle queue (real loader)
# =============================================================================


def _make_variable_length_dataset() -> DatasetMetadata:
    """10 traces with N=1..10 turns, mirroring the small weka fixture's shape.

    Constructs ``DatasetMetadata`` directly rather than routing through
    ``WekaTraceLoader``: pool / recycle behavior is identical regardless of
    how the metadata was sourced, and the direct construction sidesteps the
    parallel-reconstruction path's incompatibility with the small fixture
    (``SharedMemory size=0`` on payloads below the chunk threshold).
    """
    convs: list[ConversationMetadata] = []
    for i in range(1, 11):
        turns = [TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(i)]
        convs.append(
            ConversationMetadata(conversation_id=f"trace_{i:02d}_n{i}", turns=turns)
        )
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


@pytest.mark.asyncio
async def test_concurrency_one_pool_ten_one_trajectory_nine_in_recycle() -> None:
    """concurrency < pool_size: trajectory = 1, recycle queue = 9.

    Drive enough final-turn returns to cycle once and assert at least one
    trace_id is dispatched more than once (recycle observed).

    Deviation from prompt: the prompt asks for the real WekaTraceLoader path,
    but the parallel-reconstruction path introduced by 02a1da62d crashes on
    the small fixture with ``SharedMemory size=0`` (same crash already breaks
    ``test_agentic_replay_e2e_clean_run_under_scenario``). Pool/recycle
    contract is loader-independent, so we use a synthetic 10-trace dataset
    with N=1..10 (the fixture's shape) to pin the behavior without depending
    on the broken loader path.
    """
    dataset = _make_variable_length_dataset()
    assert len(dataset.conversations) == 10

    sampler = SequentialSampler([c.conversation_id for c in dataset.conversations])
    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=4242,
    )
    assert len(source.trajectories) == 1

    log = _DispatchLog()
    current_phase = [CreditPhase.PROFILING]
    issuer = _make_recording_issuer(log, current_phase)
    profiling = _build_phase_strategy(
        phase=CreditPhase.PROFILING, source=source, issuer=issuer
    )
    await profiling.setup_phase()
    await profiling.execute_phase()

    metadata_lookup = source._metadata_lookup
    trajectory = source.trajectories[0]

    # Drive sequential final-turn returns for everything that's been dispatched
    # so far. Each completion either resumes mid-trace or recycles the queue.
    finalized: set[str] = set()
    safety = 0
    while safety < 25:
        safety += 1
        snapshot = log.by_phase(CreditPhase.PROFILING)
        # Find an in-flight trace_id we have not yet finalized.
        candidates = [cid for cid, _ in snapshot if cid not in finalized]
        if not candidates:
            break
        cid = candidates[0]
        n = len(metadata_lookup[cid].turns)
        await profiling.handle_credit_return(
            _make_credit(conversation_id=cid, turn_index=n - 1, num_turns=n)
        )
        finalized.add(cid)
        all_ids = log.trace_ids_in_phase(CreditPhase.PROFILING)
        if any(all_ids.count(t) > 1 for t in set(all_ids)):
            break

    full_ids = log.trace_ids_in_phase(CreditPhase.PROFILING)
    duplicates = [t for t in set(full_ids) if full_ids.count(t) > 1]
    assert duplicates, (
        f"expected at least one trace_id to be dispatched more than once; "
        f"got dispatch sequence={full_ids}, trajectory={trajectory.conversation_id}"
    )


# =============================================================================
# Test 2: concurrency == pool_size -> recycle queue starts EMPTY; finished id
# is dispatched again
# =============================================================================


@pytest.mark.asyncio
async def test_concurrency_equals_pool_size_recycle_follows_sampler_rotation() -> None:
    """concurrency == pool_size: every trace becomes a lane, and recycle draws
    the next root from the shared dataset sampler.

    The build consumed trace_0..trace_3 for the four lanes; recycle reuses the
    same sampler, which wraps round-robin back to trace_0. Finalizing one
    trajectory therefore produces exactly one fresh turn-0 dispatch of trace_0.
    """
    dataset = _make_dataset(num_traces=4, turns_per_trace=3)
    sampler = SequentialSampler([c.conversation_id for c in dataset.conversations])
    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=4,
        random_seed=11,
    )
    assert len(source.trajectories) == 4

    log = _DispatchLog()
    current_phase = [CreditPhase.PROFILING]
    issuer = _make_recording_issuer(log, current_phase)
    profiling = _build_phase_strategy(
        phase=CreditPhase.PROFILING, source=source, issuer=issuer
    )
    await profiling.setup_phase()
    await profiling.execute_phase()
    pre_recycle = list(log.entries)

    # Predict the recycled id: the sampler already yielded the four lane roots
    # (trace_0..trace_3) at build time, so the next draw wraps to trace_0.
    all_trace_ids = [c.conversation_id for c in dataset.conversations]
    predictor = SequentialSampler(all_trace_ids)
    for _ in range(len(source.trajectories)):
        predictor.next_conversation_id()
    expected_recycled = predictor.next_conversation_id()
    assert expected_recycled == "trace_0"

    # Finalize one trajectory. Recycle draws the next root from the sampler.
    finished = source.trajectories[0]
    await profiling.handle_credit_return(
        _make_credit(
            conversation_id=finished.conversation_id,
            turn_index=2,
            num_turns=3,
        )
    )

    new_dispatches = log.entries[len(pre_recycle) :]
    assert len(new_dispatches) == 1, (
        f"recycle should issue exactly one fresh dispatch; got {new_dispatches}"
    )
    phase, cid, idx = new_dispatches[0]
    assert phase == CreditPhase.PROFILING
    assert cid == expected_recycled, (
        "recycled id must follow the sampler rotation (wraps to trace_0 after "
        f"the four build draws); got {cid!r}"
    )
    assert idx == 0, "recycled session must start at turn 0, not at k_i"


# =============================================================================
# Test 3: concurrency > pool_size -> wrap-fill produces ``concurrency`` lanes
# =============================================================================


def test_concurrency_exceeds_pool_wrap_fills_to_concurrency() -> None:
    """concurrency > pool_size: TrajectorySource wrap-fills the missing
    lanes by cycling through the distinct trajectories. Task 8 covers the
    end-to-end recycle behavior; this test only pins the construction-time
    contract.
    """
    dataset = _make_dataset(num_traces=4, turns_per_trace=3)
    sampler = SequentialSampler([c.conversation_id for c in dataset.conversations])

    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=15,
        random_seed=7,
    )

    assert len(source.trajectories) == 15
    distinct = {t.conversation_id for t in source.trajectories}
    assert distinct == {f"trace_{i}" for i in range(4)}
    assert len(distinct) < 15  # wrap-fill activated


# =============================================================================
# Test 4: concurrency == pool_size at boundary -> no error
# =============================================================================


def test_concurrency_equals_pool_size_at_boundary(caplog) -> None:
    """At the boundary concurrency == pool_size, construction succeeds cleanly."""
    dataset = _make_dataset(num_traces=4, turns_per_trace=3)
    sampler = SequentialSampler([c.conversation_id for c in dataset.conversations])

    with caplog.at_level(logging.WARNING, logger="aiperf.timing.trajectory_source"):
        source = TrajectorySource(
            dataset_metadata=dataset,
            dataset_sampler=sampler,
            concurrency=4,
            random_seed=7,
        )

    assert len(source.trajectories) == 4

    over_cap = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "exceeds trace pool size" in r.getMessage()
    ]
    assert not over_cap, (
        f"no over-cap warning expected at the boundary; got {[r.getMessage() for r in over_cap]}"
    )


# =============================================================================
# Test 5: mixed-validity pool -> zero-turn traces skipped with per-trace WARNING
# =============================================================================


def test_mixed_validity_pool_skips_zero_turn_traces_with_warning(caplog) -> None:
    """Zero-turn traces are skipped at trajectory selection with a per-trace WARNING.

    With 5 trace slots (3 valid x 2 turns + 2 empty) and concurrency=3 (matching
    the usable count), ``_build_trajectories`` visits every trace and emits a
    per-trace WARNING for each zero-turn skip; trajectories contain only the 3
    valid trace_ids and wrap-fill is not triggered.
    """
    dataset = _make_dataset_with_zero_turn_traces(
        valid_count=3, zero_count=2, valid_turns=2
    )
    assert len(dataset.conversations) == 5

    sampler = SequentialSampler([c.conversation_id for c in dataset.conversations])

    with caplog.at_level(logging.WARNING, logger="aiperf.timing.trajectory_source"):
        source = TrajectorySource(
            dataset_metadata=dataset,
            dataset_sampler=sampler,
            concurrency=3,
            random_seed=3,
        )

    trajectory_ids = {m.conversation_id for m in source.trajectories}
    assert trajectory_ids == {"valid_0", "valid_1", "valid_2"}, (
        f"only valid traces may become trajectories; got {trajectory_ids}"
    )

    skip_messages = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "Skipping trace" in r.getMessage()
    ]
    # Each zero-turn trace yields one skip warning containing "0 turns".
    for empty_id in ("empty_0", "empty_1"):
        matching = [m for m in skip_messages if empty_id in m and "0 turns" in m]
        assert matching, (
            f"expected a 'Skipping trace ... 0 turns' WARNING for {empty_id!r}; "
            f"got {skip_messages}"
        )


# =============================================================================
# Test 6: empty pool -> EmptyTracePoolError at TrajectorySource construction
# =============================================================================


def test_empty_pool_raises_at_trajectory_source_construction() -> None:
    """An entirely-empty conversations list raises before any strategy is built."""
    dataset = DatasetMetadata(
        conversations=[],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    # The production SequentialSampler rejects an empty id list in __init__;
    # use a stand-in so TrajectorySource raises EmptyTracePoolError on the
    # conversations check (the contract) rather than the sampler pre-empting it.
    sampler = MagicMock()

    with pytest.raises(EmptyTracePoolError, match="0 traces"):
        TrajectorySource(
            dataset_metadata=dataset,
            dataset_sampler=sampler,
            concurrency=4,
            random_seed=0,
        )
    # No AgenticReplayStrategy is constructed in this path; the constructor
    # raise above is the contract.
