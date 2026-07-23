# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extended adversarial unit tests for ``TrajectorySource`` and ``SampledSession.build_turn_at_index``."""

from __future__ import annotations

import pytest

from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.common.scenario.base import (
    EmptyTracePoolError,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.conversation_source import SampledSession
from aiperf.timing.trajectory_source import (
    TrajectorySource,
    _seed_for_trace,
)

# Helpers


def _make_dataset(turns_per_trace_by_id: dict[str, int]) -> DatasetMetadata:
    """Build a real DatasetMetadata where each conversation has the given turn count."""
    convs: list[ConversationMetadata] = []
    for cid, n in turns_per_trace_by_id.items():
        turns = [TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(n)]
        convs.append(ConversationMetadata(conversation_id=cid, turns=turns))
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def _uniform_dataset(num_traces: int, turns_per_trace: int) -> DatasetMetadata:
    return _make_dataset({f"trace_{i}": turns_per_trace for i in range(num_traces)})


class _Sampler:
    """Wrapping stub sampler (like the production SequentialSampler): cycles
    through the provided ids indefinitely; raises StopIteration only when the
    pool is empty."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = list(ids)
        self._i = 0

    def next_conversation_id(self) -> str:
        if not self._ids:
            raise StopIteration
        cid = self._ids[self._i % len(self._ids)]
        self._i += 1
        return cid


# concurrency=0 -> _target_size=0 -> empty trajectories -> EmptyTracePoolError


def test_concurrency_zero_yields_empty_trajectories_then_raises() -> None:
    """concurrency=0 makes ``_target_size`` 0; the build loop never runs and
    the empty-trajectory guard at the end of ``__init__`` fires."""
    ds = _uniform_dataset(num_traces=5, turns_per_trace=4)
    sampler = _Sampler([c.conversation_id for c in ds.conversations])

    with pytest.raises(EmptyTracePoolError):
        TrajectorySource(
            dataset_metadata=ds,
            dataset_sampler=sampler,
            concurrency=0,
            random_seed=1,
        )


# Mixed valid + invalid traces: 0-turn ones are skipped, valid ones survive


def test_mixed_valid_and_invalid_traces_skips_zero_turn_traces() -> None:
    """Traces 1 and 3 have 0 turns; the trajectory list must exclude them.

    Concurrency 3 matches the 3 valid traces, so the run is accepted; the
    zero-turn skip path is exercised inside ``_build_trajectories`` and
    wrap-fill is not triggered.
    """
    ds = _make_dataset(
        {
            "trace_0": 4,
            "trace_1": 0,
            "trace_2": 4,
            "trace_3": 0,
            "trace_4": 4,
        }
    )
    sampler = _Sampler([c.conversation_id for c in ds.conversations])

    src = TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=sampler,
        concurrency=3,
        random_seed=99,
    )

    cids = {t.conversation_id for t in src.trajectories}
    assert "trace_1" not in cids
    assert "trace_3" not in cids
    assert cids == {"trace_0", "trace_2", "trace_4"}


def test_mixed_valid_and_invalid_traces_concurrency_over_usable_wrap_fills() -> None:
    """When zero-turn skips push usable trajectories below concurrency,
    wrap-fill activates so the run still honours ``--concurrency``. Pool=5
    (3 valid + 2 zero-turn), concurrency=5 -> 3 distinct trajectories
    fanned out to 5 lanes.
    """
    ds = _make_dataset(
        {
            "trace_0": 4,
            "trace_1": 0,
            "trace_2": 4,
            "trace_3": 0,
            "trace_4": 4,
        }
    )
    sampler = _Sampler([c.conversation_id for c in ds.conversations])

    src = TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=sampler,
        concurrency=5,
        random_seed=99,
    )

    assert len(src.trajectories) == 5
    distinct = {t.conversation_id for t in src.trajectories}
    assert distinct == {"trace_0", "trace_2", "trace_4"}
    assert len(distinct) < 5  # wrap-fill activated


# Seed sensitivity: different seeds drive at least one differing k_i


def test_different_seeds_can_yield_different_k_i() -> None:
    """Different base seeds for the same dataset must drive at least one differing k_i.

    With N=10 -> k_max=7, the k-space has 8 values. Across 5 traces the
    chance of full collision under two different seeds is vanishingly small;
    pinning ANY difference is robust.
    """
    ds = _uniform_dataset(num_traces=5, turns_per_trace=10)
    ids = [c.conversation_id for c in ds.conversations]

    src_a = TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=_Sampler(list(ids)),
        concurrency=5,
        random_seed=1,
    )
    src_b = TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=_Sampler(list(ids)),
        concurrency=5,
        random_seed=2,
    )

    by_cid_a = {t.conversation_id: t.start_turn_index for t in src_a.trajectories}
    by_cid_b = {t.conversation_id: t.start_turn_index for t in src_b.trajectories}
    differing = [cid for cid in by_cid_a if by_cid_a[cid] != by_cid_b.get(cid)]
    assert differing, (
        f"Expected at least one k_i to differ across seeds 1 and 2; got "
        f"{by_cid_a} vs {by_cid_b}"
    )


# _seed_for_trace independence across distinct trace_ids


def test_seed_for_trace_independence_across_traces() -> None:
    """SHA-256-derived per-trace seeds must be distinct across distinct trace_ids."""
    base_seed = 42
    trace_ids = [f"trace_{i}" for i in range(10)]
    seeds = [_seed_for_trace(base_seed, tid) for tid in trace_ids]
    assert len(set(seeds)) == len(seeds), (
        f"Expected all per-trace seeds distinct; got duplicates in {seeds}"
    )


# session_for: persistent trajectory correlation_id when no override


def test_session_for_reuses_trajectory_correlation_id_per_call() -> None:
    """``session_for`` must preserve lane identity when no override is passed."""
    ds = _make_dataset({"trace_0": 4})
    sampler = _Sampler(["trace_0"])
    src = TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=11,
    )
    trajectory = src.trajectories[0]

    s1 = src.session_for(trajectory)
    s2 = src.session_for(trajectory)

    assert s1.x_correlation_id == trajectory.x_correlation_id
    assert s2.x_correlation_id == trajectory.x_correlation_id
    assert s1.start_turn_index == trajectory.start_turn_index
    assert s2.start_turn_index == trajectory.start_turn_index


# session_for: explicit x_correlation_id used verbatim


def test_session_for_accepts_explicit_correlation_id() -> None:
    """Explicit ``x_correlation_id`` is used verbatim, no UUID minting."""
    ds = _make_dataset({"trace_0": 4})
    sampler = _Sampler(["trace_0"])
    src = TrajectorySource(
        dataset_metadata=ds,
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=22,
    )
    trajectory = src.trajectories[0]

    session = src.session_for(trajectory, x_correlation_id="my-fixed-id")

    assert session.x_correlation_id == "my-fixed-id"


# SampledSession.build_turn_at_index: negative index rejected


def test_build_turn_at_index_negative_raises_index_error() -> None:
    """Negative indices must be rejected with a clear out-of-range message."""
    meta = ConversationMetadata(
        conversation_id="trace_0",
        turns=[TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(3)],
    )
    session = SampledSession(
        conversation_id="trace_0",
        metadata=meta,
        x_correlation_id="xcorr",
    )

    with pytest.raises(IndexError, match="out of range"):
        session.build_turn_at_index(-1)


# SampledSession.build_turn_at_index: index at or beyond length rejected


def test_build_turn_at_index_at_or_beyond_length_raises() -> None:
    """Indices at len(turns) and beyond must raise."""
    meta = ConversationMetadata(
        conversation_id="trace_0",
        turns=[TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(3)],
    )
    session = SampledSession(
        conversation_id="trace_0",
        metadata=meta,
        x_correlation_id="xcorr",
    )

    with pytest.raises(IndexError, match="out of range"):
        session.build_turn_at_index(3)
    with pytest.raises(IndexError, match="out of range"):
        session.build_turn_at_index(99)


# SampledSession.build_turn_at_index: first and last in-range indices succeed


def test_build_turn_at_index_first_and_last_succeed() -> None:
    """First (0) and last (len-1) turn indices both produce valid TurnToSend."""
    meta = ConversationMetadata(
        conversation_id="trace_0",
        turns=[TurnMetadata(timestamp_ms=None, delay_ms=None) for _ in range(3)],
    )
    session = SampledSession(
        conversation_id="trace_0",
        metadata=meta,
        x_correlation_id="xcorr",
    )

    first = session.build_turn_at_index(0)
    assert first.turn_index == 0
    assert first.num_turns == 3

    last = session.build_turn_at_index(2)
    assert last.turn_index == 2
    assert last.num_turns == 3
