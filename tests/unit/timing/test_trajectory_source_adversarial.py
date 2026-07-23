# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial unit tests for TrajectorySource (spec §8.4.2)."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from aiperf.common.scenario.base import (
    EmptyTracePoolError,
)
from aiperf.timing.trajectory_source import TrajectorySource
from tests.unit.timing._shared_helpers import _make_dataset_metadata


def _sampler_for(ids: list[str]) -> MagicMock:
    """A wrapping sampler (like the production SequentialSampler): cycles over"""
    import itertools

    sampler = MagicMock()
    if not ids:
        sampler.next_conversation_id.side_effect = StopIteration
    else:
        cycle = itertools.cycle(ids)
        sampler.next_conversation_id.side_effect = lambda: next(cycle)
    return sampler


def test_pool_one_concurrency_ten_repeats_single_trace_across_ten_lanes(caplog):
    """concurrency > pool: the wrapping sampler hands the single trace to all"""
    md = _make_dataset_metadata({"only": 5})
    sampler = _sampler_for(["only"])

    with caplog.at_level(logging.INFO, logger="aiperf.timing.trajectory_source"):
        src = TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=10,
            random_seed=42,
        )

    assert len(src.trajectories) == 10
    distinct_cids = {t.conversation_id for t in src.trajectories}
    assert distinct_cids == {"only"}
    reuse_logs = [
        r.getMessage() for r in caplog.records if "distinct traces" in r.getMessage()
    ]
    assert reuse_logs, "expected an INFO log about traces repeating across lanes"


def test_empty_pool_raises_at_construction():
    md = _make_dataset_metadata({})
    sampler = _sampler_for([])

    with pytest.raises(EmptyTracePoolError):
        TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=4,
            random_seed=1,
        )


def test_zero_turn_trace_skipped_then_pool_exhaustion_raises():
    md = _make_dataset_metadata({"empty_a": 0, "empty_b": 0, "empty_c": 0})
    sampler = _sampler_for(["empty_a", "empty_b", "empty_c"])

    with pytest.raises(EmptyTracePoolError):
        TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=2,
            random_seed=1,
        )


def test_single_turn_trace_skipped_with_warning_deterministically(caplog):
    """n=1 traces are rejected at trajectory selection (no profile turn after"""
    md = _make_dataset_metadata({"only": 1})
    sampler = _sampler_for(["only"])

    with caplog.at_level(logging.WARNING), pytest.raises(EmptyTracePoolError):
        TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=1,
            random_seed=12345,
        )
    assert any("Skipping trace" in r.getMessage() for r in caplog.records)


def test_two_turn_trace_k_i_is_zero_for_all_seeds():
    """N=2 forces k_i=0 unconditionally (only k_i=0 leaves a profile turn at"""
    md = _make_dataset_metadata({"t0": 2})
    for seed in (0, 6, 42, 123456789, (2**63) - 1):
        src = TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=_sampler_for(["t0"]),
            concurrency=1,
            random_seed=seed,
        )
        assert src.trajectories[0].start_turn_index == 0, (
            f"seed={seed} produced k_i={src.trajectories[0].start_turn_index} (expected 0)"
        )


def test_trajectories_follow_sampler_order_including_repeats():
    """Trace selection is the sampler's job: trajectories mirror the sampler's"""
    md = _make_dataset_metadata({"a": 5, "b": 5, "c": 5})
    sampler = _sampler_for(["a", "a", "b", "c"])

    src = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=3,
        random_seed=42,
    )

    cids = [t.conversation_id for t in src.trajectories]
    assert cids == ["a", "a", "b"]


def test_same_seed_two_independent_constructions_yield_identical_trajectories():
    md = _make_dataset_metadata({f"t{i}": 10 for i in range(4)})
    ids = [f"t{i}" for i in range(4)]

    s1 = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=_sampler_for(list(ids)),
        concurrency=4,
        random_seed=123456789,
    )
    s2 = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=_sampler_for(list(ids)),
        concurrency=4,
        random_seed=123456789,
    )

    k1 = [(t.conversation_id, t.start_turn_index) for t in s1.trajectories]
    k2 = [(t.conversation_id, t.start_turn_index) for t in s2.trajectories]
    assert k1 == k2


def test_seed_zero_is_accepted_and_deterministic():
    md = _make_dataset_metadata({f"t{i}": 10 for i in range(3)})
    ids = [f"t{i}" for i in range(3)]

    s1 = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=_sampler_for(list(ids)),
        concurrency=3,
        random_seed=0,
    )
    s2 = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=_sampler_for(list(ids)),
        concurrency=3,
        random_seed=0,
    )

    assert [(t.conversation_id, t.start_turn_index) for t in s1.trajectories] == [
        (t.conversation_id, t.start_turn_index) for t in s2.trajectories
    ]


def test_seed_max_int64_is_accepted_and_deterministic():
    max_int64 = (2**63) - 1
    md = _make_dataset_metadata({f"t{i}": 10 for i in range(3)})
    ids = [f"t{i}" for i in range(3)]

    s1 = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=_sampler_for(list(ids)),
        concurrency=3,
        random_seed=max_int64,
    )
    s2 = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=_sampler_for(list(ids)),
        concurrency=3,
        random_seed=max_int64,
    )

    assert [(t.conversation_id, t.start_turn_index) for t in s1.trajectories] == [
        (t.conversation_id, t.start_turn_index) for t in s2.trajectories
    ]


def test_per_trace_salting_yields_different_k_for_different_trace_ids():
    n = 20
    md = _make_dataset_metadata({f"t{i}": n for i in range(6)})
    ids = [f"t{i}" for i in range(6)]

    src = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=_sampler_for(list(ids)),
        concurrency=6,
        random_seed=42,
    )

    ks = {t.start_turn_index for t in src.trajectories}
    assert len(ks) > 1
