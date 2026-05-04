# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import pytest

from aiperf.common.scenario.base import (
    EmptyTracePoolError,
)
from aiperf.timing.trajectory_source import TrajectorySource


def _make_dataset_metadata(turn_counts_by_id: dict[str, int]):
    md = MagicMock()
    convs = []
    for cid, n in turn_counts_by_id.items():
        c = MagicMock()
        c.conversation_id = cid
        c.turns = [MagicMock(has_forks=False) for _ in range(n)]
        convs.append(c)
    md.conversations = convs
    return md


def test_trajectory_count_matches_min_concurrency_and_pool():
    md = _make_dataset_metadata({"a": 5, "b": 5, "c": 5, "d": 5})
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = ["a", "b", "c", "d"]

    src = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=2,
        random_seed=42,
    )
    assert len(src.trajectories) == 2


def test_k_i_within_bounds_for_each_trajectory():
    md = _make_dataset_metadata({f"t{i}": 10 for i in range(5)})
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = [
        md.conversations[i].conversation_id for i in range(5)
    ]

    src = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=5,
        random_seed=7,
    )
    for trajectory in src.trajectories:
        assert 0 <= trajectory.start_turn_index <= 7  # floor(0.7 * 10) = 7


def test_seed_determinism():
    md = _make_dataset_metadata({"a": 10, "b": 10, "c": 10})
    sampler1 = MagicMock()
    sampler1.next_conversation_id.side_effect = ["a", "b", "c"]
    sampler2 = MagicMock()
    sampler2.next_conversation_id.side_effect = ["a", "b", "c"]

    s1 = TrajectorySource(
        dataset_metadata=md, dataset_sampler=sampler1, concurrency=3, random_seed=999
    )
    s2 = TrajectorySource(
        dataset_metadata=md, dataset_sampler=sampler2, concurrency=3, random_seed=999
    )

    k1 = [(t.conversation_id, t.start_turn_index) for t in s1.trajectories]
    k2 = [(t.conversation_id, t.start_turn_index) for t in s2.trajectories]
    assert k1 == k2


def test_skips_zero_turn_traces_and_replenishes():
    md = _make_dataset_metadata({"good_a": 5, "empty_b": 0, "good_c": 5})
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = ["empty_b", "good_a", "good_c"]

    src = TrajectorySource(
        dataset_metadata=md, dataset_sampler=sampler, concurrency=2, random_seed=1
    )
    trajectory_ids = {t.conversation_id for t in src.trajectories}
    assert trajectory_ids == {"good_a", "good_c"}


def test_empty_pool_raises():
    md = _make_dataset_metadata({})
    sampler = MagicMock()
    with pytest.raises(EmptyTracePoolError):
        TrajectorySource(
            dataset_metadata=md, dataset_sampler=sampler, concurrency=2, random_seed=1
        )


def test_single_turn_trace_skipped_with_warning(caplog):
    """n=1 traces have no profiling turn after the warmup split; the source
    skips them with a warning. When only n=1 traces exist, the trajectory
    pool is empty and EmptyTracePoolError is raised."""
    md = _make_dataset_metadata({"only": 1})
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = ["only"]
    with caplog.at_level("WARNING"), pytest.raises(EmptyTracePoolError):
        TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=1,
            random_seed=42,
        )
    assert any(
        "Skipping trace" in r.getMessage() and "only" in r.getMessage()
        for r in caplog.records
    )
