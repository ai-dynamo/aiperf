# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.common.scenario.base import (
    EmptyTracePoolError,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
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


def _make_timestamped_subagent_dataset() -> DatasetMetadata:
    branch_id = "trace:spawn:agent_0"
    return DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=12000.0, branch_ids=[branch_id]),
                    TurnMetadata(
                        timestamp_ms=20000.0,
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id=branch_id,
                            )
                        ],
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id=branch_id,
                        child_conversation_ids=["trace::sa:agent_0"],
                        mode=ConversationBranchMode.SPAWN,
                        start_timestamp_ms=13000.0,
                    )
                ],
            ),
            ConversationMetadata(
                conversation_id="trace::sa:agent_0",
                turns=[
                    TurnMetadata(timestamp_ms=13000.0),
                    TurnMetadata(timestamp_ms=14000.0),
                    TurnMetadata(timestamp_ms=17000.0),
                ],
                is_root=False,
                agent_depth=1,
                parent_conversation_id="trace",
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


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
        # defaults: floor(0.25 * 10) = 2, floor(0.75 * 10) = 7
        assert 2 <= trajectory.start_turn_index <= 7


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


def test_timestamped_snapshot_includes_inflight_subagent_and_gated_parent():
    branch_id = "trace:spawn:agent_0"
    md = _make_timestamped_subagent_dataset()
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = ["trace"]

    src = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=123,
        start_min_ratio=0.675,
        start_max_ratio=0.675,
    )

    trajectory = src.trajectories[0]
    assert trajectory.snapshot is not None
    assert trajectory.snapshot.t_star_ms == pytest.approx(13500.0)
    by_cid = {state.conversation_id: state for state in trajectory.snapshot.states}
    parent = by_cid["trace"]
    child = by_cid["trace::sa:agent_0"]
    resume_boundaries = {
        boundary.conversation_id: boundary.next_turn_index
        for boundary in trajectory.snapshot.replay_resume_boundaries
    }

    assert parent.waiting_on_children is True
    assert parent.next_turn_index == 2
    assert child.next_turn_index == 1
    assert child.parent_correlation_id == parent.x_correlation_id
    assert child.branch_id == branch_id
    assert child.branch_mode == ConversationBranchMode.SPAWN
    assert child.next_dispatch_offset_ms == pytest.approx(500.0)
    assert resume_boundaries == {"trace": 2, "trace::sa:agent_0": 1}
    # Both active-at-t* sessions are warmed: the mid-flight child (turn 0) and
    # the gated parent (turn 1, priming its join turn). Gated parents are no
    # longer excluded from warmup.
    assert src.warmup_credit_count == 2


def test_timestamped_summary_logs_sample_time_not_root_turn_pct(caplog):
    md = _make_timestamped_subagent_dataset()
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = ["trace"]

    with caplog.at_level(logging.INFO, logger="aiperf.timing.trajectory_source"):
        TrajectorySource(
            dataset_metadata=md,
            dataset_sampler=sampler,
            concurrency=1,
            random_seed=123,
            start_min_ratio=0.675,
            start_max_ratio=0.675,
        )

    log_text = "\n".join(record.getMessage() for record in caplog.records)
    assert "observed sample pct:" in log_text
    assert "sample_time= 68%" in log_text
    assert "root_next=  2/3" in log_text
    assert "live=2 ready=1" in log_text


def test_timestamped_snapshot_after_spawning_turn_schedules_future_child_start():
    branch_id = "trace:spawn:agent_0"
    md = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace",
                turns=[
                    TurnMetadata(timestamp_ms=0.0),
                    TurnMetadata(timestamp_ms=12000.0, branch_ids=[branch_id]),
                    TurnMetadata(
                        timestamp_ms=20000.0,
                        prerequisites=[
                            TurnPrerequisite(
                                kind=PrerequisiteKind.SPAWN_JOIN,
                                branch_id=branch_id,
                            )
                        ],
                    ),
                ],
                branches=[
                    ConversationBranchInfo(
                        branch_id=branch_id,
                        child_conversation_ids=["trace::sa:agent_0"],
                        mode=ConversationBranchMode.SPAWN,
                        start_timestamp_ms=13000.0,
                    )
                ],
            ),
            ConversationMetadata(
                conversation_id="trace::sa:agent_0",
                turns=[
                    TurnMetadata(timestamp_ms=13000.0),
                    TurnMetadata(timestamp_ms=14000.0),
                ],
                is_root=False,
                agent_depth=1,
                parent_conversation_id="trace",
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    sampler = MagicMock()
    sampler.next_conversation_id.side_effect = ["trace"]

    src = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=123,
        start_min_ratio=0.625,
        start_max_ratio=0.625,
    )

    trajectory = src.trajectories[0]
    assert trajectory.snapshot is not None
    assert trajectory.snapshot.t_star_ms == pytest.approx(12500.0)
    by_cid = {state.conversation_id: state for state in trajectory.snapshot.states}
    parent = by_cid["trace"]
    child = by_cid["trace::sa:agent_0"]
    resume_boundaries = {
        boundary.conversation_id: boundary.next_turn_index
        for boundary in trajectory.snapshot.replay_resume_boundaries
    }

    assert parent.waiting_on_children is True
    assert parent.next_turn_index == 2
    assert child.next_turn_index == 0
    assert child.next_dispatch_offset_ms == pytest.approx(500.0)
    assert resume_boundaries == {"trace": 2}


def test_next_recycle_conversation_id_uses_sampler_round_robin():
    """Recycle draws the next root from the dataset sampler.

    A SequentialSampler yields every root in order and wraps indefinitely, so
    over a whole number of cycles each root is reused exactly equally -- no
    trace is favored. This replaces the old strategy-side recycle queue, whose
    copy accumulation favored short, rootless-heavy traces.
    """
    from collections import Counter

    from aiperf.dataset.dataset_samplers import SequentialSampler

    dataset = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id=f"trace_{i}",
                turns=[
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                ],
                is_root=True,
            )
            for i in range(4)
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    sampler = SequentialSampler([c.conversation_id for c in dataset.conversations])
    src = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=2,
        random_seed=0,
    )

    seen = [src.next_recycle_conversation_id() for _ in range(8)]
    counts = Counter(seen)
    assert set(counts) == {f"trace_{i}" for i in range(4)}
    assert all(v == 2 for v in counts.values()), counts


def test_next_recycle_conversation_id_skips_unspawnable_roots():
    """Roots with no spawnable session (zero turns) are skipped so recycle never
    hands back a dead trace; bounded so an all-empty pool returns None."""
    from aiperf.dataset.dataset_samplers import SequentialSampler

    dataset = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="ok",
                turns=[
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                    TurnMetadata(timestamp_ms=None, delay_ms=None),
                ],
                is_root=True,
            ),
            ConversationMetadata(conversation_id="empty", turns=[], is_root=True),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    sampler = SequentialSampler([c.conversation_id for c in dataset.conversations])
    src = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=0,
    )

    for _ in range(5):
        assert src.next_recycle_conversation_id() == "ok"
