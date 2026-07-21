# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TrajectorySource lane sampling when concurrency exceeds the pool.

Trace selection is the dataset sampler's job: ``TrajectorySource`` samples one
trace per concurrency lane straight from the sampler (which wraps), so when
concurrency > pool the same trace recurs across lanes, each with its own t*
snapshot. There is no separate wrap-fill step. The full happy path lives in
``tests/component_integration/test_agentic_replay_wrap_fill.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.timing.trajectory_source import Trajectory, TrajectorySource


def _make_metadata(num_traces: int, turns_per_trace: int) -> MagicMock:
    """Build a MagicMock DatasetMetadata with N conversations of M turns each."""
    convs = []
    for i in range(num_traces):
        cid = f"trace_{i}"
        turns = [MagicMock(turn_index=t) for t in range(turns_per_trace)]
        convs.append(MagicMock(conversation_id=cid, turns=turns))
    md = MagicMock()
    md.conversations = convs
    return md


def _build_source(
    num_traces: int, turns_per_trace: int, concurrency: int
) -> TrajectorySource:
    md = _make_metadata(num_traces, turns_per_trace)
    # Real SequentialSampler: wraps round-robin over the pool, so concurrency >
    # pool repeats traces across lanes (the production behavior).
    sampler = SequentialSampler([c.conversation_id for c in md.conversations])
    return TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=concurrency,
        random_seed=42,
    )


def test_init_pool_1_concurrency_4_produces_4_trajectories_same_trace():
    src = _build_source(num_traces=1, turns_per_trace=10, concurrency=4)
    assert len(src.trajectories) == 4
    assert {t.conversation_id for t in src.trajectories} == {"trace_0"}


def test_init_pool_3_concurrency_10_produces_balanced_distribution():
    src = _build_source(num_traces=3, turns_per_trace=10, concurrency=10)
    assert len(src.trajectories) == 10
    counts = {"trace_0": 0, "trace_1": 0, "trace_2": 0}
    for t in src.trajectories:
        counts[t.conversation_id] += 1
    assert sorted(counts.values()) == [3, 3, 4]


def test_init_pool_5_concurrency_5_no_wrap_fill_distinct_only():
    src = _build_source(num_traces=5, turns_per_trace=10, concurrency=5)
    assert len(src.trajectories) == 5
    assert len({t.conversation_id for t in src.trajectories}) == 5


def test_init_logs_info_when_traces_repeat_across_lanes(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="aiperf.timing.trajectory_source"):
        _build_source(num_traces=2, turns_per_trace=10, concurrency=8)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("distinct traces" in m for m in msgs), msgs


def test_init_does_not_log_repeat_info_when_all_lanes_distinct(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="aiperf.timing.trajectory_source"):
        _build_source(num_traces=4, turns_per_trace=10, concurrency=4)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("distinct traces" in m for m in msgs), msgs


def test_sendable_start_rejects_empty_raw_messages_without_prefix():
    meta = MagicMock()
    meta.system_message = None
    meta.user_context_message = None
    turn = MagicMock()
    turn.raw_messages_count = 0
    meta.turns = [turn]

    assert TrajectorySource._trajectory_start_is_sendable(meta, 0) is False


def test_sendable_start_accepts_empty_raw_messages_with_system_prefix():
    meta = MagicMock()
    meta.system_message = "system"
    meta.user_context_message = None
    turn = MagicMock()
    turn.raw_messages_count = 0
    meta.turns = [turn]

    assert TrajectorySource._trajectory_start_is_sendable(meta, 0) is True


def test_sendable_start_accepts_old_metadata_without_raw_message_count():
    meta = MagicMock()
    meta.system_message = None
    meta.user_context_message = None
    turn = MagicMock(spec=[])
    meta.turns = [turn]

    assert TrajectorySource._trajectory_start_is_sendable(meta, 0) is True


def test_conversation_metadata_preserves_raw_message_count_without_payload_copy():
    conv = Conversation(
        session_id="trace_0",
        turns=[
            Turn(raw_messages=None),
            Turn(raw_messages=[]),
            Turn(raw_messages=[{"role": "user", "content": "hello"}]),
        ],
    )

    counts = [turn.raw_messages_count for turn in conv.metadata().turns]
    assert counts == [None, 0, 1]


def test_conversation_metadata_preserves_prefix_messages():
    conv = Conversation(
        session_id="trace_0",
        system_message="system",
        user_context_message="context",
        turns=[Turn(raw_messages=[])],
    )

    meta = conv.metadata()

    assert meta.system_message == "system"
    assert meta.user_context_message == "context"


def test_conversation_to_metadata_preserves_prefix_messages():
    conv = Conversation(
        session_id="trace_0",
        system_message="system",
        user_context_message="context",
        turns=[Turn(raw_messages=[])],
    )

    meta = conv.to_metadata()

    assert meta.system_message == "system"
    assert meta.user_context_message == "context"


def test_init_samples_only_warmup_profile_pairs_with_nonempty_first_payloads():
    def turn(raw_messages_count):
        t = MagicMock()
        t.raw_messages_count = raw_messages_count
        return t

    conv = MagicMock()
    conv.conversation_id = "trace_0"
    conv.system_message = None
    conv.user_context_message = None
    conv.turns = [
        turn(1),
        turn(0),
        turn(1),
        turn(1),
    ]
    md = MagicMock()
    md.conversations = [conv]
    sampler = SequentialSampler(["trace_0"])

    src = TrajectorySource(
        dataset_metadata=md,
        dataset_sampler=sampler,
        concurrency=1,
        random_seed=42,
        start_min_ratio=0.0,
        start_max_ratio=1.0,
    )

    # k=0 is invalid because profiling would start on empty turn 1.
    # k=1 is invalid because warmup would start on empty turn 1.
    assert src.trajectories == [
        Trajectory(conversation_id="trace_0", start_turn_index=2)
    ]
