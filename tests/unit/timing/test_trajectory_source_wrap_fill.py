# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TrajectorySource wrap-fill helper.

These tests exercise the wrap-fill helper in isolation. Task 3 wires it
into ``TrajectorySource.__init__``; the full happy path lives in
``tests/component_integration/test_agentic_replay_wrap_fill.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aiperf.common.models.dataset_models import Conversation, Turn
from aiperf.timing.trajectory_source import Trajectory, TrajectorySource


def _make_metadata_lookup(num_traces: int, turns_per_trace: int) -> dict:
    """Build a minimal metadata lookup with N traces, each with M turns."""
    lookup = {}
    for i in range(num_traces):
        cid = f"trace_{i}"
        turns = [MagicMock(turn_index=t) for t in range(turns_per_trace)]
        conv = MagicMock(conversation_id=cid, turns=turns)
        lookup[cid] = conv
    return lookup


def _make_source_for_helper(num_traces: int, turns_per_trace: int) -> TrajectorySource:
    """Construct a TrajectorySource via __new__ to bypass __init__ for helper testing.

    Task 3 will exercise the full __init__ path; here we only want to call
    _wrap_fill_lanes() directly without triggering the distinct-build loop.
    """
    src = TrajectorySource.__new__(TrajectorySource)
    src._random_seed = 42
    src._start_min_ratio = 0.0
    src._start_max_ratio = 0.7
    src._metadata_lookup = _make_metadata_lookup(num_traces, turns_per_trace)
    return src


def test_wrap_fill_extends_to_target_count():
    src = _make_source_for_helper(num_traces=3, turns_per_trace=5)
    distinct = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    extras = src._wrap_fill_lanes(distinct, extra_count=7)
    assert len(extras) == 7


def test_wrap_fill_cycles_conversation_ids_in_order():
    src = _make_source_for_helper(num_traces=3, turns_per_trace=5)
    distinct = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(3)
    ]
    extras = src._wrap_fill_lanes(distinct, extra_count=7)
    assert [e.conversation_id for e in extras] == [
        "trace_0",
        "trace_1",
        "trace_2",
        "trace_0",
        "trace_1",
        "trace_2",
        "trace_0",
    ]


def test_wrap_fill_start_turn_index_is_deterministic():
    src1 = _make_source_for_helper(num_traces=2, turns_per_trace=10)
    src2 = _make_source_for_helper(num_traces=2, turns_per_trace=10)
    distinct = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(2)
    ]
    extras1 = src1._wrap_fill_lanes(distinct, extra_count=4)
    extras2 = src2._wrap_fill_lanes(distinct, extra_count=4)
    assert [e.start_turn_index for e in extras1] == [
        e.start_turn_index for e in extras2
    ]


def test_wrap_fill_decorrelates_k_i_across_lanes_sharing_trace():
    src = _make_source_for_helper(num_traces=1, turns_per_trace=20)
    distinct = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    extras = src._wrap_fill_lanes(distinct, extra_count=16)
    k_values = {e.start_turn_index for e in extras}
    assert len(k_values) >= 2, f"Expected decorrelated k_i, got {k_values!r}"


def test_wrap_fill_pool_of_two_turns_uses_k_zero():
    src = _make_source_for_helper(num_traces=1, turns_per_trace=2)
    distinct = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    extras = src._wrap_fill_lanes(distinct, extra_count=3)
    assert all(e.start_turn_index == 0 for e in extras)


def test_wrap_fill_respects_configured_start_range():
    src = _make_source_for_helper(num_traces=1, turns_per_trace=20)
    src._start_min_ratio = 0.9
    src._start_max_ratio = 1.0
    distinct = [Trajectory(conversation_id="trace_0", start_turn_index=18)]

    extras = src._wrap_fill_lanes(distinct, extra_count=8)

    assert {e.start_turn_index for e in extras} == {18}


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


class _FakeSampler:
    """Hands out conversation_ids in order; raises StopIteration when exhausted.

    Mirrors what the production sampler does at end-of-pool.
    """

    def __init__(self, cids: list[str]) -> None:
        self._cids = list(cids)
        self._i = 0

    def next_conversation_id(self) -> str:
        if self._i >= len(self._cids):
            raise StopIteration
        cid = self._cids[self._i]
        self._i += 1
        return cid


def _build_source(
    num_traces: int, turns_per_trace: int, concurrency: int
) -> TrajectorySource:
    md = _make_metadata(num_traces, turns_per_trace)
    sampler = _FakeSampler([c.conversation_id for c in md.conversations])
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


def test_init_logs_info_when_wrap_fill_activates(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="aiperf.timing.trajectory_source"):
        _build_source(num_traces=2, turns_per_trace=10, concurrency=8)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Trajectory reuse" in m for m in msgs), msgs


def test_init_does_not_log_info_when_no_wrap_fill_needed(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="aiperf.timing.trajectory_source"):
        _build_source(num_traces=4, turns_per_trace=10, concurrency=4)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("Trajectory reuse" in m for m in msgs), msgs


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
    sampler = _FakeSampler(["trace_0"])

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
