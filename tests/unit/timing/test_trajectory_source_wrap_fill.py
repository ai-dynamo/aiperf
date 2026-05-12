# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TrajectorySource wrap-fill helper.

These tests exercise the wrap-fill helper in isolation. Task 3 wires it
into ``TrajectorySource.__init__``; the full happy path lives in
``tests/component_integration/test_agentic_replay_wrap_fill.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

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
