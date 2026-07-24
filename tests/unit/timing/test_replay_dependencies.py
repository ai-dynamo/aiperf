# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

from aiperf.timing.replay_dependencies import (
    RecordedTurnInterval,
    ReplayTurnKey,
    infer_cross_stream_predecessors,
)


def _interval(
    name: str,
    stream: str,
    start_ms: float | None,
    api_time_ms: float | None,
) -> RecordedTurnInterval:
    return RecordedTurnInterval(
        key=ReplayTurnKey(name, 0),
        stream_id=stream,
        start_ms=start_ms,
        api_time_ms=api_time_ms,
    )


def test_sequential_intervals_form_singleton_frontiers() -> None:
    a = _interval("a", "a", 0.0, 10.0)
    b = _interval("b", "b", 10.0, 10.0)
    c = _interval("c", "c", 20.0, 10.0)

    result = infer_cross_stream_predecessors([a, b, c])

    assert result[a.key] == ()
    assert result[b.key] == (a.key,)
    assert result[c.key] == (b.key,)


def test_parallel_abc_all_join_before_d() -> None:
    a = _interval("a", "a", 0.0, 10.0)
    b = _interval("b", "b", 1.0, 8.0)
    c = _interval("c", "c", 2.0, 7.0)
    d = _interval("d", "d", 10.0, 1.0)

    result = infer_cross_stream_predecessors([a, b, c, d])

    assert result[a.key] == ()
    assert result[b.key] == ()
    assert result[c.key] == ()
    assert result[d.key] == (a.key, b.key, c.key)


def test_boundary_touch_is_sequential_but_equal_starts_are_parallel() -> None:
    a = _interval("a", "a", 0.0, 5.0)
    b = _interval("b", "b", 5.0, 0.0)
    c = _interval("c", "c", 5.0, 0.0)

    result = infer_cross_stream_predecessors([a, b, c])

    assert result[b.key] == (a.key,)
    assert result[c.key] == (a.key,)
    assert b.key not in result[c.key]
    assert c.key not in result[b.key]


def test_missing_and_invalid_durations_are_zero_width() -> None:
    missing = _interval("missing", "missing", 1.0, None)
    negative = _interval("negative", "negative", 2.0, -5.0)
    non_finite = _interval("non-finite", "non-finite", 3.0, math.inf)
    target = _interval("target", "target", 4.0, 1.0)

    result = infer_cross_stream_predecessors([missing, negative, non_finite, target])

    assert result[negative.key] == (missing.key,)
    assert result[non_finite.key] == (negative.key,)
    assert result[target.key] == (non_finite.key,)


def test_missing_and_non_finite_starts_add_no_cross_stream_edges() -> None:
    missing = _interval("missing", "missing", None, 1.0)
    non_finite = _interval("non-finite", "non-finite", math.nan, 1.0)
    target = _interval("target", "target", 10.0, 1.0)

    result = infer_cross_stream_predecessors([missing, non_finite, target])

    assert result[target.key] == ()


def test_transitive_overlap_uses_precise_frontier_not_connected_component() -> None:
    long_a = _interval("long-a", "a", 0.0, 10.0)
    short_b = _interval("short-b", "b", 1.0, 1.0)
    later_c = _interval("later-c", "c", 3.0, 1.0)

    result = infer_cross_stream_predecessors([long_a, short_b, later_c])

    assert result[short_b.key] == ()
    assert result[later_c.key] == (short_b.key,)
    assert long_a.key not in result[later_c.key]


def test_same_stream_order_is_left_to_conversation_replay() -> None:
    a = RecordedTurnInterval(ReplayTurnKey("stream", 0), "stream", 0.0, 5.0)
    b = RecordedTurnInterval(ReplayTurnKey("stream", 1), "stream", 1.0, 5.0)

    result = infer_cross_stream_predecessors([a, b])

    assert result[a.key] == ()
    assert result[b.key] == ()
