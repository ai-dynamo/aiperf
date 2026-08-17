# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A context-overflow must only excuse orphans on its OWN branch.

``_TraceContext.overflow_terminated`` used to be a trace-wide bool: once any
node overflowed, every subsequent ``ChannelOrphanedError`` anywhere in the trace
was swallowed as a clean exit. On a fan-out graph that meant a genuine producer
failure on an unrelated branch -- ``mark_producer_done(success=False)`` -- was
silently reclassified as an expected consequence of someone else's overflow and
never counted as a trace error.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from aiperf.graph.executor import TraceExecutor


def _executor_with_successors(edges: dict[str, list[str]]) -> TraceExecutor:
    """A TraceExecutor whose scheduler exposes ``edges`` as the successor map."""
    executor = TraceExecutor.__new__(TraceExecutor)
    scheduler = MagicMock()
    scheduler.successors_after = lambda node_id: edges.get(node_id, [])
    executor._scheduler = scheduler
    return executor


def _ctx(overflowed: set[str]):
    ctx = MagicMock()
    ctx.overflow_terminated_nodes = overflowed
    return ctx


def test_no_overflow_means_no_exemption() -> None:
    executor = _executor_with_successors({"a": ["b"]})
    assert executor._is_overflow_descendant("b", _ctx(set())) is False


def test_overflowed_node_itself_is_exempt() -> None:
    executor = _executor_with_successors({})
    assert executor._is_overflow_descendant("a", _ctx({"a"})) is True


def test_direct_successor_of_overflow_is_exempt() -> None:
    executor = _executor_with_successors({"a": ["b"]})
    assert executor._is_overflow_descendant("b", _ctx({"a"})) is True


def test_transitive_successor_of_overflow_is_exempt() -> None:
    executor = _executor_with_successors({"a": ["b"], "b": ["c"], "c": ["d"]})
    assert executor._is_overflow_descendant("d", _ctx({"a"})) is True


def test_parallel_branch_is_NOT_exempt() -> None:
    """The regression: 'x' overflowed, but 'y2' is on a sibling branch.

    A trace-wide flag exempted 'y2' too, hiding a real producer failure.
    """
    executor = _executor_with_successors(
        {"root": ["x", "y1"], "x": ["x2"], "y1": ["y2"]}
    )
    assert executor._is_overflow_descendant("x2", _ctx({"x"})) is True
    assert executor._is_overflow_descendant("y2", _ctx({"x"})) is False


def test_ancestor_of_overflow_is_NOT_exempt() -> None:
    """Overflow downstream cannot retroactively excuse an upstream orphan."""
    executor = _executor_with_successors({"a": ["b"], "b": ["c"]})
    assert executor._is_overflow_descendant("a", _ctx({"c"})) is False


def test_walk_terminates_on_cyclic_successors() -> None:
    """A malformed cyclic successor map must not hang the orphan path."""
    executor = _executor_with_successors({"a": ["b"], "b": ["a"]})
    assert executor._is_overflow_descendant("zzz", _ctx({"a"})) is False


def test_trace_scoped_property_still_answers_did_any_overflow() -> None:
    from aiperf.graph.context import _TraceContext

    ctx = _TraceContext(trace=MagicMock(), store=MagicMock())
    assert ctx.overflow_terminated is False
    ctx.overflow_terminated_nodes.add("a")
    assert ctx.overflow_terminated is True
