# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flat-IR structural coverage for the nested-subagent Dynamo fixtures.

Every parent/child session flattens into ONE graph of ``{session_id}:{k}``
``LlmNode``s. Concurrency is EMERGENT from the recorded intervals: a child whose
first request overlaps a still-running parent turn START-ANCHORS to that
parent (a single ``delay_after_predecessor_start_us`` edge, so the child
tracks the parent's dispatch causally), while disjoint intervals yield a
finished-before edge.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.graph.adapters.dynamo.trace import (
    DynamoTraceAdapterError,
    from_dynamo_trace,
)
from aiperf.dataset.graph.models import LlmNode, StaticEdge

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dynamo_nested"


def _node_ids(pb) -> set[str]:
    return set(pb.graph.nodes)


def _incoming(pb, target: str) -> list[StaticEdge]:
    return [
        e for e in pb.graph.edges if isinstance(e, StaticEdge) and e.target == target
    ]


def test_nested_2_level_flattens_parent_and_child() -> None:
    pb = from_dynamo_trace(FIXTURES / "nested_2_level.jsonl.gz")
    assert _node_ids(pb) == {
        "sess_A:0",
        "sess_A:1",
        "sess_A:2",
        "sess_B:0",
        "sess_B:1",
    }
    assert all(isinstance(n, LlmNode) for n in pb.graph.nodes.values())
    assert [t.id for t in pb.traces] == ["sess_A"]


def test_nested_3_level_flattens_all_three_sessions() -> None:
    pb = from_dynamo_trace(FIXTURES / "nested_3_level.jsonl.gz")
    for sid in ("sess_A", "sess_B", "sess_C"):
        assert f"{sid}:0" in _node_ids(pb)
    assert [t.id for t in pb.traces] == ["sess_A"]


def test_parallel_subagents_flatten_side_by_side() -> None:
    pb = from_dynamo_trace(FIXTURES / "parallel_subagents.jsonl.gz")
    for sid in ("sess_B", "sess_C"):
        assert f"{sid}:0" in _node_ids(pb)
        assert f"{sid}:1" in _node_ids(pb)


def test_parallel_two_root_parses_one_trace_per_root() -> None:
    pb = from_dynamo_trace(FIXTURES / "parallel_two_root.jsonl.gz")
    # Two independent roots -> two per-tree traces (multi-graph), no multi-root tag.
    assert [t.id for t in pb.traces] == ["sess_A", "sess_B"]
    for trace in pb.traces:
        assert trace.graph_ref == trace.id
        assert "multi-root" not in trace.tags
    assert set(pb.graphs) == {"sess_A", "sess_B"}


def test_tool_call_id_linkage_fixture_flattens_without_subgraph_nodes() -> None:
    pb = from_dynamo_trace(FIXTURES / "tool_call_id_linkage.jsonl.gz")
    assert all(isinstance(n, LlmNode) for n in pb.graph.nodes.values())
    assert "sess_B:0" in _node_ids(pb)


def test_mixed_turn_tool_events_do_not_lower() -> None:
    """Recorded tool events parse cleanly but produce no per-node metadata:
    tool time is implicit in the recorded end-to-start gaps the replay honors."""
    pb = from_dynamo_trace(FIXTURES / "mixed_turn.jsonl.gz")
    a2 = pb.graph.nodes["sess_A:1"]
    assert "tool_breakdown" not in a2.metadata["dynamo"]


def test_cycle_AB_A_raises_with_cycle_message() -> None:
    with pytest.raises(DynamoTraceAdapterError) as excinfo:
        from_dynamo_trace(FIXTURES / "cycle_AB_A.jsonl.gz")
    assert "cycle" in str(excinfo.value).lower(), (
        f"DynamoTraceAdapterError message should mention 'cycle'; got: {excinfo.value!s}"
    )


# --- emergent concurrency: overlap vs disjoint intervals --------------------


def _rec(
    *,
    ts: int,
    sid: str,
    parent: str | None = None,
    received: int | None = None,
    total: float | None = None,
) -> dict:
    ctx: dict = {"session_id": sid}
    if parent is not None:
        ctx["parent_session_id"] = parent
    req: dict = {
        "request_id": f"{sid}-{ts}",
        "model": "m",
        "input_tokens": 32,
        "output_tokens": 16,
        "cached_tokens": 0,
    }
    if received is not None:
        req["request_received_ms"] = received
    if total is not None:
        req["total_time_ms"] = total
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": ctx,
        "request": req,
    }


def test_child_overlapping_parent_turn_start_anchors_to_parent(
    tmp_path: Path,
) -> None:
    """The child starts while the parent's turn is still running -> one
    start-anchored edge from the parent (``delay_after_predecessor_start_us``
    set), so the child tracks the parent's dispatch causally instead of
    freezing to the recorded wall clock."""
    p = tmp_path / "overlap.jsonl"
    records = [
        # Parent turn 1 active [1000, 1400]; child fires at 1150, inside it.
        _rec(ts=1400, sid="parent", received=1000, total=400.0),
        _rec(ts=1150, sid="child", parent="parent"),
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))

    pb = from_dynamo_trace(p)
    parent_a1 = "parent:0"
    child_a1 = "child:0"
    incoming = _incoming(pb, child_a1)
    assert [e.source for e in incoming] == [parent_a1], (
        f"overlapping child must start-anchor to the parent; got {incoming}"
    )
    (edge,) = incoming
    # The start-anchor edge keeps the recorded parent-start-to-child-start gap
    # (150ms, below the shared 60s idle-gap cap) while the end-delay stays None.
    assert edge.delay_after_predecessor_start_us == pytest.approx(150_000)
    assert edge.delay_after_predecessor_us is None


def test_child_overlapping_parent_turn_start_anchor_warps_under_cap(
    tmp_path: Path,
) -> None:
    """Same overlap fixture with the idle-gap cap collapsed to zero: the
    start-anchor delay rides the warped clock (every idle gap compressed), while
    the anchor kind survives (end-delay stays None)."""
    p = tmp_path / "overlap_warp.jsonl"
    records = [
        # Parent turn 1 active [1000, 1400]; child fires at 1150, inside it.
        _rec(ts=1400, sid="parent", received=1000, total=400.0),
        _rec(ts=1150, sid="child", parent="parent"),
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))

    pb = from_dynamo_trace(p, idle_gap_cap_seconds=0.0)
    parent_a1 = "parent:0"
    child_a1 = "child:0"
    incoming = _incoming(pb, child_a1)
    assert [e.source for e in incoming] == [parent_a1], (
        f"overlapping child must start-anchor to the parent; got {incoming}"
    )
    (edge,) = incoming
    # Both starts sit inside ONE active interval (no idle gap between them), so
    # the start-to-start gap survives even a cap of zero.
    assert edge.delay_after_predecessor_start_us == pytest.approx(150_000)
    assert edge.delay_after_predecessor_us is None


def test_child_after_parent_turn_gets_finished_before_edge(tmp_path: Path) -> None:
    """Disjoint intervals: the parent turn finished before the child started,
    so the child carries a finished-before edge from it."""
    p = tmp_path / "disjoint.jsonl"
    records = [
        # Parent turn 1 active [1000, 1100]; child fires at 1150, after it.
        _rec(ts=1100, sid="parent", received=1000, total=100.0),
        _rec(ts=1150, sid="child", parent="parent"),
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))

    pb = from_dynamo_trace(p)
    parent_a1 = "parent:0"
    child_a1 = "child:0"
    incoming = _incoming(pb, child_a1)
    assert [e.source for e in incoming] == [parent_a1]
