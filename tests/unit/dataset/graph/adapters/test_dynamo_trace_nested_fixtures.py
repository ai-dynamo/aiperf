# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Flat-graph structural coverage for the nested-subagent Dynamo fixtures: every parent/child session flattens into one graph of ``{session_id}:{k}`` LlmNodes."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import param

from aiperf.dataset.graph.adapters.dynamo.trace import (
    DynamoTraceAdapterError,
    from_dynamo_trace,
)
from aiperf.dataset.graph.models import LlmNode, ParsedGraph, StaticEdge

from .conftest import write_jsonl

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "dynamo_nested"


def _node_ids(pb: ParsedGraph) -> set[str]:
    return set(pb.graph.nodes)


def _incoming(pb: ParsedGraph, target: str) -> list[StaticEdge]:
    return [
        e for e in pb.graph.edges if isinstance(e, StaticEdge) and e.target == target
    ]


def test_nested_2_level_flattens_parent_and_child() -> None:
    """A parent and its single subagent flatten to the exact five-node id set."""
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


@pytest.mark.parametrize(
    "fixture_name, expected_node_ids, expected_trace_ids, expect_all_llm_nodes",
    [
        param(
            "nested_3_level.jsonl.gz",
            ["sess_A:0", "sess_B:0", "sess_C:0"],
            ["sess_A"],
            False,
            id="nested_3_level_grandchild_joins_root_trace",
        ),
        param(
            "parallel_subagents.jsonl.gz",
            ["sess_B:0", "sess_B:1", "sess_C:0", "sess_C:1"],
            None,
            False,
            id="parallel_subagents_flatten_side_by_side",
        ),
        param(
            "tool_call_id_linkage.jsonl.gz",
            ["sess_B:0"],
            None,
            True,
            id="tool_call_id_linkage_has_no_subgraph_nodes",
        ),
    ],
)  # fmt: skip
def test_nested_fixtures_flatten_into_llm_nodes(
    fixture_name: str,
    expected_node_ids: list[str],
    expected_trace_ids: list[str] | None,
    expect_all_llm_nodes: bool,
) -> None:
    """Nesting depth and sibling parallelism collapse into flat ``{session_id}:{k}`` nodes on one graph."""
    pb = from_dynamo_trace(FIXTURES / fixture_name)
    if expect_all_llm_nodes:
        assert all(isinstance(n, LlmNode) for n in pb.graph.nodes.values())
    ids = _node_ids(pb)
    for node_id in expected_node_ids:
        assert node_id in ids
    if expected_trace_ids is not None:
        assert [t.id for t in pb.traces] == expected_trace_ids


def test_parallel_two_root_parses_one_trace_per_root() -> None:
    """Two independent roots yield two per-tree traces (multi-graph), not one multi-root graph."""
    pb = from_dynamo_trace(FIXTURES / "parallel_two_root.jsonl.gz")
    assert [t.id for t in pb.traces] == ["sess_A", "sess_B"]
    for trace in pb.traces:
        assert trace.graph_ref == trace.id
        assert "multi-root" not in trace.tags
    assert set(pb.graphs) == {"sess_A", "sess_B"}


def test_mixed_turn_tool_events_do_not_lower() -> None:
    """Recorded tool events parse cleanly but produce no per-node metadata: tool time is implicit in the recorded end-to-start gaps."""
    pb = from_dynamo_trace(FIXTURES / "mixed_turn.jsonl.gz")
    a2 = pb.graph.nodes["sess_A:1"]
    assert "tool_breakdown" not in a2.metadata["dynamo"]


def test_cycle_AB_A_raises_with_cycle_message() -> None:
    """A parent/child linkage cycle aborts the parse with a message naming the cycle."""
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
    """One ``request_end`` record; ``received``/``total`` define the recorded active interval."""
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


def _overlap_records() -> list[dict]:
    """Parent turn 1 active [1000, 1400]; child fires at 1150, inside it."""
    return [
        _rec(ts=1400, sid="parent", received=1000, total=400.0),
        _rec(ts=1150, sid="child", parent="parent"),
    ]


@pytest.mark.parametrize(
    "name, idle_gap_cap_seconds",
    [
        param("overlap.jsonl", "default", id="default_idle_gap_cap"),
        param("overlap_warp.jsonl", 0.0, id="idle_gap_cap_collapsed_to_zero"),
    ],
)  # fmt: skip
def test_child_overlapping_parent_turn_start_anchors_to_parent(
    tmp_path: Path,
    name: str,
    idle_gap_cap_seconds: float | str,
) -> None:
    """A child starting inside a still-running parent turn start-anchors to it, so the child tracks the parent's dispatch causally instead of freezing to the recorded wall clock."""
    p = write_jsonl(tmp_path / name, _overlap_records())
    pb = (
        from_dynamo_trace(p)
        if idle_gap_cap_seconds == "default"
        else from_dynamo_trace(p, idle_gap_cap_seconds=idle_gap_cap_seconds)
    )

    incoming = _incoming(pb, "child:0")
    assert [e.source for e in incoming] == ["parent:0"], (
        f"overlapping child must start-anchor to the parent; got {incoming}"
    )
    (edge,) = incoming
    # Both starts sit inside ONE active interval, so the recorded 150ms
    # start-to-start gap survives even a cap of zero; the end-delay stays None,
    # which is what marks the edge as start-anchored rather than finished-before.
    assert edge.delay_after_predecessor_start_us == pytest.approx(150_000)
    assert edge.delay_after_predecessor_us is None


def test_child_after_parent_turn_gets_finished_before_edge(tmp_path: Path) -> None:
    """Disjoint intervals: the parent turn finished before the child started, so the child carries a finished-before edge from it."""
    # Parent turn 1 active [1000, 1100]; child fires at 1150, after it.
    p = write_jsonl(
        tmp_path / "disjoint.jsonl",
        [
            _rec(ts=1100, sid="parent", received=1000, total=100.0),
            _rec(ts=1150, sid="child", parent="parent"),
        ],
    )
    pb = from_dynamo_trace(p)
    incoming = _incoming(pb, "child:0")
    assert [e.source for e in incoming] == ["parent:0"]
