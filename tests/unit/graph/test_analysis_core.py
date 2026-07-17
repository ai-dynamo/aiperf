# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct unit cover for the graph analysis core.

These functions (`elaborate_trace`, `compute_snapshot`) were previously
exercised only transitively through the strategy/adapter E2E path, so a
structural regression in any of them could slip past the suite. These tests
build a `ParsedGraph` and assert directly on each function's output -- the
parallel-readiness cohort keying and the snapshot t* partition.
"""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.models import (
    GraphRecord,
    LlmNode,
    ParsedGraph,
    StaticEdge,
    TraceRecord,
)
from aiperf.graph.analysis.snapshot import compute_snapshot
from aiperf.graph.analysis.timeline import GraphCycleError, elaborate_trace


def _llm(output: str, *, offset: int | None = None) -> LlmNode:
    return LlmNode(prompt=[f"@{output}"], output=output, arrival_offset_us=offset)


def _parsed(
    nodes: dict[str, object],
    edges: list[StaticEdge],
) -> ParsedGraph:
    graph = GraphRecord(nodes=nodes, edges=edges, state={})
    return ParsedGraph(
        graph=graph,
        traces=[TraceRecord(id="t")],
    )


def test_compute_snapshot_inclusive_tstar_boundary_and_warmup_split():
    """arrival < t* -> warmup; arrival >= t* -> profiled (t* itself is kept)."""
    parsed = _parsed(
        {
            "A": _llm("a", offset=0),
            "B": _llm("b", offset=100),
            "C": _llm("c", offset=200),
        },
        [
            StaticEdge(source="START", target="A"),
            StaticEdge(source="A", target="B"),
            StaticEdge(source="B", target="C"),
        ],
    )
    snap = compute_snapshot(parsed, parsed.traces[0], t_star_us=100)

    assert {sf.firing.node_id for sf in snap.warmup} == {"A"}
    profiled = {sf.firing.node_id: sf for sf in snap.profiled}
    assert set(profiled) == {"B", "C"}
    # The firing exactly at t* is kept (inclusive) and rebased to dispatch 0.
    assert profiled["B"].dispatch_offset_us == 0
    assert profiled["C"].dispatch_offset_us == 100


def test_elaborate_trace_depth_cap_raises_graph_cycle_error():
    """A firing count over depth_cap raises GraphCycleError (validator guard)."""
    parsed = _parsed(
        {"A": _llm("a"), "B": _llm("b")},
        [
            StaticEdge(source="START", target="A"),
            StaticEdge(source="A", target="B"),
        ],
    )
    with pytest.raises(GraphCycleError):
        elaborate_trace(parsed, parsed.traces[0], depth_cap=1)


# ---------------------------------------------------------------------------
# R1 -- start-anchored subtrees must elaborate (analysis/runtime agreement)
# ---------------------------------------------------------------------------


def _start_anchored_parsed() -> ParsedGraph:
    """The R1 review graph: START->a; a->b (completion); a->c (start-anchored);
    c->d (completion). The runtime fires all four; the dry-run must too."""
    return _parsed(
        {
            "a": _llm("a", offset=0),
            "b": _llm("b", offset=1_000_000),
            "c": _llm("c", offset=2_500_000),
            "d": _llm("d", offset=9_000_000),
        },
        [
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="b", delay_after_predecessor_us=0.0),
            StaticEdge(
                source="a", target="c", delay_after_predecessor_start_us=2_500_000.0
            ),
            StaticEdge(source="c", target="d", delay_after_predecessor_us=0.0),
        ],
    )


def test_elaborate_trace_follows_start_anchored_edges():
    """The whole subtree under a start-anchored edge fires in the dry-run.

    Before the fix the elaborator consulted only ``successors_after`` and the
    timeline stopped at ['a', 'b'] -- the anchored child ``c`` and its
    completion successor ``d`` never fired, so ``duration_us`` under-measured
    and snapshot planning misclassified the subtree.
    """
    parsed = _start_anchored_parsed()

    timeline = elaborate_trace(parsed, parsed.traces[0])

    assert [f.node_id for f in timeline.firings] == ["a", "b", "c", "d"]
    assert timeline.duration_us() == 9_000_000


def test_compute_snapshot_partitions_start_anchored_subtree():
    """Snapshot-at-t* sees anchored-subtree firings as warmup/profiled members."""
    parsed = _start_anchored_parsed()

    snap = compute_snapshot(parsed, parsed.traces[0], t_star_us=2_500_000)

    assert {sf.firing.node_id for sf in snap.warmup} == {"a", "b"}
    profiled = {sf.firing.node_id: sf for sf in snap.profiled}
    assert set(profiled) == {"c", "d"}
    assert profiled["c"].dispatch_offset_us == 0
    assert profiled["d"].dispatch_offset_us == 6_500_000


def test_elaborate_trace_pending_fan_in_waits_for_late_arrival():
    """A scheduled node whose fan-in is unsatisfied stays pending (not dropped).

    Mirrors the runtime's parked ``await_inputs``: ``join`` is scheduled via a
    start-anchored edge from ``a`` but requires ``b``'s output channel, so it
    fires only after ``b`` does.
    """
    from aiperf.dataset.graph.models import ChannelRequirement, ChannelSpec

    join = LlmNode(
        prompt=["@j"],
        output="j",
        inputs=[ChannelRequirement(channel="b", count=1)],
    )
    graph = GraphRecord(
        nodes={
            "a": _llm("a", offset=0),
            "b": _llm("b", offset=100),
            "join": join,
        },
        edges=[
            StaticEdge(source="START", target="a"),
            StaticEdge(source="a", target="b", delay_after_predecessor_us=0.0),
            StaticEdge(
                source="a", target="join", delay_after_predecessor_start_us=50.0
            ),
        ],
        state={
            "a": ChannelSpec(),
            "b": ChannelSpec(),
            "j": ChannelSpec(),
        },
    )
    parsed = ParsedGraph(graph=graph, traces=[TraceRecord(id="t")])

    timeline = elaborate_trace(parsed, parsed.traces[0])

    order = [f.node_id for f in timeline.firings]
    assert set(order) == {"a", "b", "join"}
    assert order.index("join") > order.index("b")
