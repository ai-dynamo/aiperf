# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R3 -- ``collapse_leading_start_offsets`` (the `--burst-phase-starts` collapse).

The leading t*-relative phase-start offset lives on a firing's START in-edge
``min_start_delay_us`` (stamped by ``interval_order.build_interval_edges`` and
the ``snapshot_chop`` frontier re-root); NO producer stamps the node-level
field. Burst must zero exactly those leading offsets while every non-START
edge keeps its recorded inter-turn pacing. The timing strategy's
``_burst_collapse_leading_offsets`` is expected to delegate here.
"""

from __future__ import annotations

from aiperf.dataset.graph.models import GraphRecord, LlmNode, StaticEdge
from aiperf.graph.scheduler import collapse_leading_start_offsets


def _llm(output: str, *, min_start: float | None = None) -> LlmNode:
    return LlmNode(prompt=[f"@{output}"], output=output, min_start_delay_us=min_start)


def _graph() -> GraphRecord:
    return GraphRecord(
        nodes={
            "a": _llm("a"),
            "b": _llm("b", min_start=7_000_000.0),
            "c": _llm("c"),
        },
        edges=[
            StaticEdge(source="START", target="a", min_start_delay_us=5_000_000.0),
            StaticEdge(
                source="a",
                target="b",
                delay_after_predecessor_us=2_000_000.0,
                min_start_delay_us=1_000_000.0,
            ),
            StaticEdge(
                source="a", target="c", delay_after_predecessor_start_us=3_000_000.0
            ),
        ],
        state={},
    )


def test_collapse_zeroes_start_edge_min_start_delay():
    """The START-edge leading offset (the t* anchor carrier) is zeroed."""
    collapsed = collapse_leading_start_offsets(_graph())

    start_edges = [e for e in collapsed.edges if e.source == "START"]
    assert len(start_edges) == 1
    assert start_edges[0].min_start_delay_us == 0.0


def test_collapse_preserves_inter_turn_edge_pacing():
    """Non-START edges keep every recorded delay (burst governs starts only)."""
    collapsed = collapse_leading_start_offsets(_graph())

    by_target = {e.target: e for e in collapsed.edges if e.source != "START"}
    assert by_target["b"].delay_after_predecessor_us == 2_000_000.0
    assert by_target["b"].min_start_delay_us == 1_000_000.0
    assert by_target["c"].delay_after_predecessor_start_us == 3_000_000.0


def test_collapse_node_level_offsets_zero_leading_preserve_mid_graph():
    """Node-level ``min_start_delay_us`` is a leading offset ONLY on a node with
    no real predecessor.

    Burst zeroes those (a hand-authored leading anchor) and leaves mid-graph
    node-level pacing intact -- the AND-fan-in residual fold lands node-level on
    a node that KEEPS a surviving-pred edge, so zeroing it would silently revert
    the fold.
    """
    graph = GraphRecord(
        nodes={
            "x": _llm("x", min_start=4_000_000.0),
            "y": _llm("y", min_start=2_000_000.0),
        },
        edges=[
            StaticEdge(source="START", target="x", min_start_delay_us=6_000_000.0),
            StaticEdge(source="x", target="y", delay_after_predecessor_us=1_000_000.0),
        ],
        state={},
    )

    collapsed = collapse_leading_start_offsets(graph)

    # x roots at START only -> its node-level delay is a leading offset, zeroed.
    assert collapsed.nodes["x"].min_start_delay_us == 0.0
    # y has a real predecessor -> its node-level delay is mid-graph pacing, kept.
    assert collapsed.nodes["y"].min_start_delay_us == 2_000_000.0


def test_collapse_is_pure_and_identity_preserving():
    """The input graph is untouched; untouched nodes/edges keep identity."""
    graph = _graph()

    collapsed = collapse_leading_start_offsets(graph)

    # Purity: the original still carries its offsets.
    assert graph.edges[0].min_start_delay_us == 5_000_000.0
    assert graph.nodes["b"].min_start_delay_us == 7_000_000.0
    # Identity for untouched members (no needless struct rebuilds).
    assert collapsed.nodes["a"] is graph.nodes["a"]
    assert collapsed.edges[1] is graph.edges[1]


def test_collapse_noop_graph_round_trips():
    """A graph with no leading offsets comes back structurally identical."""
    graph = GraphRecord(
        nodes={"a": _llm("a")},
        edges=[StaticEdge(source="START", target="a")],
        state={},
    )

    collapsed = collapse_leading_start_offsets(graph)

    assert collapsed.nodes["a"] is graph.nodes["a"]
    assert collapsed.edges[0] is graph.edges[0]


def test_collapse_leading_offsets_preserves_folded_join_residual():
    """Burst zeroes true leading offsets, never mid-graph folded residuals.

    collapse_leading_start_offsets' contract: burst governs only the phase
    START. A folded join residual sits on a node with a real (non-START)
    in-edge -- zeroing it would silently revert the AND-fan-in fold on
    --burst-phase-starts runs.
    """
    graph = GraphRecord(
        nodes={
            "b": _llm("b"),
            "j": _llm("j", min_start=200_000.0),
        },
        edges=[
            StaticEdge(source="START", target="b", min_start_delay_us=3_000_000.0),
            StaticEdge(source="b", target="j", delay_after_predecessor_us=100_000.0),
        ],
        state={},
    )

    out = collapse_leading_start_offsets(graph)

    # leading START-edge offset zeroed; the folded node-level residual survives
    start_edge = next(e for e in out.edges if e.source == "START")
    assert not start_edge.min_start_delay_us
    assert out.nodes["j"].min_start_delay_us == 200_000.0
