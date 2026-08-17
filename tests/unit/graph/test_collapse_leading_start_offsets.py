# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R3 -- ``collapse_leading_start_offsets`` (the ``--burst-phase-starts`` collapse)."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pytest import param

from aiperf.dataset.graph.models import GraphRecord, LlmNode, StaticEdge
from aiperf.graph.scheduler import collapse_leading_start_offsets


def _llm(output: str, *, min_start: float | None = None) -> LlmNode:
    return LlmNode(prompt=[f"@{output}"], output=output, min_start_delay_us=min_start)


def _graph() -> GraphRecord:
    """START->a (leading offset); a->b (paced, node-level offset); a->c (start-anchored)."""
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


def _root_and_mid_graph() -> GraphRecord:
    """A START-rooted node and a node with a real predecessor, both carrying node-level offsets."""
    return GraphRecord(
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


def _folded_join_residual_graph() -> GraphRecord:
    """A join whose AND-fan-in residual was folded onto its node-level offset."""
    return GraphRecord(
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


def test_collapse_zeroes_start_edge_min_start_delay() -> None:
    """The START-edge leading offset (the t* anchor carrier) is zeroed."""
    collapsed = collapse_leading_start_offsets(_graph())

    start_edges = [e for e in collapsed.edges if e.source == "START"]
    assert len(start_edges) == 1
    assert start_edges[0].min_start_delay_us == 0.0


def test_collapse_preserves_inter_turn_edge_pacing() -> None:
    """Non-START edges keep every recorded delay -- burst governs starts only."""
    collapsed = collapse_leading_start_offsets(_graph())

    by_target = {e.target: e for e in collapsed.edges if e.source != "START"}
    assert by_target["b"].delay_after_predecessor_us == 2_000_000.0
    assert by_target["b"].min_start_delay_us == 1_000_000.0
    assert by_target["c"].delay_after_predecessor_start_us == 3_000_000.0


@pytest.mark.parametrize(
    ("build", "expected_node_offsets"),
    [
        param(
            _root_and_mid_graph,
            {"x": 0.0, "y": 2_000_000.0},
            id="root-node-zeroed-mid-graph-node-kept",
        ),
        param(
            _folded_join_residual_graph,
            {"b": None, "j": 200_000.0},
            id="folded-join-residual-survives",
        ),
    ],
)  # fmt: skip
def test_collapse_zeroes_only_node_level_offsets_without_a_real_predecessor(
    build: Callable[[], GraphRecord], expected_node_offsets: dict[str, float | None]
) -> None:
    """Node-level ``min_start_delay_us`` is leading (zeroed) only on a START-rooted node."""
    # A node keeping a real in-edge carries mid-graph pacing -- notably the
    # AND-fan-in residual fold, which zeroing would silently revert.
    graph = build()

    collapsed = collapse_leading_start_offsets(graph)

    start_edge = next(e for e in collapsed.edges if e.source == "START")
    assert not start_edge.min_start_delay_us
    assert {
        nid: node.min_start_delay_us for nid, node in collapsed.nodes.items()
    } == expected_node_offsets


def test_collapse_is_pure_and_identity_preserving() -> None:
    """The input graph is untouched and untouched nodes/edges keep object identity."""
    graph = _graph()

    collapsed = collapse_leading_start_offsets(graph)

    # Purity: the original still carries its offsets.
    assert graph.edges[0].min_start_delay_us == 5_000_000.0
    assert graph.nodes["b"].min_start_delay_us == 7_000_000.0
    # Identity for untouched members (no needless struct rebuilds).
    assert collapsed.nodes["a"] is graph.nodes["a"]
    assert collapsed.edges[1] is graph.edges[1]


def test_collapse_noop_graph_round_trips() -> None:
    """A graph with no leading offsets comes back structurally identical."""
    graph = GraphRecord(
        nodes={"a": _llm("a")},
        edges=[StaticEdge(source="START", target="a")],
        state={},
    )

    collapsed = collapse_leading_start_offsets(graph)

    assert collapsed.nodes["a"] is graph.nodes["a"]
    assert collapsed.edges[0] is graph.edges[0]
