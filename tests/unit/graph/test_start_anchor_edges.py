# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start-anchor post-pass over interval-order edges: only an in-flight parent anchors a child."""

from __future__ import annotations

import pytest

from aiperf.dataset.graph.models import LlmNode, StaticEdge
from aiperf.dataset.graph.segment_trie.interval_order import (
    apply_start_anchors,
    build_interval_edges,
    compute_ranks,
)
from aiperf.dataset.graph.segment_trie.trie_content import (
    TrieNode,
    TrieRequest,
    with_fan_in_inputs,
)

_EdgeMap = dict[str, list[StaticEdge]]


def _node(nid: str, t: float, api: float, causal: str | None = None) -> TrieNode:
    """A trie node starting at wall-clock ``t`` and occupying ``api`` seconds."""
    node = TrieNode(
        node_id=nid,
        request=TrieRequest(
            hash_ids=[], input_length=64, output_length=8, t=t, api_time=api
        ),
        order=0,
        causal_parent_id=causal,
    )
    node.warped_start = t
    return node


def _anchored_edges(nodes: list[TrieNode]) -> _EdgeMap:
    """Run the full rank -> interval-edge -> start-anchor pipeline over ``nodes``."""
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    apply_start_anchors(nodes, edges)
    return edges


def test_overlapped_child_gets_single_start_anchored_edge() -> None:
    """A child launched while its causal parent is still in flight gets one START-anchored edge."""
    # Parent spans 0.0-8.0, child starts at 2.5, so waiting for completion would
    # serialize what the recording shows as overlapping.
    nodes = [_node("p", 0.0, 8.0), _node("c", 2.5, 1.0, causal="p")]

    edges = _anchored_edges(nodes)

    (edge,) = edges["c"]
    assert edge.source == "p"
    assert edge.delay_after_predecessor_start_us == pytest.approx(2.5e6)
    assert edge.delay_after_predecessor_us is None


def test_non_overlapped_child_keeps_interval_edges() -> None:
    """A child starting after its parent finished keeps its completion-ordered edges."""
    nodes = [_node("p", 0.0, 1.0), _node("c", 2.5, 1.0, causal="p")]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    before = list(edges["c"])

    apply_start_anchors(nodes, edges)

    assert edges["c"] == before
    assert edges["c"][0].delay_after_predecessor_start_us is None


def test_missing_or_unknown_causal_parent_untouched() -> None:
    """A causal parent id that names no known node leaves the whole edge map unchanged."""
    nodes = [_node("a", 0.0, 1.0), _node("b", 0.5, 1.0, causal="ghost")]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    before = {k: list(v) for k, v in edges.items()}

    apply_start_anchors(nodes, edges)

    assert edges == before


def test_delay_computed_on_warped_clock() -> None:
    """The anchor delay uses warped starts, so idle-gap warping is not double-counted."""
    parent, child = _node("p", 100.0, 8.0), _node("c", 103.0, 1.0, causal="p")
    parent.warped_start, child.warped_start = 10.0, 13.0  # warp compressed by 90s

    edges = _anchored_edges([parent, child])

    assert edges["c"][0].delay_after_predecessor_start_us == pytest.approx(3.0e6)


def test_fan_in_inputs_skip_start_anchored_edges() -> None:
    """AND-fan-in requirements come from completion edges only, never start-anchored ones."""
    llm = LlmNode(prompt=[], output="c_out")
    edges = [
        StaticEdge(source="p", target="c", delay_after_predecessor_start_us=1e6),
        StaticEdge(source="q", target="c", delay_after_predecessor_us=0.0),
    ]

    out = with_fan_in_inputs(llm, edges)

    assert [r.channel for r in out.inputs] == ["q_out"]
