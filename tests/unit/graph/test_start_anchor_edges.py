# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start-anchor post-pass over interval-order edges.

``apply_start_anchors`` replaces an overlapped node's interval-order edges with a
single start-anchored edge (``delay_after_predecessor_start_us``) when the node's
``causal_parent_id`` names a parent that is still IN FLIGHT at the node's recorded
start. Nodes whose causal parent had already finished keep their interval-order
edges, and nodes with no / unknown causal parent are untouched. Also covers the
``with_fan_in_inputs`` exclusion of start-anchored edges.
"""

import pytest

from aiperf.dataset.graph.models import LlmNode, StaticEdge
from aiperf.dataset.graph.segment_ir.interval_order import (
    apply_start_anchors,
    build_interval_edges,
    compute_ranks,
)
from aiperf.dataset.graph.segment_ir.trie_content import (
    TrieNode,
    TrieRequest,
    with_fan_in_inputs,
)


def _node(nid, t, api, causal=None):
    n = TrieNode(
        node_id=nid,
        request=TrieRequest(
            hash_ids=[], input_length=64, output_length=8, t=t, api_time=api
        ),
        order=0,
        causal_parent_id=causal,
    )
    n.warped_start = t
    return n


def test_overlapped_child_gets_single_start_anchored_edge():
    p = _node("p", 0.0, 8.0)
    c = _node("c", 2.5, 1.0, causal="p")
    nodes = [p, c]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    apply_start_anchors(nodes, edges)
    (e,) = edges["c"]
    assert e.source == "p"
    assert e.delay_after_predecessor_start_us == pytest.approx(2.5e6)
    assert e.delay_after_predecessor_us is None


def test_non_overlapped_child_keeps_interval_edges():
    p = _node("p", 0.0, 1.0)
    c = _node("c", 2.5, 1.0, causal="p")  # p ended at 1.0 < 2.5: no overlap
    nodes = [p, c]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    before = list(edges["c"])
    apply_start_anchors(nodes, edges)
    assert edges["c"] == before
    assert edges["c"][0].delay_after_predecessor_start_us is None


def test_missing_or_unknown_causal_parent_untouched():
    a = _node("a", 0.0, 1.0)
    b = _node("b", 0.5, 1.0, causal="ghost")
    nodes = [a, b]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    before = {k: list(v) for k, v in edges.items()}
    apply_start_anchors(nodes, edges)
    assert edges == before


def test_delay_computed_on_warped_clock():
    p = _node("p", 100.0, 8.0)
    c = _node("c", 103.0, 1.0, causal="p")
    p.warped_start, c.warped_start = 10.0, 13.0  # warp compressed by 90s
    nodes = [p, c]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)
    apply_start_anchors(nodes, edges)
    assert edges["c"][0].delay_after_predecessor_start_us == pytest.approx(3.0e6)


def test_fan_in_inputs_skip_start_anchored_edges():
    llm = LlmNode(prompt=[], output="c_out")
    edges = [
        StaticEdge(source="p", target="c", delay_after_predecessor_start_us=1e6),
        StaticEdge(source="q", target="c", delay_after_predecessor_us=0.0),
    ]
    out = with_fan_in_inputs(llm, edges)
    assert [r.channel for r in out.inputs] == ["q_out"]
