# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontier transitive-reduction soundness for the interval-order edge rule."""

from __future__ import annotations

from aiperf.dataset.graph.segment_ir.interval_order import (
    build_interval_edges,
    compute_ranks,
)
from aiperf.dataset.graph.segment_ir.trie_content import TrieNode, TrieRequest


def _node(
    nid: str, t: float, api: float, async_ancestors: set[str] | None = None
) -> TrieNode:
    node = TrieNode(
        node_id=nid,
        request=TrieRequest(
            hash_ids=[], input_length=0, output_length=0, t=t, api_time=api
        ),
        order=0,
        async_ancestors=frozenset(async_ancestors or set()),
    )
    node.warped_start = t
    return node


def test_frontier_reduction_keeps_async_sibling_not_covered_by_main_chain() -> None:
    """Dropping candidate c for a later candidate d is only sound when the
    covering edge c -> d exists -- i.e. when d does NOT async-exclude c.

    Counterexample (S1): node n inside async subtree X, sibling c in X ending
    t=10, main-chain d (not in X) running t=11..12, n starting t=13. No c -> d
    edge exists anywhere (d excludes c), so c must stay in n's frontier or the
    recorded c-before-n ordering inside the subtree is silently lost.
    """
    c = _node("c", 5.0, 5.0, {"X"})  # in X, ends t=10
    d = _node("d", 11.0, 1.0)  # main chain, t=11..12
    n = _node("n", 13.0, 1.0, {"X"})  # in X, starts t=13
    nodes = [c, d, n]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)

    n_sources = {e.source for e in edges["n"]}
    assert "c" in n_sources, "c dropped despite no covering c -> d edge existing"
    assert n_sources == {"c", "d"}
    # d async-excludes c, so d has no predecessors at all -- the covering edge
    # the unsound reduction assumed cannot exist.
    assert {e.source for e in edges["d"]} == {"START"}
