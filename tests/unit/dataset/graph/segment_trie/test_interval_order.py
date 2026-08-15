# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontier transitive-reduction soundness for the interval-order edge rule."""

from __future__ import annotations

from aiperf.dataset.graph.segment_trie.interval_order import (
    build_interval_edges,
    compute_ranks,
)
from tests.unit.dataset.graph.segment_trie.conftest import trie_node


def test_frontier_reduction_keeps_async_sibling_not_covered_by_main_chain() -> None:
    """S1: dropping an earlier candidate in favour of a later one is only sound when the covering edge between them actually exists."""
    c = trie_node(
        "c", t=5.0, api_time=5.0, warped_start=5.0, async_ancestors=frozenset({"X"})
    )  # in X, ends t=10
    d = trie_node("d", t=11.0, api_time=1.0, warped_start=11.0)  # main chain, t=11..12
    n = trie_node(
        "n", t=13.0, api_time=1.0, warped_start=13.0, async_ancestors=frozenset({"X"})
    )  # in X, starts t=13
    nodes = [c, d, n]
    compute_ranks(nodes)
    edges = build_interval_edges(nodes)

    n_sources = {e.source for e in edges["n"]}
    assert "c" in n_sources, "c dropped despite no covering c -> d edge existing"
    assert n_sources == {"c", "d"}
    # d async-excludes c, so d has no predecessors at all -- the covering edge
    # the unsound reduction assumed cannot exist.
    assert {e.source for e in edges["d"]} == {"START"}
