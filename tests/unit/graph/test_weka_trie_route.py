# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Routing gate: :func:`from_weka_trace` builds the segment-trie IR.

The dependency-only segment-trie builder (:func:`build_trie_graph`) is the
SOLE weka path: the returned :class:`ParsedGraph` carries ONLY :class:`LlmNode`
nodes and :class:`StaticEdge` edges.
"""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.models import LlmNode, ParsedGraph, StaticEdge

FIX_SUBAGENT = Path(__file__).parent / "fixtures" / "weka_subagent.json"


def _all_nodes(pg: ParsedGraph) -> list:
    """Every node across the top-level graph and any subgraphs."""
    nodes: list = list(pg.graph.nodes.values())
    return nodes


def _all_edges(pg: ParsedGraph) -> list:
    """Every edge across the top-level graph and any subgraphs."""
    edges: list = list(pg.graph.edges)
    return edges


def test_weka_parse_routes_to_trie_builder(monkeypatch) -> None:
    """Every weka parse -> graph is purely LlmNode + StaticEdge (trie IR).

    Hermetic: ``build_trie_graph``'s default callbacks build a real gpt2-backed
    :class:`CorpusContentSynthesizer`. ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE``
    pin the tokenizer load to the local HuggingFace cache so the unit run never
    issues a live ``huggingface.co`` HEAD -- the same offline mechanism the other
    ``tests/unit/graph`` weka tokenizer paths use.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    pg = from_weka_trace(FIX_SUBAGENT)

    nodes = _all_nodes(pg)
    edges = _all_edges(pg)
    assert nodes, "trie graph must have at least one node"
    assert all(isinstance(n, LlmNode) for n in nodes), [type(n).__name__ for n in nodes]
    assert all(isinstance(e, StaticEdge) for e in edges), [
        type(e).__name__ for e in edges
    ]
    # The trie builder produces no subgraphs (no subagent primitives).
