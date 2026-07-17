# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-path/node-ordinal catalog over the weka segment-trie ``ParsedGraph``.

The trie IR is a single flat top graph of ``LlmNode``s (one per recorded
request, no subgraphs/namespaces); :func:`build_graph_path_catalog` resolves each
fired node to the SAME dense ordinal the build-plane segment store was written at
(recorded ``arrival_offset_us`` order, ties broken by node id).
"""

from pathlib import Path

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.graph_path_catalog import build_graph_path_catalog
from aiperf.dataset.graph.models import LlmNode, resolve_trace_graph

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def test_catalog_top_only_is_stable_and_dense() -> None:
    """The trie catalog assigns dense, unique, stable ordinals in arrival order."""
    parsed = from_weka_trace(str(FIX))
    cat = build_graph_path_catalog(parsed)
    trace = parsed.traces[0]
    assert trace.id in cat
    keys = cat[trace.id]

    # One ordinal per recorded-request LlmNode; dense ``0..N-1`` and unique.
    graph = resolve_trace_graph(parsed, trace)
    llm_ids = {nid for nid, n in graph.nodes.items() if isinstance(n, LlmNode)}
    assert set(keys) == llm_ids
    assert sorted(keys.values()) == list(range(len(keys)))

    # Ordinals follow recorded arrival order (the three weka_min turns at
    # t=0/1.5/3.0 fire in that order).
    by_offset = sorted(llm_ids, key=lambda nid: graph.nodes[nid].arrival_offset_us or 0)
    assert [keys[nid] for nid in by_offset] == list(range(len(by_offset)))

    # Stable across runs.
    assert build_graph_path_catalog(parsed)[trace.id] == keys
