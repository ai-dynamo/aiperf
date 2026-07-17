# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R6 wiring unit tests: input-type -> GRAPH_IR + graph-store build path symmetry.

These pin the three R6 seams that connect the existing graph components into a
real run WITHOUT standing up the full multiprocess pipeline:

* a weka input file is detected as a graph workload (``workload_detect``),
* the DatasetManager build store and the worker read client resolve the SAME
  graph-store directory from ``(base_path, benchmark_id)`` -- the symmetry
  the worker depends on to find what the build wrote, and
* build-plane unified-store ordinals match the dispatch-time catalog ordinals.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from aiperf.dataset.graph.adapters.weka.trace import (
    WekaTraceAdapter,
    from_weka_trace,
)
from aiperf.dataset.graph.graph_path_catalog import (
    build_catalog_context,
    node_ordinal_for,
)
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def test_weka_fixture_detected_as_graph_workload():
    assert WekaTraceAdapter.can_load(FIX) is True


@pytest.mark.asyncio
async def test_build_envelope_ordinals_match_dispatch_catalog():
    # weka always parses to the segment-trie IR (``segment_pool`` set), so the
    # build plane writes per-node manifests via
    # ``build_unified_trie_store_interned`` and the dispatch plane resolves the
    # SAME trie ordinals via ``build_catalog_context`` (which detects the trie
    # and reuses the shared ordinal scheme).
    parsed = from_weka_trace(str(FIX), content_root_seed=42)
    assert parsed.segment_pool is not None, "weka now always builds the trie IR"
    catalog = build_catalog_context(parsed)
    with tempfile.TemporaryDirectory() as d:
        store = GraphSegmentUnifiedBackingStore(base_path=d, benchmark_id="b1")
        addr = await build_unified_trie_store_interned(parsed, store)

        client = GraphSegmentUnifiedClient(base_path=d, benchmark_id="b1").open()
        trace_id = parsed.traces[0].id
        node_map = addr[trace_id]
        assert node_map, "expected per-node ordinals"
        for node_key, ordinal in node_map.items():
            # The build ordinal is what the dispatch adapter resolves via the
            # catalog -- they MUST agree or the worker reads the wrong manifest.
            assert node_ordinal_for(catalog, trace_id, node_key) == ordinal
            assert client.get_node_envelope(trace_id, ordinal, "profiling") is not None
        client.close()
