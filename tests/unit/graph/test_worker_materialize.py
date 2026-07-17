# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker-side materialization of a graph node's request on the segment-trie IR.

A trie node's interned manifest carries ``handles`` (an int-handle path into the
unified content pool); the materializer walks that path against the unified
store client to produce ``messages`` -- NO ancestor accumulation, NO reset
handling -- then applies the node's ``dispatch_overrides`` (mapping the
recorded token cap to the wire field) and ``stream`` flag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.worker_materialize import (
    materialize_graph_request_unified,
)

FIX_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"


@pytest.mark.asyncio
async def test_materialize_maps_max_output_tokens_to_wire_field(tmp_path):
    """``max_output_tokens`` -> ``max_tokens`` (legacy) / ``max_completion_tokens``
    (modern) on the trie materialization path.

    A trie node carries its prompt as interned ``handles``; its
    ``dispatch_overrides`` token cap must map to the endpoint-appropriate wire
    token field exactly as on the linear path (the mapping the worker applies
    after walking the handle path).
    """
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="b1")
    handle = store.put_segment("s_u1", "user", "hi")
    store.add_node_manifest_interned(
        "t-1#0", 0, "profiling", [handle], {"max_output_tokens": 25}, False
    )
    await store.finalize()

    client = GraphSegmentUnifiedClient(base_path=tmp_path, benchmark_id="b1").open()
    try:
        legacy = materialize_graph_request_unified(
            client, "t-1#0", 0, "profiling", use_legacy_max_tokens=True
        )
        modern = materialize_graph_request_unified(
            client, "t-1#0", 0, "profiling", use_legacy_max_tokens=False
        )
    finally:
        client.close()

    assert legacy["messages"] == [{"role": "user", "content": "hi"}]
    assert legacy.get("max_tokens") == 25
    assert "max_completion_tokens" not in legacy
    assert "max_output_tokens" not in legacy

    assert modern.get("max_completion_tokens") == 25
    assert "max_tokens" not in modern


@pytest.mark.asyncio
async def test_materialize_missing_node_returns_none(tmp_path):
    """A node ordinal not in the store materializes to None (graceful miss)."""
    parsed = from_weka_trace(str(FIX_MIN))
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="b1")
    await build_unified_trie_store_interned(parsed, store)

    t0 = parsed.traces[0].id
    # The unified client duck-types BOTH the addressing and segment
    # (content) faces, so one client serves both materializers.
    client = GraphSegmentUnifiedClient(base_path=tmp_path, benchmark_id="b1").open()
    try:
        assert materialize_graph_request_unified(client, t0, 9999, "profiling") is None
    finally:
        client.close()
