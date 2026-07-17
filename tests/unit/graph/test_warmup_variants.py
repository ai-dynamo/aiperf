# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warmup materialization on the segment-trie IR (Task 11 / adv2 F2).

The trie build plane persists one profiling manifest per node plus the
content-addressed segment pool in the ONE interned unified store
(``build_unified_trie_store_interned``). A WARMUP credit reuses those profiling
bytes and materializes a payload with AgentX's unconditional 1-token output
cap, avoiding a duplicate warmup copy in the store.
"""

from pathlib import Path

import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.worker_materialize import materialize_graph_request_unified

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


async def _build_trie_store(
    fixture: Path, tmp_path: Path, benchmark_id: str
) -> tuple[dict[str, dict[str, int]], GraphSegmentUnifiedClient]:
    """Ingest ``fixture`` and drain it into an opened interned unified reader."""
    parsed = from_weka_trace(str(fixture))
    assert parsed.segment_pool is not None, "trie ingest must attach a SegmentPool"

    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=benchmark_id
    )
    addr = await build_unified_trie_store_interned(parsed, store)

    client = GraphSegmentUnifiedClient(
        base_path=tmp_path, benchmark_id=benchmark_id
    ).open()
    return addr, client


@pytest.mark.asyncio
async def test_warmup_materializes_for_every_node(tmp_path):
    addr, client = await _build_trie_store(FIX, tmp_path, "bw")
    t0 = next(iter(addr))
    try:
        for ordinal in addr[t0].values():
            # Warmup has no dedicated manifest -- it reuses the profiling bytes.
            assert client.get_node_envelope(t0, ordinal, "warmup") is None
            assert client.get_node_envelope(t0, ordinal, "profiling") is not None
            warmup = materialize_graph_request_unified(client, t0, ordinal, "warmup")
            assert warmup is not None
            assert warmup["messages"]
    finally:
        client.close()


@pytest.mark.asyncio
async def test_warmup_materialization_caps_output_to_one_token(tmp_path):
    """AgentX parity: warmup UNCONDITIONALLY caps output to 1 token."""
    addr, client = await _build_trie_store(FIX, tmp_path, "bw2")
    t0 = next(iter(addr))
    # The first recorded turn (arrival ordinal 0) carries the recorded out=25.
    ordinal = min(addr[t0].values())
    try:
        warmup = materialize_graph_request_unified(client, t0, ordinal, "warmup")
        profiling = materialize_graph_request_unified(client, t0, ordinal, "profiling")

        assert warmup is not None
        assert profiling is not None
        cap = Environment.GRAPH.WARMUP_MAX_OUTPUT_TOKENS
        assert cap == 1, "WARMUP_MAX_OUTPUT_TOKENS must default to AgentX's 1"
        # The trie manifest carries the recorded ``out`` on ``max_output_tokens``,
        # endpoint-mapped to ``max_completion_tokens`` (modern default); warmup
        # pops the modern field and unconditionally overwrites with the 1-token
        # legacy cap while profiling keeps the recorded out (25).
        assert warmup.get("max_tokens") == cap
        assert "max_completion_tokens" not in warmup
        assert profiling.get("max_completion_tokens") == 25
        assert "max_tokens" not in profiling
        # Warmup reuses the EXACT profiling input prefix -- only the cap differs.
        assert warmup["messages"] == profiling["messages"]
    finally:
        client.close()
