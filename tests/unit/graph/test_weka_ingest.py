# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import msgspec
import pytest

from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.models import GraphRecord, LlmNode, ParsedGraph
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_from_payloads,
    build_unified_trie_store_interned,
    iter_trace_segment_payloads,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
)

FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def test_from_weka_trace_yields_parsed_graph_real_content():
    # The segment-trie IR is the only weka path: ingest attaches a SegmentPool and
    # every LlmNode carries a ``prompt_segment_ids`` path that materializes to real
    # (non-placeholder) conversation text -- no legacy replay_outputs/subgraphs.
    parsed = from_weka_trace(str(FIX))
    assert parsed.traces, "expected at least one trace"
    assert parsed.segment_pool is not None, "trie ingest must attach a SegmentPool"

    pool = parsed.segment_pool
    graph = parsed.graph
    llm_nodes = [n for n in graph.nodes.values() if isinstance(n, LlmNode)]
    assert llm_nodes, "expected at least one LlmNode"

    for node in llm_nodes:
        trie = node.metadata.get("trie")
        assert trie is not None, "trie node must carry trie metadata"
        path = trie["prompt_segment_ids"]
        assert path, "trie node must carry a non-empty prompt segment path"

        # The trie route carries NO inline prompt: content lives ONLY in the
        # segment pool, reached via the prompt_segment_ids path.
        assert node.prompt == []
        materialized = pool.materialize(path)
        # Real content, not the empty/placeholder prompt: every segment has text.
        for msg in materialized:
            assert msg["role"] in {"system", "user", "assistant", "tool"}
            assert isinstance(msg["content"], str) and msg["content"]

        # The node's recorded response is a real assistant pool entry chained
        # onto the prompt tip (content-addressed; no metadata handle needed).
        tip = path[-1]
        assert any(
            s.role == "assistant" and s.parent_id == tip for s in pool.by_id.values()
        )


def _with_sentinel_prompts(parsed: ParsedGraph) -> ParsedGraph:
    """Copy ``parsed`` with a non-empty sentinel on every LlmNode.prompt, across
    ``parsed.graph`` AND every ``parsed.graphs`` value."""
    sentinel: list = [{"role": "user", "content": "SENTINEL"}]

    def _stamp(graph: GraphRecord) -> GraphRecord:
        nodes = {
            nid: (
                msgspec.structs.replace(node, prompt=sentinel)
                if isinstance(node, LlmNode)
                else node
            )
            for nid, node in graph.nodes.items()
        }
        return msgspec.structs.replace(graph, nodes=nodes)

    return msgspec.structs.replace(
        parsed,
        graph=_stamp(parsed.graph),
        graphs={ref: _stamp(g) for ref, g in parsed.graphs.items()},
    )


@pytest.mark.asyncio
async def test_weka_store_bytes_independent_of_inline_prompt(tmp_path: Path) -> None:
    """Weka trie store bytes are a function of (segment pool, trie envelope) ONLY
    -- never the inline ``LlmNode.prompt`` -- through BOTH the eager and streaming
    drains. Mirrors the dynamo pin (``test_dynamo_streaming_store_parity``) on a
    weka-shaped parse so the ``prompt=[]`` convention is pinned per format."""
    parsed = from_weka_trace(str(FIX))
    sentinel = _with_sentinel_prompts(parsed)
    # Guard the guard: EVERY LlmNode in the sentinel copy must carry the non-empty
    # sentinel prompt -- an ``all(... == sentinel)`` (not ``any``) so a partial
    # ``_stamp`` failure cannot pass this vacuously (mirrors the dynamo twin in
    # test_dynamo_streaming_store_parity.py).
    sentinel_nodes = [
        node
        for graph in (sentinel.graph, *sentinel.graphs.values())
        for node in graph.nodes.values()
        if isinstance(node, LlmNode)
    ]
    assert sentinel_nodes and all(
        node.prompt == [{"role": "user", "content": "SENTINEL"}]
        for node in sentinel_nodes
    )

    async def _dirs(p: ParsedGraph, tag: str) -> tuple[Path, Path]:
        interned = GraphSegmentUnifiedBackingStore(
            base_path=tmp_path, benchmark_id=f"{tag}-i"
        )
        await build_unified_trie_store_interned(p, interned)
        stream = GraphSegmentUnifiedBackingStore(
            base_path=tmp_path, benchmark_id=f"{tag}-s"
        )
        await build_unified_trie_store_from_payloads(
            iter_trace_segment_payloads(p), stream
        )
        return (
            tmp_path / f"aiperf_graph_segments_{tag}-i",
            tmp_path / f"aiperf_graph_segments_{tag}-s",
        )

    base_i, base_s = await _dirs(parsed, "base")
    sent_i, sent_s = await _dirs(sentinel, "sent")

    for real_dir, sent_dir in ((base_i, sent_i), (base_s, sent_s)):
        names = sorted(p.name for p in real_dir.iterdir())
        assert names == sorted(p.name for p in sent_dir.iterdir()) and names
        for name in names:
            assert (real_dir / name).read_bytes() == (sent_dir / name).read_bytes(), (
                f"weka store file {name!r} differs -- a drain read inline "
                f"node.prompt instead of the segment pool + trie envelope"
            )
