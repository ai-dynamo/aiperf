# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo store-build parity: the streamed payload drain must equal the eager interned build and the prefix-cache map must survive the structural handoff -- the invariants the dynamo streaming gate flip relies on."""

from __future__ import annotations

from pathlib import Path

import msgspec
import orjson
import pytest

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.codecs import decode_parsed_graph_msgpack
from aiperf.dataset.graph.merge import merge_parsed_graphs
from aiperf.dataset.graph.models import GraphRecord, LlmNode, ParsedGraph
from aiperf.dataset.graph.segment_trie.store_builder import (
    build_unified_trie_store_from_payloads,
    build_unified_trie_store_interned,
    iter_trace_segment_payloads,
)
from aiperf.dataset.graph.store_build import GraphStoreBuilder
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from tests.unit.dataset.conftest import (
    assert_store_dirs_identical,
    write_shared_dynamo_trace,
)


@pytest.fixture
def dyn_trace(tmp_path: Path) -> Path:
    """The canonical 3-record dynamo trace: two prefix-sharing ``s1`` turns plus a standalone ``s2``."""
    return write_shared_dynamo_trace(tmp_path / "dyn_parity.jsonl")


async def _assert_stream_matches_interned_oracle(
    parsed: ParsedGraph, tmp_path: Path, prefix: str
) -> None:
    """Build the same graph both eagerly-interned and via the payload drain, then assert identical catalogs, identical store bytes, and identical worker-side reads."""
    eager_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=f"{prefix}eager"
    )
    eager_catalog = await build_unified_trie_store_interned(parsed, eager_store)

    stream_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=f"{prefix}stream"
    )
    stream_catalog = await build_unified_trie_store_from_payloads(
        iter_trace_segment_payloads(parsed), stream_store
    )

    assert stream_catalog == eager_catalog and eager_catalog

    # Strong equivalence: the persisted unified store is byte-for-byte identical
    # on disk (mirrors ``test_hf_streaming_trie_stores`` eager-vs-streaming
    # oracle), which subsumes the interned handle map, node-manifest region, and
    # content pool -- the whole store, not just the materialized messages.
    assert_store_dirs_identical(
        tmp_path / f"aiperf_graph_segments_{prefix}eager",
        tmp_path / f"aiperf_graph_segments_{prefix}stream",
        why="differs between streaming and eager",
    )

    # Semantic equivalence through the worker read face: the byte-identical
    # stores materialize identical content and agree on the non-handle envelope
    # fields (handles are store-local ints; compare MATERIALIZED content).
    with (
        GraphSegmentUnifiedClient(tmp_path, f"{prefix}eager").open() as ec,
        GraphSegmentUnifiedClient(tmp_path, f"{prefix}stream").open() as sc,
    ):
        for trace_id, ordinals in eager_catalog.items():
            for ordinal in ordinals.values():
                e_raw = ec.get_node_envelope(trace_id, ordinal)
                s_raw = sc.get_node_envelope(trace_id, ordinal)
                assert e_raw is not None and s_raw is not None
                e_env = orjson.loads(e_raw)
                s_env = orjson.loads(s_raw)
                assert ec.materialize_handles(
                    e_env["handles"]
                ) == sc.materialize_handles(s_env["handles"])
                assert {k: v for k, v in e_env.items() if k != "handles"} == {
                    k: v for k, v in s_env.items() if k != "handles"
                }


@pytest.mark.asyncio
async def test_dynamo_payload_stream_store_matches_interned_oracle(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """A streamed dynamo payload drain lands the same store bytes, catalog, and worker-side reads as the eager interned build."""
    parsed = from_dynamo_trace(
        dyn_trace, content_root_seed=1234, content_tokenizer="builtin"
    )
    assert parsed.segment_pool is not None

    await _assert_stream_matches_interned_oracle(parsed, tmp_path, "")


def test_dynamo_prefix_cache_from_structural_matches_eager(dyn_trace: Path) -> None:
    """A prefix-cache map rebuilt from the per-trace structural blobs the drain emits equals the one built off the whole eager parse -- the map must survive the msgpack structural handoff."""
    parsed = from_dynamo_trace(
        dyn_trace, content_root_seed=42, content_tokenizer="builtin"
    )
    eager_map = GraphStoreBuilder._build_graph_prefix_cache_by_trace(parsed)
    assert eager_map, "dynamo fixture must stamp a non-empty prefix-cache map"

    structural_blobs = [
        p.structural_graph
        for p in iter_trace_segment_payloads(parsed)
        if p.structural_graph
    ]
    merged = merge_parsed_graphs(
        decode_parsed_graph_msgpack(b) for b in structural_blobs
    )
    assert GraphStoreBuilder._build_graph_prefix_cache_by_trace(merged) == eager_map


def _with_sentinel_prompts(parsed: ParsedGraph) -> ParsedGraph:
    """Return a copy of ``parsed`` whose every ``LlmNode.prompt`` is a non-empty sentinel, across ``parsed.graph`` AND every ``parsed.graphs`` value."""
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


def _all_llm_nodes(parsed: ParsedGraph) -> list[LlmNode]:
    """Every ``LlmNode`` across the root graph and all subgraphs."""
    return [
        node
        for graph in (parsed.graph, *parsed.graphs.values())
        for node in graph.nodes.values()
        if isinstance(node, LlmNode)
    ]


async def _build_interned_dir(parsed: ParsedGraph, tmp_path: Path, bid: str) -> Path:
    """Build the store via the EAGER interned route and return its directory."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id=bid)
    await build_unified_trie_store_interned(parsed, store)
    return tmp_path / f"aiperf_graph_segments_{bid}"


async def _build_streamed_dir(parsed: ParsedGraph, tmp_path: Path, bid: str) -> Path:
    """Build the store via the STREAMING payload drain and return its directory."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id=bid)
    await build_unified_trie_store_from_payloads(
        iter_trace_segment_payloads(parsed), store
    )
    return tmp_path / f"aiperf_graph_segments_{bid}"


async def _build_direct_dir(
    dyn_trace: Path, tmp_path: Path, bid: str
) -> tuple[Path, ParsedGraph]:
    """Build the store via the DIRECT write-through route and return (dir, parsed)."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id=bid)
    parsed = from_dynamo_trace(
        dyn_trace,
        content_root_seed=1234,
        content_tokenizer="builtin",
        direct_store=store,
    )
    await build_unified_trie_store_interned(parsed, store)
    return tmp_path / f"aiperf_graph_segments_{bid}", parsed


_INLINE_PROMPT_LEAK = (
    "a drain read inline node.prompt instead of the segment pool + trie envelope"
)


@pytest.mark.asyncio
async def test_store_bytes_independent_of_inline_prompt(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """The persisted store bytes are a function of (segment pool, trie envelope) ONLY."""
    parsed = from_dynamo_trace(
        dyn_trace, content_root_seed=1234, content_tokenizer="builtin"
    )
    sentinel = _with_sentinel_prompts(parsed)
    # Guard the guard: the sentinel copy must actually carry non-empty inline
    # prompts, else the byte-equality below would be vacuously true.
    assert _all_llm_nodes(sentinel) and all(
        node.prompt == [{"role": "user", "content": "SENTINEL"}]
        for node in _all_llm_nodes(sentinel)
    )

    base_interned = await _build_interned_dir(parsed, tmp_path, "base-interned")
    sent_interned = await _build_interned_dir(sentinel, tmp_path, "sent-interned")
    assert_store_dirs_identical(base_interned, sent_interned, why=_INLINE_PROMPT_LEAK)

    base_stream = await _build_streamed_dir(parsed, tmp_path, "base-stream")
    sent_stream = await _build_streamed_dir(sentinel, tmp_path, "sent-stream")
    assert_store_dirs_identical(base_stream, sent_stream, why=_INLINE_PROMPT_LEAK)


@pytest.mark.asyncio
async def test_dynamo_release_replay_store_bytes_identical(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """The ``release_replay`` adjunct is a pure build-time RAM optimization."""
    keep = from_dynamo_trace(
        dyn_trace, content_root_seed=1234, content_tokenizer="builtin"
    )
    release = from_dynamo_trace(
        dyn_trace,
        content_root_seed=1234,
        content_tokenizer="builtin",
        release_replay=True,
    )
    keep_dir = await _build_interned_dir(keep, tmp_path, "replay-keep")
    release_dir = await _build_interned_dir(release, tmp_path, "replay-release")
    assert_store_dirs_identical(keep_dir, release_dir, why=_INLINE_PROMPT_LEAK)


@pytest.mark.asyncio
async def test_dynamo_direct_store_route_matches_eager_bytes(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """THREE-WAY parity: the direct write-through store is byte-for-byte identical to the eager interned store."""
    eager_dir = await _build_interned_dir(
        from_dynamo_trace(
            dyn_trace, content_root_seed=1234, content_tokenizer="builtin"
        ),
        tmp_path,
        "eager-3way",
    )
    direct_dir, _parsed = await _build_direct_dir(dyn_trace, tmp_path, "direct-3way")
    assert_store_dirs_identical(eager_dir, direct_dir, why=_INLINE_PROMPT_LEAK)


@pytest.mark.asyncio
async def test_dynamo_direct_store_route_mechanism(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """The direct route returns an empty pool and content-free nodes."""
    _direct_dir, parsed = await _build_direct_dir(dyn_trace, tmp_path, "direct-mech")

    assert parsed.segment_pool is not None
    assert parsed.segment_pool.by_id == {}
    nodes = _all_llm_nodes(parsed)
    assert nodes, "dynamo fixture must lower to at least one LlmNode"
    for node in nodes:
        assert node.prompt == []
        trie_meta = (node.metadata or {}).get("trie") or {}
        assert set(trie_meta) == {"prompt_segment_ids"}, (
            f"node {node.node_id!r} trie metadata must carry only "
            f"prompt_segment_ids, got {sorted(trie_meta)}"
        )


def test_dynamo_direct_route_prefix_cache_matches_eager(dyn_trace: Path) -> None:
    """The direct route stamps the SAME per-node prefix-cache counts as the eager parse: the stamper reads ``node.request.hash_ids``, never the pool, so swapping in the write-through shim must leave the map untouched."""
    eager = from_dynamo_trace(
        dyn_trace, content_root_seed=1234, content_tokenizer="builtin"
    )
    eager_map = GraphStoreBuilder._build_graph_prefix_cache_by_trace(eager)
    assert eager_map, "dynamo fixture must stamp a non-empty prefix-cache map"

    # A throwaway store is fine: we only read the direct-route parse's metadata.
    class _NullStore:
        def put_segment(self, *a: object, **k: object) -> int:
            return 0

    direct = from_dynamo_trace(
        dyn_trace,
        content_root_seed=1234,
        content_tokenizer="builtin",
        direct_store=_NullStore(),  # type: ignore[arg-type]
    )
    assert GraphStoreBuilder._build_graph_prefix_cache_by_trace(direct) == eager_map
