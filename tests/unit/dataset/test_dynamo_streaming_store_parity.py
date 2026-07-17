# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo/native store-build parity: the streamed payload drain must equal the
eager interned build, and the prefix-cache map must survive the structural
handoff -- the invariants the dynamo/native streaming gate flip relies on."""

from __future__ import annotations

from pathlib import Path

import msgspec
import orjson
import pytest

from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.codecs import decode_parsed_graph_msgpack
from aiperf.dataset.graph.merge import merge_parsed_graphs
from aiperf.dataset.graph.models import GraphRecord, LlmNode, ParsedGraph
from aiperf.dataset.graph.parser import parse_native
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_from_payloads,
    build_unified_trie_store_interned,
    graph_carries_assembly_slots,
    iter_trace_segment_payloads,
)
from aiperf.dataset.graph.store_build import GraphStoreBuilder
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)


def _dynamo_record(ts: int, sid: str, input_tokens: int, hashes: list[int]) -> dict:
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": {"session_id": sid},
        "request": {
            "request_id": f"r{ts}",
            "model": "m",
            "input_tokens": input_tokens,
            "output_tokens": 8,
            "cached_tokens": 0,
            "replay": {
                "trace_block_size": 16,
                "input_length": input_tokens,
                "input_sequence_hashes": hashes,
            },
        },
    }


@pytest.fixture
def dyn_trace(tmp_path: Path) -> Path:
    p = tmp_path / "dyn_parity.jsonl"
    records = [
        _dynamo_record(1000, "s1", 32, [111, 222]),
        _dynamo_record(2000, "s1", 64, [111, 222, 333, 444]),
        _dynamo_record(3000, "s2", 48, [555, 666, 777]),
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


@pytest.mark.asyncio
async def test_dynamo_payload_stream_store_matches_interned_oracle(
    dyn_trace: Path, tmp_path: Path
) -> None:
    parsed = from_dynamo_trace(
        dyn_trace, content_root_seed=1234, content_tokenizer="builtin"
    )
    assert parsed.segment_pool is not None

    eager_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="eager"
    )
    eager_catalog = await build_unified_trie_store_interned(parsed, eager_store)

    stream_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="stream"
    )
    stream_catalog = await build_unified_trie_store_from_payloads(
        iter_trace_segment_payloads(parsed), stream_store
    )

    assert stream_catalog == eager_catalog and eager_catalog

    # Strong equivalence: the persisted unified store is byte-for-byte identical
    # on disk (mirrors ``test_hf_streaming_trie_stores`` eager-vs-streaming
    # oracle), which subsumes the interned handle map, node-manifest region, and
    # content pool -- the whole store, not just the materialized messages.
    eager_dir = tmp_path / "aiperf_graph_segments_eager"
    stream_dir = tmp_path / "aiperf_graph_segments_stream"
    eager_files = sorted(p.name for p in eager_dir.iterdir())
    stream_files = sorted(p.name for p in stream_dir.iterdir())
    assert eager_files == stream_files and eager_files, (
        f"unified store file sets differ: {eager_files} vs {stream_files}"
    )
    for name in eager_files:
        assert (eager_dir / name).read_bytes() == (stream_dir / name).read_bytes(), (
            f"unified store file {name!r} differs between streaming and eager"
        )

    # Semantic equivalence through the worker read face: the byte-identical
    # stores materialize identical content and agree on the non-handle envelope
    # fields (handles are store-local ints; compare MATERIALIZED content).
    with (
        GraphSegmentUnifiedClient(tmp_path, "eager").open() as ec,
        GraphSegmentUnifiedClient(tmp_path, "stream").open() as sc,
    ):
        for trace_id, ordinals in eager_catalog.items():
            for ordinal in ordinals.values():
                e_raw = ec.get_node_envelope(trace_id, ordinal, "profiling")
                s_raw = sc.get_node_envelope(trace_id, ordinal, "profiling")
                assert e_raw is not None and s_raw is not None
                e_env = orjson.loads(e_raw)
                s_env = orjson.loads(s_raw)
                assert ec.materialize_handles(
                    e_env["handles"]
                ) == sc.materialize_handles(s_env["handles"])
                assert {k: v for k, v in e_env.items() if k != "handles"} == {
                    k: v for k, v in s_env.items() if k != "handles"
                }


def test_dynamo_prefix_cache_from_structural_matches_eager(dyn_trace: Path) -> None:
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
    """Return a copy of ``parsed`` whose every ``LlmNode.prompt`` is a non-empty
    sentinel, across ``parsed.graph`` AND every ``parsed.graphs`` value.

    A drain that read the inline ``node.prompt`` (rather than the segment pool +
    trie envelope) would emit different bytes for this copy than for the
    real-content baseline. Sentinelling both graph surfaces keeps multi-graph
    corpus shapes pinned even though the dynamo fixture is single-graph.
    """
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
    return [
        node
        for graph in (parsed.graph, *parsed.graphs.values())
        for node in graph.nodes.values()
        if isinstance(node, LlmNode)
    ]


async def _build_interned_dir(parsed: ParsedGraph, tmp_path: Path, bid: str) -> Path:
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id=bid)
    await build_unified_trie_store_interned(parsed, store)
    return tmp_path / f"aiperf_graph_segments_{bid}"


async def _build_streamed_dir(parsed: ParsedGraph, tmp_path: Path, bid: str) -> Path:
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id=bid)
    await build_unified_trie_store_from_payloads(
        iter_trace_segment_payloads(parsed), store
    )
    return tmp_path / f"aiperf_graph_segments_{bid}"


async def _build_direct_dir(
    dyn_trace: Path, tmp_path: Path, bid: str
) -> tuple[Path, ParsedGraph]:
    """Build the store via the DIRECT write-through route and return (dir, parsed).

    The store is constructed FIRST and threaded as ``direct_store`` so
    ``build_trie_ir``'s ``pool.add`` interns each segment straight into it during
    the parse; the returned pool is an EMPTY pool (the per-tree write-through
    shims carry no ``by_id`` and the multi-graph merge normalizes them to a plain
    empty ``SegmentPool``) and the interned drain's put loop no-ops over it
    (segments already resident). The same content_root_seed / tokenizer as the
    eager/streaming legs so all three parses are the identical trie.
    """
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id=bid)
    parsed = from_dynamo_trace(
        dyn_trace,
        content_root_seed=1234,
        content_tokenizer="builtin",
        direct_store=store,
    )
    await build_unified_trie_store_interned(parsed, store)
    return tmp_path / f"aiperf_graph_segments_{bid}", parsed


def _assert_store_dirs_identical(dir_a: Path, dir_b: Path) -> None:
    files_a = sorted(p.name for p in dir_a.iterdir())
    files_b = sorted(p.name for p in dir_b.iterdir())
    assert files_a == files_b and files_a, (
        f"unified store file sets differ: {files_a} vs {files_b}"
    )
    for name in files_a:
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes(), (
            f"unified store file {name!r} differs -- a drain read inline "
            f"node.prompt instead of the segment pool + trie envelope"
        )


@pytest.mark.asyncio
async def test_store_bytes_independent_of_inline_prompt(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """The persisted store bytes are a function of (segment pool, trie envelope)
    ONLY -- never the inline ``LlmNode.prompt``.

    Pins the trie-route invariant that no drain reads ``node.prompt``: a
    real-content baseline parse and a copy whose every node carries a sentinel
    non-empty prompt build BYTE-IDENTICAL stores through BOTH the eager
    (``build_unified_trie_store_interned``) and streaming
    (``build_unified_trie_store_from_payloads``) drains. Holds before AND after
    adapters stamp ``prompt=[]``, so the pin survives the change permanently.
    """
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
    _assert_store_dirs_identical(base_interned, sent_interned)

    base_stream = await _build_streamed_dir(parsed, tmp_path, "base-stream")
    sent_stream = await _build_streamed_dir(sentinel, tmp_path, "sent-stream")
    _assert_store_dirs_identical(base_stream, sent_stream)


@pytest.mark.asyncio
async def test_dynamo_release_replay_store_bytes_identical(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """The ``release_replay`` adjunct is a pure build-time RAM optimization: a
    parse with replay-release ON produces a BYTE-IDENTICAL unified store to one
    with it OFF.

    ``dynamo_trie_nodes`` copies each record's recorded hashes / input_length
    into the ``TrieRequest`` BEFORE freeing ``req.replay``, so the trie IR --
    and therefore the persisted store (content pool + node manifests) -- is
    unchanged. Each ``from_dynamo_trace`` call reads fresh chains from disk, so
    the release on one parse never affects the other.
    """
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
    _assert_store_dirs_identical(keep_dir, release_dir)


@pytest.mark.asyncio
async def test_dynamo_direct_store_route_matches_eager_bytes(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """THREE-WAY parity: the direct write-through store is byte-for-byte identical
    to the eager interned store.

    ``test_dynamo_payload_stream_store_matches_interned_oracle`` already proves
    eager == streaming on the SAME store files; this proves direct == eager on
    the same files, so by transitivity **direct == eager == streaming**. Both the
    direct and eager routes intern in ``build_trie_ir``'s content-loop
    first-occurrence order (the single ordering authority), so the write-through
    sink assigns the same handle stream -- the parity holds by construction, not
    by luck.
    """
    eager_dir = await _build_interned_dir(
        from_dynamo_trace(
            dyn_trace, content_root_seed=1234, content_tokenizer="builtin"
        ),
        tmp_path,
        "eager-3way",
    )
    direct_dir, _parsed = await _build_direct_dir(dyn_trace, tmp_path, "direct-3way")
    _assert_store_dirs_identical(eager_dir, direct_dir)


@pytest.mark.asyncio
async def test_dynamo_direct_store_route_mechanism(
    dyn_trace: Path, tmp_path: Path
) -> None:
    """The direct route returns an empty pool and content-free nodes: segments
    live in the store, not the pool.

    The DM-level mechanism pins (original §1 ruling): the returned pool is EMPTY
    -- the write-through shim carries no ``by_id`` and the multi-graph merge
    (:func:`merge_parsed_graphs`) normalizes the per-tree shims to a plain empty
    ``SegmentPool`` -- so the interned drain's put loop no-ops over it, every
    ``LlmNode.prompt`` was stamped ``[]`` at lowering, and ``metadata["trie"]``
    carries ONLY ``prompt_segment_ids`` (the load-bearing store path --
    ``hash_ids`` and inline content are gone).
    """
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
    """The direct route stamps the SAME per-node prefix-cache counts as the eager
    parse (template: ``test_dynamo_prefix_cache_from_structural_matches_eager``).

    ``stamp_theoretical_prefix_cache`` reads ``node.request.hash_ids``, never the
    pool, so swapping in the write-through shim leaves the prefix-cache map
    untouched -- pinned here on the direct route so a future shim change that
    perturbed lowering would be caught.
    """
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


@pytest.fixture
def native_workload(tmp_path: Path) -> Path:
    p = tmp_path / "native_parity.yaml"
    p.write_text(
        """
graph:
  nodes:
    a:
      prompt:
        - {role: user, content: "question one"}
      output: a_out
    b:
      prompt:
        - {role: user, content: "question two"}
      output: b_out
  edges:
    - {source: START, target: a}
    - {source: a, target: b}
    - {source: b, target: END}
traces:
  - id: t1
  - id: t2
"""
    )
    return p


@pytest.mark.asyncio
async def test_native_payload_stream_store_matches_interned_oracle(
    native_workload: Path, tmp_path: Path
) -> None:
    parsed = parse_native(native_workload)
    assert parsed.segment_pool is not None
    # Slot-free premise: plain native graphs stream with no eager fallback.
    assert not graph_carries_assembly_slots(parsed)

    eager_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="native-eager"
    )
    eager_catalog = await build_unified_trie_store_interned(parsed, eager_store)

    stream_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="native-stream"
    )
    stream_catalog = await build_unified_trie_store_from_payloads(
        iter_trace_segment_payloads(parsed), stream_store
    )

    assert stream_catalog == eager_catalog and eager_catalog

    eager_dir = tmp_path / "aiperf_graph_segments_native-eager"
    stream_dir = tmp_path / "aiperf_graph_segments_native-stream"
    eager_files = sorted(p.name for p in eager_dir.iterdir())
    stream_files = sorted(p.name for p in stream_dir.iterdir())
    assert eager_files == stream_files and eager_files, (
        f"unified store file sets differ: {eager_files} vs {stream_files}"
    )
    for name in eager_files:
        assert (eager_dir / name).read_bytes() == (stream_dir / name).read_bytes(), (
            f"unified store file {name!r} differs between streaming and eager"
        )

    with (
        GraphSegmentUnifiedClient(tmp_path, "native-eager").open() as ec,
        GraphSegmentUnifiedClient(tmp_path, "native-stream").open() as sc,
    ):
        for trace_id, ordinals in eager_catalog.items():
            for ordinal in ordinals.values():
                e_raw = ec.get_node_envelope(trace_id, ordinal, "profiling")
                s_raw = sc.get_node_envelope(trace_id, ordinal, "profiling")
                assert e_raw is not None and s_raw is not None
                e_env = orjson.loads(e_raw)
                s_env = orjson.loads(s_raw)
                assert ec.materialize_handles(
                    e_env["handles"]
                ) == sc.materialize_handles(s_env["handles"])
                assert {k: v for k, v in e_env.items() if k != "handles"} == {
                    k: v for k, v in s_env.items() if k != "handles"
                }

    # Native never stamps prefix-cache counts (hash-id-free graphs); pin the
    # documented absence rather than a non-empty map.
    assert GraphStoreBuilder._build_graph_prefix_cache_by_trace(parsed) == {}
