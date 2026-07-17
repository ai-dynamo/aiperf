# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF-streaming trie store build path (Task T8-hf).

The HF ``--public-dataset`` build is STREAMED per row (the parent never holds
the whole corpus's synthesized real content). Before this task the streaming
worker dropped ``pg.segment_pool`` and emitted legacy ``messages_delta``
envelopes with NO segment store -- producing corrupt / empty prompts.

These tests drive the streaming trie path directly through its
worker+consumer functions (``iter_item_segment_payloads`` ->
``build_unified_trie_store_from_payloads``) on a small set of synthetic weka
rows and prove:

* the streamed payloads carry ``prompt_segment_ids`` envelopes (NOT legacy
  ``messages_delta``), and the streamed catalog ordinals match
  ``trie_node_ordinals``;
* worker ``materialize_graph_request_unified`` against the STREAMED unified
  store reproduces the SAME prompt as the EAGER interned unified store for the
  same rows + seed (byte-equal, eager store as oracle) -- and the two stores
  are byte-for-byte identical on disk.

Hermetic: the ``fake_tokenizer`` fixture pins ``Tokenizer.from_pretrained`` to
the deterministic ``FakeTokenizer`` so both the eager and streaming sides
synthesize identical content from the same seed -- no network, no real gpt2.
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.dataset.graph.adapters.weka.trace_parallel import (
    iter_item_segment_payloads,
    parse_items,
    row_work_items,
)
from aiperf.dataset.graph.models import LlmNode
from aiperf.dataset.graph.segment_ir.store_builder import (
    TraceSegmentPayload,
    build_unified_trie_store_from_payloads,
    build_unified_trie_store_interned,
    trie_node_ordinals,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.worker_materialize import materialize_graph_request_unified

_SEED = 42
_SOURCE = "synthetic/weka-trie"

# Two small synthetic weka rows (the same dict shape a .json file / HF row
# carries). Multi-turn + a subagent so the trie pool has shared-prefix dedup
# and a deepest path; two rows exercise cross-row segment dedup in the consumer.
_ROW_A = {
    "id": "trace_alpha",
    "models": ["claude-opus-4-5-20251101"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {
            "t": 0.0,
            "type": "n",
            "model": "claude-opus-4-5-20251101",
            "in": 180,
            "out": 25,
            "hash_ids": [1, 2],
            "input_types": ["text"],
            "output_types": ["text"],
            "stop": "end_turn",
            "api_time": 0.8,
            "think_time": 0.0,
        },  # noqa: E501
        {
            "t": 1.0,
            "type": "n",
            "model": "claude-opus-4-5-20251101",
            "in": 240,
            "out": 30,
            "hash_ids": [1, 2, 3],
            "input_types": ["text"],
            "output_types": ["text"],
            "stop": "end_turn",
            "api_time": 0.9,
            "think_time": 0.1,
        },  # noqa: E501
    ],
}
_ROW_B = {
    "id": "trace_beta",
    "models": ["claude-opus-4-5-20251101"],
    "block_size": 64,
    "hash_id_scope": "local",
    "requests": [
        {
            "t": 0.0,
            "type": "n",
            "model": "claude-opus-4-5-20251101",
            "in": 180,
            "out": 25,
            "hash_ids": [1, 2],
            "input_types": ["text"],
            "output_types": ["text"],
            "stop": "tool_use",
            "api_time": 0.8,
            "think_time": 0.0,
        },  # noqa: E501
        {
            "t": 1.0,
            "type": "subagent",
            "agent_id": "agent_001",
            "subagent_type": "Explore",
            "duration_ms": 4000,
            "total_tokens": 600,
            "tool_use_count": 1,
            "status": "completed",
            "models": ["claude-opus-4-5-20251101"],
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [
                {
                    "t": 1.2,
                    "type": "n",
                    "model": "claude-opus-4-5-20251101",
                    "in": 200,
                    "out": 30,
                    "hash_ids": [10, 11],
                    "input_types": ["text"],
                    "output_types": ["text"],
                    "stop": "end_turn",
                    "api_time": 0.9,
                    "think_time": 0.0,
                },  # noqa: E501
            ],
        },
        {
            "t": 6.0,
            "type": "n",
            "model": "claude-opus-4-5-20251101",
            "in": 280,
            "out": 45,
            "hash_ids": [1, 2, 3, 4],
            "input_types": ["text"],
            "output_types": ["text"],
            "stop": "end_turn",
            "api_time": 1.3,
            "think_time": 0.5,
        },  # noqa: E501
    ],
}

_PARSE_KWARGS = {
    "tag": "weka",
    "idle_gap_cap_seconds": None,
    "content_root_seed": _SEED,
    "max_osl": None,
}


def _stream_trie_payloads(rows: list[dict]) -> list[TraceSegmentPayload]:
    """Materialize the streaming worker output (serial in-process path)."""
    return list(
        iter_item_segment_payloads(
            row_work_items(rows, _SOURCE),
            source_label=_SOURCE,
            parse_kwargs=dict(_PARSE_KWARGS),
        )
    )


async def _build_eager_store(rows: list[dict], tmp_path, benchmark_id: str):
    """Eager ORACLE: merged ParsedGraph -> interned unified store."""
    parsed = parse_items(
        row_work_items(rows, _SOURCE),
        source_label=_SOURCE,
        parse_kwargs=dict(_PARSE_KWARGS),
    )
    assert parsed.segment_pool is not None, "eager trie parse must surface a pool"

    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=benchmark_id
    )
    catalog = await build_unified_trie_store_interned(parsed, store)
    return parsed, catalog


async def _build_streaming_store(rows: list[dict], tmp_path, benchmark_id: str):
    """Streaming path: per-row trie payloads -> interned unified store."""
    payloads = _stream_trie_payloads(rows)
    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=benchmark_id
    )
    return await build_unified_trie_store_from_payloads(payloads, store)


@pytest.mark.asyncio
async def test_streaming_emits_trie_payloads_not_legacy(
    fake_tokenizer: None,  # noqa: ARG001  # side-effect: deterministic tokenizer
) -> None:
    """The streaming worker emits TraceSegmentPayloads w/ envelopes."""
    payloads = _stream_trie_payloads([_ROW_A, _ROW_B])

    assert payloads, "streaming must yield at least one payload"
    assert all(isinstance(p, TraceSegmentPayload) for p in payloads)
    # At least one payload carries segments (the pool triples) -> a segment store
    # WILL be produced; legacy runs never carry these.
    assert any(p.segments for p in payloads)

    for payload in payloads:
        assert payload.envelopes, f"trace {payload.trace_id} must carry envelopes"
        for node_envelope in payload.envelopes:
            env = orjson.loads(node_envelope.envelope_bytes)
            assert "prompt_segment_ids" in env, "trie envelope, not legacy"
            assert "messages_delta" not in env, "must NOT be a legacy delta envelope"


@pytest.mark.asyncio
async def test_streaming_catalog_matches_trie_node_ordinals(
    fake_tokenizer: None,  # noqa: ARG001
    tmp_path,
) -> None:
    """Streamed catalog == eager catalog == per-trace ``trie_node_ordinals``."""
    parsed, eager_catalog = await _build_eager_store(
        [_ROW_A, _ROW_B], tmp_path / "e", "e"
    )
    stream_catalog = await _build_streaming_store([_ROW_A, _ROW_B], tmp_path / "s", "s")

    assert stream_catalog == eager_catalog, "streaming catalog must equal eager"

    for trace in parsed.traces:
        llm_nodes = {
            nid: n
            for nid, n in parsed.graphs[trace.graph_ref].nodes.items()
            if isinstance(n, LlmNode)
        }
        assert stream_catalog[trace.id] == trie_node_ordinals(llm_nodes)


@pytest.mark.asyncio
async def test_streaming_worker_materialization_byte_equal_to_eager(
    fake_tokenizer: None,  # noqa: ARG001
    tmp_path,
) -> None:
    """CRITICAL: worker materialization over the streamed store == eager, byte-equal.

    For every trie node the worker-side ``materialize_graph_request_unified``
    reads the persisted interned manifest + content pool. The EAGER unified
    store is the oracle: the streamed prompt must match it byte-for-byte (and
    equal the pool ground truth), proving the HF streaming path produces the
    identical materialized request the local eager path does.
    """
    parsed, eager_catalog = await _build_eager_store(
        [_ROW_A, _ROW_B], tmp_path / "e", "e"
    )
    stream_catalog = await _build_streaming_store([_ROW_A, _ROW_B], tmp_path / "s", "s")
    assert stream_catalog == eager_catalog

    pool = parsed.segment_pool
    eager_client = GraphSegmentUnifiedClient(base_path=tmp_path / "e", benchmark_id="e")
    stream_client = GraphSegmentUnifiedClient(
        base_path=tmp_path / "s", benchmark_id="s"
    )
    eager_client.open()
    stream_client.open()
    try:
        checked = 0
        for trace in parsed.traces:
            llm_nodes = {
                nid: n
                for nid, n in parsed.graphs[trace.graph_ref].nodes.items()
                if isinstance(n, LlmNode)
            }
            ordinals = stream_catalog[trace.id]
            for node_id, node in llm_nodes.items():
                ordinal = ordinals[node_id]
                expected = pool.materialize(node.metadata["trie"]["prompt_segment_ids"])
                eager_req = materialize_graph_request_unified(
                    eager_client, trace.id, ordinal, "profiling"
                )
                stream_req = materialize_graph_request_unified(
                    stream_client, trace.id, ordinal, "profiling"
                )
                assert eager_req is not None and stream_req is not None
                assert eager_req["messages"] == expected, node_id
                # Byte-equal eager-vs-streaming: identical materialized request.
                assert stream_req == eager_req, node_id
                checked += 1
        assert checked == sum(len(v) for v in stream_catalog.values())
    finally:
        eager_client.close()
        stream_client.close()


@pytest.mark.asyncio
async def test_streaming_payloads_ship_content_free_structural_graph(
    fake_tokenizer: None,  # noqa: ARG001  # side-effect: deterministic tokenizer
) -> None:
    """Each streamed payload carries a content-free structural graph for the sidecar.

    ``LlmNode.prompt`` (the dominant inline-content field), ``replay_outputs``, and
    the segment pool content are all emptied -- but the pool is kept non-None so the
    loaded graph keeps the trie ordinal scheme -- while topology (nodes/edges) is
    preserved. This is what makes the streaming ``graph_meta`` sidecar bounded.
    """
    from aiperf.dataset.graph.codecs import decode_parsed_graph_msgpack

    payloads = _stream_trie_payloads([_ROW_A, _ROW_B])
    structural = [p.structural_graph for p in payloads if p.structural_graph]
    assert structural, "at least one payload must ship a structural graph"

    for blob in structural:
        pg = decode_parsed_graph_msgpack(blob)
        # Trie ordinal scheme preserved: pool present but content emptied.
        assert pg.segment_pool is not None
        assert not pg.segment_pool._by_id, "structural pool must be emptied"
        # Content-free: every LlmNode.prompt emptied across every graph.
        graphs = [pg.graph, *pg.graphs.values()]
        llm_seen = False
        for g in graphs:
            for node in g.nodes.values():
                if isinstance(node, LlmNode):
                    llm_seen = True
                    assert node.prompt == [], "LlmNode.prompt must be stripped"
        assert llm_seen, "structural graph must retain its LlmNode topology"


@pytest.mark.asyncio
async def test_streaming_structural_sink_matches_store_catalog(
    fake_tokenizer: None,  # noqa: ARG001  # side-effect: deterministic tokenizer
    tmp_path,
) -> None:
    """The merged structural sink rebuilds the SAME catalog the store was built at.

    ``catalogs_match`` is the gate that decides whether the streaming sidecar is
    written; this proves it passes, so the TimingManager loads the sidecar instead
    of the whole-corpus re-parse.
    """
    from aiperf.dataset.graph.codecs import decode_parsed_graph_msgpack
    from aiperf.dataset.graph.graph_meta_sidecar import catalogs_match
    from aiperf.dataset.graph.merge import merge_parsed_graphs

    bid = "sidecar_stream_test"
    payloads = _stream_trie_payloads([_ROW_A, _ROW_B])
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id=bid)
    sink: list[bytes] = []
    catalog = await build_unified_trie_store_from_payloads(
        payloads, store, structural_sink=sink
    )

    assert sink, "structural sink must be populated when requested"
    merged = merge_parsed_graphs(decode_parsed_graph_msgpack(b) for b in sink)
    assert catalogs_match(merged, catalog), (
        "merged structural catalog must equal the store catalog (sidecar gate)"
    )


@pytest.mark.asyncio
async def test_streaming_unified_store_byte_matches_eager_interned(
    fake_tokenizer: None,  # noqa: ARG001  # side-effect: deterministic tokenizer
    tmp_path,
) -> None:
    """Streaming unified store (from payloads) == eager interned unified store, byte-for-byte.

    Proves ``build_unified_trie_store_from_payloads`` (the corpus-scale HF path)
    produces the SAME unified store as the eager
    ``build_unified_trie_store_interned``, so the unified store is the real
    store on both paths.
    """
    rows = [_ROW_A, _ROW_B]

    # Eager interned unified store.
    eager_parsed = parse_items(
        row_work_items(rows, _SOURCE),
        source_label=_SOURCE,
        parse_kwargs=dict(_PARSE_KWARGS),
    )
    assert eager_parsed.segment_pool is not None
    eager_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="eager"
    )
    eager_catalog = await build_unified_trie_store_interned(eager_parsed, eager_store)

    # Streaming unified store from per-row payloads.
    stream_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="stream"
    )
    stream_catalog = await build_unified_trie_store_from_payloads(
        _stream_trie_payloads(rows), stream_store
    )

    assert stream_catalog == eager_catalog, "catalog must match eager interned"

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
