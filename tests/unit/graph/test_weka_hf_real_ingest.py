# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-dataset ingest of the published SemiAnalysis weka corpus (062126).

Proves the HuggingFace ``org/name`` load path end-to-end on REAL data: a small
pinned slice of ``semianalysisai/cc-traces-weka-062126`` is streamed (so the
~1.8 GB single-file ``traces.jsonl`` is never fully materialized), parsed into a
segment-trie :class:`ParsedGraph` through the SAME shared core the local-file
path uses, and drained into a :class:`GraphSegmentUnifiedBackingStore` via the
same unified-store builders :class:`DatasetManager._configure_graph_workload`
runs (interned eager build + payload-streamed build).

Marked ``slow`` (real-content synthesis loads a tokenizer and synthesizes every
turn) and ``network`` (resolves the corpus from the HuggingFace cache / hub).
The slice + revision are pinned so a re-run is reproducible; the test skips
cleanly when the dataset cannot be resolved (no cache and no network).
"""

from __future__ import annotations

import orjson
import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.weka.trace import (
    WekaTraceAdapterError,
    _looks_like_hf_dataset_id,
    from_weka_trace,
)
from aiperf.dataset.graph.models import LlmNode, ParsedGraph, resolve_trace_graph
from aiperf.dataset.graph.segment_ir.store_builder import (
    TraceSegmentPayload,
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.worker_materialize import materialize_graph_request_unified

# Published SemiAnalysis weka corpus dated 062126. Public, single ``train``
# split exposed as one ~1.8 GB ``traces.jsonl``.
WEKA_HF_REPO_ID = "semianalysisai/cc-traces-weka-062126"

# Pinned commit SHA of the corpus snapshot this test was authored against, so a
# streamed re-run reads the identical bytes even if the dataset's default
# branch advances. Matches the locally cached revision.
WEKA_HF_REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"

# A handful of real traces is enough to exercise the multi-graph merge, the
# trie pool union, and real-content segment synthesis without pulling the whole
# corpus. Streaming + this cap keeps the ingest fast and cheap.
SLICE_ROWS = 3


def _parse_real_slice_or_skip() -> ParsedGraph:
    """Parse the pinned real slice; skip ONLY on availability errors.

    hub/datasets availability failures (offline mode, cache miss, repo/auth
    errors, HTTP failures) all descend from ``OSError``; the weka loader wraps
    the ``load_dataset`` failure in ``WekaTraceAdapterError`` with the original
    as ``__cause__``. Anything else -- a real parse/trie crash on real data --
    must fail the test loudly, never skip.
    """
    unavailable = f"weka corpus {WEKA_HF_REPO_ID!r} unavailable"
    try:
        return from_weka_trace(WEKA_HF_REPO_ID, content_root_seed=42)
    except OSError as exc:
        pytest.skip(f"{unavailable}: {exc!r}")
    except WekaTraceAdapterError as exc:
        if isinstance(exc.__cause__, OSError):
            pytest.skip(f"{unavailable}: {exc!r}")
        raise


def test_repo_id_routes_to_weka_hf_loader() -> None:
    """The published 062126 repo id is recognized as a weka HF dataset id."""
    assert _looks_like_hf_dataset_id(WEKA_HF_REPO_ID)


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.asyncio
async def test_real_062126_slice_ingests_to_unified_store(
    tmp_path, monkeypatch
) -> None:
    """Stream a real 062126 slice -> trie ParsedGraph -> unified segment store.

    Asserts the real traces parse through the segment-trie IR (non-None
    ``segment_pool``), produce a populated unified store with valid dense
    0..n-1 node ordinals per trace, and worker-materialize to non-empty
    prompts — the build-time proof that real weka data flows through the
    graph-IR-v1 trie ingest path.
    """
    # Pin the streamed slice + revision so the ingest is reproducible and cheap.
    # Bound the streamed rows via the split slice (WEKA_HF_MAX_ROWS was removed).
    monkeypatch.setattr(Environment.DATASET, "WEKA_HF_SPLIT", f"train[:{SLICE_ROWS}]")
    monkeypatch.setattr(Environment.DATASET, "WEKA_HF_REVISION", WEKA_HF_REVISION)

    parsed = _parse_real_slice_or_skip()

    # Streaming slice yields exactly SLICE_ROWS traces, each its own graph,
    # every one parsed through the trie builder (pool always present).
    assert len(parsed.traces) == SLICE_ROWS
    assert len(parsed.graphs) == SLICE_ROWS
    assert parsed.segment_pool is not None, "weka parse must surface the trie pool"
    assert parsed.segment_pool._by_id, "trie pool must carry real content segments"

    # Every trace's graph lowers to LLM nodes with the recorded model resolved
    # into dispatch_overrides -- the real "endpoint resolved" proof.
    for trace in parsed.traces:
        graph = resolve_trace_graph(parsed, trace)
        llm_nodes = [n for n in graph.nodes.values() if isinstance(n, LlmNode)]
        assert llm_nodes, f"trace {trace.id} must lower to at least one LLM node"
        for node in llm_nodes:
            assert (node.dispatch_overrides or {}).get("model"), (
                f"trace {trace.id}: LLM node missing dispatch_overrides['model']"
            )

    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="real062126"
    )
    catalog = await build_unified_trie_store_interned(parsed, store)

    # Every parsed trace is addressable in the catalog with valid ordinals.
    assert set(catalog) == {t.id for t in parsed.traces}
    total_nodes = 0
    for nodes in catalog.values():
        assert nodes, "expected at least one node per trace"
        ordinals = sorted(nodes.values())
        assert ordinals == list(range(len(nodes))), "ordinals must be dense 0..n-1"
        total_nodes += len(nodes)
    assert total_nodes > 0

    # The unified store materializes non-empty real-content prompts for the
    # overwhelming majority of nodes (nodes without a prompt path write no
    # manifest, so a strong floor is asserted rather than 100%).
    non_empty = 0
    with GraphSegmentUnifiedClient(tmp_path, "real062126").open() as client:
        for trace_id, nodes in catalog.items():
            for ordinal in nodes.values():
                payload = materialize_graph_request_unified(
                    client, trace_id, ordinal, "profiling"
                )
                if payload is None:
                    continue
                messages = payload["messages"]
                if messages and messages[0].get("content"):
                    non_empty += 1
    assert non_empty >= total_nodes - SLICE_ROWS
    assert non_empty > 0


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.asyncio
async def test_streaming_ingest_matches_eager_byte_for_byte(
    tmp_path, monkeypatch
) -> None:
    """Streaming ingest == eager ingest, byte-for-byte, on the same real slice.

    The streaming build plane (``build_unified_trie_store_from_payloads`` over
    ``stream_weka_trace_segment_payloads``) keeps resident memory at ~one trace so
    the full real corpus ingests without OOM. This asserts the streamed payloads
    carry the trie shape (:class:`TraceSegmentPayload` with segments +
    ``prompt_segment_ids`` envelopes) and that draining them produces the
    IDENTICAL unified store as the eager whole-corpus
    ``build_unified_trie_store_interned``: same catalog (per-trace ordinal maps)
    and byte-identical store files.
    """
    # Bound the streamed rows via the split slice (WEKA_HF_MAX_ROWS was removed).
    monkeypatch.setattr(Environment.DATASET, "WEKA_HF_SPLIT", f"train[:{SLICE_ROWS}]")
    monkeypatch.setattr(Environment.DATASET, "WEKA_HF_REVISION", WEKA_HF_REVISION)

    from aiperf.dataset.graph.adapters.weka.trace import (
        stream_weka_trace_segment_payloads,
    )
    from aiperf.dataset.graph.segment_ir.store_builder import (
        build_unified_trie_store_from_payloads,
    )

    eager_parsed = _parse_real_slice_or_skip()

    assert eager_parsed.segment_pool is not None
    eager_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="eager"
    )
    eager_catalog = await build_unified_trie_store_interned(eager_parsed, eager_store)

    payloads = list(
        stream_weka_trace_segment_payloads(WEKA_HF_REPO_ID, content_root_seed=42)
    )
    assert len(payloads) == SLICE_ROWS
    assert all(isinstance(p, TraceSegmentPayload) for p in payloads)
    assert any(p.segments for p in payloads), "trie payloads must carry segments"
    for payload in payloads:
        assert payload.envelopes, f"trace {payload.trace_id} must carry envelopes"
        for node_envelope in payload.envelopes:
            env = orjson.loads(node_envelope.envelope_bytes)
            assert "prompt_segment_ids" in env, "trie envelope, not legacy"
            assert "messages_delta" not in env, "must NOT be a legacy delta envelope"

    stream_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="stream"
    )
    stream_catalog = await build_unified_trie_store_from_payloads(
        payloads, stream_store
    )

    assert stream_catalog == eager_catalog, "streaming catalog must equal eager"

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
