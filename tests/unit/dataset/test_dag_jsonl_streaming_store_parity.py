# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dag_jsonl store-build parity: a slot-free dag graph's streamed payload
drain must equal the interned build byte-for-byte (the byte-parity oracle that
proves the in-process interned drain and the weka worker-pool trie drain build
identical stores), and ``_build_graph_store_streaming`` now routes EVERY
non-weka dag_jsonl parse -- slot-carrying or not -- through the in-process
interned drain (parse once, drain the same parse, return the full parse).
``graph_carries_assembly_slots`` is retained for the ``workload_detect``
t*-gate; its lineage detection is pinned here too."""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import orjson
import pytest

from aiperf.dataset.graph.adapters.dag_jsonl.trace import from_dag_jsonl
from aiperf.dataset.graph.graph_meta_sidecar import strip_replay_text
from aiperf.dataset.graph.merge import merge_parsed_graphs
from aiperf.dataset.graph.models import resolve_trace_graph
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

DAG_FIXTURES = Path(__file__).parents[2] / "fixtures" / "dag"
# Spawn-only dag graphs are the slot-free shape: spawned children start fresh
# histories, so no node assembles a live-reply (lineage) slot. Fork children
# inherit the parent's history including its assistant reply, so every other
# fixture in the dir carries slots.
SLOT_FREE_FIXTURE = DAG_FIXTURES / "spawn_minimal.dag.jsonl"
LINEAGE_FIXTURE = DAG_FIXTURES / "full.dag.jsonl"

# Two INDEPENDENT root sessions (neither spawns the other) with distinct
# topologies: conv-a is a lone turn, conv-b spawns a child. One file therefore
# parses to ONE ParsedGraph carrying TWO traces whose graphs live in
# ``parsed.graphs`` keyed by graph_ref -- the multi-trace-per-source shape the
# structural merge must preserve.
MULTI_TRACE_DAG_LINES = """\
{"session_id":"conv-a","turns":[{"model":"Qwen3-0.6B","messages":[{"role":"system","content":"a-sys"},{"role":"user","content":"a-u"}],"max_tokens":20}]}
{"session_id":"conv-b","turns":[{"model":"Qwen3-0.6B","messages":[{"role":"system","content":"b-sys"},{"role":"user","content":"b-u"}],"max_tokens":20,"spawns":["conv-b-child"]}]}
{"session_id":"conv-b-child","turns":[{"model":"Qwen3-0.6B","messages":[{"role":"system","content":"bc-sys"},{"role":"user","content":"bc-u"}],"max_tokens":20}]}
"""


@pytest.mark.asyncio
async def test_dag_jsonl_slot_free_payload_stream_matches_interned_oracle(
    tmp_path: Path,
) -> None:
    parsed = from_dag_jsonl(str(SLOT_FREE_FIXTURE))
    assert parsed.segment_pool is not None
    # Slot-free premise: this fixture must stream with no eager fallback.
    assert not graph_carries_assembly_slots(parsed)

    eager_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="dag-eager"
    )
    eager_catalog = await build_unified_trie_store_interned(parsed, eager_store)

    stream_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="dag-stream"
    )
    stream_catalog = await build_unified_trie_store_from_payloads(
        iter_trace_segment_payloads(parsed), stream_store
    )

    assert stream_catalog == eager_catalog and eager_catalog

    # Strong equivalence: the persisted unified store is byte-for-byte identical
    # on disk (mirrors ``test_dynamo_streaming_store_parity``'s eager-vs-stream
    # oracle), which subsumes the interned handle map, node-manifest region, and
    # content pool -- the whole store, not just the materialized messages.
    eager_dir = tmp_path / "aiperf_graph_segments_dag-eager"
    stream_dir = tmp_path / "aiperf_graph_segments_dag-stream"
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
        GraphSegmentUnifiedClient(tmp_path, "dag-eager").open() as ec,
        GraphSegmentUnifiedClient(tmp_path, "dag-stream").open() as sc,
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


def test_dag_jsonl_lineage_fixture_carries_slots() -> None:
    """Real live-reply dag workloads carry assembly slots.

    Fork children replay the parent's assistant response as a live-reply
    assembly slot, so a lineage-carrying dag parse must trip
    ``graph_carries_assembly_slots``. The store route does not branch on
    this (every non-weka format takes the interned drain, which persists slot
    envelopes), but ``workload_detect``'s t*-gate still consults the detector,
    so its lineage detection is pinned here.
    """
    parsed = from_dag_jsonl(str(LINEAGE_FIXTURE))
    assert parsed.segment_pool is not None
    assert graph_carries_assembly_slots(parsed)


@pytest.mark.asyncio
async def test_configure_route_dag_jsonl_slot_free_takes_interned_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slot-free dag_jsonl workload takes the in-process interned drain.

    Routing pin: ``_build_graph_store_streaming`` serves dag_jsonl (like every
    non-weka format) by parsing once in-process and draining that SAME parse
    through the interned builder. The second return IS the full parse
    (identity), NOT a content-free structural merge, and the weka trie
    drain/structural merge are never touched.
    """
    from aiperf.dataset.graph import workload_detect

    parsed = from_dag_jsonl(str(SLOT_FREE_FIXTURE))
    assert not graph_carries_assembly_slots(parsed)
    monkeypatch.setattr(
        workload_detect, "parse_graph_workload", lambda run, path: parsed
    )

    # Just the attributes the in-process interned branch reads from self, with
    # the REAL interned-drain/sidecar helpers bound; the weka trie drain and
    # structural merge fail loudly if reached (no non-weka format may take them).
    stub = SimpleNamespace(
        run=SimpleNamespace(benchmark_id="bench"),
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        _sidecar_path=None,
    )
    for name in ("_write_graph_sidecar", "_build_interned_unified_store"):
        setattr(stub, name, MethodType(getattr(GraphStoreBuilder, name), stub))

    async def _fail_trie(payloads, base_path):  # noqa: ANN001, ARG001
        raise AssertionError("dag_jsonl must not take the weka trie payload drain")

    def _fail_merge(structural_sink):  # noqa: ANN001, ARG001
        raise AssertionError("dag_jsonl must not merge a structural stream")

    stub._build_graph_store_streaming_trie = _fail_trie
    stub._merge_structural_graphs = _fail_merge

    catalog, returned = await GraphStoreBuilder._build_graph_store_streaming(
        stub, SLOT_FREE_FIXTURE, tmp_path, "dag_jsonl"
    )

    assert set(catalog) == {t.id for t in parsed.traces} and catalog
    # The interned drain hands back the FULL parse, not a merged structural graph.
    assert returned is parsed
    # The interned store is the real unified store the worker opens.
    with GraphSegmentUnifiedClient(tmp_path, "bench").open() as client:
        for trace_id, ordinals in catalog.items():
            for ordinal in ordinals.values():
                assert client.get_node_envelope(trace_id, ordinal, "profiling")


def test_merge_preserves_per_trace_graphs_multi_trace_source(
    tmp_path: Path,
) -> None:
    """A multi-trace SOURCE graph keeps each trace's own topology through merge.

    The structural merge used to assume one trace per source and keyed every
    trace to ``pg.graph`` (the FIRST tree's record), dropping ``pg.graphs``
    entirely -- so conv-b resolved to conv-a's topology and the mandatory
    sidecar's ``catalogs_match`` gate hard-failed the build.
    """
    src = tmp_path / "multi.dag.jsonl"
    src.write_text(MULTI_TRACE_DAG_LINES)
    parsed = from_dag_jsonl(str(src))
    assert len(parsed.traces) == 2
    # Premise: the two traces genuinely carry DIFFERENT topologies.
    node_sets = {t.id: set(resolve_trace_graph(parsed, t).nodes) for t in parsed.traces}
    assert node_sets["conv-a"] != node_sets["conv-b"]

    merged = merge_parsed_graphs([strip_replay_text(parsed)])

    assert {t.id for t in merged.traces} == set(node_sets)
    for trace in merged.traces:
        assert set(resolve_trace_graph(merged, trace).nodes) == node_sets[trace.id], (
            f"trace {trace.id!r} lost its own topology through the merge"
        )


@pytest.mark.asyncio
async def test_configure_route_dag_jsonl_multi_trace_slot_free_takes_interned_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-trace slot-free dag_jsonl file builds through the interned route.

    Regression pin: the mandatory ``_write_graph_sidecar`` cross-checks the
    build catalog against the structural graph (``catalogs_match``). The two
    traces carry DIFFERENT topologies, so a route that collapsed them to one
    tree's graph (the historical merge bug) would fail that gate. The in-process
    interned drain writes the sidecar from the FULL parse directly, so the build
    must SUCCEED with both traces on their own topologies.
    """
    from aiperf.dataset.graph import workload_detect

    src = tmp_path / "multi.dag.jsonl"
    src.write_text(MULTI_TRACE_DAG_LINES)
    parsed = from_dag_jsonl(str(src))
    assert len(parsed.traces) == 2
    assert not graph_carries_assembly_slots(parsed)
    # Premise: the two traces genuinely carry DIFFERENT topologies.
    node_sets = {t.id: set(resolve_trace_graph(parsed, t).nodes) for t in parsed.traces}
    assert node_sets["conv-a"] != node_sets["conv-b"]
    monkeypatch.setattr(
        workload_detect, "parse_graph_workload", lambda run, path: parsed
    )

    stub = SimpleNamespace(
        run=SimpleNamespace(benchmark_id="bench-multi"),
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        _sidecar_path=None,
    )
    for name in ("_write_graph_sidecar", "_build_interned_unified_store"):
        setattr(stub, name, MethodType(getattr(GraphStoreBuilder, name), stub))

    async def _fail_trie(payloads, base_path):  # noqa: ANN001, ARG001
        raise AssertionError("dag_jsonl must not take the weka trie payload drain")

    stub._build_graph_store_streaming_trie = _fail_trie

    catalog, returned = await GraphStoreBuilder._build_graph_store_streaming(
        stub, src, tmp_path, "dag_jsonl"
    )

    assert set(catalog) == {t.id for t in parsed.traces} and len(catalog) == 2
    # The interned drain returns the FULL parse. The per-trace topology is
    # cross-checked STORE-SIDE (against the persisted index) in
    # ``test_nonweka_interned_route`` rather than re-derived from ``returned is
    # parsed`` here (which would be tautological -- ``node_sets`` came from the
    # same parse).
    assert returned is parsed
    # The mandatory sidecar's catalogs_match gate passed (path recorded), so the
    # multi-topology build was NOT collapsed to a single tree's graph.
    assert stub._sidecar_path is not None
    # The store carries manifests for BOTH traces' real nodes.
    with GraphSegmentUnifiedClient(tmp_path, "bench-multi").open() as client:
        for trace_id, ordinals in catalog.items():
            for ordinal in ordinals.values():
                assert client.get_node_envelope(trace_id, ordinal, "profiling")
