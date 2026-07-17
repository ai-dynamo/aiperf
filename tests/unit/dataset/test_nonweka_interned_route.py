# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-weka interned-route contract: every non-weka graph build (dynamo /
native / dag_jsonl) parses ONCE in-process and drains that SAME parse through
the eager interned builder, writing the mandatory graph_meta sidecar DIRECTLY
from the stripped parse; weka alone still takes the worker-pool payload stream +
structural merge.

These pins cover the route face that the byte-parity oracle
(``test_dag_jsonl_streaming_store_parity`` / ``test_dynamo_streaming_store_parity``)
does not: the returned prefix source identity, the persisted store's per-trace
topology, the sidecar's PARSE-ORDER trace list (parse order is the contract --
a sorted-by-id list would silently reorder traces), the sidecar's
prefix-cache-count round trip, and the weka-only routing inverse.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import orjson
import pytest

from aiperf.dataset.graph.adapters.dag_jsonl.trace import from_dag_jsonl
from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.codecs import decode_graph_meta_sidecar
from aiperf.dataset.graph.graph_meta_sidecar import catalogs_match
from aiperf.dataset.graph.models import LlmNode, resolve_trace_graph
from aiperf.dataset.graph.segment_ir.store_builder import (
    graph_carries_assembly_slots,
    iter_trace_segment_payloads,
)
from aiperf.dataset.graph.store_build import GraphStoreBuilder
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedClient,
    _encode_inner_key,
)
from tests.unit.dataset.test_dag_jsonl_streaming_store_parity import (
    MULTI_TRACE_DAG_LINES,
)

WEKA_FIXTURE = Path(__file__).parents[1] / "graph" / "fixtures" / "weka_min.json"

# Two INDEPENDENT lone-turn traces whose ids sort in the REVERSE of parse order
# ("aa-second" < "zz-first"), so a route that preserves parse order and one that
# sorts by id produce DIFFERENT trace lists -- the pin for the
# parse-order contract.
PARSE_ORDER_DAG_LINES = """\
{"session_id":"zz-first","turns":[{"model":"Qwen3-0.6B","messages":[{"role":"system","content":"zz-sys"},{"role":"user","content":"zz-u"}],"max_tokens":20}]}
{"session_id":"aa-second","turns":[{"model":"Qwen3-0.6B","messages":[{"role":"system","content":"aa-sys"},{"role":"user","content":"aa-u"}],"max_tokens":20}]}
"""


def _dynamo_record(ts: int, sid: str, input_tokens: int, hashes: list[int]) -> dict:
    """A minimal dynamo request-end record that stamps prefix-cache hash ids."""
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


def _make_interned_route_stub(benchmark_id: str) -> SimpleNamespace:
    """Stub carrying only what the in-process interned branch reads from self.

    Binds the REAL interned-drain/sidecar helpers; the weka trie drain and
    structural merge fail loudly if reached (no non-weka format may take them).
    """
    stub = SimpleNamespace(
        run=SimpleNamespace(benchmark_id=benchmark_id),
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        _sidecar_path=None,
    )
    for name in ("_write_graph_sidecar", "_build_interned_unified_store"):
        setattr(stub, name, MethodType(getattr(GraphStoreBuilder, name), stub))

    async def _fail_trie(payloads, base_path):  # noqa: ANN001, ARG001
        raise AssertionError(
            "non-weka format must not take the weka trie payload drain"
        )

    def _fail_merge(structural_sink):  # noqa: ANN001, ARG001
        raise AssertionError("non-weka format must not merge a structural stream")

    stub._build_graph_store_streaming_trie = _fail_trie
    stub._merge_structural_graphs = _fail_merge
    return stub


@pytest.mark.asyncio
async def test_multi_trace_interned_route_persists_both_topologies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multi-trace slot-free dag_jsonl file takes the interned route and the
    persisted store serves each trace's OWN topology.

    ``_build_graph_store_streaming`` returns the FULL parse as the prefix source
    (identity), the catalog covers both traces, and -- read back from the
    unified store index -- each trace's node ordinals match its own catalog
    (a merge-collapse bug would give both traces one tree's single-node graph).
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

    stub = _make_interned_route_stub("bench-multi-topo")
    catalog, returned = await GraphStoreBuilder._build_graph_store_streaming(
        stub, src, tmp_path, "dag_jsonl"
    )

    assert returned is parsed
    assert set(catalog) == {"conv-a", "conv-b"} and len(catalog) == 2
    # Store-side topology check: the persisted index carries each trace's own
    # node ordinals and its envelopes materialize.
    with GraphSegmentUnifiedClient(tmp_path, "bench-multi-topo").open() as client:
        for trace_id, ordinals in catalog.items():
            store_keys = set(client._node_offsets.get(trace_id, {}))
            expected_keys = {
                _encode_inner_key(ordinal, "profiling") for ordinal in ordinals.values()
            }
            assert store_keys == expected_keys and store_keys
            for ordinal in ordinals.values():
                assert client.get_node_envelope(trace_id, ordinal, "profiling")
    # The two traces did NOT collapse to a single shared topology: distinct
    # per-trace node-id sets survived into the store's build catalog.
    assert set(catalog["conv-a"]) != set(catalog["conv-b"])


@pytest.mark.asyncio
async def test_interned_route_sidecar_preserves_parse_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph_meta sidecar ships traces in PARSE order, content-free.

    The contract: the in-process interned route writes the sidecar
    DIRECTLY from the stripped parse, so the loaded trace list is the parse
    order ("zz-first", "aa-second"), NOT a
    sorted-by-id order. The loaded graph is content-free (empty pool, empty node
    prompts) and its catalog still matches the build catalog.
    """
    from aiperf.dataset.graph import workload_detect

    src = tmp_path / "order.dag.jsonl"
    src.write_text(PARSE_ORDER_DAG_LINES)
    parsed = from_dag_jsonl(str(src))
    assert [t.id for t in parsed.traces] == ["zz-first", "aa-second"]
    assert not graph_carries_assembly_slots(parsed)
    monkeypatch.setattr(
        workload_detect, "parse_graph_workload", lambda run, path: parsed
    )

    stub = _make_interned_route_stub("bench-order")
    catalog, returned = await GraphStoreBuilder._build_graph_store_streaming(
        stub, src, tmp_path, "dag_jsonl"
    )
    assert returned is parsed
    assert stub._sidecar_path is not None

    loaded, _fingerprint, _version = decode_graph_meta_sidecar(
        stub._sidecar_path.read_bytes()
    )

    # PARSE order, not sorted-by-id (sorted would be ["aa-second", "zz-first"]).
    assert [t.id for t in loaded.traces] == ["zz-first", "aa-second"]
    # Content-free: the pool is emptied and every LlmNode prompt is stripped.
    assert not loaded.segment_pool.by_id
    for trace in loaded.traces:
        for node in resolve_trace_graph(loaded, trace).nodes.values():
            if isinstance(node, LlmNode):
                assert node.prompt == []
    # The loaded structural graph still describes the build's topology.
    assert catalogs_match(loaded, catalog)


@pytest.mark.asyncio
async def test_interned_route_sidecar_roundtrips_prefix_cache_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sidecar preserves the per-node prefix-cache counts the prefix map reads.

    A dynamo parse stamps ``metadata["dispatch"]`` prefix-cache counts. The
    prefix map computed from the LOADED sidecar graph must equal the map from
    the live parse -- proving ``strip_replay_text`` + encode/decode preserve the
    dispatch metadata the prefix-cache metric consumes.
    """
    from aiperf.dataset.graph import workload_detect

    dyn = tmp_path / "dyn.jsonl"
    dyn.write_bytes(
        b"\n".join(
            orjson.dumps(r)
            for r in (
                _dynamo_record(1000, "s1", 32, [111, 222]),
                _dynamo_record(2000, "s1", 64, [111, 222, 333, 444]),
                _dynamo_record(3000, "s2", 48, [555, 666, 777]),
            )
        )
    )
    parsed = from_dynamo_trace(dyn, content_root_seed=42, content_tokenizer="builtin")
    eager_map = GraphStoreBuilder._build_graph_prefix_cache_by_trace(parsed)
    assert eager_map, "dynamo fixture must stamp a non-empty prefix-cache map"
    monkeypatch.setattr(
        workload_detect, "parse_graph_workload", lambda run, path: parsed
    )

    stub = _make_interned_route_stub("bench-prefix")
    _catalog, returned = await GraphStoreBuilder._build_graph_store_streaming(
        stub, dyn, tmp_path, "dynamo"
    )
    assert returned is parsed
    assert stub._sidecar_path is not None

    loaded, _fingerprint, _version = decode_graph_meta_sidecar(
        stub._sidecar_path.read_bytes()
    )
    loaded_map = GraphStoreBuilder._build_graph_prefix_cache_by_trace(loaded)
    assert loaded_map == eager_map


@pytest.mark.asyncio
async def test_weka_route_still_takes_payload_stream_and_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The weka-only inverse: ``fmt="weka_trace"`` takes the worker-pool payload
    stream + structural merge, NOT the in-process interned drain.

    Canned ``TraceSegmentPayload``s (captured from a real weka parse so the
    structural graph and catalog stay consistent) replace the streaming source;
    the route must drain them through ``_build_graph_store_streaming_trie``, so
    the returned prefix source IS the MERGED structural graph (content-free,
    identity-checked), the sidecar is written, and the interned in-process drain
    is never reached.
    """
    from aiperf.dataset.graph import workload_detect
    from aiperf.dataset.graph.adapters.weka import trace as weka_trace
    from aiperf.dataset.graph.parse_context import GraphParseContext

    parsed_weka = from_weka_trace(str(WEKA_FIXTURE), content_root_seed=0)
    canned_payloads = list(iter_trace_segment_payloads(parsed_weka))
    assert canned_payloads and any(p.structural_graph for p in canned_payloads)

    monkeypatch.setattr(
        weka_trace,
        "stream_weka_trace_segment_payloads",
        lambda source, **kwargs: iter(canned_payloads),
    )
    # The route resolves the run-derived knob bundle (the ONE
    # ``resolve_graph_parse_context`` resolution) before calling the (patched)
    # stream; neutralize it so the bare stub run suffices.
    monkeypatch.setattr(
        workload_detect,
        "resolve_graph_parse_context",
        lambda run: GraphParseContext(content_root_seed=0, idle_gap_cap_seconds=None),
    )

    stub = SimpleNamespace(
        run=SimpleNamespace(benchmark_id="bench-weka-route"),
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        _sidecar_path=None,
    )
    for name in (
        "_build_graph_store_streaming_trie",
        "_write_graph_sidecar",
    ):
        setattr(stub, name, MethodType(getattr(GraphStoreBuilder, name), stub))

    captured: dict[str, object] = {}
    real_merge = MethodType(GraphStoreBuilder._merge_structural_graphs, stub)

    def _capturing_merge(structural_sink):  # noqa: ANN001
        merged = real_merge(structural_sink)
        captured["merged"] = merged
        return merged

    stub._merge_structural_graphs = _capturing_merge

    async def _fail_interned(parsed, unified):  # noqa: ANN001, ARG001
        raise AssertionError("weka_trace must not take the in-process interned drain")

    stub._build_interned_unified_store = _fail_interned
    monkeypatch.setattr(
        workload_detect,
        "parse_graph_workload",
        lambda run, path: (_ for _ in ()).throw(
            AssertionError("weka_trace must not re-parse in-process")
        ),
    )

    catalog, returned = await GraphStoreBuilder._build_graph_store_streaming(
        stub, WEKA_FIXTURE, tmp_path, "weka_trace"
    )

    # Routed through the trie drain: the returned prefix source IS the merged
    # structural graph, not the parse.
    assert returned is captured["merged"]
    assert returned is not parsed_weka
    # The merged structural graph is content-free (empty pool, stripped prompts).
    assert not returned.segment_pool.by_id
    assert set(catalog) == {t.id for t in parsed_weka.traces} and catalog
    assert stub._sidecar_path is not None
    assert stub._sidecar_path.exists()
    # The store serves the drained trace's node envelopes.
    with GraphSegmentUnifiedClient(tmp_path, "bench-weka-route").open() as client:
        for trace_id, ordinals in catalog.items():
            for ordinal in ordinals.values():
                assert client.get_node_envelope(trace_id, ordinal, "profiling")
