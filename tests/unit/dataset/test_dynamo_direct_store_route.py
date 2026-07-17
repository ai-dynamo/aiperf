# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamo direct write-through store route.

Covers the route face the byte-parity oracle does not:

* the ``StoreBackedSegmentPool`` shim -- ``add()`` dedups identically to
  ``SegmentPool.add`` and every non-``add`` method fails loud (dynamo-only);
* the ``**adapter_kwargs`` seam -- ``direct_store`` threads
  ``parse_graph_workload -> parse_graph -> _parse_via_adapter -> adapter.parse``
  with the two documented fail-loud modes (``GraphParseError`` for native,
  ``TypeError`` for an adapter that does not accept the kwarg);
* the store-before-parse cleanup -- a mid-parse failure (a block-inconsistent
  record, or any exception once content has spilled) leaves NO store files, i.e.
  the abort()+rmtree cleanup composes with the parse, not just the
  drain;
* the ``GraphStoreBuilder`` route on a REAL resolved ``BenchmarkRun`` (not a
  MagicMock -- the documented path-drift trap): the dynamo route hands a live
  store to the parse and the produced store + facet match the eager route.
"""

from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import orjson
import pytest

from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.graph import workload_detect
from aiperf.dataset.graph.adapters.dynamo.store_backed_pool import (
    InterningSegmentPool,
    StoreBackedSegmentPool,
)
from aiperf.dataset.graph.codecs import GRAPH_META_SCHEMA_VERSION
from aiperf.dataset.graph.graph_meta_sidecar import write_graph_meta_sidecar
from aiperf.dataset.graph.parser import GraphParseError, parse_graph
from aiperf.dataset.graph.segment_ir.pool import SegmentPool, segment_id
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph.store_build import GraphStoreBuilder
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
)
from tests.unit.conftest import make_run_from_cli


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
    p = tmp_path / "dyn_route.jsonl"
    records = [
        _dynamo_record(1000, "s1", 32, [111, 222]),
        _dynamo_record(2000, "s1", 64, [111, 222, 333, 444]),
        _dynamo_record(3000, "s2", 48, [555, 666, 777]),
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


# --- StoreBackedSegmentPool shim ------------------------------------------------


_ADD_CALLS = [
    {"role": "user", "content": "one", "tokens": [1, 2, 3], "parent_id": None},
    {"role": "assistant", "content": "two", "tokens": [4, 5], "parent_id": "p"},
    # A verbatim repeat of the first call: first-occurrence dedup must return the
    # SAME sid and intern nothing new.
    {"role": "user", "content": "one", "tokens": [1, 2, 3], "parent_id": None},
]


def test_shim_add_dedups_identically_to_segmentpool(tmp_path: Path) -> None:
    """The shim's ``add()`` returns the SAME sid stream as ``SegmentPool.add`` --
    same content-addressed ids, same first-occurrence dedup -- and interns exactly
    the unique segments into the store (the repeat is a no-op)."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="shim-dd")
    shim = StoreBackedSegmentPool(store)
    pool = SegmentPool()

    shim_sids = [shim.add(**call) for call in _ADD_CALLS]
    pool_sids = [pool.add(**call) for call in _ADD_CALLS]

    assert shim_sids == pool_sids
    # First-occurrence dedup: the repeated call yields the first call's sid.
    assert shim_sids[0] == shim_sids[2]
    # The store interned exactly the two UNIQUE segments (dedup no-op on repeat),
    # matching the plain pool's two-entry _by_id.
    assert len(store._ids) == 2
    assert len(pool.by_id) == 2


def test_interning_pool_add_returns_canonical_sid_object(tmp_path: Path) -> None:
    """``InterningSegmentPool.add`` returns the FIRST-BORN ``Segment.id`` object on
    a dedup hit (not a fresh equal string), while values and ``_by_id`` insertion
    order stay identical to a plain ``SegmentPool``.

    The eager route relies on this canonical-object return so every node's
    ``prompt_segment_ids`` re-listing shares one string object per unique value.
    """
    interning = InterningSegmentPool()
    plain = SegmentPool()

    interning_sids = [interning.add(**call) for call in _ADD_CALLS]
    plain_sids = [plain.add(**call) for call in _ADD_CALLS]

    # Values are byte-identical to the plain pool (interning changes identity only).
    assert interning_sids == plain_sids
    assert list(interning.by_id.keys()) == list(plain.by_id.keys())
    # The repeat (call 2 == call 0) returns the SAME object as the first add...
    assert interning_sids[0] is interning_sids[2]
    # ...which is exactly the first-born Segment.id stored on first occurrence.
    assert all(sid is interning.by_id[sid].id for sid in interning_sids)
    # A plain SegmentPool does NOT canonicalize -- the dedup hit is a fresh object.
    assert plain_sids[0] is not plain_sids[2]


def test_shim_add_returns_canonical_sid_object(tmp_path: Path) -> None:
    """``StoreBackedSegmentPool.add`` returns the canonical (first-born) sid object
    on a repeat via its handle-indexed ``_sids`` list, values unchanged, and
    ``_sids`` is dense (one entry per unique segment, in first-occurrence order)."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="shim-can")
    shim = StoreBackedSegmentPool(store)
    plain = SegmentPool()

    shim_sids = [shim.add(**call) for call in _ADD_CALLS]
    plain_sids = [plain.add(**call) for call in _ADD_CALLS]

    # Values identical to the plain pool's sid stream (identity-only change).
    assert shim_sids == plain_sids
    # The repeat returns the SAME object as the first add (canonical).
    assert shim_sids[0] is shim_sids[2]
    # _sids is dense: exactly the two unique sids, in first-occurrence order, and
    # holding the canonical objects the shim returns.
    assert shim._sids == [shim_sids[0], shim_sids[1]]
    assert shim._sids[0] is shim_sids[0]
    assert len(shim._sids) == len(store._ids) == 2


def test_shim_add_pre_populated_store_falls_through_to_fresh_sid(
    tmp_path: Path,
) -> None:
    """When the store was written before this shim existed (``handle`` outruns
    ``_sids``), ``add`` degrades to returning the fresh value-correct sid and
    interns NOTHING into ``_sids`` -- the defensive third branch, never hit in
    production (the shim is the store's sole writer during a parse)."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="shim-pre")
    # Pre-populate the store with two segments the shim never recorded in _sids.
    store.put_segment("pre-a", "user", "a")
    store.put_segment("pre-b", "user", "b")

    shim = StoreBackedSegmentPool(store)
    assert shim._sids == []

    got = shim.add(role="user", content="one", tokens=[1, 2, 3], parent_id=None)

    # The handle (2) outran the empty _sids, so add fell through: value is still the
    # correct content-addressed sid, and _sids stayed empty (nothing canonicalized).
    assert got == segment_id(None, "user", [1, 2, 3])
    assert shim._sids == []


def test_shim_non_add_methods_fail_loud(tmp_path: Path) -> None:
    """Every non-``add`` pool method raises ``NotImplementedError`` naming the
    dynamo-only write-through contract -- so a non-dynamo adopter cannot silently
    intern into an empty ``_by_id`` the store never sees."""
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="shim-fl")
    shim = StoreBackedSegmentPool(store)

    with pytest.raises(NotImplementedError, match="add_text"):
        shim.add_text(role="user", content="x", parent_id=None)
    with pytest.raises(NotImplementedError, match="add_raw_message"):
        shim.add_raw_message(message={"role": "user"}, parent_id=None)
    with pytest.raises(NotImplementedError, match="get"):
        shim.get("some-sid")
    with pytest.raises(NotImplementedError, match="materialize"):
        shim.materialize(["some-sid"])


# --- **adapter_kwargs seam: fail-loud contract ----------------------------------


def test_parse_graph_native_rejects_adapter_kwargs(tmp_path: Path) -> None:
    """``adapter_kwargs`` with ``format="native"`` raise ``GraphParseError`` up
    front: the native parse path has no adapter ``parse`` to receive them."""
    p = tmp_path / "native.yaml"
    p.write_text("graph:\n  nodes: {}\n")
    with pytest.raises(GraphParseError, match="native parse path accepts no"):
        parse_graph(p, format="native", direct_store=object())


def test_parse_graph_unknown_adapter_kwarg_fails_loud_typeerror(
    tmp_path: Path,
) -> None:
    """A kwarg reaching an adapter whose ``parse`` does not accept it fails loud
    with ``TypeError`` (never silently dropped). weka's ``parse(path, ctx)`` takes
    no ``direct_store``, so threading one is a build-plane wiring bug that must
    surface, not vanish."""
    p = tmp_path / "t.json"
    p.write_text("{}")
    with pytest.raises(TypeError):
        parse_graph(p, format="weka_trace", direct_store=object())


# --- store-before-parse cleanup -------------------------------------------------


def _cleanup_stub(benchmark_id: str) -> SimpleNamespace:
    """Stub self carrying only what the dynamo direct branch reads.

    Binds the REAL interned drain + sidecar helpers so the branch runs
    end-to-end; ``parse_graph_workload`` is monkeypatched per-test.
    """
    stub = SimpleNamespace(
        run=SimpleNamespace(benchmark_id=benchmark_id),
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        _sidecar_path=None,
    )
    for name in ("_write_graph_sidecar", "_build_interned_unified_store"):
        setattr(stub, name, MethodType(getattr(GraphStoreBuilder, name), stub))
    return stub


@pytest.mark.asyncio
async def test_direct_route_cleans_partial_content_on_mid_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-parse failure AFTER content has spilled leaves NO store files.

    The store is now constructed BEFORE the parse and content.blob spills
    incrementally, so the abort()+rmtree cleanup MUST cover the parse. A fake
    parse writes a real segment through the passed ``direct_store`` (proving the
    write-through is live and content was written) then raises; the store dir must
    be gone afterward.
    """
    captured: dict[str, object] = {}

    def _fake_parse(run, path, *, direct_store):  # noqa: ANN001, ANN202
        direct_store.put_segment("sid-partial", "user", "partial content")
        captured["bytes_written"] = direct_store._content_bytes_written
        captured["data_dir"] = direct_store.data_dir
        raise GraphParseError(f"{path}: boom mid-parse")

    monkeypatch.setattr(workload_detect, "parse_graph_workload", _fake_parse)

    stub = _cleanup_stub("bench-partial")
    with pytest.raises(GraphParseError, match="boom mid-parse"):
        await GraphStoreBuilder._build_graph_store_streaming(
            stub, tmp_path / "dyn.jsonl", tmp_path, "dynamo_trace"
        )

    # Content really was written through before the failure...
    assert captured["bytes_written"] > 0
    # ...and the whole store dir (partial content.blob included) is gone.
    assert not Path(captured["data_dir"]).exists()


@pytest.mark.asyncio
async def test_direct_route_block_inconsistent_record_aborts_store(
    tmp_path: Path,
) -> None:
    """A block-inconsistent record through the direct route surfaces as
    ``GraphParseError`` (the seam re-wraps ``DynamoISLMismatchError``) AND leaves
    no store dir -- the real end-to-end proof the parse-failure cleanup composes
    with Task-5's abort path.

    Built on a REAL resolved ``BenchmarkRun`` (the MagicMock path-drift trap:
    a stub config silently no-ops the wrong attribute read)."""
    bad = tmp_path / "bad.jsonl"
    # input_length=100 with 2 hashes at block_size=16 violates
    # (n-1)*bs < input_length <= n*bs  (16 < 100 <= 32 is False).
    bad.write_bytes(orjson.dumps(_dynamo_record(1000, "s1", 100, [111, 222])))

    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(bad),
            tokenizer_name="builtin",
        )
    )
    builder = GraphStoreBuilder(run)
    store_dir = tmp_path / f"aiperf_graph_segments_{run.benchmark_id}"

    with pytest.raises(GraphParseError, match="not.*block-aligned"):
        await builder._build_graph_store_streaming(bad, tmp_path, "dynamo_trace")

    assert not store_dir.exists()


# --- REAL-BenchmarkRun route parity ---------------------------------------------


@pytest.mark.asyncio
async def test_real_run_dynamo_route_hands_store_and_matches_eager(
    dyn_trace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a REAL resolved run the dynamo route (i) hands a live store to the parse
    and (ii) produces a store + facet byte/value-identical to the eager route.

    Spies the real ``parse_graph_workload`` to capture the threaded
    ``direct_store``; the eager reference re-parses the SAME run (same ctx -> same
    seed) so the byte comparison needs no knowledge of the resolved seed."""
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            input_file=str(dyn_trace),
            tokenizer_name="builtin",
        )
    )

    real_parse = workload_detect.parse_graph_workload
    captured: dict[str, object] = {}

    def _spy(run_arg, path_arg, **kwargs):  # noqa: ANN001, ANN202
        captured["direct_store"] = kwargs.get("direct_store")
        return real_parse(run_arg, path_arg, **kwargs)

    # Direct route (real parse under the spy). The eager reference below calls
    # ``real_parse`` directly, so the spy never intercepts it.
    monkeypatch.setattr(workload_detect, "parse_graph_workload", _spy)
    builder = GraphStoreBuilder(run)
    catalog, prefix_source = await builder._build_graph_store_streaming(
        dyn_trace, tmp_path, "dynamo_trace"
    )

    # (i) The dynamo route threaded a LIVE unified store into the parse.
    assert isinstance(captured["direct_store"], GraphSegmentUnifiedBackingStore)
    assert builder._sidecar_path is not None and builder._sidecar_path.exists()

    # Eager reference: the SAME run, no direct_store.
    eager_parsed = real_parse(run, dyn_trace)
    eager_store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id="eager-route-ref"
    )
    eager_catalog = await build_unified_trie_store_interned(eager_parsed, eager_store)

    # (ii-a) Store bytes match the eager route.
    direct_dir = tmp_path / f"aiperf_graph_segments_{run.benchmark_id}"
    eager_dir = tmp_path / "aiperf_graph_segments_eager-route-ref"
    direct_files = sorted(p.name for p in direct_dir.iterdir())
    eager_files = sorted(p.name for p in eager_dir.iterdir())
    assert direct_files == eager_files and direct_files
    for name in direct_files:
        assert (direct_dir / name).read_bytes() == (eager_dir / name).read_bytes(), (
            f"direct-route store file {name!r} diverged from the eager route"
        )
    assert catalog == eager_catalog

    # (ii-b) Facet (per-node prefix-cache map) matches the eager route.
    assert GraphStoreBuilder._build_graph_prefix_cache_by_trace(
        prefix_source
    ) == GraphStoreBuilder._build_graph_prefix_cache_by_trace(eager_parsed)

    # (ii-c) Sidecar bytes match the eager route (benchmark_id affects only the
    # path, never the content-free encoded structural graph).
    eager_sidecar = write_graph_meta_sidecar(
        eager_parsed,
        base_path=tmp_path,
        benchmark_id="eager-sidecar-ref",
        source_fingerprint={},
        schema_version=GRAPH_META_SCHEMA_VERSION,
    )
    assert builder._sidecar_path.read_bytes() == eager_sidecar.read_bytes()
