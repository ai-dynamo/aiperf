# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused read+build parallel dynamo tree build: grouping scan, shuffle, byte-equivalence to serial."""

from __future__ import annotations

import gzip
import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo import trace_parallel
from aiperf.dataset.graph.adapters.dynamo.trace import (
    DynamoTraceAdapterError,
    from_dynamo_trace,
    root_of_sessions,
)
from aiperf.dataset.graph.adapters.dynamo.trace_reader import (
    DynamoTraceReadError,
    scan_dynamo_trace,
    write_ingest_sidecar,
)
from aiperf.dataset.graph.adapters.shared.content import CorpusContentSynthesizer
from aiperf.dataset.graph.models import ParsedGraph
from tests.unit.dataset.graph.adapters.conftest import write_jsonl

_SEED = 12345

# The fused path decides tree membership in the parent from a cheap grouping scan
# (session ids + parent links only, NO input_sequence_hashes parse), shuffles raw
# record lines into per-batch temp files, and READS+BUILDS each batch inside a
# worker so recorded hash arrays never cross a process boundary. The pinned
# content seed + block size plus CONTIGUOUS weight-balanced batching make the
# parent's in-order union of worker results identical to the serial per-tree loop.
# The pool is real (forkserver on Linux), so fixtures stay tiny and worker counts
# low to keep these multiprocess tests fast and non-flaky.


@pytest.fixture(autouse=True)
def _fresh_synth_cache() -> Iterator[None]:
    """Isolate the process-level synthesizer cache from other tests."""
    CorpusContentSynthesizer.reset_worker_cache()
    yield
    CorpusContentSynthesizer.reset_worker_cache()


# --- fixture builders (real capture agent_context shape) ------------------


def _re(
    *,
    ts: int,
    sid: str,
    hashes: list[int],
    ilen: int,
    parent: str | None = None,
    otok: int = 8,
    bs: int = 16,
) -> dict[str, Any]:
    """A ``request_end`` record in the shape real dynamo captures emit."""
    ctx: dict = {"session_id": sid, "trajectory_id": sid}
    if parent is not None:
        ctx["parent_trajectory_id"] = parent
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": ctx,
        "request": {
            "request_id": f"r-{sid}-{ts}",
            "model": "m",
            "input_tokens": ilen,
            "output_tokens": otok,
            "cached_tokens": 0,
            "ttft_ms": 10.0,
            "replay": {
                "trace_block_size": bs,
                "input_length": ilen,
                "input_sequence_hashes": hashes,
            },
        },
    }


def _multi_tree_records() -> list[dict[str, Any]]:
    """One parent+subagent tree ('P'<-'S') plus three independent root trees."""
    return [
        _re(ts=1000, sid="P", hashes=[1, 2], ilen=32),
        _re(ts=2000, sid="S", parent="P", hashes=[7, 8], ilen=32),
        _re(ts=3000, sid="P", hashes=[1, 2, 3], ilen=48),
        _re(ts=1000, sid="a", hashes=[10, 11], ilen=32),
        _re(ts=1100, sid="a", hashes=[10, 11, 12], ilen=48),
        _re(ts=5000, sid="b", hashes=[20, 21], ilen=32),
        _re(ts=6000, sid="c", hashes=[30, 31], ilen=32),
        _re(ts=6100, sid="c", hashes=[30, 31, 32], ilen=48),
    ]


def _single_tree_records() -> list[dict[str, Any]]:
    """A lone two-turn root session, i.e. exactly one tree."""
    return [
        _re(ts=1000, sid="only", hashes=[1, 2], ilen=32),
        _re(ts=2000, sid="only", hashes=[1, 2, 3], ilen=48),
    ]


def _force_parallel(monkeypatch: pytest.MonkeyPatch, *, workers: int) -> None:
    """Drop the tree-count threshold to 0 so the fused pool path always runs."""
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 0)
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_WORKERS", workers)


def _edge_set(pg: ParsedGraph) -> set[tuple[str, str]]:
    """The ``(source, target)`` pairs of a parsed graph's edges."""
    return {(e.source, e.target) for e in pg.graph.edges}


def _assert_same_shape_and_content(actual: ParsedGraph, expected: ParsedGraph) -> None:
    """Assert two builds agree on node keys, edge set, and content-addressed pool."""
    assert set(actual.graph.nodes) == set(expected.graph.nodes)
    assert _edge_set(actual) == _edge_set(expected)
    assert actual.segment_pool is not None and expected.segment_pool is not None
    assert actual.segment_pool.by_id == expected.segment_pool.by_id


def _forbid_pool(monkeypatch: pytest.MonkeyPatch, reason: str) -> None:
    """Make entering the fused pool an outright test failure."""

    def _no_pool(*_a: Any, **_k: Any) -> None:
        raise AssertionError(reason)

    monkeypatch.setattr(trace_parallel, "_build_fused_parallel", _no_pool)


def _maybe_build(path: Path) -> ParsedGraph | None:
    """Call ``maybe_build_fused_parallel`` with the standard test parameters."""
    return trace_parallel.maybe_build_fused_parallel(
        path,
        content_root_seed=_SEED,
        idle_gap_cap_seconds=60.0,
        content_tokenizer=None,
        prompt_corpus="coding",
        release_replay=False,
        max_depth=8,
    )


def _track_fused_temp_dirs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every temp dir the fused build itself creates, ignoring other machinery."""
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*a: Any, **k: Any) -> str:
        d = real_mkdtemp(*a, **k)
        # Pool and shared-memory machinery also call mkdtemp and clean up on
        # their own schedule; only the fused build's dir is under test here.
        if Path(d).name.startswith("aiperf-dynamo-fused-"):
            created.append(d)
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)
    return created


# --- 1. contiguous weight batching (pure) ---------------------------------


@pytest.mark.parametrize(
    "items,weights,num_batches,expected_first",
    [
        param(list(range(10)), [1] * 10, 3, None, id="uniform_weights_split_in_three"),
        param(list(range(7)), [100] + [1] * 6, 3, [0], id="heavy_leader_closes_batch"),
        param([0, 1, 2, 3], [1, 1, 1, 1], 1, [0, 1, 2, 3], id="single_batch_takes_all"),
    ],
)  # fmt: skip
def test_contiguous_batches_preserve_order_and_cover_all(
    items: list[int],
    weights: list[int],
    num_batches: int,
    expected_first: list[int] | None,
) -> None:
    """Weight batching is contiguous and lossless: flattening yields the input order back."""
    batches = trace_parallel._contiguous_weight_batches(
        items, weights, num_batches=num_batches
    )
    assert [x for batch in batches for x in batch] == items
    assert 1 <= len(batches) <= num_batches
    if expected_first is not None:
        assert batches[0] == expected_first


# --- 2. grouping scan: NO hash parse, cross-file links, block size --------


def _write_plain(path: Path) -> Path:
    """Plain JSONL capture of the multi-tree fixture."""
    return write_jsonl(path / "trace.jsonl", _multi_tree_records())


def _write_gz_enveloped(path: Path) -> Path:
    """Gzipped capture whose lines are wrapped in the sink's timestamp/event envelope."""
    p = path / "trace.jsonl.gz"
    with gzip.open(p, "wb") as f:
        for rec in _multi_tree_records():
            f.write(orjson.dumps({"timestamp": 1, "event": rec}))
            f.write(b"\n")
    return p


@pytest.mark.parametrize(
    "writer",
    [
        param(_write_plain, id="plain_jsonl"),
        param(_write_gz_enveloped, id="gzip_sink_envelope"),
    ],
)  # fmt: skip
def test_scan_groups_sessions_and_parent_links(tmp_path: Path, writer: Any) -> None:
    """The grouping scan finds every session and the parent<-subagent link in both capture layouts."""
    scan = trace_parallel._scan_grouping(writer(tmp_path), threads=2)

    assert scan.request_end_sessions == {"P", "S", "a", "b", "c"}
    assert scan.parent_link == {"S": "P"}


def test_scan_reports_block_size_and_per_session_weight(tmp_path: Path) -> None:
    """The scan carries the capture's replay block size plus a positive per-session batching weight."""
    scan = trace_parallel._scan_grouping(_write_plain(tmp_path), threads=2)

    assert scan.block_size == 16
    assert scan.record_count == len(_multi_tree_records())
    # Weight is the summed record-line byte length per session -- a hash-free proxy.
    assert set(scan.session_weight) == {"P", "S", "a", "b", "c"}
    assert all(w > 0 for w in scan.session_weight.values())


def test_scan_synthesizes_contextless_request_sessions_and_ignores_markers(
    tmp_path: Path,
) -> None:
    """S3 markers are ignored while context-free requests become root sessions."""
    p = write_jsonl(
        tmp_path / "trace.jsonl", [_re(ts=1000, sid="P", hashes=[1, 2], ilen=32)]
    )
    # Append a schema-less S3 marker + a replay-only (no agent_context) record.
    with p.open("ab") as f:
        f.write(orjson.dumps({"verification": "trace-s3-uploader"}))
        f.write(b"\n")
        f.write(
            orjson.dumps(
                {
                    "schema": "dynamo.request.trace.v1",
                    "event_type": "request_end",
                    "event_time_unix_ms": 9,
                    "request": {
                        "request_id": "noctx",
                        "replay": {
                            "trace_block_size": 16,
                            "input_length": 32,
                            "input_sequence_hashes": [99, 98],
                        },
                    },
                }
            )
        )
        f.write(b"\n")

    scan = trace_parallel._scan_grouping(p, threads=1)
    assert scan.request_end_sessions == {"P", "request-noctx"}
    assert scan.parent_link == {}


def test_grouping_scan_reuses_matching_ingest_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valid metadata sidecar avoids reparsing the parent grouping scan."""
    p = write_jsonl(tmp_path / "trace.jsonl", _multi_tree_records())
    write_ingest_sidecar(p, scan_dynamo_trace(p))

    def fail_scan(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("sidecar path should not rescan source segments")

    monkeypatch.setattr(trace_parallel, "scan_dynamo_trace", fail_scan)
    scan = trace_parallel._scan_grouping(p, threads=1)

    assert scan.record_count == len(_multi_tree_records())
    assert scan.request_end_sessions == {"P", "S", "a", "b", "c"}


def test_scan_uses_canonical_json_and_envelope_semantics(tmp_path: Path) -> None:
    """Escapes, whitespace, envelopes, and nested decoys match the serial reader."""
    record = _re(ts=1000, sid='child\\"é', parent='parent\\"é', hashes=[1], ilen=32)
    decoy = {"event": {"session_id": "wrong"}}
    path = tmp_path / "trace.jsonl"
    with path.open("wb") as handle:
        handle.write(
            json.dumps({"event": record, "timestamp": 1}, indent=None).encode()
        )
        handle.write(b"\n")
        handle.write(orjson.dumps(decoy))
        handle.write(b"\n")
        handle.write(orjson.dumps({"verification": {"session_id": "wrong"}}))
        handle.write(b"\n")

    scan = trace_parallel._scan_grouping(path, threads=2)
    assert scan.request_end_sessions == {'child\\"é'}
    assert scan.parent_link == {'child\\"é': 'parent\\"é'}


def test_scan_rejects_mixed_block_size(tmp_path: Path) -> None:
    """A capture mixing replay block sizes is unbatchable and must raise."""
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [
            _re(ts=1000, sid="P", hashes=[1, 2], ilen=32, bs=16),
            _re(ts=2000, sid="Q", hashes=[1, 2], ilen=64, bs=32),
        ],
    )
    with pytest.raises(DynamoTraceAdapterError, match="mixed replay trace_block_size"):
        trace_parallel._scan_grouping(p, threads=2)


def test_scan_rejects_zero_block_size(tmp_path: Path) -> None:
    """A recorded ``trace_block_size`` of 0 is corrupt and must hard-fail.

    The serial reader rejects it via Pydantic ``ge=1``; the scan must not fall
    through to ``next(iter(block_sizes), DEFAULT_VIRTUAL_BLOCK_SIZE)`` and
    silently substitute a default, or the same malformed corpus would
    mis-benchmark above the parallel threshold and abort below it.
    """
    p = write_jsonl(
        tmp_path / "trace.jsonl",
        [_re(ts=1000, sid="P", hashes=[1, 2], ilen=32, bs=0)],
    )
    with pytest.raises(DynamoTraceReadError, match="greater than or equal to 1"):
        trace_parallel._scan_grouping(p, threads=1)


def test_scan_cross_file_parent_link(tmp_path: Path) -> None:
    """A subagent in one segment file linking to a parent in another roots into one tree."""
    d = tmp_path / "capture"
    write_jsonl(d / "a.jsonl", [_re(ts=1000, sid="P", hashes=[1, 2], ilen=32)])
    write_jsonl(
        d / "b.jsonl", [_re(ts=2000, sid="S", parent="P", hashes=[7, 8], ilen=32)]
    )

    scan = trace_parallel._scan_grouping(d, threads=2)
    root_of = root_of_sessions(scan.request_end_sessions, scan.parent_link)
    assert root_of["S"] == "P"  # child rooted at cross-file parent
    assert sorted(set(root_of.values())) == ["P"]  # one tree


# --- 3. byte-equivalence: parallel == serial ------------------------------


def test_parallel_build_byte_identical_to_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fused parallel build reproduces the serial build exactly: node keys and order, edges, pool, trace identity."""
    p = write_jsonl(tmp_path / "trace.jsonl", _multi_tree_records())

    # Serial reference: default threshold (8) with 4 trees stays in-process.
    serial = from_dynamo_trace(p, content_root_seed=_SEED)

    _force_parallel(monkeypatch, workers=2)
    parallel = from_dynamo_trace(p, content_root_seed=_SEED)

    # Assert equality, not counts.
    _assert_same_shape_and_content(parallel, serial)
    assert list(parallel.graph.nodes) == list(serial.graph.nodes)  # insertion order
    assert parallel.traces[0].tags == serial.traces[0].tags
    assert parallel.traces[0].id == serial.traces[0].id


def test_parallel_preserves_within_tree_parent_subagent_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batching keeps a whole tree together, so the parent-turn -> subagent edge survives the pool."""
    p = write_jsonl(tmp_path / "trace.jsonl", _multi_tree_records())

    _force_parallel(monkeypatch, workers=2)
    parallel = from_dynamo_trace(p, content_root_seed=_SEED)

    # P's first turn (1000) finished before S started (2000): within-tree edge.
    assert ("P:0", "S:0") in _edge_set(parallel)


def test_parallel_cross_file_subagent_tree_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent in one segment file plus its subagent in another still form ONE tree, byte-identical to the serial directory build."""
    d = tmp_path / "capture"
    write_jsonl(
        d / "a.jsonl",
        [
            _re(ts=1000, sid="P", hashes=[1, 2], ilen=32),
            _re(ts=3000, sid="P", hashes=[1, 2, 3], ilen=48),
            _re(ts=1000, sid="x", hashes=[40, 41], ilen=32),
        ],
    )
    write_jsonl(
        d / "b.jsonl",
        [
            _re(ts=2000, sid="S", parent="P", hashes=[7, 8], ilen=32),
            _re(ts=5000, sid="y", hashes=[50, 51], ilen=32),
            _re(ts=6000, sid="z", hashes=[60, 61], ilen=32),
        ],
    )

    serial = from_dynamo_trace(d, content_root_seed=_SEED)

    _force_parallel(monkeypatch, workers=2)
    parallel = from_dynamo_trace(d, content_root_seed=_SEED)

    # Grouping runs ONCE in the parent over the FULL scan, so the cross-file
    # link is unioned before batching and the tree lands in a single batch.
    assert ("P:0", "S:0") in _edge_set(parallel)
    _assert_same_shape_and_content(parallel, serial)


def test_parallel_preserves_duplicate_identity_source_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first duplicate request record remains authoritative across segments."""
    d = tmp_path / "capture"
    first = _re(ts=1000, sid="a", hashes=[10, 11], ilen=32)
    second = _re(ts=2000, sid="a", hashes=[10, 11, 12], ilen=48)
    second["request"]["request_id"] = first["request"]["request_id"]
    write_jsonl(d / "a.jsonl", [first])
    write_jsonl(d / "b.jsonl", [second])

    serial = from_dynamo_trace(d, content_root_seed=_SEED)
    _force_parallel(monkeypatch, workers=2)
    parallel = from_dynamo_trace(d, content_root_seed=_SEED)

    _assert_same_shape_and_content(parallel, serial)
    assert set(parallel.graph.nodes) == {"a:0"}


def test_parallel_preserves_first_parent_source_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first non-self parent remains authoritative across segments."""
    d = tmp_path / "capture"
    child_a = _re(ts=1000, sid="child", parent="parent-a", hashes=[7, 8], ilen=32)
    child_b = _re(ts=2000, sid="child", parent="parent-b", hashes=[9, 10], ilen=32)
    parent_a = _re(ts=500, sid="parent-a", hashes=[1, 2], ilen=32)
    parent_b = _re(ts=600, sid="parent-b", hashes=[3, 4], ilen=32)
    write_jsonl(d / "a.jsonl", [parent_a, child_a])
    write_jsonl(d / "b.jsonl", [parent_b, child_b])

    serial = from_dynamo_trace(d, content_root_seed=_SEED)
    _force_parallel(monkeypatch, workers=2)
    parallel = from_dynamo_trace(d, content_root_seed=_SEED)

    _assert_same_shape_and_content(parallel, serial)
    assert ("parent-a:0", "child:0") in _edge_set(parallel)
    assert ("parent-b:0", "child:0") not in _edge_set(parallel)


def test_shuffle_merges_fragments_in_segment_order(tmp_path: Path) -> None:
    """Final batch files concatenate source segments in order."""
    d = tmp_path / "capture"
    first = _re(ts=1000, sid="a", hashes=[10, 11], ilen=32)
    second = _re(ts=2000, sid="a", hashes=[10, 11, 12], ilen=48)
    second["request"]["request_id"] = first["request"]["request_id"]
    write_jsonl(d / "a.jsonl", [first])
    write_jsonl(d / "b.jsonl", [second])
    tmpdir = tmp_path / "shuffle"
    tmpdir.mkdir()

    batch_files = trace_parallel._shuffle_to_batch_files(
        d, {"a": 0}, tmpdir, threads=2, batch_count=1
    )
    with gzip.open(batch_files[0], "rb") as handle:
        lines = handle.readlines()
    assert orjson.loads(lines[0])["request"]["replay"]["input_length"] == 32
    assert orjson.loads(lines[1])["request"]["replay"]["input_length"] == 48


# --- 4. threshold fallback: no pool below threshold -----------------------


@pytest.mark.parametrize(
    "records,threshold,workers,reason",
    [
        param(
            _multi_tree_records(),
            8,
            None,
            "pool must not spawn below threshold",
            id="tree_count_below_threshold",
        ),
        param(
            _single_tree_records(),
            0,
            4,
            "single tree must not spawn a pool",
            id="single_tree_collapses_to_one_worker",
        ),
    ],
)  # fmt: skip
def test_maybe_build_fused_parallel_declines_and_skips_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[dict[str, Any]],
    threshold: int,
    workers: int | None,
    reason: str,
) -> None:
    """``maybe_build_fused_parallel`` returns None (serial fallback) without ever entering the pool."""
    p = write_jsonl(tmp_path / "trace.jsonl", records)
    monkeypatch.setattr(
        Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", threshold
    )
    if workers is not None:
        monkeypatch.setattr(
            Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_WORKERS", workers
        )
    _forbid_pool(monkeypatch, reason)

    assert _maybe_build(p) is None


def test_below_threshold_end_to_end_equals_and_skips_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A below-threshold end-to-end load builds the same graph as the reference while provably never spawning a pool."""
    p = write_jsonl(tmp_path / "trace.jsonl", _multi_tree_records())

    reference = from_dynamo_trace(p, content_root_seed=_SEED)

    # Threshold above the 4-tree count keeps the build serial; the pool entry
    # raising proves it is never reached.
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 8)
    _forbid_pool(monkeypatch, "pool must not spawn below threshold")
    serial = from_dynamo_trace(p, content_root_seed=_SEED)

    _assert_same_shape_and_content(serial, reference)


# --- 5. temp-dir cleanup on success AND exception -------------------------


def test_temp_dir_cleaned_up_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful fused build removes the shuffle temp dir it created."""
    created = _track_fused_temp_dirs(monkeypatch)

    p = write_jsonl(tmp_path / "trace.jsonl", _multi_tree_records())
    _force_parallel(monkeypatch, workers=2)
    from_dynamo_trace(p, content_root_seed=_SEED)

    assert created, "fused path must have created a temp dir"
    assert not any(Path(d).exists() for d in created), "temp dir left behind"


def test_temp_dir_cleaned_up_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pool round that raises after the shuffle still leaves no temp dir behind."""
    created = _track_fused_temp_dirs(monkeypatch)

    # Make the pool round blow up AFTER the temp dir + shuffle exist.
    from aiperf.dataset.graph.adapters.shared import pool as shared_pool

    def _boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("boom in pool round")

    monkeypatch.setattr(shared_pool, "run_pool_streaming", _boom)

    p = write_jsonl(tmp_path / "trace.jsonl", _multi_tree_records())
    _force_parallel(monkeypatch, workers=2)

    with pytest.raises(RuntimeError, match="boom in pool round"):
        from_dynamo_trace(p, content_root_seed=_SEED)

    assert created, "fused path must have created a temp dir"
    assert not any(Path(d).exists() for d in created), "temp dir left behind on error"


def _write_shuffle_segments(src: Path, *, segments: int, lines: int) -> None:
    """Write small segmented JSONL inputs for shuffle failure tests."""
    src.mkdir(exist_ok=True)
    for segment in range(segments):
        with gzip.open(src / f"trace.{segment:06d}.jsonl.gz", "wb") as handle:
            for index in range(lines):
                handle.write(
                    orjson.dumps(
                        _re(
                            ts=segment * lines + index,
                            sid=f"s{index % 8}",
                            hashes=[index],
                            ilen=1,
                        )
                    )
                    + b"\n"
                )


def test_shuffle_producer_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A segment producer failure aborts ordered fragment production."""
    src = tmp_path / "src"
    _write_shuffle_segments(src, segments=2, lines=4)
    out = tmp_path / "out"
    out.mkdir()

    def _boom_produce(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("producer-only-failure")

    monkeypatch.setattr(trace_parallel, "_shuffle_produce_segment", _boom_produce)

    with pytest.raises(RuntimeError, match="producer-only-failure"):
        trace_parallel._shuffle_to_batch_files(
            src, {f"s{i}": i % 4 for i in range(8)}, out, threads=4, batch_count=4
        )


def test_shuffle_fragment_close_failure_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fragment writer close failure cannot produce a partial success."""
    real_gzip_open = gzip.open

    class _CloseFailure:
        def write(self, data: bytes) -> int:
            return len(data)

        def close(self) -> None:
            raise OSError("fragment-close-failure")

    def _open(path: Any, *args: Any, **kwargs: Any) -> Any:
        mode = args[0] if args else kwargs.get("mode", "rb")
        if "w" in str(mode) and "segment_" in str(path):
            return _CloseFailure()
        return real_gzip_open(path, *args, **kwargs)

    src = tmp_path / "src"
    _write_shuffle_segments(src, segments=2, lines=4)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(gzip, "open", _open)

    with pytest.raises(OSError, match="fragment-close-failure"):
        trace_parallel._shuffle_to_batch_files(
            src, {f"s{i}": i % 4 for i in range(8)}, out, threads=2, batch_count=4
        )


# --- 6. determinism --------------------------------------------------------


def test_two_parallel_builds_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two independent fused builds of the same capture and seed agree exactly."""
    p = write_jsonl(tmp_path / "trace.jsonl", _multi_tree_records())

    _force_parallel(monkeypatch, workers=3)
    first = from_dynamo_trace(p, content_root_seed=999)
    second = from_dynamo_trace(p, content_root_seed=999)

    _assert_same_shape_and_content(first, second)
