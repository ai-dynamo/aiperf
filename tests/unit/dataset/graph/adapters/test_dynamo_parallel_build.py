# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused read+build parallel dynamo session-tree build: scan, shuffle, equivalence.

The fused path replaces the earlier ship-collected-trees pool: a cheap grouping
scan (session ids + parent links only, NO ``input_sequence_hashes`` parse)
decides tree membership in the parent, raw record lines are shuffled to per-batch
temp files, and each batch is READ+BUILT inside a worker so the recorded hash
arrays never cross a process boundary. The acceptance bar is byte-equivalence to
the serial tree-scoped build: the pinned content seed + block size and CONTIGUOUS
weight-balanced batching make the parent's in-order union of worker results
identical to the serial per-tree loop -- same node keys, same edge set, same
content-addressed pool.

The pool is real (forkserver on Linux); fixtures are kept tiny and workers low so
the multiprocess tests stay fast and non-flaky.
"""

from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

import orjson
import pytest

from aiperf.common.environment import Environment
from aiperf.dataset.graph.adapters.dynamo import trace_parallel
from aiperf.dataset.graph.adapters.dynamo.trace import (
    DynamoTraceAdapterError,
    from_dynamo_trace,
    root_of_sessions,
)
from aiperf.dataset.graph.adapters.shared.content import CorpusContentSynthesizer

_SEED = 12345


@pytest.fixture(autouse=True)
def _fresh_synth_cache():
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
) -> dict:
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


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for r in records:
            f.write(orjson.dumps(r))
            f.write(b"\n")


def _multi_tree_records() -> list[dict]:
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


def _force_parallel(monkeypatch: pytest.MonkeyPatch, *, workers: int) -> None:
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 0)
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_WORKERS", workers)


def _edge_set(pg) -> set[tuple[str, str]]:
    return {(e.source, e.target) for e in pg.graph.edges}


# --- 1. contiguous weight batching (pure) ---------------------------------


def test_contiguous_batches_are_order_preserving_and_cover_all() -> None:
    items = list(range(10))
    weights = [1] * 10
    batches = trace_parallel._contiguous_weight_batches(items, weights, num_batches=3)
    flat = [x for batch in batches for x in batch]
    assert flat == items  # contiguous, every item once
    assert 1 <= len(batches) <= 3


def test_contiguous_batches_heavy_item_closes_its_own_batch() -> None:
    items = list(range(7))
    weights = [100] + [1] * 6  # one heavy leader
    batches = trace_parallel._contiguous_weight_batches(items, weights, num_batches=3)
    assert batches[0] == [0]
    assert [x for batch in batches for x in batch] == items


def test_contiguous_batches_single_batch_returns_all() -> None:
    items = [0, 1, 2, 3]
    assert trace_parallel._contiguous_weight_batches(
        items, [1, 1, 1, 1], num_batches=1
    ) == [items]


# --- 2. grouping scan: NO hash parse, cross-file links, block size --------


def test_scan_groups_sessions_and_parent_links(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())

    scan = trace_parallel._scan_grouping(p, threads=2)

    assert scan.request_end_sessions == {"P", "S", "a", "b", "c"}
    assert scan.parent_link == {"S": "P"}
    assert scan.block_size == 16
    # weight is the summed record-line byte length per session (a hash-free proxy)
    assert set(scan.session_weight) == {"P", "S", "a", "b", "c"}
    assert all(w > 0 for w in scan.session_weight.values())


def test_scan_ignores_no_context_and_marker_lines(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, [_re(ts=1000, sid="P", hashes=[1, 2], ilen=32)])
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
    # Only the session-bearing record is grouped; the marker / no-context record
    # carry no session id and are dropped (exactly as the serial reader drops them).
    assert scan.request_end_sessions == {"P"}
    assert scan.parent_link == {}


def test_scan_handles_sink_envelope_and_gzip(tmp_path: Path) -> None:
    """Real captures wrap each line in a ``{"timestamp","event"}`` envelope + gzip."""
    p = tmp_path / "trace.jsonl.gz"
    with gzip.open(p, "wb") as f:
        for rec in _multi_tree_records():
            f.write(orjson.dumps({"timestamp": 1, "event": rec}))
            f.write(b"\n")

    scan = trace_parallel._scan_grouping(p, threads=2)
    assert scan.request_end_sessions == {"P", "S", "a", "b", "c"}
    assert scan.parent_link == {"S": "P"}


def test_scan_rejects_mixed_block_size(tmp_path: Path) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _re(ts=1000, sid="P", hashes=[1, 2], ilen=32, bs=16),
            _re(ts=2000, sid="Q", hashes=[1, 2], ilen=64, bs=32),
        ],
    )
    with pytest.raises(DynamoTraceAdapterError, match="mixed replay trace_block_size"):
        trace_parallel._scan_grouping(p, threads=2)


def test_scan_cross_file_parent_link(tmp_path: Path) -> None:
    """A subagent in one file linking to a parent in another is one tree."""
    d = tmp_path / "capture"
    _write_jsonl(d / "a.jsonl", [_re(ts=1000, sid="P", hashes=[1, 2], ilen=32)])
    _write_jsonl(
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
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())

    # Serial reference: default threshold (8) with 4 trees stays in-process.
    serial = from_dynamo_trace(p, content_root_seed=_SEED)

    _force_parallel(monkeypatch, workers=2)
    parallel = from_dynamo_trace(p, content_root_seed=_SEED)

    # Assert equality, not counts.
    assert set(parallel.graph.nodes) == set(serial.graph.nodes)
    assert list(parallel.graph.nodes) == list(serial.graph.nodes)  # insertion order
    assert _edge_set(parallel) == _edge_set(serial)
    assert parallel.segment_pool is not None and serial.segment_pool is not None
    assert parallel.segment_pool.by_id == serial.segment_pool.by_id
    assert parallel.traces[0].tags == serial.traces[0].tags
    assert parallel.traces[0].id == serial.traces[0].id


def test_parallel_preserves_within_tree_parent_subagent_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())

    _force_parallel(monkeypatch, workers=2)
    parallel = from_dynamo_trace(p, content_root_seed=_SEED)

    # P's first turn (1000) finished before S started (2000): within-tree edge.
    assert ("P:0", "S:0") in _edge_set(parallel)


def test_parallel_cross_file_subagent_tree_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent in one segment file + its subagent in another still form ONE tree.

    Grouping happens ONCE in the parent over the FULL scan (all files), so a
    cross-file parent<->subagent link is unioned into a single tree BEFORE
    batching; the whole tree's records are shuffled to one batch and its
    within-tree edge survives. Byte-identical to the serial directory build.
    """
    d = tmp_path / "capture"
    _write_jsonl(
        d / "a.jsonl",
        [
            _re(ts=1000, sid="P", hashes=[1, 2], ilen=32),
            _re(ts=3000, sid="P", hashes=[1, 2, 3], ilen=48),
            _re(ts=1000, sid="x", hashes=[40, 41], ilen=32),
        ],
    )
    _write_jsonl(
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

    assert ("P:0", "S:0") in _edge_set(parallel)
    assert set(parallel.graph.nodes) == set(serial.graph.nodes)
    assert _edge_set(parallel) == _edge_set(serial)
    assert parallel.segment_pool.by_id == serial.segment_pool.by_id


# --- 4. threshold fallback: no pool below threshold -----------------------


def test_below_threshold_returns_none_no_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At/below threshold, ``maybe_build_fused_parallel`` returns None (serial)."""
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 8)

    def _no_pool(*_a, **_k):
        raise AssertionError("pool must not spawn below threshold")

    monkeypatch.setattr(trace_parallel, "_build_fused_parallel", _no_pool)
    result = trace_parallel.maybe_build_fused_parallel(
        p,
        content_root_seed=_SEED,
        idle_gap_cap_seconds=60.0,
        content_tokenizer=None,
        prompt_corpus="coding",
        release_replay=False,
        max_depth=8,
    )
    assert result is None


def test_below_threshold_end_to_end_equals_and_skips_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())

    reference = from_dynamo_trace(p, content_root_seed=_SEED)

    # Threshold above the 4-tree count keeps the build serial; the pool entry
    # raising proves it is never reached.
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 8)

    def _no_pool(*_a, **_k):
        raise AssertionError("pool must not spawn below threshold")

    monkeypatch.setattr(trace_parallel, "_build_fused_parallel", _no_pool)
    serial = from_dynamo_trace(p, content_root_seed=_SEED)

    assert set(serial.graph.nodes) == set(reference.graph.nodes)
    assert _edge_set(serial) == _edge_set(reference)
    assert serial.segment_pool.by_id == reference.segment_pool.by_id


def test_single_tree_stays_serial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Threshold 0 but one tree -> worker count collapses to 1 -> serial."""
    p = tmp_path / "trace.jsonl"
    _write_jsonl(
        p,
        [
            _re(ts=1000, sid="only", hashes=[1, 2], ilen=32),
            _re(ts=2000, sid="only", hashes=[1, 2, 3], ilen=48),
        ],
    )
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_THRESHOLD", 0)
    monkeypatch.setattr(Environment.DATASET, "DYNAMO_GRAPH_PARALLEL_WORKERS", 4)

    def _no_pool(*_a, **_k):
        raise AssertionError("single tree must not spawn a pool")

    monkeypatch.setattr(trace_parallel, "_build_fused_parallel", _no_pool)
    result = trace_parallel.maybe_build_fused_parallel(
        p,
        content_root_seed=_SEED,
        idle_gap_cap_seconds=60.0,
        content_tokenizer=None,
        prompt_corpus="coding",
        release_replay=False,
        max_depth=8,
    )
    assert result is None


# --- 5. temp-dir cleanup on success AND exception -------------------------


def test_temp_dir_cleaned_up_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*a, **k):
        d = real_mkdtemp(*a, **k)
        # Only the fused build's own temp dir; other machinery (pool, shm) may
        # also call mkdtemp and clean up on its own schedule.
        if Path(d).name.startswith("aiperf-dynamo-fused-"):
            created.append(d)
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)

    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())
    _force_parallel(monkeypatch, workers=2)
    from_dynamo_trace(p, content_root_seed=_SEED)

    assert created, "fused path must have created a temp dir"
    assert not any(Path(d).exists() for d in created), "temp dir left behind"


def test_temp_dir_cleaned_up_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def _tracking_mkdtemp(*a, **k):
        d = real_mkdtemp(*a, **k)
        # Only the fused build's own temp dir; other machinery (pool, shm) may
        # also call mkdtemp and clean up on its own schedule.
        if Path(d).name.startswith("aiperf-dynamo-fused-"):
            created.append(d)
        return d

    monkeypatch.setattr(tempfile, "mkdtemp", _tracking_mkdtemp)

    # Make the pool round blow up AFTER the temp dir + shuffle exist.
    from aiperf.dataset.graph.adapters.weka import trace_parallel as weka_tp

    def _boom(*_a, **_k):
        raise RuntimeError("boom in pool round")

    monkeypatch.setattr(weka_tp, "_run_pool_streaming", _boom)

    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())
    _force_parallel(monkeypatch, workers=2)

    with pytest.raises(RuntimeError, match="boom in pool round"):
        from_dynamo_trace(p, content_root_seed=_SEED)

    assert created, "fused path must have created a temp dir"
    assert not any(Path(d).exists() for d in created), "temp dir left behind on error"


# --- 6. determinism --------------------------------------------------------


def test_two_parallel_builds_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "trace.jsonl"
    _write_jsonl(p, _multi_tree_records())

    _force_parallel(monkeypatch, workers=3)
    first = from_dynamo_trace(p, content_root_seed=999)
    second = from_dynamo_trace(p, content_root_seed=999)

    assert set(first.graph.nodes) == set(second.graph.nodes)
    assert _edge_set(first) == _edge_set(second)
    assert first.segment_pool.by_id == second.segment_pool.by_id
