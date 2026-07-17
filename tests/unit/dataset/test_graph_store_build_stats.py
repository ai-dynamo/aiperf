# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph store build-stats snapshot: exact-field pins, cross-payload dedup, and
the GraphStoreBuilder build-complete log line.

The stats are a cheap ``O(traces) + O(1)`` snapshot computed at
``GraphSegmentUnifiedBackingStore.finalize()`` ENTRY -- the measurement baseline
that turns pool/envelope-size regressions into a visible log delta and a hard
CI failure instead of mystery RSS at corpus scale.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from aiperf.dataset import graph_segment_unified_store as store_mod
from aiperf.dataset.graph.adapters.dag_jsonl.trace import from_dag_jsonl
from aiperf.dataset.graph.segment_ir.store_builder import (
    TraceSegmentPayload,
    build_unified_trie_store_from_payloads,
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph.store_build import (
    GraphStoreBuilder,
    _format_store_build_stats,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphStoreBuildStats,
    NodeEnvelope,
)

DAG_FIXTURES = Path(__file__).parents[2] / "fixtures" / "dag"
# Spawn-only dag graph: a fast, slot-free trie parse (one trace carrying the
# root plus its spawned child) drained through the frozen trie pipeline.
SPAWN_MINIMAL_FIXTURE = DAG_FIXTURES / "spawn_minimal.dag.jsonl"


def _envelope_bytes(prompt_segment_ids: list[str]) -> bytes:
    """A minimal slot-free manifest envelope the streaming drain resolves."""
    return orjson.dumps(
        {
            "prompt_segment_ids": prompt_segment_ids,
            "dispatch_overrides": {},
            "stream": False,
        }
    )


@pytest.mark.asyncio
async def test_finalize_computes_build_stats_exact_fields(tmp_path: Path) -> None:
    """Pin the exact drained-store field values for a deterministic trie parse.

    These literals are structural products of the FROZEN trie pipeline over the
    spawn_minimal fixture. Tripping this assertion on an INTENTIONAL corpus or
    trie change is the DESIRED tripwire: update the literal deliberately after
    confirming the new build shape, do not delete the check.
    """
    parsed = from_dag_jsonl(str(SPAWN_MINIMAL_FIXTURE))
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="stats")

    assert store.build_stats is None, "build_stats must be None until finalize"

    await build_unified_trie_store_interned(parsed, store)

    stats = store.build_stats
    assert stats is not None
    assert stats.segment_count == 4, (
        "spawn_minimal interns 4 unique content segments (system+user per node, "
        "two nodes); a change here means the trie/corpus shape moved -- re-pin "
        "deliberately"
    )
    assert stats.content_bytes == 146, (
        "total interned content blob bytes for spawn_minimal; drifts with an "
        "intentional corpus-text change -- re-pin deliberately"
    )
    assert stats.node_manifest_count == 2, (
        "one manifest per trie LlmNode (root + spawned child); a change means "
        "the node set moved -- re-pin deliberately"
    )


@pytest.mark.asyncio
async def test_cross_payload_overlapping_segment_counted_once(tmp_path: Path) -> None:
    """A segment id shipped in TWO payloads is interned once (put_segment dedup).

    This is the memory property the snapshot guards: at weka corpus scale
    per-trace workers re-ship shared-prefix segments, so ``segment_count`` and
    ``content_bytes`` must count each content-addressed id exactly once no matter
    how many payloads carry it.
    """
    shared = ("seg-shared", "user", "shared-content", None)
    a_only = ("seg-a", "user", "a-only", None)
    b_only = ("seg-b", "user", "b-only", None)

    payloads = [
        TraceSegmentPayload(
            trace_id="t0",
            node_ordinals={"n0": 0},
            envelopes=[
                NodeEnvelope(0, "profiling", _envelope_bytes(["seg-shared", "seg-a"]))
            ],
            segments=[shared, a_only],
        ),
        TraceSegmentPayload(
            trace_id="t1",
            node_ordinals={"n0": 0},
            envelopes=[
                NodeEnvelope(0, "profiling", _envelope_bytes(["seg-shared", "seg-b"]))
            ],
            segments=[shared, b_only],
        ),
    ]

    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="dedup")
    await build_unified_trie_store_from_payloads(payloads, store)

    stats = store.build_stats
    assert stats is not None
    # 4 segment tuples supplied across the two payloads, but seg-shared repeats.
    assert stats.segment_count == 3, "overlapping segment must be interned once"

    expected_content_bytes = sum(
        len(orjson.dumps({"role": role, "content": content}))
        for _id, role, content, _wire in (shared, a_only, b_only)
    )
    assert stats.content_bytes == expected_content_bytes, (
        "content_bytes must count the shared segment's blob once, not twice"
    )


class _LogCaptureStub:
    """Minimal GraphStoreBuilder stand-in for the streaming build log site.

    Carries only what ``_build_graph_store_streaming_trie`` reads from ``self``,
    capturing ``info`` calls; the merge/sidecar hooks are no-ops so the drain
    runs and its build-complete log line (which fires before the merge) lands.
    """

    def __init__(self) -> None:
        self.run = SimpleNamespace(benchmark_id="bench-stats-log")
        self.infos: list[str] = []

    def info(self, msg: object) -> None:
        self.infos.append(msg() if callable(msg) else msg)

    def _merge_structural_graphs(self, structural_sink: list[bytes]) -> object:
        return object()

    def _write_graph_sidecar(
        self,
        merged: object,
        catalog: dict[str, dict[str, int]],
        base_path: Path,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_streaming_drain_logs_build_stats(tmp_path: Path) -> None:
    """The streaming build-complete log line carries the formatted store stats."""
    payload = TraceSegmentPayload(
        trace_id="t0",
        node_ordinals={"n0": 0},
        envelopes=[NodeEnvelope(0, "profiling", _envelope_bytes(["seg-x"]))],
        segments=[("seg-x", "user", "hello", None)],
    )
    stub = _LogCaptureStub()

    await GraphStoreBuilder._build_graph_store_streaming_trie(stub, [payload], tmp_path)

    build_logs = [m for m in stub.infos if "UNIFIED store built (streaming)" in m]
    assert len(build_logs) == 1, stub.infos
    line = build_logs[0]
    assert "segments=1" in line, line
    assert "content_bytes=" in line, line
    assert "node_manifests=1" in line, line
    assert "peak_rss_mib=" in line, line


@pytest.mark.asyncio
async def test_streaming_drain_failure_aborts_and_removes_store_dir(
    tmp_path: Path,
) -> None:
    """A mid-drain exception aborts the store and removes its dir.

    The store spills ``content.blob`` incrementally, so a drain that raises
    before finalize leaves a partial blob on disk. ``GraphStoreBuilder`` wraps
    the drain in ``abort()`` + ``rmtree``, so no half-written store survives for
    a later open. The error re-surfaces unchanged.
    """

    def _raising_payloads():
        yield TraceSegmentPayload(
            trace_id="t0",
            node_ordinals={"n0": 0},
            envelopes=[NodeEnvelope(0, "profiling", _envelope_bytes(["seg-x"]))],
            segments=[("seg-x", "user", "hello", None)],
        )
        raise RuntimeError("boom mid-drain")

    stub = _LogCaptureStub()
    stub.run = SimpleNamespace(benchmark_id="drain-abort")

    with pytest.raises(RuntimeError, match="boom mid-drain"):
        await GraphStoreBuilder._build_graph_store_streaming_trie(
            stub, _raising_payloads(), tmp_path
        )

    assert not (tmp_path / "aiperf_graph_segments_drain-abort").exists()


def test_format_store_build_stats_none_renders_unavailable() -> None:
    """A pre-finalize store (``build_stats is None``) logs a marker, not a crash."""
    assert "build_stats=unavailable" in _format_store_build_stats(None)


def test_format_store_build_stats_peak_rss_none_renders_na() -> None:
    """``peak_rss_mib=None`` (Windows / unavailable) renders as ``n/a``."""
    stats = GraphStoreBuildStats(
        segment_count=0,
        content_bytes=0,
        node_manifest_count=0,
        manifest_bytes=0,
        trace_count=0,
        peak_rss_mib=None,
    )
    assert "n/a" in _format_store_build_stats(stats)


@pytest.mark.skipif(
    store_mod.IS_WINDOWS, reason="resource module is unavailable on Windows"
)
def test_peak_rss_mib_macos_converts_bytes_to_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """macOS ``ru_maxrss`` is BYTES; the Darwin branch divides by 1024*1024."""
    import resource

    monkeypatch.setattr(store_mod, "IS_WINDOWS", False)
    monkeypatch.setattr(store_mod, "IS_MACOS", True)
    monkeypatch.setattr(
        resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=10 * 1024 * 1024)
    )
    assert store_mod._peak_rss_mib() == pytest.approx(10.0)


@pytest.mark.skipif(
    store_mod.IS_WINDOWS, reason="resource module is unavailable on Windows"
)
def test_peak_rss_mib_linux_converts_kib_to_mib(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Linux ``ru_maxrss`` is KiB; the non-Darwin branch divides by 1024."""
    import resource

    monkeypatch.setattr(store_mod, "IS_WINDOWS", False)
    monkeypatch.setattr(store_mod, "IS_MACOS", False)
    monkeypatch.setattr(
        resource, "getrusage", lambda who: SimpleNamespace(ru_maxrss=10 * 1024)
    )
    assert store_mod._peak_rss_mib() == pytest.approx(10.0)
