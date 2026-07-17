# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DM3: graph store builds must not freeze the DatasetManager event loop.

The HF streaming drain consumes a multiprocessing-backed iterator whose
``__next__`` blocks in ``result.get()`` per trace. Run synchronously on the
service loop that freezes heartbeats for the entire corpus-scale build (and
lets ``PROFILE_CONFIGURE_TIMEOUT`` kill a still-progressing configure).
``_build_graph_store_streaming_trie`` therefore drains the stream in a worker
thread; this test drives a deliberately slow payload iterator and asserts a
concurrent ticker keeps ticking while the stream is mid-drain.

The EAGER (local file/dir) path has the same failure shape: the interned
unified-store build is a zero-yield CPU drain (orjson-encode + pool copy of
ALL synthesized content) until its trailing ``finalize`` await, so
``_build_interned_unified_store`` runs it in a worker thread too. The second
test drives a deliberately blocking builder through that seam and asserts the
same ticker liveness.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiperf.dataset.graph.segment_ir.store_builder import TraceSegmentPayload
from aiperf.dataset.graph.store_build import GraphStoreBuilder


class _StubManager:
    """Just the attributes ``_build_graph_store_streaming_trie`` reads from self."""

    def __init__(self) -> None:
        self.run = SimpleNamespace(benchmark_id="bench-offload-test")
        self.infos: list[object] = []
        self.merge_calls: list[list[bytes]] = []
        self.sidecar_calls: list[tuple] = []
        self.merged_sentinel = object()

    def info(self, msg: object) -> None:
        self.infos.append(msg() if callable(msg) else msg)

    def _merge_structural_graphs(self, structural_sink: list[bytes]) -> object:
        self.merge_calls.append(structural_sink)
        return self.merged_sentinel

    def _write_graph_sidecar(
        self,
        merged: object,
        catalog: dict[str, dict[str, int]],
        base_path: Path,
    ) -> None:
        self.sidecar_calls.append((merged, catalog, base_path))


@pytest.mark.asyncio
async def test_streaming_trie_drain_keeps_event_loop_responsive(
    tmp_path: Path,
) -> None:
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticks_at_payload: list[int] = []

    def slow_payloads():
        for i in range(3):
            # Stands in for the blocking multiprocessing result.get() the
            # real payload iterator performs per trace.
            time.sleep(0.05)
            ticks_at_payload.append(ticks)
            yield TraceSegmentPayload(
                trace_id=f"t{i}",
                node_ordinals={f"n{i}": 0},
                envelopes=[],
            )

    manager = _StubManager()
    ticker_task = asyncio.create_task(ticker())
    try:
        catalog, merged = await GraphStoreBuilder._build_graph_store_streaming_trie(
            manager, slow_payloads(), tmp_path
        )
    finally:
        ticker_task.cancel()

    assert set(catalog) == {"t0", "t1", "t2"}
    assert merged is manager.merged_sentinel
    assert manager.sidecar_calls, "sidecar write must still run after the drain"
    # A synchronous drain freezes the loop for the whole stream: the ticker
    # never gets scheduled and every sample stays 0.
    assert ticks_at_payload[-1] > 0, (
        "event loop made no progress while the payload stream was draining; "
        "the drain is blocking the DatasetManager loop"
    )


@pytest.mark.asyncio
async def test_eager_interned_store_build_keeps_event_loop_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eager interned build runs off-loop; a concurrent ticker keeps ticking."""
    from aiperf.dataset.graph.segment_ir import store_builder

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    ticks_at_step: list[int] = []

    async def slow_interned_build(parsed: object, unified: object) -> dict:
        # Stands in for the zero-yield orjson-encode + pool-copy drain the
        # real builder performs before its trailing ``store.finalize()``.
        for _ in range(3):
            time.sleep(0.05)
            ticks_at_step.append(ticks)
        return {"t0": {"n0": 0}}

    monkeypatch.setattr(
        store_builder, "build_unified_trie_store_interned", slow_interned_build
    )

    manager = _StubManager()
    ticker_task = asyncio.create_task(ticker())
    try:
        catalog = await GraphStoreBuilder._build_interned_unified_store(
            manager, SimpleNamespace(), SimpleNamespace()
        )
    finally:
        ticker_task.cancel()

    assert catalog == {"t0": {"n0": 0}}
    # A synchronous build freezes the loop for the whole drain: the ticker
    # never gets scheduled and every sample stays 0.
    assert ticks_at_step[-1] > 0, (
        "event loop made no progress while the eager interned store build was "
        "running; the build is blocking the DatasetManager loop"
    )
