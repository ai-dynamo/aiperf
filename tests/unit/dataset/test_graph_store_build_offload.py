# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DM3: both graph store builds run in worker threads, so their blocking drains must not freeze the DatasetManager event loop past ``PROFILE_CONFIGURE_TIMEOUT``."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiperf.dataset.graph.segment_trie.store_builder import TraceSegmentPayload
from aiperf.dataset.graph.store_build import GraphStoreBuilder


@contextlib.asynccontextmanager
async def _loop_ticker() -> AsyncIterator[Callable[[], int]]:
    """Run a background task that counts event-loop iterations, yielding a reader for the live count."""
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0)

    task = asyncio.create_task(ticker())
    try:
        yield lambda: ticks
    finally:
        task.cancel()


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
    """The HF streaming payload drain runs off-loop, so a concurrent ticker keeps advancing while payloads block."""
    ticks_at_payload: list[int] = []
    manager = _StubManager()

    async with _loop_ticker() as ticks:

        def slow_payloads():
            for i in range(3):
                # Stands in for the blocking multiprocessing result.get() the
                # real payload iterator performs per trace.
                time.sleep(0.05)
                ticks_at_payload.append(ticks())
                yield TraceSegmentPayload(
                    trace_id=f"t{i}",
                    node_ordinals={f"n{i}": 0},
                    envelopes=[],
                )

        catalog, merged = await GraphStoreBuilder._build_graph_store_streaming_trie(
            manager, slow_payloads(), tmp_path
        )

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
    from aiperf.dataset.graph.segment_trie import store_builder

    ticks_at_step: list[int] = []
    manager = _StubManager()

    async with _loop_ticker() as ticks:

        async def slow_interned_build(parsed: object, unified: object) -> dict:
            # Stands in for the zero-yield orjson-encode + pool-copy drain the
            # real builder performs before its trailing ``store.finalize()``.
            for _ in range(3):
                time.sleep(0.05)
                ticks_at_step.append(ticks())
            return {"t0": {"n0": 0}}

        monkeypatch.setattr(
            store_builder, "build_unified_trie_store_interned", slow_interned_build
        )
        catalog = await GraphStoreBuilder._build_interned_unified_store(
            manager, SimpleNamespace(), SimpleNamespace()
        )

    assert catalog == {"t0": {"n0": 0}}
    # A synchronous build freezes the loop for the whole drain: the ticker
    # never gets scheduled and every sample stays 0.
    assert ticks_at_step[-1] > 0, (
        "event loop made no progress while the eager interned store build was "
        "running; the build is blocking the DatasetManager loop"
    )
