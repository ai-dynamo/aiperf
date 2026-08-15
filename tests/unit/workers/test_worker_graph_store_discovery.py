# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker opens the graph unified store from the broadcast, not env conventions."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from aiperf.common.models.dataset_models import GraphSegmentClientMetadata
from aiperf.workers.worker import Worker


def _bare_worker(meta: GraphSegmentClientMetadata | None) -> Worker:
    """Uninitialized Worker with only the graph-store discovery state populated."""
    w = Worker.__new__(Worker)
    w.run = SimpleNamespace(benchmark_id="b1")
    w._graph_client_metadata = meta
    w._graph_unified_open_attempted = False
    w._graph_unified_client = None
    w._graph_unified_open_failure = None
    w.warning = lambda *a, **k: None
    w.debug = lambda *a, **k: None
    return w


def test_reader_without_graph_broadcast_records_failure() -> None:
    """Absent broadcast metadata yields no reader and an actionable recorded failure."""
    # The failure string feeds GraphStoreUnavailable; there is deliberately no
    # env-convention fallback path.
    w = _bare_worker(None)
    assert w._graph_unified_reader() is None
    assert "GraphSegmentClientMetadata" in (w._graph_unified_open_failure or "")


def test_reader_opens_from_broadcast_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reader opens the client with exactly the base path and id from the broadcast."""
    meta = GraphSegmentClientMetadata(
        store_base_path=tmp_path,
        benchmark_id="b1",
        sidecar_path=tmp_path / "aiperf_graph_meta_b1" / "graph_meta.msgpack",
    )
    w = _bare_worker(meta)
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, base_path, benchmark_id) -> None:
            captured["base_path"] = base_path
            captured["benchmark_id"] = benchmark_id

        def open(self) -> "_FakeClient":
            return self

    # worker.py binds ``GraphSegmentUnifiedClient`` at import time, so patch the
    # name in the worker module, not its defining module.
    monkeypatch.setattr("aiperf.workers.worker.GraphSegmentUnifiedClient", _FakeClient)

    assert w._graph_unified_reader() is not None
    assert captured == {"base_path": tmp_path, "benchmark_id": "b1"}
