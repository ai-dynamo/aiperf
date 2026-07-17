# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GraphStoreBuilder._write_graph_sidecar`` mandatory-write contract.

The TimingManager only ingests the graph_meta sidecar from the path the
graph-typed dataset broadcast advertises (no re-parse fallback), so every
graph build route MUST land the file, record its path, or fail the run.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from aiperf.common.exceptions import DatasetError
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.codecs import encode_parsed_graph_msgpack
from aiperf.dataset.graph.graph_meta_sidecar import (
    sidecar_path_for,
    strip_replay_text,
)
from aiperf.dataset.graph.graph_path_catalog import build_graph_path_catalog
from aiperf.dataset.graph.merge import merge_parsed_graphs
from aiperf.dataset.graph.store_build import GraphStoreBuilder

WEKA_FIXTURE = Path(__file__).parents[1] / "graph" / "fixtures" / "weka_min.json"


class _StubManager:
    """Just the attributes ``_write_graph_sidecar`` reads/writes on self."""

    def __init__(self) -> None:
        self.run = SimpleNamespace(benchmark_id="bench-sidecar-test")
        self._sidecar_path: Path | None = None
        self.infos: list[object] = []

    def info(self, msg: object) -> None:
        self.infos.append(msg() if callable(msg) else msg)


@pytest.mark.skipif(not WEKA_FIXTURE.exists(), reason="weka fixture missing")
def test_write_graph_sidecar_writes_and_records_path(tmp_path):
    parsed = from_weka_trace(WEKA_FIXTURE, content_root_seed=0)
    structural = strip_replay_text(parsed)
    sink = [encode_parsed_graph_msgpack(structural)]
    catalog = build_graph_path_catalog(merge_parsed_graphs([structural]))
    manager = _StubManager()

    merged = GraphStoreBuilder._merge_structural_graphs(manager, sink)
    GraphStoreBuilder._write_graph_sidecar(manager, merged, catalog, tmp_path)

    expected = sidecar_path_for(tmp_path, "bench-sidecar-test")
    assert expected.exists()
    assert manager._sidecar_path == expected


@pytest.mark.skipif(not WEKA_FIXTURE.exists(), reason="weka fixture missing")
def test_write_graph_sidecar_catalog_mismatch_raises(tmp_path):
    parsed = from_weka_trace(WEKA_FIXTURE, content_root_seed=0)
    manager = _StubManager()
    with pytest.raises(DatasetError, match="catalog"):
        GraphStoreBuilder._write_graph_sidecar(
            manager, parsed, {"ghost-trace": {"n": 0}}, tmp_path
        )
    assert not sidecar_path_for(tmp_path, "bench-sidecar-test").exists()
    assert manager._sidecar_path is None


def test_merge_structural_graphs_empty_sink_raises():
    manager = _StubManager()
    with pytest.raises(DatasetError, match="no structural graphs"):
        GraphStoreBuilder._merge_structural_graphs(manager, [])


def test_merge_structural_graphs_undecodable_blob_raises():
    manager = _StubManager()
    with pytest.raises(DatasetError, match="failed to merge"):
        GraphStoreBuilder._merge_structural_graphs(manager, [b"not-msgpack"])
