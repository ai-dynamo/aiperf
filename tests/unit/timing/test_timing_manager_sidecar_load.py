# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TimingManager._load_graph_sidecar`` mandatory broadcast-ingest contract.

The DatasetManager writes the graph_meta sidecar on EVERY graph build route
and advertises its exact path on the graph-typed
``DatasetConfiguredNotification.client_metadata``; the schedule plane ingests
the sidecar from that broadcast path only. A graph run whose broadcast is not
graph-typed, or whose advertised file is missing, undecodable, or
store-divergent, is a hard configure-time failure. There is NO re-parse
fallback and NO env-convention path re-derivation.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.common.exceptions import InvalidStateError
from aiperf.common.models.dataset_models import GraphSegmentClientMetadata
from aiperf.config.resolution.plan import GraphWorkloadRef
from aiperf.dataset.graph.graph_meta_sidecar import write_graph_meta_sidecar
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord
from aiperf.timing.manager import TimingManager


def _graph_meta(tmp_path: Path, sidecar: Path) -> GraphSegmentClientMetadata:
    return GraphSegmentClientMetadata(
        store_base_path=tmp_path, benchmark_id="b1", sidecar_path=sidecar
    )


def _make_tm(client_metadata: GraphSegmentClientMetadata | None) -> TimingManager:
    tm = TimingManager.__new__(TimingManager)  # bypass full service init
    tm.run = MagicMock()
    tm.run.benchmark_id = "b1"
    tm._graph_client_metadata = client_metadata
    tm.info = lambda *a, **k: None
    tm.debug = lambda *a, **k: None
    return tm


def _force_graph_run(monkeypatch) -> None:
    # The single detection seam every consumer reads. Patched (not derived):
    # this MagicMock run never carries a real memo, and the accessor's strict
    # ``graph_workload_resolved is True`` check refuses the mock's
    # auto-truthified attributes.
    monkeypatch.setattr(
        "aiperf.dataset.graph.workload_detect.resolve_graph_workload",
        lambda run: GraphWorkloadRef(
            path=Path("/does/not/matter.json"), format="weka_trace"
        ),
    )


def _write_sidecar(tmp_path: Path) -> Path:
    pg = ParsedGraph(graph=GraphRecord(), traces=[TraceRecord(id="t-7", tags=["x"])])
    return write_graph_meta_sidecar(
        pg,
        base_path=tmp_path,
        benchmark_id="b1",
        source_fingerprint={},
        schema_version=1,
    )


def test_loads_sidecar_from_broadcast_path(tmp_path, monkeypatch):
    sidecar = _write_sidecar(tmp_path)
    tm = _make_tm(_graph_meta(tmp_path, sidecar))
    _force_graph_run(monkeypatch)
    result = tm._load_graph_sidecar()
    assert result is not None
    assert [t.id for t in result.traces] == ["t-7"]


def test_non_graph_run_returns_none(monkeypatch):
    tm = _make_tm(None)
    monkeypatch.setattr(
        "aiperf.dataset.graph.workload_detect.resolve_graph_workload",
        lambda run: None,
    )
    assert tm._load_graph_sidecar() is None


def test_graph_run_without_graph_broadcast_raises(monkeypatch):
    tm = _make_tm(None)
    _force_graph_run(monkeypatch)
    with pytest.raises(InvalidStateError, match="GraphSegmentClientMetadata"):
        tm._load_graph_sidecar()


def test_advertised_but_missing_file_raises(tmp_path, monkeypatch):
    missing = tmp_path / "aiperf_graph_meta_b1" / "graph_meta.msgpack"
    tm = _make_tm(_graph_meta(tmp_path, missing))
    _force_graph_run(monkeypatch)
    with pytest.raises(InvalidStateError, match="missing"):
        tm._load_graph_sidecar()


def test_corrupt_sidecar_raises(tmp_path, monkeypatch):
    sidecar = tmp_path / "aiperf_graph_meta_b1" / "graph_meta.msgpack"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"not-a-msgpack-frame")
    tm = _make_tm(_graph_meta(tmp_path, sidecar))
    _force_graph_run(monkeypatch)
    with pytest.raises(InvalidStateError, match="unreadable"):
        tm._load_graph_sidecar()


def test_index_mismatch_raises(tmp_path, monkeypatch):
    sidecar = _write_sidecar(tmp_path)
    tm = _make_tm(_graph_meta(tmp_path, sidecar))
    _force_graph_run(monkeypatch)
    tm._sidecar_passes_index_check = lambda graph, sidecar: False
    with pytest.raises(InvalidStateError, match="cross-check"):
        tm._load_graph_sidecar()


@pytest.mark.asyncio
async def test_notification_handler_stores_graph_client_metadata(tmp_path):
    import asyncio

    from aiperf.common.messages import DatasetConfiguredNotification
    from aiperf.common.models.dataset_models import DatasetMetadata
    from aiperf.plugin.enums import DatasetSamplingStrategy

    tm = _make_tm(None)
    tm._dataset_metadata = None
    tm._dataset_configured_event = asyncio.Event()
    sidecar = tmp_path / "aiperf_graph_meta_b1" / "graph_meta.msgpack"
    message = DatasetConfiguredNotification(
        service_id="dataset_manager",
        metadata=DatasetMetadata(
            conversations=[], sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
        ),
        client_metadata=_graph_meta(tmp_path, sidecar),
    )
    await tm._on_dataset_configured_notification(message)
    assert tm._graph_client_metadata is not None
    assert tm._graph_client_metadata.sidecar_path == sidecar
    assert tm._dataset_configured_event.is_set()
