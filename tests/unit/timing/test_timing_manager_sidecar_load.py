# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TimingManager._load_graph_sidecar`` mandatory broadcast-ingest contract."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.common.exceptions import InvalidStateError
from aiperf.common.messages import DatasetConfiguredNotification
from aiperf.common.models.dataset_models import (
    DatasetMetadata,
    GraphSegmentClientMetadata,
)
from aiperf.config.resolution.plan import GraphWorkloadRef
from aiperf.dataset.graph.graph_meta_sidecar import write_graph_meta_sidecar
from aiperf.dataset.graph.models import GraphRecord, ParsedGraph, TraceRecord
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.manager import TimingManager


def _graph_meta(tmp_path: Path, sidecar: Path) -> GraphSegmentClientMetadata:
    """Broadcast client metadata pointing at ``sidecar`` under ``tmp_path``."""
    return GraphSegmentClientMetadata(
        store_base_path=tmp_path, benchmark_id="b1", sidecar_path=sidecar
    )


def _make_tm(client_metadata: GraphSegmentClientMetadata | None) -> TimingManager:
    """Bare TimingManager carrying only what the sidecar load path reads."""
    tm = TimingManager.__new__(TimingManager)  # bypass full service init
    tm.run = MagicMock()
    tm.run.benchmark_id = "b1"
    tm._graph_client_metadata = client_metadata
    tm.info = lambda *a, **k: None
    tm.debug = lambda *a, **k: None
    return tm


def _force_graph_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the workload detection seam report a graph run."""
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
    """Write a valid one-trace graph-meta sidecar and return its path."""
    pg = ParsedGraph(graph=GraphRecord(), traces=[TraceRecord(id="t-7", tags=["x"])])
    return write_graph_meta_sidecar(
        pg,
        base_path=tmp_path,
        benchmark_id="b1",
        source_fingerprint={},
        schema_version=1,
    )


def _no_broadcast(tmp_path: Path) -> GraphSegmentClientMetadata | None:
    """Graph run whose dataset broadcast never carried graph client metadata."""
    return None


def _advertised_but_absent(tmp_path: Path) -> GraphSegmentClientMetadata:
    """Broadcast advertises a sidecar path that was never written."""
    return _graph_meta(
        tmp_path, tmp_path / "aiperf_graph_meta_b1" / "graph_meta.msgpack"
    )


def _corrupt_sidecar(tmp_path: Path) -> GraphSegmentClientMetadata:
    """Broadcast points at a file that is not a msgpack frame."""
    sidecar = tmp_path / "aiperf_graph_meta_b1" / "graph_meta.msgpack"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"not-a-msgpack-frame")
    return _graph_meta(tmp_path, sidecar)


def _valid_sidecar(tmp_path: Path) -> GraphSegmentClientMetadata:
    """Broadcast points at a well-formed sidecar."""
    return _graph_meta(tmp_path, _write_sidecar(tmp_path))


def test_loads_sidecar_from_broadcast_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A graph run ingests the sidecar advertised on the dataset broadcast."""
    tm = _make_tm(_valid_sidecar(tmp_path))
    _force_graph_run(monkeypatch)
    result = tm._load_graph_sidecar()
    assert result is not None
    assert [t.id for t in result.traces] == ["t-7"]


def test_non_graph_run_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-graph run skips sidecar ingest entirely rather than erroring."""
    tm = _make_tm(None)
    monkeypatch.setattr(
        "aiperf.dataset.graph.workload_detect.resolve_graph_workload",
        lambda run: None,
    )
    assert tm._load_graph_sidecar() is None


@pytest.mark.parametrize(
    "make_metadata,fail_index_check,match",
    [
        param(_no_broadcast, False, "GraphSegmentClientMetadata", id="no_broadcast_metadata"),
        param(_advertised_but_absent, False, "missing", id="advertised_file_absent"),
        param(_corrupt_sidecar, False, "unreadable", id="corrupt_sidecar_bytes"),
        param(_valid_sidecar, True, "cross-check", id="index_check_verdict_false_raises"),
    ],
)  # fmt: skip
def test_graph_run_sidecar_faults_raise_invalid_state(
    make_metadata: Callable[[Path], GraphSegmentClientMetadata | None],
    fail_index_check: bool,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every sidecar ingest fault on a graph run is a fail-fast InvalidStateError."""
    tm = _make_tm(make_metadata(tmp_path))
    _force_graph_run(monkeypatch)
    if fail_index_check:
        tm._sidecar_passes_index_check = lambda graph, sidecar: False
    with pytest.raises(InvalidStateError, match=match):
        tm._load_graph_sidecar()


@pytest.mark.asyncio
async def test_notification_handler_stores_graph_client_metadata(
    tmp_path: Path,
) -> None:
    """The dataset-configured handler latches graph client metadata and wakes waiters."""
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


class _FakeClient:
    """Minimal stand-in for ``GraphSegmentUnifiedClient``."""

    def __init__(self, offsets: dict[str, dict[str, list[int]]]) -> None:
        self._node_offsets = offsets
        self.closed = False

    def open(self) -> "_FakeClient":
        return self

    def close(self) -> None:
        self.closed = True


def _patch_store(
    monkeypatch: pytest.MonkeyPatch, offsets: dict[str, dict[str, list[int]]]
) -> _FakeClient:
    """Make the unified store exist and open onto ``offsets``."""
    client = _FakeClient(offsets)
    monkeypatch.setattr(
        "aiperf.dataset.graph_segment_unified_store._unified_dir",
        lambda base_path, benchmark_id: Path("/"),
    )
    monkeypatch.setattr(
        "aiperf.dataset.graph_segment_unified_store.GraphSegmentUnifiedClient",
        lambda base_path, benchmark_id: client,
    )
    return client


def test_unparseable_store_inner_key_fails_index_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that opens but whose ordinals do not decode IS a divergence."""
    # Previously the decode ran inside the same blanket ``except Exception:
    # return True`` as the open probe, so a malformed manifest key silently
    # reported "index check passed" -- the exact failure this gate exists for.
    tm = _make_tm(_graph_meta(tmp_path, tmp_path / "graph_meta.msgpack"))
    client = _patch_store(monkeypatch, {"t-7": {"not-an-ordinal:base": [0, 1]}})
    pg = ParsedGraph(graph=GraphRecord(), traces=[TraceRecord(id="t-7")])
    assert tm._sidecar_passes_index_check(pg, tmp_path) is False
    assert client.closed


def test_unreachable_store_still_accepts_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store that cannot be opened remains "not reachable", not a divergence."""
    tm = _make_tm(_graph_meta(tmp_path, tmp_path / "graph_meta.msgpack"))
    monkeypatch.setattr(
        "aiperf.dataset.graph_segment_unified_store._unified_dir",
        lambda base_path, benchmark_id: Path("/"),
    )

    def _boom(base_path: Path, benchmark_id: str):
        raise OSError("store gone")

    monkeypatch.setattr(
        "aiperf.dataset.graph_segment_unified_store.GraphSegmentUnifiedClient",
        _boom,
    )
    pg = ParsedGraph(graph=GraphRecord(), traces=[TraceRecord(id="t-7")])
    assert tm._sidecar_passes_index_check(pg, tmp_path) is True
