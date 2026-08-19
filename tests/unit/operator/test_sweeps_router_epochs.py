# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.operator.routers.sweeps import create_sweeps_router


def _client(api: object | None, base: Path) -> TestClient:
    holder: list = [api]
    app = FastAPI()
    app.include_router(create_sweeps_router(holder, base))
    return TestClient(app)


def _write_aggregate(base: Path, ns: str, name: str, epoch: str, body: dict) -> Path:
    d = base / ns / "sweeps" / name / epoch
    d.mkdir(parents=True)
    (d / "aggregate.json").write_text(json.dumps(body))
    return d


def _write_children(base: Path, ns: str, name: str, epoch: str, children: list) -> None:
    p = base / ns / "sweeps" / name / epoch / "children.json"
    p.write_text(json.dumps({"sweep_run_epoch": epoch, "children": children}))


def test_get_sweep_with_epoch(tmp_path: Path) -> None:
    _write_aggregate(
        tmp_path,
        "bench",
        "s1",
        "1714069323",
        {
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
            "specSummary": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
            },
        },
    )
    _write_aggregate(
        tmp_path,
        "bench",
        "s1",
        "1714069400",
        {
            "phase": "Succeeded",
            "totalVariations": 8,
            "completedRuns": 8,
            "failedRuns": 0,
            "completedAt": "2026-04-26T01:00:00Z",
            "specSummary": {
                "sweep_type": "grid",
                "dimensions": [
                    {"name": "concurrency", "values": [1, 2, 4, 8, 16, 32, 64, 128]}
                ],
            },
        },
    )
    from aiperf.operator.results_layout import write_sweep_latest

    write_sweep_latest(tmp_path, "bench", "s1", "1714069400")
    api = MagicMock()
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep", AsyncMock()
        ) as mock_find,
        patch(
            "aiperf.operator.routers.sweeps.list_all_jobs", AsyncMock(return_value=[])
        ),
    ):
        from aiperf.operator import sweep_union

        async def _real_find(*args, **kw):
            return await sweep_union.find_any_sweep(*args, **kw)

        mock_find.side_effect = _real_find
        with patch(
            "aiperf.operator.sweep_union.find_aiperfsweep",
            AsyncMock(return_value=None),
        ):
            c = _client(api, tmp_path)
            r = c.get("/api/v1/sweeps/bench/s1?epoch=1714069323")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep"]["total_variations"] == 4


def test_list_sweep_epochs(tmp_path: Path) -> None:
    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {"phase": "Succeeded"})
    _write_aggregate(tmp_path, "bench", "s1", "1714069400", {"phase": "Succeeded"})
    from aiperf.operator.results_layout import write_sweep_latest

    write_sweep_latest(tmp_path, "bench", "s1", "1714069400")
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/sweeps/bench/s1/epochs")
    assert r.status_code == 200
    body = r.json()
    assert len(body["epochs"]) == 2
    assert body["epochs"][-1]["isLatest"] is True


def test_get_children_manifest(tmp_path: Path) -> None:
    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {"phase": "Succeeded"})
    _write_children(
        tmp_path,
        "bench",
        "s1",
        "1714069323",
        [
            {
                "namespace": "bench",
                "name": "s1-v00-t0",
                "variation_index": 0,
                "trial_index": 0,
                "child_run_epoch": "1714069324",
                "variation_label": "concurrency-1",
            },
        ],
    )
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/sweeps/bench/s1/children?epoch=1714069323")
    assert r.status_code == 200
    body = r.json()
    assert body["sweepRunEpoch"] == "1714069323"
    assert len(body["children"]) == 1
    assert body["children"][0]["childRunEpoch"] == "1714069324"


def test_get_children_missing_404(tmp_path: Path) -> None:
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/sweeps/bench/nope/children?epoch=1714069323")
    assert r.status_code == 404


def _live_record_with_aggregate_children(
    name: str, epoch: str, children: list[dict]
) -> object:
    """Build a SweepRecord whose live status mirrors the operator-pod reality:
    the controller-pod's children.json envelope is embedded under
    ``status.aggregate.children``, but no on-disk file is visible to the
    operator pod (the controller and operator use different PVCs).
    """
    from aiperf.operator.sweep_union import SweepRecord

    return SweepRecord(
        namespace="bench",
        name=name,
        source="live",
        phase="Succeeded",
        total_variations=len(children),
        completed_runs=len(children),
        failed_runs=0,
        age_seconds=120,
        model="m",
        raw_status={
            "phase": "Succeeded",
            "aggregate": {
                "children": {"sweep_run_epoch": epoch, "children": children},
            },
        },
    )


def test_get_children_reads_live_cr_when_disk_empty(tmp_path: Path) -> None:
    """Reproduces the prod 404: live sweep with status.aggregate.children
    populated, but no children.json on the operator pod's PVC."""
    rec = _live_record_with_aggregate_children(
        "s1",
        "1714069323",
        [
            {
                "namespace": "bench",
                "name": "s1-v00-t0",
                "variation_index": 0,
                "trial_index": 0,
                "child_run_epoch": "1714069324",
                "variation_label": "concurrency-1",
            },
        ],
    )
    api = MagicMock()
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=rec),
    ):
        c = _client(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/children")
    assert r.status_code == 200
    body = r.json()
    assert body["sweepRunEpoch"] == "1714069323"
    assert len(body["children"]) == 1
    assert body["children"][0]["name"] == "s1-v00-t0"
    assert body["children"][0]["childRunEpoch"] == "1714069324"


def test_get_children_explicit_epoch_skips_cr(tmp_path: Path) -> None:
    """An explicit ?epoch=X is a historical lookup — must read disk only,
    never fall back to the live CR (whose aggregate may be a different epoch)."""
    rec = _live_record_with_aggregate_children(
        "s1", "9999999999", [{"namespace": "bench", "name": "wrong-epoch-child"}]
    )
    api = MagicMock()
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=rec),
    ):
        c = _client(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/children?epoch=1714069323")
    assert r.status_code == 404


def test_get_children_falls_back_to_disk_when_cr_absent(tmp_path: Path) -> None:
    """Archived (post-TTL) sweep: CR is gone, but children.json sits on the
    PVC — the disk fallback must still work."""
    _write_aggregate(tmp_path, "bench", "s1", "1714069323", {"phase": "Succeeded"})
    _write_children(
        tmp_path,
        "bench",
        "s1",
        "1714069323",
        [
            {
                "namespace": "bench",
                "name": "s1-v00-t0",
                "variation_index": 0,
                "trial_index": 0,
                "child_run_epoch": "1714069324",
                "variation_label": "concurrency-1",
            },
        ],
    )
    from aiperf.operator.results_layout import write_sweep_latest

    write_sweep_latest(tmp_path, "bench", "s1", "1714069323")
    api = MagicMock()
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=None),
    ):
        c = _client(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/children")
    assert r.status_code == 200
    body = r.json()
    assert len(body["children"]) == 1
    assert body["children"][0]["childRunEpoch"] == "1714069324"


def test_get_cells_with_epoch_param(tmp_path: Path) -> None:
    _write_aggregate(
        tmp_path,
        "bench",
        "s1",
        "1714069323",
        {
            "phase": "Succeeded",
            "completedAt": "2026-04-25T01:00:00Z",
            "totalVariations": 1,
            "completedRuns": 1,
            "failedRuns": 0,
            "specSummary": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [8]}],
            },
            "per_cell_aggregates": [
                {
                    "variation_index": 0,
                    "variation_label": "concurrency-8",
                    "values": {"concurrency": 8},
                    "trials_completed": 1,
                    "trials_failed": 0,
                    "metrics": {"request_throughput": {"avg": 100.0}},
                    "children": [],
                },
            ],
        },
    )
    api = MagicMock()
    with patch(
        "aiperf.operator.sweep_union.find_aiperfsweep",
        AsyncMock(return_value=None),
    ):
        c = _client(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1/cells?epoch=1714069323")
    assert r.status_code == 200
    body = r.json()
    assert len(body["cells"]) == 1
    assert body["cells"][0]["metrics"]["request_throughput"]["avg"] == 100.0
