# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.operator.routers.sweeps import create_sweeps_router
from aiperf.operator.sweep_union import SweepRecord


def _client_with(api: object | None, base_dir: Path) -> TestClient:
    holder: list = [api]
    app = FastAPI()
    app.include_router(create_sweeps_router(holder, base_dir))
    return TestClient(app)


def _live_record(name: str = "s1") -> SweepRecord:
    return SweepRecord(
        namespace="bench",
        name=name,
        source="live",
        phase="Running",
        total_variations=4,
        completed_runs=1,
        failed_runs=0,
        age_seconds=10,
        model="m",
        raw_spec={
            "template": {"spec": {"models": [{"name": "m"}]}},
            "sweep": {
                "type": "grid",
                "axes": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
            },
        },
        raw_status={
            "phase": "Running",
            "totalVariations": 4,
            "completedRuns": 1,
            "failedRuns": 0,
        },
    )


def test_list_returns_503_when_api_missing(tmp_path: Path) -> None:
    c = _client_with(None, tmp_path)
    r = c.get("/api/v1/sweeps")
    assert r.status_code == 503


def test_list_returns_records(tmp_path: Path) -> None:
    api = MagicMock()
    with patch(
        "aiperf.operator.routers.sweeps.list_all_sweeps",
        AsyncMock(return_value=[_live_record()]),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps")
    assert r.status_code == 200
    body = r.json()
    assert len(body["sweeps"]) == 1
    assert body["sweeps"][0]["name"] == "s1"
    assert body["sweeps"][0]["source"] == "live"


def test_detail_404_when_missing(tmp_path: Path) -> None:
    api = MagicMock()
    with patch(
        "aiperf.operator.routers.sweeps.find_any_sweep",
        AsyncMock(return_value=None),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/nope")
    assert r.status_code == 404


def test_detail_returns_spec_summary_from_live(tmp_path: Path) -> None:
    api = MagicMock()
    rec = _live_record()
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch(
            "aiperf.operator.routers.sweeps.list_all_jobs",
            AsyncMock(return_value=[]),
        ),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep"]["name"] == "s1"
    assert body["spec_summary"]["sweep_type"] == "grid"
    dim_names = [d["name"] for d in body["spec_summary"]["dimensions"]]
    assert "concurrency" in dim_names


def test_detail_archived_uses_synthesized_status(tmp_path: Path) -> None:
    sweep_dir = tmp_path / "bench" / "sweeps" / "s1"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "aggregate.json").write_text(
        json.dumps(
            {
                "phase": "Succeeded",
                "totalVariations": 4,
                "completedRuns": 4,
                "failedRuns": 0,
                "completedAt": "2026-04-25T01:00:00Z",
                "spec_snapshot": {
                    "sweep_type": "grid",
                    "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
                },
                "model": "m",
            }
        )
    )
    api = MagicMock()
    rec = SweepRecord(
        namespace="bench",
        name="s1",
        source="archived",
        phase="Succeeded",
        total_variations=4,
        completed_runs=4,
        failed_runs=0,
        age_seconds=999,
        model="m",
        aggregate_path=str(sweep_dir / "aggregate.json"),
        aggregate_doc={
            "phase": "Succeeded",
            "totalVariations": 4,
            "completedRuns": 4,
            "failedRuns": 0,
            "completedAt": "2026-04-25T01:00:00Z",
            "spec_snapshot": {
                "sweep_type": "grid",
                "dimensions": [{"name": "concurrency", "values": [1, 2, 4, 8]}],
            },
            "model": "m",
        },
    )
    with (
        patch(
            "aiperf.operator.routers.sweeps.find_any_sweep",
            AsyncMock(return_value=rec),
        ),
        patch(
            "aiperf.operator.routers.sweeps.list_all_jobs",
            AsyncMock(return_value=[]),
        ),
    ):
        c = _client_with(api, tmp_path)
        r = c.get("/api/v1/sweeps/bench/s1")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep"]["source"] == "archived"
    assert body["status"]["phase"] == "Succeeded"
    assert body["status"]["completedAt"] == "2026-04-25T01:00:00Z"
