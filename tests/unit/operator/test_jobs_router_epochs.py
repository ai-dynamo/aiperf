# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the jobs router ``?epoch=`` query and ``/epochs`` listing."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.operator.routers.jobs import create_jobs_router


def _client(api: object | None, base: Path) -> TestClient:
    holder: list = [api]
    app = FastAPI()
    app.include_router(create_jobs_router(holder, base))
    return TestClient(app)


def _write_summary(base: Path, ns: str, name: str, epoch: str) -> None:
    d = base / ns / name / epoch
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_text(
        json.dumps(
            {
                "status": "Succeeded",
                "input_config": {
                    "models": {"items": [{"name": "m"}]},
                    "endpoint": {"urls": ["x"]},
                },
                "request_throughput": {"avg": float(epoch[-3:])},
            }
        )
    )


def _patch_no_live_cr(monkeypatch) -> None:
    """Force the CR half of ``find_any_job`` / ``list_all_jobs`` to be empty."""
    from aiperf.operator import job_union as ju

    async def _no_cr(*_args, **_kwargs):
        return None

    monkeypatch.setattr(ju, "find_aiperf_job", _no_cr)


def test_get_job_with_epoch_param(tmp_path: Path, monkeypatch) -> None:
    _write_summary(tmp_path, "bench", "j1", "1714069323")
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")
    _patch_no_live_cr(monkeypatch)
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1?epoch=1714069323")
    assert r.status_code == 200, r.text
    body = r.json()
    assert abs(body["job"]["throughputRps"] - 323.0) < 0.001


def test_get_job_unknown_epoch_404(tmp_path: Path, monkeypatch) -> None:
    _patch_no_live_cr(monkeypatch)
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1?epoch=9999999999")
    assert r.status_code == 404


def test_get_job_malformed_epoch_400(tmp_path: Path) -> None:
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1?epoch=not-an-epoch")
    assert r.status_code == 400


def test_list_job_epochs(tmp_path: Path) -> None:
    _write_summary(tmp_path, "bench", "j1", "1714069323")
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1/epochs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["epochs"]) == 2
    epoch_strs = [e["epoch"] for e in body["epochs"]]
    assert epoch_strs == ["1714069323", "1714069400"]
    assert body["epochs"][-1]["isLatest"] is True
    assert body["epochs"][0]["isLatest"] is False
