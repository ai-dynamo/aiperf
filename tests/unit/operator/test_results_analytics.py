# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator web UI results_analytics router.

Covers the ``/api/v1/config/{namespace}/{job_id}`` fallback chain — specifically
the live-CR spec fallback that keeps the dashboard hero's SLO chips working
for running jobs with no on-disk artifacts.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiperf.operator import runs_index
from aiperf.operator.results_db import ResultsDB
from aiperf.operator.routers import results_analytics as mod
from aiperf.operator.routers.results_analytics import create_results_analytics_router


@pytest.fixture
async def _open_runs_index(tmp_path):
    """Open an empty runs_index DB so the get_run_spec lookup returns None
    cleanly instead of erroring on the unopened singleton."""
    await runs_index.open(tmp_path / ".aiperf_index.sqlite")
    yield
    await runs_index.close()


def _write_profile_export(
    base: Path,
    namespace: str,
    job_id: str,
    *,
    epoch: str = "1714064523",
    metric_val: float = 100.0,
    model: str = "llama-7b",
) -> None:
    from aiperf.operator.results_layout import run_dir, write_latest

    payload = orjson.dumps(
        {
            "request_throughput": {
                "avg": metric_val,
                "p50": metric_val * 0.9,
                "p99": metric_val * 1.5,
                "unit": "req/s",
            },
            "request_latency": {
                "avg": 50.0,
                "p50": 45.0,
                "p99": 120.0,
                "unit": "ms",
            },
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T10:05:00Z",
            "input_config": {
                "models": {"items": [{"name": model}]},
                "endpoint": {"urls": ["http://localhost:8000"]},
            },
        }
    )
    path = run_dir(base, namespace, job_id, epoch)
    path.mkdir(parents=True, exist_ok=True)
    (path / "profile_export_aiperf.json").write_bytes(payload)
    write_latest(base, namespace, job_id, epoch)


@pytest.mark.asyncio
async def test_results_db_leaderboard_falls_back_to_disk_without_index(tmp_path):
    await runs_index.close()
    _write_profile_export(tmp_path, "ns", "job-1", metric_val=123.0)
    db = ResultsDB(tmp_path)

    rows = await db.leaderboard(metric="request_throughput", stat="avg")

    assert rows == [
        {
            "namespace": "ns",
            "job_id": "job-1",
            "epoch": "1714064523",
            "value": 123.0,
            "unit": "req/s",
            "start_time": "2026-01-15T10:00:00Z",
            "end_time": "2026-01-15T10:05:00Z",
            "model": "llama-7b",
            "endpoint": "http://localhost:8000",
        }
    ]


@pytest.mark.asyncio
async def test_results_db_history_falls_back_to_disk_without_index(tmp_path):
    await runs_index.close()
    _write_profile_export(tmp_path, "ns", "job-1", metric_val=123.0)
    db = ResultsDB(tmp_path)

    rows = await db.history(metric="request_throughput", stat="avg")

    assert rows == [
        {
            "namespace": "ns",
            "job_id": "job-1",
            "epoch": "1714064523",
            "value": 123.0,
            "unit": "req/s",
            "start_time": "2026-01-15T10:00:00Z",
            "model": "llama-7b",
            "endpoint": "http://localhost:8000",
        }
    ]


@pytest.mark.asyncio
async def test_results_db_compare_falls_back_to_disk_without_index(tmp_path):
    await runs_index.close()
    _write_profile_export(tmp_path, "ns", "job-1", metric_val=123.0)
    db = ResultsDB(tmp_path)

    rows = await db.compare(job_ids=["job-1"], metrics=["request_throughput"])

    assert rows == [
        {
            "namespace": "ns",
            "job_id": "job-1",
            "epoch": "1714064523",
            "start_time": "2026-01-15T10:00:00Z",
            "model": "llama-7b",
            "endpoint": "http://localhost:8000",
            "gpu_count": 0,
            "gpu_name": None,
            "request_throughput_avg": 123.0,
            "request_throughput_p50": 110.7,
            "request_throughput_p99": 184.5,
            "request_throughput_unit": "req/s",
        }
    ]


@pytest.mark.asyncio
async def test_results_db_leaderboard_merges_stale_index_with_newer_disk_run(tmp_path):
    await runs_index.close()
    await runs_index.open(tmp_path / ".aiperf_index.sqlite")
    old_summary = {
        "request_throughput": {"avg": 10.0, "p50": 9.0, "p99": 15.0, "unit": "req/s"},
        "start_time": "2026-01-15T09:00:00Z",
        "end_time": "2026-01-15T09:05:00Z",
        "input_config": {
            "models": {"items": [{"name": "old-model"}]},
            "endpoint": {"urls": ["http://old.svc:8000"]},
        },
    }
    await runs_index.upsert_run_created(
        "ns", "job-1", "1714064523", spec={"benchmark": old_summary["input_config"]}
    )
    await runs_index.upsert_run_completed(
        "ns",
        "job-1",
        "1714064523",
        summary_blob=runs_index._zstd_compress(old_summary),
        metrics=old_summary,
        files=["profile_export_aiperf.json"],
        mtime_epoch=1714064523,
        start_time=old_summary["start_time"],
        end_time=old_summary["end_time"],
    )
    await runs_index.set_latest("ns", "job-1", "1714064523")
    _write_profile_export(
        tmp_path, "ns", "job-1", epoch="1714069999", metric_val=123.0, model="new-model"
    )
    db = ResultsDB(tmp_path)

    rows = await db.leaderboard(metric="request_throughput", stat="avg")

    assert [row["epoch"] for row in rows] == ["1714069999"]
    assert rows[0]["value"] == 123.0
    assert rows[0]["model"] == "new-model"
    await runs_index.close()


@pytest.mark.asyncio
async def test_results_db_history_merges_stale_index_with_newer_disk_run(tmp_path):
    await runs_index.close()
    await runs_index.open(tmp_path / ".aiperf_index.sqlite")
    old_summary = {
        "request_throughput": {"avg": 10.0, "p50": 9.0, "p99": 15.0, "unit": "req/s"},
        "start_time": "2026-01-15T09:00:00Z",
        "input_config": {
            "models": {"items": [{"name": "old-model"}]},
            "endpoint": {"urls": ["http://old.svc:8000"]},
        },
    }
    await runs_index.upsert_run_created(
        "ns", "job-1", "1714064523", spec={"benchmark": old_summary["input_config"]}
    )
    await runs_index.upsert_run_completed(
        "ns",
        "job-1",
        "1714064523",
        summary_blob=runs_index._zstd_compress(old_summary),
        metrics=old_summary,
        files=["profile_export_aiperf.json"],
        mtime_epoch=1714064523,
        start_time=old_summary["start_time"],
    )
    await runs_index.set_latest("ns", "job-1", "1714064523")
    _write_profile_export(
        tmp_path, "ns", "job-1", epoch="1714069999", metric_val=123.0, model="new-model"
    )
    db = ResultsDB(tmp_path)

    rows = await db.history(metric="request_throughput", stat="avg")

    assert [row["epoch"] for row in rows] == ["1714069999"]
    assert rows[0]["value"] == 123.0
    assert rows[0]["model"] == "new-model"
    await runs_index.close()


@pytest.mark.asyncio
async def test_results_db_compare_merges_stale_index_with_newer_disk_run(tmp_path):
    await runs_index.close()
    await runs_index.open(tmp_path / ".aiperf_index.sqlite")
    old_summary = {
        "request_throughput": {"avg": 10.0, "p50": 9.0, "p99": 15.0, "unit": "req/s"},
        "start_time": "2026-01-15T09:00:00Z",
        "input_config": {
            "models": {"items": [{"name": "old-model"}]},
            "endpoint": {"urls": ["http://old.svc:8000"]},
        },
    }
    await runs_index.upsert_run_created(
        "ns", "job-1", "1714064523", spec={"benchmark": old_summary["input_config"]}
    )
    await runs_index.upsert_run_completed(
        "ns",
        "job-1",
        "1714064523",
        summary_blob=runs_index._zstd_compress(old_summary),
        metrics=old_summary,
        files=["profile_export_aiperf.json"],
        mtime_epoch=1714064523,
        start_time=old_summary["start_time"],
    )
    await runs_index.set_latest("ns", "job-1", "1714064523")
    _write_profile_export(
        tmp_path, "ns", "job-1", epoch="1714069999", metric_val=123.0, model="new-model"
    )
    db = ResultsDB(tmp_path)

    rows = await db.compare(job_ids=["job-1"], metrics=["request_throughput"])

    assert [row["epoch"] for row in rows] == ["1714069999"]
    assert rows[0]["request_throughput_avg"] == 123.0
    assert rows[0]["model"] == "new-model"
    await runs_index.close()


@pytest.mark.asyncio
async def test_get_job_config_falls_back_to_live_cr_spec_when_index_closed(
    tmp_path, monkeypatch
):
    await runs_index.close()
    fake_cr = {
        "metadata": {"name": "live-job", "namespace": "aiperf-bench"},
        "spec": {"benchmark": {"slos": {"time_to_first_token": 500}}},
    }

    async def fake_get_raw(api, namespace, name):
        if namespace == "aiperf-bench" and name == "live-job":
            return fake_cr
        return None

    monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw, raising=False)
    db = ResultsDB(tmp_path)
    router = create_results_analytics_router(lambda: db, tmp_path, [object()])
    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/config/aiperf-bench/live-job")
        assert resp.status_code == 200, resp.text
        assert resp.json()["source"] == "cr"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_index_route_falls_back_to_disk_when_index_closed(tmp_path):
    await runs_index.close()
    _write_profile_export(tmp_path, "ns", "job-1", metric_val=123.0)
    db = ResultsDB(tmp_path)
    router = create_results_analytics_router(lambda: db, tmp_path, [None])
    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/index")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ns/job-1"]["epoch"] == "1714064523"
        assert body["ns/job-1"]["model"] == "llama-7b"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_get_job_config_falls_back_to_live_cr_spec(
    tmp_path, monkeypatch, _open_runs_index
):
    """When no file + no summary, config endpoint returns the live CR spec."""
    fake_cr = {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "metadata": {"name": "live-job", "namespace": "aiperf-bench"},
        "spec": {
            "benchmark": {
                "models": {"items": [{"name": "llama3-8b"}]},
                "endpoint": {"urls": ["http://llama3.svc:8000/v1"], "type": "chat"},
                "slos": {"time_to_first_token": 500},
            },
        },
    }

    async def fake_get_raw(api, namespace, name):
        if namespace == "aiperf-bench" and name == "live-job":
            return fake_cr
        return None

    monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw, raising=False)

    api_holder: list = [object()]  # sentinel so the None-guard passes
    db = ResultsDB(tmp_path)
    router = create_results_analytics_router(lambda: db, tmp_path, api_holder)

    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/config/aiperf-bench/live-job")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["source"] == "cr"
        assert body["spec"]["benchmark"]["slos"] == {"time_to_first_token": 500}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_get_job_config_returns_404_when_cr_missing(
    tmp_path, monkeypatch, _open_runs_index
):
    """When all fallbacks miss (no file, no summary, no live CR), still 404."""

    async def fake_get_raw(api, namespace, name):
        return None

    monkeypatch.setattr(mod, "get_raw_aiperfjob", fake_get_raw, raising=False)

    api_holder: list = [object()]
    db = ResultsDB(tmp_path)
    router = create_results_analytics_router(lambda: db, tmp_path, api_holder)

    app = FastAPI()
    app.include_router(router)
    try:
        with TestClient(app) as client:
            resp = client.get("/api/v1/config/aiperf-bench/missing-job")
        assert resp.status_code == 404
    finally:
        db.close()
