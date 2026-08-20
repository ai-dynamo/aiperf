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


def _patch_no_lazy_backfill(monkeypatch) -> None:
    """Keep index-backed endpoint tests from scheduling background DB writes."""
    from aiperf.operator import results_layout

    monkeypatch.setattr(
        results_layout, "_schedule_lazy_backfill_runs", lambda *args: None
    )


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


def test_get_job_epoch_dir_without_summary_returns_200_archived(
    tmp_path: Path, monkeypatch
) -> None:
    """An epoch dir on disk with no profile_export must serve a 200 archived
    stub instead of 404, so the run-detail page can render epochs that
    ``/epochs`` already enumerates (failed/cancelled runs).
    """
    (tmp_path / "bench" / "j1" / "1714069323").mkdir(parents=True)
    _patch_no_live_cr(monkeypatch)
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1?epoch=1714069323")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job"]["source"] == "archived"
    assert body["job"]["phase"] == "Unknown"
    assert body["job"]["throughputRps"] is None
    assert body["status"]["jobId"] == "j1"
    assert body["pods"] == []


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


def test_list_job_epochs_returns_status_unknown_when_index_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """Index-miss path: every row returns status=unknown."""
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")
    _patch_no_live_cr(monkeypatch)

    async def _no_raw(*_args, **_kwargs):
        return None

    from aiperf.operator.routers import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "get_raw_aiperfjob", _no_raw)

    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1/epochs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["epochs"]) == 1
    e = body["epochs"][0]
    assert e["status"] == "unknown"
    assert e["startedAt"] is None
    assert e["endedAt"] is None


def test_list_job_epochs_merges_stale_index_with_newer_disk_epoch(
    tmp_path: Path, monkeypatch
) -> None:
    """An indexed old epoch does not hide a newer disk-only epoch."""
    from aiperf.operator.results_layout import write_latest
    from aiperf.operator.routers import jobs as jobs_module
    from aiperf.operator.runs_index_models import RunIndexRow

    old_epoch = "1714069323"
    new_epoch = "1714069400"
    _write_summary(tmp_path, "bench", "j1", old_epoch)
    _write_summary(tmp_path, "bench", "j1", new_epoch)
    write_latest(tmp_path, "bench", "j1", new_epoch)

    async def _no_raw(*_args, **_kwargs):
        return None

    async def _stale_index_rows(namespace: str, job_id: str) -> list[RunIndexRow]:
        assert (namespace, job_id) == ("bench", "j1")
        return [
            RunIndexRow(
                namespace="bench",
                job_id="j1",
                epoch=old_epoch,
                phase="Succeeded",
                is_latest=True,
                start_time="2026-05-01T00:00:00+00:00",
                end_time="2026-05-01T00:05:00+00:00",
                created_unix=0,
                mtime_epoch=1,
                error=None,
                model=None,
                endpoint=None,
                gpu_count=0,
                gpu_name=None,
                file_count=0,
                total_size_bytes=0,
                sweep_namespace=None,
                sweep_name=None,
                sweep_epoch=None,
                sweep_variation_idx=None,
            )
        ]

    monkeypatch.setattr(jobs_module, "get_raw_aiperfjob", _no_raw)
    monkeypatch.setattr(jobs_module.runs_index, "list_runs_for_job", _stale_index_rows)

    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1/epochs")
    assert r.status_code == 200, r.text
    epochs = {entry["epoch"]: entry for entry in r.json()["epochs"]}
    assert set(epochs) == {old_epoch, new_epoch}
    assert epochs[old_epoch]["status"] == "succeeded"
    assert epochs[old_epoch]["startedAt"] == 1777593600
    assert epochs[new_epoch]["status"] == "unknown"
    assert epochs[new_epoch]["isLatest"] is True


def test_list_job_epochs_running_overrides_index_phase(
    tmp_path: Path, monkeypatch
) -> None:
    """Live in-flight epoch reports status=running even if index phase is stale."""
    import asyncio

    from aiperf.operator import runs_index
    from aiperf.operator.routers import jobs as jobs_module

    _patch_no_lazy_backfill(monkeypatch)
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")

    db = tmp_path / "_index.sqlite"
    asyncio.run(runs_index.open(db))
    try:
        asyncio.run(
            runs_index.upsert_run_created(
                "bench",
                "j1",
                "1714069400",
                spec={"models": {"items": [{"name": "m"}]}},
            )
        )
        # Stale phase: index says Succeeded; CR will say Running below.
        asyncio.run(
            runs_index.upsert_run_phase("bench", "j1", "1714069400", phase="Succeeded")
        )
        asyncio.run(runs_index.set_latest("bench", "j1", "1714069400"))

        async def _running_cr(*_args, **_kwargs):
            return {
                "metadata": {"name": "j1", "namespace": "bench"},
                "status": {"phase": "Running", "runEpoch": 1714069400},
            }

        monkeypatch.setattr(jobs_module, "get_raw_aiperfjob", _running_cr)

        api = MagicMock()
        c = _client(api, tmp_path)
        r = c.get("/api/v1/jobs/bench/j1/epochs")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["epochs"][0]["status"] == "running"
    finally:
        asyncio.run(runs_index.close())


def test_list_job_epochs_disk_fallback_marks_running_epoch(
    tmp_path: Path, monkeypatch
) -> None:
    """Disk-fallback path: row whose epoch matches the live runEpoch reports running."""
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")

    async def _running_cr(*_args, **_kwargs):
        return {
            "metadata": {"name": "j1", "namespace": "bench"},
            "status": {"phase": "Running", "runEpoch": 1714069400},
        }

    from aiperf.operator.routers import jobs as jobs_module

    monkeypatch.setattr(jobs_module, "get_raw_aiperfjob", _running_cr)

    # No runs_index.open() — the index path returns nothing, disk fallback runs.
    api = MagicMock()
    c = _client(api, tmp_path)
    r = c.get("/api/v1/jobs/bench/j1/epochs")
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["epochs"]) == 1
    assert body["epochs"][0]["status"] == "running"
    # No timestamps from disk fallback.
    assert body["epochs"][0]["startedAt"] is None
    assert body["epochs"][0]["endedAt"] is None


def test_list_job_epochs_phase_failed(tmp_path: Path, monkeypatch) -> None:
    """Phase=Failed in the index produces status=failed in the response."""
    import asyncio

    from aiperf.operator import runs_index
    from aiperf.operator.routers import jobs as jobs_module

    _patch_no_lazy_backfill(monkeypatch)
    _write_summary(tmp_path, "bench", "j1", "1714069400")
    from aiperf.operator.results_layout import write_latest

    write_latest(tmp_path, "bench", "j1", "1714069400")

    async def _no_raw(*_args, **_kwargs):
        return None

    monkeypatch.setattr(jobs_module, "get_raw_aiperfjob", _no_raw)

    db = tmp_path / "_index.sqlite"
    asyncio.run(runs_index.open(db))
    try:
        asyncio.run(
            runs_index.upsert_run_created(
                "bench",
                "j1",
                "1714069400",
                spec={"models": {"items": [{"name": "m"}]}},
            )
        )
        asyncio.run(
            runs_index.upsert_run_completed(
                "bench",
                "j1",
                "1714069400",
                summary_blob=b"",
                metrics={"metrics": {}},
                files=[],
                mtime_epoch=2,
                start_time="2026-05-01T00:00:00+00:00",
                end_time="2026-05-01T00:05:00+00:00",
                total_size_bytes=0,
                phase="Failed",
            )
        )
        asyncio.run(runs_index.set_latest("bench", "j1", "1714069400"))
        api = MagicMock()
        c = _client(api, tmp_path)
        r = c.get("/api/v1/jobs/bench/j1/epochs")
        body = r.json()
        assert body["epochs"][0]["status"] == "failed"
        # ISO timestamps from the index are surfaced as unix-seconds ints.
        assert body["epochs"][0]["startedAt"] == 1777593600
        assert body["epochs"][0]["endedAt"] == 1777593900
    finally:
        asyncio.run(runs_index.close())
