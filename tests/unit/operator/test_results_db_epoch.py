# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for epoch surfacing + filtering in ResultsDB analytics queries.

The on-disk layout is ``<base>/<ns>/<job>/<epoch>/profile_export_aiperf.json``.
Multiple epoch dirs per job represent re-runs of the same AIPerfJob name.
Today's behaviour (one row per job, latest run only) must be preserved when
no ``epoch`` filter is supplied; explicit ``epoch=`` selects that historical
run instead.
"""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

pytest.importorskip("duckdb", reason="duckdb required for results_db tests")

from aiperf.operator.results_db import ResultsDB
from aiperf.operator.results_layout import write_latest


def _write_run(
    base_dir: Path,
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    throughput_avg: float = 100.0,
    model: str = "llama-7b",
    endpoint: str = "http://localhost:8000",
    start_time: str = "2026-01-15T10:00:00Z",
    end_time: str = "2026-01-15T10:05:00Z",
    is_latest: bool = False,
) -> Path:
    """Write a profile_export_aiperf.json for one (ns, job, epoch) run."""
    data = {
        "request_throughput": {
            "avg": throughput_avg,
            "p50": throughput_avg * 0.9,
            "p99": throughput_avg * 1.5,
            "unit": "req/s",
        },
        "request_latency": {
            "avg": 50.0,
            "p50": 40.0,
            "p99": 150.0,
            "unit": "ms",
        },
        "start_time": start_time,
        "end_time": end_time,
        "input_config": {
            "models": {"items": [{"name": model}]},
            "endpoint": {"urls": [endpoint]},
        },
    }
    job_dir = base_dir / namespace / job_id / epoch
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "profile_export_aiperf.json"
    path.write_bytes(orjson.dumps(data))
    if is_latest:
        write_latest(base_dir, namespace, job_id, epoch)
    return path


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    d = tmp_path / "results"
    d.mkdir()
    return d


@pytest.fixture
def db(results_dir: Path):
    instance = ResultsDB(results_dir)
    yield instance
    instance.close()


# ---------------------------------------------------------------------------
# Leaderboard surfaces epoch and defaults to latest
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_leaderboard_includes_epoch_column(
    results_dir: Path, db: ResultsDB
) -> None:
    _write_run(
        results_dir, "ns", "job-1", "1714069323", throughput_avg=100.0, is_latest=True
    )
    rows = await db.leaderboard()
    assert rows, "expected at least one leaderboard row"
    assert "epoch" in rows[0]
    assert rows[0]["epoch"] == "1714069323"


@pytest.mark.asyncio
async def test_leaderboard_default_returns_latest_epoch_only(
    results_dir: Path, db: ResultsDB
) -> None:
    # Two runs of the same (ns, job): older + latest.
    _write_run(results_dir, "ns", "job-1", "1714069323", throughput_avg=100.0)
    _write_run(
        results_dir, "ns", "job-1", "1714069400", throughput_avg=200.0, is_latest=True
    )
    rows = await db.leaderboard()
    # Today's contract: one row per job, freshest run.
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1714069400"
    assert rows[0]["value"] == 200.0


@pytest.mark.asyncio
async def test_leaderboard_explicit_epoch_returns_that_run(
    results_dir: Path, db: ResultsDB
) -> None:
    _write_run(results_dir, "ns", "job-1", "1714069323", throughput_avg=100.0)
    _write_run(
        results_dir, "ns", "job-1", "1714069400", throughput_avg=200.0, is_latest=True
    )
    rows = await db.leaderboard(epoch="1714069323")
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1714069323"
    assert rows[0]["value"] == 100.0


# ---------------------------------------------------------------------------
# History surfaces epoch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_includes_epoch_column(results_dir: Path, db: ResultsDB) -> None:
    _write_run(
        results_dir, "ns", "job-1", "1714069323", throughput_avg=100.0, is_latest=True
    )
    rows = await db.history()
    assert rows
    assert "epoch" in rows[0]
    assert rows[0]["epoch"] == "1714069323"


@pytest.mark.asyncio
async def test_history_default_only_latest_per_job(
    results_dir: Path, db: ResultsDB
) -> None:
    _write_run(results_dir, "ns", "job-1", "1714069323", throughput_avg=100.0)
    _write_run(
        results_dir, "ns", "job-1", "1714069400", throughput_avg=200.0, is_latest=True
    )
    rows = await db.history()
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1714069400"


@pytest.mark.asyncio
async def test_history_explicit_epoch_filter(results_dir: Path, db: ResultsDB) -> None:
    _write_run(results_dir, "ns", "job-1", "1714069323", throughput_avg=100.0)
    _write_run(
        results_dir, "ns", "job-1", "1714069400", throughput_avg=200.0, is_latest=True
    )
    rows = await db.history(epoch="1714069323")
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1714069323"
    assert rows[0]["value"] == 100.0


# ---------------------------------------------------------------------------
# Compare honours epoch surfacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compare_includes_epoch_per_row(results_dir: Path, db: ResultsDB) -> None:
    _write_run(
        results_dir, "ns", "job-1", "1714069323", throughput_avg=100.0, is_latest=True
    )
    _write_run(
        results_dir, "ns", "job-2", "1714069400", throughput_avg=200.0, is_latest=True
    )
    rows = await db.compare(["job-1", "job-2"])
    assert len(rows) == 2
    epochs = {r["job_id"]: r["epoch"] for r in rows}
    assert epochs == {"job-1": "1714069323", "job-2": "1714069400"}


@pytest.mark.asyncio
async def test_compare_default_only_latest_epoch_per_job(
    results_dir: Path, db: ResultsDB
) -> None:
    # Two runs of the same job; compare should only see the latest.
    _write_run(results_dir, "ns", "job-1", "1714069323", throughput_avg=100.0)
    _write_run(
        results_dir, "ns", "job-1", "1714069400", throughput_avg=200.0, is_latest=True
    )
    rows = await db.compare(["job-1"])
    assert len(rows) == 1
    assert rows[0]["epoch"] == "1714069400"
