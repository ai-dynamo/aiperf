# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runs_index.py — the SQLite-backed runs + sweep variation index."""

from __future__ import annotations

import asyncio
from pathlib import Path

import orjson
import pytest
import zstandard
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aiperf.operator import runs_index


@pytest.fixture
async def index_path(tmp_path: Path) -> Path:
    """Open a fresh runs_index DB rooted at tmp_path; close on teardown."""
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    yield path
    await runs_index.close()


@pytest.mark.asyncio
async def test_open_creates_schema_idempotently(tmp_path: Path) -> None:
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    await runs_index.close()
    assert path.exists()

    # Re-opening must not raise and must not duplicate rows in meta
    await runs_index.open(path)
    schema_version = await runs_index.get_meta("schema_version")
    assert schema_version == "1"
    await runs_index.close()


@pytest.mark.asyncio
async def test_integrity_check_detects_corruption(tmp_path: Path) -> None:
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    await runs_index.close()

    # Stomp the file with garbage
    path.write_bytes(b"not a sqlite db")
    ok = await runs_index.integrity_check(path)
    assert ok is False


@pytest.mark.asyncio
async def test_upsert_run_created_inserts_row(index_path) -> None:
    spec = {
        "benchmark": {
            "models": {"items": [{"name": "llama-3"}]},
            "endpoint": {"urls": ["http://server:8000"]},
        }
    }
    await runs_index.upsert_run_created("ns", "job-a", "1714069323", spec=spec)

    row = await runs_index.get_run("ns", "job-a", "1714069323")
    assert row is not None
    assert row.phase == "Pending"
    assert row.model == "llama-3"
    assert row.endpoint == "http://server:8000"
    assert row.is_latest is False  # not flipped yet


@pytest.mark.asyncio
async def test_upsert_run_phase_updates_only_phase(index_path) -> None:
    await runs_index.upsert_run_created("ns", "j", "100", spec={})
    await runs_index.upsert_run_phase("ns", "j", "100", phase="Running")

    row = await runs_index.get_run("ns", "j", "100")
    assert row.phase == "Running"
    assert row.end_time is None  # completion didn't happen


@pytest.mark.asyncio
async def test_upsert_run_completed_populates_metrics_and_blob(index_path) -> None:
    await runs_index.upsert_run_created("ns", "j", "100", spec={})

    metrics = {
        "request_throughput": {"avg": 42.5, "p50": 40.0, "p99": 50.0, "unit": "rps"},
        "request_latency": {"avg": 0.123, "p50": 0.1, "p99": 0.2, "unit": "s"},
        "telemetry_data": {
            "endpoints": {
                "e1": {"gpus": {"g1": {"gpu_name": "H100"}, "g2": {"gpu_name": "H100"}}}
            }
        },
    }
    summary_blob = zstandard.ZstdCompressor().compress(orjson.dumps(metrics))

    await runs_index.upsert_run_completed(
        "ns",
        "j",
        "100",
        summary_blob=summary_blob,
        metrics=metrics,
        files=["a.json", "b.parquet"],
        mtime_epoch=1714069400,
        end_time="2024-04-25T18:23:20Z",
    )

    row = await runs_index.get_run("ns", "j", "100")
    assert row.phase == "Succeeded"
    assert row.file_count == 2
    assert row.gpu_count == 2
    assert row.gpu_name == "H100"
    assert row.mtime_epoch == 1714069400

    blob = await runs_index.get_summary_blob("ns", "j", "100")
    assert (
        orjson.loads(zstandard.ZstdDecompressor().decompress(blob))[
            "request_throughput"
        ]["avg"]
        == 42.5
    )


@pytest.mark.asyncio
async def test_upsert_run_failed_records_error(index_path) -> None:
    await runs_index.upsert_run_created("ns", "j", "100", spec={})
    await runs_index.upsert_run_failed(
        "ns", "j", "100", error="OOMKilled", phase="Failed"
    )

    row = await runs_index.get_run("ns", "j", "100")
    assert row.phase == "Failed"
    assert row.error == "OOMKilled"


@pytest.mark.asyncio
async def test_set_latest_flips_one_row_only(index_path) -> None:
    for ep in ("100", "200", "300"):
        await runs_index.upsert_run_created("ns", "j", ep, spec={})
    await runs_index.set_latest("ns", "j", "200")

    rows = await runs_index.list_runs_for_job("ns", "j")
    latest = [r for r in rows if r.is_latest]
    assert len(latest) == 1
    assert latest[0].epoch == "200"

    # Re-pointing must atomically clear the prior is_latest
    await runs_index.set_latest("ns", "j", "300")
    rows = await runs_index.list_runs_for_job("ns", "j")
    assert sum(1 for r in rows if r.is_latest) == 1
    assert next(r for r in rows if r.is_latest).epoch == "300"


@pytest.mark.asyncio
async def test_delete_run_removes_row(index_path) -> None:
    await runs_index.upsert_run_created("ns", "j", "100", spec={})
    await runs_index.delete_run("ns", "j", "100")
    assert await runs_index.get_run("ns", "j", "100") is None


@pytest.mark.asyncio
async def test_concurrent_upsert_no_clobber(index_path) -> None:
    """The bug ``_index_lock`` papered over for jobs_index.json must not regress."""
    await runs_index.upsert_run_created("ns", "j", "100", spec={})

    await asyncio.gather(
        *[
            runs_index.upsert_run_phase("ns", "j", "100", phase=p)
            for p in ("Running", "Aggregating", "Running", "Succeeded")
        ]
    )

    # All four upserts must have written the row exactly once each — final
    # phase is one of the four, no row duplication, no missing row.
    rows = await runs_index.list_runs_for_job("ns", "j")
    assert len(rows) == 1
    assert rows[0].phase in {"Running", "Aggregating", "Succeeded"}


_PHASE_EVENTS = st.sampled_from(
    ["created", "phase_running", "phase_aggregating", "completed", "failed"]
)


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(events=st.lists(_PHASE_EVENTS, min_size=1, max_size=6))
@pytest.mark.asyncio
async def test_upsert_reordering_invariants(tmp_path: Path, events) -> None:
    db = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db)
    try:
        for e in events:
            if e == "created":
                await runs_index.upsert_run_created("ns", "j", "100", spec={})
            elif e == "phase_running":
                await runs_index.upsert_run_phase("ns", "j", "100", phase="Running")
            elif e == "phase_aggregating":
                await runs_index.upsert_run_phase("ns", "j", "100", phase="Aggregating")
            elif e == "completed":
                await runs_index.upsert_run_completed(
                    "ns",
                    "j",
                    "100",
                    summary_blob=b"",
                    metrics={},
                    files=[],
                    mtime_epoch=100,
                )
            elif e == "failed":
                await runs_index.upsert_run_failed("ns", "j", "100", error="x")

        # Invariants: exactly one row, phase set, no NULL pk fields
        rows = await runs_index.list_runs_for_job("ns", "j")
        assert len(rows) == 1
        assert rows[0].phase
        assert rows[0].namespace == "ns"
        assert rows[0].job_id == "j"
        assert rows[0].epoch == "100"
    finally:
        await runs_index.close()
