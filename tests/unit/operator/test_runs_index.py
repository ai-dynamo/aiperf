# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for runs_index.py — the SQLite-backed runs + sweep variation index."""

from __future__ import annotations

import asyncio
import sqlite3
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

    metrics_payload = {
        "request_throughput": {"avg": 42.5, "p50": 40.0, "p99": 50.0, "unit": "rps"},
        "request_latency": {"avg": 0.123, "p50": 0.1, "p99": 0.2, "unit": "s"},
        "telemetry_data": {
            "endpoints": {
                "e1": {"gpus": {"g1": {"gpu_name": "H100"}, "g2": {"gpu_name": "H100"}}}
            }
        },
    }
    summary_blob = zstandard.ZstdCompressor().compress(orjson.dumps(metrics_payload))
    metrics_envelope = {
        "aiperf_version": "0.8.0",
        "benchmark_id": "bench-1",
        "model": "mock-model",
        "endpoint_type": "chat",
        "streaming": False,
        "concurrency": 4,
        "request_rate": None,
        "metrics": metrics_payload,
        "telemetry_data": metrics_payload["telemetry_data"],
    }

    await runs_index.upsert_run_completed(
        "ns",
        "j",
        "100",
        summary_blob=summary_blob,
        metrics=metrics_envelope,
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

    narrow = await runs_index.get_run_narrow_metrics("ns", "j", "100")
    assert narrow is not None
    assert narrow["request_throughput_avg"] == 42.5
    assert narrow["request_latency_avg"] == 0.123

    blob = await runs_index.get_summary_blob("ns", "j", "100")
    assert (
        orjson.loads(zstandard.ZstdDecompressor().decompress(blob))[
            "request_throughput"
        ]["avg"]
        == 42.5
    )


@pytest.mark.asyncio
async def test_upsert_run_completed_uses_nested_metrics_payload(index_path) -> None:
    """Narrow compare columns come from the nested /api/metrics payload.

    Why: the completion handler passes the full controller ``/api/metrics``
    envelope into ``upsert_run_completed``. Flattening the envelope itself
    produces all-NULL narrow columns even though the summary blob is present
    and valid; only the nested ``metrics`` mapping carries the six compare
    metrics.
    """
    await runs_index.upsert_run_created("ns", "j", "101", spec={})
    payload = {
        "request_throughput": {"avg": 100.0, "p50": 95.0, "p99": 120.0, "unit": "rps"},
        "request_latency": {"avg": 0.5, "p50": 0.4, "p99": 0.9, "unit": "ms"},
        "output_token_throughput": {
            "avg": 900.0,
            "p50": 880.0,
            "p99": 990.0,
            "unit": "tok/s",
        },
    }
    await runs_index.upsert_run_completed(
        "ns",
        "j",
        "101",
        summary_blob=zstandard.ZstdCompressor().compress(orjson.dumps(payload)),
        metrics={
            "aiperf_version": "0.8.0",
            "benchmark_id": "bench-2",
            "model": "mock-model",
            "endpoint_type": "chat",
            "streaming": False,
            "concurrency": 4,
            "request_rate": None,
            "metrics": payload,
        },
        files=["profile_export_aiperf.json"],
        mtime_epoch=1714069401,
        end_time="2024-04-25T18:23:21Z",
    )

    narrow = await runs_index.get_run_narrow_metrics("ns", "j", "101")
    assert narrow is not None
    assert narrow["request_throughput_avg"] == 100.0
    assert narrow["request_latency_p50"] == 0.4
    assert narrow["output_token_throughput_p99"] == 990.0


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


@pytest.mark.asyncio
async def test_upsert_sweep_variation_inserts(index_path) -> None:
    metrics = {
        "request_throughput": {"avg": 100.0, "p50": 95.0, "p99": 110.0, "unit": "rps"},
        "request_latency": {"avg": 0.05, "p50": 0.05, "p99": 0.08, "unit": "s"},
    }
    await runs_index.upsert_sweep_variation(
        "ns",
        "satsweep",
        "1714069323",
        0,
        variation_values={"concurrency": 10},
        mode="REPEATED",
        phase="Succeeded",
        metrics=metrics,
        child_ref=("ns", "satsweep-c10", "1714069324"),
        metrics_blob=b"\x28\xb5\x2f\xfd",  # not real zstd; just a sentinel
    )

    rows = await runs_index.list_sweep_variations("ns", "satsweep", "1714069323")
    assert len(rows) == 1
    assert rows[0].variation_idx == 0
    assert rows[0].mode == "REPEATED"
    assert rows[0].child_job_id == "satsweep-c10"


@pytest.mark.asyncio
async def test_mark_sweep_pareto_sets_ranks_and_best(index_path) -> None:
    for idx in range(3):
        await runs_index.upsert_sweep_variation(
            "ns",
            "s1",
            "100",
            idx,
            variation_values={"concurrency": 10 * (idx + 1)},
            mode="INDEPENDENT",
            phase="Succeeded",
            metrics={},
            child_ref=None,
            metrics_blob=b"",
        )

    await runs_index.mark_sweep_pareto(
        "ns",
        "s1",
        "100",
        rankings=[(0, 1, False), (1, 0, True), (2, 2, False)],
    )

    rows = sorted(
        await runs_index.list_sweep_variations("ns", "s1", "100"),
        key=lambda r: r.variation_idx,
    )
    assert rows[0].pareto_rank == 1 and not rows[0].is_best
    assert rows[1].pareto_rank == 0 and rows[1].is_best
    assert rows[2].pareto_rank == 2 and not rows[2].is_best


@pytest.mark.asyncio
async def test_list_all_latest_returns_only_latest_rows(index_path) -> None:
    for ns, job, ep in [
        ("a", "j1", "100"),
        ("a", "j1", "200"),
        ("a", "j2", "100"),
        ("b", "j3", "100"),
    ]:
        await runs_index.upsert_run_created(ns, job, ep, spec={})
    await runs_index.set_latest("a", "j1", "200")
    await runs_index.set_latest("a", "j2", "100")
    await runs_index.set_latest("b", "j3", "100")

    rows = await runs_index.list_all_latest()
    keys = sorted((r.namespace, r.job_id, r.epoch) for r in rows)
    assert keys == [("a", "j1", "200"), ("a", "j2", "100"), ("b", "j3", "100")]


@pytest.mark.asyncio
async def test_bootstrap_walks_pvc_and_indexes_runs(tmp_path: Path) -> None:
    base = tmp_path / "results"
    # <base>/<ns>/<job>/<epoch>/profile_export_aiperf.json + ready marker
    run = base / "ns1" / "job-a" / "1714069323"
    run.mkdir(parents=True)
    (run / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(
            {
                "request_throughput": {
                    "avg": 5.0,
                    "p50": 5.0,
                    "p99": 6.0,
                    "unit": "rps",
                },
                "input_config": {
                    "models": {"items": [{"name": "m"}]},
                    "endpoint": {"urls": ["http://e"]},
                },
            }
        )
    )
    (run / ".aiperf_results_ready.json").write_text("{}")
    (base / "ns1" / "job-a" / "latest.txt").write_text("1714069323")

    # A sweeps-collision distractor — must be skipped
    (base / "ns1" / "sweeps" / "satsweep" / "1714069324").mkdir(parents=True)

    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 1
        rows = await runs_index.list_all_latest()
        assert len(rows) == 1
        assert rows[0].is_latest is True
        assert rows[0].model == "m"
    finally:
        await runs_index.close()


@pytest.mark.asyncio
async def test_bootstrap_indexes_k8s_sweep_children_from_real_run_artifacts(
    tmp_path: Path,
) -> None:
    """K8s sweep aggregate exports are run-level; per-cell metrics come from children."""
    base = tmp_path / "results"
    epoch_dir = base / "ns" / "sweeps" / "satsweep" / "1714069323"
    aggregate_dir = epoch_dir / "sweep_aggregate"
    aggregate_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(
        orjson.dumps(
            {
                "phase": "Succeeded",
                "totalVariations": 2,
                "completedRuns": 2,
                "failedRuns": 0,
                "childRuns": [
                    {"label": "concurrency=10", "status": "Succeeded", "error": ""},
                    {"label": "concurrency=20", "status": "Succeeded", "error": ""},
                ],
            }
        )
    )
    (epoch_dir / "children.json").write_bytes(
        orjson.dumps(
            {
                "sweep_run_epoch": "1714069323",
                "children": [
                    {
                        "namespace": "ns",
                        "name": "satsweep-v00",
                        "variation_index": 0,
                        "variation_label": "concurrency=10",
                        "trial_index": None,
                        "child_run_epoch": "1714069324",
                        "status": "Succeeded",
                    },
                    {
                        "namespace": "ns",
                        "name": "satsweep-v01",
                        "variation_index": 1,
                        "variation_label": "concurrency=20",
                        "trial_index": None,
                        "child_run_epoch": "1714069325",
                        "status": "Succeeded",
                    },
                ],
            }
        )
    )
    # Shape produced by AggregateConfidenceJsonExporter: run-level metadata + metrics,
    # not per_combination_metrics. It cannot identify per-variation rows by itself.
    (aggregate_dir / "profile_export_aiperf_aggregate.json").write_bytes(
        orjson.dumps(
            {
                "schema_version": "1.0",
                "aiperf_version": "0.8.0",
                "metadata": {
                    "aggregation_type": "confidence",
                    "num_profile_runs": 2,
                    "num_successful_runs": 2,
                    "failed_runs": [],
                    "sweep_mode": "INDEPENDENT",
                },
                "metrics": {
                    "request_throughput": {
                        "mean": 140.0,
                        "std": 56.6,
                        "min": 100.0,
                        "max": 180.0,
                        "cv": 0.4,
                        "se": 40.0,
                        "ci_low": -368.2,
                        "ci_high": 648.2,
                        "t_critical": 12.706,
                        "unit": "rps",
                    }
                },
            }
        )
    )
    for child_name, child_epoch, throughput in (
        ("satsweep-v00", "1714069324", 100.0),
        ("satsweep-v01", "1714069325", 180.0),
    ):
        run_dir = base / "ns" / child_name / child_epoch
        run_dir.mkdir(parents=True)
        (run_dir / "profile_export_aiperf.json").write_bytes(
            orjson.dumps(
                {
                    "metrics": {
                        "request_throughput": {
                            "avg": throughput,
                            "p50": throughput - 5.0,
                            "p99": throughput + 10.0,
                            "unit": "rps",
                        }
                    },
                    "input_config": {"models": {"items": [{"name": "m"}]}},
                }
            )
        )

    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 2
        assert stats.sweep_variations_indexed == 2
        rows = await runs_index.list_sweep_variations("ns", "satsweep", "1714069323")
        assert [r.variation_idx for r in rows] == [0, 1]
        assert rows[0].child_job_id == "satsweep-v00"
        assert rows[1].child_epoch == "1714069325"
        cur = await runs_index._conn().execute(
            "SELECT request_throughput_avg FROM sweep_variations "
            "WHERE namespace = ? AND sweep_name = ? AND variation_idx = ?",
            ("ns", "satsweep", 1),
        )
        assert (await cur.fetchone())[0] == 180.0
        await cur.close()
    finally:
        await runs_index.close()


@pytest.mark.asyncio
async def test_bootstrap_aggregates_repeated_trial_sweep_rows_by_variation_idx(
    tmp_path: Path,
) -> None:
    """Repeated-trial children aggregate into one truthful variation row."""
    base = tmp_path / "results"
    epoch_dir = base / "ns" / "sweeps" / "satsweep" / "1714069323"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(
        orjson.dumps(
            {
                "phase": "Succeeded",
                "totalVariations": 1,
                "completedRuns": 2,
                "failedRuns": 0,
            }
        )
    )
    (epoch_dir / "children.json").write_bytes(
        orjson.dumps(
            {
                "sweep_run_epoch": "1714069323",
                "children": [
                    {
                        "namespace": "ns",
                        "name": "satsweep-v00-t0",
                        "variation_index": 0,
                        "variation_label": "concurrency=10",
                        "trial_index": 0,
                        "child_run_epoch": "1714069324",
                        "status": "Succeeded",
                    },
                    {
                        "namespace": "ns",
                        "name": "satsweep-v00-t1",
                        "variation_index": 0,
                        "variation_label": "concurrency=10",
                        "trial_index": 1,
                        "child_run_epoch": "1714069325",
                        "status": "Succeeded",
                    },
                ],
            }
        )
    )
    for child_name, child_epoch, throughput in (
        ("satsweep-v00-t0", "1714069324", 100.0),
        ("satsweep-v00-t1", "1714069325", 120.0),
    ):
        run_dir = base / "ns" / child_name / child_epoch
        run_dir.mkdir(parents=True)
        (run_dir / "profile_export_aiperf.json").write_bytes(
            orjson.dumps(
                {
                    "metrics": {
                        "request_throughput": {
                            "avg": throughput,
                            "p50": throughput,
                            "p99": throughput,
                            "unit": "rps",
                        }
                    }
                }
            )
        )

    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        stats = await runs_index.bootstrap(base)
        rows = await runs_index.list_sweep_variations("ns", "satsweep", "1714069323")
        assert stats.sweep_variations_indexed == len(rows) == 1
        assert rows[0].variation_idx == 0
        assert rows[0].child_job_id is None
        assert rows[0].child_epoch is None

        cur = await runs_index._conn().execute(
            "SELECT request_throughput_avg, metrics_json FROM sweep_variations "
            "WHERE namespace = ? AND sweep_name = ? AND variation_idx = ?",
            ("ns", "satsweep", 0),
        )
        stored = await cur.fetchone()
        await cur.close()
        assert stored[0] == 110.0
        blob = orjson.loads(runs_index.zstd_decompress(stored[1]))
        assert (
            blob["aggregation_method"] == "mean_finite_numeric_default_compare_metrics"
        )
        assert blob["trial_count"] == 2
        assert [trial["child_ref"]["job_id"] for trial in blob["trials"]] == [
            "satsweep-v00-t0",
            "satsweep-v00-t1",
        ]
    finally:
        await runs_index.close()


@pytest.mark.asyncio
async def test_bootstrap_skips_incomplete_repeated_trial_variation(
    tmp_path: Path,
) -> None:
    """Duplicate child variation_idx must not become a false single-row aggregate."""
    base = tmp_path / "results"
    epoch_dir = base / "ns" / "sweeps" / "satsweep" / "1714069323"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(
        orjson.dumps(
            {
                "phase": "Succeeded",
                "totalVariations": 1,
                "completedRuns": 1,
                "failedRuns": 1,
            }
        )
    )
    (epoch_dir / "children.json").write_bytes(
        orjson.dumps(
            {
                "sweep_run_epoch": "1714069323",
                "children": [
                    {
                        "namespace": "ns",
                        "name": "satsweep-v00-t0",
                        "variation_index": 0,
                        "variation_label": "concurrency=10",
                        "trial_index": 0,
                        "child_run_epoch": "1714069324",
                        "status": "Succeeded",
                    },
                    {
                        "namespace": "ns",
                        "name": "satsweep-v00-t1",
                        "variation_index": 0,
                        "variation_label": "concurrency=10",
                        "trial_index": 1,
                        "child_run_epoch": "1714069325",
                        "status": "Failed",
                    },
                ],
            }
        )
    )
    run_dir = base / "ns" / "satsweep-v00-t0" / "1714069324"
    run_dir.mkdir(parents=True)
    (run_dir / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(
            {
                "metrics": {
                    "request_throughput": {
                        "avg": 100.0,
                        "p50": 100.0,
                        "p99": 100.0,
                        "unit": "rps",
                    }
                }
            }
        )
    )
    # The t1 profile is intentionally missing: idx=0 is a repeated-trial group,
    # so indexing only t0 would misrepresent it as a complete variation aggregate.

    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        stats = await runs_index.bootstrap(base)
        rows = await runs_index.list_sweep_variations("ns", "satsweep", "1714069323")
        assert stats.sweep_variations_indexed == 0
        assert rows == []
    finally:
        await runs_index.close()


@pytest.mark.asyncio
async def test_bootstrap_ingests_legacy_runs_without_ready_marker(
    tmp_path: Path,
) -> None:
    """Pre-marker-convention runs on the PVC must still be ingested at startup.

    The ``.aiperf_results_ready.json`` marker guards lazy-backfill against
    capturing mid-write runs; bootstrap runs at operator startup before any
    write is in flight, so the marker is not required there. Skipping these
    legacy runs would leave a fresh operator deploy onto an existing PVC
    showing an empty leaderboard until new runs land.
    """
    base = tmp_path / "results"
    # Use a real epoch matching EPOCH_RE (^\d{9,11}$) so list_run_epochs picks it up
    run = base / "ns" / "j" / "1714069323"
    run.mkdir(parents=True)
    (run / "profile_export_aiperf.json").write_bytes(orjson.dumps({}))
    (base / "ns" / "j" / "latest.txt").write_text("1714069323")
    # No .aiperf_results_ready.json — legacy run

    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        stats = await runs_index.bootstrap(base)
        assert stats.runs_indexed == 1
        rows = await runs_index.list_all_latest()
        assert len(rows) == 1
        assert rows[0].epoch == "1714069323"
    finally:
        await runs_index.close()


@pytest.mark.asyncio
async def test_bootstrap_prunes_index_rows_for_deleted_run_dirs(
    tmp_path: Path,
) -> None:
    """A completed index row whose run dir vanished from disk is pruned.

    Retention deletes the run dir first and schedules the index drop second;
    an operator crash between the two strands the row forever because the
    forward ingest walk only ever adds. Bootstrap's reverse pass must drop it.
    """
    base = tmp_path / "results"
    live = base / "ns1" / "job-a" / "1714069323"
    live.mkdir(parents=True)
    (live / "profile_export_aiperf.json").write_bytes(
        orjson.dumps({"request_throughput": {"avg": 5.0, "unit": "rps"}})
    )
    (base / "ns1" / "job-a" / "latest.txt").write_text("1714069323")

    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        # Stale completed row: dir was retention-deleted, drop was crash-lost.
        await runs_index.upsert_run_created("ns1", "job-a", "1714000000", spec={})
        await runs_index.upsert_run_completed(
            "ns1",
            "job-a",
            "1714000000",
            summary_blob=b"",
            metrics={"request_throughput": {"avg": 1.0, "unit": "rps"}},
            files=["profile_export_aiperf.json"],
            mtime_epoch=1714000001,
        )
        assert await runs_index.get_run("ns1", "job-a", "1714000000") is not None

        await runs_index.bootstrap(base)

        assert await runs_index.get_run("ns1", "job-a", "1714000000") is None, (
            "stale row for the deleted run dir must be pruned"
        )
        assert await runs_index.get_run("ns1", "job-a", "1714069323") is not None, (
            "the on-disk run must survive the prune and be (re)ingested"
        )
    finally:
        await runs_index.close()


@pytest.mark.asyncio
async def test_bootstrap_prune_spares_pending_rows_without_artifacts(
    tmp_path: Path,
) -> None:
    """A Pending stub (CR created, controller not yet writing results) has no
    run dir by design and must NOT be pruned at operator restart."""
    base = tmp_path / "results"
    base.mkdir(parents=True)

    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        await runs_index.upsert_run_created(
            "ns1", "job-pending", "1714069999", spec={"benchmark": {}}
        )

        await runs_index.bootstrap(base)

        row = await runs_index.get_run("ns1", "job-pending", "1714069999")
        assert row is not None
        assert row.phase == "Pending"
    finally:
        await runs_index.close()


@pytest.mark.asyncio
async def test_bootstrap_missing_base_does_not_prune(tmp_path: Path) -> None:
    """A missing results base (PVC unmounted mid-startup) must not be treated
    as "all runs deleted" — the prune pass is skipped entirely."""
    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    try:
        await runs_index.upsert_run_created("ns1", "job-a", "1714000000", spec={})
        await runs_index.upsert_run_completed(
            "ns1",
            "job-a",
            "1714000000",
            summary_blob=b"",
            metrics={"request_throughput": {"avg": 1.0, "unit": "rps"}},
            files=[],
            mtime_epoch=1714000001,
        )

        await runs_index.bootstrap(tmp_path / "not-mounted")

        assert await runs_index.get_run("ns1", "job-a", "1714000000") is not None
    finally:
        await runs_index.close()


@pytest.mark.asyncio
async def test_leaderboard_orders_by_metric(index_path) -> None:
    for ep, tput in [("100", 10.0), ("200", 50.0), ("300", 25.0)]:
        spec = {}
        await runs_index.upsert_run_created("ns", "j", ep, spec=spec)
        await runs_index.upsert_run_completed(
            "ns",
            "j",
            ep,
            summary_blob=b"",
            metrics={
                "request_throughput": {
                    "avg": tput,
                    "p50": tput,
                    "p99": tput,
                    "unit": "rps",
                },
            },
            files=[],
            mtime_epoch=int(ep),
        )
    await runs_index.set_latest("ns", "j", "200")  # only "200" is latest

    rows = await runs_index.leaderboard(
        metric="request_throughput", stat="avg", order="desc", limit=10
    )
    # Only the latest epoch participates
    assert len(rows) == 1
    assert rows[0]["value"] == 50.0


@pytest.mark.asyncio
async def test_compare_returns_metrics_for_named_jobs(index_path) -> None:
    metrics = {
        "request_throughput": {
            "avg": 100.0,
            "p50": 95.0,
            "p99": 110.0,
            "unit": "rps",
        },
        "request_latency": {"avg": 0.05, "p50": 0.05, "p99": 0.08, "unit": "s"},
    }
    for j in ("j1", "j2"):
        await runs_index.upsert_run_created("ns", j, "100", spec={})
        await runs_index.upsert_run_completed(
            "ns",
            j,
            "100",
            summary_blob=b"",
            metrics=metrics,
            files=[],
            mtime_epoch=100,
        )
        await runs_index.set_latest("ns", j, "100")

    rows = await runs_index.compare(["j1", "j2"], metrics=["request_throughput"])
    assert {r["job_id"] for r in rows} == {"j1", "j2"}
    assert all(r["request_throughput_avg"] == 100.0 for r in rows)


# ---------------------------------------------------------------------------
# Connection cleanup on failed open
# ---------------------------------------------------------------------------


def _spy_connect_close(monkeypatch: pytest.MonkeyPatch, closes: list[bool]) -> None:
    """Patch runs_index's aiosqlite.connect to record close() awaits."""
    real_connect = runs_index.aiosqlite.connect

    def spying_connect(*args: object, **kwargs: object):
        conn = real_connect(*args, **kwargs)
        real_close = conn.close

        async def tracked_close() -> None:
            closes.append(True)
            await real_close()

        conn.close = tracked_close
        return conn

    monkeypatch.setattr(runs_index.aiosqlite, "connect", spying_connect)


@pytest.mark.asyncio
async def test_open_closes_connection_when_post_connect_setup_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure between connect and _DB assignment must close the connection.

    aiosqlite spawns a non-daemon thread per connection and ResultsDB retries
    open() per request, so a leak here grows unboundedly.
    """
    closes: list[bool] = []
    real_connect = runs_index.aiosqlite.connect

    def connect_with_failing_setup(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        real_close = conn.close

        async def tracked_close() -> None:
            closes.append(True)
            await real_close()

        async def failing_executescript(*_args, **_kwargs):
            raise RuntimeError("simulated schema install failure")

        conn.close = tracked_close
        conn.executescript = failing_executescript
        return conn

    monkeypatch.setattr(runs_index.aiosqlite, "connect", connect_with_failing_setup)

    with pytest.raises(RuntimeError, match="simulated schema install failure"):
        await runs_index.open(tmp_path / ".aiperf_index.sqlite")

    assert closes == [True]
    assert runs_index.is_open() is False


@pytest.mark.asyncio
async def test_open_readonly_closes_connection_when_meta_table_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign sqlite file (no meta table) must not leak the connection."""
    path = tmp_path / ".aiperf_index.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE unrelated (id INTEGER)")
    conn.commit()
    conn.close()

    closes: list[bool] = []
    _spy_connect_close(monkeypatch, closes)

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        await runs_index.open_readonly(path)

    assert closes == [True]
    assert runs_index.is_open() is False


@pytest.mark.asyncio
async def test_open_readonly_closes_connection_exactly_once_on_missing_schema_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".aiperf_index.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.commit()
    conn.close()

    closes: list[bool] = []
    _spy_connect_close(monkeypatch, closes)

    with pytest.raises(RuntimeError, match="missing schema_version"):
        await runs_index.open_readonly(path)

    assert closes == [True]
    assert runs_index.is_open() is False


@pytest.mark.asyncio
async def test_open_closes_connection_when_schema_version_too_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    await runs_index._conn().execute(
        "UPDATE meta SET value = '999' WHERE key = 'schema_version'"
    )
    await runs_index.close()

    closes: list[bool] = []
    _spy_connect_close(monkeypatch, closes)

    with pytest.raises(RuntimeError, match="schema_version=999"):
        await runs_index.open(path)

    assert closes == [True]
    assert runs_index.is_open() is False
