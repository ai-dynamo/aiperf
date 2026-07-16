# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the Kubernetes runs_index SQLite cache.

Focuses on cache-never-source-of-truth failure modes:
- corrupt summary blobs during bootstrap and sweep child harvesting
- stale index rows merging with disk truth instead of hiding newer runs
- lazy backfill convergence after an index miss
- schema-version and partial-unique-index SQLite guards
- missing metric columns, duplicate sweep variation keys, and zstd decode fallout

Out of scope (covered elsewhere):
- Happy-path row upserts: tests/unit/operator/test_runs_index.py
- HTTP router response models: tests/unit/operator/test_results_api.py
"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import orjson
import pytest
import zstandard
from pytest import param

from aiperf.operator import results_layout, runs_index
from aiperf.operator.results_db import ResultsDB

# ============================================================================
# Helpers
# ============================================================================


def _zstd(payload: dict[str, object]) -> bytes:
    """Return a zstd-compressed JSON payload matching result artifact storage."""
    return zstandard.ZstdCompressor().compress(orjson.dumps(payload))


def _summary(
    *,
    throughput: float = 100.0,
    model_name: str = "meta-llama/Llama-3-8B",
    endpoint_url: str = "http://inference.local:8000",
) -> dict[str, object]:
    """Build a realistic profile summary with one compare metric and input config."""
    return {
        "start_time": "2026-05-18T11:00:00Z",
        "end_time": "2026-05-18T11:05:00Z",
        "request_throughput": {
            "avg": throughput,
            "p50": throughput - 5.0,
            "p99": throughput + 10.0,
            "unit": "rps",
        },
        "input_config": {
            "models": {"items": [{"name": model_name}]},
            "endpoint": {"urls": [endpoint_url]},
        },
    }


def _write_run_artifact(
    base: Path,
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    summary: dict[str, object] | None = None,
    zstd_summary: bool = False,
    corrupt_zstd: bool = False,
    latest: bool = True,
) -> Path:
    """Write one on-disk run directory and optionally point latest.txt at it."""
    run_dir = base / namespace / job_id / epoch
    run_dir.mkdir(parents=True)
    if corrupt_zstd:
        (run_dir / "profile_export_aiperf.json.zst").write_bytes(b"not-a-zstd-frame")
    elif zstd_summary:
        (run_dir / "profile_export_aiperf.json.zst").write_bytes(
            _zstd(summary or _summary())
        )
    else:
        (run_dir / "profile_export_aiperf.json").write_bytes(
            orjson.dumps(summary or _summary())
        )
    (run_dir / runs_index.READY_MARKER).write_bytes(b"{}")
    if latest:
        results_layout.write_latest(base, namespace, job_id, epoch)
    return run_dir


@pytest.fixture
async def opened_index(tmp_path: Path) -> AsyncGenerator[Path, None]:
    """Open a fresh writable runs_index DB for a single adversarial test."""
    path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(path)
    try:
        yield path
    finally:
        await runs_index.close()


# ============================================================================
# Corrupt blobs and stale cache fallbacks
# ============================================================================


@pytest.mark.asyncio
async def test_bootstrap_corrupt_zstd_summary_skips_bad_run_and_indexes_sibling(
    tmp_path: Path,
    opened_index: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A corrupt summary blob must not abort PVC bootstrap for adjacent runs."""
    base = tmp_path / "results"
    _write_run_artifact(
        base,
        "bench-prod",
        "llama-smoke-corrupt",
        "1716060001",
        corrupt_zstd=True,
        latest=False,
    )
    _write_run_artifact(
        base,
        "bench-prod",
        "llama-smoke-good",
        "1716060002",
        summary=_summary(throughput=212.5),
    )

    stats = await runs_index.bootstrap(base)

    assert stats.runs_indexed == 1
    assert (
        await runs_index.get_run("bench-prod", "llama-smoke-corrupt", "1716060001")
        is None
    )
    good = await runs_index.get_run("bench-prod", "llama-smoke-good", "1716060002")
    assert good is not None
    assert good.model == "meta-llama/Llama-3-8B"
    assert "cannot read summary" in caplog.text


@pytest.mark.asyncio
async def test_resultsdb_summary_stale_latest_index_falls_back_to_disk_latest(
    tmp_path: Path,
    opened_index: Path,
) -> None:
    """A stale latest row whose directory was pruned must not hide disk truth."""
    base = tmp_path / "results"
    _write_run_artifact(
        base,
        "bench-prod",
        "llama-regression-7f2a",
        "1716060102",
        summary=_summary(throughput=314.0),
    )
    await runs_index.upsert_run_created(
        "bench-prod",
        "llama-regression-7f2a",
        "1716060101",
        spec={"benchmark": {}},
    )
    await runs_index.upsert_run_completed(
        "bench-prod",
        "llama-regression-7f2a",
        "1716060101",
        summary_blob=_zstd(_summary(throughput=1.0)),
        metrics=_summary(throughput=1.0),
        files=["profile_export_aiperf.json"],
        mtime_epoch=1716060101,
    )
    await runs_index.set_latest("bench-prod", "llama-regression-7f2a", "1716060101")

    summary = await ResultsDB(base).summary("bench-prod", "llama-regression-7f2a")

    assert summary is not None
    assert summary["request_throughput"]["avg"] == 314.0


@pytest.mark.asyncio
async def test_list_runs_async_index_miss_schedules_lazy_backfill(
    tmp_path: Path,
    opened_index: Path,
) -> None:
    """Disk fallback should converge the writable index via lazy backfill."""
    base = tmp_path / "results"
    _write_run_artifact(
        base,
        "bench-prod",
        "lazy-backfill-bench-9c3a",
        "1716060201",
        summary=_summary(throughput=155.0),
    )

    before = asyncio.all_tasks()
    entries = await results_layout.list_runs_async(
        base, "bench-prod", "lazy-backfill-bench-9c3a"
    )
    scheduled = asyncio.all_tasks() - before - {asyncio.current_task()}
    if scheduled:
        await asyncio.gather(*scheduled)

    assert [entry.epoch for entry in entries] == ["1716060201"]
    row = await runs_index.get_run(
        "bench-prod", "lazy-backfill-bench-9c3a", "1716060201"
    )
    assert row is not None
    assert row.phase == "Succeeded"
    assert row.is_latest is True
    narrow = await runs_index.get_run_narrow_metrics(
        "bench-prod", "lazy-backfill-bench-9c3a", "1716060201"
    )
    assert narrow is not None
    assert narrow["request_throughput_avg"] == 155.0


# ============================================================================
# SQLite guardrails and missing-column contracts
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "readonly",
    [
        False,
        True,
    ],
)  # fmt: skip
async def test_open_schema_version_newer_than_code_refuses_db(
    tmp_path: Path,
    readonly: bool,
) -> None:
    """Forward schema versions must fail closed instead of reading unknown rows."""
    db_path = tmp_path / ".aiperf_index.sqlite"
    await runs_index.open(db_path)
    await runs_index.close()
    with sqlite3.connect(db_path) as db:
        db.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(runs_index.SCHEMA_VERSION + 7),),
        )

    opener = runs_index.open_readonly if readonly else runs_index.open
    with pytest.raises(RuntimeError, match=r"schema_version=8.*only knows up to 1"):
        await opener(db_path)
    await runs_index.close()


@pytest.mark.asyncio
async def test_runs_one_latest_partial_unique_index_rejects_dual_latest(
    opened_index: Path,
) -> None:
    """A second writer mistake must hit SQLite's unique guard, not split brain."""
    for epoch in ("1716060301", "1716060302"):
        await runs_index.upsert_run_created(
            "bench-prod", "unique-latest-bench-4b1d", epoch, spec={"benchmark": {}}
        )
    await runs_index.set_latest("bench-prod", "unique-latest-bench-4b1d", "1716060301")

    with pytest.raises(sqlite3.IntegrityError, match=r"UNIQUE constraint failed"):
        await runs_index._conn().execute(
            "UPDATE runs SET is_latest = 1 "
            "WHERE namespace = ? AND job_id = ? AND epoch = ?",
            ("bench-prod", "unique-latest-bench-4b1d", "1716060302"),
        )

    latest = await runs_index.list_all_latest()
    assert [(row.job_id, row.epoch) for row in latest] == [
        ("unique-latest-bench-4b1d", "1716060301")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_name",
    [
        param("leaderboard", id="leaderboard-missing-metric-column"),
        param("history", id="history-missing-metric-column"),
        param("compare", id="compare-missing-metric-column"),
    ],
)  # fmt: skip
async def test_analytics_missing_metric_column_returns_empty_list(
    opened_index: Path,
    query_name: str,
) -> None:
    """User-supplied metric names without flat columns return no rows, not 500s."""
    await runs_index.upsert_run_created(
        "bench-prod", "custom-metric-bench-2e9f", "1716060401", spec={"benchmark": {}}
    )
    await runs_index.set_latest("bench-prod", "custom-metric-bench-2e9f", "1716060401")

    if query_name == "leaderboard":
        rows = await runs_index.leaderboard(metric="custom_metric", stat="avg")
    elif query_name == "history":
        rows = await runs_index.history(metric="custom_metric", stat="avg")
    else:
        rows = await runs_index.compare(
            ["custom-metric-bench-2e9f"], metrics=["custom_metric"]
        )

    assert rows == []


# ============================================================================
# Sweep variation adversaries
# ============================================================================


@pytest.mark.asyncio
async def test_upsert_sweep_variation_duplicate_idx_replaces_row_not_duplicates(
    opened_index: Path,
) -> None:
    """The sweep PK is one row per variation_idx; reruns refresh the same cell."""
    await runs_index.upsert_sweep_variation(
        "bench-prod",
        "latency-sweep-5d1c",
        "1716060501",
        0,
        variation_values={"benchmark.phases.default.concurrency": 16},
        mode="INDEPENDENT",
        phase="Running",
        metrics={"request_throughput": {"avg": 100.0, "unit": "rps"}},
        child_ref=("bench-prod", "latency-sweep-v00-old", "1716060502"),
        metrics_blob=_zstd({"attempt": "old"}),
    )
    await runs_index.upsert_sweep_variation(
        "bench-prod",
        "latency-sweep-5d1c",
        "1716060501",
        0,
        variation_values={"benchmark.phases.default.concurrency": 32},
        mode="INDEPENDENT",
        phase="Succeeded",
        metrics={"request_throughput": {"avg": 225.0, "unit": "rps"}},
        child_ref=("bench-prod", "latency-sweep-v00-new", "1716060503"),
        metrics_blob=_zstd({"attempt": "new"}),
    )

    rows = await runs_index.list_sweep_variations(
        "bench-prod", "latency-sweep-5d1c", "1716060501"
    )
    assert len(rows) == 1
    assert rows[0].child_job_id == "latency-sweep-v00-new"
    assert rows[0].phase == "Succeeded"
    cur = await runs_index._conn().execute(
        "SELECT request_throughput_avg, variation_values_json FROM sweep_variations "
        "WHERE namespace = ? AND sweep_name = ? AND variation_idx = ?",
        ("bench-prod", "latency-sweep-5d1c", 0),
    )
    stored = await cur.fetchone()
    await cur.close()
    assert stored[0] == 225.0
    assert orjson.loads(runs_index.zstd_decompress(stored[1])) == {
        "benchmark.phases.default.concurrency": 32
    }


@pytest.mark.asyncio
async def test_bootstrap_sweep_child_corrupt_zstd_metrics_skips_only_bad_variation(
    tmp_path: Path,
    opened_index: Path,
) -> None:
    """One corrupt child metrics blob must not poison the whole sweep index."""
    base = tmp_path / "results"
    epoch_dir = base / "bench-prod" / "sweeps" / "token-sweep-8a4e" / "1716060601"
    epoch_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(orjson.dumps({"phase": "Succeeded"}))
    (epoch_dir / "children.json").write_bytes(
        orjson.dumps(
            {
                "children": [
                    {
                        "namespace": "bench-prod",
                        "name": "token-sweep-v00",
                        "variation_index": 0,
                        "variation_label": "output_tokens=64",
                        "child_run_epoch": "1716060602",
                    },
                    {
                        "namespace": "bench-prod",
                        "name": "token-sweep-v01",
                        "variation_index": 1,
                        "variation_label": "output_tokens=128",
                        "child_run_epoch": "1716060603",
                    },
                ]
            }
        )
    )
    _write_run_artifact(
        base,
        "bench-prod",
        "token-sweep-v00",
        "1716060602",
        summary={"metrics": _summary(throughput=180.0)},
        zstd_summary=True,
        latest=False,
    )
    _write_run_artifact(
        base,
        "bench-prod",
        "token-sweep-v01",
        "1716060603",
        corrupt_zstd=True,
        latest=False,
    )

    stats = await runs_index.bootstrap(base)

    rows = await runs_index.list_sweep_variations(
        "bench-prod", "token-sweep-8a4e", "1716060601"
    )
    assert stats.sweep_variations_indexed == 1
    assert [row.variation_idx for row in rows] == [0]
    assert rows[0].child_job_id == "token-sweep-v00"


@pytest.mark.asyncio
async def test_bootstrap_sweep_duplicate_child_refs_drop_ambiguous_child_link(
    tmp_path: Path,
    opened_index: Path,
) -> None:
    """Duplicate manifest indices cannot pick an arbitrary child_ref for the variation."""
    base = tmp_path / "results"
    epoch_dir = base / "bench-prod" / "sweeps" / "dup-ref-sweep-3f7b" / "1716060701"
    aggregate_dir = epoch_dir / "sweep_aggregate"
    aggregate_dir.mkdir(parents=True)
    (epoch_dir / "aggregate.json").write_bytes(orjson.dumps({"phase": "Succeeded"}))
    (epoch_dir / "children.json").write_bytes(
        orjson.dumps(
            {
                "children": [
                    {
                        "namespace": "bench-prod",
                        "name": "dup-ref-v00-t0",
                        "variation_index": 0,
                        "child_run_epoch": "1716060702",
                    },
                    {
                        "namespace": "bench-prod",
                        "name": "dup-ref-v00-t1",
                        "variation_index": 0,
                        "child_run_epoch": "1716060703",
                    },
                ]
            }
        )
    )
    (aggregate_dir / "profile_export_aiperf_sweep.json").write_bytes(
        orjson.dumps(
            {
                "metadata": {"sweep_mode": "INDEPENDENT"},
                "per_combination_metrics": [
                    {
                        "variation_idx": 0,
                        "variation_values": {"benchmark.phases.default.concurrency": 4},
                        "metrics": {
                            "request_throughput": {
                                "avg": 42.0,
                                "p50": 40.0,
                                "p99": 45.0,
                                "unit": "rps",
                            }
                        },
                    }
                ],
            }
        )
    )

    stats = await runs_index.bootstrap(base)

    rows = await runs_index.list_sweep_variations(
        "bench-prod", "dup-ref-sweep-3f7b", "1716060701"
    )
    assert stats.sweep_variations_indexed == 1
    assert rows[0].variation_idx == 0
    assert rows[0].child_job_id is None
    assert rows[0].child_epoch is None
