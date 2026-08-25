# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the operator ResultsDB reader facade.

Focuses on cache-never-source-of-truth reader behavior:
- read-only SQLite opens for sidecar-style query serving
- missing and corrupt indexes falling back to disk without creating writer state
- stale index rows merging with disk truth for latest and explicit epochs
- malformed summary blobs and malformed disk artifacts at the trust boundary
- DEFAULT_COMPARE_METRICS projection, sweep-directory exclusion, and query filters

Out of scope (covered elsewhere):
- runs_index write-side schema and bootstrap guardrails:
  tests/unit/operator/test_runs_index_adversarial.py
- HTTP response models and route wiring: tests/unit/operator/test_results_analytics.py
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import NoReturn

import orjson
import pytest
import pytest_asyncio
import zstandard
from pytest import param

from aiperf.common.results_markers import (
    write_processing_marker,
    write_ready_marker,
)
from aiperf.operator import results_layout, runs_index
from aiperf.operator.results_db import DEFAULT_COMPARE_METRICS, ResultsDB

# ============================================================================
# Helpers
# ============================================================================

_EPOCH_OLD = "1716064501"
_EPOCH_NEW = "1716069901"


def _zstd(payload: dict[str, object]) -> bytes:
    """Return a zstd-compressed JSON payload matching result artifact storage."""
    return zstandard.ZstdCompressor().compress(orjson.dumps(payload))


def _metric_payload(base: float, *, unit: str = "ms") -> dict[str, object]:
    """Build one metric summary with distinct compare stats for projection tests."""
    return {
        "avg": base,
        "p50": base + 0.5,
        "p99": base + 9.9,
        "unit": unit,
    }


def _summary(
    *,
    throughput: float = 100.0,
    latency: float = 50.0,
    model_name: str = "meta-llama/Llama-3-8B",
    endpoint_url: str = "http://inference.aiperf.local:8000/v1",
    start_time: str = "2026-05-18T11:00:00Z",
) -> dict[str, object]:
    """Build a realistic profile summary containing all default compare metrics."""
    return {
        "start_time": start_time,
        "end_time": "2026-05-18T11:05:00Z",
        "request_throughput": _metric_payload(throughput, unit="req/s"),
        "request_latency": _metric_payload(latency),
        "time_to_first_token": _metric_payload(latency / 2),
        "output_token_throughput": _metric_payload(throughput * 10, unit="tok/s"),
        "output_token_throughput_per_user": _metric_payload(
            throughput / 10,
            unit="tok/s/user",
        ),
        "inter_token_latency": _metric_payload(1.25),
        "telemetry_data": {
            "endpoints": {
                "http://inference.aiperf.local:8000": {
                    "gpus": {"0": {"gpu_name": "NVIDIA H100 80GB HBM3"}},
                },
            },
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
    latest: bool = True,
    compressed: bool = False,
) -> Path:
    """Write one completed run directory and optionally point latest.txt at it."""
    run_dir = results_layout.run_dir(base, namespace, job_id, epoch)
    run_dir.mkdir(parents=True)
    payload = summary or _summary()
    if compressed:
        (run_dir / "profile_export_aiperf.json.zst").write_bytes(_zstd(payload))
    else:
        (run_dir / "profile_export_aiperf.json").write_bytes(orjson.dumps(payload))
    (run_dir / runs_index.READY_MARKER).write_bytes(b"{}")
    if latest:
        results_layout.write_latest(base, namespace, job_id, epoch)
    return run_dir


def _write_malformed_run_artifact(
    base: Path,
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    latest: bool = True,
) -> None:
    """Write a run directory whose summary JSON is malformed at the disk boundary."""
    run_dir = results_layout.run_dir(base, namespace, job_id, epoch)
    run_dir.mkdir(parents=True)
    (run_dir / "profile_export_aiperf.json").write_bytes(b'{"request_throughput":')
    (run_dir / runs_index.READY_MARKER).write_bytes(b"{}")
    if latest:
        results_layout.write_latest(base, namespace, job_id, epoch)


async def _write_index_run(
    namespace: str,
    job_id: str,
    epoch: str,
    *,
    summary: dict[str, object] | None = None,
    latest: bool = True,
) -> None:
    """Insert one completed run into the open runs_index connection."""
    payload = summary or _summary()
    await runs_index.upsert_run_created(
        namespace,
        job_id,
        epoch,
        spec={"benchmark": payload["input_config"]},
    )
    await runs_index.upsert_run_completed(
        namespace,
        job_id,
        epoch,
        summary_blob=_zstd(payload),
        metrics=payload,
        files=["profile_export_aiperf.json", runs_index.READY_MARKER],
        mtime_epoch=int(epoch),
        start_time=(
            payload.get("start_time")
            if isinstance(payload.get("start_time"), str)
            else None
        ),
        end_time=(
            payload.get("end_time")
            if isinstance(payload.get("end_time"), str)
            else None
        ),
    )
    if latest:
        await runs_index.set_latest(namespace, job_id, epoch)


async def _open_writable_index(path: Path) -> None:
    """Reset and open runs_index at path so tests never inherit singleton state."""
    await runs_index.close()
    await runs_index.open(path)


@pytest_asyncio.fixture(autouse=True)
async def _close_runs_index() -> AsyncGenerator[None, None]:
    """Close the module-global runs_index connection around every ResultsDB test."""
    await runs_index.close()
    try:
        yield
    finally:
        await runs_index.close()


# ============================================================================
# Read-only open and corrupt-index fallback
# ============================================================================


class TestResultsDBReadonlyAndCorruptIndex:
    """The SQLite cache accelerates reads but must never be the only source of truth."""

    @pytest.mark.asyncio
    async def test_leaderboard_existing_index_opens_readonly_and_returns_rows(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        db_path = base / ".aiperf_index.sqlite"
        base.mkdir()
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-readonly-bench-7f2a",
            _EPOCH_NEW,
            summary=_summary(throughput=231.5),
        )
        await _open_writable_index(db_path)
        await _write_index_run(
            "bench-prod",
            "llama-readonly-bench-7f2a",
            _EPOCH_NEW,
            summary=_summary(throughput=231.5),
        )
        await runs_index.close()

        rows = await ResultsDB(base).leaderboard(
            metric="request_throughput", stat="avg"
        )

        assert runs_index.is_readonly() is True
        assert rows[0]["job_id"] == "llama-readonly-bench-7f2a"
        assert rows[0]["value"] == 231.5

    @pytest.mark.asyncio
    async def test_leaderboard_missing_index_uses_disk_without_creating_sqlite(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-disk-only-bench-9c3a",
            _EPOCH_NEW,
            summary=_summary(throughput=119.0),
        )

        rows = await ResultsDB(base).leaderboard(
            metric="request_throughput", stat="avg"
        )

        assert rows[0]["job_id"] == "llama-disk-only-bench-9c3a"
        assert rows[0]["value"] == 119.0
        assert (base / ".aiperf_index.sqlite").exists() is False
        assert runs_index.is_open() is False

    @pytest.mark.asyncio
    async def test_leaderboard_disk_fallback_excludes_processing_summary_without_ready_marker(
        self,
        tmp_path: Path,
    ) -> None:
        """A partial export must not appear in analytics before finalization."""
        base = tmp_path / "results"
        run = _write_run_artifact(
            base,
            "bench-prod",
            "llama-processing-bench-8c2f",
            _EPOCH_NEW,
            summary=_summary(throughput=119.0),
        )
        write_processing_marker(run)

        rows = await ResultsDB(base).leaderboard(
            metric="request_throughput", stat="avg"
        )

        assert rows == []

    @pytest.mark.asyncio
    async def test_history_corrupt_index_file_uses_disk_and_logs_readonly_failure(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base = tmp_path / "results"
        base.mkdir()
        (base / ".aiperf_index.sqlite").write_bytes(b"not-a-sqlite-database")
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-corrupt-index-bench-2d8e",
            _EPOCH_NEW,
            summary=_summary(throughput=77.0),
        )

        caplog.set_level("DEBUG", logger="aiperf.operator.results_db")

        rows = await ResultsDB(base).history(metric="request_throughput", stat="avg")

        assert rows[0]["job_id"] == "llama-corrupt-index-bench-2d8e"
        assert rows[0]["value"] == 77.0
        assert "runs_index read-only open unavailable" in caplog.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_name,kwargs,expected_key,expected_value",
        [
            param(
                "leaderboard",
                {"metric": "request_throughput", "stat": "avg"},
                "value",
                231.5,
                id="leaderboard",
            ),
            param(
                "history",
                {"metric": "request_throughput", "stat": "avg"},
                "value",
                231.5,
                id="history",
            ),
            param(
                "compare",
                {
                    "job_ids": ["llama-index-fast-path-bench-7f2a"],
                    "metrics": ["request_throughput"],
                },
                "request_throughput_avg",
                231.5,
                id="compare",
            ),
            param(
                "index_entries",
                {},
                "job_id",
                "llama-index-fast-path-bench-7f2a",
                id="index-entries",
            ),
        ],
    )  # fmt: skip
    async def test_populated_current_index_analytics_avoids_disk_summary_walk(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        method_name: str,
        kwargs: dict[str, object],
        expected_key: str,
        expected_value: object,
    ) -> None:
        """A healthy index must answer polling routes without scanning summaries."""
        base = tmp_path / "results"
        db_path = base / ".aiperf_index.sqlite"
        base.mkdir()
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-index-fast-path-bench-7f2a",
            _EPOCH_NEW,
            summary=_summary(throughput=231.5),
            compressed=True,
        )
        await _open_writable_index(db_path)
        await _write_index_run(
            "bench-prod",
            "llama-index-fast-path-bench-7f2a",
            _EPOCH_NEW,
            summary=_summary(throughput=231.5),
        )
        runs_index.mark_catalog_complete(base)

        def fail_disk_walk(self: ResultsDB, epoch: str | None) -> NoReturn:
            raise AssertionError(f"unexpected disk summary walk for epoch={epoch}")

        monkeypatch.setattr(ResultsDB, "_iter_disk_summaries", fail_disk_walk)

        rows = await getattr(ResultsDB(base), method_name)(**kwargs)

        assert len(rows) == 1
        assert rows[0][expected_key] == expected_value

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            param(
                "leaderboard",
                {"metric": "request_throughput", "stat": "avg"},
                id="leaderboard",
            ),
            param(
                "history",
                {"metric": "request_throughput", "stat": "avg"},
                id="history",
            ),
            param(
                "compare",
                {
                    "job_ids": ["llama-partial-index-bench-7f2a"],
                    "metrics": ["request_throughput"],
                },
                id="compare",
            ),
            param("index_entries", {}, id="index-entries"),
        ],
    )  # fmt: skip
    async def test_nonempty_incomplete_index_preserves_disk_only_ready_job(
        self,
        tmp_path: Path,
        method_name: str,
        kwargs: dict[str, object],
    ) -> None:
        """Rows returned by an index query do not prove that its catalog is complete."""
        base = tmp_path / "results"
        db_path = base / ".aiperf_index.sqlite"
        base.mkdir()
        for namespace, throughput in (("bench-a", 231.5), ("bench-b", 119.0)):
            _write_run_artifact(
                base,
                namespace,
                "llama-partial-index-bench-7f2a",
                _EPOCH_NEW,
                summary=_summary(throughput=throughput),
            )
        await _open_writable_index(db_path)
        await _write_index_run(
            "bench-a",
            "llama-partial-index-bench-7f2a",
            _EPOCH_NEW,
            summary=_summary(throughput=231.5),
        )

        rows = await getattr(ResultsDB(base), method_name)(**kwargs)

        assert {row["namespace"] for row in rows} == {"bench-a", "bench-b"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            param("leaderboard", {}, id="leaderboard"),
            param("history", {"model": "no-such-model"}, id="history"),
            param(
                "compare",
                {"job_ids": ["no-such-job"]},
                id="compare",
            ),
            param("index_entries", {}, id="index-entries"),
        ],
    )  # fmt: skip
    async def test_proven_complete_empty_index_query_avoids_disk_summary_walk(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        method_name: str,
        kwargs: dict[str, object],
    ) -> None:
        """A successful empty query is authoritative for a proven catalog."""
        base = tmp_path / "results"
        base.mkdir()
        await _open_writable_index(base / ".aiperf_index.sqlite")
        runs_index.mark_catalog_complete(base)

        def fail_disk_walk(self: ResultsDB, epoch: str | None) -> NoReturn:
            raise AssertionError(f"unexpected disk summary walk for epoch={epoch}")

        monkeypatch.setattr(ResultsDB, "_iter_disk_summaries", fail_disk_walk)

        rows = await getattr(ResultsDB(base), method_name)(**kwargs)

        assert rows == []

    @pytest.mark.asyncio
    async def test_unsupported_index_metric_still_uses_disk_summary(
        self,
        tmp_path: Path,
    ) -> None:
        """An empty SQLite result for a non-indexed metric is not authoritative."""
        base = tmp_path / "results"
        summary = _summary()
        summary["custom_quality"] = _metric_payload(0.91, unit="score")
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-custom-metric-7f2a",
            _EPOCH_NEW,
            summary=summary,
        )
        await _open_writable_index(base / ".aiperf_index.sqlite")
        await _write_index_run(
            "bench-prod",
            "llama-custom-metric-7f2a",
            _EPOCH_NEW,
            summary=summary,
        )
        runs_index.mark_catalog_complete(base)

        rows = await ResultsDB(base).leaderboard(metric="custom_quality", stat="avg")

        assert rows[0]["value"] == 0.91


# ============================================================================
# Summary fallback and malformed payloads
# ============================================================================


class TestResultsDBSummaryFallbacks:
    """Summary reads should prefer valid index blobs but preserve disk truth on misses."""

    @pytest.mark.asyncio
    async def test_summary_index_blob_missing_falls_back_to_compressed_disk_summary(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        db_path = base / ".aiperf_index.sqlite"
        base.mkdir()
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-null-blob-bench-4b1c",
            _EPOCH_NEW,
            summary=_summary(throughput=184.0),
            compressed=True,
        )
        await _open_writable_index(db_path)
        await runs_index.upsert_run_created(
            "bench-prod",
            "llama-null-blob-bench-4b1c",
            _EPOCH_NEW,
            spec={"benchmark": _summary()["input_config"]},
        )
        await runs_index.set_latest(
            "bench-prod",
            "llama-null-blob-bench-4b1c",
            _EPOCH_NEW,
        )

        summary = await ResultsDB(base).summary(
            "bench-prod", "llama-null-blob-bench-4b1c"
        )

        assert summary is not None
        assert summary["request_throughput"]["avg"] == 184.0

    @pytest.mark.asyncio
    async def test_summary_fallback_resolves_custom_artifact_prefix(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        run = results_layout.run_dir(base, "bench-prod", "custom-prefix", _EPOCH_NEW)
        run.mkdir(parents=True)
        (run / "job_spec.json").write_bytes(
            orjson.dumps({"benchmark": {"artifacts": {"prefix": "nightly"}}})
        )
        (run / "nightly.json").write_bytes(orjson.dumps(_summary(throughput=213.0)))
        write_ready_marker(run)
        results_layout.write_latest(base, "bench-prod", "custom-prefix", _EPOCH_NEW)

        summary = await ResultsDB(base).summary("bench-prod", "custom-prefix")

        assert summary is not None
        assert summary["request_throughput"]["avg"] == 213.0

    @pytest.mark.asyncio
    async def test_summary_malformed_index_blob_falls_back_to_disk_summary(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        db_path = base / ".aiperf_index.sqlite"
        base.mkdir()
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-corrupt-blob-bench-6e4a",
            _EPOCH_NEW,
            summary=_summary(throughput=201.0),
        )
        await _open_writable_index(db_path)
        await _write_index_run(
            "bench-prod",
            "llama-corrupt-blob-bench-6e4a",
            _EPOCH_NEW,
            summary=_summary(throughput=1.0),
        )
        await runs_index._conn().execute(
            "UPDATE runs SET metrics_json = ? "
            "WHERE namespace = ? AND job_id = ? AND epoch = ?",
            (
                b"not-a-zstd-frame",
                "bench-prod",
                "llama-corrupt-blob-bench-6e4a",
                _EPOCH_NEW,
            ),
        )

        summary = await ResultsDB(base).summary(
            "bench-prod", "llama-corrupt-blob-bench-6e4a"
        )

        assert summary is not None
        assert summary["request_throughput"]["avg"] == 201.0

    @pytest.mark.asyncio
    async def test_leaderboard_malformed_disk_summary_skips_bad_run_and_keeps_sibling(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        base = tmp_path / "results"
        _write_malformed_run_artifact(
            base,
            "bench-prod",
            "llama-malformed-summary-bench-5a7d",
            _EPOCH_OLD,
        )
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-good-summary-bench-5a7d",
            _EPOCH_NEW,
            summary=_summary(throughput=166.0),
        )

        rows = await ResultsDB(base).leaderboard(
            metric="request_throughput", stat="avg"
        )

        assert [row["job_id"] for row in rows] == ["llama-good-summary-bench-5a7d"]
        assert "cannot read summary" in caplog.text


# ============================================================================
# Compare columns, sweep shape, and query filters
# ============================================================================


class TestResultsDBCompareAndFilters:
    """Compare/history reader contracts for metrics, sweeps, and user filters."""

    @pytest.mark.asyncio
    async def test_compare_default_metrics_projects_all_default_compare_columns(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-default-metrics-bench-1f9e",
            _EPOCH_NEW,
            summary=_summary(throughput=144.0),
        )

        rows = await ResultsDB(base).compare(
            job_ids=["llama-default-metrics-bench-1f9e"]
        )

        assert len(rows) == 1
        row = rows[0]
        for metric in DEFAULT_COMPARE_METRICS:
            assert f"{metric}_avg" in row
            assert f"{metric}_p50" in row
            assert f"{metric}_p99" in row
            assert f"{metric}_unit" in row
        assert row["request_throughput_avg"] == 144.0
        assert row["gpu_count"] == 1
        assert row["gpu_name"] == "NVIDIA H100 80GB HBM3"

    @pytest.mark.asyncio
    async def test_index_entries_sweep_directory_is_not_reported_as_benchmark_run(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        _write_run_artifact(
            base,
            "bench-prod",
            "saturation-sweep-v00-t0",
            _EPOCH_NEW,
            summary=_summary(throughput=133.0),
        )
        sweep_epoch_dir = (
            base / "bench-prod" / "sweeps" / "saturation-sweep" / _EPOCH_NEW
        )
        sweep_epoch_dir.mkdir(parents=True)
        (sweep_epoch_dir / "aggregate.json").write_bytes(
            orjson.dumps({"metadata": {"mode": "INDEPENDENT"}})
        )
        results_layout.write_sweep_latest(
            base,
            "bench-prod",
            "saturation-sweep",
            _EPOCH_NEW,
        )

        rows = await ResultsDB(base).index_entries()

        assert {(row["namespace"], row["job_id"]) for row in rows} == {
            ("bench-prod", "saturation-sweep-v00-t0")
        }

    @pytest.mark.asyncio
    async def test_history_model_and_endpoint_filters_match_substrings_only(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-prod-bench-8d2b",
            _EPOCH_NEW,
            summary=_summary(
                throughput=240.0,
                model_name="meta-llama/Llama-3-70B-Instruct",
                endpoint_url="http://prod-inference.aiperf.local:8000/v1",
                start_time="2026-05-18T12:00:00Z",
            ),
        )
        _write_run_artifact(
            base,
            "bench-stage",
            "mistral-stage-bench-8d2b",
            _EPOCH_NEW,
            summary=_summary(
                throughput=91.0,
                model_name="mistralai/Mistral-7B-Instruct-v0.3",
                endpoint_url="http://stage-inference.aiperf.local:8000/v1",
                start_time="2026-05-18T12:01:00Z",
            ),
        )

        rows = await ResultsDB(base).history(
            model="Llama-3",
            endpoint="prod-inference",
            metric="request_throughput",
            stat="avg",
        )

        assert [(row["namespace"], row["job_id"], row["value"]) for row in rows] == [
            ("bench-prod", "llama-prod-bench-8d2b", 240.0)
        ]

    @pytest.mark.asyncio
    async def test_compare_qualified_job_ids_disambiguate_same_name_across_namespaces(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        for namespace, throughput in (
            ("bench-prod", 210.0),
            ("bench-stage", 99.0),
        ):
            _write_run_artifact(
                base,
                namespace,
                "shared-name-bench-3c1f",
                _EPOCH_NEW,
                summary=_summary(throughput=throughput),
            )

        rows = await ResultsDB(base).compare(
            job_ids=["bench-prod/shared-name-bench-3c1f"],
            metrics=["request_throughput"],
        )

        assert [(row["namespace"], row["request_throughput_avg"]) for row in rows] == [
            ("bench-prod", 210.0)
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "method_name,kwargs",
        [
            param(
                "leaderboard",
                {"metric": "request_throughput;drop", "stat": "avg"},
                id="leaderboard-invalid-metric",
            ),
            param(
                "leaderboard",
                {"metric": "request_throughput", "stat": "avg desc"},
                id="leaderboard-invalid-stat",
            ),
            param(
                "history",
                {"metric": "request-throughput", "stat": "avg"},
                id="history-invalid-metric",
            ),
            param(
                "compare",
                {
                    "job_ids": ["llama-prod-bench-8d2b"],
                    "metrics": ["request_throughput)"],
                },
                id="compare-invalid-metric",
            ),
        ],
    )  # fmt: skip
    async def test_query_invalid_identifier_returns_empty_rows(
        self,
        tmp_path: Path,
        method_name: str,
        kwargs: dict[str, object],
    ) -> None:
        base = tmp_path / "results"
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-prod-bench-8d2b",
            _EPOCH_NEW,
            summary=_summary(throughput=240.0),
        )
        method = getattr(ResultsDB(base), method_name)

        rows = await method(**kwargs)

        assert rows == []

    @pytest.mark.asyncio
    async def test_leaderboard_explicit_epoch_keeps_old_and_latest_rows_distinct(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-epoch-filter-bench-7b6d",
            _EPOCH_OLD,
            summary=_summary(throughput=88.0),
            latest=False,
        )
        _write_run_artifact(
            base,
            "bench-prod",
            "llama-epoch-filter-bench-7b6d",
            _EPOCH_NEW,
            summary=_summary(throughput=188.0),
        )

        old_rows = await ResultsDB(base).leaderboard(
            metric="request_throughput",
            stat="avg",
            epoch=_EPOCH_OLD,
        )
        latest_rows = await ResultsDB(base).leaderboard(
            metric="request_throughput",
            stat="avg",
        )

        assert [(row["epoch"], row["value"]) for row in old_rows] == [
            (_EPOCH_OLD, 88.0)
        ]
        assert [(row["epoch"], row["value"]) for row in latest_rows] == [
            (_EPOCH_NEW, 188.0)
        ]


# ============================================================================
# Sweep variation rows through the shared read cache
# ============================================================================


class TestResultsDBSweepVariationAdjacency:
    """ResultsDB shares the same SQLite file as sweep variation readers."""

    @pytest.mark.asyncio
    async def test_readonly_resultsdb_does_not_block_sweep_variation_reads(
        self,
        tmp_path: Path,
    ) -> None:
        base = tmp_path / "results"
        db_path = base / ".aiperf_index.sqlite"
        base.mkdir()
        _write_run_artifact(
            base,
            "bench-prod",
            "saturation-sweep-v00-t0",
            _EPOCH_NEW,
            summary=_summary(throughput=133.0),
        )
        await _open_writable_index(db_path)
        await _write_index_run(
            "bench-prod",
            "saturation-sweep-v00-t0",
            _EPOCH_NEW,
            summary=_summary(throughput=133.0),
        )
        await runs_index.upsert_sweep_variation(
            "bench-prod",
            "saturation-sweep",
            _EPOCH_NEW,
            0,
            variation_values={"request_rate": 1200, "trial_index": 0},
            mode="INDEPENDENT",
            phase="Succeeded",
            metrics={"request_throughput": _metric_payload(133.0, unit="req/s")},
            child_ref=("bench-prod", "saturation-sweep-v00-t0", _EPOCH_NEW),
            metrics_blob=_zstd({"variation_idx": 0}),
        )
        await runs_index.close()

        index_rows = await ResultsDB(base).index_entries()
        variations = await runs_index.list_sweep_variations(
            "bench-prod",
            "saturation-sweep",
            _EPOCH_NEW,
        )

        assert runs_index.is_readonly() is True
        assert [row["job_id"] for row in index_rows] == ["saturation-sweep-v00-t0"]
        assert [(row.variation_idx, row.child_job_id) for row in variations] == [
            (0, "saturation-sweep-v00-t0")
        ]
