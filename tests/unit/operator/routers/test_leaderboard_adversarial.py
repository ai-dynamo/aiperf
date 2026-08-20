# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for leaderboard and summary analytics tolerance.

Focuses on:
- malformed leaderboard metric payloads at the profile-export trust boundary
- invalid metric identifiers with and without an open runs_index cache
- duplicate AIPerfJob names across namespaces staying namespace-qualified
- stale runs_index rows falling back to disk-authoritative summaries
- missing summary files returning stable empty/404 contracts
- no aggregation across unrelated runs that share model and endpoint metadata

Out of scope: compare pivot semantics and UI leaderboard rendering, covered by
``test_compare_adversarial.py`` and ``test_operator_leaderboard_edges.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import orjson
import pytest
from aiperf.operator.results_db import ResultsDB
from aiperf.operator.results_layout import run_dir, write_latest
from aiperf.operator.routers.results_analytics import create_results_analytics_router
from fastapi import FastAPI
from pytest import param

from aiperf.kubernetes.results_sidecar import write_ready_marker
from aiperf.operator import runs_index

# ============================================================
# Helpers
# ============================================================

_EPOCH_OLD = "1714064523"
_EPOCH_NEW = "1714150923"


def _metric_payload(
    *, avg: object = 100.0, p50: object | None = None, p99: object | None = None
) -> dict[str, object]:
    """Build one metric block in the shape emitted by profile exports."""
    return {
        "avg": avg,
        "p50": p50 if p50 is not None else 90.0,
        "p99": p99 if p99 is not None else 150.0,
        "unit": "req/s",
    }


def _summary_payload(
    *,
    throughput: object = 100.0,
    model: str = "meta-llama/Llama-3-8B",
    endpoint: str = "http://llama3.svc.cluster.local:8000/v1",
    start_time: str = "2026-04-21T09:00:00Z",
) -> dict[str, object]:
    """Build a profile export with leaderboard-visible metadata."""
    return {
        "request_throughput": _metric_payload(avg=throughput),
        "request_latency": {
            "avg": 50.0,
            "p50": 45.0,
            "p99": 120.0,
            "unit": "ms",
        },
        "start_time": start_time,
        "end_time": "2026-04-21T09:05:00Z",
        "input_config": {
            "models": {"items": [{"name": model}]},
            "endpoint": {"urls": [endpoint]},
        },
    }


def _seed_summary_run(
    base_dir: Path,
    *,
    namespace: str = "bench-prod",
    job_id: str = "llama-3-8b-load",
    epoch: str = _EPOCH_NEW,
    payload: dict[str, object] | None = None,
    write_latest_pointer: bool = True,
) -> Path:
    """Create one PVC-style run directory with a profile-export summary."""
    target = run_dir(base_dir, namespace, job_id, epoch)
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("profile_export_aiperf.json").write_bytes(
        orjson.dumps(payload or _summary_payload())
    )
    write_ready_marker(target)
    if write_latest_pointer:
        write_latest(base_dir, namespace, job_id, epoch)
    return target


def _seed_summaryless_run(
    base_dir: Path,
    *,
    namespace: str = "bench-prod",
    job_id: str = "summary-not-ready-load",
    epoch: str = _EPOCH_NEW,
) -> Path:
    """Create a latest run directory before the summary export lands."""
    target = run_dir(base_dir, namespace, job_id, epoch)
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("worker_stdout.log").write_text("controller still processing\n")
    write_latest(base_dir, namespace, job_id, epoch)
    return target


def _analytics_app(results_dir: Path) -> FastAPI:
    """Build only the analytics router against a real ResultsDB facade."""
    db = ResultsDB(results_dir)
    app = FastAPI()
    app.include_router(create_results_analytics_router(lambda: db, results_dir, [None]))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by an isolated operator results directory."""
    await runs_index.close()
    transport = httpx.ASGITransport(
        app=_analytics_app(tmp_path), raise_app_exceptions=False
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf-operator.local"
    ) as c:
        yield c
    await runs_index.close()


async def _write_index_run(
    base_dir: Path,
    namespace: str,
    job_id: str,
    *,
    epoch: str,
    payload: dict[str, object],
    create_disk_run: bool,
    is_latest: bool = True,
) -> None:
    """Insert one runs_index row, optionally backed by a matching disk run."""
    if create_disk_run:
        _seed_summary_run(
            base_dir,
            namespace=namespace,
            job_id=job_id,
            epoch=epoch,
            payload=payload,
            write_latest_pointer=is_latest,
        )
    await runs_index.upsert_run_created(
        namespace,
        job_id,
        epoch,
        spec={"benchmark": payload.get("input_config", {})},
    )
    await runs_index.upsert_run_completed(
        namespace,
        job_id,
        epoch,
        summary_blob=runs_index._zstd_compress(payload),
        metrics=payload,
        files=["profile_export_aiperf.json"],
        mtime_epoch=int(epoch),
        start_time=str(payload.get("start_time")),
        end_time=str(payload.get("end_time")),
    )
    if is_latest:
        await runs_index.set_latest(namespace, job_id, epoch)


# ============================================================
# Malformed metric payloads
# ============================================================


class TestLeaderboardMalformedMetricPayloads:
    """Bad profile-export metric cells are skipped without poisoning good runs."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_metric",
        [
            param({"avg": "fast", "unit": "req/s"}, id="string-stat"),
            param({"avg": {"nested": 42}, "unit": "req/s"}, id="object-stat"),
            param({"avg": [101.0], "unit": "req/s"}, id="list-stat"),
        ],
    )  # fmt: skip
    async def test_leaderboard_malformed_metric_stat_skips_bad_run_and_keeps_good_run(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        bad_metric: dict[str, object],
    ) -> None:
        _seed_summary_run(
            tmp_path,
            job_id="bad-metric-load",
            payload={**_summary_payload(), "request_throughput": bad_metric},
        )
        _seed_summary_run(
            tmp_path,
            job_id="healthy-h100-load",
            payload=_summary_payload(throughput=321.0),
        )

        response = await client.get("/api/v1/analytics/leaderboard")

        assert response.status_code == 200
        body = response.json()
        assert [entry["job_id"] for entry in body["entries"]] == ["healthy-h100-load"]
        assert body["entries"][0]["value"] == 321.0

    @pytest.mark.asyncio
    async def test_leaderboard_null_metric_stat_is_missing_value_not_zero(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            job_id="null-throughput-load",
            payload={
                **_summary_payload(),
                "request_throughput": _metric_payload(avg=None),
            },
        )

        response = await client.get("/api/v1/analytics/leaderboard")

        assert response.status_code == 200
        assert response.json()["entries"] == []


# ============================================================
# Query trust boundary
# ============================================================


class TestLeaderboardInvalidMetricNames:
    """Metric identifiers fail closed whether analytics is disk-backed or index-backed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metric_identifier",
        [
            "request_throughput;drop",
            param("request-throughput", id="hyphen-rejected"),
            param("request throughput", id="space-rejected"),
        ],
    )  # fmt: skip
    async def test_leaderboard_invalid_metric_identifier_returns_empty_entries_not_500(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        metric_identifier: str,
    ) -> None:
        _seed_summary_run(tmp_path)

        response = await client.get(
            "/api/v1/analytics/leaderboard", params={"metric": metric_identifier}
        )

        assert response.status_code == 200
        assert response.json()["entries"] == []

    @pytest.mark.asyncio
    async def test_leaderboard_invalid_metric_identifier_with_open_index_returns_empty_not_500(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        await runs_index.open(tmp_path / ".aiperf_index.sqlite")
        await _write_index_run(
            tmp_path,
            "bench-prod",
            "indexed-h100-load",
            epoch=_EPOCH_NEW,
            payload=_summary_payload(throughput=777.0),
            create_disk_run=True,
        )

        response = await client.get(
            "/api/v1/analytics/leaderboard?metric=request_throughput%3Bdrop"
        )

        assert response.status_code == 200
        assert response.json()["entries"] == []


# ============================================================
# Namespace identity and aggregation guards
# ============================================================


class TestLeaderboardNamespaceAndAggregationGuards:
    """Leaderboard rows are per run identity, not grouped by display attributes."""

    @pytest.mark.asyncio
    async def test_leaderboard_duplicate_job_names_across_namespaces_keeps_both_rows(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            namespace="bench-prod",
            job_id="shared-load",
            payload=_summary_payload(throughput=100.0),
        )
        _seed_summary_run(
            tmp_path,
            namespace="bench-canary",
            job_id="shared-load",
            payload=_summary_payload(throughput=200.0),
        )

        response = await client.get("/api/v1/analytics/leaderboard")

        assert response.status_code == 200
        assert [
            (entry["namespace"], entry["job_id"], entry["value"])
            for entry in response.json()["entries"]
        ] == [
            ("bench-canary", "shared-load", 200.0),
            ("bench-prod", "shared-load", 100.0),
        ]

    @pytest.mark.asyncio
    async def test_leaderboard_related_runs_with_same_model_endpoint_are_not_aggregated(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        model = "meta-llama/Llama-3.1-70B-Instruct"
        endpoint = "http://prod-router.svc.cluster.local:8000/v1"
        for job_id, throughput in (
            ("llama70b-baseline-load", 10.0),
            ("llama70b-tuned-load", 20.0),
            ("llama70b-canary-load", 30.0),
        ):
            _seed_summary_run(
                tmp_path,
                job_id=job_id,
                payload=_summary_payload(
                    throughput=throughput, model=model, endpoint=endpoint
                ),
            )

        response = await client.get("/api/v1/analytics/leaderboard")

        assert response.status_code == 200
        assert [
            (entry["job_id"], entry["value"]) for entry in response.json()["entries"]
        ] == [
            ("llama70b-canary-load", 30.0),
            ("llama70b-tuned-load", 20.0),
            ("llama70b-baseline-load", 10.0),
        ]


# ============================================================
# Stale index and missing summary fallback
# ============================================================


class TestLeaderboardSummaryFallbacks:
    """The SQLite cache is advisory; summary responses stay disk-authoritative."""

    @pytest.mark.asyncio
    async def test_leaderboard_stale_index_row_missing_run_dir_returns_disk_run_only(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        await runs_index.open(tmp_path / ".aiperf_index.sqlite")
        await _write_index_run(
            tmp_path,
            "bench-prod",
            "deleted-index-load",
            epoch=_EPOCH_OLD,
            payload=_summary_payload(throughput=999.0),
            create_disk_run=False,
        )
        _seed_summary_run(
            tmp_path,
            namespace="bench-prod",
            job_id="disk-authoritative-load",
            payload=_summary_payload(throughput=321.0),
        )

        response = await client.get("/api/v1/analytics/leaderboard")

        assert response.status_code == 200
        assert [entry["job_id"] for entry in response.json()["entries"]] == [
            "disk-authoritative-load"
        ]

    @pytest.mark.asyncio
    async def test_summary_stale_index_latest_missing_run_dir_uses_disk_latest_summary(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        await runs_index.open(tmp_path / ".aiperf_index.sqlite")
        await _write_index_run(
            tmp_path,
            "bench-prod",
            "resubmitted-load",
            epoch=_EPOCH_OLD,
            payload=_summary_payload(throughput=11.0),
            create_disk_run=False,
        )
        _seed_summary_run(
            tmp_path,
            namespace="bench-prod",
            job_id="resubmitted-load",
            epoch=_EPOCH_NEW,
            payload=_summary_payload(throughput=222.0),
        )

        response = await client.get(
            "/api/v1/analytics/summary/bench-prod/resubmitted-load"
        )

        assert response.status_code == 200
        assert response.json()["request_throughput"]["avg"] == 222.0

    @pytest.mark.asyncio
    async def test_leaderboard_and_summary_missing_export_file_return_stable_contracts(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summaryless_run(tmp_path)

        leaderboard_response = await client.get("/api/v1/analytics/leaderboard")
        summary_response = await client.get(
            "/api/v1/analytics/summary/bench-prod/summary-not-ready-load"
        )

        assert leaderboard_response.status_code == 200
        assert leaderboard_response.json()["entries"] == []
        assert summary_response.status_code == 404
        assert summary_response.json()["detail"] == (
            "No summary for bench-prod/summary-not-ready-load"
        )
