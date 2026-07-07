# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator analytics/dashboard API routers.

Focuses on:
- empty and malformed result trees returning stable API shapes
- compare ambiguity, qualified filters, and invalid query parameters
- namespace/model/endpoint filtering across latest-run summaries
- stale runs_index rows falling back to the PVC artifact tree
- path and query encoding at the FastAPI route boundary

Out of scope: file download gating and dashboard sidecar proxying, covered by
``test_results_files_adversarial.py`` and ``test_dashboard_proxy_adversarial.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import orjson
import pytest
from fastapi import FastAPI
from pytest import param

from aiperf.operator import runs_index
from aiperf.operator.results_db import ResultsDB
from aiperf.operator.results_layout import run_dir, write_latest
from aiperf.operator.routers.results_analytics import create_results_analytics_router

# ============================================================
# Helpers
# ============================================================

_EPOCH_OLD = "1714064523"
_EPOCH_NEW = "1714150923"


def _summary_payload(
    *,
    metric_value: float = 100.0,
    model: str = "meta-llama/Llama-3-8B",
    endpoint: str = "http://llama3.svc.cluster.local:8000/v1",
    start_time: str = "2026-04-21T09:00:00Z",
) -> dict[str, object]:
    """Build a profile export with the fields analytics endpoints surface."""
    return {
        "request_throughput": {
            "avg": metric_value,
            "p50": metric_value * 0.9,
            "p99": metric_value * 1.5,
            "unit": "req/s",
        },
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
        "telemetry_data": {
            "endpoints": {
                "gpu-node-a": {
                    "gpus": {
                        "0": {"gpu_name": "NVIDIA H100 80GB HBM3"},
                        "1": {"gpu_name": "NVIDIA H100 80GB HBM3"},
                    }
                }
            }
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
    """Create one run directory with ``profile_export_aiperf.json``."""
    target = run_dir(base_dir, namespace, job_id, epoch)
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("profile_export_aiperf.json").write_bytes(
        orjson.dumps(payload or _summary_payload())
    )
    if write_latest_pointer:
        write_latest(base_dir, namespace, job_id, epoch)
    return target


def _seed_malformed_summary_run(
    base_dir: Path,
    *,
    namespace: str = "bench-prod",
    job_id: str = "broken-export",
    epoch: str = _EPOCH_NEW,
) -> Path:
    """Create a run whose summary JSON cannot be parsed."""
    target = run_dir(base_dir, namespace, job_id, epoch)
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("profile_export_aiperf.json").write_bytes(b'{"request_throughput":')
    write_latest(base_dir, namespace, job_id, epoch)
    return target


def _analytics_app(results_dir: Path) -> FastAPI:
    """Build only the analytics router with a real ResultsDB facade."""
    db = ResultsDB(results_dir)
    app = FastAPI()
    app.include_router(create_results_analytics_router(lambda: db, results_dir, [None]))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by a temp PVC-like results directory."""
    await runs_index.close()
    transport = httpx.ASGITransport(app=_analytics_app(tmp_path))
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf-operator.local"
    ) as c:
        yield c
    await runs_index.close()


async def _write_stale_index_run(
    namespace: str,
    job_id: str,
    *,
    epoch: str = _EPOCH_OLD,
    metric_value: float = 999.0,
) -> None:
    """Insert an index row without creating its matching run directory."""
    summary = _summary_payload(metric_value=metric_value, model="stale-index-model")
    await runs_index.upsert_run_created(
        namespace,
        job_id,
        epoch,
        spec={"benchmark": summary["input_config"]},
    )
    await runs_index.upsert_run_completed(
        namespace,
        job_id,
        epoch,
        summary_blob=runs_index._zstd_compress(summary),
        metrics=summary,
        files=["profile_export_aiperf.json"],
        mtime_epoch=int(epoch),
        start_time=str(summary["start_time"]),
        end_time=str(summary["end_time"]),
    )
    await runs_index.set_latest(namespace, job_id, epoch)


# ============================================================
# Empty and malformed runs
# ============================================================


class TestAnalyticsEmptyAndMalformedRuns:
    """Empty PVCs and corrupt summaries keep returning documented JSON shapes."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path,expected_body",
        [
            param(
                "/api/v1/analytics/leaderboard",
                {
                    "metric": "request_throughput",
                    "stat": "avg",
                    "order": "desc",
                    "entries": [],
                },
                id="leaderboard-empty",
            ),
            param(
                "/api/v1/analytics/history",
                {"metric": "request_throughput", "stat": "avg", "entries": []},
                id="history-empty",
            ),
            param(
                "/api/v1/analytics/compare?jobs=llama-3-8b-load",
                {
                    "job_ids": ["llama-3-8b-load"],
                    "metrics": [
                        "request_throughput",
                        "request_latency",
                        "time_to_first_token",
                        "output_token_throughput",
                        "output_token_throughput_per_user",
                        "inter_token_latency",
                    ],
                    "entries": [],
                    "meta": {},
                },
                id="compare-empty",
            ),
            param("/api/v1/index", {}, id="index-empty"),
        ],
    )  # fmt: skip
    async def test_analytics_empty_results_dir_returns_stable_schema(
        self, client: httpx.AsyncClient, path: str, expected_body: dict[str, object]
    ) -> None:
        response = await client.get(path)

        assert response.status_code == 200
        assert response.json() == expected_body

    @pytest.mark.asyncio
    async def test_leaderboard_malformed_summary_skips_bad_run_and_keeps_good_run(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_malformed_summary_run(tmp_path)
        _seed_summary_run(
            tmp_path,
            job_id="healthy-h100-load",
            payload=_summary_payload(
                metric_value=321.0, model="meta-llama/Llama-3.1-70B"
            ),
        )

        response = await client.get("/api/v1/analytics/leaderboard")

        assert response.status_code == 200
        body = response.json()
        assert [entry["job_id"] for entry in body["entries"]] == ["healthy-h100-load"]
        assert body["entries"][0]["value"] == 321.0

    @pytest.mark.asyncio
    async def test_leaderboard_non_mapping_metric_shape_skips_run_not_crash(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            job_id="scalar-metric-load",
            payload={
                **_summary_payload(),
                "request_throughput": 42.0,
            },
        )

        response = await client.get("/api/v1/analytics/leaderboard")

        assert response.status_code == 200
        assert response.json()["entries"] == []


# ============================================================
# Query validation and compare filters
# ============================================================


class TestAnalyticsQueryAndCompareFilters:
    """Invalid query shapes fail at the HTTP boundary; valid filters are precise."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path,expected_field",
        [
            ("/api/v1/analytics/leaderboard?limit=0", "limit"),
            ("/api/v1/analytics/leaderboard?limit=1001", "limit"),
            ("/api/v1/analytics/history?limit=0", "limit"),
            ("/api/v1/analytics/history?limit=10001", "limit"),
            param("/api/v1/analytics/compare", "jobs", id="compare-missing-jobs"),
        ],
    )  # fmt: skip
    async def test_analytics_invalid_query_params_return_422_with_field_context(
        self, client: httpx.AsyncClient, path: str, expected_field: str
    ) -> None:
        response = await client.get(path)

        assert response.status_code == 422
        assert expected_field in response.text

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path,response_key",
        [
            (
                "/api/v1/analytics/leaderboard?metric=request_throughput;drop",
                "entries",
            ),
            (
                "/api/v1/analytics/history?metric=request_throughput%3Bdrop",
                "entries",
            ),
            param(
                "/api/v1/analytics/compare?jobs=llama-3-8b-load&metrics=request_throughput%3Bdrop",
                "entries",
                id="compare-invalid-metric",
            ),
        ],
    )  # fmt: skip
    async def test_analytics_invalid_metric_identifier_returns_empty_data_not_500(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        path: str,
        response_key: str,
    ) -> None:
        _seed_summary_run(tmp_path)

        response = await client.get(path)

        assert response.status_code == 200
        assert response.json()[response_key] == []

    @pytest.mark.asyncio
    async def test_compare_bare_job_name_ambiguous_across_namespaces_returns_409(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(tmp_path, namespace="bench-prod", job_id="shared-load")
        _seed_summary_run(tmp_path, namespace="bench-canary", job_id="shared-load")

        response = await client.get("/api/v1/analytics/compare?jobs=shared-load")

        assert response.status_code == 409
        assert "namespace/job syntax" in response.json()["detail"]
        assert "bench-canary/shared-load" in response.json()["detail"]
        assert "bench-prod/shared-load" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_compare_qualified_job_filter_disambiguates_namespace_and_keeps_meta_key(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            namespace="bench-prod",
            job_id="shared-load",
            payload=_summary_payload(metric_value=100.0, model="meta-llama/Llama-3-8B"),
        )
        _seed_summary_run(
            tmp_path,
            namespace="bench-canary",
            job_id="shared-load",
            payload=_summary_payload(
                metric_value=200.0, model="meta-llama/Llama-3.1-70B"
            ),
        )

        response = await client.get(
            "/api/v1/analytics/compare"
            "?jobs=bench-canary%2Fshared-load&metrics=request_throughput"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["job_ids"] == ["bench-canary/shared-load"]
        assert body["metrics"] == ["request_throughput"]
        assert body["entries"] == [
            {
                "metric": "request_throughput",
                "stat": "avg",
                "unit": "req/s",
                "values": {"bench-canary/shared-load": 200.0},
            },
            {
                "metric": "request_throughput",
                "stat": "p50",
                "unit": "req/s",
                "values": {"bench-canary/shared-load": 180.0},
            },
            {
                "metric": "request_throughput",
                "stat": "p99",
                "unit": "req/s",
                "values": {"bench-canary/shared-load": 300.0},
            },
        ]
        assert body["meta"] == {
            "bench-canary/shared-load": {
                "gpu_count": 2,
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "model": "meta-llama/Llama-3.1-70B",
                "endpoint": "http://llama3.svc.cluster.local:8000/v1",
            }
        }


# ============================================================
# Namespace/model filtering and encoding
# ============================================================


class TestAnalyticsFilteringAndEncoding:
    """Dashboard filters rely on model substrings and URL-decoded path segments."""

    @pytest.mark.asyncio
    async def test_history_model_and_endpoint_query_filters_return_only_matching_run(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            job_id="llama3-prod-load",
            payload=_summary_payload(
                metric_value=111.0,
                model="meta-llama/Llama-3.1-70B-Instruct",
                endpoint="http://prod-router.svc.cluster.local:8000/v1",
                start_time="2026-04-21T09:00:00Z",
            ),
        )
        _seed_summary_run(
            tmp_path,
            job_id="mistral-canary-load",
            payload=_summary_payload(
                metric_value=222.0,
                model="mistralai/Mistral-7B-Instruct-v0.3",
                endpoint="http://canary-router.svc.cluster.local:8000/v1",
                start_time="2026-04-21T09:10:00Z",
            ),
        )

        response = await client.get(
            "/api/v1/analytics/history"
            "?model=meta-llama%2FLlama-3.1&endpoint=prod-router.svc"
        )

        assert response.status_code == 200
        body = response.json()
        assert [entry["job_id"] for entry in body["entries"]] == ["llama3-prod-load"]
        assert body["entries"][0]["model"] == "meta-llama/Llama-3.1-70B-Instruct"
        assert body["entries"][0]["value"] == 111.0

    @pytest.mark.asyncio
    async def test_summary_url_encoded_namespace_and_job_returns_exact_payload(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        # Valid Kubernetes identifiers; the dot in the job name still exercises
        # percent-decoding while passing path validation.
        _seed_summary_run(
            tmp_path,
            namespace="team-alpha",
            job_id="llama-3.1-8b-load",
            payload=_summary_payload(
                metric_value=444.0, model="meta-llama/Llama-3.1-8B"
            ),
        )

        response = await client.get(
            "/api/v1/analytics/summary/team%2Dalpha/llama-3%2E1-8b-load"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["request_throughput"]["avg"] == 444.0
        assert body["input_config"]["models"]["items"][0]["name"] == (
            "meta-llama/Llama-3.1-8B"
        )

    @pytest.mark.asyncio
    async def test_leaderboard_epoch_query_uses_requested_historical_epoch_only(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            epoch=_EPOCH_OLD,
            payload=_summary_payload(metric_value=10.0, model="meta-llama/Llama-3-8B"),
            write_latest_pointer=False,
        )
        _seed_summary_run(
            tmp_path,
            epoch=_EPOCH_NEW,
            payload=_summary_payload(
                metric_value=99.0, model="meta-llama/Llama-3.1-8B"
            ),
        )

        response = await client.get(f"/api/v1/analytics/leaderboard?epoch={_EPOCH_OLD}")

        assert response.status_code == 200
        entries = response.json()["entries"]
        assert [(entry["epoch"], entry["value"]) for entry in entries] == [
            (_EPOCH_OLD, 10.0)
        ]


# ============================================================
# Stale index fallback and schema stability
# ============================================================


class TestAnalyticsIndexFallbackAndSchema:
    """SQLite is a cache; disk remains authoritative for dashboard reads."""

    @pytest.mark.asyncio
    async def test_index_route_stale_index_missing_run_dir_returns_disk_run(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        await runs_index.open(tmp_path / ".aiperf_index.sqlite")
        await _write_stale_index_run("bench-prod", "deleted-index-run")
        _seed_summary_run(
            tmp_path,
            namespace="bench-prod",
            job_id="disk-authoritative-run",
            payload=_summary_payload(
                metric_value=123.0, model="meta-llama/Llama-3.1-8B"
            ),
        )

        response = await client.get("/api/v1/index")

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"bench-prod/disk-authoritative-run"}
        assert body["bench-prod/disk-authoritative-run"]["model"] == (
            "meta-llama/Llama-3.1-8B"
        )

    @pytest.mark.asyncio
    async def test_leaderboard_response_entry_schema_remains_stable(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(tmp_path)

        response = await client.get("/api/v1/analytics/leaderboard")

        assert response.status_code == 200
        body = response.json()
        assert set(body) == {"metric", "stat", "order", "entries"}
        assert [set(entry) for entry in body["entries"]] == [
            {
                "namespace",
                "job_id",
                "epoch",
                "value",
                "unit",
                "start_time",
                "end_time",
                "model",
                "endpoint",
            }
        ]

    @pytest.mark.asyncio
    async def test_summary_missing_namespace_job_returns_404_with_run_identity(
        self, client: httpx.AsyncClient
    ) -> None:
        response = await client.get("/api/v1/analytics/summary/bench-prod/missing-load")

        assert response.status_code == 404
        assert response.json()["detail"] == "No summary for bench-prod/missing-load"
