# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for operator compare analytics query semantics.

Focuses on:
- namespace-qualified compare identities and duplicate job names across namespaces
- default metrics, missing metrics, and invalid metric identifiers
- preserving per-run values without aggregating unrelated benchmark runs
- model/endpoint metadata carried through compare responses
- stale runs_index rows falling back to disk-authoritative summaries
- malformed compare query parameters at the FastAPI boundary

Out of scope: result-file download authorization and dashboard proxy behavior, covered by
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

from aiperf.common.results_markers import write_ready_marker
from aiperf.operator import runs_index
from aiperf.operator.results_db import DEFAULT_COMPARE_METRICS, ResultsDB
from aiperf.operator.results_layout import run_dir, write_latest
from aiperf.operator.routers.results_analytics import create_results_analytics_router

# ============================================================
# Helpers
# ============================================================

_EPOCH_OLD = "1714064523"
_EPOCH_NEW = "1714150923"


def _metric_payload(
    *,
    avg: float = 100.0,
    p50: float | None = None,
    p99: float | None = None,
    unit: str = "req/s",
) -> dict[str, object]:
    """Build one summary metric block with realistic percentile defaults."""
    return {
        "avg": avg,
        "p50": p50 if p50 is not None else avg * 0.9,
        "p99": p99 if p99 is not None else avg * 1.5,
        "unit": unit,
    }


def _summary_payload(
    *,
    throughput: float = 100.0,
    latency: float | None = 50.0,
    ttft: float | None = None,
    model: str = "meta-llama/Llama-3-8B",
    endpoint: str = "http://llama3.svc.cluster.local:8000/v1",
    start_time: str = "2026-04-21T09:00:00Z",
) -> dict[str, object]:
    """Build a profile export with compare metrics and metadata."""
    payload: dict[str, object] = {
        "request_throughput": _metric_payload(avg=throughput),
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
    if latency is not None:
        payload["request_latency"] = _metric_payload(avg=latency, unit="ms")
    if ttft is not None:
        payload["time_to_first_token"] = _metric_payload(avg=ttft, unit="ms")
    return payload


def _seed_summary_run(
    base_dir: Path,
    *,
    namespace: str = "bench-prod",
    job_id: str = "llama-3-8b-load",
    epoch: str = _EPOCH_NEW,
    payload: dict[str, object] | None = None,
    write_latest_pointer: bool = True,
) -> Path:
    """Create one PVC-style run directory with a profile export summary."""
    target = run_dir(base_dir, namespace, job_id, epoch)
    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("profile_export_aiperf.json").write_bytes(
        orjson.dumps(payload or _summary_payload())
    )
    write_ready_marker(target)
    if write_latest_pointer:
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
# Namespace-qualified identity filters
# ============================================================


class TestCompareNamespaceQualifiedFilters:
    """Compare identities are namespace-aware, not just display-name filters."""

    @pytest.mark.asyncio
    async def test_compare_bare_duplicate_job_name_across_namespaces_returns_409(
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
    async def test_compare_qualified_duplicate_job_names_keeps_each_namespace_separate(
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

        response = await client.get(
            "/api/v1/analytics/compare"
            "?jobs=bench-prod%2Fshared-load"
            "&jobs=bench-canary%2Fshared-load"
            "&metrics=request_throughput"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["job_ids"] == [
            "bench-prod/shared-load",
            "bench-canary/shared-load",
        ]
        avg_entry = next(
            entry
            for entry in body["entries"]
            if entry["metric"] == "request_throughput" and entry["stat"] == "avg"
        )
        assert avg_entry["values"] == {
            "bench-prod/shared-load": 100.0,
            "bench-canary/shared-load": 200.0,
        }

    @pytest.mark.asyncio
    async def test_compare_bare_unique_job_name_preserves_bare_response_key(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(tmp_path, namespace="bench-prod", job_id="unique-h100-load")

        response = await client.get(
            "/api/v1/analytics/compare?jobs=unique-h100-load&metrics=request_throughput"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["job_ids"] == ["unique-h100-load"]
        assert body["meta"] == {
            "unique-h100-load": {
                "gpu_count": 2,
                "gpu_name": "NVIDIA H100 80GB HBM3",
                "model": "meta-llama/Llama-3-8B",
                "endpoint": "http://llama3.svc.cluster.local:8000/v1",
            }
        }


# ============================================================
# Metrics and no-cross-run aggregation
# ============================================================


class TestCompareMetricsAndAggregationGuards:
    """Metric projection preserves missing values and never blends benchmarks."""

    @pytest.mark.asyncio
    async def test_compare_default_metrics_returns_contract_list_but_only_populated_entries(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            job_id="minimal-summary-load",
            payload=_summary_payload(throughput=123.0, latency=None),
        )

        response = await client.get(
            "/api/v1/analytics/compare?jobs=bench-prod%2Fminimal-summary-load"
        )

        assert response.status_code == 200
        body = response.json()
        assert body["metrics"] == DEFAULT_COMPARE_METRICS
        assert {entry["metric"] for entry in body["entries"]} == {"request_throughput"}
        assert body["entries"][0]["values"] == {
            "bench-prod/minimal-summary-load": 123.0
        }

    @pytest.mark.asyncio
    async def test_compare_requested_metric_missing_for_one_job_preserves_none_not_zero(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            job_id="throughput-only-load",
            payload=_summary_payload(throughput=111.0, ttft=None),
        )
        _seed_summary_run(
            tmp_path,
            job_id="ttft-enabled-load",
            payload=_summary_payload(throughput=222.0, ttft=37.0),
        )

        response = await client.get(
            "/api/v1/analytics/compare"
            "?jobs=bench-prod%2Fthroughput-only-load"
            "&jobs=bench-prod%2Fttft-enabled-load"
            "&metrics=time_to_first_token"
        )

        assert response.status_code == 200
        avg_entry = next(
            entry
            for entry in response.json()["entries"]
            if entry["metric"] == "time_to_first_token" and entry["stat"] == "avg"
        )
        assert avg_entry["values"] == {
            "bench-prod/throughput-only-load": None,
            "bench-prod/ttft-enabled-load": 37.0,
        }

    @pytest.mark.asyncio
    async def test_compare_same_job_historical_epoch_filter_does_not_average_epochs(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            job_id="resubmitted-load",
            epoch=_EPOCH_OLD,
            payload=_summary_payload(throughput=10.0),
            write_latest_pointer=False,
        )
        _seed_summary_run(
            tmp_path,
            job_id="resubmitted-load",
            epoch=_EPOCH_NEW,
            payload=_summary_payload(throughput=90.0),
        )

        response = await client.get(
            "/api/v1/analytics/compare"
            f"?jobs=bench-prod%2Fresubmitted-load"
            f"&metrics=request_throughput&epoch={_EPOCH_OLD}"
        )

        assert response.status_code == 200
        avg_entry = next(
            entry for entry in response.json()["entries"] if entry["stat"] == "avg"
        )
        assert avg_entry["values"] == {"bench-prod/resubmitted-load": 10.0}

    @pytest.mark.asyncio
    async def test_compare_unrequested_job_with_same_model_and_endpoint_is_not_included(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        model = "meta-llama/Llama-3.1-70B-Instruct"
        endpoint = "http://prod-router.svc.cluster.local:8000/v1"
        _seed_summary_run(
            tmp_path,
            job_id="requested-h100-load",
            payload=_summary_payload(throughput=300.0, model=model, endpoint=endpoint),
        )
        _seed_summary_run(
            tmp_path,
            job_id="unrequested-h100-load",
            payload=_summary_payload(throughput=900.0, model=model, endpoint=endpoint),
        )

        response = await client.get(
            "/api/v1/analytics/compare"
            "?jobs=bench-prod%2Frequested-h100-load"
            "&metrics=request_throughput"
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body["meta"]) == {"bench-prod/requested-h100-load"}
        avg_entry = next(entry for entry in body["entries"] if entry["stat"] == "avg")
        assert avg_entry["values"] == {"bench-prod/requested-h100-load": 300.0}


# ============================================================
# Model and endpoint metadata
# ============================================================


class TestCompareModelEndpointMetadata:
    """Compare carries model and endpoint context without broadening selection."""

    @pytest.mark.asyncio
    async def test_compare_model_and_endpoint_query_params_do_not_expand_requested_jobs(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        _seed_summary_run(
            tmp_path,
            job_id="llama3-prod-load",
            payload=_summary_payload(
                throughput=444.0,
                model="meta-llama/Llama-3.1-70B-Instruct",
                endpoint="http://prod-router.svc.cluster.local:8000/v1",
            ),
        )
        _seed_summary_run(
            tmp_path,
            job_id="mistral-canary-load",
            payload=_summary_payload(
                throughput=555.0,
                model="mistralai/Mistral-7B-Instruct-v0.3",
                endpoint="http://canary-router.svc.cluster.local:8000/v1",
            ),
        )

        response = await client.get(
            "/api/v1/analytics/compare"
            "?jobs=bench-prod%2Fllama3-prod-load"
            "&metrics=request_throughput"
            "&model=mistralai%2FMistral&endpoint=canary-router.svc"
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body["meta"]) == {"bench-prod/llama3-prod-load"}
        assert body["meta"]["bench-prod/llama3-prod-load"]["model"] == (
            "meta-llama/Llama-3.1-70B-Instruct"
        )
        assert body["meta"]["bench-prod/llama3-prod-load"]["endpoint"] == (
            "http://prod-router.svc.cluster.local:8000/v1"
        )


# ============================================================
# Stale index fallback
# ============================================================


class TestCompareStaleIndexFallback:
    """SQLite is a cache; compare falls back to disk when index rows are stale."""

    @pytest.mark.asyncio
    async def test_compare_stale_index_row_missing_run_dir_returns_disk_run_only(
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

        response = await client.get(
            "/api/v1/analytics/compare"
            "?jobs=bench-prod%2Fdeleted-index-load"
            "&jobs=bench-prod%2Fdisk-authoritative-load"
            "&metrics=request_throughput"
        )

        assert response.status_code == 200
        body = response.json()
        assert set(body["meta"]) == {"bench-prod/disk-authoritative-load"}
        avg_entry = next(entry for entry in body["entries"] if entry["stat"] == "avg")
        assert avg_entry["values"] == {"bench-prod/disk-authoritative-load": 321.0}


# ============================================================
# Malformed query params and trust-boundary shapes
# ============================================================


class TestCompareMalformedQueryParams:
    """Trust-boundary query shapes fail closed without SQL or server errors."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "path,expected_field",
        [
            ("/api/v1/analytics/compare", "jobs"),
            param(
                "/api/v1/analytics/compare?jobs=bench-prod%2Fllama-3-8b-load&epoch=not-an-epoch",
                "entries",
                id="malformed-epoch-is-empty-result",
            ),
        ],
    )  # fmt: skip
    async def test_compare_malformed_query_params_return_stable_response(
        self, tmp_path: Path, client: httpx.AsyncClient, path: str, expected_field: str
    ) -> None:
        _seed_summary_run(tmp_path)

        response = await client.get(path)

        if expected_field == "jobs":
            assert response.status_code == 422
            assert "jobs" in response.text
        else:
            assert response.status_code == 200
            assert response.json()[expected_field] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "metric_identifier",
        [
            "request_throughput;drop",
            param("request-throughput", id="hyphen-rejected"),
            param("request throughput", id="space-rejected"),
        ],
    )  # fmt: skip
    async def test_compare_invalid_metric_identifier_returns_empty_entries_not_500(
        self,
        tmp_path: Path,
        client: httpx.AsyncClient,
        metric_identifier: str,
    ) -> None:
        _seed_summary_run(tmp_path)

        response = await client.get(
            "/api/v1/analytics/compare",
            params={
                "jobs": "bench-prod/llama-3-8b-load",
                "metrics": metric_identifier,
            },
        )

        assert response.status_code == 200
        assert response.json()["entries"] == []

    @pytest.mark.asyncio
    async def test_compare_invalid_metric_identifier_with_open_index_returns_empty_not_500(
        self, tmp_path: Path, client: httpx.AsyncClient
    ) -> None:
        await runs_index.open(tmp_path / ".aiperf_index.sqlite")
        await _write_index_run(
            tmp_path,
            "bench-prod",
            "indexed-load",
            epoch=_EPOCH_NEW,
            payload=_summary_payload(throughput=777.0),
            create_disk_run=True,
        )

        response = await client.get(
            "/api/v1/analytics/compare"
            "?jobs=bench-prod%2Findexed-load&metrics=request_throughput%3Bdrop"
        )

        assert response.status_code == 200
        assert response.json()["entries"] == []

    @pytest.mark.asyncio
    async def test_compare_non_mapping_metric_shape_skips_run_not_500(
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

        response = await client.get(
            "/api/v1/analytics/compare"
            "?jobs=bench-prod%2Fscalar-metric-load&metrics=request_throughput"
        )

        assert response.status_code == 200
        assert response.json()["entries"] == []
