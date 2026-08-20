# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for the operator jobs HTTP router.

Focuses on:
- namespace/name path decoding and encoded-slash smuggling rejection
- 404 vs Kubernetes API 5xx behavior on detail and cancel routes
- list response schema stability when CR status/spec payloads are sparse
- sweep child linkage fields from live CR labels and archived sweep markers
- cancel endpoint merge-patch wire shape

Out of scope: logs streaming and epoch history, covered by sibling jobs router tests.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import orjson
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from kubernetes_asyncio.client.exceptions import ApiException
from pytest import param

from aiperf.common.redact import REDACTED_VALUE
from aiperf.kubernetes.models import AIPerfJobInfo
from aiperf.operator.results_layout import write_latest
from aiperf.operator.routers.jobs import create_jobs_router

# ============================================================
# Helpers
# ============================================================

_EPOCH = "1714064523"


def _app(api: object | None, results_dir: Path) -> FastAPI:
    """Build only the jobs router with the production Kubernetes exception shape."""
    app = FastAPI()

    @app.exception_handler(ApiException)
    async def _api_exception_handler(
        request: Request, exc: ApiException
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exc.status or 500,
            content={"detail": str(exc.body or exc.reason or "Kubernetes API error")},
        )

    app.include_router(create_jobs_router([api], results_dir))
    return app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client backed by a temp PVC-like results directory and live API token."""
    transport = httpx.ASGITransport(
        app=_app(object(), tmp_path), raise_app_exceptions=False
    )
    async with httpx.AsyncClient(
        transport=transport, base_url="http://aiperf.operator.local"
    ) as c:
        yield c


def _live_job_info(
    *,
    namespace: str = "aiperf-benchmarks",
    name: str = "llama-3-8b-load",
    phase: str = "Running",
    source: Literal["live", "archived", "both"] = "live",
    sweep_name: str | None = None,
    variation_index: int | None = None,
    variation_label: str | None = None,
    endpoint: str = "http://vllm-router.aiperf-system:8000/v1",
) -> AIPerfJobInfo:
    """Return a real display model matching the router's list/detail schema."""
    return AIPerfJobInfo(
        name=name,
        namespace=namespace,
        phase=phase,
        job_id=name,
        jobset_name=f"aiperf-{name}",
        workers_ready=1,
        workers_total=2,
        current_phase="profiling",
        created="2026-05-18T17:00:00Z",
        source=source,
        sweep_name=sweep_name,
        variation_index=variation_index,
        variation_label=variation_label,
        model="meta-llama/Llama-3-8B",
        endpoint=endpoint,
    )


def _raw_aiperf_job(
    *,
    namespace: str = "aiperf-benchmarks",
    name: str = "llama-3-8b-load",
    labels: dict[str, str] | None = None,
    spec: dict[str, object] | None = None,
    status: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return a raw CR dict close to a Kubernetes CustomObjectsApi response."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": "2026-05-18T17:00:00Z",
            "labels": labels or {},
        },
        "spec": spec
        if spec is not None
        else {
            "benchmark": {
                "models": {"items": [{"name": "meta-llama/Llama-3-8B"}]},
                "endpoint": {"urls": ["http://vllm-router:8000/v1"]},
            }
        },
        "status": status
        if status is not None
        else {
            "phase": "Running",
            "jobId": name,
            "jobSetName": f"aiperf-{name}",
            "workers": {"ready": 1, "total": 2},
            "currentPhase": "profiling",
            "runEpoch": int(_EPOCH),
        },
    }


def _seed_archived_run(
    base_dir: Path,
    *,
    namespace: str = "aiperf-sweeps",
    name: str = "llama-sweep-v01",
    sweep_marker: dict[str, object] | None = None,
) -> None:
    """Create an archived run with optional sweep-controller linkage marker."""
    job_dir = base_dir / namespace / name
    run_dir = job_dir / _EPOCH
    run_dir.mkdir(parents=True)
    (run_dir / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(
            {
                "status": "Succeeded",
                "start_time": "2026-05-18T17:00:00Z",
                "end_time": "2026-05-18T17:30:00Z",
                "request_throughput": {"avg": 125.5},
                "input_config": {
                    "models": {"items": [{"name": "meta-llama/Llama-3-8B"}]},
                    "endpoint": {"urls": ["http://vllm-router:8000/v1"]},
                },
            }
        )
    )
    if sweep_marker is not None:
        (job_dir / "sweep.json").write_bytes(orjson.dumps(sweep_marker))
    write_latest(base_dir, namespace, name, _EPOCH)


# ============================================================
# Namespace/name path decoding
# ============================================================


class TestJobsRouterPathEncoding:
    """Route parameters are decoded once and never let slashes cross segments."""

    @pytest.mark.asyncio
    async def test_get_job_url_encoded_dns_characters_resolve_to_decoded_job(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator import job_union as ju
        from aiperf.operator.routers import jobs as jobs_module

        seen: list[tuple[str, str]] = []

        async def fake_find(
            api: object,
            results_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> AIPerfJobInfo:
            del api, results_dir, epoch
            seen.append((namespace, name))
            return _live_job_info(namespace=namespace, name=name)

        monkeypatch.setattr(jobs_module, "find_any_job", fake_find)
        monkeypatch.setattr(
            jobs_module,
            "get_raw_aiperfjob_status",
            AsyncMock(return_value={"phase": "Running"}),
        )
        monkeypatch.setattr(jobs_module, "get_pods", AsyncMock(return_value=[]))

        response = await client.get(
            "/api/v1/jobs/aiperf%2Dbenchmarks/llama%2D3%2E1%2D8b%2Drun"
        )

        assert response.status_code == 200, response.text
        assert seen == [("aiperf-benchmarks", "llama-3.1-8b-run")]
        assert response.json()["job"]["name"] == "llama-3.1-8b-run"
        assert ju.find_any_job is not fake_find

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "encoded_path",
        [
            param("aiperf-benchmarks/llama-3-8b%2Fstolen", id="name-encoded-slash"),
            param("aiperf%2Fsystem/llama-3-8b-load", id="namespace-encoded-slash"),
        ],
    )  # fmt: skip
    async def test_get_job_encoded_slash_smuggling_returns_404_without_lookup(
        self,
        client: httpx.AsyncClient,
        monkeypatch: pytest.MonkeyPatch,
        encoded_path: str,
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        fake_find = AsyncMock(return_value=_live_job_info())
        monkeypatch.setattr(jobs_module, "find_any_job", fake_find)

        response = await client.get(f"/api/v1/jobs/{encoded_path}")

        assert response.status_code == 404
        fake_find.assert_not_awaited()


# ============================================================
# 404 vs Kubernetes API 5xx behavior
# ============================================================


class TestJobsRouterErrorContracts:
    """Missing jobs are 404; accepted live jobs still surface API patch failures."""

    @pytest.mark.asyncio
    async def test_get_job_missing_live_and_archived_job_returns_404_with_identity(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        async def fake_missing(
            api: object,
            results_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> None:
            del api, results_dir, namespace, name, epoch
            return None

        monkeypatch.setattr(jobs_module, "find_any_job", fake_missing)

        response = await client.get("/api/v1/jobs/aiperf-benchmarks/missing-llama-run")

        assert response.status_code == 404
        assert "aiperf-benchmarks/missing-llama-run" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_cancel_job_patch_api_500_surfaces_kubernetes_detail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator import job_union as ju

        async def fake_find(api: object, name: str, namespace: str) -> AIPerfJobInfo:
            del api
            return _live_job_info(namespace=namespace, name=name)

        monkeypatch.setattr(ju, "find_aiperf_job", fake_find)
        api_error = ApiException(status=500, reason="Internal Server Error")
        api_error.body = "apiserver etcd timeout"
        mock_custom = MagicMock(
            patch_namespaced_custom_object=AsyncMock(side_effect=api_error)
        )
        transport = httpx.ASGITransport(
            app=_app(object(), tmp_path), raise_app_exceptions=False
        )

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://aiperf.operator.local"
            ) as c:
                response = await c.post(
                    "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-load/cancel"
                )

        assert response.status_code == 500
        assert "apiserver etcd timeout" in response.json()["detail"]


# ============================================================
# List schema, sparse payloads, and sweep child fields
# ============================================================


class TestJobsRouterListSchema:
    """The jobs list stays schema-stable for sparse CRs and sweep children."""

    @pytest.mark.asyncio
    async def test_list_jobs_sparse_status_and_spec_defaults_keep_stable_schema(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator import job_union as ju

        async def fake_list(
            api: object,
            *,
            all_namespaces: bool = True,
            namespace: str | None = None,
        ) -> list[AIPerfJobInfo]:
            del api, namespace
            assert all_namespaces is True
            cr = _raw_aiperf_job(
                name="sparse-cr-load",
                spec={},
                status={"phase": None, "workers": {}},
            )
            from aiperf.kubernetes.models import AIPerfJobCR

            return [AIPerfJobCR.model_validate(cr).to_info()]

        monkeypatch.setattr(ju, "list_aiperf_jobs", fake_list)

        response = await client.get("/api/v1/jobs?namespace=ignored&status=Failed")

        assert response.status_code == 200, response.text
        body = response.json()
        assert len(body["jobs"]) == 1
        job = body["jobs"][0]
        expected_keys = {
            "name",
            "namespace",
            "phase",
            "jobId",
            "jobsetName",
            "workersReady",
            "workersTotal",
            "currentPhase",
            "error",
            "startTime",
            "completionTime",
            "created",
            "progressPercent",
            "throughputRps",
            "latencyP99Ms",
            "ttftMs",
            "outputTokenThroughputTps",
            "interTokenLatencyMs",
            "totalRequests",
            "errorRate",
            "model",
            "endpoint",
            "source",
            "sweepName",
            "variationIndex",
            "variationLabel",
        }
        assert expected_keys <= set(job)
        assert job["phase"] == "Pending"
        assert job["workersReady"] == 0
        assert job["workersTotal"] == 0
        assert job["model"] is None
        assert job["endpoint"] is None

    @pytest.mark.asyncio
    async def test_list_jobs_live_sweep_child_labels_surface_variation_fields(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator import job_union as ju

        async def fake_list(
            api: object,
            *,
            all_namespaces: bool = True,
            namespace: str | None = None,
        ) -> list[AIPerfJobInfo]:
            del api, all_namespaces, namespace
            cr = _raw_aiperf_job(
                namespace="aiperf-sweeps",
                name="llama-sweep-v02",
                labels={
                    "aiperf.nvidia.com/sweep": "llama-throughput-sweep",
                    "aiperf.nvidia.com/variation-index": "2",
                    "aiperf.nvidia.com/variation-label": "request_rate=2000",
                },
            )
            from aiperf.kubernetes.models import AIPerfJobCR

            return [AIPerfJobCR.model_validate(cr).to_info()]

        monkeypatch.setattr(ju, "list_aiperf_jobs", fake_list)

        response = await client.get("/api/v1/jobs")

        assert response.status_code == 200, response.text
        job = response.json()["jobs"][0]
        assert job["sweepName"] == "llama-throughput-sweep"
        assert job["variationIndex"] == 2
        assert job["variationLabel"] == "request_rate=2000"

    @pytest.mark.asyncio
    async def test_list_jobs_redacts_live_endpoint_credentials(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        endpoint = "https://user:password@llm/v1?api_key=secret&model=llama"
        monkeypatch.setattr(
            jobs_module,
            "list_all_jobs",
            AsyncMock(return_value=[_live_job_info(endpoint=endpoint)]),
        )

        response = await client.get("/api/v1/jobs")

        assert response.status_code == 200, response.text
        assert response.json()["jobs"][0]["endpoint"] == (
            f"https://{REDACTED_VALUE}@llm/v1?api_key={REDACTED_VALUE}&model=llama"
        )

    @pytest.mark.asyncio
    async def test_list_jobs_archived_sweep_marker_surfaces_variation_fields(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from aiperf.operator import job_union as ju

        async def fake_empty_list(
            api: object,
            *,
            all_namespaces: bool = True,
            namespace: str | None = None,
        ) -> list[AIPerfJobInfo]:
            del api, all_namespaces, namespace
            return []

        monkeypatch.setattr(ju, "list_aiperf_jobs", fake_empty_list)
        _seed_archived_run(
            tmp_path,
            sweep_marker={
                "sweep_name": "llama-throughput-sweep",
                "variation_index": "7",
                "variation_label": "concurrency=512",
            },
        )
        transport = httpx.ASGITransport(
            app=_app(object(), tmp_path), raise_app_exceptions=False
        )

        async with httpx.AsyncClient(
            transport=transport, base_url="http://aiperf.operator.local"
        ) as c:
            response = await c.get("/api/v1/jobs")

        assert response.status_code == 200, response.text
        job = response.json()["jobs"][0]
        assert job["source"] == "archived"
        assert job["sweepName"] == "llama-throughput-sweep"
        assert job["variationIndex"] == 7
        assert job["variationLabel"] == "concurrency=512"


# ============================================================
# Detail schema and sweep children
# ============================================================


class TestJobsRouterDetailSchema:
    """Job detail keeps sweep-child fields stable alongside raw status."""

    @pytest.mark.asyncio
    async def test_get_job_live_sweep_child_preserves_linkage_and_status(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        async def fake_find(
            api: object,
            results_dir: Path,
            namespace: str,
            name: str,
            *,
            epoch: str | None = None,
        ) -> AIPerfJobInfo:
            del api, results_dir, epoch
            return _live_job_info(
                namespace=namespace,
                name=name,
                source="both",
                sweep_name="llama-throughput-sweep",
                variation_index=11,
                variation_label="temperature=0.2",
            )

        raw_status = {
            "phase": "Running",
            "subPhase": "profiling",
            "runEpoch": int(_EPOCH),
            "summary": {"request_throughput": {"avg": 1500.0}},
        }
        monkeypatch.setattr(jobs_module, "find_any_job", fake_find)
        monkeypatch.setattr(
            jobs_module, "get_raw_aiperfjob_status", AsyncMock(return_value=raw_status)
        )
        monkeypatch.setattr(jobs_module, "get_pods", AsyncMock(return_value=[]))

        response = await client.get("/api/v1/jobs/aiperf-sweeps/llama-sweep-v11")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["job"]["source"] == "both"
        assert body["job"]["sweepName"] == "llama-throughput-sweep"
        assert body["job"]["variationIndex"] == 11
        assert body["job"]["variationLabel"] == "temperature=0.2"
        assert body["status"] == raw_status
        assert body["pods"] == []

    @pytest.mark.asyncio
    async def test_get_job_redacts_live_endpoint_credentials(
        self, client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator.routers import jobs as jobs_module

        endpoint = "https://user:password@llm/v1?token=secret&model=llama"
        monkeypatch.setattr(
            jobs_module,
            "find_any_job",
            AsyncMock(return_value=_live_job_info(endpoint=endpoint)),
        )
        monkeypatch.setattr(
            jobs_module, "get_raw_aiperfjob_status", AsyncMock(return_value={})
        )
        monkeypatch.setattr(jobs_module, "get_pods", AsyncMock(return_value=[]))

        response = await client.get("/api/v1/jobs/aiperf-benchmarks/llama-3-8b-load")

        assert response.status_code == 200, response.text
        assert response.json()["job"]["endpoint"] == (
            f"https://{REDACTED_VALUE}@llm/v1?token={REDACTED_VALUE}&model=llama"
        )


# ============================================================
# Cancel patch shape
# ============================================================


class TestJobsRouterCancelShape:
    """Cancellation must remain a minimal merge patch on spec.cancel."""

    @pytest.mark.asyncio
    async def test_cancel_job_live_job_uses_merge_patch_without_force_or_field_manager(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from aiperf.operator import job_union as ju

        async def fake_find(api: object, name: str, namespace: str) -> AIPerfJobInfo:
            del api
            return _live_job_info(namespace=namespace, name=name)

        monkeypatch.setattr(ju, "find_aiperf_job", fake_find)
        mock_patch = AsyncMock(return_value={})
        mock_custom = MagicMock(patch_namespaced_custom_object=mock_patch)
        transport = httpx.ASGITransport(
            app=_app(object(), tmp_path), raise_app_exceptions=False
        )

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://aiperf.operator.local"
            ) as c:
                response = await c.post(
                    "/api/v1/jobs/aiperf-benchmarks/llama-3-8b-load/cancel"
                )

        assert response.status_code == 200, response.text
        assert response.json() == {"cancelled": True}
        mock_patch.assert_awaited_once()
        kwargs = mock_patch.call_args.kwargs
        assert kwargs["namespace"] == "aiperf-benchmarks"
        assert kwargs["name"] == "llama-3-8b-load"
        assert kwargs["body"] == {"spec": {"cancel": True}}
        assert kwargs["_content_type"] == "application/merge-patch+json"
        assert "force" not in kwargs
        assert "field_manager" not in kwargs
