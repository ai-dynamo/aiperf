# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator web UI jobs API router."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from httpx import ASGITransport, AsyncClient
from kubernetes_asyncio.client.exceptions import ApiException

from aiperf.operator.routers.jobs import create_jobs_router


def _make_app(api=None, results_dir: Path | None = None):
    """Create a minimal FastAPI app with the jobs router for testing."""
    from fastapi import FastAPI

    app = FastAPI()
    holder = [api]
    router = create_jobs_router(holder, results_dir or Path("/tmp/aiperf-test-empty"))
    app.include_router(router)
    return app


def _aiperf_job_cr(
    *,
    name: str = "test-bench",
    namespace: str = "aiperf-benchmarks",
    phase: str = "Running",
) -> dict:
    """Return a minimal AIPerfJob CR dict that AIPerfJobCR can validate."""
    return {
        "apiVersion": "aiperf.nvidia.com/v1alpha1",
        "kind": "AIPerfJob",
        "metadata": {
            "name": name,
            "namespace": namespace,
            "creationTimestamp": "2026-03-19T18:00:00Z",
        },
        "spec": {
            "endpoint": {
                "url": "http://vllm-server:8000/v1",
                "model": "Qwen/Qwen3-0.6B",
            }
        },
        "status": {
            "phase": phase,
            "jobId": name,
            "jobSetName": f"aiperf-{name}",
            "workers": {"ready": 1, "total": 1},
            "startTime": "2026-03-19T18:00:00Z",
        },
    }


def _node_obj(name: str = "node1", gpu: str = "1") -> MagicMock:
    """Build a typed-ish V1Node mock with an ``nvidia.com/gpu`` allocation."""
    node = MagicMock()
    node.metadata = MagicMock()
    node.metadata.name = name
    node.status = MagicMock()
    node.status.allocatable = {"nvidia.com/gpu": gpu}
    return node


class TestListJobs:
    @pytest.mark.asyncio
    async def test_list_jobs_returns_jobs(self):
        mock_api = MagicMock()
        mock_custom = MagicMock()
        mock_custom.list_cluster_custom_object = AsyncMock(
            return_value={"items": [_aiperf_job_cr()]}
        )
        app = _make_app(mock_api)

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/v1/jobs")

        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["name"] == "test-bench"

    @pytest.mark.asyncio
    async def test_list_jobs_no_client_returns_503(self):
        app = _make_app(api=None)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/jobs")

        assert resp.status_code == 503


class TestGetJob:
    @pytest.mark.asyncio
    async def test_get_job_found(self):
        mock_api = MagicMock()
        cr = _aiperf_job_cr()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(return_value=cr)
        # Empty pod list to keep the pods section simple
        mock_core = MagicMock(
            list_namespaced_pod=AsyncMock(return_value=MagicMock(items=[]))
        )
        app = _make_app(mock_api)

        with (
            patch(
                "aiperf.kubernetes.client.client.CustomObjectsApi",
                return_value=mock_custom,
            ),
            patch(
                "aiperf.kubernetes.client.client.CoreV1Api",
                return_value=mock_core,
            ),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/v1/jobs/aiperf-benchmarks/test-bench")

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_get_job_not_found(self):
        mock_api = MagicMock()
        mock_custom = MagicMock()
        # Both direct lookup and cluster scan return 404/empty
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=404)
        )
        mock_custom.list_cluster_custom_object = AsyncMock(return_value={"items": []})
        app = _make_app(mock_api)

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/v1/jobs/aiperf-benchmarks/nonexistent")

        assert resp.status_code == 404


class TestCluster:
    @pytest.mark.asyncio
    async def test_cluster_info(self):
        mock_api = MagicMock()
        node = _node_obj(gpu="1")
        mock_core = MagicMock(list_node=AsyncMock(return_value=MagicMock(items=[node])))

        # cluster_version builds its result from VersionApi.get_code; mock it.
        version_info = MagicMock()
        version_info.major = "1"
        version_info.minor = "33"
        version_info.git_version = "v1.33.1"
        version_info.git_commit = "abc"
        version_info.platform = "linux/amd64"
        mock_version = MagicMock(get_code=AsyncMock(return_value=version_info))

        app = _make_app(mock_api)

        with (
            patch(
                "aiperf.kubernetes.client.client.VersionApi",
                return_value=mock_version,
            ),
            patch(
                "aiperf.operator.routers.jobs.client.CoreV1Api",
                return_value=mock_core,
            ),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/api/v1/cluster")

        assert resp.status_code == 200
        data = resp.json()
        assert data["nodes"] == 1
        assert data["gpus"] == 1
        assert data["kubernetes_version"] == "v1.33.1"


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_job(self):
        mock_api = MagicMock()
        mock_patch = AsyncMock(return_value={})
        mock_get = AsyncMock(return_value=_aiperf_job_cr())
        mock_custom = MagicMock(
            patch_namespaced_custom_object=mock_patch,
            get_namespaced_custom_object=mock_get,
        )
        app = _make_app(mock_api)

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/v1/jobs/aiperf-benchmarks/test-bench/cancel"
                )

        assert resp.status_code == 200
        # Verify we issued the {"spec": {"cancel": True}} merge patch
        mock_patch.assert_awaited_once()
        kwargs = mock_patch.call_args.kwargs
        assert kwargs["body"] == {"spec": {"cancel": True}}
        assert kwargs["namespace"] == "aiperf-benchmarks"
        assert kwargs["name"] == "test-bench"

    @pytest.mark.asyncio
    async def test_cancel_archived_job_returns_400(self, tmp_path: Path):
        """Archived (PVC-only) jobs cannot be cancelled — should return 400."""
        from fastapi import FastAPI

        d = tmp_path / "ns" / "ghost"
        d.mkdir(parents=True)
        (d / "profile_export_aiperf.json").write_bytes(
            orjson.dumps({"status": "Succeeded"})
        )

        mock_api = MagicMock()
        mock_custom = MagicMock()
        # No CR for this job — 404 on direct lookup, empty on cluster scan
        mock_custom.get_namespaced_custom_object = AsyncMock(
            side_effect=ApiException(status=404)
        )
        mock_custom.list_cluster_custom_object = AsyncMock(return_value={"items": []})

        app = FastAPI()
        app.include_router(create_jobs_router([mock_api], tmp_path))

        with patch(
            "aiperf.kubernetes.client.client.CustomObjectsApi",
            return_value=mock_custom,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post("/api/v1/jobs/ns/ghost/cancel")

        assert resp.status_code == 400
        assert "archived" in resp.text.lower()


def test_list_jobs_includes_archived_only_entry(tmp_path: Path, monkeypatch):
    """GET /api/v1/jobs returns a PVC-only entry when no CR exists for it."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from aiperf.operator import job_union as ju
    from aiperf.operator.routers.jobs import create_jobs_router

    d = tmp_path / "aiperf-bench" / "archive-only"
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(
            {
                "status": "Succeeded",
                "request_throughput": {"avg": 50.0, "unit": "requests/sec"},
            }
        )
    )

    async def fake_list(api, *, all_namespaces=True, namespace=None, **_):
        return []

    monkeypatch.setattr(ju, "list_aiperf_jobs", fake_list)

    api_holder = [object()]
    router = create_jobs_router(api_holder, tmp_path)

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    names = {j["name"]: j for j in body["jobs"]}
    assert "archive-only" in names
    assert names["archive-only"]["source"] == "archived"
