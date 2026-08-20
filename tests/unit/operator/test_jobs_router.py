# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the operator web UI jobs API router."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
from aiperf.operator.results_layout import write_latest
from aiperf.operator.routers.jobs import create_jobs_router
from httpx import ASGITransport, AsyncClient
from kubernetes_asyncio.client.exceptions import ApiException

_TEST_EPOCH = "1714064523"


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
        mock_core = MagicMock(
            list_node=AsyncMock(return_value=MagicMock(items=[node])),
            list_pod_for_all_namespaces=AsyncMock(return_value=MagicMock(items=[])),
        )

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
        assert data["gpus_used"] == 0
        assert data["gpus_free"] == 1
        assert data["gpu_nodes"] == 1
        assert data["nodes_free"] == 1
        assert data["nodes_partial"] == 0
        assert data["nodes_full"] == 0
        assert data["utilization_percent"] == 0.0
        assert data["kubernetes_version"] == "v1.33.1"

    @pytest.mark.asyncio
    async def test_cluster_info_gpu_usage_breakdown(self):
        """Two 8-GPU nodes: one fully used, one partially used → mixed states."""
        mock_api = MagicMock()
        node_a = _node_obj(name="node-a", gpu="8")
        node_b = _node_obj(name="node-b", gpu="8")

        def _gpu_pod(name: str, node: str, gpus: int, phase: str) -> MagicMock:
            pod = MagicMock()
            pod.metadata = MagicMock(name=name, namespace="bench")
            pod.spec = MagicMock(node_name=node)
            container = MagicMock()
            container.resources = MagicMock(requests={"nvidia.com/gpu": str(gpus)})
            pod.spec.containers = [container]
            pod.status = MagicMock(phase=phase)
            return pod

        # node-a: 8 used (full); node-b: 4 used (partial); plus a Succeeded pod
        # that should be ignored entirely.
        pods = [
            _gpu_pod("worker-a", "node-a", 8, "Running"),
            _gpu_pod("worker-b", "node-b", 4, "Running"),
            _gpu_pod("done", "node-b", 4, "Succeeded"),
        ]

        mock_core = MagicMock(
            list_node=AsyncMock(return_value=MagicMock(items=[node_a, node_b])),
            list_pod_for_all_namespaces=AsyncMock(return_value=MagicMock(items=pods)),
        )

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
        assert data["gpus"] == 16
        assert data["gpus_used"] == 12
        assert data["gpus_free"] == 4
        assert data["utilization_percent"] == 75.0
        assert data["gpu_nodes"] == 2
        assert data["nodes_free"] == 0
        assert data["nodes_partial"] == 1
        assert data["nodes_full"] == 1


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

        d = tmp_path / "ns" / "ghost" / _TEST_EPOCH
        d.mkdir(parents=True)
        (d / "profile_export_aiperf.json").write_bytes(
            orjson.dumps({"status": "Succeeded"})
        )
        write_latest(tmp_path, "ns", "ghost", _TEST_EPOCH)

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


def test_get_job_archived_synthesizes_status(tmp_path: Path, monkeypatch):
    """GET /api/v1/jobs/{ns}/{name} for an archived job returns a full synthesized status."""
    from aiperf.operator.routers.jobs import create_jobs_router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from aiperf.operator import job_union as ju

    d = tmp_path / "ns" / "ghost" / _TEST_EPOCH
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(
            {
                "status": "Succeeded",
                "start_time": "2026-04-22T10:00:00Z",
                "end_time": "2026-04-22T10:45:00Z",
                "request_count": 7777,
                "request_throughput": {"avg": 42.1, "unit": "requests/sec"},
                "request_latency": {"avg": 180.0, "p99": 390.0, "unit": "ms"},
                "time_to_first_token": {"avg": 45.5, "p99": 120.0, "unit": "ms"},
                "output_token_throughput": {"avg": 2048.5, "unit": "tokens/sec"},
            }
        )
    )
    write_latest(tmp_path, "ns", "ghost", _TEST_EPOCH)

    async def fake_find(api, name, namespace):
        return None

    monkeypatch.setattr(ju, "find_aiperf_job", fake_find)

    api_holder = [object()]
    router = create_jobs_router(api_holder, tmp_path)

    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        resp = client.get("/api/v1/jobs/ns/ghost")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["job"]["source"] == "archived"
    status = body["status"]
    assert status["phase"] == "Archived"
    assert status["currentPhase"] == "completed"
    assert status["workers"] == {"ready": 0, "total": 0}
    assert status["summary"]["request_throughput"]["avg"] == 42.1
    assert status["summary"]["request_latency"]["p99"] == 390.0
    assert status["summary"]["time_to_first_token"]["avg"] == 45.5
    assert status["summary"]["total_requests"] == 7777
    assert status["phases"]["benchmark"]["requestsCompleted"] == 7777
    assert body["pods"] == []


def test_list_jobs_includes_archived_only_entry(tmp_path: Path, monkeypatch):
    """GET /api/v1/jobs returns a PVC-only entry when no CR exists for it."""
    from aiperf.operator.routers.jobs import create_jobs_router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from aiperf.operator import job_union as ju

    d = tmp_path / "aiperf-bench" / "archive-only" / _TEST_EPOCH
    d.mkdir(parents=True)
    (d / "profile_export_aiperf.json").write_bytes(
        orjson.dumps(
            {
                "status": "Succeeded",
                "request_throughput": {"avg": 50.0, "unit": "requests/sec"},
            }
        )
    )
    write_latest(tmp_path, "aiperf-bench", "archive-only", _TEST_EPOCH)

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


def _v1_event(
    *,
    reason: str,
    message: str,
    type_: str = "Warning",
    involved_kind: str = "Pod",
    involved_name: str = "test-bench-controller-0",
) -> MagicMock:
    """Build a V1Event-shaped mock with the attributes _event_to_entry reads."""
    from datetime import datetime

    ts = datetime(2026, 4, 30, 12, 0, 0, tzinfo=UTC)
    ev = MagicMock()
    ev.type = type_
    ev.reason = reason
    ev.message = message
    ev.first_timestamp = ts
    ev.last_timestamp = ts
    ev.event_time = None
    ev.count = 1
    involved = MagicMock()
    involved.kind = involved_kind
    involved.name = involved_name
    involved.namespace = "aiperf-benchmarks"
    ev.involved_object = involved
    src = MagicMock()
    src.component = "kubelet"
    src.host = "node-1"
    ev.source = src
    return ev


class TestListJobEvents:
    @pytest.mark.asyncio
    async def test_filters_known_gke_admission_policy_noise(self):
        """PolicyViolation events from the buggy GKE p4sa-audience policy are dropped."""
        cr = _aiperf_job_cr()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(return_value=cr)

        noise = _v1_event(
            reason="PolicyViolation",
            message=(
                "policy validating-node-p4sa-audience/validating-node-p4sa-audience "
                "error: expression '![\"system:addon-manager\", ...]' resulted in "
                "error: no such key: username"
            ),
            involved_name="test-bench-controller-0",
        )
        real = _v1_event(
            reason="FailedScheduling",
            message="0/3 nodes are available: insufficient nvidia.com/gpu",
            involved_name="test-bench-controller-0",
        )

        mock_core = MagicMock(
            list_namespaced_event=AsyncMock(
                return_value=MagicMock(items=[noise, real])
            ),
            list_namespaced_pod=AsyncMock(return_value=MagicMock(items=[])),
        )
        app = _make_app(MagicMock())

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
                resp = await client.get(
                    "/api/v1/jobs/aiperf-benchmarks/test-bench/events"
                )

        assert resp.status_code == 200
        body = resp.json()
        reasons = [e["reason"] for e in body["events"]]
        assert "PolicyViolation" not in reasons
        assert reasons == ["FailedScheduling"]

    @pytest.mark.asyncio
    async def test_unrelated_policy_violation_is_kept(self):
        """PolicyViolation events from non-allowlisted policies still surface."""
        cr = _aiperf_job_cr()
        mock_custom = MagicMock()
        mock_custom.get_namespaced_custom_object = AsyncMock(return_value=cr)

        kept = _v1_event(
            reason="PolicyViolation",
            message="policy some-real-workload-policy/foo violated by container image",
        )
        mock_core = MagicMock(
            list_namespaced_event=AsyncMock(return_value=MagicMock(items=[kept])),
            list_namespaced_pod=AsyncMock(return_value=MagicMock(items=[])),
        )
        app = _make_app(MagicMock())

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
                resp = await client.get(
                    "/api/v1/jobs/aiperf-benchmarks/test-bench/events"
                )

        assert resp.status_code == 200
        assert [e["reason"] for e in resp.json()["events"]] == ["PolicyViolation"]
