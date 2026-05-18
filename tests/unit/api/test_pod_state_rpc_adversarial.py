# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial tests for pod-state control RPC surfaces.

Focuses on:
- GET_POD_STATES timeout propagation and command-channel failure boundaries.
- Malformed controller payloads that must degrade to cache fallback, not 500s.
- Partial pod-state snapshots crossing the debug/progress router trust boundary.
- Kubernetes patch helper name/namespace pass-through for annotation mirrors.

Out of scope: WorkerGroupManager state production; see worker pod-manager tests.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi import FastAPI
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.pod_state_rpc import query_controller_pod_states
from aiperf.api.routers.debug import DebugRouter
from aiperf.api.routers.progress import ProgressRouter
from aiperf.common.control_structs import CommandErr, CommandOk
from aiperf.common.enums import CommandType
from aiperf.common.messages import WorkerPodStateMessage
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.controller.system_controller import AggregateWorkerStatus


# ============================================================================
# Helpers
# ============================================================================


class _CommandSender(Protocol):
    async def send_command_to_controller(self, cmd: str, timeout: float) -> object: ...


class _FakeApp:
    """Stand-in for ``request.app`` exposing only ``app.state``."""

    def __init__(self, service: object | None, *, install_service: bool = True) -> None:
        self.state = SimpleNamespace()
        if install_service:
            self.state.service = service


class _FakeConn:
    """Stand-in for ``starlette.requests.HTTPConnection``."""

    def __init__(self, service: object | None, *, install_service: bool = True) -> None:
        self.app = _FakeApp(service, install_service=install_service)


class _Service:
    """Captures controller-command arguments and replays a response or exception."""

    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[tuple[str, float]] = []

    async def send_command_to_controller(self, cmd: str, timeout: float) -> object:
        self.calls.append((cmd, timeout))
        if isinstance(self._response, BaseException):
            raise self._response
        return self._response


class _Controller:
    """In-process controller stub that wins over sidecar RPC fallback."""

    def get_aggregate_worker_status(self) -> AggregateWorkerStatus:
        return AggregateWorkerStatus(ready=9, total=9, ready_pods=3, total_pods=3)


def _ok(payload: object) -> CommandOk:
    return CommandOk(
        cid="cid-aiperf-bench-7f2a",
        sid="system_controller",
        payload=orjson.dumps(payload),
    )


def _pod(
    pod_index: str,
    *,
    declared: int = 4,
    ready: int = 4,
    service_id: str | None = None,
) -> WorkerPodStateMessage:
    return WorkerPodStateMessage(
        service_id=service_id or f"wpm-aiperf-bench-7f2a-{pod_index}",
        pod_index=pod_index,
        benchmark_generation="bench-gen-20260518",
        dataset_generation="dataset-gen-20260518",
        declared_workers=declared,
        declared_record_processors=1,
        router_connected_workers=ready,
        dispatchable_workers=ready,
        ready_workers=ready,
        ready_record_processors=1,
        degraded_workers=max(0, declared - ready),
        degraded_record_processors=0,
        pod_state="ready" if ready >= 1 else "starting",
        admission_state="dispatchable" if ready >= 1 else "admitting",
    )


@pytest.fixture
def benchmark_run() -> BenchmarkRun:
    """Real Pydantic config for router construction."""
    config = AIPerfConfig(
        benchmark={
            "models": ["meta-llama/Llama-3-8B"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": [
                {
                    "name": "synthetic-chat-main",
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            ],
            "phases": [
                {
                    "name": "steady-state",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id="aiperf-bench-7f2a",
        cfg=config.benchmark,
        artifact_dir=Path("/tmp/aiperf-bench-7f2a"),
    )


@pytest.fixture
def debug_router(mock_zmq: object, benchmark_run: BenchmarkRun) -> DebugRouter:
    return DebugRouter(run=benchmark_run)


@pytest.fixture
def debug_client(debug_router: DebugRouter) -> TestClient:
    app = FastAPI()
    app.state.debug = debug_router
    app.include_router(debug_router.get_router())
    return TestClient(app)


@pytest.fixture
def progress_router(mock_zmq: object, benchmark_run: BenchmarkRun) -> ProgressRouter:
    return ProgressRouter(run=benchmark_run)


@pytest.fixture
def progress_client(progress_router: ProgressRouter) -> TestClient:
    app = FastAPI()
    app.state.progress = progress_router
    app.include_router(progress_router.get_router())
    return TestClient(app)


# ============================================================================
# query_controller_pod_states trust-boundary behavior
# ============================================================================


class TestQueryControllerPodStatesRPC:
    """Control RPC command dispatch and failure-to-None boundaries."""

    @pytest.mark.asyncio
    async def test_query_controller_pod_states_timeout_propagates_to_command_sender(
        self,
    ) -> None:
        payload = {"pod_states": {}, "worker_startup_states": {}}
        service = _Service(_ok(payload))

        result = await query_controller_pod_states(_FakeConn(service), timeout=0.125)

        assert result == payload
        assert service.calls == [(CommandType.GET_POD_STATES, 0.125)]

    @pytest.mark.parametrize(
        "response",
        [
            param(
                CommandErr(
                    cid="cid-aiperf-bench-7f2a",
                    sid="system_controller",
                    error="controller refused GET_POD_STATES",
                ),
                id="command-err",
            ),
            param(TimeoutError("GET_POD_STATES timed out"), id="timeout-error"),
            param(RuntimeError("control connector closed"), id="runtime-error"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_query_controller_pod_states_command_failure_returns_none(
        self, response: object
    ) -> None:
        assert await query_controller_pod_states(_FakeConn(_Service(response)), 2.0) is None

    @pytest.mark.parametrize(
        "conn",
        [
            param(_FakeConn(None, install_service=False), id="service-attribute-missing"),
            param(_FakeConn(None), id="service-is-none"),
            param(_FakeConn(SimpleNamespace()), id="send-command-missing"),
            param(
                _FakeConn(SimpleNamespace(send_command_to_controller=None)),
                id="send-command-not-callable-none",
            ),
            param(
                _FakeConn(SimpleNamespace(send_command_to_controller="closed")),
                id="send-command-not-callable-string",
            ),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_query_controller_pod_states_missing_connector_returns_none(
        self, conn: _FakeConn
    ) -> None:
        assert await query_controller_pod_states(conn, timeout=2.0) is None

    @pytest.mark.parametrize(
        "payload",
        [
            param(b"", id="empty-bytes"),
            param(b"not-json", id="invalid-json"),
            param(orjson.dumps(None), id="json-null"),
            param(orjson.dumps(["pod_states"]), id="json-list"),
            param(orjson.dumps("pod_states"), id="json-string"),
            param(orjson.dumps(7), id="json-scalar"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_query_controller_pod_states_malformed_payload_returns_none(
        self, payload: bytes
    ) -> None:
        service = _Service(
            CommandOk(
                cid="cid-aiperf-bench-7f2a",
                sid="system_controller",
                payload=payload,
            )
        )

        assert await query_controller_pod_states(_FakeConn(service), timeout=2.0) is None


# ============================================================================
# Router fallback and partial-snapshot behavior
# ============================================================================


class TestPodStateRoutersAdversarialPayloads:
    """Debug/progress endpoints must not let malformed controller data become 500s."""

    def test_debug_pod_states_missing_pod_states_key_returns_empty_controller_snapshot(
        self, debug_client: TestClient
    ) -> None:
        debug_client.app.state.service = _Service(
            _ok({"worker_startup_states": {"worker-0": "ready"}})
        )

        response = debug_client.get("/api/debug/pod-states")

        assert response.status_code == 200
        assert response.json()["source"] == "controller"
        assert response.json()["pod_count"] == 0
        assert response.json()["pods"] == {}

    def test_debug_worker_startup_states_missing_startup_key_returns_empty_controller_snapshot(
        self, debug_client: TestClient
    ) -> None:
        debug_client.app.state.service = _Service(_ok({"pod_states": {}}))

        response = debug_client.get("/api/debug/worker-startup-states")

        assert response.status_code == 200
        assert response.json()["source"] == "controller"
        assert response.json()["worker_count"] == 0
        assert response.json()["workers"] == {}

    @pytest.mark.asyncio
    async def test_debug_pod_states_controller_preserves_namespace_like_pod_keys(
        self, debug_client: TestClient, debug_router: DebugRouter
    ) -> None:
        await debug_router._on_worker_pod_state(_pod("cache-only", declared=2, ready=2))
        controller_key = "bench-ns/aiperf-bench-7f2a:pod-0"
        debug_client.app.state.service = _Service(
            _ok(
                {
                    "pod_states": {controller_key: _pod(controller_key).model_dump()},
                    "worker_startup_states": {},
                }
            )
        )

        response = debug_client.get("/api/debug/pod-states")

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "controller"
        assert set(data["pods"].keys()) == {controller_key}
        assert "cache-only" not in data["pods"]

    @pytest.mark.asyncio
    async def test_progress_malformed_controller_payload_falls_back_to_cache(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        await progress_router._on_worker_pod_state(_pod("0", declared=2, ready=2))
        progress_client.app.state.service = _Service(_ok(["not", "a", "snapshot"]))

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        assert response.json()["workers"]["ready"] == 2
        assert response.json()["workers"]["total"] == 2

    @pytest.mark.asyncio
    async def test_progress_partial_pod_state_from_controller_skips_bad_entry(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        await progress_router._on_worker_pod_state(_pod("cache", declared=1, ready=1))
        progress_client.app.state.service = _Service(
            _ok(
                {
                    "pod_states": {
                        "0": _pod("0", declared=4, ready=4).model_dump(),
                        "1": {"pod_index": "1", "pod_state": "starting"},
                    },
                    "worker_startup_states": {},
                }
            )
        )

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        assert response.json()["workers"]["ready"] == 4
        assert response.json()["workers"]["total"] == 4

    def test_progress_inprocess_controller_takes_precedence_over_malformed_rpc(
        self, progress_client: TestClient
    ) -> None:
        progress_client.app.state.controller = _Controller()
        progress_client.app.state.service = _Service(_ok(["malformed", "sidecar", "rpc"]))

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        assert response.json()["workers"]["ready"] == 9
        assert response.json()["workers"]["total_pods"] == 3


# ============================================================================
# Kubernetes annotation mirror name/namespace pass-through
# ============================================================================


class TestProgressPatchNames:
    """Patch helpers must pass the exact Kubernetes name/namespace strings."""

    @pytest.mark.parametrize(
        "helper_name,expected_name",
        [
            ("_patch_jobset_annotations", "aiperf-aiperf-bench-7f2a.20260518"),
            ("_patch_aiperfjob_annotations", "aiperf-bench-7f2a.20260518"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_patch_annotations_preserves_namespace_and_resource_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        helper_name: str,
        expected_name: str,
    ) -> None:
        import kubernetes_asyncio
        import aiperf.api.routers.progress as progress_mod
        import aiperf.kubernetes.client as kclient

        custom = MagicMock()
        custom.patch_namespaced_custom_object = AsyncMock()
        monkeypatch.setattr(
            kubernetes_asyncio,
            "client",
            SimpleNamespace(CustomObjectsApi=lambda _api: custom),
            raising=False,
        )

        @asynccontextmanager
        async def fake_k8s_client() -> AsyncGenerator[MagicMock, None]:
            yield MagicMock(name="ApiClient")

        monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

        helper = getattr(progress_mod, helper_name)
        await helper(
            job_id="aiperf-bench-7f2a.20260518",
            namespace="team-aiperf-prod",
            annotations={"aiperf.nvidia.com/system-state": "profiling"},
        )

        kwargs = custom.patch_namespaced_custom_object.call_args.kwargs
        assert kwargs["namespace"] == "team-aiperf-prod"
        assert kwargs["name"] == expected_name
        assert kwargs["_content_type"] == "application/merge-patch+json"
