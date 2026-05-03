# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProgressRouter."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.progress import ProgressRouter
from aiperf.common.enums import SystemState
from aiperf.common.messages import SystemStateChangedMessage, WorkerPodStateMessage
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.config import AIPerfConfig
from aiperf.controller.system_controller import (
    AggregateWorkerStatus,
    build_aggregate_worker_status,
)


@pytest.fixture
def progress_router(mock_zmq, router_config: AIPerfConfig) -> ProgressRouter:
    return ProgressRouter(
        run=router_config,
    )


@pytest.fixture
def progress_client(progress_router: ProgressRouter) -> TestClient:
    app = FastAPI()
    app.state.progress = progress_router
    app.include_router(progress_router.get_router())
    return TestClient(app)


class TestProgressEndpoint:
    """Test the /api/progress endpoint."""

    def test_progress_empty(self, progress_client: TestClient) -> None:
        response = progress_client.get("/api/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["phases"] == {}

    def test_progress_with_phases(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        progress_router._progress_tracker._phases = {
            "warmup": CombinedPhaseStats(
                phase="warmup",
                total_expected_requests=100,
                requests_completed=50,
                start_ns=1000,
                last_update_ns=2000,
            )
        }
        response = progress_client.get("/api/progress")
        data = response.json()
        assert "warmup" in data["phases"]
        warmup = data["phases"]["warmup"]
        assert warmup["total_expected_requests"] == 100
        assert warmup["requests_completed"] == 50

    def test_progress_includes_aggregate_worker_status(
        self, progress_client: TestClient
    ) -> None:
        class FakeController:
            def get_aggregate_worker_status(self) -> AggregateWorkerStatus:
                return AggregateWorkerStatus(
                    ready=1,
                    total=2,
                    dispatchable=1,
                    router_connected=2,
                    ready_record_processors=1,
                    declared_record_processors=1,
                    ready_pods=1,
                    total_pods=1,
                    degraded_pods=0,
                )

        progress_client.app.state.service = SimpleNamespace(controller=FakeController())
        response = progress_client.get("/api/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["workers"] == {
            "ready": 1,
            "total": 2,
            "dispatchable": 1,
            "router_connected": 2,
            "ready_record_processors": 1,
            "declared_record_processors": 1,
            "ready_pods": 1,
            "total_pods": 1,
            "degraded_pods": 0,
        }

    def test_progress_falls_back_to_bus_cache_when_no_inprocess_controller(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        """K8s mode: API runs in a separate container, so app.state.controller
        is None. ProgressRouter must serve workers from its own bus-fed
        cache instead of returning all zeros."""
        # Simulate two WorkerGroupManagers publishing pod state on the bus.
        import asyncio

        async def feed() -> None:
            await progress_router._on_worker_pod_state(
                WorkerPodStateMessage(
                    service_id="wpm-0",
                    pod_index="0",
                    benchmark_generation="g",
                    dataset_generation="d",
                    declared_workers=4,
                    declared_record_processors=1,
                    router_connected_workers=4,
                    dispatchable_workers=4,
                    ready_workers=4,
                    ready_record_processors=1,
                    degraded_workers=0,
                    degraded_record_processors=0,
                    pod_state="ready",
                    admission_state="dispatchable",
                )
            )
            await progress_router._on_worker_pod_state(
                WorkerPodStateMessage(
                    service_id="wpm-1",
                    pod_index="1",
                    benchmark_generation="g",
                    dataset_generation="d",
                    declared_workers=4,
                    declared_record_processors=1,
                    router_connected_workers=4,
                    dispatchable_workers=2,
                    ready_workers=2,
                    ready_record_processors=1,
                    degraded_workers=2,
                    degraded_record_processors=0,
                    pod_state="ready",
                    admission_state="dispatchable",
                )
            )

        asyncio.get_event_loop().run_until_complete(feed())
        # No app.state.controller AND no app.state.service.controller.
        data = progress_client.get("/api/progress").json()
        assert data["workers"]["ready"] == 6
        assert data["workers"]["total"] == 8
        assert data["workers"]["ready_pods"] == 2
        assert data["workers"]["total_pods"] == 2

    def test_progress_inprocess_controller_takes_precedence_over_bus_cache(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        """When both an in-process controller and a bus-fed cache are available,
        the controller wins. Local/single-process mode keeps the existing
        contract (controller owns the canonical view) instead of double-counting
        from its own mirror."""
        import asyncio

        async def feed_bus_cache() -> None:
            await progress_router._on_worker_pod_state(
                WorkerPodStateMessage(
                    service_id="wpm-bus",
                    pod_index="0",
                    benchmark_generation="g",
                    dataset_generation="d",
                    declared_workers=8,
                    declared_record_processors=1,
                    router_connected_workers=6,
                    dispatchable_workers=6,
                    ready_workers=6,
                    ready_record_processors=1,
                    degraded_workers=2,
                    degraded_record_processors=0,
                    pod_state="ready",
                    admission_state="dispatchable",
                )
            )

        asyncio.get_event_loop().run_until_complete(feed_bus_cache())

        class FakeController:
            def get_aggregate_worker_status(self) -> AggregateWorkerStatus:
                return AggregateWorkerStatus(ready=1, total=2)

        progress_client.app.state.service = SimpleNamespace(controller=FakeController())
        data = progress_client.get("/api/progress").json()
        assert data["workers"]["ready"] == 1
        assert data["workers"]["total"] == 2

    def test_progress_returns_zeros_when_no_controller_and_no_bus_messages(
        self, progress_client: TestClient
    ) -> None:
        """No in-process controller, no pod-state messages received yet —
        legitimate startup window before any WorkerGroupManager reported."""
        data = progress_client.get("/api/progress").json()
        assert data["workers"]["ready"] == 0
        assert data["workers"]["total"] == 0
        assert data["workers"]["total_pods"] == 0

    def test_build_aggregate_worker_status_mixed_states(self) -> None:
        aggregate = build_aggregate_worker_status(
            {
                "0": WorkerPodStateMessage(
                    service_id="worker-pod-manager-0",
                    pod_index="0",
                    benchmark_generation="bench-1",
                    dataset_generation="data-1",
                    declared_workers=2,
                    declared_record_processors=1,
                    router_connected_workers=2,
                    dispatchable_workers=1,
                    ready_workers=1,
                    ready_record_processors=1,
                    degraded_workers=0,
                    degraded_record_processors=0,
                    pod_state="ready",
                    admission_state="dispatchable",
                ),
                "1": WorkerPodStateMessage(
                    service_id="worker-pod-manager-1",
                    pod_index="1",
                    benchmark_generation="bench-1",
                    dataset_generation="data-1",
                    declared_workers=3,
                    declared_record_processors=2,
                    router_connected_workers=1,
                    dispatchable_workers=0,
                    ready_workers=0,
                    ready_record_processors=0,
                    degraded_workers=2,
                    degraded_record_processors=1,
                    pod_state="degraded",
                    admission_state="blocked_waiting_for_dataset",
                ),
            }
        )

        assert aggregate == AggregateWorkerStatus(
            ready=1,
            total=5,
            dispatchable=1,
            router_connected=3,
            ready_record_processors=1,
            declared_record_processors=3,
            ready_pods=1,
            total_pods=2,
            degraded_pods=0,
        )

    def test_build_aggregate_worker_status_counts_only_usable_degraded_pods(
        self,
    ) -> None:
        aggregate = build_aggregate_worker_status(
            {
                "0": WorkerPodStateMessage(
                    service_id="worker-pod-manager-0",
                    pod_index="0",
                    benchmark_generation="bench-1",
                    dataset_generation="data-1",
                    declared_workers=2,
                    declared_record_processors=1,
                    router_connected_workers=2,
                    dispatchable_workers=1,
                    ready_workers=1,
                    ready_record_processors=1,
                    degraded_workers=1,
                    degraded_record_processors=0,
                    pod_state="degraded",
                    admission_state="dispatchable",
                )
            }
        )

        assert aggregate.degraded_pods == 1


class TestProgressRouterSystemState:
    """Tests for SYSTEM_STATE_CHANGED handling and system_state on /api/progress."""

    def test_default_system_state_is_initializing(
        self, progress_router: ProgressRouter
    ) -> None:
        assert progress_router._system_state == SystemState.INITIALIZING

    @pytest.mark.asyncio
    async def test_on_system_state_changed_updates_attribute(
        self, progress_router: ProgressRouter
    ) -> None:
        await progress_router._on_system_state_changed(
            SystemStateChangedMessage(
                service_id="system_controller",
                state=SystemState.PROFILING,
            )
        )
        assert progress_router._system_state == SystemState.PROFILING

    def test_progress_response_initializes_system_state_initializing(
        self, progress_client: TestClient
    ) -> None:
        data = progress_client.get("/api/progress").json()
        assert data["system_state"] == SystemState.INITIALIZING.value

    def test_progress_response_reflects_latest_system_state(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        import asyncio

        async def feed() -> None:
            for state in (
                SystemState.CONFIGURING,
                SystemState.READY,
                SystemState.PROFILING,
            ):
                await progress_router._on_system_state_changed(
                    SystemStateChangedMessage(
                        service_id="system_controller", state=state
                    )
                )

        asyncio.get_event_loop().run_until_complete(feed())
        data = progress_client.get("/api/progress").json()
        assert data["system_state"] == SystemState.PROFILING.value


@pytest.mark.asyncio
async def test_patch_jobset_annotations_uses_merge_patch_content_type(
    monkeypatch,
) -> None:
    from contextlib import asynccontextmanager

    import kubernetes_asyncio

    custom = MagicMock()
    custom.patch_namespaced_custom_object = AsyncMock()
    monkeypatch.setattr(
        kubernetes_asyncio,
        "client",
        SimpleNamespace(CustomObjectsApi=lambda _api: custom),
        raising=False,
    )

    @asynccontextmanager
    async def fake_k8s_client():
        yield MagicMock(name="ApiClient")

    import aiperf.kubernetes.client as kclient

    monkeypatch.setattr(kclient, "k8s_client", fake_k8s_client)

    from aiperf.api.routers.progress import _patch_jobset_annotations

    await _patch_jobset_annotations(
        job_id="job-1",
        namespace="ns",
        annotations={"k": "v"},
    )

    kwargs = custom.patch_namespaced_custom_object.call_args.kwargs
    assert kwargs["body"] == {"metadata": {"annotations": {"k": "v"}}}
    assert kwargs["_content_type"] == "application/merge-patch+json"


class TestProgressEndpointControllerRPC:
    """Verifies /api/progress prefers the GET_POD_STATES RPC over the cache."""

    def _ok_payload(self, pods: dict[str, dict]) -> object:
        import orjson

        from aiperf.common.control_structs import CommandOk

        return CommandOk(
            cid="cid-1",
            sid="system_controller",
            payload=orjson.dumps({"pod_states": pods, "worker_startup_states": {}}),
        )

    def test_progress_uses_controller_rpc_when_available(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        from aiperf.common.enums import CommandType

        # Cache deliberately empty — the only way the response can be
        # non-zero is if the RPC path runs and decodes the payload.
        controller_pods = {
            "0": WorkerPodStateMessage(
                service_id="wpm-0",
                pod_index="0",
                benchmark_generation="g",
                dataset_generation="d",
                declared_workers=4,
                declared_record_processors=1,
                router_connected_workers=4,
                dispatchable_workers=4,
                ready_workers=4,
                ready_record_processors=1,
                degraded_workers=0,
                degraded_record_processors=0,
                pod_state="ready",
                admission_state="dispatchable",
            ).model_dump(),
        }
        ok = self._ok_payload(controller_pods)

        class _FakeService:
            async def send_command_to_controller(
                self, cmd: str, timeout: float = 2.0
            ) -> object:
                assert cmd == CommandType.GET_POD_STATES
                return ok

        progress_client.app.state.service = _FakeService()
        data = progress_client.get("/api/progress").json()
        assert data["workers"]["ready"] == 4
        assert data["workers"]["total"] == 4
        assert data["workers"]["ready_pods"] == 1
        assert data["workers"]["total_pods"] == 1

    @pytest.mark.asyncio
    async def test_progress_falls_back_to_cache_when_rpc_fails(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        # Seed cache with a known shape; RPC raises → fallback.
        await progress_router._on_worker_pod_state(
            WorkerPodStateMessage(
                service_id="wpm-cache",
                pod_index="0",
                benchmark_generation="g",
                dataset_generation="d",
                declared_workers=2,
                declared_record_processors=1,
                router_connected_workers=2,
                dispatchable_workers=2,
                ready_workers=2,
                ready_record_processors=1,
                degraded_workers=0,
                degraded_record_processors=0,
                pod_state="ready",
                admission_state="dispatchable",
            )
        )

        class _Boom:
            async def send_command_to_controller(self, *_args, **_kw):
                raise RuntimeError("control client not initialized")

        progress_client.app.state.service = _Boom()
        data = progress_client.get("/api/progress").json()
        # Falls back to cache → 2 ready, 2 total from the seeded message.
        assert data["workers"]["ready"] == 2
        assert data["workers"]["total"] == 2
