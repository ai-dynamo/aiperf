# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DebugRouter."""

from __future__ import annotations

import orjson
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.debug import DebugRouter
from aiperf.common.control_structs import CommandErr, CommandOk
from aiperf.common.enums import CommandType, WorkerStartupState
from aiperf.common.messages import (
    WorkerPodStateMessage,
    WorkerStartupStateMessage,
)
from aiperf.config import AIPerfConfig


@pytest.fixture
def debug_router(mock_zmq, router_config: AIPerfConfig) -> DebugRouter:
    return DebugRouter(run=router_config)


@pytest.fixture
def debug_client(debug_router: DebugRouter) -> TestClient:
    app = FastAPI()
    app.state.debug = debug_router
    app.include_router(debug_router.get_router())
    return TestClient(app)


def _pod(
    pod_index: str,
    *,
    declared: int,
    ready: int,
    record_processors: int = 1,
) -> WorkerPodStateMessage:
    return WorkerPodStateMessage(
        service_id=f"wpm-{pod_index}",
        pod_index=pod_index,
        benchmark_generation="gen-1",
        dataset_generation="data-1",
        declared_workers=declared,
        declared_record_processors=record_processors,
        router_connected_workers=ready,
        dispatchable_workers=ready,
        ready_workers=ready,
        ready_record_processors=record_processors,
        degraded_workers=max(0, declared - ready),
        degraded_record_processors=0,
        pod_state="ready" if ready >= 1 else "starting",
        admission_state="dispatchable" if ready >= 1 else "admitting",
    )


class TestPodStatesEndpoint:
    """Test /api/debug/pod-states served from the bus-fed cache."""

    def test_returns_empty_before_any_messages(self, debug_client: TestClient) -> None:
        data = debug_client.get("/api/debug/pod-states").json()
        assert data["pod_count"] == 0
        assert data["pods"] == {}

    @pytest.mark.asyncio
    async def test_records_pod_state_message_from_bus(
        self, debug_client: TestClient, debug_router: DebugRouter
    ) -> None:
        await debug_router._on_worker_pod_state(_pod("0", declared=4, ready=4))
        await debug_router._on_worker_pod_state(_pod("1", declared=4, ready=2))
        data = debug_client.get("/api/debug/pod-states").json()
        assert data["pod_count"] == 2
        assert set(data["pods"].keys()) == {"0", "1"}
        assert data["pods"]["0"]["ready_workers"] == 4
        assert data["pods"]["1"]["ready_workers"] == 2
        assert data["pods"]["1"]["degraded_workers"] == 2

    @pytest.mark.asyncio
    async def test_subsequent_message_overwrites_pod_entry(
        self, debug_client: TestClient, debug_router: DebugRouter
    ) -> None:
        await debug_router._on_worker_pod_state(_pod("0", declared=4, ready=1))
        await debug_router._on_worker_pod_state(_pod("0", declared=4, ready=4))
        data = debug_client.get("/api/debug/pod-states").json()
        assert data["pod_count"] == 1
        assert data["pods"]["0"]["ready_workers"] == 4


class TestWorkerStartupStatesEndpoint:
    """Test /api/debug/worker-startup-states served from the bus-fed cache."""

    def test_returns_empty_before_any_messages(self, debug_client: TestClient) -> None:
        data = debug_client.get("/api/debug/worker-startup-states").json()
        assert data["worker_count"] == 0
        assert data["ready_count"] == 0
        assert data["workers"] == {}

    @pytest.mark.asyncio
    async def test_counts_ready_workers(
        self, debug_client: TestClient, debug_router: DebugRouter
    ) -> None:
        for service_id, state in [
            ("w-0", WorkerStartupState.READY),
            ("w-1", WorkerStartupState.READY),
            ("w-2", WorkerStartupState.WAITING_FOR_DATASET),
            ("w-3", WorkerStartupState.ROUTER_PROBING),
        ]:
            await debug_router._on_worker_startup_state(
                WorkerStartupStateMessage(service_id=service_id, startup_state=state)
            )
        data = debug_client.get("/api/debug/worker-startup-states").json()
        assert data["worker_count"] == 4
        assert data["ready_count"] == 2
        assert data["workers"]["w-2"] == str(WorkerStartupState.WAITING_FOR_DATASET)

    @pytest.mark.asyncio
    async def test_zero_ready_with_workers_present_signals_stuck_startup(
        self, debug_client: TestClient, debug_router: DebugRouter
    ) -> None:
        for service_id in ("w-0", "w-1"):
            await debug_router._on_worker_startup_state(
                WorkerStartupStateMessage(
                    service_id=service_id,
                    startup_state=WorkerStartupState.WAITING_FOR_DATASET,
                )
            )
        data = debug_client.get("/api/debug/worker-startup-states").json()
        assert data["worker_count"] == 2
        assert data["ready_count"] == 0


def _service_with_controller_response(
    response: object,
) -> object:
    """Build an ``app.state.service`` stub that returns ``response`` from
    ``send_command_to_controller``."""

    class _FakeService:
        async def send_command_to_controller(
            self, cmd: str, timeout: float = 2.0
        ) -> object:
            assert cmd == CommandType.GET_POD_STATES
            return response

    return _FakeService()


def _ok_payload(pods: dict[str, dict], startup: dict[str, str]) -> CommandOk:
    """Build a CommandOk whose payload mirrors what the controller emits."""
    return CommandOk(
        cid="cid-1",
        sid="system_controller",
        payload=orjson.dumps({"pod_states": pods, "worker_startup_states": startup}),
    )


class TestDebugRouterPrefersControllerRPC:
    """Verifies the new authoritative RPC path is preferred over the cache."""

    def test_pod_states_served_from_controller_rpc(
        self, debug_router: DebugRouter, debug_client: TestClient
    ) -> None:
        # Local cache says nothing; controller says pod_count=1 ready_workers=4.
        controller_pods = {
            "0": _pod("0", declared=4, ready=4).model_dump(),
        }
        debug_client.app.state.service = _service_with_controller_response(
            _ok_payload(controller_pods, {"w-0": "ready"})
        )
        data = debug_client.get("/api/debug/pod-states").json()
        assert data["pod_count"] == 1
        assert data["pods"]["0"]["ready_workers"] == 4
        assert data["source"] == "controller"

    def test_worker_startup_states_served_from_controller_rpc(
        self, debug_router: DebugRouter, debug_client: TestClient
    ) -> None:
        debug_client.app.state.service = _service_with_controller_response(
            _ok_payload({}, {"w-0": "ready", "w-1": "waiting_for_dataset"})
        )
        data = debug_client.get("/api/debug/worker-startup-states").json()
        assert data["worker_count"] == 2
        assert data["ready_count"] == 1
        assert data["source"] == "controller"


class TestDebugRouterFallsBackOnRPCFailure:
    """Verifies cache fallback when the controller is unavailable."""

    @pytest.mark.asyncio
    async def test_pod_states_falls_back_to_cache_on_controller_err(
        self, debug_router: DebugRouter, debug_client: TestClient
    ) -> None:
        await debug_router._on_worker_pod_state(_pod("0", declared=4, ready=2))
        debug_client.app.state.service = _service_with_controller_response(
            CommandErr(cid="cid-1", sid="system_controller", error="boom")
        )
        data = debug_client.get("/api/debug/pod-states").json()
        assert data["source"] == "cache"
        assert data["pods"]["0"]["ready_workers"] == 2

    @pytest.mark.asyncio
    async def test_pod_states_falls_back_when_rpc_raises(
        self, debug_router: DebugRouter, debug_client: TestClient
    ) -> None:
        await debug_router._on_worker_pod_state(_pod("0", declared=4, ready=4))

        class _Boom:
            async def send_command_to_controller(self, *_args, **_kw):
                raise RuntimeError("control client not initialized")

        debug_client.app.state.service = _Boom()
        data = debug_client.get("/api/debug/pod-states").json()
        assert data["source"] == "cache"
        assert data["pod_count"] == 1

    def test_pod_states_falls_back_when_no_service_in_app_state(
        self, debug_client: TestClient
    ) -> None:
        # No service => no RPC path. Empty cache => empty cache response.
        data = debug_client.get("/api/debug/pod-states").json()
        assert data["source"] == "cache"
        assert data["pod_count"] == 0
