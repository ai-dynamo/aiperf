# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DebugRouter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.debug import DebugRouter
from aiperf.api.routers.progress import ProgressRouter
from aiperf.common.enums import WorkerStartupState
from aiperf.common.messages import (
    CommandSuccessResponse,
    GetPodStatesCommand,
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
    the typed command facade."""

    class _FakeService:
        service_id = "api-service"

        async def send_command_and_wait_for_response(
            self, command: GetPodStatesCommand, timeout: float
        ) -> object:
            assert isinstance(command, GetPodStatesCommand)
            assert timeout > 0
            return response

    return _FakeService()


def _success_response(
    pods: dict[str, dict], startup: dict[str, str]
) -> CommandSuccessResponse:
    """Build the typed response produced by the controller command handler."""
    command = GetPodStatesCommand(service_id="api-service")
    return CommandSuccessResponse.from_command_message(
        command,
        "system-controller",
        data={"pod_states": pods, "worker_startup_states": startup},
    )


class TestDebugRouterControllerQuery:
    """The controller query is authoritative; the bus mirror is fallback."""

    def test_pod_states_prefers_controller_for_late_subscriber(
        self, debug_client: TestClient
    ) -> None:
        pods = {"0": _pod("0", declared=4, ready=4).model_dump(mode="json")}
        debug_client.app.state.service = _service_with_controller_response(
            _success_response(pods, {"worker-0": "ready"})
        )

        data = debug_client.get("/api/debug/pod-states").json()

        assert data["source"] == "controller"
        assert data["pod_count"] == 1
        assert data["pods"]["0"]["ready_workers"] == 4

    def test_worker_startup_states_prefers_controller(
        self, debug_client: TestClient
    ) -> None:
        debug_client.app.state.service = _service_with_controller_response(
            _success_response(
                {}, {"worker-0": "ready", "worker-1": "waiting_for_dataset"}
            )
        )

        data = debug_client.get("/api/debug/worker-startup-states").json()

        assert data["source"] == "controller"
        assert data["worker_count"] == 2
        assert data["ready_count"] == 1

    def test_pod_states_falls_back_when_no_service_in_app_state(
        self, debug_client: TestClient
    ) -> None:
        # No service => no RPC path. Empty cache => empty cache response.
        data = debug_client.get("/api/debug/pod-states").json()
        assert data["source"] == "cache"
        assert data["pod_count"] == 0

    @pytest.mark.asyncio
    async def test_pod_states_falls_back_when_query_raises(
        self, debug_client: TestClient, debug_router: DebugRouter
    ) -> None:
        await debug_router._on_worker_pod_state(_pod("0", declared=4, ready=2))

        class _UnavailableService:
            service_id = "api-service"

            async def send_command_and_wait_for_response(
                self, _command: GetPodStatesCommand, timeout: float
            ) -> object:
                raise RuntimeError(f"controller unavailable after {timeout}s")

        debug_client.app.state.service = _UnavailableService()

        data = debug_client.get("/api/debug/pod-states").json()

        assert data["source"] == "cache"
        assert data["pods"]["0"]["ready_workers"] == 2


@pytest.mark.asyncio
async def test_debug_and_progress_share_authoritative_snapshot(
    mock_zmq, router_config: AIPerfConfig
) -> None:
    """Both API views must report the controller snapshot, not divergent caches."""
    debug = DebugRouter(run=router_config)
    progress = ProgressRouter(run=router_config)
    await debug._on_worker_pod_state(_pod("stale", declared=9, ready=1))
    await progress._on_worker_pod_state(_pod("stale", declared=7, ready=2))
    authoritative = _pod("0", declared=4, ready=3)

    app = FastAPI()
    app.state.debug = debug
    app.state.progress = progress
    app.state.service = _service_with_controller_response(
        _success_response(
            {"0": authoritative.model_dump(mode="json")},
            {"worker-0": "ready"},
        )
    )
    app.include_router(debug.get_router())
    app.include_router(progress.get_router())
    client = TestClient(app)

    debug_data = client.get("/api/debug/pod-states").json()
    progress_data = client.get("/api/progress").json()

    assert debug_data["source"] == "controller"
    assert debug_data["pods"]["0"]["ready_workers"] == 3
    assert progress_data["workers"]["ready"] == 3
    assert progress_data["workers"]["total"] == 4
