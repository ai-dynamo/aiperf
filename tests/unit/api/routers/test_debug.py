# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DebugRouter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.debug import DebugRouter
from aiperf.common.enums import WorkerStartupState
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
