# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DebugRouter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.debug import DebugRouter
from aiperf.common.enums import WorkerStartupState
from aiperf.common.messages import WorkerPodStateMessage
from aiperf.config import AIPerfConfig


@pytest.fixture
def debug_router(mock_zmq, router_config: AIPerfConfig) -> DebugRouter:
    return DebugRouter(run=router_config)


def _make_client(
    debug_router: DebugRouter,
    *,
    pod_states: dict[str, WorkerPodStateMessage] | None = None,
    worker_startup_states: dict[str, str] | None = None,
) -> TestClient:
    app = FastAPI()
    app.state.debug = debug_router
    if pod_states is not None or worker_startup_states is not None:

        class FakeController:
            _pod_states = pod_states or {}
            _worker_startup_states = worker_startup_states or {}

        app.state.controller = FakeController()
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
    """Test /api/debug/pod-states."""

    def test_returns_empty_when_controller_missing(
        self, debug_router: DebugRouter
    ) -> None:
        client = _make_client(debug_router)
        response = client.get("/api/debug/pod-states")
        assert response.status_code == 200
        data = response.json()
        assert data["pod_count"] == 0
        assert data["pods"] == {}

    def test_returns_per_pod_payload(self, debug_router: DebugRouter) -> None:
        pod_states = {
            "0": _pod("0", declared=4, ready=4),
            "1": _pod("1", declared=4, ready=2),
        }
        client = _make_client(debug_router, pod_states=pod_states)
        data = client.get("/api/debug/pod-states").json()
        assert data["pod_count"] == 2
        assert set(data["pods"].keys()) == {"0", "1"}
        assert data["pods"]["0"]["ready_workers"] == 4
        assert data["pods"]["1"]["ready_workers"] == 2
        assert data["pods"]["1"]["degraded_workers"] == 2

    def test_zero_pods_signals_no_messages_received(
        self, debug_router: DebugRouter
    ) -> None:
        client = _make_client(debug_router, pod_states={})
        data = client.get("/api/debug/pod-states").json()
        assert data["pod_count"] == 0
        assert data["pods"] == {}


class TestWorkerStartupStatesEndpoint:
    """Test /api/debug/worker-startup-states."""

    def test_returns_empty_when_controller_missing(
        self, debug_router: DebugRouter
    ) -> None:
        client = _make_client(debug_router)
        data = client.get("/api/debug/worker-startup-states").json()
        assert data["worker_count"] == 0
        assert data["ready_count"] == 0
        assert data["workers"] == {}

    def test_counts_ready_workers(self, debug_router: DebugRouter) -> None:
        states = {
            "w-0": str(WorkerStartupState.READY),
            "w-1": str(WorkerStartupState.READY),
            "w-2": str(WorkerStartupState.WAITING_FOR_DATASET),
            "w-3": str(WorkerStartupState.ROUTER_PROBING),
        }
        client = _make_client(debug_router, worker_startup_states=states)
        data = client.get("/api/debug/worker-startup-states").json()
        assert data["worker_count"] == 4
        assert data["ready_count"] == 2
        assert data["workers"]["w-2"] == str(WorkerStartupState.WAITING_FOR_DATASET)

    def test_zero_ready_with_workers_present_signals_stuck_startup(
        self, debug_router: DebugRouter
    ) -> None:
        states = {
            "w-0": str(WorkerStartupState.WAITING_FOR_DATASET),
            "w-1": str(WorkerStartupState.WAITING_FOR_DATASET),
        }
        client = _make_client(debug_router, worker_startup_states=states)
        data = client.get("/api/debug/worker-startup-states").json()
        assert data["worker_count"] == 2
        assert data["ready_count"] == 0
