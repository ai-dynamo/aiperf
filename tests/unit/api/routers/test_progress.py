# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProgressRouter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.progress import ProgressRouter
from aiperf.common.messages import WorkerPodStateMessage
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
