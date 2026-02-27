# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for WorkersRouter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.routers.workers import WorkersRouter
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import WorkerStatus
from aiperf.common.models import WorkerStats


@pytest.fixture
def workers_router(
    mock_zmq, router_service_config: ServiceConfig, router_user_config: UserConfig
) -> WorkersRouter:
    return WorkersRouter(
        service_config=router_service_config,
        user_config=router_user_config,
    )


@pytest.fixture
def workers_client(workers_router: WorkersRouter) -> TestClient:
    app = FastAPI()
    app.state.workers = workers_router
    app.include_router(workers_router.get_router())
    return TestClient(app)


class TestWorkersEndpoint:
    """Test the /api/workers endpoint."""

    def test_workers_empty(self, workers_client: TestClient) -> None:
        response = workers_client.get("/api/workers")
        assert response.status_code == 200
        data = response.json()
        assert data["workers"] == {}

    @pytest.mark.parametrize(
        "statuses,expected_active",
        [
            param([WorkerStatus.HEALTHY], 1, id="one-healthy"),
            param([WorkerStatus.IDLE], 0, id="one-idle"),
            param([WorkerStatus.HIGH_LOAD], 1, id="one-high-load"),
            param([WorkerStatus.ERROR], 0, id="one-error"),
            param([WorkerStatus.STALE], 0, id="one-stale"),
            param([WorkerStatus.HEALTHY, WorkerStatus.HEALTHY], 2, id="two-healthy"),
            param([WorkerStatus.HEALTHY, WorkerStatus.IDLE], 1, id="one-healthy-one-idle"),
            param([WorkerStatus.HIGH_LOAD, WorkerStatus.HEALTHY], 2, id="high-load-and-healthy"),
        ],
    )  # fmt: skip
    def test_workers_active_count(
        self,
        workers_client: TestClient,
        workers_router: WorkersRouter,
        statuses: list[WorkerStatus],
        expected_active: int,
    ) -> None:
        workers_router._worker_tracker._workers_stats = {
            f"worker-{i}": WorkerStats(worker_id=f"worker-{i}", status=status)
            for i, status in enumerate(statuses)
        }
        response = workers_client.get("/api/workers")
        data = response.json()
        assert len(data["workers"]) == len(statuses)
        active = sum(
            1
            for w in data["workers"].values()
            if w["status"] in (WorkerStatus.HEALTHY, WorkerStatus.HIGH_LOAD)
        )
        assert active == expected_active
