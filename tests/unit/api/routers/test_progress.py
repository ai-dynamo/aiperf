# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProgressRouter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.progress import ProgressRouter
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CreditPhase
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats


@pytest.fixture
def progress_router(
    mock_zmq, router_service_config: ServiceConfig, router_user_config: UserConfig
) -> ProgressRouter:
    return ProgressRouter(
        service_config=router_service_config,
        user_config=router_user_config,
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
            CreditPhase.WARMUP: CombinedPhaseStats(
                phase=CreditPhase.WARMUP,
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
