# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ProgressRouter."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.progress import ProgressRouter
from aiperf.common.enums import SystemState
from aiperf.common.messages import SystemStateChangedMessage
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.config import AIPerfConfig


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

    @pytest.mark.asyncio
    async def test_out_of_order_system_state_does_not_regress(
        self, progress_router: ProgressRouter
    ) -> None:
        """A late/duplicate SYSTEM_STATE_CHANGED with a lower rank must not
        walk the reported state backwards."""
        await progress_router._on_system_state_changed(
            SystemStateChangedMessage(
                service_id="system_controller",
                state=SystemState.PROFILING,
            )
        )
        # Late-arriving message for an earlier state (e.g. reordered/duplicated
        # over ZMQ) must be ignored.
        await progress_router._on_system_state_changed(
            SystemStateChangedMessage(
                service_id="system_controller",
                state=SystemState.CONFIGURING,
            )
        )
        assert progress_router._system_state == SystemState.PROFILING

    @pytest.mark.asyncio
    async def test_duplicate_system_state_is_idempotent(
        self, progress_router: ProgressRouter
    ) -> None:
        """A duplicate delivery of the current state is a no-op, not a regression."""
        await progress_router._on_system_state_changed(
            SystemStateChangedMessage(
                service_id="system_controller",
                state=SystemState.PROFILING,
            )
        )
        await progress_router._on_system_state_changed(
            SystemStateChangedMessage(
                service_id="system_controller",
                state=SystemState.PROFILING,
            )
        )
        assert progress_router._system_state == SystemState.PROFILING

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

        asyncio.run(feed())
        data = progress_client.get("/api/progress").json()
        assert data["system_state"] == SystemState.PROFILING.value
