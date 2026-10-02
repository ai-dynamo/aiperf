# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsRouter.

Focuses on:
- The /api/server-metrics REST endpoint contract (placeholder shape vs.
  populated shape).
- The @on_message handler stashes the latest payload (without the
  message_type / service_id envelope keys).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.server_metrics import ServerMetricsRouter
from aiperf.common.messages import RealtimeServerMetricsMessage
from aiperf.common.models import (
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
)
from aiperf.config import AIPerfConfig


def make_endpoint_summary(
    endpoint_url: str = "http://localhost:9090/metrics",
) -> ServerMetricsEndpointSummary:
    """Construct a minimally-valid ServerMetricsEndpointSummary."""
    return ServerMetricsEndpointSummary(
        endpoint_url=endpoint_url,
        info=ServerMetricsEndpointInfo(
            total_fetches=10,
            first_fetch_ns=1_000_000_000,
            last_fetch_ns=2_000_000_000,
            avg_fetch_latency_ms=2.5,
            unique_updates=8,
            first_update_ns=1_100_000_000,
            last_update_ns=1_900_000_000,
            duration_seconds=1.0,
            avg_update_interval_ms=125.0,
        ),
        metrics={},
    )


@pytest.fixture
def server_metrics_router(mock_zmq, router_config: AIPerfConfig) -> ServerMetricsRouter:
    """Create a ServerMetricsRouter for testing."""
    return ServerMetricsRouter(run=router_config)


@pytest.fixture
def server_metrics_client(
    server_metrics_router: ServerMetricsRouter,
) -> TestClient:
    """Create a TestClient wired to the server metrics router."""
    app = FastAPI()
    app.state.server_metrics = server_metrics_router
    app.include_router(server_metrics_router.get_router())
    return TestClient(app)


class TestServerMetricsEndpoint:
    """Test the /api/server-metrics endpoint."""

    def test_returns_placeholder_when_no_data_received(
        self,
        server_metrics_client: TestClient,
        server_metrics_router: ServerMetricsRouter,
    ) -> None:
        assert server_metrics_router._latest is None
        response = server_metrics_client.get("/api/server-metrics")

        assert response.status_code == 200
        data = response.json()
        assert data == {
            "endpoint_summaries": {},
            "message": "No server metrics available yet",
        }

    def test_returns_latest_payload_when_populated(
        self,
        server_metrics_client: TestClient,
        server_metrics_router: ServerMetricsRouter,
    ) -> None:
        server_metrics_router._latest = {
            "endpoint_summaries": {"ep1": {"endpoint_url": "ep1"}},
        }
        response = server_metrics_client.get("/api/server-metrics")

        assert response.status_code == 200
        data = response.json()
        assert data == {"endpoint_summaries": {"ep1": {"endpoint_url": "ep1"}}}

    def test_response_is_json_content_type(
        self, server_metrics_client: TestClient
    ) -> None:
        response = server_metrics_client.get("/api/server-metrics")
        assert response.headers["content-type"].startswith("application/json")


class TestServerMetricsRealtimeHandler:
    """Test the @on_message handler that captures real-time metrics."""

    @pytest.mark.asyncio
    async def test_handler_stores_payload_excluding_envelope_fields(
        self, server_metrics_router: ServerMetricsRouter
    ) -> None:
        server_metrics_router.run_hooks = AsyncMock()
        message = RealtimeServerMetricsMessage(
            service_id="test-svc",
            endpoint_summaries={"ep1": make_endpoint_summary("ep1")},
        )

        await server_metrics_router._on_realtime_server_metrics(message)

        latest = server_metrics_router._latest
        assert latest is not None
        # Envelope keys must be stripped — only the payload remains.
        assert "message_type" not in latest
        assert "service_id" not in latest
        assert "endpoint_summaries" in latest
        assert "ep1" in latest["endpoint_summaries"]

    @pytest.mark.asyncio
    async def test_handler_overwrites_previous_payload(
        self, server_metrics_router: ServerMetricsRouter
    ) -> None:
        server_metrics_router.run_hooks = AsyncMock()
        first = RealtimeServerMetricsMessage(
            service_id="svc",
            endpoint_summaries={"ep1": make_endpoint_summary("ep1")},
        )
        second = RealtimeServerMetricsMessage(
            service_id="svc",
            endpoint_summaries={"ep2": make_endpoint_summary("ep2")},
        )

        await server_metrics_router._on_realtime_server_metrics(first)
        await server_metrics_router._on_realtime_server_metrics(second)

        latest = server_metrics_router._latest
        assert latest is not None
        assert set(latest["endpoint_summaries"].keys()) == {"ep2"}

    @pytest.mark.asyncio
    async def test_handler_with_empty_endpoint_summaries(
        self, server_metrics_router: ServerMetricsRouter
    ) -> None:
        server_metrics_router.run_hooks = AsyncMock()
        message = RealtimeServerMetricsMessage(service_id="svc", endpoint_summaries={})

        await server_metrics_router._on_realtime_server_metrics(message)

        assert server_metrics_router._latest == {
            "endpoint_summaries": {},
            "snapshot": {},
        }


class TestServerMetricsRouterConstruction:
    """Test the router construction and APIRouter exposure."""

    def test_get_router_returns_module_router(
        self, server_metrics_router: ServerMetricsRouter
    ) -> None:
        from aiperf.api.routers.server_metrics import (
            server_metrics_router as module_router,
        )

        assert server_metrics_router.get_router() is module_router

    def test_initial_latest_is_none(
        self, server_metrics_router: ServerMetricsRouter
    ) -> None:
        assert server_metrics_router._latest is None
