# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MetricsRouter."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from aiperf.api.routers.metrics import MetricsRouter
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.messages import RealtimeMetricsMessage
from aiperf.common.models import MetricResult
from tests.unit.api.routers.conftest import make_latency_metric


def make_metric_result(
    tag: str = "test_metric",
    header: str = "Test Metric",
    unit: str = "ms",
    avg: float | None = None,
    min: float | None = None,
    max: float | None = None,
    sum: float | None = None,
    p50: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    std: float | None = None,
    **kwargs,
) -> MetricResult:
    """Create a MetricResult with sensible defaults."""
    return MetricResult(
        tag=tag,
        header=header,
        unit=unit,
        avg=avg,
        min=min,
        max=max,
        sum=sum,
        p50=p50,
        p95=p95,
        p99=p99,
        std=std,
        **kwargs,
    )


@pytest.fixture
def metrics_router(
    mock_zmq, router_service_config: ServiceConfig, router_user_config: UserConfig
) -> MetricsRouter:
    """Create a MetricsRouter for testing."""
    return MetricsRouter(
        service_config=router_service_config,
        user_config=router_user_config,
    )


@pytest.fixture
def metrics_client(metrics_router: MetricsRouter) -> TestClient:
    """Create a TestClient wired to the metrics router."""
    app = FastAPI()
    app.state.metrics = metrics_router
    app.include_router(metrics_router.get_router())
    return TestClient(app)


class TestPrometheusMetricsEndpoint:
    """Test the /metrics endpoint."""

    def test_empty_metrics(
        self, metrics_client: TestClient, metrics_router: MetricsRouter
    ) -> None:
        metrics_router._metrics = []
        response = metrics_client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_with_metrics(
        self, metrics_client: TestClient, metrics_router: MetricsRouter
    ) -> None:
        metrics_router._metrics = [make_latency_metric(avg=100.0)]
        response = metrics_client.get("/metrics")
        assert response.status_code == 200
        assert "aiperf_latency" in response.text


class TestJsonMetricsEndpoint:
    """Test the /api/metrics endpoint."""

    def test_empty_metrics(
        self, metrics_client: TestClient, metrics_router: MetricsRouter
    ) -> None:
        metrics_router._metrics = []
        response = metrics_client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"] == {}

    def test_with_data(
        self, metrics_client: TestClient, metrics_router: MetricsRouter
    ) -> None:
        metrics_router._metrics = [make_latency_metric(avg=100.0)]
        response = metrics_client.get("/api/metrics")
        data = response.json()
        assert data["metrics"]["latency"]["avg"] == 100.0

    def test_multiple_metrics(
        self, metrics_client: TestClient, metrics_router: MetricsRouter
    ) -> None:
        metrics_router._metrics = [
            make_latency_metric(avg=100.0),
            make_metric_result(
                tag="throughput", header="Throughput", unit="req/s", avg=50.0
            ),
        ]
        response = metrics_client.get("/api/metrics")
        data = response.json()
        assert "latency" in data["metrics"]
        assert "throughput" in data["metrics"]


class TestInfoLabelsCache:
    """Test the info labels caching behavior."""

    def test_get_info_labels_creates_and_caches(
        self, metrics_router: MetricsRouter
    ) -> None:
        assert metrics_router._info_labels is None

        labels1 = metrics_router.get_info_labels()
        assert labels1 is not None
        assert metrics_router._info_labels is not None

        labels2 = metrics_router.get_info_labels()
        assert labels1 is labels2


class TestRealtimeMetricsHandler:
    """Test the @on_message handler from RealtimeMetricsMixin."""

    @pytest.mark.asyncio
    async def test_on_realtime_metrics_updates_state(
        self, metrics_router: MetricsRouter
    ) -> None:
        metrics_router.run_hooks = AsyncMock()

        metric = make_latency_metric(avg=42.0)
        message = RealtimeMetricsMessage(service_id="test", metrics=[metric])
        await metrics_router._on_realtime_metrics(message)

        assert len(metrics_router._metrics) == 1
        assert metrics_router._metrics[0].avg == 42.0
