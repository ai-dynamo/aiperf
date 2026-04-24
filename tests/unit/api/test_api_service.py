# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FastAPI-based API service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import ORJSONResponse
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.api_service import FastAPIService, main
from aiperf.api.routers.static import _read_static
from aiperf.api.routers.websocket import WebSocketManager
from aiperf.common.compression import (
    CompressionEncoding,
    is_zstd_available,
    select_encoding,
)
from aiperf.common.enums import LifecycleState, WorkerStatus
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.common.models import WorkerStats
from aiperf.config import BenchmarkRun
from aiperf.plugin.enums import ServiceType

from .conftest import (
    create_test_app,
    make_latency_metric,
    make_metric_result,
    make_mock_websocket,
    make_process_records_result,
)


class TestWebSocketManager:
    """Test WebSocketManager functionality."""

    def test_add_and_remove_client(self) -> None:
        """Test adding and removing clients."""
        manager = WebSocketManager()
        ws = MagicMock()

        manager.add("client-1", ws)
        assert manager.client_count == 1

        removed_subs = manager.remove("client-1")
        assert manager.client_count == 0
        assert removed_subs == set()

    def test_remove_returns_subscriptions(self) -> None:
        """Test that remove returns the client's subscriptions."""
        manager = WebSocketManager()
        ws = MagicMock()

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["topic_a", "topic_b"])

        removed_subs = manager.remove("client-1")
        assert removed_subs == {"topic_a", "topic_b"}

    def test_remove_nonexistent_client(self) -> None:
        """Test removing a client that doesn't exist."""
        manager = WebSocketManager()
        removed_subs = manager.remove("nonexistent")
        assert removed_subs == set()

    def test_subscribe_and_unsubscribe(self) -> None:
        """Test subscription management."""
        manager = WebSocketManager()
        ws = MagicMock()

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["topic_a", "topic_b"])

        all_subs = manager.all_subscriptions()
        assert "topic_a" in all_subs
        assert "topic_b" in all_subs

        manager.unsubscribe("client-1", ["topic_a"])
        all_subs = manager.all_subscriptions()
        assert "topic_a" not in all_subs
        assert "topic_b" in all_subs

    def test_subscribe_nonexistent_client(self) -> None:
        """Test subscribing for a nonexistent client does nothing."""
        manager = WebSocketManager()
        manager.subscribe("nonexistent", ["topic"])
        assert manager.all_subscriptions() == set()

    def test_unsubscribe_nonexistent_client(self) -> None:
        """Test unsubscribing for a nonexistent client does nothing."""
        manager = WebSocketManager()
        manager.unsubscribe("nonexistent", ["topic"])
        # Should not raise

    def test_all_subscriptions_empty(self) -> None:
        """Test all_subscriptions with no clients."""
        manager = WebSocketManager()
        assert manager.all_subscriptions() == set()

    def test_all_subscriptions_multiple_clients(self) -> None:
        """Test all_subscriptions aggregates across clients."""
        manager = WebSocketManager()
        ws1, ws2 = MagicMock(), MagicMock()

        manager.add("client-1", ws1)
        manager.add("client-2", ws2)
        manager.subscribe("client-1", ["topic_a"])
        manager.subscribe("client-2", ["topic_b", "topic_c"])

        all_subs = manager.all_subscriptions()
        assert all_subs == {"topic_a", "topic_b", "topic_c"}

    def test_all_subscriptions_with_overlap(self) -> None:
        """Test all_subscriptions deduplicates overlapping subscriptions."""
        manager = WebSocketManager()
        ws1, ws2 = MagicMock(), MagicMock()

        manager.add("client-1", ws1)
        manager.add("client-2", ws2)
        manager.subscribe("client-1", ["topic_a", "topic_b"])
        manager.subscribe("client-2", ["topic_b", "topic_c"])

        all_subs = manager.all_subscriptions()
        assert all_subs == {"topic_a", "topic_b", "topic_c"}

    @pytest.mark.asyncio
    async def test_broadcast_to_subscribed_clients(self) -> None:
        """Test broadcasting to subscribed clients."""
        manager = WebSocketManager()
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        manager.add("client-1", ws1)
        manager.add("client-2", ws2)
        manager.subscribe("client-1", ["realtime_metrics"])
        manager.subscribe("client-2", ["other"])

        msg = MagicMock()
        msg.message_type = "realtime_metrics"
        msg.model_dump_json.return_value = '{"data": "test"}'
        sent = await manager.broadcast(msg)
        assert sent == 1
        ws1.send_text.assert_called_once()
        ws2.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_wildcard(self) -> None:
        """Test wildcard subscription receives all messages."""
        manager = WebSocketManager()
        ws = AsyncMock()

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["*"])

        msg = MagicMock()
        msg.message_type = "any_message_type"
        msg.model_dump_json.return_value = '{"data": "test"}'
        sent = await manager.broadcast(msg)
        assert sent == 1
        ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_empty_snapshot(self) -> None:
        """Test broadcast with no clients returns 0."""
        manager = WebSocketManager()
        msg = MagicMock()
        msg.message_type = "topic"
        msg.model_dump_json.return_value = '{"data": "test"}'
        sent = await manager.broadcast(msg)
        assert sent == 0

    @pytest.mark.asyncio
    async def test_broadcast_no_subscribers(self) -> None:
        """Test broadcast when no clients are subscribed to the topic."""
        manager = WebSocketManager()
        ws = AsyncMock()

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["other_topic"])

        msg = MagicMock()
        msg.message_type = "unsubscribed_topic"
        msg.model_dump_json.return_value = '{"data": "test"}'
        sent = await manager.broadcast(msg)
        assert sent == 0
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_client(self) -> None:
        """Test that broadcast removes clients that fail to send."""
        manager = WebSocketManager()
        ws = make_mock_websocket(send_side_effect=Exception("Connection lost"))

        manager.add("client-1", ws)
        manager.subscribe("client-1", ["topic"])

        msg = MagicMock()
        msg.message_type = "topic"
        msg.model_dump_json.return_value = '{"data": "test"}'
        sent = await manager.broadcast(msg)
        assert sent == 0
        assert manager.client_count == 0

    @pytest.mark.asyncio
    async def test_close_all(self) -> None:
        """Test closing all WebSocket connections."""
        from starlette.websockets import WebSocketState

        manager = WebSocketManager()
        ws1, ws2 = AsyncMock(), AsyncMock()
        ws1.client_state = WebSocketState.CONNECTED
        ws2.client_state = WebSocketState.CONNECTED

        manager.add("client-1", ws1)
        manager.add("client-2", ws2)

        await manager.close_all()

        ws1.close.assert_called_once()
        ws2.close.assert_called_once()
        assert manager.client_count == 0

    @pytest.mark.asyncio
    async def test_close_all_empty(self) -> None:
        """Test closing all with no clients does nothing."""
        manager = WebSocketManager()
        await manager.close_all()
        # Should not raise

    @pytest.mark.asyncio
    async def test_close_all_already_disconnected(self) -> None:
        """Test closing connections that are already disconnected."""
        from starlette.websockets import WebSocketState

        manager = WebSocketManager()
        ws = AsyncMock()
        ws.client_state = WebSocketState.DISCONNECTED

        manager.add("client-1", ws)
        await manager.close_all()

        ws.close.assert_not_called()
        assert manager.client_count == 0


class TestOrjsonResponse:
    """Test ORJSONResponse class."""

    def test_render_simple_content(self) -> None:
        """Test rendering simple content."""
        response = ORJSONResponse({"key": "value"})
        body = response.body
        assert b'"key"' in body
        assert b'"value"' in body

    def test_media_type(self) -> None:
        """Test that media type is application/json."""
        response = ORJSONResponse({})
        assert response.media_type == "application/json"


class TestHTTPEndpoints:
    """Test HTTP API endpoints using TestClient."""

    def test_health_returns_ok(self, api_test_client: TestClient) -> None:
        """Test healthz endpoint returns ok."""
        response = api_test_client.get("/healthz")
        assert response.status_code == 200
        assert response.text == "ok"

    @pytest.mark.parametrize(
        "state,expected_code,expected_text",
        [
            param(LifecycleState.RUNNING, 200, "ok", id="running-healthy"),
            param(LifecycleState.INITIALIZING, 200, "ok", id="initializing-healthy"),
            param(LifecycleState.STARTING, 200, "ok", id="starting-healthy"),
            param(LifecycleState.STOPPING, 200, "ok", id="stopping-healthy"),
            param(LifecycleState.STOPPED, 200, "ok", id="stopped-healthy"),
            param(LifecycleState.FAILED, 503, "unhealthy", id="failed-unhealthy"),
        ],
    )  # fmt: skip
    def test_healthz_by_state(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        state: LifecycleState,
        expected_code: int,
        expected_text: str,
    ) -> None:
        """Test K8s liveness probe returns correct status based on lifecycle state."""
        mock_fastapi_service._state = state
        response = api_test_client.get("/healthz")
        assert response.status_code == expected_code
        assert response.text == expected_text

    @pytest.mark.parametrize(
        "state,expected_code,expected_text",
        [
            param(LifecycleState.RUNNING, 200, "ok", id="running-ready"),
            param(LifecycleState.CREATED, 503, "not ready", id="created-not-ready"),
            param(LifecycleState.INITIALIZING, 503, "not ready", id="initializing-not-ready"),
            param(LifecycleState.STARTING, 503, "not ready", id="starting-not-ready"),
            param(LifecycleState.STOPPING, 503, "not ready", id="stopping-not-ready"),
            param(LifecycleState.STOPPED, 503, "not ready", id="stopped-not-ready"),
            param(LifecycleState.FAILED, 503, "not ready", id="failed-not-ready"),
        ],
    )  # fmt: skip
    def test_readyz_by_state(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        state: LifecycleState,
        expected_code: int,
        expected_text: str,
    ) -> None:
        """Test K8s readiness probe returns correct status based on lifecycle state."""
        mock_fastapi_service._state = state
        response = api_test_client.get("/readyz")
        assert response.status_code == expected_code
        assert response.text == expected_text

    def test_config_returns_json(self, api_test_client: TestClient) -> None:
        """Test config endpoint returns JSON config."""
        response = api_test_client.get("/api/config")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "endpoint" in data

    def test_prometheus_empty_metrics(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test Prometheus endpoint with no metrics."""
        mock_fastapi_service._routers["metrics"]._metrics = []
        response = api_test_client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

    def test_prometheus_with_metrics(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test Prometheus endpoint with metrics."""
        mock_fastapi_service._routers["metrics"]._metrics = [
            make_latency_metric(avg=100.0)
        ]
        response = api_test_client.get("/metrics")
        assert response.status_code == 200
        assert "aiperf_latency" in response.text

    def test_json_metrics_empty(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test JSON metrics endpoint with no metrics."""
        mock_fastapi_service._routers["metrics"]._metrics = []
        response = api_test_client.get("/api/metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"] == {}

    def test_json_metrics_with_data(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test JSON metrics endpoint with metrics."""
        mock_fastapi_service._routers["metrics"]._metrics = [
            make_latency_metric(avg=100.0)
        ]
        response = api_test_client.get("/api/metrics")
        data = response.json()
        assert data["metrics"]["latency"]["avg"] == 100.0

    def test_json_metrics_multiple(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test JSON metrics endpoint with multiple metrics."""
        mock_fastapi_service._routers["metrics"]._metrics = [
            make_latency_metric(avg=100.0),
            make_metric_result(
                tag="throughput", header="Throughput", unit="req/s", avg=50.0
            ),
        ]
        response = api_test_client.get("/api/metrics")
        data = response.json()
        assert "latency" in data["metrics"]
        assert "throughput" in data["metrics"]

    def test_progress_empty(self, api_test_client: TestClient) -> None:
        """Test progress endpoint with no progress data."""
        response = api_test_client.get("/api/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["phases"] == {}

    def test_progress_with_phases(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test progress endpoint with phase data."""
        mock_fastapi_service._routers["progress"]._progress_tracker._phases = {
            "warmup": CombinedPhaseStats(
                phase="warmup",
                total_expected_requests=100,
                requests_completed=50,
                start_ns=1000,
                last_update_ns=2000,
            )
        }
        response = api_test_client.get("/api/progress")
        data = response.json()
        assert "warmup" in data["phases"]
        assert data["phases"]["warmup"]["total_expected_requests"] == 100
        assert data["phases"]["warmup"]["requests_completed"] == 50

    def test_workers_empty(self, api_test_client: TestClient) -> None:
        """Test workers endpoint with no workers."""
        response = api_test_client.get("/api/workers")
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
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        statuses: list[WorkerStatus],
        expected_active: int,
    ) -> None:
        """Test workers endpoint returns correct worker statuses."""
        mock_fastapi_service._routers["workers"]._worker_tracker._workers_stats = {
            f"worker-{i}": WorkerStats(worker_id=f"worker-{i}", status=status)
            for i, status in enumerate(statuses)
        }
        response = api_test_client.get("/api/workers")
        data = response.json()
        assert len(data["workers"]) == len(statuses)

    @pytest.mark.skip(
        reason="/api/server-metrics not yet migrated to router-based architecture"
    )
    def test_server_metrics_empty(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test server metrics endpoint with no data returns empty response."""
        mock_fastapi_service._server_metrics = None
        response = api_test_client.get("/api/server-metrics")
        assert response.status_code == 200
        data = response.json()
        assert data["endpoint_summaries"] == {}
        assert "message" in data

    @pytest.mark.skip(
        reason="/api/server-metrics not yet migrated to router-based architecture"
    )
    def test_server_metrics_with_data(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test server metrics endpoint returns cached data."""
        mock_fastapi_service._server_metrics = {
            "endpoint_summaries": {
                "http://server:8000/metrics": {
                    "queue_depth": {"avg": 5.0, "max": 10.0},
                    "cache_hit_rate": {"avg": 0.85},
                }
            },
            "message_type": "realtime_server_metrics",
        }
        response = api_test_client.get("/api/server-metrics")
        assert response.status_code == 200
        data = response.json()
        assert "http://server:8000/metrics" in data["endpoint_summaries"]
        assert (
            data["endpoint_summaries"]["http://server:8000/metrics"]["queue_depth"][
                "avg"
            ]
            == 5.0
        )


class TestResultsEndpoint:
    """Test the /api/results endpoint for benchmark results retrieval."""

    def test_results_running_no_results(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test results endpoint returns running status when no results available."""
        mock_fastapi_service._routers["results"]._final_results = None
        mock_fastapi_service._routers["results"]._benchmark_complete = False

        response = api_test_client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["results"] is None

    def test_results_complete_with_results(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test results endpoint returns complete status with results."""

        mock_fastapi_service._routers[
            "results"
        ]._final_results = make_process_records_result(
            completed=100, was_cancelled=False
        )
        mock_fastapi_service._routers["results"]._benchmark_complete = True

        response = api_test_client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "complete"
        assert data["results"] is not None
        assert data["results"]["results"]["completed"] == 100
        assert data["results"]["results"]["was_cancelled"] is False

    @pytest.mark.parametrize(
        "was_cancelled,expected_status",
        [
            param(False, "complete", id="not-cancelled-complete"),
            param(True, "cancelled", id="was-cancelled"),
        ],
    )  # fmt: skip
    def test_results_status_based_on_cancellation(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        was_cancelled: bool,
        expected_status: str,
    ) -> None:
        """Test results endpoint status reflects cancellation state."""

        mock_fastapi_service._routers[
            "results"
        ]._final_results = make_process_records_result(was_cancelled=was_cancelled)
        mock_fastapi_service._routers["results"]._benchmark_complete = True

        response = api_test_client.get("/api/results")
        data = response.json()
        assert data["status"] == expected_status

    @pytest.mark.parametrize(
        "completed_count",
        [
            param(0, id="zero-completed"),
            param(1, id="one-completed"),
            param(100, id="hundred-completed"),
            param(10000, id="ten-thousand-completed"),
        ],
    )  # fmt: skip
    def test_results_completed_counts(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        completed_count: int,
    ) -> None:
        """Test results endpoint returns correct completed count."""

        mock_fastapi_service._routers[
            "results"
        ]._final_results = make_process_records_result(completed=completed_count)
        mock_fastapi_service._routers["results"]._benchmark_complete = True

        response = api_test_client.get("/api/results")
        data = response.json()
        assert data["results"]["results"]["completed"] == completed_count

    def test_results_contains_metric_records(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test results contain metric records with expected structure."""
        latency = make_latency_metric(avg=150.0, p95=200.0, p99=250.0)
        mock_fastapi_service._routers[
            "results"
        ]._final_results = make_process_records_result(records=[latency])
        mock_fastapi_service._routers["results"]._benchmark_complete = True

        response = api_test_client.get("/api/results")
        data = response.json()

        records = data["results"]["results"]["records"]
        assert len(records) == 1
        assert records[0]["tag"] == "latency"
        assert records[0]["avg"] == 150.0
        assert records[0]["p95"] == 200.0
        assert records[0]["p99"] == 250.0


class TestStaticFileServing:
    """Test static file serving with path traversal protection."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            param("../secret.txt", id="parent-dir"),
            param("../../etc/passwd", id="etc-passwd"),
            param("static/../../../secret.txt", id="nested-traversal"),
            param("foo/../../../etc/passwd", id="deep-traversal"),
        ],
    )  # fmt: skip
    async def test_path_traversal_blocked(self, filename: str) -> None:
        """Test that path traversal attempts are blocked with 400."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _read_static(filename)
        assert exc_info.value.status_code == 400
        assert "Invalid filename" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_nonexistent_file_returns_404(self) -> None:
        """Test that non-existent files return 404."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await _read_static("nonexistent.html")
        assert exc_info.value.status_code == 404


class TestWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""

    def test_websocket_subscribe(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test WebSocket subscribe message."""
        mock_fastapi_service._ensure_zmq_subscriptions = AsyncMock()

        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {"type": "subscribe", "message_types": ["realtime_metrics"]}
            )
            response = websocket.receive_json()
            assert response["type"] == "subscribed"
            assert "realtime_metrics" in response["message_types"]

    def test_websocket_subscribe_multiple_types(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test WebSocket subscribe to multiple message types."""
        mock_fastapi_service._ensure_zmq_subscriptions = AsyncMock()

        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json(
                {
                    "type": "subscribe",
                    "message_types": ["realtime_metrics", "worker_status"],
                }
            )
            response = websocket.receive_json()
            assert response["type"] == "subscribed"
            assert set(response["message_types"]) == {
                "realtime_metrics",
                "worker_status",
            }

    def test_websocket_unsubscribe(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test WebSocket unsubscribe message."""
        mock_fastapi_service._ensure_zmq_subscriptions = AsyncMock()

        with api_test_client.websocket_connect("/ws") as websocket:
            # Subscribe first
            websocket.send_json(
                {"type": "subscribe", "message_types": ["realtime_metrics"]}
            )
            websocket.receive_json()

            # Unsubscribe
            websocket.send_json(
                {"type": "unsubscribe", "message_types": ["realtime_metrics"]}
            )
            response = websocket.receive_json()
            assert response["type"] == "unsubscribed"
            assert "realtime_metrics" in response["message_types"]

    def test_websocket_ping_pong(self, api_test_client: TestClient) -> None:
        """Test WebSocket ping/pong."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_json({"type": "ping"})
            response = websocket.receive_json()
            assert response["type"] == "pong"

    def test_websocket_invalid_json(self, api_test_client: TestClient) -> None:
        """Test WebSocket handles invalid JSON gracefully."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_text("not valid json")
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Invalid JSON" in response["message"]

    @pytest.mark.parametrize(
        "invalid_text",
        [
            param("not valid json", id="plain-text"),
            param("{incomplete", id="incomplete-json"),
            param("{'single': 'quotes'}", id="single-quotes"),
        ],
    )  # fmt: skip
    def test_websocket_various_invalid_json(
        self, api_test_client: TestClient, invalid_text: str
    ) -> None:
        """Test WebSocket handles various forms of invalid JSON."""
        with api_test_client.websocket_connect("/ws") as websocket:
            websocket.send_text(invalid_text)
            response = websocket.receive_json()
            assert response["type"] == "error"


@pytest.mark.skip(
    reason="ZMQ subscription cleanup not yet migrated to router-based architecture"
)
class TestOrphanedSubscriptionCleanup:
    """Test orphaned ZMQ subscription cleanup."""

    def test_cleanup_removes_orphaned_subscriptions(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that subscriptions are removed when no client uses them."""
        mock_fastapi_service.debug = MagicMock()

        # Setup: client2 uses topic_b, zmq has both topics
        ws = MagicMock()
        mock_fastapi_service.ws_manager.add("client2", ws)
        mock_fastapi_service.ws_manager.subscribe("client2", ["topic_b"])
        mock_fastapi_service._zmq_subscriptions = {"topic_a", "topic_b"}

        # Client 1 disconnects with topic_a (orphaned) and topic_b (still used)
        mock_fastapi_service._cleanup_zmq_subscriptions({"topic_a", "topic_b"})

        assert "topic_a" not in mock_fastapi_service._zmq_subscriptions
        assert "topic_b" in mock_fastapi_service._zmq_subscriptions

    def test_cleanup_keeps_subscriptions_used_by_other_clients(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that subscriptions still used by other clients are kept."""
        mock_fastapi_service.debug = MagicMock()

        # Setup: multiple clients with overlapping subscriptions
        ws2, ws3 = MagicMock(), MagicMock()
        mock_fastapi_service.ws_manager.add("client2", ws2)
        mock_fastapi_service.ws_manager.add("client3", ws3)
        mock_fastapi_service.ws_manager.subscribe("client2", ["topic_a", "topic_c"])
        mock_fastapi_service.ws_manager.subscribe("client3", ["topic_b"])
        mock_fastapi_service._zmq_subscriptions = {"topic_a", "topic_b", "topic_c"}

        # Client 1 disconnects with all topics, but they're all still used
        mock_fastapi_service._cleanup_zmq_subscriptions(
            {"topic_a", "topic_b", "topic_c"}
        )

        assert mock_fastapi_service._zmq_subscriptions == {
            "topic_a",
            "topic_b",
            "topic_c",
        }

    def test_cleanup_handles_empty_removed_subs(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test cleanup with empty removed subscriptions."""
        mock_fastapi_service.debug = MagicMock()
        mock_fastapi_service._zmq_subscriptions = {"topic_a"}

        mock_fastapi_service._cleanup_zmq_subscriptions(set())

        # Nothing should change
        assert mock_fastapi_service._zmq_subscriptions == {"topic_a"}

    def test_cleanup_handles_non_zmq_subscriptions(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test cleanup doesn't affect subscriptions not in zmq_subscriptions."""
        mock_fastapi_service.debug = MagicMock()
        mock_fastapi_service._zmq_subscriptions = {"topic_a"}

        # topic_b was never in zmq_subscriptions
        mock_fastapi_service._cleanup_zmq_subscriptions({"topic_b"})

        assert mock_fastapi_service._zmq_subscriptions == {"topic_a"}


class TestCreateTestApp:
    """Test the create_test_app factory and dependency injection patterns."""

    def test_create_test_app_with_mock_service(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test create_test_app creates a working app with injected service."""
        app = create_test_app(mock_fastapi_service)
        client = TestClient(app)

        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.text == "ok"

    def test_dependency_overrides_pattern(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that app.state.service injection works for mocking service."""
        app = create_test_app(None)  # No service initially
        app.state.service = mock_fastapi_service

        client = TestClient(app)
        response = client.get("/healthz")
        assert response.status_code == 200

    def test_create_test_app_without_service_raises(self) -> None:
        """Test that endpoints fail gracefully without a service."""
        app = create_test_app(None)
        client = TestClient(app, raise_server_exceptions=False)

        response = client.get("/healthz")
        assert response.status_code == 500


@pytest.mark.skip(
    reason="ZMQ subscription management not yet migrated to router-based architecture"
)
class TestZmqSubscriptionManagement:
    """Test ZMQ subscription management in FastAPIService."""

    @pytest.mark.asyncio
    async def test_ensure_zmq_subscriptions_skips_handled_types(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that handled message types are not dynamically subscribed."""
        mock_fastapi_service.subscribe = AsyncMock()

        # REALTIME_METRICS is in _handled_types, so it should be skipped
        await mock_fastapi_service._ensure_zmq_subscriptions(["realtime_metrics"])

        mock_fastapi_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_zmq_subscriptions_skips_wildcard(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that wildcard subscription is skipped."""
        mock_fastapi_service.subscribe = AsyncMock()

        await mock_fastapi_service._ensure_zmq_subscriptions(["*"])

        mock_fastapi_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_zmq_subscriptions_skips_already_subscribed(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that already subscribed types are not re-subscribed."""
        mock_fastapi_service.subscribe = AsyncMock()
        mock_fastapi_service._zmq_subscriptions = {"custom_type"}

        await mock_fastapi_service._ensure_zmq_subscriptions(["custom_type"])

        mock_fastapi_service.subscribe.assert_not_called()

    @pytest.mark.asyncio
    async def test_ensure_zmq_subscriptions_adds_new_subscription(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that new message types are subscribed."""
        from aiperf.common.enums import MessageType

        mock_fastapi_service.subscribe = AsyncMock()

        # Use a valid MessageType that's not in _handled_types
        await mock_fastapi_service._ensure_zmq_subscriptions(
            [str(MessageType.HEARTBEAT)]
        )

        mock_fastapi_service.subscribe.assert_called_once()
        assert str(MessageType.HEARTBEAT) in mock_fastapi_service._zmq_subscriptions

    @pytest.mark.asyncio
    async def test_ensure_zmq_subscriptions_handles_invalid_type(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that invalid message types are handled gracefully."""
        mock_fastapi_service.subscribe = AsyncMock()

        await mock_fastapi_service._ensure_zmq_subscriptions(["invalid_type_xyz"])

        # Should not raise, should log warning
        mock_fastapi_service.warning.assert_called()
        mock_fastapi_service.subscribe.assert_not_called()


class TestFinalResultsMixin:
    """Test FinalResultsMixin message handling in FastAPIService."""

    @pytest.mark.asyncio
    async def test_on_process_records_result_stores_results(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that ProcessRecordsResultMessage stores results."""
        from aiperf.common.messages import ProcessRecordsResultMessage

        # Initial state
        assert mock_fastapi_service._routers["results"]._final_results is None
        assert mock_fastapi_service._routers["results"]._benchmark_complete is False

        # Simulate receiving the message
        result = make_process_records_result(completed=200)
        message = ProcessRecordsResultMessage(
            service_id="records_manager", results=result
        )

        await mock_fastapi_service._routers["results"]._on_process_records_result(
            message
        )

        assert mock_fastapi_service._routers["results"]._final_results is not None
        assert (
            mock_fastapi_service._routers["results"]._final_results.results.completed
            == 200
        )
        # _benchmark_complete stays False until BenchmarkCompleteMessage arrives
        # (after export is done). This ensures external consumers don't fetch
        # results before files are written to disk.
        assert mock_fastapi_service._routers["results"]._benchmark_complete is False

    @pytest.mark.asyncio
    async def test_on_process_records_result_replaces_previous(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that subsequent messages replace previous results."""
        from aiperf.common.messages import ProcessRecordsResultMessage

        # First message
        first_result = make_process_records_result(completed=100)
        message1 = ProcessRecordsResultMessage(
            service_id="records_manager", results=first_result
        )
        await mock_fastapi_service._routers["results"]._on_process_records_result(
            message1
        )
        assert (
            mock_fastapi_service._routers["results"]._final_results.results.completed
            == 100
        )

        # Second message (replaces first)
        second_result = make_process_records_result(completed=200)
        message2 = ProcessRecordsResultMessage(
            service_id="records_manager", results=second_result
        )
        await mock_fastapi_service._routers["results"]._on_process_records_result(
            message2
        )
        assert (
            mock_fastapi_service._routers["results"]._final_results.results.completed
            == 200
        )

    @pytest.mark.parametrize(
        "completed,was_cancelled",
        [
            param(0, False, id="zero-completed-not-cancelled"),
            param(100, False, id="hundred-completed-not-cancelled"),
            param(50, True, id="fifty-completed-cancelled"),
            param(0, True, id="zero-completed-cancelled"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_on_process_records_result_various_states(
        self,
        mock_fastapi_service: FastAPIService,
        completed: int,
        was_cancelled: bool,
    ) -> None:
        """Test message handling with various completion and cancellation states."""
        from aiperf.common.messages import ProcessRecordsResultMessage

        result = make_process_records_result(
            completed=completed, was_cancelled=was_cancelled
        )
        message = ProcessRecordsResultMessage(
            service_id="records_manager", results=result
        )

        await mock_fastapi_service._routers["results"]._on_process_records_result(
            message
        )

        assert (
            mock_fastapi_service._routers["results"]._final_results.results.completed
            == completed
        )
        assert (
            mock_fastapi_service._routers[
                "results"
            ]._final_results.results.was_cancelled
            == was_cancelled
        )
        assert mock_fastapi_service._routers["results"]._benchmark_complete is False


class TestBenchmarkCompleteHandler:
    """Test BenchmarkCompleteMessage handling in FastAPIService."""

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_sets_flag(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that BenchmarkCompleteMessage sets benchmark_complete flag."""
        from aiperf.common.messages import BenchmarkCompleteMessage

        assert mock_fastapi_service._routers["results"]._benchmark_complete is False

        message = BenchmarkCompleteMessage(
            service_id="system_controller", was_cancelled=False
        )

        await mock_fastapi_service._routers["results"]._on_benchmark_complete(message)

        assert mock_fastapi_service._routers["results"]._benchmark_complete is True

    @pytest.mark.parametrize(
        "was_cancelled",
        [
            param(False, id="not-cancelled"),
            param(True, id="was-cancelled"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_on_benchmark_complete_with_cancellation_states(
        self,
        mock_fastapi_service: FastAPIService,
        was_cancelled: bool,
    ) -> None:
        """Test handler works with both cancelled and non-cancelled states."""
        from aiperf.common.messages import BenchmarkCompleteMessage

        message = BenchmarkCompleteMessage(
            service_id="system_controller", was_cancelled=was_cancelled
        )

        await mock_fastapi_service._routers["results"]._on_benchmark_complete(message)

        # Flag should be set regardless of cancellation state
        assert mock_fastapi_service._routers["results"]._benchmark_complete is True

    @pytest.mark.asyncio
    async def test_on_benchmark_complete_idempotent(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that multiple calls are idempotent."""
        from aiperf.common.messages import BenchmarkCompleteMessage

        message = BenchmarkCompleteMessage(
            service_id="system_controller", was_cancelled=False
        )

        # Call multiple times
        await mock_fastapi_service._routers["results"]._on_benchmark_complete(message)
        await mock_fastapi_service._routers["results"]._on_benchmark_complete(message)
        await mock_fastapi_service._routers["results"]._on_benchmark_complete(message)

        # Should still be True
        assert mock_fastapi_service._routers["results"]._benchmark_complete is True


# =============================================================================
# Compression encoding selection
# =============================================================================


class TestSelectEncoding:
    """Test compression encoding selection."""

    @pytest.mark.parametrize(
        "accept_encoding,expected",
        [
            param("zstd, gzip", "zstd", id="prefers-zstd"),
            param("gzip", "gzip", id="fallback-gzip"),
            param("deflate, br", "identity", id="unknown-identity-fallback"),
            param(None, "gzip", id="none-fallback-gzip"),
            param("", "gzip", id="empty-fallback-gzip"),
            param("ZSTD, GZIP", "zstd", id="case-insensitive"),
            param("  zstd  ,  gzip  ", "zstd", id="whitespace-handling"),
        ],
    )  # fmt: skip
    def test_select_encoding(self, accept_encoding: str | None, expected: str) -> None:
        """Test encoding selection based on Accept-Encoding header."""
        result = select_encoding(accept_encoding)
        expected_encoding = CompressionEncoding(expected)
        if expected_encoding == CompressionEncoding.ZSTD and not is_zstd_available():
            assert result == CompressionEncoding.GZIP
        else:
            assert result == expected_encoding


@pytest.mark.skip(
    reason="_get_content_type moved to inline dict lookup in results router"
)
class TestGetContentType:
    """Test the _get_content_type helper function."""

    @pytest.mark.parametrize(
        "suffix,expected",
        [
            param(".json", "application/json", id="json"),
            param(".jsonl", "application/x-ndjson", id="jsonl"),
            param(".csv", "text/csv", id="csv"),
            param(".parquet", "application/vnd.apache.parquet", id="parquet"),
            param(".txt", "application/octet-stream", id="txt-fallback"),
            param(".dat", "application/octet-stream", id="dat-fallback"),
            param("", "application/octet-stream", id="no-extension"),
        ],
    )  # fmt: skip
    def test_get_content_type_returns_correct_type(
        self, suffix: str, expected: str
    ) -> None:
        """Test that _get_content_type returns correct content type for various extensions."""
        import pathlib

        from aiperf.api.api_service import _get_content_type

        file_path = pathlib.Path(f"/results/test{suffix}")
        assert _get_content_type(file_path) == expected


class TestResultsListEndpoint:
    """Test the /api/results/list endpoint."""

    def test_list_results_empty_directory(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test listing results when directory doesn't exist."""
        from unittest.mock import MagicMock

        mock_output = MagicMock()
        mock_output.artifact_directory.exists.return_value = False
        mock_fastapi_service._routers["results"].run.cfg.artifacts = mock_output

        response = api_test_client.get("/api/results/list")
        assert response.status_code == 200
        data = response.json()
        assert data["files"] == []

    def test_list_results_with_files(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        tmp_path,
    ) -> None:
        """Test listing results with files in directory."""
        from unittest.mock import MagicMock

        # Create test files
        (tmp_path / "metrics.json").write_text('{"test": 1}')
        (tmp_path / "records.jsonl").write_text('{"id": 1}')

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        mock_fastapi_service._routers["results"].run.cfg.artifacts = mock_output

        response = api_test_client.get("/api/results/list")
        assert response.status_code == 200
        data = response.json()

        file_names = [f["name"] for f in data["files"]]
        assert "metrics.json" in file_names
        assert "records.jsonl" in file_names
        for f in data["files"]:
            assert "size" in f
            assert f["size"] > 0


class TestResultsFileEndpoints:
    """Test generic result file download endpoint."""

    def test_file_returns_404_when_missing(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        tmp_path,
    ) -> None:
        """Test returns 404 for nonexistent file."""
        from unittest.mock import MagicMock

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        mock_fastapi_service._routers["results"].run.cfg.artifacts = mock_output

        response = api_test_client.get("/api/results/files/nonexistent.json")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_file_streams_content_with_correct_headers(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        tmp_path,
    ) -> None:
        """Test file streams content with correct headers."""
        from unittest.mock import MagicMock

        test_file = tmp_path / "profile_export.json"
        test_file.write_text('{"metrics": {"latency": 100}}')

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        mock_fastapi_service._routers["results"].run.cfg.artifacts = mock_output

        response = api_test_client.get(
            "/api/results/files/profile_export.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert response.status_code == 200
        assert "profile_export.json" in response.headers["content-disposition"]
        assert "profile_export.json" in response.headers["x-filename"]

    def test_file_rejects_path_traversal(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        tmp_path,
    ) -> None:
        """Test path traversal attempts are rejected."""
        from unittest.mock import MagicMock

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        mock_fastapi_service._routers["results"].run.cfg.artifacts = mock_output

        response = api_test_client.get("/api/results/files/../../../etc/passwd")
        assert response.status_code in (400, 404)

    def test_file_supports_compression(
        self,
        api_test_client: TestClient,
        mock_fastapi_service: FastAPIService,
        tmp_path,
    ) -> None:
        """Test result file endpoint supports gzip compression."""
        from unittest.mock import MagicMock

        test_file = tmp_path / "metrics.json"
        test_file.write_text('{"metrics": {"latency": 100}}')

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        mock_fastapi_service._routers["results"].run.cfg.artifacts = mock_output

        response = api_test_client.get(
            "/api/results/files/metrics.json",
            headers={"Accept-Encoding": "gzip"},
        )
        assert response.status_code == 200
        assert response.headers["content-encoding"] == "gzip"


class TestWebSocketBroadcast:
    """Test broadcasting messages via WebSocketManager."""

    @pytest.mark.asyncio
    async def test_broadcast_message_to_subscribers(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that messages are broadcast to subscribed WebSocket clients."""
        from aiperf.common.enums import LifecycleState
        from aiperf.common.messages import HeartbeatMessage

        mock_ws = AsyncMock()
        mock_fastapi_service.ws_manager.add("client-1", mock_ws)
        mock_fastapi_service.ws_manager.subscribe("client-1", ["heartbeat"])

        message = HeartbeatMessage(
            service_id="test",
            service_type="worker",
            state=LifecycleState.RUNNING,
        )

        await mock_fastapi_service.ws_manager.broadcast(message)

        mock_ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_message_no_subscribers(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test broadcast with no subscribers does not raise."""
        from aiperf.common.enums import LifecycleState
        from aiperf.common.messages import HeartbeatMessage

        message = HeartbeatMessage(
            service_id="test",
            service_type="worker",
            state=LifecycleState.RUNNING,
        )

        sent = await mock_fastapi_service.ws_manager.broadcast(message)
        assert sent == 0


class TestStaticPageEndpoints:
    """Test the static page serving endpoints."""

    def test_index_page_returns_html(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test index page serves HTML."""
        from unittest.mock import patch

        with patch(
            "aiperf.api.routers.static._read_static",
            return_value="<html>Index</html>",
        ):
            response = api_test_client.get("/")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]

    def test_dashboard_page_returns_html(
        self, api_test_client: TestClient, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test dashboard page serves HTML."""
        from unittest.mock import patch

        with patch(
            "aiperf.api.routers.static._read_static",
            return_value="<html>Dashboard</html>",
        ):
            response = api_test_client.get("/dashboard")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]


# =============================================================================
# Service properties
# =============================================================================


class TestServiceBaseUrl:
    """Test the _base_url property."""

    def test_base_url_format(self, mock_fastapi_service: FastAPIService) -> None:
        """Test _base_url returns correct format."""
        mock_fastapi_service.api_host = "0.0.0.0"
        mock_fastapi_service.api_port = 8080

        assert mock_fastapi_service._base_url == "http://0.0.0.0:8080"

    def test_base_url_localhost(self, mock_fastapi_service: FastAPIService) -> None:
        """Test _base_url with localhost."""
        mock_fastapi_service.api_host = "127.0.0.1"
        mock_fastapi_service.api_port = 9999

        assert mock_fastapi_service._base_url == "http://127.0.0.1:9999"


class TestInfoLabelsCache:
    """Test the info labels caching behavior."""

    def test_get_info_labels_creates_and_caches(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test get_info_labels creates and caches labels on MetricsRouter."""
        metrics_router = mock_fastapi_service._routers["metrics"]
        assert metrics_router._info_labels is None

        labels1 = metrics_router.get_info_labels()
        assert labels1 is not None
        assert metrics_router._info_labels is not None

        # Second call should return cached value
        labels2 = metrics_router.get_info_labels()
        assert labels1 is labels2


# =============================================================================
# FastAPIService lifecycle tests (init, start, stop, main)
# =============================================================================


class TestFastAPIServiceInit:
    """Test FastAPIService.__init__ via direct instantiation."""

    def test_init_sets_host_port_from_config(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        assert mock_fastapi_service.api_host == "0.0.0.0"
        assert mock_fastapi_service.api_port == 8080

    def test_init_creates_app(self, mock_fastapi_service: FastAPIService) -> None:
        assert mock_fastapi_service.app is not None
        assert mock_fastapi_service.app.title == "AIPerf API"

    def test_init_defaults_server_to_none(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        assert mock_fastapi_service._server is None
        assert mock_fastapi_service._server_task is None

    def test_init_loads_routers(self, mock_fastapi_service: FastAPIService) -> None:
        assert len(mock_fastapi_service._routers) > 0

    def test_init_with_custom_host(
        self, mock_zmq: None, api_run: BenchmarkRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type("_Fake", (), {"HOST": "0.0.0.0", "PORT": 8080, "CORS_ORIGINS": []})(),
        )
        service = FastAPIService(
            run=api_run,
            service_id="api-custom",
        )
        assert service.api_host == "0.0.0.0"
        assert service.api_port == 8080


class TestFastAPIServiceCORSMiddleware:
    """Test CORS middleware is added when cors_origins is set."""

    def test_cors_middleware_added_when_origins_set(
        self,
        mock_zmq: None,
        api_run: BenchmarkRun,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type(
                "_Fake",
                (),
                {"HOST": "127.0.0.1", "PORT": 8080, "CORS_ORIGINS": ["*"]},
            )(),
        )
        service = FastAPIService(
            run=api_run,
            service_id="api-cors",
        )
        middleware_names = [m.cls.__name__ for m in service.app.user_middleware]
        assert "CORSMiddleware" in middleware_names

    def test_no_cors_middleware_when_origins_empty(
        self,
        mock_zmq: None,
        api_run: BenchmarkRun,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type(
                "_Fake", (), {"HOST": "127.0.0.1", "PORT": 8080, "CORS_ORIGINS": []}
            )(),
        )
        service = FastAPIService(
            run=api_run,
            service_id="api-no-cors",
        )
        middleware_names = [m.cls.__name__ for m in service.app.user_middleware]
        assert "CORSMiddleware" not in middleware_names


class TestFastAPIServiceStartStop:
    """Test _start_api_server and _stop_api_server."""

    @pytest.mark.asyncio
    async def test_start_raises_when_port_not_configured(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_fastapi_service.api_port = None
        with pytest.raises(ValueError, match="API port is not configured"):
            await mock_fastapi_service._start_api_server()

    @pytest.mark.asyncio
    async def test_start_creates_server_and_task(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()

        with (
            patch("aiperf.api.api_service.uvicorn.Config"),
            patch("aiperf.api.api_service.uvicorn.Server", return_value=mock_server),
        ):
            await mock_fastapi_service._start_api_server()

        assert mock_fastapi_service._server is mock_server
        assert mock_fastapi_service._server_task is not None

        mock_fastapi_service._server_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await mock_fastapi_service._server_task

    @pytest.mark.asyncio
    async def test_stop_sets_should_exit_and_waits(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_server = MagicMock()
        completed = asyncio.Event()

        async def fake_serve():
            await completed.wait()

        task = asyncio.create_task(fake_serve())
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = task

        completed.set()
        await mock_fastapi_service._stop_api_server()

        assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_stop_cancels_on_timeout(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_server = MagicMock()

        async def hang_forever():
            await asyncio.Future()

        task = asyncio.create_task(hang_forever())
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = task

        real_wait_for = asyncio.wait_for
        call_count = 0

        async def first_call_times_out(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError
            return await real_wait_for(*args, **kwargs)

        with patch(
            "aiperf.api.api_service.asyncio.wait_for",
            side_effect=first_call_times_out,
        ):
            await mock_fastapi_service._stop_api_server()

        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_stop_handles_no_server(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        mock_fastapi_service._server = None
        mock_fastapi_service._server_task = None
        await mock_fastapi_service._stop_api_server()

    @pytest.mark.asyncio
    async def test_stop_propagates_cancelled_error(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test _stop_api_server re-raises CancelledError for cooperative cancellation."""
        mock_server = MagicMock()
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = asyncio.create_task(asyncio.sleep(100))

        with (
            patch(
                "aiperf.api.api_service.asyncio.wait_for",
                side_effect=asyncio.CancelledError,
            ),
            pytest.raises(asyncio.CancelledError),
        ):
            await mock_fastapi_service._stop_api_server()

        assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_on_server_task_done_schedules_stop_on_exception(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test _on_server_task_done schedules stop when server task fails."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = RuntimeError("server crashed")

        with patch.object(
            mock_fastapi_service, "stop", new_callable=AsyncMock
        ) as mock_stop:
            mock_fastapi_service._on_server_task_done(task)
            assert mock_fastapi_service._stop_task is not None
            await asyncio.sleep(0)
            mock_stop.assert_called_once()

    def test_on_server_task_done_ignores_cancelled(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test _on_server_task_done does nothing for cancelled tasks."""
        task = MagicMock()
        task.cancelled.return_value = True
        mock_fastapi_service._on_server_task_done(task)
        task.exception.assert_not_called()
        assert mock_fastapi_service._stop_task is None

    def test_on_server_task_done_no_exception(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test _on_server_task_done does nothing when task succeeds."""
        task = MagicMock()
        task.cancelled.return_value = False
        task.exception.return_value = None
        mock_fastapi_service._on_server_task_done(task)
        assert mock_fastapi_service._stop_task is None


class TestFastAPIServiceLifespan:
    """Test FastAPI lifespan hooks."""

    def test_lifespan_logs_startup_and_shutdown(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that lifespan logs on startup and shutdown."""
        mock_fastapi_service.info = MagicMock()

        with TestClient(mock_fastapi_service.app):
            pass

        info_calls = [call[0][0] for call in mock_fastapi_service.info.call_args_list]
        assert any("FastAPI starting" in msg for msg in info_calls)
        assert any("FastAPI stopped" in msg for msg in info_calls)


class TestFastAPIServiceMain:
    """Test the main() entry point."""

    def test_main_calls_bootstrap(self) -> None:
        with patch(
            "aiperf.api.api_service.bootstrap_and_run_service"
        ) as mock_bootstrap:
            main()
            mock_bootstrap.assert_called_once_with(ServiceType.API)
