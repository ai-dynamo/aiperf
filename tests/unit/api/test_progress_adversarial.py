# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adversarial progress and server-metrics API tests.

Focuses on:
- /api/progress snapshot schema stability beyond pod-state RPC aggregation.
- SYSTEM_STATE_CHANGED deserialization boundaries and malformed-state rejection.
- Results-exported and concurrent progress updates reflected through FastAPI.
- Server-metrics summaries, missing required fields, and non-finite JSON output.

Out of scope: pod-state RPC fallback behavior; see test_pod_state_rpc_adversarial.py.
"""

from __future__ import annotations

from pathlib import Path

import msgspec
import orjson
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.routers.progress import ProgressRouter
from aiperf.api.routers.server_metrics import ServerMetricsRouter
from aiperf.common.enums import CreditPhase, SystemState
from aiperf.common.messages import (
    RealtimeServerMetricsMessage,
    ResultsExportedMessage,
    SystemStateChangedMessage,
)
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.common.models.server_metrics_models import (
    GaugeMetricData,
    GaugeSeries,
    GaugeStats,
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
)
from aiperf.config import AIPerfConfig, BenchmarkRun


# ============================================================================
# Helpers
# ============================================================================


def _benchmark_run() -> BenchmarkRun:
    """Real Pydantic benchmark run for router construction."""
    config = AIPerfConfig(
        benchmark={
            "models": ["meta-llama/Llama-3-8B"],
            "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
            "datasets": [
                {
                    "name": "synthetic-chat-main",
                    "type": "synthetic",
                    "entries": 100,
                    "prompts": {"isl": 128, "osl": 64},
                }
            ],
            "phases": [
                {
                    "name": "steady-state",
                    "type": "concurrency",
                    "requests": 10,
                    "concurrency": 1,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id="aiperf-bench-7f2a",
        cfg=config.benchmark,
        artifact_dir=Path("/tmp/aiperf-bench-7f2a"),
    )


def _progress_app(router: ProgressRouter) -> FastAPI:
    app = FastAPI()
    app.state.progress = router
    app.include_router(router.get_router())
    return app


def _server_metrics_app(router: ServerMetricsRouter) -> FastAPI:
    app = FastAPI()
    app.state.server_metrics = router
    app.include_router(router.get_router())
    return app


def _endpoint_info(**overrides: float | int | None) -> ServerMetricsEndpointInfo:
    values: dict[str, float | int | None] = {
        "total_fetches": 4,
        "first_fetch_ns": 1_000,
        "last_fetch_ns": 4_000,
        "avg_fetch_latency_ms": 2.5,
        "unique_updates": 3,
        "first_update_ns": 1_100,
        "last_update_ns": 3_900,
        "duration_seconds": 2.8,
        "avg_update_interval_ms": 900.0,
        "median_update_interval_ms": 875.0,
    }
    values.update(overrides)
    return ServerMetricsEndpointInfo(**values)  # type: ignore[arg-type]


def _endpoint_summary(
    endpoint_url: str = "http://vllm-a.metrics:8000/metrics",
    *,
    avg_fetch_latency_ms: float = 2.5,
    kv_cache_avg: float = 0.72,
) -> ServerMetricsEndpointSummary:
    return ServerMetricsEndpointSummary(
        endpoint_url=endpoint_url,
        info=_endpoint_info(avg_fetch_latency_ms=avg_fetch_latency_ms),
        metrics={
            "vllm:gpu_cache_usage_perc": GaugeMetricData(
                description="KV cache usage percentage",
                unit="percent",
                series=[
                    GaugeSeries(
                        endpoint_url=endpoint_url,
                        labels={"model_name": "meta-llama/Llama-3-8B"},
                        stats=GaugeStats(avg=kv_cache_avg, min=0.5, max=0.95),
                    )
                ],
            )
        },
    )


@pytest.fixture
def progress_router(mock_zmq: object) -> ProgressRouter:
    return ProgressRouter(run=_benchmark_run())


@pytest.fixture
def progress_client(progress_router: ProgressRouter) -> TestClient:
    return TestClient(_progress_app(progress_router))


@pytest.fixture
def server_metrics_router(mock_zmq: object) -> ServerMetricsRouter:
    return ServerMetricsRouter(run=_benchmark_run())


@pytest.fixture
def server_metrics_client(server_metrics_router: ServerMetricsRouter) -> TestClient:
    return TestClient(_server_metrics_app(server_metrics_router))


# ============================================================================
# /api/progress snapshot schema and lifecycle messages
# ============================================================================


class TestProgressSnapshotSchema:
    """Snapshot fields are stable for operator and dashboard consumers."""

    def test_get_progress_empty_snapshot_has_stable_top_level_schema(
        self, progress_client: TestClient
    ) -> None:
        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        data = orjson.loads(response.content)
        assert set(data) == {"phases", "workers", "results_exported", "system_state"}
        assert data["phases"] == {}
        assert data["results_exported"] is False
        assert data["system_state"] == "initializing"
        assert data["workers"]["ready"] == 0
        assert data["workers"]["total_pods"] == 0

    def test_get_progress_phase_snapshot_preserves_false_zero_and_null_fields(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        progress_router._progress_tracker._phases = {
            CreditPhase.PROFILING: CombinedPhaseStats(
                phase=CreditPhase.PROFILING,
                exclude_from_results=False,
                start_ns=1_000,
                total_expected_requests=0,
                requests_completed=0,
                requests_per_second=None,
            )
        }

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        phase = orjson.loads(response.content)["phases"]["profiling"]
        assert phase["phase"] == "profiling"
        assert phase["exclude_from_results"] is False
        assert phase["total_expected_requests"] == 0
        assert phase["requests_completed"] == 0
        assert phase["requests_per_second"] is None

    @pytest.mark.asyncio
    async def test_on_results_exported_sets_snapshot_gate_true(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        await progress_router._on_results_exported(
            ResultsExportedMessage(service_id="system_controller", was_cancelled=False)
        )

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        assert orjson.loads(response.content)["results_exported"] is True

    @pytest.mark.parametrize(
        "wire_state,expected_state",
        [
            ("profiling", SystemState.PROFILING),
            param("PROFILING", SystemState.PROFILING, id="uppercase-accepted"),
            param("Ready", SystemState.READY, id="mixed-case-accepted"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_system_state_changed_message_wire_values_decode_to_enum(
        self,
        progress_client: TestClient,
        progress_router: ProgressRouter,
        wire_state: str,
        expected_state: SystemState,
    ) -> None:
        message = SystemStateChangedMessage.from_json(
            {
                "message_type": "system_state_changed",
                "service_id": "system_controller",
                "state": wire_state,
            }
        )

        await progress_router._on_system_state_changed(message)

        response = progress_client.get("/api/progress")
        assert response.status_code == 200
        assert orjson.loads(response.content)["system_state"] == expected_state.value

    @pytest.mark.parametrize(
        "payload,match",
        [
            param(
                {
                    "message_type": "system_state_changed",
                    "service_id": "system_controller",
                    "state": "profilling",
                },
                r"state.*profilling|profilling.*state",
                id="typo-state",
            ),
            param(
                {
                    "message_type": "system_state_changed",
                    "service_id": "system_controller",
                    "state": None,
                },
                r"state.*null|null.*state",
                id="null-state",
            ),
            param(
                {
                    "message_type": "system_state_changed",
                    "service_id": "system_controller",
                },
                r"state",
                id="missing-state",
            ),
        ],
    )  # fmt: skip
    def test_system_state_changed_message_malformed_wire_values_are_rejected(
        self, payload: dict[str, object], match: str
    ) -> None:
        with pytest.raises(msgspec.ValidationError, match=match):
            SystemStateChangedMessage.from_json(payload)

    def test_get_progress_non_finite_phase_numbers_serialize_as_null(
        self, progress_client: TestClient, progress_router: ProgressRouter
    ) -> None:
        progress_router._progress_tracker._phases = {
            CreditPhase.PROFILING: CombinedPhaseStats(
                phase=CreditPhase.PROFILING,
                start_ns=1_000,
                total_expected_requests=10,
                requests_completed=1,
                requests_per_second=float("nan"),
                records_per_second=float("inf"),
            )
        }

        response = progress_client.get("/api/progress")

        assert response.status_code == 200
        assert b"NaN" not in response.content
        assert b"Infinity" not in response.content
        phase = orjson.loads(response.content)["phases"]["profiling"]
        assert phase["requests_per_second"] is None
        assert phase["records_per_second"] is None

    @pytest.mark.asyncio
    async def test_get_progress_concurrent_system_state_updates_never_emit_invalid_state(
        self, progress_router: ProgressRouter
    ) -> None:
        app = _progress_app(progress_router)
        transport = ASGITransport(app=app)
        valid_states = {state.value for state in SystemState}

        async with AsyncClient(transport=transport, base_url="http://api.local") as client:
            for state in (
                SystemState.CONFIGURING,
                SystemState.READY,
                SystemState.PROFILING,
                SystemState.PROCESSING,
                SystemState.STOPPING,
            ):
                await progress_router._on_system_state_changed(
                    SystemStateChangedMessage(
                        service_id="system_controller",
                        state=state,
                    )
                )
                response = await client.get("/api/progress")
                assert response.status_code == 200
                assert response.json()["system_state"] in valid_states

        await progress_router._on_system_state_changed(
            SystemStateChangedMessage(
                service_id="system_controller",
                state=SystemState.SHUTDOWN,
            )
        )
        final = TestClient(app).get("/api/progress")
        assert final.json()["system_state"] == "shutdown"


# ============================================================================
# Server metrics summaries and JSON boundaries
# ============================================================================


class TestServerMetricsSummaryBoundaries:
    """Realtime server-metrics summaries stay JSON-safe at the API boundary."""

    def test_get_server_metrics_before_message_returns_empty_summary_contract(
        self, server_metrics_client: TestClient
    ) -> None:
        response = server_metrics_client.get("/api/server-metrics")

        assert response.status_code == 200
        data = orjson.loads(response.content)
        assert data["endpoint_summaries"] == {}
        assert "server metrics" in data["message"]

    @pytest.mark.asyncio
    async def test_realtime_server_metrics_summary_excludes_message_identity_fields(
        self,
        server_metrics_client: TestClient,
        server_metrics_router: ServerMetricsRouter,
    ) -> None:
        endpoint_url = "http://vllm-a.metrics:8000/metrics"
        await server_metrics_router._on_realtime_server_metrics(
            RealtimeServerMetricsMessage(
                service_id="server_metrics_manager",
                endpoint_summaries={endpoint_url: _endpoint_summary(endpoint_url)},
            )
        )

        response = server_metrics_client.get("/api/server-metrics")

        assert response.status_code == 200
        data = orjson.loads(response.content)
        assert "message_type" not in data
        assert "service_id" not in data
        summary = data["endpoint_summaries"][endpoint_url]
        assert summary["endpoint_url"] == endpoint_url
        assert summary["info"]["total_fetches"] == 4
        assert (
            summary["metrics"]["vllm:gpu_cache_usage_perc"]["series"][0]["stats"]["avg"]
            == 0.72
        )

    @pytest.mark.parametrize(
        "summary_payload,match",
        [
            param(
                {"endpoint_url": "http://vllm-a.metrics:8000/metrics"},
                r"info",
                id="missing-info",
            ),
            param(
                {"info": {"total_fetches": 1}},
                r"endpoint_url|first_fetch_ns",
                id="missing-endpoint-and-info-fields",
            ),
        ],
    )  # fmt: skip
    def test_realtime_server_metrics_message_missing_summary_fields_are_rejected(
        self, summary_payload: dict[str, object], match: str
    ) -> None:
        payload = {
            "message_type": "realtime_server_metrics",
            "service_id": "server_metrics_manager",
            "endpoint_summaries": {
                "http://vllm-a.metrics:8000/metrics": summary_payload
            },
        }

        with pytest.raises(msgspec.ValidationError, match=match):
            RealtimeServerMetricsMessage.from_json(payload)

    @pytest.mark.asyncio
    async def test_server_metrics_non_finite_summary_numbers_serialize_as_null(
        self,
        server_metrics_client: TestClient,
        server_metrics_router: ServerMetricsRouter,
    ) -> None:
        endpoint_url = "http://vllm-a.metrics:8000/metrics"
        await server_metrics_router._on_realtime_server_metrics(
            RealtimeServerMetricsMessage(
                service_id="server_metrics_manager",
                endpoint_summaries={
                    endpoint_url: _endpoint_summary(
                        endpoint_url,
                        avg_fetch_latency_ms=float("nan"),
                        kv_cache_avg=float("inf"),
                    )
                },
            )
        )

        response = server_metrics_client.get("/api/server-metrics")

        assert response.status_code == 200
        assert b"NaN" not in response.content
        assert b"Infinity" not in response.content
        summary = orjson.loads(response.content)["endpoint_summaries"][endpoint_url]
        assert summary["info"]["avg_fetch_latency_ms"] is None
        assert (
            summary["metrics"]["vllm:gpu_cache_usage_perc"]["series"][0]["stats"]["avg"]
            is None
        )

    @pytest.mark.asyncio
    async def test_server_metrics_repeated_updates_return_latest_endpoint_summary(
        self,
        server_metrics_client: TestClient,
        server_metrics_router: ServerMetricsRouter,
    ) -> None:
        first_endpoint = "http://vllm-a.metrics:8000/metrics"
        second_endpoint = "http://vllm-b.metrics:8000/metrics"

        await server_metrics_router._on_realtime_server_metrics(
            RealtimeServerMetricsMessage(
                service_id="server_metrics_manager",
                endpoint_summaries={first_endpoint: _endpoint_summary(first_endpoint)},
            )
        )
        await server_metrics_router._on_realtime_server_metrics(
            RealtimeServerMetricsMessage(
                service_id="server_metrics_manager",
                endpoint_summaries={second_endpoint: _endpoint_summary(second_endpoint)},
            )
        )

        response = server_metrics_client.get("/api/server-metrics")

        assert response.status_code == 200
        endpoint_summaries = orjson.loads(response.content)["endpoint_summaries"]
        assert set(endpoint_summaries) == {second_endpoint}
        assert first_endpoint not in endpoint_summaries
