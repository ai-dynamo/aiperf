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
from aiperf.common.compression import (
    CompressionEncoding,
    is_zstd_available,
    select_encoding,
)
from aiperf.common.enums import LifecycleState
from aiperf.common.mixins.progress_tracker_mixin import CombinedPhaseStats
from aiperf.config import BenchmarkRun
from aiperf.plugin.enums import ServiceType

from .conftest import (
    create_test_app,
    make_latency_metric,
    make_metric_result,
    make_process_records_result,
)


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


class TestResultsRouterFinalResults:
    """Test ResultsRouter final-results message handling in FastAPIService."""

    @pytest.mark.asyncio
    async def test_on_process_records_result_stores_results(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        """Test that ProcessAllResultsMessage stores results."""
        from aiperf.common.messages import ProcessAllResultsMessage

        # Initial state
        assert mock_fastapi_service._routers["results"]._final_results is None
        assert mock_fastapi_service._routers["results"]._benchmark_complete is False

        # Simulate receiving the message
        result = make_process_records_result(completed=200)
        message = ProcessAllResultsMessage(service_id="records_manager", results=result)

        await mock_fastapi_service._routers["results"]._on_process_all_results(message)

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
        from aiperf.common.messages import ProcessAllResultsMessage

        # First message
        first_result = make_process_records_result(completed=100)
        message1 = ProcessAllResultsMessage(
            service_id="records_manager", results=first_result
        )
        await mock_fastapi_service._routers["results"]._on_process_all_results(message1)
        assert (
            mock_fastapi_service._routers["results"]._final_results.results.completed
            == 100
        )

        # Second message (replaces first)
        second_result = make_process_records_result(completed=200)
        message2 = ProcessAllResultsMessage(
            service_id="records_manager", results=second_result
        )
        await mock_fastapi_service._routers["results"]._on_process_all_results(message2)
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
        from aiperf.common.messages import ProcessAllResultsMessage

        result = make_process_records_result(
            completed=completed, was_cancelled=was_cancelled
        )
        message = ProcessAllResultsMessage(service_id="records_manager", results=result)

        await mock_fastapi_service._routers["results"]._on_process_all_results(message)

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

    @pytest.mark.asyncio
    async def test_initialize_attaches_every_router_as_lifecycle_child(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        # Routers are lifecycle children, so initialize/stop must propagate to
        # each one exactly once rather than leaving orphaned components behind.
        children = mock_fastapi_service._children
        assert all(
            router in children for router in mock_fastapi_service._routers.values()
        )
        await mock_fastapi_service.initialize()
        await mock_fastapi_service.stop()

    def test_init_uses_constructor_api_port(
        self, mock_zmq: None, api_run: BenchmarkRun, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        api_run.cfg.runtime.api_port = None
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type("_Fake", (), {"HOST": "0.0.0.0", "PORT": None, "CORS_ORIGINS": []})(),
        )
        service = FastAPIService(
            run=api_run,
            service_id="api-custom-port",
            api_port=9090,
        )
        assert service.api_port == 9090

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
    async def test_start_constructor_port_bind_conflict_raises(
        self, mock_zmq: None, api_run: BenchmarkRun
    ) -> None:
        api_run.cfg.runtime.api_port = None
        service = FastAPIService(
            run=api_run,
            service_id="api-constructor-port",
            api_port=9090,
        )
        bind = MagicMock(side_effect=OSError("address already in use"))

        with (
            patch("aiperf.api.api_service.socket.socket") as socket_mock,
            pytest.raises(RuntimeError, match="API server cannot bind 0.0.0.0:9090"),
        ):
            socket_mock.return_value.__enter__.return_value.bind = bind
            await service._start_api_server()

    @pytest.mark.asyncio
    async def test_start_runtime_port_bind_conflict_raises(
        self, mock_fastapi_service: FastAPIService
    ) -> None:
        bind = MagicMock(side_effect=OSError("address already in use"))

        with (
            patch("aiperf.api.api_service.socket.socket") as socket_mock,
            pytest.raises(RuntimeError, match="API server cannot bind 0.0.0.0:8080"),
        ):
            socket_mock.return_value.__enter__.return_value.bind = bind
            await mock_fastapi_service._start_api_server()

    @pytest.mark.asyncio
    async def test_start_implicit_port_bind_conflict_warns_and_skips_server(
        self,
        mock_zmq: None,
        api_run: BenchmarkRun,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        api_run.cfg.runtime.api_port = None
        monkeypatch.setattr(
            "aiperf.common.environment.Environment.API_SERVER",
            type("_Fake", (), {"HOST": "0.0.0.0", "PORT": 8080, "CORS_ORIGINS": []})(),
        )
        service = FastAPIService(run=api_run, service_id="api-implicit-port")
        bind = MagicMock(side_effect=OSError("address already in use"))

        with (
            patch("aiperf.api.api_service.socket.socket") as socket_mock,
            patch.object(service, "warning") as warning,
        ):
            socket_mock.return_value.__enter__.return_value.bind = bind
            await service._start_api_server()

        warning.assert_called_once()
        assert service._server is None
        assert service._server_task is None

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
    async def test_stop_holds_grace_window_before_should_exit(
        self, mock_fastapi_service: FastAPIService, time_traveler
    ) -> None:
        """Grace sleep must precede setting should_exit so the listener stays open.

        Uses time_traveler.sleeps_for(grace) to assert the function spends exactly
        the grace duration in asyncio.sleep — any sleep AFTER should_exit was set
        would push the duration past the asserted value.
        """
        mock_server = MagicMock()
        completed = asyncio.Event()

        async def fake_serve():
            """Pretend to be uvicorn.serve(): block until completed is set."""
            await completed.wait()

        mock_server.should_exit = False
        task = asyncio.create_task(fake_serve())
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = task
        completed.set()

        with (
            patch(
                "aiperf.api.api_service.Environment.API_SERVER.POST_COMPLETE_GRACE",
                2.5,
            ),
            time_traveler.sleeps_for(2.5),
        ):
            await mock_fastapi_service._stop_api_server()

        assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_stop_skips_grace_when_zero(
        self, mock_fastapi_service: FastAPIService, time_traveler
    ) -> None:
        """POST_COMPLETE_GRACE=0 must skip the sleep entirely (back-compat path)."""
        mock_server = MagicMock()
        completed = asyncio.Event()

        async def fake_serve():
            """Pretend to be uvicorn.serve(): block until completed is set."""
            await completed.wait()

        task = asyncio.create_task(fake_serve())
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = task
        completed.set()

        with (
            patch(
                "aiperf.api.api_service.Environment.API_SERVER.POST_COMPLETE_GRACE",
                0.0,
            ),
            time_traveler.sleeps_for(0.0),
        ):
            await mock_fastapi_service._stop_api_server()

        assert mock_server.should_exit is True

    @pytest.mark.asyncio
    async def test_stop_skips_grace_when_server_task_done(
        self, mock_fastapi_service: FastAPIService, time_traveler
    ) -> None:
        """Grace must be skipped when there is no live serve task to keep open."""
        mock_server = MagicMock()
        # Finished task simulates a crashed/exited server.
        finished_task = asyncio.create_task(asyncio.sleep(0))
        await finished_task
        mock_fastapi_service._server = mock_server
        mock_fastapi_service._server_task = finished_task

        with (
            patch(
                "aiperf.api.api_service.Environment.API_SERVER.POST_COMPLETE_GRACE",
                5.0,
            ),
            time_traveler.sleeps_for(0.0),
        ):
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
                raise TimeoutError
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
