# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ResultsRouter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from pytest import param
from starlette.testclient import TestClient

from aiperf.api.routers.results import ResultsRouter
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.messages import ProcessRecordsResultMessage
from aiperf.common.models import MetricResult
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults
from tests.unit.api.routers.conftest import make_latency_metric


def make_throughput_metric(
    avg: float = 50.0,
    sum: float = 5000.0,
) -> MetricResult:
    """Create a typical throughput metric for testing."""
    return MetricResult(
        tag="throughput",
        header="Throughput",
        unit="req/s",
        avg=avg,
        sum=sum,
    )


def make_profile_results(
    records: list[MetricResult] | None = None,
    completed: int = 100,
    start_ns: int = 1000000000,
    end_ns: int = 2000000000,
    was_cancelled: bool = False,
) -> ProfileResults:
    """Create a ProfileResults with sensible defaults."""
    if records is None:
        records = [make_latency_metric(), make_throughput_metric()]
    return ProfileResults(
        records=records,
        completed=completed,
        start_ns=start_ns,
        end_ns=end_ns,
        was_cancelled=was_cancelled,
    )


def make_process_records_result(
    records: list[MetricResult] | None = None,
    completed: int = 100,
    was_cancelled: bool = False,
) -> ProcessRecordsResult:
    """Create a ProcessRecordsResult with sensible defaults."""
    profile_results = make_profile_results(
        records=records,
        completed=completed,
        was_cancelled=was_cancelled,
    )
    return ProcessRecordsResult(results=profile_results)


@pytest.fixture
def results_router(
    mock_zmq, router_service_config: ServiceConfig, router_user_config: UserConfig
) -> ResultsRouter:
    return ResultsRouter(
        service_config=router_service_config,
        user_config=router_user_config,
    )


@pytest.fixture
def results_client(results_router: ResultsRouter) -> TestClient:
    app = FastAPI()
    app.state.results = results_router
    app.include_router(results_router.get_router())
    return TestClient(app)


class TestResultsEndpoint:
    """Test the /api/results endpoint for benchmark results retrieval."""

    def test_results_running_no_results(
        self, results_client: TestClient, results_router: ResultsRouter
    ) -> None:
        results_router._final_results = None
        results_router._benchmark_complete = False

        response = results_client.get("/api/results")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "running"
        assert data["results"] is None

    def test_results_complete_with_results(
        self, results_client: TestClient, results_router: ResultsRouter
    ) -> None:
        results_router._final_results = make_process_records_result(
            completed=100, was_cancelled=False
        )
        results_router._benchmark_complete = True

        response = results_client.get("/api/results")
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
        results_client: TestClient,
        results_router: ResultsRouter,
        was_cancelled: bool,
        expected_status: str,
    ) -> None:
        results_router._final_results = make_process_records_result(
            was_cancelled=was_cancelled
        )
        results_router._benchmark_complete = True

        response = results_client.get("/api/results")
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
        results_client: TestClient,
        results_router: ResultsRouter,
        completed_count: int,
    ) -> None:
        results_router._final_results = make_process_records_result(
            completed=completed_count
        )
        results_router._benchmark_complete = True

        response = results_client.get("/api/results")
        data = response.json()
        assert data["results"]["results"]["completed"] == completed_count

    def test_results_contains_metric_records(
        self, results_client: TestClient, results_router: ResultsRouter
    ) -> None:
        latency = make_latency_metric(avg=150.0, p95=200.0, p99=250.0)
        results_router._final_results = make_process_records_result(records=[latency])
        results_router._benchmark_complete = True

        response = results_client.get("/api/results")
        data = response.json()

        records = data["results"]["results"]["records"]
        assert len(records) == 1
        assert records[0]["tag"] == "latency"
        assert records[0]["avg"] == 150.0
        assert records[0]["p95"] == 200.0
        assert records[0]["p99"] == 250.0


class TestResultsListEndpoint:
    """Test the /api/results/list endpoint."""

    def test_list_results_empty_directory(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path / "nonexistent"
        results_router.user_config.output = mock_output

        response = results_client.get("/api/results/list")
        assert response.status_code == 200
        data = response.json()
        assert data["files"] == []

    def test_list_results_with_files(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        (tmp_path / "metrics.json").write_text('{"test": 1}')
        (tmp_path / "records.jsonl").write_text('{"id": 1}')

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.user_config.output = mock_output

        response = results_client.get("/api/results/list")
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
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.user_config.output = mock_output

        response = results_client.get("/api/results/files/nonexistent.json")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_file_streams_content_with_correct_headers(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        test_file = tmp_path / "profile_export.json"
        test_file.write_text('{"metrics": {"latency": 100}}')

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.user_config.output = mock_output

        response = results_client.get(
            "/api/results/files/profile_export.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert response.status_code == 200
        assert "profile_export.json" in response.headers["content-disposition"]
        assert "profile_export.json" in response.headers["x-filename"]

    def test_file_rejects_path_traversal(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.user_config.output = mock_output

        response = results_client.get("/api/results/files/../../../etc/passwd")
        assert response.status_code in (400, 404)

    def test_file_supports_compression(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        test_file = tmp_path / "metrics.json"
        test_file.write_text('{"metrics": {"latency": 100}}')

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.user_config.output = mock_output

        response = results_client.get(
            "/api/results/files/metrics.json",
            headers={"Accept-Encoding": "gzip"},
        )
        assert response.status_code == 200
        assert response.headers["content-encoding"] == "gzip"


class TestResultsFileContentType:
    """Test content type detection by file extension for result files."""

    @pytest.mark.parametrize(
        "filename,expected_content_type",
        [
            param("metrics.json", "application/json", id="json"),
            param("records.jsonl", "application/x-ndjson", id="jsonl"),
            param("data.csv", "text/csv", id="csv"),
            param("data.parquet", "application/vnd.apache.parquet", id="parquet"),
            param("notes.txt", "text/plain", id="txt"),
            param("data.bin", "application/octet-stream", id="unknown-extension"),
        ],
    )  # fmt: skip
    def test_content_type_by_extension(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
        filename: str,
        expected_content_type: str,
    ) -> None:
        test_file = tmp_path / filename
        test_file.write_text("test content")

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.user_config.output = mock_output

        response = results_client.get(
            f"/api/results/files/{filename}",
            headers={"Accept-Encoding": "identity"},
        )
        assert response.status_code == 200
        assert expected_content_type in response.headers["content-type"]

    def test_identity_encoding_omits_content_encoding_header(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        test_file = tmp_path / "data.json"
        test_file.write_text('{"key": "value"}')

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.user_config.output = mock_output

        response = results_client.get(
            "/api/results/files/data.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert response.status_code == 200
        assert "content-encoding" not in response.headers


class TestFinalResultsHandler:
    """Test the @on_message handler from FinalResultsMixin."""

    @pytest.mark.asyncio
    async def test_on_process_records_result_stores_results(
        self, results_router: ResultsRouter
    ) -> None:
        assert results_router._final_results is None
        assert results_router._benchmark_complete is False

        result = make_process_records_result(completed=200)
        message = ProcessRecordsResultMessage(
            service_id="records_manager", results=result
        )
        await results_router._on_process_records_result(message)

        assert results_router._final_results is not None
        assert results_router._final_results.results.completed == 200
        assert results_router._benchmark_complete is True

    @pytest.mark.asyncio
    async def test_on_process_records_result_replaces_previous(
        self, results_router: ResultsRouter
    ) -> None:
        first_result = make_process_records_result(completed=100)
        message1 = ProcessRecordsResultMessage(
            service_id="records_manager", results=first_result
        )
        await results_router._on_process_records_result(message1)
        assert results_router._final_results.results.completed == 100

        second_result = make_process_records_result(completed=200)
        message2 = ProcessRecordsResultMessage(
            service_id="records_manager", results=second_result
        )
        await results_router._on_process_records_result(message2)
        assert results_router._final_results.results.completed == 200

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
        results_router: ResultsRouter,
        completed: int,
        was_cancelled: bool,
    ) -> None:
        result = make_process_records_result(
            completed=completed, was_cancelled=was_cancelled
        )
        message = ProcessRecordsResultMessage(
            service_id="records_manager", results=result
        )
        await results_router._on_process_records_result(message)

        assert results_router._final_results.results.completed == completed
        assert results_router._final_results.results.was_cancelled == was_cancelled
        assert results_router._benchmark_complete is True
