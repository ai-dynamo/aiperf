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
from aiperf.common.messages import ProcessAllResultsMessage
from aiperf.common.models import MetricResult
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults
from aiperf.config import BenchmarkRun
from aiperf.config.defaults import OutputDefaults
from aiperf.kubernetes.results_sidecar import (
    CHECKPOINTS_DIR_NAME,
    READY_MARKER_NAME,
    write_ready_marker,
)
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
def results_router(mock_zmq, router_config: BenchmarkRun) -> ResultsRouter:
    return ResultsRouter(
        run=router_config,
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


class TestResultsUploadEndpoint:
    def test_upload_atomically_publishes_complete_raw_file(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output
        content = b'{"record": 1}\n{"record": 2}\n'

        response = results_client.post(
            "/api/results/upload/raw_records_record_processor_0.jsonl",
            files={"file": ("records.jsonl", content, "application/x-ndjson")},
        )

        assert response.status_code == 201
        assert int(response.json()["size"]) == len(content)
        raw_dir = tmp_path / OutputDefaults.RAW_RECORDS_FOLDER
        assert (
            raw_dir / "raw_records_record_processor_0.jsonl"
        ).read_bytes() == content
        assert list(raw_dir.glob("*.uploading")) == []


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
        results_router.run.cfg.artifacts = mock_output

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
        write_ready_marker(tmp_path)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

        response = results_client.get("/api/results/list")
        assert response.status_code == 200
        data = response.json()

        file_names = [f["name"] for f in data["files"]]
        assert "metrics.json" in file_names
        assert "records.jsonl" in file_names
        assert READY_MARKER_NAME not in file_names
        for f in data["files"]:
            assert "size" in f
            assert f["size"] > 0

    def test_list_hides_top_level_until_marker(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        """Without the marker the export may still be in flight; the operator
        must not see partial profile_export_*.json files (sub-second-job race)."""
        (tmp_path / "profile_export_aiperf.json").write_text("{}")
        (tmp_path / "profile_export_aiperf.csv").write_text("a,b\n")

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

        response = results_client.get("/api/results/list")
        assert response.status_code == 200
        names = [f["name"] for f in response.json()["files"]]
        assert names == []

    def test_list_exposes_only_checkpoints_until_marker(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        """Checkpoints bypass the gate so the operator's stagnation-byte signal
        can still observe progress; top-level summary files stay hidden."""
        from aiperf.kubernetes.results_sidecar import write_processing_marker

        (tmp_path / "profile_export_aiperf.json").write_text("{}")
        cp_dir = tmp_path / CHECKPOINTS_DIR_NAME
        cp_dir.mkdir()
        (cp_dir / "cp0.json").write_text("{}")
        (cp_dir / "cp1.json").write_text("{}")
        write_processing_marker(tmp_path)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

        response = results_client.get("/api/results/list")
        payload = response.json()
        names = {f["name"] for f in payload["files"]}
        assert names == {"checkpoints/cp0.json", "checkpoints/cp1.json"}
        assert payload["processing"] is True
        assert payload["ready"] is False

    def test_list_full_after_marker(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        (tmp_path / "profile_export_aiperf.json").write_text("{}")
        cp_dir = tmp_path / CHECKPOINTS_DIR_NAME
        cp_dir.mkdir()
        (cp_dir / "cp0.json").write_text("{}")
        write_ready_marker(tmp_path)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

        response = results_client.get("/api/results/list")
        names = {f["name"] for f in response.json()["files"]}
        assert names == {"profile_export_aiperf.json", "checkpoints/cp0.json"}
        assert READY_MARKER_NAME not in names


class TestResultsFileEndpoints:
    """Test generic result file download endpoint."""

    def test_file_returns_404_when_missing(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        write_ready_marker(tmp_path)
        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

        response = results_client.get("/api/results/files/nonexistent.json")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_file_returns_404_with_ready_marker_detail_when_not_ready(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        """Sub-second jobs can produce partial summary files before export
        finishes; until the marker lands the primary endpoint must refuse."""
        from aiperf.kubernetes.results_sidecar import write_processing_marker

        (tmp_path / "profile_export_aiperf.json").write_text("{}")
        write_processing_marker(tmp_path)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

        response = results_client.get("/api/results/files/profile_export_aiperf.json")
        assert response.status_code == 404
        assert READY_MARKER_NAME in response.json()["detail"]
        assert "processing" in response.json()["detail"].lower()

    def test_file_checkpoint_bypasses_marker_gate(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        cp_dir = tmp_path / CHECKPOINTS_DIR_NAME
        cp_dir.mkdir()
        content = b'{"cp": true}'
        (cp_dir / "cp0.json").write_bytes(content)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

        response = results_client.get(
            "/api/results/files/checkpoints/cp0.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert response.status_code == 200
        assert response.content == content

    def test_file_streams_content_with_correct_headers(
        self,
        results_client: TestClient,
        results_router: ResultsRouter,
        tmp_path,
    ) -> None:
        test_file = tmp_path / "profile_export.json"
        test_file.write_text('{"metrics": {"latency": 100}}')
        write_ready_marker(tmp_path)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

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
        write_ready_marker(tmp_path)
        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

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
        write_ready_marker(tmp_path)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

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
        write_ready_marker(tmp_path)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

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
        write_ready_marker(tmp_path)

        mock_output = MagicMock()
        mock_output.artifact_directory = tmp_path
        results_router.run.cfg.artifacts = mock_output

        response = results_client.get(
            "/api/results/files/data.json",
            headers={"Accept-Encoding": "identity"},
        )
        assert response.status_code == 200
        assert "content-encoding" not in response.headers


class TestFinalResultsHandler:
    """Test the @on_message handler on ResultsRouter."""

    @pytest.mark.asyncio
    async def test_on_process_records_result_stores_results(
        self, results_router: ResultsRouter
    ) -> None:
        assert results_router._final_results is None
        assert results_router._benchmark_complete is False

        result = make_process_records_result(completed=200)
        message = ProcessAllResultsMessage(service_id="records_manager", results=result)
        await results_router._on_process_all_results(message)

        assert results_router._final_results is not None
        assert results_router._final_results.results.completed == 200
        assert results_router._benchmark_complete is False

    @pytest.mark.asyncio
    async def test_on_process_records_result_replaces_previous(
        self, results_router: ResultsRouter
    ) -> None:
        first_result = make_process_records_result(completed=100)
        message1 = ProcessAllResultsMessage(
            service_id="records_manager", results=first_result
        )
        await results_router._on_process_all_results(message1)
        assert results_router._final_results.results.completed == 100

        second_result = make_process_records_result(completed=200)
        message2 = ProcessAllResultsMessage(
            service_id="records_manager", results=second_result
        )
        await results_router._on_process_all_results(message2)
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
        message = ProcessAllResultsMessage(service_id="records_manager", results=result)
        await results_router._on_process_all_results(message)

        assert results_router._final_results.results.completed == completed
        assert results_router._final_results.results.was_cancelled == was_cancelled
        assert results_router._benchmark_complete is False

    @pytest.mark.asyncio
    async def test_on_process_all_results_accepts_optional_summary_payloads(
        self, results_router: ResultsRouter
    ) -> None:
        """ProcessAllResultsMessage carries telemetry/server/energy/exported_artifacts fields.

        The router only stores ``results`` today, but the message contract must
        accept the additional payloads so the unified path stays a single hop.
        """
        result = make_process_records_result(completed=42)
        message = ProcessAllResultsMessage(
            service_id="records_manager",
            results=result,
            telemetry_results=None,
            server_metrics_results=None,
            energy_efficiency_results=None,
            exported_artifacts={},
        )

        await results_router._on_process_all_results(message)

        assert results_router._final_results is not None
        assert results_router._final_results.results.completed == 42
