# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.console module.

Focuses on:
- Last benchmark persistence (save/get/clear lifecycle)
- Metrics summary extraction and display logic
- Jobs table rendering with various pod states
- Print helper formatting behavior
"""

from typing import Any
from unittest.mock import patch

import orjson
import pytest
from pytest import param

from aiperf.kubernetes.console import (
    LastBenchmarkInfo,
    clear_last_benchmark,
    get_last_benchmark,
    print_aiperfjob_table,
    print_cr_submission_summary,
    print_detach_info,
    print_file_table,
    print_header,
    print_metrics_summary,
    print_step,
    save_last_benchmark,
    status_log,
)
from aiperf.kubernetes.models import AIPerfJobInfo

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def benchmark_file(tmp_path):
    """Patch _LAST_BENCHMARK_FILE to use a temp directory."""
    fake_path = tmp_path / ".aiperf" / "last_kube_benchmark.json"
    with patch("aiperf.kubernetes.console._LAST_BENCHMARK_FILE", fake_path):
        yield fake_path


@pytest.fixture
def mock_logger():
    """Patch the console module's logger and return the mock."""
    with patch("aiperf.kubernetes.console.logger") as m:
        yield m


# ============================================================
# LastBenchmarkInfo Dataclass
# ============================================================


class TestLastBenchmarkInfo:
    """Verify LastBenchmarkInfo dataclass construction."""

    def test_create_with_required_fields(self) -> None:
        info = LastBenchmarkInfo(job_id="abc", namespace="default")
        assert info.job_id == "abc"
        assert info.namespace == "default"
        assert info.name is None

    def test_create_with_name(self) -> None:
        info = LastBenchmarkInfo(job_id="abc", namespace="ns", name="my-bench")
        assert info.name == "my-bench"


# ============================================================
# save_last_benchmark / get_last_benchmark / clear_last_benchmark
# ============================================================


class TestLastBenchmarkPersistence:
    """Verify save/get/clear lifecycle for last benchmark file."""

    def test_save_and_get_roundtrip(self, benchmark_file) -> None:
        save_last_benchmark("job-1", "ns-1")
        result = get_last_benchmark()
        assert result is not None
        assert result.job_id == "job-1"
        assert result.namespace == "ns-1"
        assert result.name is None

    def test_save_with_name_roundtrip(self, benchmark_file) -> None:
        save_last_benchmark("job-2", "ns-2", name="my-benchmark")
        result = get_last_benchmark()
        assert result is not None
        assert result.name == "my-benchmark"

    def test_get_returns_none_when_no_file(self, benchmark_file) -> None:
        result = get_last_benchmark()
        assert result is None

    def test_clear_removes_file(self, benchmark_file) -> None:
        save_last_benchmark("job-1", "ns-1")
        assert benchmark_file.exists()
        clear_last_benchmark()
        assert not benchmark_file.exists()

    def test_clear_noop_when_no_file(self, benchmark_file) -> None:
        clear_last_benchmark()  # should not raise

    def test_get_returns_none_on_corrupt_json(self, benchmark_file) -> None:
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        benchmark_file.write_bytes(b"not valid json{{{")
        result = get_last_benchmark()
        assert result is None

    def test_get_returns_none_on_missing_keys(self, benchmark_file) -> None:
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        benchmark_file.write_bytes(orjson.dumps({"unrelated": "data"}))
        result = get_last_benchmark()
        assert result is None

    def test_save_creates_parent_directories(self, benchmark_file) -> None:
        assert not benchmark_file.parent.exists()
        save_last_benchmark("job-1", "ns-1")
        assert benchmark_file.exists()

    def test_save_without_name_omits_key(self, benchmark_file) -> None:
        save_last_benchmark("job-1", "ns-1")
        data = orjson.loads(benchmark_file.read_bytes())
        assert "name" not in data

    def test_save_with_empty_name_omits_key(self, benchmark_file) -> None:
        save_last_benchmark("job-1", "ns-1", name="")
        data = orjson.loads(benchmark_file.read_bytes())
        assert "name" not in data


# ============================================================
# print_step
# ============================================================


class TestPrintStep:
    """Verify print_step formatting with and without step counters."""

    def test_print_step_with_counter(self, mock_logger) -> None:
        print_step("Deploying", step=1, total=3)
        mock_logger.info.assert_called_once()
        msg = mock_logger.info.call_args[0][0]
        assert "1/3" in msg
        assert "Deploying" in msg

    def test_print_step_without_counter(self, mock_logger) -> None:
        print_step("Starting")
        mock_logger.info.assert_called_once()
        msg = mock_logger.info.call_args[0][0]
        assert "Starting" in msg
        # Step/total bracket pattern should not appear
        assert "\\[" not in msg or "Starting" in msg

    def test_print_step_only_step_no_total_omits_counter(self, mock_logger) -> None:
        print_step("Partial", step=1, total=None)
        msg = mock_logger.info.call_args[0][0]
        assert "1/" not in msg


# ============================================================
# print_header
# ============================================================


class TestPrintHeader:
    """Verify print_header outputs title and separator."""

    def test_header_logs_title_and_separator(self, mock_logger) -> None:
        print_header("My Section")
        calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert len(calls) == 3  # blank, title, separator
        assert "My Section" in calls[1]

    def test_header_separator_matches_title_length(self, mock_logger) -> None:
        print_header("Test")
        separator_call = mock_logger.info.call_args_list[2][0][0]
        assert "\u2500" * len("Test") in separator_call


# ============================================================
# status_log context manager
# ============================================================


class TestStatusLog:
    """Verify status_log context manager logs on entry."""

    def test_status_log_logs_message(self, mock_logger) -> None:
        with status_log("Waiting for pods"):
            pass
        mock_logger.info.assert_called_once()
        msg = mock_logger.info.call_args[0][0]
        assert "Waiting for pods" in msg

    def test_status_log_yields_control(self, mock_logger) -> None:
        executed = False
        with status_log("test"):
            executed = True
        assert executed


# ============================================================
# print_metrics_summary
# ============================================================


class TestPrintMetricsSummary:
    """Verify metrics summary extracts and displays key metrics."""

    def test_displays_matching_metrics(self, mock_logger) -> None:
        data: dict[str, Any] = {
            "metrics": [
                {"tag": "throughput", "value": 42.5, "unit": "req/s"},
                {"tag": "latency_p50", "value": 100.0, "display_unit": "ms"},
                {"tag": "some_other_metric", "value": 999.0, "unit": "x"},
            ],
        }
        print_metrics_summary(data)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "throughput" in joined
        assert "latency_p50" in joined
        assert "some_other_metric" not in joined

    def test_no_matching_metrics_no_header(self, mock_logger) -> None:
        data: dict[str, Any] = {
            "metrics": [{"tag": "irrelevant", "value": 1.0}],
        }
        print_metrics_summary(data)
        for call in mock_logger.info.call_args_list:
            assert "Benchmark Metrics" not in call[0][0]

    def test_empty_metrics_list(self, mock_logger) -> None:
        print_metrics_summary({"metrics": []})
        assert mock_logger.info.call_count == 0

    def test_no_metrics_key(self, mock_logger) -> None:
        print_metrics_summary({})  # should not raise

    def test_benchmark_id_displayed(self, mock_logger) -> None:
        data: dict[str, Any] = {"benchmark_id": "bench-42", "metrics": []}
        print_metrics_summary(data)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert any("bench-42" in c for c in info_calls)

    def test_display_unit_preferred_over_unit(self, mock_logger) -> None:
        data: dict[str, Any] = {
            "metrics": [
                {
                    "tag": "throughput",
                    "value": 10.0,
                    "unit": "raw_unit",
                    "display_unit": "tok/s",
                },
            ],
        }
        print_metrics_summary(data)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        joined = " ".join(info_calls)
        assert "tok/s" in joined

    def test_limits_to_ten_metrics(self, mock_logger) -> None:
        data: dict[str, Any] = {
            "metrics": [
                {"tag": f"throughput_{i}", "value": float(i), "unit": "x"}
                for i in range(15)
            ],
        }
        print_metrics_summary(data)
        metric_lines = [
            c for c in mock_logger.info.call_args_list if "throughput_" in c[0][0]
        ]
        assert len(metric_lines) <= 10


# ============================================================
# print_file_table
# ============================================================


class TestPrintFileTable:
    """Verify file table display."""

    def test_files_sorted_alphabetically(self, mock_logger) -> None:
        files = [("z_file.json", 100), ("a_file.csv", 200)]
        print_file_table(files)
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        file_lines = [c for c in info_calls if "file" in c.lower()]
        a_idx = next(i for i, c in enumerate(file_lines) if "a_file" in c)
        z_idx = next(i for i, c in enumerate(file_lines) if "z_file" in c)
        assert a_idx < z_idx

    def test_custom_verb(self, mock_logger) -> None:
        print_file_table([("f.txt", 50)], verb="Copied")
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert any("Copied" in c for c in info_calls)

    def test_empty_file_list(self, mock_logger) -> None:
        print_file_table([])
        info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
        assert any("0 file(s)" in c for c in info_calls)


# ============================================================
# print_detach_info
# ============================================================


class TestPrintDetachInfo:
    """Verify detach info includes job commands."""

    def test_detach_with_name(self, mock_logger) -> None:
        print_detach_info("job-1", "default", name="my-bench")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "my-bench" in calls
        assert "deployed successfully" in calls

    def test_detach_without_name(self, mock_logger) -> None:
        print_detach_info("job-1", "default")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "Benchmark" in calls
        assert "deployed successfully" in calls

    def test_detach_includes_watch_command(self, mock_logger) -> None:
        print_detach_info("job-abc", "ns-1")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "aiperf kube watch job-abc --namespace ns-1" in calls

    def test_detach_includes_results_command(self, mock_logger) -> None:
        print_detach_info("job-abc", "ns-1")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "aiperf kube results job-abc --namespace ns-1" in calls


# ============================================================
# Helper: build AIPerfJobInfo for table tests
# ============================================================


def _make_aiperfjob_info(**overrides: Any) -> AIPerfJobInfo:
    """Build an AIPerfJobInfo with sensible defaults, overridden by kwargs."""
    defaults: dict[str, Any] = {
        "name": "bench-run",
        "namespace": "default",
        "phase": "Running",
        "job_id": "abc123",
        "jobset_name": "aiperf-abc123",
        "workers_ready": 2,
        "workers_total": 4,
        "created": "2026-01-15T10:30:00Z",
    }
    defaults.update(overrides)
    return AIPerfJobInfo(**defaults)


# ============================================================
# print_aiperfjob_table
# ============================================================


class TestPrintAIPerfJobTable:
    """Verify AIPerfJob table rendering with various inputs."""

    def test_empty_jobs_list_renders(self) -> None:
        """Test that an empty list produces a table without error."""
        with patch("aiperf.kubernetes.console.console") as mock_console:
            print_aiperfjob_table([])
            mock_console.print.assert_called_once()

    def test_single_running_job_renders(self) -> None:
        """Test that a single running job renders in the table."""
        job = _make_aiperfjob_info(phase="Running", workers_ready=2, workers_total=4)
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="5m"),
        ):
            print_aiperfjob_table([job])
            mock_console.print.assert_called_once()
            table = mock_console.print.call_args[0][0]
            assert table.row_count == 1

    def test_multiple_jobs_render(self) -> None:
        """Test that multiple jobs each get a row in the table."""
        jobs = [
            _make_aiperfjob_info(name="job-1", phase="Running"),
            _make_aiperfjob_info(name="job-2", phase="Completed"),
            _make_aiperfjob_info(name="job-3", phase="Failed"),
        ]
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1h"),
        ):
            print_aiperfjob_table(jobs)
            table = mock_console.print.call_args[0][0]
            assert table.row_count == 3

    def test_narrow_mode_columns(self) -> None:
        """Test that narrow mode has the expected base columns."""
        job = _make_aiperfjob_info()
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_aiperfjob_table([job], wide=False)
            table = mock_console.print.call_args[0][0]
            col_names = [c.header for c in table.columns]
            assert "NAME" in col_names
            assert "NAMESPACE" in col_names
            assert "PHASE" in col_names
            assert "AGE" in col_names
            assert "MODEL" not in col_names
            assert "ENDPOINT" not in col_names
            assert "ERROR" not in col_names

    def test_wide_mode_adds_extra_columns(self) -> None:
        """Test that wide mode includes MODEL, ENDPOINT, ERROR columns."""
        job = _make_aiperfjob_info(model="llama-3", endpoint="http://llm:8000")
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_aiperfjob_table([job], wide=True)
            table = mock_console.print.call_args[0][0]
            col_names = [c.header for c in table.columns]
            assert "MODEL" in col_names
            assert "ENDPOINT" in col_names
            assert "ERROR" in col_names

    def test_progress_shown_when_present(self) -> None:
        """Test that a job with progress_percent renders a percentage."""
        job = _make_aiperfjob_info(progress_percent=42.0)
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_aiperfjob_table([job])
            mock_console.print.assert_called_once()

    def test_throughput_and_latency_shown_when_present(self) -> None:
        """Test that throughput and latency values render without error."""
        job = _make_aiperfjob_info(throughput_rps=120.5, latency_p99_ms=45.3)
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="2m"),
        ):
            print_aiperfjob_table([job])
            mock_console.print.assert_called_once()

    def test_missing_optional_fields_render_dash(self) -> None:
        """Test that missing progress/throughput/latency render without error."""
        job = _make_aiperfjob_info(
            progress_percent=None,
            throughput_rps=None,
            latency_p99_ms=None,
        )
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="3m"),
        ):
            print_aiperfjob_table([job])
            mock_console.print.assert_called_once()

    @pytest.mark.parametrize(
        "phase",
        [
            "Running",
            "Completed",
            "Failed",
            "Cancelled",
            "Pending",
            "Queued",
            "Initializing",
            param("UnknownPhase", id="unrecognized-phase"),
        ],
    )  # fmt: skip
    def test_various_phases_render_without_error(self, phase: str) -> None:
        """Test that all known and unknown phases render without error."""
        job = _make_aiperfjob_info(phase=phase)
        with (
            patch("aiperf.kubernetes.console.console"),
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_aiperfjob_table([job])

    def test_zero_workers_ready_renders(self) -> None:
        """Test that zero workers ready renders with dim styling."""
        job = _make_aiperfjob_info(workers_ready=0, workers_total=4)
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="1m"),
        ):
            print_aiperfjob_table([job])
            mock_console.print.assert_called_once()

    def test_wide_mode_error_field(self) -> None:
        """Test that the error field is shown in wide mode for failed jobs."""
        job = _make_aiperfjob_info(phase="Failed", error="OOMKilled")
        with (
            patch("aiperf.kubernetes.console.console") as mock_console,
            patch("aiperf.kubernetes.cli_helpers.format_age", return_value="10m"),
        ):
            print_aiperfjob_table([job], wide=True)
            table = mock_console.print.call_args[0][0]
            col_names = [c.header for c in table.columns]
            assert "ERROR" in col_names


# ============================================================
# print_cr_submission_summary
# ============================================================


class TestPrintCrSubmissionSummary:
    """Verify CR submission summary output."""

    def test_required_fields_shown(self, mock_logger) -> None:
        """Test that name, namespace, and image always appear."""
        print_cr_submission_summary(
            name="my-bench",
            namespace="bench-ns",
            image="nvcr.io/aiperf:latest",
        )
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "my-bench" in calls
        assert "bench-ns" in calls
        assert "nvcr.io/aiperf:latest" in calls

    def test_header_shown(self, mock_logger) -> None:
        """Test that the header line is present."""
        print_cr_submission_summary(name="n", namespace="ns", image="img")
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "AIPerf Kubernetes Benchmark" in calls

    def test_optional_endpoint_shown(self, mock_logger) -> None:
        """Test that endpoint_url appears when provided."""
        print_cr_submission_summary(
            name="n",
            namespace="ns",
            image="img",
            endpoint_url="http://llm:8000/v1",
        )
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "http://llm:8000/v1" in calls

    def test_optional_endpoint_omitted_when_none(self, mock_logger) -> None:
        """Test that endpoint label is not emitted when endpoint_url is None."""
        print_cr_submission_summary(
            name="n", namespace="ns", image="img", endpoint_url=None
        )
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "Endpoint" not in calls

    def test_optional_model_names_shown(self, mock_logger) -> None:
        """Test that model_names appear comma-separated when provided."""
        print_cr_submission_summary(
            name="n",
            namespace="ns",
            image="img",
            model_names=["llama-3", "gpt-4"],
        )
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "llama-3, gpt-4" in calls

    def test_optional_model_names_omitted_when_none(self, mock_logger) -> None:
        """Test that model label is not emitted when model_names is None."""
        print_cr_submission_summary(
            name="n", namespace="ns", image="img", model_names=None
        )
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "Model" not in calls

    def test_optional_connections_per_worker_shown(self, mock_logger) -> None:
        """Test that connections_per_worker appears when provided."""
        print_cr_submission_summary(
            name="n",
            namespace="ns",
            image="img",
            connections_per_worker=16,
        )
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "16" in calls

    def test_optional_connections_per_worker_omitted_when_none(
        self, mock_logger
    ) -> None:
        """Test that connections label is not emitted when value is None."""
        print_cr_submission_summary(
            name="n",
            namespace="ns",
            image="img",
            connections_per_worker=None,
        )
        calls = " ".join(c[0][0] for c in mock_logger.info.call_args_list)
        assert "Connections" not in calls

    def test_to_stderr_uses_stderr_logger(self) -> None:
        """Test that to_stderr routes output to the stderr logger."""
        with (
            patch("aiperf.kubernetes.console.logger") as mock_logger,
            patch("aiperf.kubernetes.console._stderr_logger") as mock_stderr,
        ):
            print_cr_submission_summary(
                name="n",
                namespace="ns",
                image="img",
                to_stderr=True,
            )
            assert mock_stderr.info.call_count > 0
            assert mock_logger.info.call_count == 0

    def test_to_stdout_by_default(self, mock_logger) -> None:
        """Test that output goes to stdout logger by default."""
        print_cr_submission_summary(name="n", namespace="ns", image="img")
        assert mock_logger.info.call_count > 0
