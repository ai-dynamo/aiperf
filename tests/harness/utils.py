# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import platform
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.context import ForkProcess, SpawnProcess
from pathlib import Path
from typing import Any, TypeAlias

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import (
    InputsFile,
    JsonExportData,
    MetricRecordInfo,
    RawRecordInfo,
    ServerMetricsExportData,
    SessionPayloads,
    SlimRecord,
)


@dataclass(frozen=True)
class AIPerfRunnerResult:
    """AIPerf subprocess result."""

    exit_code: int
    output_dir: Path
    stdout: str = ""
    stderr: str = ""


AIPerfRunnerFn: TypeAlias = Callable[[list[str], float], AIPerfRunnerResult]


@dataclass
class AIPerfMockServer:
    """AIPerfMockServer server info."""

    host: str
    port: int
    url: str
    process: SpawnProcess | ForkProcess

    @property
    def dcgm_urls(self) -> list[str]:
        """AIPerfMockServer server DCGM metrics URLs."""
        return [f"{self.url}/dcgm{i}/metrics" for i in [1, 2]]

    @property
    def server_metrics_urls(self) -> dict[str, str]:
        """Server metrics URLs for different server types."""
        return {
            "aiperf": f"{self.url}/metrics",
            "vllm": f"{self.url}/vllm/metrics",
            "sglang": f"{self.url}/sglang/metrics",
            "trtllm": f"{self.url}/trtllm/metrics",
            "dynamo_frontend": f"{self.url}/dynamo_frontend/metrics",
            "dynamo_prefill": f"{self.url}/dynamo_component/prefill/metrics",
            "dynamo_decode": f"{self.url}/dynamo_component/decode/metrics",
        }

    def get_server_metrics_url(self, *server_types: str) -> list[str]:
        """Get server metrics URLs for specified server types.

        Args:
            *server_types: Server types to get URLs for (e.g., 'vllm', 'sglang')

        Returns:
            List of URLs for the specified server types.
        """
        return [self.server_metrics_urls[t] for t in server_types]


@dataclass(frozen=True)
class VideoDetails:
    """Video file metadata extracted from ffprobe."""

    format_name: str
    duration: float
    codec_name: str
    width: int
    height: int
    fps: float
    pix_fmt: str | None = None
    is_fragmented: bool = False


class AIPerfResults:
    """Simple wrapper for AIPerf results.

    All JSON-based artifacts are loaded as Pydantic models for type safety and validation.
    """

    def __init__(self, result: AIPerfRunnerResult) -> None:
        self.artifacts_dir = result.output_dir
        self.exit_code = result.exit_code
        self.runner_result = result

        self.json = self._load_json_export()
        self.csv = self._load_text_file("**/*aiperf.csv")
        self.inputs = self._load_inputs()
        self.jsonl = self._load_jsonl_records()
        self.raw_records = self._load_raw_records()
        self.log = self._load_text_file("**/logs/aiperf*.log")

        # Server metrics outputs
        self.server_metrics_json = self._load_server_metrics_json()
        self.server_metrics_jsonl = self._load_server_metrics_jsonl()
        self.server_metrics_csv = self._load_text_file("**/*server_metrics_export.csv")
        self.server_metrics_parquet_path = self._find_file(
            "**/*server_metrics_export.parquet"
        )

    @property
    def stdout(self) -> str:
        """Captured stdout from the CLI run."""
        return getattr(self.runner_result, "stdout", "")

    @property
    def stderr(self) -> str:
        """Captured stderr from the CLI run."""
        return getattr(self.runner_result, "stderr", "")

    def _find_file(self, pattern: str) -> Path | None:
        """Find first file matching pattern in artifacts directory."""
        return next(self.artifacts_dir.glob(pattern), None)

    def _load_text_file(self, pattern: str) -> str:
        """Load text file content or return empty string."""
        file_path = self._find_file(pattern)
        return file_path.read_text() if file_path else ""

    def _load_json_export(self) -> JsonExportData | None:
        """Load JSON export as Pydantic model."""
        file_path = self._find_file("**/*aiperf.json")
        if not file_path:
            return None
        return JsonExportData.model_validate_json(file_path.read_text())

    def _load_inputs(self) -> InputsFile | None:
        """Load inputs file as Pydantic model."""
        file_path = self._find_file("**/inputs.json")
        return (
            InputsFile.model_validate_json(file_path.read_text()) if file_path else None
        )

    def _load_jsonl_records(self) -> list[MetricRecordInfo] | None:
        """Load JSONL records as Pydantic models."""
        file_path = self._find_file("**/*profile_export.jsonl")
        if not file_path:
            return None

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(MetricRecordInfo.model_validate_json(line))
        return records

    def _load_raw_records(self) -> list[RawRecordInfo] | None:
        """Load raw records as Pydantic models."""
        file_path = self._find_file("**/*profile_export_raw.jsonl")
        if not file_path:
            return None

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(RawRecordInfo.model_validate_json(line))
        return records

    def _load_server_metrics_json(self) -> ServerMetricsExportData | None:
        """Load server metrics JSON export as Pydantic model."""
        file_path = self._find_file("**/*server_metrics_export.json")
        if not file_path:
            return None
        return ServerMetricsExportData.model_validate_json(file_path.read_text())

    def _load_server_metrics_jsonl(self) -> list[SlimRecord] | None:
        """Load server metrics JSONL records as Pydantic models."""
        file_path = self._find_file("**/*server_metrics_export.jsonl")
        if not file_path:
            return None

        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(SlimRecord.model_validate_json(line))
        return records

    @property
    def has_all_outputs(self) -> bool:
        """Check if all outputs exist."""
        return all(
            (
                self.json is not None,
                bool(self.csv),
                self.inputs is not None,
                self.jsonl is not None,
            )
        )

    def validate_pydantic_models(self) -> None:
        """Validate that all Pydantic models are properly loaded."""
        if self.json:
            assert isinstance(self.json, JsonExportData), (
                "json should be JsonExportData"
            )

        if self.inputs:
            assert isinstance(self.inputs, InputsFile), "inputs should be InputsFile"
            if self.inputs.data:
                assert all(isinstance(s, SessionPayloads) for s in self.inputs.data), (
                    "All inputs.data entries should be SessionPayloads"
                )

        if self.jsonl:
            assert all(isinstance(r, MetricRecordInfo) for r in self.jsonl), (
                "All jsonl records should be MetricRecordInfo"
            )

        if self.raw_records:
            assert all(isinstance(r, RawRecordInfo) for r in self.raw_records), (
                "All raw records should be RawRecordInfo"
            )

    @property
    def request_count(self) -> int:
        """Get number of completed requests from JsonExportData Pydantic model."""
        if not self.json or not self.json.request_count:
            return 0
        return int(self.json.request_count.avg)

    @property
    def has_streaming_metrics(self) -> bool:
        """Check if streaming metrics exist."""
        return self._has_all_metrics(
            (
                "time_to_first_token",
                "inter_token_latency",
                "inter_chunk_latency",
                "time_to_second_token",
            )
        )

    @property
    def has_non_streaming_metrics(self) -> bool:
        """Check if non-streaming metrics exist."""
        return self._has_all_metrics(
            (
                "request_latency",
                "request_throughput",
                "output_token_throughput",
                "output_token_throughput_per_user",
                "output_sequence_length",
                "input_sequence_length",
            )
        )

    def _has_all_metrics(self, metrics: tuple[str, ...]) -> bool:
        """Check if all specified metrics exist in the JsonExportData Pydantic model."""
        return bool(self.json) and all(
            getattr(self.json, metric, None) is not None for metric in metrics
        )

    def _has_input_media(self, media_attr: str) -> bool:
        """Check if inputs contain media of the specified type."""
        if not (self.inputs and self.inputs.data):
            return False

        media_type_map = {
            "images": "image_url",
            "audios": "input_audio",
            "videos": "video_url",
        }
        content_type = media_type_map.get(media_attr, media_attr)

        for session in self.inputs.data:
            if not session.payloads:
                continue

            for payload in session.payloads:
                if self._has_openai_media(payload, content_type):
                    return True
                if self._has_top_level_media(payload, media_attr):
                    return True

        return False

    def _has_openai_media(self, payload: dict[str, Any], content_type: str) -> bool:
        """Check for media in OpenAI message format."""
        for message in payload.get("messages", []):
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get(content_type):
                        return True
        return False

    def _has_top_level_media(self, payload: dict[str, Any], media_attr: str) -> bool:
        """Check for media at top level of payload."""
        media_list = payload.get(media_attr, [])
        return bool(media_list)

    @property
    def has_input_images(self) -> bool:
        """Check if inputs contain images."""
        return self._has_input_media("images")

    @property
    def has_input_audio(self) -> bool:
        """Check if inputs contain audio."""
        return self._has_input_media("audios")

    @property
    def has_input_videos(self) -> bool:
        """Check if inputs contain videos."""
        return self._has_input_media("videos")

    @property
    def has_gpu_telemetry(self) -> bool:
        """Check if GPU telemetry exists."""
        return self.json is not None and self.json.telemetry_data is not None

    # ========================================================================
    # Server Metrics Properties
    # ========================================================================

    @property
    def has_server_metrics(self) -> bool:
        """Check if server metrics data exists."""
        return self.server_metrics_json is not None

    @property
    def has_server_metrics_jsonl(self) -> bool:
        """Check if server metrics JSONL records exist."""
        return (
            self.server_metrics_jsonl is not None and len(self.server_metrics_jsonl) > 0
        )

    @property
    def has_server_metrics_csv(self) -> bool:
        """Check if server metrics CSV exists."""
        return bool(self.server_metrics_csv)

    @property
    def has_server_metrics_parquet(self) -> bool:
        """Check if server metrics parquet file exists."""
        return self.server_metrics_parquet_path is not None

    @property
    def has_all_server_metrics_outputs(self) -> bool:
        """Check if all server metrics output files exist."""
        return (
            self.has_server_metrics
            and self.has_server_metrics_csv
            and self.has_server_metrics_parquet
            and self.has_server_metrics_jsonl
        )

    @property
    def server_metrics_endpoints_configured(self) -> list[str]:
        """Get list of configured server metrics endpoints."""
        if not self.server_metrics_json:
            return []
        return self.server_metrics_json.summary.endpoints_configured

    @property
    def server_metrics_endpoints_successful(self) -> list[str]:
        """Get list of successful server metrics endpoints."""
        if not self.server_metrics_json:
            return []
        return self.server_metrics_json.summary.endpoints_successful

    @property
    def server_metrics_names(self) -> set[str]:
        """Get set of all server metric names collected."""
        if not self.server_metrics_json:
            return set()
        return set(self.server_metrics_json.metrics.keys())

    @property
    def server_metrics_record_count(self) -> int:
        """Get total number of server metrics JSONL records."""
        if not self.server_metrics_jsonl:
            return 0
        return len(self.server_metrics_jsonl)

    def has_server_metric(self, metric_name: str) -> bool:
        """Check if a specific server metric was collected.

        Args:
            metric_name: Full metric name (e.g., 'vllm:kv_cache_usage_perc')

        Returns:
            True if the metric exists in the export data.
        """
        return metric_name in self.server_metrics_names

    def get_server_metric(self, metric_name: str) -> Any:
        """Get server metric data by name.

        Args:
            metric_name: Full metric name (e.g., 'vllm:kv_cache_usage_perc')

        Returns:
            ServerMetricData for the metric, or None if not found.
        """
        if not self.server_metrics_json:
            return None
        return self.server_metrics_json.metrics.get(metric_name)

    def assert_server_metrics_valid(self) -> None:
        """Assert that server metrics are valid and complete."""
        assert self.has_server_metrics, "Server metrics JSON should exist"
        assert self.has_server_metrics_csv, "Server metrics CSV should exist"

        # Validate at least one endpoint was successful
        assert len(self.server_metrics_endpoints_successful) > 0, (
            "At least one server metrics endpoint should be successful"
        )

        # Validate we have metrics data
        assert len(self.server_metrics_names) > 0, (
            "Server metrics should have metric data"
        )

        if self.has_server_metrics_jsonl:
            # Validate JSONL records
            assert self.server_metrics_record_count > 0, (
                "Server metrics JSONL should have records"
            )
            # Validate each JSONL record
            for record in self.server_metrics_jsonl:
                assert isinstance(record, SlimRecord), (
                    "All server metrics records should be ServerMetricsSlimRecord"
                )
                assert record.endpoint_url, "Record should have endpoint_url"
                assert record.timestamp_ns > 0, "Record should have valid timestamp"

    def assert_valid(self) -> None:
        """Assert that the results are valid and all Pydantic models are properly loaded."""
        assert self.has_all_outputs, "Not all output files exist"
        assert self.request_count > 0, "Request count should be greater than 0"
        self.validate_pydantic_models()

    @property
    def average_inter_send_time(self) -> float:
        """Calculate the average inter-send time (time between requests)."""
        start_times_ns = sorted(
            record.metadata.request_start_ns for record in self.jsonl
        )
        inter_send_times_s = [
            (start_times_ns[i] - start_times_ns[i - 1]) / NANOS_PER_SECOND
            for i in range(1, len(start_times_ns))
        ]
        return sum(inter_send_times_s) / len(inter_send_times_s)

    @property
    def average_request_send_rate(self) -> float:
        """Calculate the average request rate based on HTTP start times."""
        start_times_ns = sorted(
            record.metadata.request_start_ns for record in self.jsonl
        )
        duration_s = (start_times_ns[-1] - start_times_ns[0]) / NANOS_PER_SECOND
        return (len(start_times_ns) - 1) / duration_s

    @property
    def average_credit_issue_rate(self) -> float:
        """Calculate the average rate based on credit issuance times.

        This is more accurate for rate limiter testing as it measures
        when the rate limiter issued credits, not when HTTP requests started.
        """
        issue_times_ns = sorted(
            record.metadata.credit_issued_ns
            for record in self.jsonl
            if record.metadata.credit_issued_ns is not None
        )
        if len(issue_times_ns) < 2:
            raise ValueError("Need at least 2 records with credit_issued_ns")
        duration_s = (issue_times_ns[-1] - issue_times_ns[0]) / NANOS_PER_SECOND
        return (len(issue_times_ns) - 1) / duration_s

    @property
    def average_inter_issue_time(self) -> float:
        """Calculate the average inter-issue time (time between credit issuances).

        This is more accurate for rate limiter testing as it measures
        the rate limiter's actual behavior, not HTTP timing.
        """
        issue_times_ns = sorted(
            record.metadata.credit_issued_ns
            for record in self.jsonl
            if record.metadata.credit_issued_ns is not None
        )
        if len(issue_times_ns) < 2:
            raise ValueError("Need at least 2 records with credit_issued_ns")
        inter_issue_times_s = [
            (issue_times_ns[i] - issue_times_ns[i - 1]) / NANOS_PER_SECOND
            for i in range(1, len(issue_times_ns))
        ]
        return sum(inter_issue_times_s) / len(inter_issue_times_s)


class AIPerfCLI:
    """Clean CLI wrapper for running AIPerf benchmarks."""

    def __init__(
        self,
        aiperf_runner: AIPerfRunnerFn,
    ) -> None:
        self._runner = aiperf_runner

    def run_sync(
        self,
        command: str,
        timeout: float = 200.0,
        assert_success: bool = True,
    ) -> AIPerfResults:
        """Run aiperf command and return results."""
        args = self._parse_command(command)
        result = self._runner(args, timeout)
        perf_results = AIPerfResults(result)

        if assert_success and result.exit_code != 0:
            self._raise_failure_error(result, perf_results)

        return perf_results

    async def run(
        self,
        command: str,
        timeout: float = 200.0,
        assert_success: bool = True,
    ) -> AIPerfResults:
        """Run aiperf command and return results.

        Args:
            command: The aiperf command to run (e.g., "aiperf profile ...")
            timeout: Command timeout in seconds
            assert_success: Whether to raise an error if the command fails

        Returns:
            AIPerfResults object containing all output artifacts

        Raises:
            AssertionError: If assert_success is True and the command fails
        """
        args = self._parse_command(command)
        result = await self._runner(args, timeout)
        perf_results = AIPerfResults(result)

        if assert_success and result.exit_code != 0:
            # TODO: HACK: FIXME: This is a temporary workaround for a known issue with macOS where the
            # process can exit with -6 when the process is terminated by a signal, failing the test.
            # More research is needed to root cause this issue, so for now, we will ignore it.
            if result.exit_code == -6 and platform.system() == "Darwin":
                pytest.xfail(
                    "AIPerf exited with SIGABRT (-6) on macOS - known platform issue"
                )

            self._raise_failure_error(result, perf_results)

        return perf_results

    def _raise_failure_error(
        self, result: AIPerfRunnerResult, perf_results: AIPerfResults
    ) -> None:
        """Raise detailed error for failed AIPerf run."""
        error_parts = [f"AIPerf process failed with exit code {result.exit_code}\n"]

        if hasattr(perf_results, "log") and perf_results.log:
            error_parts.append(
                f"\n{'=' * 80}\nAIPERF LOG (logs/aiperf.log):\n{'=' * 80}\n"
                f"{perf_results.log}\n"
            )

        raise AssertionError("".join(error_parts))

    @staticmethod
    def _parse_command(cmd: str) -> list[str]:
        """Parse command string into args.

        Args:
            cmd: Command string to parse

        Returns:
            List of command arguments
        """
        cmd = cmd.strip().replace("\\\n", " ")
        args = shlex.split(cmd)
        return args[1:] if args and args[0] == "aiperf" else args
