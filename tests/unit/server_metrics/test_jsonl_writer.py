# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import orjson
import pytest

from aiperf.common.enums import PrometheusMetricType, ServerMetricsFormat
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.resolution.plan import BenchmarkRun
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.jsonl_writer import ServerMetricsJSONLWriter
from tests.unit.conftest import make_run_from_cli
from tests.unit.post_processors.conftest import aiperf_lifecycle


@pytest.fixture
def cfg_server_metrics_export(
    tmp_artifact_dir: Path,
    cli_config: CLIConfig,
) -> BenchmarkRun:
    """Build a v2 BenchmarkRun configured for JSONL server-metrics export.

    The fixture name is preserved for ergonomic reasons (callers reference
    ``cfg_run.cfg.artifacts.server_metrics_export_jsonl_file``); the v1 ->
    v2 conversion happens here.
    """
    user_cfg = CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        artifact_directory=tmp_artifact_dir,
        server_metrics_formats=[ServerMetricsFormat.JSONL],
    )
    return make_run_from_cli(user_cfg)


@pytest.fixture
def sample_server_metrics_record_for_export() -> ServerMetricsRecord:
    """Create sample ServerMetricsRecord for export testing."""
    return ServerMetricsRecord(
        endpoint_url="http://localhost:8081/metrics",
        timestamp_ns=1_000_000_000,
        endpoint_latency_ns=5_000_000,
        metrics={
            "requests_total": MetricFamily(
                type=PrometheusMetricType.COUNTER,
                description="Total requests",
                samples=[
                    MetricSample(
                        labels={"status": "success"},
                        value=100.0,
                    )
                ],
            ),
        },
    )


class TestServerMetricsJSONLWriterInitialization:
    """Test ServerMetricsJSONLWriter initialization."""

    def test_initialization(
        self,
        cfg_server_metrics_export: BenchmarkRun,
    ):
        """Test processor initializes with correct file paths."""
        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        assert (
            processor.output_file
            == cfg_server_metrics_export.cfg.artifacts.server_metrics_export_jsonl_file
        )

    @pytest.mark.asyncio
    async def test_files_cleared_on_initialization(
        self,
        cfg_server_metrics_export: BenchmarkRun,
        tmp_artifact_dir: Path,
    ):
        """Test that output files are cleared on initialization."""
        jsonl_file = tmp_artifact_dir / "server_metrics_export.jsonl"
        jsonl_file.write_text("old data")

        writer = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )
        await writer.initialize()

        try:
            assert not jsonl_file.exists() or jsonl_file.stat().st_size == 0
        finally:
            await writer.stop()


class TestServerMetricsRecordProcessing:
    """Test processing ServerMetricsRecord objects."""

    @pytest.mark.asyncio
    async def test_process_single_record(
        self,
        cfg_server_metrics_export: BenchmarkRun,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test processing single server metrics record."""
        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        output_file = (
            cfg_server_metrics_export.cfg.artifacts.server_metrics_export_jsonl_file
        )
        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 1

        data = orjson.loads(lines[0])
        assert data["endpoint_url"] == "http://localhost:8081/metrics"
        assert data["timestamp_ns"] == 1_000_000_000
        assert data["endpoint_latency_ns"] == 5_000_000
        assert "metrics" in data

    @pytest.mark.asyncio
    async def test_process_multiple_records(
        self,
        cfg_server_metrics_export: BenchmarkRun,
    ):
        """Test processing multiple server metrics records with different metrics."""
        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        async with aiperf_lifecycle(processor):
            for i in range(5):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={
                        "counter": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            description="Test counter",
                            samples=[
                                MetricSample(
                                    labels={},
                                    value=float(
                                        i
                                    ),  # Different values to avoid deduplication
                                )
                            ],
                        ),
                    },
                )
                await processor.process_server_metrics_record(record)

        output_file = (
            cfg_server_metrics_export.cfg.artifacts.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_record_converted_to_slim_format(
        self,
        cfg_server_metrics_export: BenchmarkRun,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test that records are converted to slim format before writing."""
        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        output_file = (
            cfg_server_metrics_export.cfg.artifacts.server_metrics_export_jsonl_file
        )
        data = orjson.loads(output_file.read_text().strip())

        assert "metrics" in data
        assert "requests_total" in data["metrics"]

    @pytest.mark.asyncio
    async def test_histogram_written_in_slim_format(
        self,
        cfg_server_metrics_export: BenchmarkRun,
    ):
        """Test that histogram records are exported correctly in slim format."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "ttft": MetricFamily(
                    type=PrometheusMetricType.HISTOGRAM,
                    description="Time to first token",
                    samples=[
                        MetricSample(
                            labels={"model": "test"},
                            buckets={"0.01": 5.0, "0.1": 15.0, "+Inf": 50.0},
                            sum=5.5,
                            count=50.0,
                        )
                    ],
                )
            },
        )

        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        output_file = (
            cfg_server_metrics_export.cfg.artifacts.server_metrics_export_jsonl_file
        )
        data = orjson.loads(output_file.read_text().strip())

        # Verify histogram is in slim format
        assert "ttft" in data["metrics"]
        samples = data["metrics"]["ttft"]
        assert len(samples) == 1
        sample = samples[0]
        assert sample["labels"] == {"model": "test"}
        assert sample["buckets"] == {"0.01": 5.0, "0.1": 15.0, "+Inf": 50.0}
        assert sample["sum"] == 5.5
        assert sample["count"] == 50.0


class TestDuplicateRecordHandling:
    """Test duplicate record handling."""

    @pytest.mark.asyncio
    async def test_duplicate_records_skipped(
        self,
        cfg_server_metrics_export: BenchmarkRun,
    ):
        """Test that duplicate records are not written to JSONL."""
        unique_record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Total requests",
                    samples=[MetricSample(value=100.0)],
                )
            },
            is_duplicate=False,
        )

        duplicate_record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=2_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Total requests",
                    samples=[MetricSample(value=100.0)],
                )
            },
            is_duplicate=True,
        )

        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(unique_record)
            await processor.process_server_metrics_record(duplicate_record)

        output_file = (
            cfg_server_metrics_export.cfg.artifacts.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Should only have 1 line (duplicate skipped)
        assert len(lines) == 1
        data = orjson.loads(lines[0])
        assert data["timestamp_ns"] == 1_000_000_000


class TestSummarizeMethod:
    """Test summarize method behavior."""

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_list(
        self,
        cfg_server_metrics_export: BenchmarkRun,
    ):
        """Test that summarize returns empty list (export processors don't summarize)."""
        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        async with aiperf_lifecycle(processor):
            results = await processor.summarize()

        assert results == []


class TestInfoMetricsHandling:
    """Test that _info metrics are properly excluded from slim records."""

    @pytest.mark.asyncio
    async def test_info_metrics_excluded_from_slim_records(
        self,
        cfg_server_metrics_export: BenchmarkRun,
    ):
        """Test that metrics ending in _info are excluded from slim JSONL records."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "python_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Python platform information",
                    samples=[
                        MetricSample(
                            labels={"version": "3.10.0"},
                            value=1.0,
                        )
                    ],
                ),
                "process_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Process information",
                    samples=[
                        MetricSample(
                            labels={"pid": "1234"},
                            value=1.0,
                        )
                    ],
                ),
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    description="Total requests",
                    samples=[
                        MetricSample(
                            labels={"status": "success"},
                            value=100.0,
                        )
                    ],
                ),
            },
        )

        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        jsonl_file = (
            cfg_server_metrics_export.cfg.artifacts.server_metrics_export_jsonl_file
        )
        lines = jsonl_file.read_text().strip().split("\n")

        # Should have 1 line
        assert len(lines) == 1

        slim_record = orjson.loads(lines[0])

        # Verify _info metrics are NOT in the slim record
        assert "python_info" not in slim_record["metrics"]
        assert "process_info" not in slim_record["metrics"]

        # Verify regular metrics ARE in the slim record
        assert "requests_total" in slim_record["metrics"]

    @pytest.mark.asyncio
    async def test_mixed_info_and_regular_metrics(
        self,
        cfg_server_metrics_export: BenchmarkRun,
    ):
        """Test handling of multiple _info metrics alongside regular metrics."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "python_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Python info",
                    samples=[MetricSample(labels={}, value=1.0)],
                ),
                "server_info": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Server info",
                    samples=[MetricSample(labels={}, value=1.0)],
                ),
                "cpu_usage": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="CPU usage",
                    samples=[MetricSample(labels={}, value=42.0)],
                ),
                "memory_usage": MetricFamily(
                    type=PrometheusMetricType.GAUGE,
                    description="Memory usage",
                    samples=[MetricSample(labels={}, value=1024.0)],
                ),
            },
        )

        processor = ServerMetricsJSONLWriter(
            run=cfg_server_metrics_export,
            service_id="records-manager",
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        # Check JSONL file
        jsonl_file = (
            cfg_server_metrics_export.cfg.artifacts.server_metrics_export_jsonl_file
        )
        slim_record = orjson.loads(jsonl_file.read_text().strip())

        # Only regular metrics in slim record
        assert len(slim_record["metrics"]) == 2
        assert "cpu_usage" in slim_record["metrics"]
        assert "memory_usage" in slim_record["metrics"]
        assert "python_info" not in slim_record["metrics"]
        assert "server_info" not in slim_record["metrics"]
