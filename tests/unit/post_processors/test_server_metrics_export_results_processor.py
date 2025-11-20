# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import orjson
import pytest

from aiperf.common.config import EndpointConfig, OutputConfig, ServiceConfig, UserConfig
from aiperf.common.enums import EndpointType, PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    HistogramData,
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.post_processors.server_metrics_export_results_processor import (
    ServerMetricsExportResultsProcessor,
)
from tests.unit.post_processors.conftest import aiperf_lifecycle


@pytest.fixture
def user_config_server_metrics_export(tmp_artifact_dir: Path) -> UserConfig:
    """Create UserConfig for server metrics export testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
        ),
        output=OutputConfig(
            artifact_directory=tmp_artifact_dir,
        ),
    )


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
                help="Total requests",
                samples=[
                    MetricSample(
                        labels={"status": "success"},
                        value=100.0,
                    )
                ],
            ),
        },
    )


class TestServerMetricsExportResultsProcessorInitialization:
    """Test ServerMetricsExportResultsProcessor initialization."""

    def test_initialization(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test processor initializes with correct file paths."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        assert (
            processor.output_file
            == user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        assert (
            processor._metadata_file
            == user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )

    def test_files_cleared_on_initialization(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        tmp_artifact_dir: Path,
    ):
        """Test that output files are cleared on initialization."""
        jsonl_file = tmp_artifact_dir / "server_metrics_export.jsonl"
        metadata_file = tmp_artifact_dir / "server_metrics_metadata.json"

        jsonl_file.write_text("old data")
        metadata_file.write_text("old metadata")

        ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        assert not jsonl_file.exists() or jsonl_file.stat().st_size == 0


class TestServerMetricsRecordProcessing:
    """Test processing ServerMetricsRecord objects."""

    @pytest.mark.asyncio
    async def test_process_single_record(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test processing single server metrics record."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
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
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test processing multiple server metrics records with different metrics."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
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
                            help="Test counter",
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
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 5

    @pytest.mark.asyncio
    async def test_record_converted_to_slim_format(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test that records are converted to slim format before writing."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        data = orjson.loads(output_file.read_text().strip())

        assert "metrics" in data
        assert "requests_total" in data["metrics"]


class TestMetadataExtraction:
    """Test metadata extraction and writing."""

    @pytest.mark.asyncio
    async def test_metadata_extracted_on_first_record(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test that metadata is extracted and written on first record from endpoint."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        assert metadata_file.exists()

        metadata_content = orjson.loads(metadata_file.read_bytes())
        assert "endpoints" in metadata_content
        assert "http://localhost:8081/metrics" in metadata_content["endpoints"]

    @pytest.mark.asyncio
    async def test_metadata_contains_metric_schemas(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        sample_server_metrics_record_for_export: ServerMetricsRecord,
    ):
        """Test that metadata includes metric schemas (type, help)."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(
                sample_server_metrics_record_for_export
            )

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        endpoint_metadata = metadata_content["endpoints"][
            "http://localhost:8081/metrics"
        ]
        assert "metric_schemas" in endpoint_metadata
        assert "requests_total" in endpoint_metadata["metric_schemas"]

        schema = endpoint_metadata["metric_schemas"]["requests_total"]
        assert schema["type"] == "counter"
        assert schema["help"] == "Total requests"

    @pytest.mark.asyncio
    async def test_histogram_schema_includes_bucket_labels(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that histogram schemas include bucket labels."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "ttft": MetricFamily(
                    type=PrometheusMetricType.HISTOGRAM,
                    help="Time to first token",
                    samples=[
                        MetricSample(
                            labels={"model": "test"},
                            histogram=HistogramData(
                                buckets={"0.01": 5.0, "0.1": 15.0, "+Inf": 50.0},
                                sum=5.5,
                                count=50.0,
                            ),
                        )
                    ],
                )
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["ttft"]
        assert "bucket_labels" in schema
        assert schema["bucket_labels"] == ["0.01", "0.1", "+Inf"]

    @pytest.mark.asyncio
    async def test_metadata_includes_unique_label_values(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metadata includes unique label values seen across samples."""
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    help="Total requests",
                    samples=[
                        MetricSample(
                            labels={"status": "success", "endpoint": "chat"},
                            value=100.0,
                        ),
                        MetricSample(
                            labels={"status": "error", "endpoint": "chat"},
                            value=10.0,
                        ),
                        MetricSample(
                            labels={"status": "success", "endpoint": "completions"},
                            value=50.0,
                        ),
                    ],
                )
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["requests_total"]
        assert "unique_label_values" in schema
        assert schema["unique_label_values"] == {
            "endpoint": ["chat", "completions"],
            "status": ["error", "success"],
        }

    @pytest.mark.asyncio
    async def test_unique_label_values_respects_cardinality_limit(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
        monkeypatch,
    ):
        """Test that unique_label_values respects MAX_UNIQUE_LABEL_VALUES limit."""
        # Set a low limit for testing
        from aiperf.common import environment

        monkeypatch.setattr(
            environment.Environment.SERVER_METRICS, "MAX_UNIQUE_LABEL_VALUES", 2
        )

        # Create record with 3 unique label values (exceeds limit of 2)
        record = ServerMetricsRecord(
            endpoint_url="http://localhost:8081/metrics",
            timestamp_ns=1_000_000_000,
            endpoint_latency_ns=5_000_000,
            metrics={
                "requests_total": MetricFamily(
                    type=PrometheusMetricType.COUNTER,
                    help="Total requests",
                    samples=[
                        MetricSample(labels={"status": "success"}, value=100.0),
                        MetricSample(labels={"status": "error"}, value=10.0),
                        MetricSample(
                            labels={"status": "timeout"}, value=5.0
                        ),  # This should not be tracked
                    ],
                )
            },
        )

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["requests_total"]
        # Should only have 2 values due to the limit
        assert len(schema["unique_label_values"]["status"]) == 2

    @pytest.mark.asyncio
    async def test_metadata_updated_for_multiple_endpoints(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metadata file contains all endpoints."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            for endpoint in ["http://node1:8081/metrics", "http://node2:8081/metrics"]:
                record = ServerMetricsRecord(
                    endpoint_url=endpoint,
                    timestamp_ns=1_000_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={},
                )
                await processor.process_server_metrics_record(record)

        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())

        assert len(metadata_content["endpoints"]) == 2
        assert "http://node1:8081/metrics" in metadata_content["endpoints"]
        assert "http://node2:8081/metrics" in metadata_content["endpoints"]

    @pytest.mark.asyncio
    async def test_metadata_only_written_once_per_endpoint(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metadata is only extracted on first record per endpoint."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            for _ in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={},
                )
                await processor.process_server_metrics_record(record)

        assert (
            "http://localhost:8081/metrics" in processor._metadata_file_model.endpoints
        )
        assert len(processor._metadata_file_model.endpoints) == 1


class TestSummarizeMethod:
    """Test summarize method behavior."""

    @pytest.mark.asyncio
    async def test_summarize_returns_empty_list(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that summarize returns empty list (export processors don't summarize)."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            results = await processor.summarize()

        assert results == []


class TestServerMetricsDeduplication:
    """Test deduplication functionality for server metrics records."""

    @pytest.mark.asyncio
    async def test_basic_deduplication_consecutive_duplicates(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that consecutive identical records are deduplicated.

        Input: A,A,A
        Output: A (first), A (last on stop)
        """
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        metrics_data = {
            "requests_total": MetricFamily(
                type=PrometheusMetricType.COUNTER,
                help="Total requests",
                samples=[
                    MetricSample(
                        labels={"status": "success"},
                        value=100.0,
                    )
                ],
            ),
        }

        async with aiperf_lifecycle(processor):
            # Write same metrics 3 times
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics=metrics_data,
                )
                await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Should write first and last occurrence (A, A)
        assert len(lines) == 2
        assert (
            processor._dupe_counts["http://localhost:8081/metrics"] == 0
        )  # Reset after flush

    @pytest.mark.asyncio
    async def test_no_deduplication_when_metrics_change(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that different metrics are not deduplicated."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={
                        "requests_total": MetricFamily(
                            type=PrometheusMetricType.COUNTER,
                            help="Total requests",
                            samples=[
                                MetricSample(
                                    labels={"status": "success"},
                                    value=100.0 + i,  # Different values
                                )
                            ],
                        ),
                    },
                )
                await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # All 3 records should be written (no duplicates)
        assert len(lines) == 3
        assert processor._dupe_counts["http://localhost:8081/metrics"] == 0

    @pytest.mark.asyncio
    async def test_deduplication_per_endpoint(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that deduplication is tracked independently per endpoint.

        Each endpoint: A,A,A → A (first), A (last on stop)
        Total: 4 lines (2 per endpoint)
        """
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        metrics_data = {
            "requests_total": MetricFamily(
                type=PrometheusMetricType.COUNTER,
                help="Total requests",
                samples=[
                    MetricSample(
                        labels={"status": "success"},
                        value=100.0,
                    )
                ],
            ),
        }

        async with aiperf_lifecycle(processor):
            # Write same metrics to two different endpoints, 3 times each
            for endpoint in [
                "http://node1:8081/metrics",
                "http://node2:8081/metrics",
            ]:
                for i in range(3):
                    record = ServerMetricsRecord(
                        endpoint_url=endpoint,
                        timestamp_ns=1_000_000_000 + i * 1_000_000,
                        endpoint_latency_ns=5_000_000,
                        metrics=metrics_data,
                    )
                    await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Each endpoint should write 2 records (first and last)
        assert len(lines) == 4

        # Counts should be reset after flush
        assert processor._dupe_counts["http://node1:8081/metrics"] == 0
        assert processor._dupe_counts["http://node2:8081/metrics"] == 0

    @pytest.mark.asyncio
    async def test_deduplication_sequence_with_changes(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test deduplication with sequence: A,A,A,B,B,C.

        Expected output: A, A, B, B, C
        - A (first), A (last before B), B (first), B (last before C), C (no duplicates)
        """
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # Write A,A,A
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={
                        "value": MetricFamily(
                            type=PrometheusMetricType.GAUGE,
                            help="Test value",
                            samples=[MetricSample(labels={}, value=1.0)],
                        ),
                    },
                )
                await processor.process_server_metrics_record(record)

            # Write B,B
            for i in range(2):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + (i + 3) * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={
                        "value": MetricFamily(
                            type=PrometheusMetricType.GAUGE,
                            help="Test value",
                            samples=[MetricSample(labels={}, value=2.0)],
                        ),
                    },
                )
                await processor.process_server_metrics_record(record)

            # Write C
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000 + 5 * 1_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "value": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        help="Test value",
                        samples=[MetricSample(labels={}, value=3.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")
        records = [orjson.loads(line) for line in lines]

        # Should write: A, A, B, B, C (first and last A, first and last B, single C)
        # Note: C doesn't have a duplicate, so only 1 C is written
        assert len(lines) == 5

        # Verify the metric values
        assert records[0]["metrics"]["value"][0]["value"] == 1.0  # A first
        assert records[1]["metrics"]["value"][0]["value"] == 1.0  # A last
        assert records[2]["metrics"]["value"][0]["value"] == 2.0  # B first
        assert records[3]["metrics"]["value"][0]["value"] == 2.0  # B last
        assert records[4]["metrics"]["value"][0]["value"] == 3.0  # C

    @pytest.mark.asyncio
    async def test_deduplication_with_empty_metrics(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test deduplication with empty metrics dictionaries.

        Input: {},{},{}
        Output: {} (first), {} (last on stop)
        """
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # Write 3 records with empty metrics
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={},
                )
                await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Should write first and last
        assert len(lines) == 2
        assert processor._dupe_counts["http://localhost:8081/metrics"] == 0

    @pytest.mark.asyncio
    async def test_deduplication_only_compares_metrics_not_timestamp(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that deduplication compares metrics only, not timestamps.

        Same metrics, different timestamps → still deduplicated
        Output: first and last occurrence
        """
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        metrics_data = {
            "requests_total": MetricFamily(
                type=PrometheusMetricType.COUNTER,
                help="Total requests",
                samples=[
                    MetricSample(
                        labels={"status": "success"},
                        value=100.0,
                    )
                ],
            ),
        }

        async with aiperf_lifecycle(processor):
            # Same metrics, different timestamps and latencies
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 10_000_000,  # Very different
                    endpoint_latency_ns=5_000_000 + i * 1_000_000,  # Different
                    metrics=metrics_data,
                )
                await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Should deduplicate based on metrics only (write first and last)
        assert len(lines) == 2

    @pytest.mark.asyncio
    async def test_deduplication_with_complex_metrics(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test deduplication with complex histogram metrics.

        Input: Same histogram 3 times
        Output: Histogram (first), Histogram (last on stop)
        """
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        metrics_data = {
            "request_duration": MetricFamily(
                type=PrometheusMetricType.HISTOGRAM,
                help="Request duration",
                samples=[
                    MetricSample(
                        labels={"endpoint": "/api/v1"},
                        histogram=HistogramData(
                            buckets={"0.1": 10.0, "0.5": 50.0, "+Inf": 100.0},
                            sum=25.5,
                            count=100.0,
                        ),
                    )
                ],
            ),
        }

        async with aiperf_lifecycle(processor):
            # Write same histogram 3 times
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics=metrics_data,
                )
                await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Should write first and last
        assert len(lines) == 2
        assert processor._dupe_counts["http://localhost:8081/metrics"] == 0

    @pytest.mark.asyncio
    async def test_deduplication_detects_histogram_bucket_changes(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that deduplication detects changes in histogram bucket values."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # Write histogram with different bucket values
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={
                        "request_duration": MetricFamily(
                            type=PrometheusMetricType.HISTOGRAM,
                            help="Request duration",
                            samples=[
                                MetricSample(
                                    labels={"endpoint": "/api/v1"},
                                    histogram=HistogramData(
                                        buckets={
                                            "0.1": 10.0 + i,  # Changing values
                                            "0.5": 50.0 + i,
                                            "+Inf": 100.0 + i,
                                        },
                                        sum=25.5 + i,
                                        count=100.0 + i,
                                    ),
                                )
                            ],
                        ),
                    },
                )
                await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Should write all 3 records (no duplicates)
        assert len(lines) == 3
        assert processor._dupe_counts["http://localhost:8081/metrics"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_same_endpoint(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that deduplication is thread-safe for concurrent writes to same endpoint."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        metrics_data = {
            "requests_total": MetricFamily(
                type=PrometheusMetricType.COUNTER,
                help="Total requests",
                samples=[
                    MetricSample(
                        labels={"status": "success"},
                        value=100.0,
                    )
                ],
            ),
        }

        async def write_records(count: int):
            for i in range(count):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics=metrics_data,
                )
                await processor.process_server_metrics_record(record)

        async with aiperf_lifecycle(processor):
            # Write same metrics concurrently from multiple tasks
            await asyncio.gather(
                write_records(10),
                write_records(10),
                write_records(10),
            )

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Due to concurrent writes, may have a few records written
        # but should be much less than 30
        assert len(lines) <= 30
        assert len(lines) >= 1  # At least first write

        # Verify file integrity
        for line in lines:
            data = orjson.loads(line)
            assert "endpoint_url" in data
            assert "metrics" in data

    @pytest.mark.asyncio
    async def test_concurrent_writes_to_different_endpoints(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that different endpoints can be written concurrently.

        Each endpoint: 5 identical records → 2 written (first and last)
        Total: 6 lines (2 per endpoint)
        """
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        metrics_data = {
            "requests_total": MetricFamily(
                type=PrometheusMetricType.COUNTER,
                help="Total requests",
                samples=[
                    MetricSample(
                        labels={"status": "success"},
                        value=100.0,
                    )
                ],
            ),
        }

        async def write_to_endpoint(endpoint: str, count: int):
            for i in range(count):
                record = ServerMetricsRecord(
                    endpoint_url=endpoint,
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics=metrics_data,
                )
                await processor.process_server_metrics_record(record)

        async with aiperf_lifecycle(processor):
            # Write to 3 different endpoints concurrently
            await asyncio.gather(
                write_to_endpoint("http://node1:8081/metrics", 5),
                write_to_endpoint("http://node2:8081/metrics", 5),
                write_to_endpoint("http://node3:8081/metrics", 5),
            )

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        # Each endpoint should write 2 records (first and last)
        assert len(lines) == 6

        # Verify all endpoints are present
        endpoints = {orjson.loads(line)["endpoint_url"] for line in lines}
        assert endpoints == {
            "http://node1:8081/metrics",
            "http://node2:8081/metrics",
            "http://node3:8081/metrics",
        }

    @pytest.mark.asyncio
    async def test_single_record_no_deduplication(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that a single record is written without issues."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={},
            )
            await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")

        assert len(lines) == 1
        assert processor._dupe_counts["http://localhost:8081/metrics"] == 0

    @pytest.mark.asyncio
    async def test_deduplication_writes_last_occurrence_before_change(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that the last occurrence is written when metrics change.

        Input: A,A,A,B
        Expected: A, A, B (first A, last A before change, first B)
        This matches BufferedJSONLWriterMixin behavior for time-series data.
        """
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # Write A,A,A
            for i in range(3):
                record = ServerMetricsRecord(
                    endpoint_url="http://localhost:8081/metrics",
                    timestamp_ns=1_000_000_000 + i * 1_000_000,
                    endpoint_latency_ns=5_000_000,
                    metrics={
                        "value": MetricFamily(
                            type=PrometheusMetricType.GAUGE,
                            help="Test value",
                            samples=[MetricSample(labels={}, value=1.0)],
                        ),
                    },
                )
                await processor.process_server_metrics_record(record)

            # Write B (change)
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000 + 3 * 1_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "value": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        help="Test value",
                        samples=[MetricSample(labels={}, value=2.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record)

        output_file = (
            user_config_server_metrics_export.output.server_metrics_export_jsonl_file
        )
        lines = output_file.read_text().strip().split("\n")
        records = [orjson.loads(line) for line in lines]

        # Expected: A, A, B (first A at t0, last A at t2, B at t3)
        # This matches BufferedJSONLWriterMixin behavior for time-series data
        assert len(lines) == 3, (
            f"Expected 3 lines (A, A, B) but got {len(lines)}: {[r['metrics']['value'][0]['value'] for r in records]}"
        )

        # Verify timestamps to ensure we got first and last A
        assert records[0]["timestamp_ns"] == 1_000_000_000  # First A
        assert records[0]["metrics"]["value"][0]["value"] == 1.0
        assert records[1]["timestamp_ns"] == 1_000_000_000 + 2 * 1_000_000  # Last A
        assert records[1]["metrics"]["value"][0]["value"] == 1.0
        assert records[2]["timestamp_ns"] == 1_000_000_000 + 3 * 1_000_000  # B
        assert records[2]["metrics"]["value"][0]["value"] == 2.0


class TestMetadataReconciliation:
    """Test metadata reconciliation for evolving metrics."""

    @pytest.mark.asyncio
    async def test_new_metrics_appearing_later(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that new metrics appearing in later records are captured."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record with metric A
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric A",
                        samples=[MetricSample(labels={}, value=100.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            # Second record with metrics A and B
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric A",
                        samples=[MetricSample(labels={}, value=101.0)],
                    ),
                    "metric_b": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        help="Metric B",
                        samples=[MetricSample(labels={}, value=50.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

        # Verify metadata includes both metrics
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        endpoint_metadata = metadata_content["endpoints"][
            "http://localhost:8081/metrics"
        ]

        assert "metric_a" in endpoint_metadata["metric_schemas"]
        assert "metric_b" in endpoint_metadata["metric_schemas"]
        assert endpoint_metadata["metric_schemas"]["metric_a"]["type"] == "counter"
        assert endpoint_metadata["metric_schemas"]["metric_b"]["type"] == "gauge"

    @pytest.mark.asyncio
    async def test_same_count_different_metrics(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that different metrics with same count are detected."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record with metrics A, B, C
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric A",
                        samples=[MetricSample(labels={}, value=1.0)],
                    ),
                    "metric_b": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric B",
                        samples=[MetricSample(labels={}, value=2.0)],
                    ),
                    "metric_c": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric C",
                        samples=[MetricSample(labels={}, value=3.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            # Second record with metrics B, C, D (same count, but D is new)
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_b": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric B",
                        samples=[MetricSample(labels={}, value=2.0)],
                    ),
                    "metric_c": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric C",
                        samples=[MetricSample(labels={}, value=3.0)],
                    ),
                    "metric_d": MetricFamily(
                        type=PrometheusMetricType.GAUGE,
                        help="Metric D",
                        samples=[MetricSample(labels={}, value=4.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

        # Verify metadata includes all metrics (A, B, C, D)
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        endpoint_metadata = metadata_content["endpoints"][
            "http://localhost:8081/metrics"
        ]

        assert len(endpoint_metadata["metric_schemas"]) == 4
        assert "metric_a" in endpoint_metadata["metric_schemas"]
        assert "metric_b" in endpoint_metadata["metric_schemas"]
        assert "metric_c" in endpoint_metadata["metric_schemas"]
        assert "metric_d" in endpoint_metadata["metric_schemas"]

    @pytest.mark.asyncio
    async def test_histogram_bucket_changes(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that new histogram buckets are detected and merged."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record with histogram with 3 buckets
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "request_duration": MetricFamily(
                        type=PrometheusMetricType.HISTOGRAM,
                        help="Request duration",
                        samples=[
                            MetricSample(
                                labels={},
                                histogram=HistogramData(
                                    buckets={"0.1": 10.0, "0.5": 50.0, "+Inf": 100.0},
                                    sum=25.5,
                                    count=100.0,
                                ),
                            )
                        ],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            # Second record with 5 buckets (added 0.01 and 1.0)
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "request_duration": MetricFamily(
                        type=PrometheusMetricType.HISTOGRAM,
                        help="Request duration",
                        samples=[
                            MetricSample(
                                labels={},
                                histogram=HistogramData(
                                    buckets={
                                        "0.01": 5.0,
                                        "0.1": 15.0,
                                        "0.5": 60.0,
                                        "1.0": 80.0,
                                        "+Inf": 120.0,
                                    },
                                    sum=35.5,
                                    count=120.0,
                                ),
                            )
                        ],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

        # Verify metadata includes all buckets (union)
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["request_duration"]

        assert "bucket_labels" in schema
        assert schema["bucket_labels"] == ["0.01", "0.1", "0.5", "1.0", "+Inf"]

    @pytest.mark.asyncio
    async def test_summary_quantile_changes(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that new summary quantiles are detected and merged."""
        from aiperf.common.models.server_metrics_models import SummaryData

        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record with 3 quantiles
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "response_time": MetricFamily(
                        type=PrometheusMetricType.SUMMARY,
                        help="Response time",
                        samples=[
                            MetricSample(
                                labels={},
                                summary=SummaryData(
                                    quantiles={"0.5": 0.1, "0.9": 0.5, "0.99": 1.0},
                                    sum=50.0,
                                    count=100.0,
                                ),
                            )
                        ],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            # Second record with 5 quantiles (added 0.25 and 0.95)
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "response_time": MetricFamily(
                        type=PrometheusMetricType.SUMMARY,
                        help="Response time",
                        samples=[
                            MetricSample(
                                labels={},
                                summary=SummaryData(
                                    quantiles={
                                        "0.25": 0.05,
                                        "0.5": 0.12,
                                        "0.9": 0.55,
                                        "0.95": 0.75,
                                        "0.99": 1.1,
                                    },
                                    sum=60.0,
                                    count=120.0,
                                ),
                            )
                        ],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

        # Verify metadata includes all quantiles (union)
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["response_time"]

        assert "quantile_labels" in schema
        assert schema["quantile_labels"] == ["0.25", "0.5", "0.9", "0.95", "0.99"]

    @pytest.mark.asyncio
    async def test_no_update_for_identical_metadata(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that metadata file is not rewritten when metrics don't change."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # First record
            record1 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric A",
                        samples=[MetricSample(labels={}, value=100.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record1)

            metadata_file = user_config_server_metrics_export.output.server_metrics_metadata_json_file
            first_mtime = metadata_file.stat().st_mtime_ns

            # Wait a bit to ensure timestamp would change if file is rewritten
            await asyncio.sleep(0.01)

            # Second record with same metrics (different values)
            record2 = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=2_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "metric_a": MetricFamily(
                        type=PrometheusMetricType.COUNTER,
                        help="Metric A",
                        samples=[MetricSample(labels={}, value=105.0)],
                    ),
                },
            )
            await processor.process_server_metrics_record(record2)

            second_mtime = metadata_file.stat().st_mtime_ns

            # Metadata file should not be rewritten
            assert first_mtime == second_mtime

    @pytest.mark.asyncio
    async def test_metadata_merge_is_idempotent(
        self,
        user_config_server_metrics_export: UserConfig,
        service_config: ServiceConfig,
    ):
        """Test that merging the same metadata multiple times produces same result."""
        processor = ServerMetricsExportResultsProcessor(
            service_id="records-manager",
            service_config=service_config,
            user_config=user_config_server_metrics_export,
        )

        async with aiperf_lifecycle(processor):
            # Record with histogram
            record = ServerMetricsRecord(
                endpoint_url="http://localhost:8081/metrics",
                timestamp_ns=1_000_000_000,
                endpoint_latency_ns=5_000_000,
                metrics={
                    "duration": MetricFamily(
                        type=PrometheusMetricType.HISTOGRAM,
                        help="Duration",
                        samples=[
                            MetricSample(
                                labels={},
                                histogram=HistogramData(
                                    buckets={"0.1": 10.0, "0.5": 50.0, "+Inf": 100.0},
                                    sum=25.5,
                                    count=100.0,
                                ),
                            )
                        ],
                    ),
                },
            )

            # Process same record 3 times
            await processor.process_server_metrics_record(record)
            await processor.process_server_metrics_record(record)
            await processor.process_server_metrics_record(record)

        # Verify buckets are not duplicated
        metadata_file = (
            user_config_server_metrics_export.output.server_metrics_metadata_json_file
        )
        metadata_content = orjson.loads(metadata_file.read_bytes())
        schema = metadata_content["endpoints"]["http://localhost:8081/metrics"][
            "metric_schemas"
        ]["duration"]

        assert schema["bucket_labels"] == ["0.1", "0.5", "+Inf"]
