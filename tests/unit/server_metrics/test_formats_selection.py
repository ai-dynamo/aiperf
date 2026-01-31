# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for server metrics format selection feature."""

import pytest
from pydantic import ValidationError

from aiperf.common.config import UserConfig
from aiperf.common.config.endpoint_config import EndpointConfig
from aiperf.common.enums import ServerMetricsFormat
from aiperf.common.exceptions import DataExporterDisabled, PostProcessorDisabled
from aiperf.common.models.server_metrics_models import (
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
    ServerMetricsResults,
)
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.csv_exporter import ServerMetricsCsvExporter
from aiperf.server_metrics.json_exporter import ServerMetricsJsonExporter
from aiperf.server_metrics.jsonl_writer import ServerMetricsJSONLWriter


@pytest.fixture
def mock_server_metrics_results():
    """Create minimal ServerMetricsResults for testing exporters."""
    endpoint_summaries = {
        "http://localhost:8081/metrics": ServerMetricsEndpointSummary(
            endpoint_url="http://localhost:8081/metrics",
            info=ServerMetricsEndpointInfo(
                total_fetches=10,
                first_fetch_ns=1_000_000_000_000,
                last_fetch_ns=1_100_000_000_000,
                avg_fetch_latency_ms=10.0,
                unique_updates=10,
                first_update_ns=1_000_000_000_000,
                last_update_ns=1_100_000_000_000,
                duration_seconds=100.0,
                avg_update_interval_ms=10000.0,
            ),
            metrics={},
        )
    }

    return ServerMetricsResults(
        benchmark_id="test-benchmark-id",
        server_metrics_data=None,
        endpoint_summaries=endpoint_summaries,
        start_ns=1_000_000_000_000,
        end_ns=1_100_000_000_000,
        endpoints_configured=["http://localhost:8081/metrics"],
        endpoints_successful=["http://localhost:8081/metrics"],
        error_summary=[],
    )


class TestServerMetricsFormatSelection:
    """Test server metrics format selection configuration."""

    def test_default_includes_json_and_csv_only(self, tmp_path):
        """Test that default config enables JSON and CSV (not JSONL due to file size)."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            output={"artifact_dir": str(tmp_path)},
        )

        assert ServerMetricsFormat.JSON in config.server_metrics_formats
        assert ServerMetricsFormat.CSV in config.server_metrics_formats
        assert ServerMetricsFormat.JSONL not in config.server_metrics_formats

    def test_single_format_selection(self, tmp_path):
        """Test selecting a single format."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=[ServerMetricsFormat.JSON],
            output={"artifact_dir": str(tmp_path)},
        )

        assert ServerMetricsFormat.JSON in config.server_metrics_formats
        assert ServerMetricsFormat.CSV not in config.server_metrics_formats
        assert ServerMetricsFormat.JSONL not in config.server_metrics_formats

    def test_multiple_formats_selection(self, tmp_path):
        """Test selecting multiple formats."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=[ServerMetricsFormat.JSON, ServerMetricsFormat.CSV],
            output={"artifact_dir": str(tmp_path)},
        )

        assert ServerMetricsFormat.JSON in config.server_metrics_formats
        assert ServerMetricsFormat.CSV in config.server_metrics_formats
        assert ServerMetricsFormat.JSONL not in config.server_metrics_formats

    def test_invalid_format_raises_error(self, tmp_path):
        """Test that invalid format name raises ValidationError."""
        with pytest.raises(
            ValidationError, match="Input should be 'json', 'csv', 'jsonl' or 'parquet'"
        ):
            UserConfig(
                endpoint=EndpointConfig(
                    model_names=["test-model"],
                    type=EndpointType.CHAT,
                    custom_endpoint="/v1/chat/completions",
                ),
                server_metrics_formats=["invalid_format"],
                output={"artifact_dir": str(tmp_path)},
            )


class TestJsonExporterFormatSelection:
    """Test JSON exporter respects format selection."""

    def test_json_exporter_enabled_when_format_selected(
        self, tmp_path, mock_server_metrics_results
    ):
        """Test JSON exporter is enabled when JSON format is selected."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=[ServerMetricsFormat.JSON],
            output={"artifact_dir": str(tmp_path)},
        )

        exporter_config = ExporterConfig(
            results=None,
            user_config=config,
            service_config=None,
            telemetry_results=None,
            server_metrics_results=mock_server_metrics_results,
        )

        # Should not raise exception
        exporter = ServerMetricsJsonExporter(exporter_config=exporter_config)
        assert exporter is not None

    def test_json_exporter_disabled_when_format_not_selected(
        self, tmp_path, mock_server_metrics_results
    ):
        """Test JSON exporter is disabled when JSON format is not selected."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=[ServerMetricsFormat.CSV],  # Only CSV, no JSON
            output={"artifact_dir": str(tmp_path)},
        )

        exporter_config = ExporterConfig(
            results=None,
            user_config=config,
            service_config=None,
            telemetry_results=None,
            server_metrics_results=mock_server_metrics_results,
        )

        with pytest.raises(DataExporterDisabled, match="format not selected"):
            ServerMetricsJsonExporter(exporter_config=exporter_config)


class TestCsvExporterFormatSelection:
    """Test CSV exporter respects format selection."""

    def test_csv_exporter_enabled_when_format_selected(
        self, tmp_path, mock_server_metrics_results
    ):
        """Test CSV exporter is enabled when CSV format is selected."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=[ServerMetricsFormat.CSV],
            output={"artifact_dir": str(tmp_path)},
        )

        exporter_config = ExporterConfig(
            results=None,
            user_config=config,
            service_config=None,
            telemetry_results=None,
            server_metrics_results=mock_server_metrics_results,
        )

        # Should not raise exception
        exporter = ServerMetricsCsvExporter(exporter_config=exporter_config)
        assert exporter is not None

    def test_csv_exporter_disabled_when_format_not_selected(
        self, tmp_path, mock_server_metrics_results
    ):
        """Test CSV exporter is disabled when CSV format is not selected."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=[ServerMetricsFormat.JSON],  # Only JSON, no CSV
            output={"artifact_dir": str(tmp_path)},
        )

        exporter_config = ExporterConfig(
            results=None,
            user_config=config,
            service_config=None,
            telemetry_results=None,
            server_metrics_results=mock_server_metrics_results,
        )

        with pytest.raises(DataExporterDisabled, match="format not selected"):
            ServerMetricsCsvExporter(exporter_config=exporter_config)


class TestJsonlWriterFormatSelection:
    """Test JSONL writer respects format selection."""

    def test_jsonl_writer_enabled_when_format_selected(self, tmp_path):
        """Test JSONL writer is enabled when JSONL format is selected."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=["jsonl"],
            output={"artifact_dir": str(tmp_path)},
        )

        # Should not raise exception
        writer = ServerMetricsJSONLWriter(user_config=config)
        assert writer is not None

    def test_jsonl_writer_disabled_when_format_not_selected(self, tmp_path):
        """Test JSONL writer is disabled when JSONL format is not selected."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=[
                ServerMetricsFormat.JSON,
                ServerMetricsFormat.CSV,
            ],  # No JSONL
            output={"artifact_dir": str(tmp_path)},
        )

        with pytest.raises(PostProcessorDisabled, match="format not selected"):
            ServerMetricsJSONLWriter(user_config=config)


class TestAllExportersEnabled:
    """Test that all exporters work when all formats are selected."""

    def test_all_exporters_enabled_with_all_formats(
        self, tmp_path, mock_server_metrics_results
    ):
        """Test all exporters are enabled when all formats are selected."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            server_metrics_formats=[
                ServerMetricsFormat.JSON,
                ServerMetricsFormat.CSV,
                ServerMetricsFormat.JSONL,
            ],
            output={"artifact_dir": str(tmp_path)},
        )

        exporter_config = ExporterConfig(
            results=None,
            user_config=config,
            service_config=None,
            telemetry_results=None,
            server_metrics_results=mock_server_metrics_results,
        )

        # All should initialize without exceptions
        json_exporter = ServerMetricsJsonExporter(exporter_config=exporter_config)
        csv_exporter = ServerMetricsCsvExporter(exporter_config=exporter_config)
        jsonl_writer = ServerMetricsJSONLWriter(user_config=config)

        assert json_exporter is not None
        assert csv_exporter is not None
        assert jsonl_writer is not None

    def test_default_config_enables_json_and_csv_only(
        self, tmp_path, mock_server_metrics_results
    ):
        """Test default config enables JSON and CSV (JSONL excluded due to file size)."""
        config = UserConfig(
            endpoint=EndpointConfig(
                model_names=["test-model"],
                type=EndpointType.CHAT,
                custom_endpoint="/v1/chat/completions",
            ),
            # No server_metrics_formats specified - should default to JSON and CSV only
            output={"artifact_dir": str(tmp_path)},
        )

        exporter_config = ExporterConfig(
            results=None,
            user_config=config,
            service_config=None,
            telemetry_results=None,
            server_metrics_results=mock_server_metrics_results,
        )

        # JSON and CSV should initialize without exceptions
        json_exporter = ServerMetricsJsonExporter(exporter_config=exporter_config)
        csv_exporter = ServerMetricsCsvExporter(exporter_config=exporter_config)

        assert json_exporter is not None
        assert csv_exporter is not None

        # JSONL should be disabled by default
        with pytest.raises(PostProcessorDisabled, match="format not selected"):
            ServerMetricsJSONLWriter(user_config=config)
