# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.constants import NANOS_PER_MILLIS
from aiperf.common.enums import EndpointType
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.json_exporter import JsonExporter


@pytest.fixture
def sample_records():
    """Sample metric records for JSON export testing."""
    return [
        MetricResult(
            tag="ttft",
            header="Time to First Token",
            unit="ns",
            avg=123.0 * NANOS_PER_MILLIS,
            min=100.0 * NANOS_PER_MILLIS,
            max=150.0 * NANOS_PER_MILLIS,
            p1=101.0 * NANOS_PER_MILLIS,
            p5=105.0 * NANOS_PER_MILLIS,
            p25=110.0 * NANOS_PER_MILLIS,
            p50=120.0 * NANOS_PER_MILLIS,
            p75=130.0 * NANOS_PER_MILLIS,
            p90=140.0 * NANOS_PER_MILLIS,
            p95=None,
            p99=149.0 * NANOS_PER_MILLIS,
            std=10.0 * NANOS_PER_MILLIS,
        )
    ]


@pytest.fixture
def mock_user_config():
    """Mock user configuration for JSON export testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
        )
    )


@pytest.fixture
def mock_results(sample_records):
    """Mock results object for JSON export testing."""

    class MockResults:
        def __init__(self, metrics):
            self.metrics = metrics
            self.start_ns = None
            self.end_ns = None

        @property
        def records(self):
            return self.metrics

        @property
        def has_results(self):
            return bool(self.metrics)

        @property
        def was_cancelled(self):
            return False

        @property
        def error_summary(self):
            return []

    return MockResults(sample_records)


class TestJsonExporter:
    @pytest.mark.asyncio
    async def test_json_exporter_creates_expected_json(
        self, mock_results, mock_user_config
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            assert expected_file.exists()

            with open(expected_file) as f:
                data = json.load(f)

            assert "records" in data
            records = data["records"]
            assert isinstance(records, dict)
            assert len(records) == 1
            assert "ttft" in records
            assert records["ttft"]["unit"] == "ms"
            assert records["ttft"]["avg"] == 123.0
            assert records["ttft"]["p1"] == 101.0

            assert "input_config" in data
            assert isinstance(data["input_config"], dict)
            # TODO: Uncomment this once we have expanded the output config to include all important fields
            # assert "output" in data["input_config"]
            # assert data["input_config"]["output"]["artifact_directory"] == str(
            #     output_dir
            # )


class TestJsonExporterTelemetry:
    """Test JSON export with telemetry data."""

    @pytest.mark.asyncio
    async def test_json_export_with_telemetry_data(
        self, mock_results, mock_user_config, sample_telemetry_results
    ):
        """Test that JSON export includes telemetry_data field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=sample_telemetry_results,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            assert expected_file.exists()

            with open(expected_file) as f:
                data = json.load(f)

            # Verify telemetry_data exists
            assert "telemetry_data" in data
            assert data["telemetry_data"] is not None

            # Verify summary section
            assert "summary" in data["telemetry_data"]
            summary = data["telemetry_data"]["summary"]
            assert "endpoints_tested" in summary
            assert "endpoints_successful" in summary

            # Verify endpoints section with GPU data
            assert "endpoints" in data["telemetry_data"]
            endpoints = data["telemetry_data"]["endpoints"]
            assert len(endpoints) > 0

            # Check for GPU metrics in at least one endpoint
            first_endpoint = list(endpoints.values())[0]
            assert "gpus" in first_endpoint

    @pytest.mark.asyncio
    async def test_json_export_without_telemetry_data(
        self, mock_results, mock_user_config
    ):
        """Test that JSON export works when telemetry_results is None."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            assert expected_file.exists()

            with open(expected_file) as f:
                data = json.load(f)

            # telemetry_data should not be present or be null
            assert "telemetry_data" not in data or data.get("telemetry_data") is None

    @pytest.mark.asyncio
    async def test_json_export_telemetry_structure(
        self, mock_results, mock_user_config, sample_telemetry_results
    ):
        """Test that JSON telemetry data has correct structure with metrics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=sample_telemetry_results,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            with open(expected_file) as f:
                data = json.load(f)

            endpoints = data["telemetry_data"]["endpoints"]
            # Get first GPU from first endpoint
            first_endpoint = list(endpoints.values())[0]
            first_gpu = list(first_endpoint["gpus"].values())[0]

            # Verify GPU metadata
            assert "gpu_index" in first_gpu
            assert "gpu_name" in first_gpu
            assert "gpu_uuid" in first_gpu

            # Verify metrics structure
            assert "metrics" in first_gpu
            metrics = first_gpu["metrics"]

            # Check for at least one metric
            assert len(metrics) > 0

            # Check that metrics have statistical data
            first_metric = list(metrics.values())[0]
            assert "avg" in first_metric
            assert "min" in first_metric
            assert "max" in first_metric
            assert "unit" in first_metric

    @pytest.mark.asyncio
    async def test_json_export_telemetry_exception_handling(
        self, mock_results, mock_user_config
    ):
        """Test that telemetry export handles metric retrieval exceptions."""
        from unittest.mock import Mock

        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            TelemetryHierarchy,
            TelemetryResults,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # Create GPU data that will fail on metric retrieval
            gpu_data = Mock(spec=GpuTelemetryData)
            gpu_data.metadata = GpuMetadata(
                gpu_index=0,
                model_name="Test GPU",
                gpu_uuid="GPU-123",
            )
            gpu_data.get_metric_result = Mock(
                side_effect=Exception("Metric not available")
            )

            hierarchy = TelemetryHierarchy()
            hierarchy.dcgm_endpoints = {
                "http://localhost:9400/metrics": {"GPU-123": gpu_data}
            }

            telemetry_results = TelemetryResults(
                telemetry_data=hierarchy,
                start_ns=0,
                end_ns=0,
                endpoints_tested=["http://localhost:9400/metrics"],
                endpoints_successful=["http://localhost:9400/metrics"],
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = JsonExporter(exporter_config)
            # Should not raise exception despite metric retrieval failures
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            assert expected_file.exists()

            with open(expected_file) as f:
                data = json.load(f)

            # Should still have telemetry structure even if metrics fail
            assert "telemetry_data" in data

    @pytest.mark.asyncio
    async def test_json_export_telemetry_with_none_values(
        self, mock_results, mock_user_config
    ):
        """Test JSON export when metric values are None."""
        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            TelemetryHierarchy,
            TelemetryResults,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # Create GPU data with None values (no snapshots = no data)
            gpu_data = GpuTelemetryData(
                metadata=GpuMetadata(
                    gpu_index=0,
                    model_name="Test GPU",
                    gpu_uuid="GPU-123",
                )
            )
            # No snapshots added = empty metrics

            hierarchy = TelemetryHierarchy()
            hierarchy.dcgm_endpoints = {
                "http://localhost:9400/metrics": {"GPU-123": gpu_data}
            }

            telemetry_results = TelemetryResults(
                telemetry_data=hierarchy,
                start_ns=0,
                end_ns=0,
                endpoints_tested=["http://localhost:9400/metrics"],
                endpoints_successful=["http://localhost:9400/metrics"],
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            with open(expected_file) as f:
                data = json.load(f)

            # Should handle None values gracefully
            assert "telemetry_data" in data

    @pytest.mark.asyncio
    async def test_json_export_telemetry_empty_hierarchy(
        self, mock_results, mock_user_config
    ):
        """Test JSON export with empty telemetry hierarchy."""
        from aiperf.common.models import TelemetryHierarchy, TelemetryResults

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # Empty hierarchy
            telemetry_results = TelemetryResults(
                telemetry_data=TelemetryHierarchy(),
                start_ns=0,
                end_ns=0,
                endpoints_tested=[],
                endpoints_successful=[],
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            with open(expected_file) as f:
                data = json.load(f)

            # Should have telemetry_data section but empty
            assert "telemetry_data" in data
            endpoints = data["telemetry_data"]["endpoints"]
            assert endpoints == {}

    @pytest.mark.asyncio
    async def test_json_export_telemetry_endpoint_normalization(
        self, mock_results, mock_user_config
    ):
        """Test that endpoint URLs are normalized in JSON output."""
        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            TelemetryHierarchy,
            TelemetryResults,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            gpu_data = GpuTelemetryData(
                metadata=GpuMetadata(
                    gpu_index=0,
                    model_name="Test GPU",
                    gpu_uuid="GPU-123",
                )
            )
            gpu_data.time_series.append_snapshot(
                {"gpu_power_usage": 100.0}, timestamp_ns=1000000000
            )

            hierarchy = TelemetryHierarchy()
            hierarchy.dcgm_endpoints = {
                "http://node1.example.com:9400/metrics": {"GPU-123": gpu_data}
            }

            telemetry_results = TelemetryResults(
                telemetry_data=hierarchy,
                start_ns=0,
                end_ns=0,
                endpoints_tested=["http://node1.example.com:9400/metrics"],
                endpoints_successful=["http://node1.example.com:9400/metrics"],
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            with open(expected_file) as f:
                data = json.load(f)

            endpoints = data["telemetry_data"]["endpoints"]
            # Check that endpoint was normalized (removed http:// and /metrics)
            assert "node1.example.com:9400" in endpoints

    @pytest.mark.asyncio
    async def test_json_export_telemetry_multi_endpoint(
        self, mock_results, mock_user_config
    ):
        """Test JSON export with multiple DCGM endpoints."""
        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            TelemetryHierarchy,
            TelemetryResults,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # Create two endpoints with GPU data
            gpu1_data = GpuTelemetryData(
                metadata=GpuMetadata(
                    gpu_index=0,
                    model_name="GPU Model 1",
                    gpu_uuid="GPU-111",
                )
            )
            gpu1_data.time_series.append_snapshot(
                {"gpu_power_usage": 100.0}, timestamp_ns=1000000000
            )
            gpu1_data.time_series.append_snapshot(
                {"gpu_power_usage": 110.0}, timestamp_ns=2000000000
            )

            gpu2_data = GpuTelemetryData(
                metadata=GpuMetadata(
                    gpu_index=0,
                    model_name="GPU Model 2",
                    gpu_uuid="GPU-222",
                )
            )
            gpu2_data.time_series.append_snapshot(
                {"gpu_power_usage": 200.0}, timestamp_ns=1000000000
            )
            gpu2_data.time_series.append_snapshot(
                {"gpu_power_usage": 210.0}, timestamp_ns=2000000000
            )

            hierarchy = TelemetryHierarchy()
            hierarchy.dcgm_endpoints = {
                "http://node1:9400/metrics": {"GPU-111": gpu1_data},
                "http://node2:9400/metrics": {"GPU-222": gpu2_data},
            }

            telemetry_results = TelemetryResults(
                telemetry_data=hierarchy,
                start_ns=0,
                end_ns=0,
                endpoints_tested=[
                    "http://node1:9400/metrics",
                    "http://node2:9400/metrics",
                ],
                endpoints_successful=[
                    "http://node1:9400/metrics",
                    "http://node2:9400/metrics",
                ],
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = JsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            with open(expected_file) as f:
                data = json.load(f)

            endpoints = data["telemetry_data"]["endpoints"]
            # Should have both endpoints
            assert "node1:9400" in endpoints
            assert "node2:9400" in endpoints

            # Check GPU data exists for both
            assert "gpus" in endpoints["node1:9400"]
            assert "gpus" in endpoints["node2:9400"]
