# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import JsonExportData
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter
from aiperf.plugin.enums import EndpointType


@pytest.fixture
def sample_records():
    """Create sample records already in display units (ms) as they would be from summarize()."""
    return [
        MetricResult(
            tag="time_to_first_token",
            header="Time to First Token",
            unit="ms",  # Already in display units from summarize()
            avg=123.0,
            min=100.0,
            max=150.0,
            p1=101.0,
            p5=105.0,
            p25=110.0,
            p50=120.0,
            p75=130.0,
            p90=140.0,
            p95=None,
            p99=149.0,
            std=10.0,
        )
    ]


@pytest.fixture
def mock_user_config():
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
        )
    )


@pytest.fixture
def mock_results(sample_records):
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


class TestMetricsJsonExporter:
    @pytest.mark.asyncio
    async def test_metrics_json_exporter_creates_expected_json(
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

            exporter = MetricsJsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            assert expected_file.exists()

            with open(expected_file) as f:
                data = JsonExportData.model_validate_json(f.read())

            assert isinstance(data, JsonExportData)
            assert data.time_to_first_token is not None
            assert data.time_to_first_token.unit == "ms"
            assert data.time_to_first_token.avg == 123.0
            assert data.time_to_first_token.p1 == 101.0

            assert data.input_config is not None
            assert isinstance(data.input_config, UserConfig)
            # TODO: Uncomment this once we have expanded the output config to include all important fields
            # assert "output" in data["input_config"]
            # assert data["input_config"]["output"]["artifact_directory"] == str(
            #     output_dir
            # )

    def test_metrics_json_exporter_inherits_from_base(self, mock_user_config):
        """Verify MetricsJsonExporter inherits from MetricsBaseExporter."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            mock_results = type(
                "MockResults",
                (),
                {
                    "records": [],
                    "start_ns": None,
                    "end_ns": None,
                    "has_results": False,
                    "was_cancelled": False,
                    "error_summary": [],
                },
            )()

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = MetricsJsonExporter(exporter_config)

            from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter

            assert isinstance(exporter, MetricsBaseExporter)

    @pytest.mark.asyncio
    async def test_metrics_json_exporter_uses_base_export(
        self, mock_results, mock_user_config
    ):
        """Verify uses base class export() method."""
        from unittest.mock import AsyncMock

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = MetricsJsonExporter(exporter_config)

            # Mock the base class export method
            from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter

            mock_export = AsyncMock()

            with patch.object(MetricsBaseExporter, "export", mock_export):
                await exporter.export()

                # Verify base export was called
                mock_export.assert_called_once()

    def test_generate_content_uses_instance_data_members(
        self, mock_results, mock_user_config
    ):
        """Verify _generate_content() uses instance data members."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = MetricsJsonExporter(exporter_config)

            content = exporter._generate_content()

            # Should contain data from instance members
            data = json.loads(content)
            assert "input_config" in data

    def test_generate_content_uses_telemetry_results_from_instance(
        self, mock_results, mock_user_config, sample_telemetry_results
    ):
        """Verify _generate_content() uses self._telemetry_results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=sample_telemetry_results,
            )

            exporter = MetricsJsonExporter(exporter_config)

            content = exporter._generate_content()

            # Should contain telemetry data
            data = json.loads(content)
            assert "telemetry_data" in data

    @pytest.mark.asyncio
    async def test_export_calls_generate_content_internally(
        self, mock_results, mock_user_config
    ):
        """Verify export() calls _generate_content() internally."""

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = MetricsJsonExporter(exporter_config)

            test_json_content = '{"test": "data"}'

            with patch.object(
                exporter, "_generate_content", return_value=test_json_content
            ) as mock_generate:
                await exporter.export()

                # Verify _generate_content was called
                mock_generate.assert_called_once()

                # Verify file contains the returned content
                expected_file = (
                    output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
                )
                with open(expected_file) as f:
                    actual_content = f.read()

                assert actual_content == test_json_content


class TestMetricsJsonExporterTelemetry:
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

            exporter = MetricsJsonExporter(exporter_config)
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
            assert "endpoints_configured" in summary
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

            exporter = MetricsJsonExporter(exporter_config)
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

            exporter = MetricsJsonExporter(exporter_config)
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
        """Test that telemetry export handles missing metrics gracefully."""
        from datetime import datetime

        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # Create TelemetryExportData with GPU that has no metrics (empty dict)
            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=["http://localhost:9400/metrics"],
                    endpoints_successful=["http://localhost:9400/metrics"],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(0),
                ),
                endpoints={
                    "localhost:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="Test GPU",
                                gpu_uuid="GPU-123",
                                hostname="test-node",
                                metrics={},  # No metrics
                            ),
                        }
                    ),
                },
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsJsonExporter(exporter_config)
            # Should not raise exception despite missing metrics
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            assert expected_file.exists()

            with open(expected_file) as f:
                data = json.load(f)

            # Should still have telemetry structure even if metrics are empty
            assert "telemetry_data" in data

    @pytest.mark.asyncio
    async def test_json_export_telemetry_with_none_values(
        self, mock_results, mock_user_config
    ):
        """Test JSON export when metric values are None."""
        from datetime import datetime

        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            JsonMetricResult,
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # Create TelemetryExportData with metrics that have None values
            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=["http://localhost:9400/metrics"],
                    endpoints_successful=["http://localhost:9400/metrics"],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(1),
                ),
                endpoints={
                    "localhost:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="Test GPU",
                                gpu_uuid="GPU-123",
                                hostname="test-host",
                                metrics={
                                    # Metric with None values for percentiles
                                    "gpu_power_usage": JsonMetricResult(
                                        unit="W",
                                        avg=100.0,
                                        min=None,
                                        max=None,
                                        p50=None,
                                        p90=None,
                                        p99=None,
                                        std=None,
                                    ),
                                },
                            ),
                        }
                    ),
                },
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsJsonExporter(exporter_config)
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
        from datetime import datetime

        from aiperf.common.models.export_models import (
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # Empty TelemetryExportData - no endpoints
            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=[],
                    endpoints_successful=[],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(1),
                ),
                endpoints={},
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsJsonExporter(exporter_config)
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
        from datetime import datetime

        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            JsonMetricResult,
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # TelemetryExportData already has normalized endpoint keys
            # (normalization happens during conversion from TelemetryResults)
            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=["http://node1.example.com:9400/metrics"],
                    endpoints_successful=["http://node1.example.com:9400/metrics"],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(1),
                ),
                endpoints={
                    "node1.example.com:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="Test GPU",
                                gpu_uuid="GPU-123",
                                hostname="node1",
                                metrics={
                                    "gpu_power_usage": JsonMetricResult(
                                        unit="W",
                                        avg=100.0,
                                        min=100.0,
                                        max=100.0,
                                        std=0.0,
                                    ),
                                },
                            ),
                        }
                    ),
                },
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsJsonExporter(exporter_config)
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
        from datetime import datetime

        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            JsonMetricResult,
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            # Create TelemetryExportData with two endpoints
            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=[
                        "http://node1:9400/metrics",
                        "http://node2:9400/metrics",
                    ],
                    endpoints_successful=[
                        "http://node1:9400/metrics",
                        "http://node2:9400/metrics",
                    ],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(2),
                ),
                endpoints={
                    "node1:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="GPU Model 1",
                                gpu_uuid="GPU-111",
                                hostname="node1",
                                metrics={
                                    "gpu_power_usage": JsonMetricResult(
                                        unit="W",
                                        avg=105.0,
                                        min=100.0,
                                        max=110.0,
                                        std=5.0,
                                    ),
                                },
                            ),
                        }
                    ),
                    "node2:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="GPU Model 2",
                                gpu_uuid="GPU-222",
                                hostname="node2",
                                metrics={
                                    "gpu_power_usage": JsonMetricResult(
                                        unit="W",
                                        avg=205.0,
                                        min=200.0,
                                        max=210.0,
                                        std=5.0,
                                    ),
                                },
                            ),
                        }
                    ),
                },
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsJsonExporter(exporter_config)
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

    @pytest.mark.asyncio
    async def test_json_export_with_hostname_metadata(
        self, mock_results, mock_user_config
    ):
        """Test JSON export includes hostname metadata."""
        from datetime import datetime

        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            JsonMetricResult,
            TelemetryExportData,
            TelemetrySummary,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            mock_user_config.output.artifact_directory = output_dir

            telemetry_results = TelemetryExportData(
                summary=TelemetrySummary(
                    endpoints_configured=["http://localhost:9400/metrics"],
                    endpoints_successful=["http://localhost:9400/metrics"],
                    start_time=datetime.fromtimestamp(0),
                    end_time=datetime.fromtimestamp(1),
                ),
                endpoints={
                    "localhost:9400": EndpointData(
                        gpus={
                            "gpu_0": GpuSummary(
                                gpu_index=0,
                                gpu_name="Test GPU",
                                gpu_uuid="GPU-123",
                                hostname="test-hostname",
                                metrics={
                                    "gpu_power_usage": JsonMetricResult(
                                        unit="W",
                                        avg=100.0,
                                        min=100.0,
                                        max=100.0,
                                        std=0.0,
                                    ),
                                },
                            ),
                        }
                    ),
                },
            )

            exporter_config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=telemetry_results,
            )

            exporter = MetricsJsonExporter(exporter_config)
            await exporter.export()

            expected_file = output_dir / OutputDefaults.PROFILE_EXPORT_AIPERF_JSON_FILE
            with open(expected_file) as f:
                data = json.load(f)

            endpoints = data["telemetry_data"]["endpoints"]
            gpu_summary = endpoints["localhost:9400"]["gpus"]["gpu_0"]
            assert gpu_summary["hostname"] == "test-hostname"
