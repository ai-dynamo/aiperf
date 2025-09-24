# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.enums import ConsoleExporterType, EndpointType
from aiperf.common.factories import ConsoleExporterFactory
from aiperf.common.models.record_models import ProfileResults
from aiperf.common.models.telemetry_models import TelemetryHierarchy, TelemetryRecord
from aiperf.exporters.gpu_telemetry_console_exporter import GPUTelemetryConsoleExporter


class TestGPUTelemetryEndToEnd:
    """Test suite for end-to-end GPU telemetry functionality."""

    @pytest.fixture
    def user_config(self):
        """Create a test user config with custom server metrics URLs."""
        config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                url="http://test:8000",
                model_names=["test-model"],
            )
        )
        config.server_metrics_url = [
            "http://node1:9401/metrics",
            "http://node2:9401/metrics",
        ]
        return config

    @pytest.fixture
    def service_config(self):
        """Create a test service config."""
        return ServiceConfig()

    @pytest.fixture
    def telemetry_records(self):
        """Create sample telemetry records for testing."""
        return [
            TelemetryRecord(
                timestamp_ns=1000000000,
                dcgm_url="http://node1:9401/metrics",
                gpu_index=0,
                gpu_uuid="GPU-12345678-1234-1234-1234-123456789abc",
                gpu_model_name="NVIDIA H100 80GB HBM3",
                hostname="node1",
                gpu_power_usage=250.5,
                gpu_utilization=85.2,
                gpu_memory_used=40.0,
                gpu_temperature=75.0,
            ),
            TelemetryRecord(
                timestamp_ns=1000000000,
                dcgm_url="http://node2:9401/metrics",
                gpu_index=1,
                gpu_uuid="GPU-87654321-4321-4321-4321-cba987654321",
                gpu_model_name="NVIDIA H100 80GB HBM3",
                hostname="node2",
                gpu_power_usage=300.8,
                gpu_utilization=92.1,
                gpu_memory_used=45.5,
                gpu_temperature=78.5,
            ),
        ]

    @pytest.fixture
    def telemetry_hierarchy(self, telemetry_records):
        """Create a populated telemetry hierarchy for testing."""
        hierarchy = TelemetryHierarchy()
        for record in telemetry_records:
            hierarchy.add_record(record)
        return hierarchy

    def test_user_config_server_metrics_url_default(self):
        """Test that server_metrics_url has correct default value."""
        config = UserConfig(
            endpoint=EndpointConfig(
                type=EndpointType.CHAT,
                url="http://test:8000",
                model_names=["test-model"],
            )
        )
        assert config.server_metrics_url == ["http://localhost:9401/metrics"]

    def test_user_config_server_metrics_url_custom(self, user_config):
        """Test that server_metrics_url can be set to custom values."""
        expected_urls = ["http://node1:9401/metrics", "http://node2:9401/metrics"]
        assert user_config.server_metrics_url == expected_urls

    def test_profile_results_has_telemetry_field(self):
        """Test that ProfileResults model has telemetry_data field defined."""

        # Check that the field exists in the model
        assert "telemetry_data" in ProfileResults.model_fields

        # Check the field metadata
        field_info = ProfileResults.model_fields["telemetry_data"]
        assert "GPU telemetry data" in field_info.description

    def test_gpu_telemetry_console_exporter_registered(self):
        """Test that GPU telemetry console exporter is properly registered."""
        # Test that the exporter type is registered
        assert ConsoleExporterType.TELEMETRY in ConsoleExporterFactory._registry

    @pytest.mark.asyncio
    async def test_gpu_telemetry_console_exporter_verbose_disabled(self):
        """Test that console exporter skips export when verbose is disabled."""
        service_config_mock = Mock()
        service_config_mock.verbose = False

        exporter_config_mock = Mock()
        exporter_config_mock.service_config = service_config_mock
        exporter_config_mock.results = Mock()

        exporter = GPUTelemetryConsoleExporter(exporter_config_mock)
        console_mock = Mock()

        # Should not print anything when verbose is disabled
        await exporter.export(console_mock)
        console_mock.print.assert_not_called()

    @pytest.mark.asyncio
    async def test_gpu_telemetry_console_exporter_no_data(self):
        """Test console exporter behavior when no telemetry data is available."""
        service_config_mock = Mock()
        service_config_mock.verbose = True

        results_mock = Mock()
        results_mock.telemetry_data = None

        exporter_config_mock = Mock()
        exporter_config_mock.service_config = service_config_mock
        exporter_config_mock.results = results_mock

        exporter = GPUTelemetryConsoleExporter(exporter_config_mock)
        console_mock = Mock()

        # Should not print anything when no telemetry data
        await exporter.export(console_mock)
        console_mock.print.assert_not_called()

    def test_telemetry_hierarchy_aggregation(self, telemetry_records):
        """Test that telemetry hierarchy properly aggregates records."""
        hierarchy = TelemetryHierarchy()

        # Add records
        for record in telemetry_records:
            hierarchy.add_record(record)

        # Verify structure
        assert len(hierarchy.dcgm_endpoints) == 2

        # Check node1 data
        node1_data = hierarchy.dcgm_endpoints["http://node1:9401/metrics"]
        assert len(node1_data) == 1
        gpu1_data = node1_data["GPU-12345678-1234-1234-1234-123456789abc"]
        assert gpu1_data.metadata.gpu_index == 0
        assert gpu1_data.metadata.hostname == "node1"

        # Check node2 data
        node2_data = hierarchy.dcgm_endpoints["http://node2:9401/metrics"]
        assert len(node2_data) == 1
        gpu2_data = node2_data["GPU-87654321-4321-4321-4321-cba987654321"]
        assert gpu2_data.metadata.gpu_index == 1
        assert gpu2_data.metadata.hostname == "node2"

    def test_telemetry_manager_urls_from_user_config(self, user_config):
        """Test that the expected URLs are set in user config."""
        expected_urls = ["http://node1:9401/metrics", "http://node2:9401/metrics"]
        assert user_config.server_metrics_url == expected_urls

    def test_console_exporter_type_enum_includes_telemetry(self):
        """Test that ConsoleExporterType includes TELEMETRY."""
        assert hasattr(ConsoleExporterType, "TELEMETRY")
        assert ConsoleExporterType.TELEMETRY == "telemetry"
