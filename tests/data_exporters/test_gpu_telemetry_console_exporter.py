# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPUTelemetryConsoleExporter."""

import pytest
from rich.console import Console

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.enums import EndpointType
from aiperf.common.models import ProfileResults
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.gpu_telemetry_console_exporter import (
    GPUTelemetryConsoleExporter,
)


@pytest.fixture
def mock_endpoint_config():
    """Create a mock endpoint configuration."""
    return EndpointConfig(
        type=EndpointType.CHAT,
        streaming=True,
        model_names=["test-model"],
    )


@pytest.fixture
def mock_user_config(mock_endpoint_config):
    """Create a mock user configuration."""
    return UserConfig(endpoint=mock_endpoint_config)


@pytest.fixture
def mock_profile_results():
    """Create mock profile results."""
    return ProfileResults(
        records=[],
        start_ns=0,
        end_ns=0,
        completed=0,
    )


class TestGPUTelemetryConsoleExporter:
    """Test suite for GPUTelemetryConsoleExporter."""

    @pytest.mark.asyncio
    async def test_export_verbose_disabled_no_output(
        self, mock_profile_results, mock_user_config, sample_telemetry_results, capsys
    ):
        """Test that export does not print when verbose mode is disabled."""
        service_config = ServiceConfig(verbose=False)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=sample_telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console()
        await exporter.export(console)

        output = capsys.readouterr().out
        assert "GPU Telemetry" not in output
        assert "H100" not in output

    @pytest.mark.asyncio
    async def test_export_none_telemetry_results_no_output(
        self, mock_profile_results, mock_user_config, capsys
    ):
        """Test that export does not print when telemetry_results is None."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=None,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console()
        await exporter.export(console)

        output = capsys.readouterr().out
        assert "GPU Telemetry" not in output

    @pytest.mark.asyncio
    async def test_export_with_telemetry_data(
        self, mock_profile_results, mock_user_config, sample_telemetry_results, capsys
    ):
        """Test export with real telemetry data displays correctly."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=sample_telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)
        await exporter.export(console)

        output = capsys.readouterr().out
        assert "GPU Telemetry Summary" in output
        assert "DCGM endpoints reachable" in output
        assert "H100" in output or "A100" in output
        assert "Power Usage" in output
        assert "Utilization" in output

    @pytest.mark.asyncio
    async def test_export_displays_all_endpoints(
        self, mock_profile_results, mock_user_config, sample_telemetry_results, capsys
    ):
        """Test that all endpoints are displayed in the summary."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=sample_telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)
        await exporter.export(console)

        output = capsys.readouterr().out
        assert "localhost:9400" in output
        assert "remote-node:9400" in output
        assert "2/2 DCGM endpoints reachable" in output

    @pytest.mark.asyncio
    async def test_export_shows_failed_endpoints(
        self,
        mock_profile_results,
        mock_user_config,
        sample_telemetry_results_with_failures,
        capsys,
    ):
        """Test that failed endpoints are marked appropriately."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=sample_telemetry_results_with_failures,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)
        await exporter.export(console)

        output = capsys.readouterr().out
        assert "1/3 DCGM endpoints reachable" in output
        assert "localhost:9400" in output
        assert "unreachable-node:9400" in output or "unreachable" in output
        assert "❌" in output or "unreachable" in output

    @pytest.mark.asyncio
    async def test_export_empty_telemetry_shows_message(
        self, mock_profile_results, mock_user_config, empty_telemetry_results, capsys
    ):
        """Test that empty telemetry data shows appropriate message."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=empty_telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)
        await exporter.export(console)

        output = capsys.readouterr().out
        assert (
            "No GPU telemetry data collected" in output
            or "Unreachable endpoints" in output
        )
        assert "unreachable-1:9400" in output or "unreachable-2:9400" in output

    @pytest.mark.asyncio
    async def test_get_renderable_with_multi_gpu_data(
        self, mock_profile_results, mock_user_config, sample_telemetry_results
    ):
        """Test get_renderable method with multi-GPU data."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=sample_telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        renderable = exporter.get_renderable(sample_telemetry_results)

        # Verify renderable is created without errors
        assert renderable is not None

    def test_normalize_endpoint_display(self):
        """Test endpoint URL normalization for display."""
        from aiperf.exporters.display_units_utils import normalize_endpoint_display

        # Standard http URL
        assert (
            normalize_endpoint_display("http://localhost:9400/metrics")
            == "localhost:9400"
        )

        # https URL
        assert normalize_endpoint_display("https://node1:9400/metrics") == "node1:9400"

        # URL with path
        assert (
            normalize_endpoint_display("http://node1:9400/api/metrics")
            == "node1:9400/api"
        )

        # URL without /metrics suffix
        assert normalize_endpoint_display("http://node1:9400/data") == "node1:9400/data"

        # URL with just host
        assert normalize_endpoint_display("http://node1:9400") == "node1:9400"

    @pytest.mark.asyncio
    async def test_export_displays_all_metrics(
        self, mock_profile_results, mock_user_config, sample_telemetry_results, capsys
    ):
        """Test that all key metrics are displayed in the output."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=sample_telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)
        await exporter.export(console)

        output = capsys.readouterr().out
        # Check for key metrics
        assert "Power Usage" in output
        assert "Energy Consumption" in output
        assert "Utilization" in output
        assert "Memory Used" in output
        assert "Temperature" in output
        # Check for statistical columns
        assert "avg" in output or "min" in output or "max" in output

    @pytest.mark.asyncio
    async def test_export_with_error_summary(
        self, mock_profile_results, mock_user_config, capsys
    ):
        """Test that error summary is displayed when errors occurred."""
        from aiperf.common.models import (
            ErrorDetails,
            ErrorDetailsCount,
            TelemetryHierarchy,
            TelemetryResults,
        )

        service_config = ServiceConfig(verbose=True)

        # Create telemetry results with errors but no data
        telemetry_results = TelemetryResults(
            telemetry_data=TelemetryHierarchy(),
            start_ns=0,
            end_ns=0,
            endpoints_tested=["http://failed-node:9400/metrics"],
            endpoints_successful=[],
            error_summary=[
                ErrorDetailsCount(
                    error_details=ErrorDetails(message="Connection timeout", code=408),
                    count=5,
                ),
                ErrorDetailsCount(
                    error_details=ErrorDetails(message="Connection refused", code=503),
                    count=1,
                ),
            ],
        )

        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)
        await exporter.export(console)

        output = capsys.readouterr().out
        assert "No GPU telemetry data collected" in output
        assert "Errors encountered" in output
        assert "Connection timeout" in output
        assert "5 occurrences" in output
        assert "Connection refused" in output

    @pytest.mark.asyncio
    async def test_export_handles_metric_retrieval_exceptions(
        self, mock_profile_results, mock_user_config, capsys
    ):
        """Test that metric retrieval exceptions are handled gracefully."""
        from unittest.mock import Mock

        from aiperf.common.models import (
            GpuMetadata,
            GpuTelemetryData,
            TelemetryHierarchy,
            TelemetryResults,
        )

        service_config = ServiceConfig(verbose=True)

        # Create telemetry results with GPU data that will fail on metric retrieval
        gpu_data = Mock(spec=GpuTelemetryData)
        gpu_data.metadata = GpuMetadata(
            gpu_index=0,
            model_name="Test GPU",
            gpu_uuid="GPU-123",
        )
        # Make get_metric_result raise exception
        gpu_data.get_metric_result = Mock(side_effect=Exception("Metric not available"))

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
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)

        # Should not raise exception despite metric retrieval failures
        await exporter.export(console)

        output = capsys.readouterr().out
        # Should still show GPU info even if metrics fail
        assert "Test GPU" in output or "GPU 0" in output

    @pytest.mark.asyncio
    async def test_export_all_endpoints_failed(
        self, mock_profile_results, mock_user_config, capsys
    ):
        """Test display when all endpoints failed."""
        from aiperf.common.models import TelemetryHierarchy, TelemetryResults

        service_config = ServiceConfig(verbose=True)

        telemetry_results = TelemetryResults(
            telemetry_data=TelemetryHierarchy(),
            start_ns=0,
            end_ns=0,
            endpoints_tested=[
                "http://node1:9400/metrics",
                "http://node2:9400/metrics",
                "http://node3:9400/metrics",
            ],
            endpoints_successful=[],
        )

        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)
        await exporter.export(console)

        output = capsys.readouterr().out
        assert "No GPU telemetry data collected" in output
        assert (
            "0/3 DCGM endpoints reachable" in output
            or "Unreachable endpoints" in output
        )
        assert "node1:9400" in output
        assert "node2:9400" in output
        assert "node3:9400" in output

    @pytest.mark.asyncio
    async def test_get_renderable_empty_gpu_data(
        self, mock_profile_results, mock_user_config
    ):
        """Test get_renderable with endpoint that has no GPU data."""
        from aiperf.common.models import TelemetryHierarchy, TelemetryResults

        service_config = ServiceConfig(verbose=True)

        # Endpoint exists but has no GPU data
        hierarchy = TelemetryHierarchy()
        hierarchy.dcgm_endpoints = {"http://localhost:9400/metrics": {}}

        telemetry_results = TelemetryResults(
            telemetry_data=hierarchy,
            start_ns=0,
            end_ns=0,
            endpoints_tested=["http://localhost:9400/metrics"],
            endpoints_successful=["http://localhost:9400/metrics"],
        )

        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        renderable = exporter.get_renderable(telemetry_results)

        # Should show no data message
        assert renderable is not None
