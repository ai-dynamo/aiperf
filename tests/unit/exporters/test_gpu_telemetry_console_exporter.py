# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPUTelemetryConsoleExporter."""

from datetime import datetime

import pytest
from rich.console import Console

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.models import (
    EndpointData,
    GpuSummary,
    JsonMetricResult,
    ProfileResults,
    TelemetryExportData,
    TelemetrySummary,
)
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.gpu_telemetry_console_exporter import (
    GPUTelemetryConsoleExporter,
)
from aiperf.plugin.enums import EndpointType


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
    """Create a mock user configuration with gpu_telemetry enabled."""
    return UserConfig(
        endpoint=mock_endpoint_config, gpu_telemetry=["http://localhost:9400/metrics"]
    )


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
        self,
        mock_profile_results,
        mock_endpoint_config,
        sample_telemetry_results,
        capsys,
    ):
        """Test that export does not print when gpu_telemetry is not enabled."""
        # Create user config without gpu_telemetry
        user_config = UserConfig(endpoint=mock_endpoint_config)
        service_config = ServiceConfig(verbose=False)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=user_config,
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
        assert "Power" in output and "Usage" in output

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
        renderable = exporter.get_renderable()

        assert renderable is not None

    def test_normalize_endpoint_display(self):
        """Test endpoint URL normalization for display."""
        from aiperf.exporters.utils import normalize_endpoint_display

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
        # Check for key metrics (may be wrapped across lines in table cells)
        assert "Power" in output
        assert "Usage" in output
        assert "Energy" in output
        assert "Utilization" in output
        assert "Memory" in output
        assert "Temperature" in output
        # Check for statistical columns
        assert "avg" in output or "min" in output or "max" in output

    @pytest.mark.asyncio
    async def test_export_with_failed_endpoint(
        self, mock_profile_results, mock_user_config, capsys
    ):
        """Test that failed endpoints show appropriate message."""
        service_config = ServiceConfig(verbose=True)

        # Create telemetry results with failed endpoint (no data)
        telemetry_results = TelemetryExportData(
            summary=TelemetrySummary(
                endpoints_configured=["http://failed-node:9400/metrics"],
                endpoints_successful=[],
                start_time=datetime.fromtimestamp(0),
                end_time=datetime.fromtimestamp(0),
            ),
            endpoints={},
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
        assert "Unreachable endpoints" in output
        assert "failed-node:9400" in output

    @pytest.mark.asyncio
    async def test_export_handles_missing_metrics(
        self, mock_profile_results, mock_user_config, capsys
    ):
        """Test that missing metrics are handled gracefully."""
        from datetime import datetime

        from aiperf.common.models.export_models import (
            EndpointData,
            GpuSummary,
            JsonMetricResult,
            TelemetryExportData,
            TelemetrySummary,
        )

        service_config = ServiceConfig(verbose=True)

        # Create telemetry results with GPU that only has some metrics
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
                            metrics={
                                # Only include one metric, others are missing
                                "gpu_power_usage": JsonMetricResult(
                                    unit="W", avg=100.0, min=90.0, max=110.0
                                ),
                            },
                        ),
                    }
                ),
            },
        )

        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        console = Console(width=150)

        # Should not raise exception despite missing metrics
        await exporter.export(console)

        output = capsys.readouterr().out
        # Should still show GPU info with available metrics
        assert "Test GPU" in output or "GPU 0" in output
        assert "Power Usage" in output

    @pytest.mark.asyncio
    async def test_export_all_endpoints_failed(
        self, mock_profile_results, mock_user_config, capsys
    ):
        """Test display when all endpoints failed."""
        service_config = ServiceConfig(verbose=True)

        telemetry_results = TelemetryExportData(
            summary=TelemetrySummary(
                endpoints_configured=[
                    "http://node1:9400/metrics",
                    "http://node2:9400/metrics",
                    "http://node3:9400/metrics",
                ],
                endpoints_successful=[],
                start_time=datetime.fromtimestamp(0),
                end_time=datetime.fromtimestamp(0),
            ),
            endpoints={},
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
        service_config = ServiceConfig(verbose=True)

        # Endpoint exists but has no GPU data
        telemetry_results = TelemetryExportData(
            summary=TelemetrySummary(
                endpoints_configured=["http://localhost:9400/metrics"],
                endpoints_successful=["http://localhost:9400/metrics"],
                start_time=datetime.fromtimestamp(0),
                end_time=datetime.fromtimestamp(0),
            ),
            endpoints={
                "localhost:9400": EndpointData(gpus={}),
            },
        )

        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=telemetry_results,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        renderable = exporter.get_renderable()

        # Should show no data message
        assert renderable is not None

    @pytest.mark.asyncio
    async def test_format_number_with_none(
        self, mock_profile_results, mock_user_config
    ):
        """Test _format_number with None value."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=None,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        result = exporter._format_number(None)
        assert result == "N/A"

    @pytest.mark.asyncio
    async def test_format_number_with_large_value(
        self, mock_profile_results, mock_user_config
    ):
        """Test _format_number with large values (scientific notation)."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=None,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        result = exporter._format_number(2_500_000.0)
        assert "2.50e+06" in result or "2.5e+06" in result

    @pytest.mark.asyncio
    async def test_format_number_with_small_value(
        self, mock_profile_results, mock_user_config
    ):
        """Test _format_number with normal values."""
        service_config = ServiceConfig(verbose=True)
        exporter_config = ExporterConfig(
            results=mock_profile_results,
            user_config=mock_user_config,
            service_config=service_config,
            telemetry_results=None,
        )

        exporter = GPUTelemetryConsoleExporter(exporter_config)
        result = exporter._format_number(123.456)
        assert result == "123.46"

    @pytest.mark.asyncio
    async def test_export_with_mixed_successful_failed_endpoints(
        self, mock_profile_results, mock_user_config, capsys
    ):
        """Test display with mix of successful and failed endpoints."""
        service_config = ServiceConfig(verbose=True)

        # Create one successful endpoint with GPU data
        telemetry_results = TelemetryExportData(
            summary=TelemetrySummary(
                endpoints_configured=[
                    "http://node1:9400/metrics",
                    "http://node2:9400/metrics",
                ],
                endpoints_successful=["http://node1:9400/metrics"],
                start_time=datetime.fromtimestamp(0),
                end_time=datetime.fromtimestamp(0),
            ),
            endpoints={
                "node1:9400": EndpointData(
                    gpus={
                        "gpu_0": GpuSummary(
                            gpu_index=0,
                            gpu_name="Test GPU",
                            gpu_uuid="GPU-123",
                            hostname="test-node",
                            metrics={
                                "gpu_power_usage": JsonMetricResult(
                                    unit="W", avg=100.0, min=90.0, max=110.0
                                ),
                            },
                        ),
                    }
                ),
            },
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
        # Should show 1/2 endpoints reachable
        assert "1/2 DCGM endpoints reachable" in output
        # Should show both endpoints with status
        assert "node1:9400" in output
        assert "node2:9400" in output
