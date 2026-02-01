# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MetricsBaseExporter base class."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.plugin.enums import EndpointType


class ConcreteExporter(MetricsBaseExporter):
    """Concrete implementation for testing MetricsBaseExporter."""

    def __init__(self, exporter_config: ExporterConfig, **kwargs):
        super().__init__(exporter_config, **kwargs)
        self._file_path = self._output_directory / "test_export.txt"

    def _generate_content(self) -> str:
        return "test content"


@pytest.fixture
def mock_user_config():
    """Create a mock UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
        )
    )


@pytest.fixture
def mock_results():
    """Create mock results with basic metrics."""

    class MockResults:
        def __init__(self):
            self.records = [
                MetricResult(
                    tag="time_to_first_token",
                    header="Time to First Token",
                    unit="ms",
                    avg=45.2,
                )
            ]
            self.start_ns = None
            self.end_ns = None
            self.has_results = True
            self.was_cancelled = False
            self.error_summary = []

    return MockResults()


@pytest.fixture
def exporter_config(mock_results, mock_user_config):
    """Create ExporterConfig for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        mock_user_config.output.artifact_directory = Path(temp_dir)
        yield ExporterConfig(
            results=mock_results,
            user_config=mock_user_config,
            service_config=ServiceConfig(),
            telemetry_results=None,
        )


class TestMetricsBaseExporterInitialization:
    """Tests for MetricsBaseExporter initialization."""

    def test_base_exporter_initialization(self, mock_results, mock_user_config):
        """Verify all instance variables are set correctly from ExporterConfig."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = ConcreteExporter(config)

            assert exporter._results is mock_results
            assert exporter._telemetry_results is None
            assert exporter._user_config is mock_user_config
            assert exporter._output_directory == Path(temp_dir)


class TestMetricsBaseExporterPrepareMetrics:
    """Tests for _prepare_metrics() method.

    Metrics are already filtered and in display units from summarize().
    _prepare_metrics() just builds a dict keyed by tag.
    """

    def test_prepare_metrics_returns_dict_keyed_by_tag(self, exporter_config):
        """Verify metrics are returned as a dict keyed by tag."""
        exporter = ConcreteExporter(exporter_config)

        metric = MetricResult(
            tag="time_to_first_token",
            header="Time to First Token",
            unit="ms",
            avg=45.2,
        )

        result = exporter._prepare_metrics([metric])

        assert "time_to_first_token" in result
        assert result["time_to_first_token"] is metric

    def test_prepare_metrics_multiple_metrics(self, exporter_config):
        """Verify multiple metrics are all included."""
        exporter = ConcreteExporter(exporter_config)

        metrics = [
            MetricResult(tag="metric_a", header="A", unit="ms", avg=1.0),
            MetricResult(tag="metric_b", header="B", unit="ms", avg=2.0),
        ]

        result = exporter._prepare_metrics(metrics)

        assert len(result) == 2
        assert "metric_a" in result
        assert "metric_b" in result

    def test_prepare_metrics_handles_empty_input(self, exporter_config):
        """Verify it returns empty dict without errors for empty input."""
        exporter = ConcreteExporter(exporter_config)

        result = exporter._prepare_metrics([])

        assert result == {}


class TestMetricsBaseExporterExport:
    """Tests for export() method."""

    @pytest.mark.asyncio
    async def test_export_creates_output_directory(
        self, mock_results, mock_user_config
    ):
        """Verify directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nested" / "output"
            mock_user_config.output.artifact_directory = output_dir

            config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = ConcreteExporter(config)

            assert not output_dir.exists()

            await exporter.export()

            assert output_dir.exists()
            assert output_dir.is_dir()

    @pytest.mark.asyncio
    async def test_export_calls_generate_content(self, mock_results, mock_user_config):
        """Verify _generate_content() is called during export."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = ConcreteExporter(config)

            with patch.object(
                exporter, "_generate_content", return_value="mocked content"
            ) as mock_generate:
                await exporter.export()

                mock_generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_export_writes_content_to_file(self, mock_results, mock_user_config):
        """Verify file contains returned content."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = ConcreteExporter(config)

            test_content = "This is test content\nWith multiple lines"

            with patch.object(exporter, "_generate_content", return_value=test_content):
                await exporter.export()

                with open(exporter._file_path) as f:
                    actual_content = f.read()

                assert actual_content == test_content

    @pytest.mark.asyncio
    async def test_export_handles_write_errors(self, mock_results, mock_user_config):
        """Verify error is logged and exception is re-raised on write failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = ConcreteExporter(config)

            # Create a dict to track if error was called
            called = {"err": None}

            def _err(msg):
                called["err"] = msg

            with patch.object(exporter, "error", _err):
                import aiperf.exporters.metrics_base_exporter as mbe

                # Create a mock that raises when used as async context manager
                mock_file = MagicMock()
                mock_file.__aenter__ = AsyncMock(side_effect=OSError("disk full"))
                mock_file.__aexit__ = AsyncMock(return_value=False)

                with patch.object(mbe.aiofiles, "open", return_value=mock_file):
                    with pytest.raises(OSError, match="disk full"):
                        await exporter.export()

                    assert called["err"] is not None
                    assert "Failed to export" in called["err"]

    @pytest.mark.asyncio
    async def test_export_logs_debug_message(self, mock_results, mock_user_config):
        """Verify debug message is logged with file path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            config = ExporterConfig(
                results=mock_results,
                user_config=mock_user_config,
                service_config=ServiceConfig(verbose=True),
                telemetry_results=None,
            )

            exporter = ConcreteExporter(config)

            debug_messages = []

            def _debug(msg_func):
                if callable(msg_func):
                    debug_messages.append(msg_func())
                else:
                    debug_messages.append(msg_func)

            with patch.object(exporter, "debug", _debug):
                await exporter.export()

                # Check that a debug message containing the file path was logged
                assert any(str(exporter._file_path) in msg for msg in debug_messages), (
                    f"Expected file path in debug messages: {debug_messages}"
                )
