# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TimesliceMetricsCsvExporter."""

import csv
import tempfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.metrics_base_exporter import MetricsBaseExporter
from aiperf.exporters.timeslice_metrics_csv_exporter import TimesliceMetricsCsvExporter
from aiperf.plugin.enums import EndpointType


@pytest.fixture
def mock_user_config():
    """Create mock UserConfig for testing."""
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.CHAT,
            custom_endpoint="custom_endpoint",
        )
    )


@pytest.fixture
def sample_timeslice_metric_results():
    """Create sample timeslice metric results."""
    return {
        0: [
            MetricResult(
                tag="time_to_first_token",
                header="Time to First Token",
                unit="ms",
                avg=45.2,
                min=12.1,
                max=89.3,
                p50=44.0,
                p90=78.0,
                p99=88.0,
                std=15.2,
            ),
            MetricResult(
                tag="inter_token_latency",
                header="Inter Token Latency",
                unit="ms",
                avg=5.1,
                min=2.3,
                max=12.4,
                p50=4.8,
                p90=9.2,
                p99=11.8,
                std=2.1,
            ),
        ],
        1: [
            MetricResult(
                tag="time_to_first_token",
                header="Time to First Token",
                unit="ms",
                avg=48.5,
                min=15.2,
                max=92.1,
                p50=47.3,
                p90=82.4,
                p99=90.5,
                std=16.1,
            ),
            MetricResult(
                tag="inter_token_latency",
                header="Inter Token Latency",
                unit="ms",
                avg=5.4,
                min=2.5,
                max=13.1,
                p50=5.1,
                p90=9.8,
                p99=12.3,
                std=2.3,
            ),
        ],
    }


@pytest.fixture
def mock_results_with_timeslices(sample_timeslice_metric_results):
    """Create mock results with timeslice data."""

    class MockResultsWithTimeslices:
        def __init__(self):
            self.timeslice_metric_results = sample_timeslice_metric_results
            self.records = []
            self.start_ns = None
            self.end_ns = None
            self.has_results = True
            self.was_cancelled = False
            self.error_summary = []

    return MockResultsWithTimeslices()


@pytest.fixture
def mock_results_without_timeslices():
    """Create mock results without timeslice data."""

    class MockResultsNoTimeslices:
        def __init__(self):
            self.timeslice_metric_results = None
            self.records = []
            self.start_ns = None
            self.end_ns = None
            self.has_results = False
            self.was_cancelled = False
            self.error_summary = []

    return MockResultsNoTimeslices()


class TestTimesliceMetricsCsvExporterInitialization:
    """Tests for TimesliceMetricsCsvExporter initialization."""

    def test_timeslice_csv_exporter_initialization(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify _file_path is set to {base_filename}_timeslices.csv."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            # Note: profile_export_csv_file is already set by default config

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            # Check that it has _timeslices suffix
            assert exporter._file_path.name.endswith("_timeslices.csv")
            assert isinstance(exporter, MetricsBaseExporter)

    def test_timeslice_csv_exporter_disabled_without_timeslice_data(
        self, mock_results_without_timeslices, mock_user_config
    ):
        """Verify raises DataExporterDisabled when no timeslice data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_without_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            with pytest.raises(DataExporterDisabled) as exc_info:
                TimesliceMetricsCsvExporter(config)

            assert "no timeslice metric results found" in str(exc_info.value)

    def test_timeslice_csv_exporter_uses_base_filename(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify uses base filename from configured CSV path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            # The default profile_export_csv_file should have a base name we can check

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            # Verify the filename pattern: base_timeslices.csv
            assert "_timeslices.csv" in exporter._file_path.name
            assert exporter._file_path.parent == Path(temp_dir)


class TestTimesliceMetricsCsvExporterGetExportInfo:
    """Tests for get_export_info() method."""

    def test_get_export_info_returns_correct_type(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify export_type and file_path are correct."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)
            info = exporter.get_export_info()

            assert info.export_type == "Timeslice CSV Export"
            assert info.file_path == exporter._file_path


class TestTimesliceMetricsCsvExporterGenerateContent:
    """Tests for _generate_content() method."""

    def test_generate_content_creates_tidy_format(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify CSV has correct header and tidy format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            # Mock conversion to return metrics as-is
            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            lines = content.strip().split("\n")
            reader = csv.reader(lines)
            rows = list(reader)

            # Check header
            assert rows[0] == ["Timeslice", "Metric", "Unit", "Stat", "Value"]

            # Check first data row has correct format
            assert len(rows[1]) == 5
            assert rows[1][0].isdigit()  # Timeslice index

    def test_generate_content_includes_all_timeslices(self, mock_user_config):
        """Verify all timeslice indices appear in output."""
        # Create 5 timeslices
        timeslice_results = {
            i: [
                MetricResult(
                    tag="test_metric",
                    header="Test Metric",
                    unit="ms",
                    avg=10.0 * i,
                )
            ]
            for i in range(5)
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            lines = content.strip().split("\n")
            reader = csv.reader(lines)
            rows = list(reader)

            # Get all timeslice indices from CSV
            timeslice_indices = set(int(row[0]) for row in rows[1:])  # Skip header

            assert timeslice_indices == {0, 1, 2, 3, 4}

    def test_generate_content_sorts_timeslices_by_index(self, mock_user_config):
        """Verify output has rows in sorted timeslice order."""
        # Create timeslices with indices [2, 0, 1]
        timeslice_results = {
            2: [MetricResult(tag="metric", header="Metric", unit="ms", avg=20.0)],
            0: [MetricResult(tag="metric", header="Metric", unit="ms", avg=0.0)],
            1: [MetricResult(tag="metric", header="Metric", unit="ms", avg=10.0)],
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            lines = content.strip().split("\n")
            reader = csv.reader(lines)
            rows = list(reader)

            # Check timeslice order
            timeslice_indices = [int(row[0]) for row in rows[1:]]
            assert timeslice_indices == [0, 1, 2]

    def test_generate_content_includes_all_stats(self, mock_user_config):
        """Verify each stat gets its own row."""
        timeslice_results = {
            0: [
                MetricResult(
                    tag="metric",
                    header="Metric",
                    unit="ms",
                    avg=45.0,
                    min=10.0,
                    max=90.0,
                    p50=44.0,
                    p90=78.0,
                    p95=85.0,
                    p99=88.0,
                    std=15.0,
                )
            ]
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            lines = content.strip().split("\n")
            reader = csv.reader(lines)
            rows = list(reader)

            # Get all stat names
            stat_names = [row[3] for row in rows[1:]]

            # Should have rows for all non-None stats
            expected_stats = ["avg", "min", "max", "p50", "p90", "p95", "p99", "std"]
            assert stat_names == expected_stats

    def test_generate_content_skips_none_stats(self, mock_user_config):
        """Verify only non-None stats appear in output."""
        timeslice_results = {
            0: [
                MetricResult(
                    tag="metric",
                    header="Metric",
                    unit="ms",
                    avg=45.0,
                    min=10.0,
                    max=90.0,
                    p50=None,  # None
                    p90=78.0,
                    p95=None,  # None
                    p99=88.0,
                    std=15.0,
                )
            ]
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            lines = content.strip().split("\n")
            reader = csv.reader(lines)
            rows = list(reader)

            # Get all stat names
            stat_names = [row[3] for row in rows[1:]]

            # Should not include p50 or p95
            assert "p50" not in stat_names
            assert "p95" not in stat_names
            assert "avg" in stat_names
            assert "p90" in stat_names

    def test_generate_content_uses_metric_header(self, mock_user_config):
        """Verify CSV uses header, not tag."""
        timeslice_results = {
            0: [
                MetricResult(
                    tag="ttft",
                    header="Time to First Token",
                    unit="ms",
                    avg=45.0,
                )
            ]
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            lines = content.strip().split("\n")
            reader = csv.reader(lines)
            rows = list(reader)

            assert rows[1][1] == "Time to First Token"

    def test_generate_content_includes_unit(self, mock_user_config):
        """Verify unit column contains unit value."""
        timeslice_results = {
            0: [MetricResult(tag="metric", header="Metric", unit="ms", avg=45.0)]
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            lines = content.strip().split("\n")
            reader = csv.reader(lines)
            rows = list(reader)

            assert rows[1][2] == "ms"

    def test_generate_content_empty_unit_for_unitless_metrics(self, mock_user_config):
        """Verify unit column is empty for unitless metrics."""
        timeslice_results = {
            0: [MetricResult(tag="metric", header="Metric", unit="", avg=45.0)]
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            lines = content.strip().split("\n")
            reader = csv.reader(lines)
            rows = list(reader)

            assert rows[1][2] == ""


class TestTimesliceMetricsCsvExporterFormatNumber:
    """Tests for _format_number() method."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, ""),
            (42, "42"),
            (0, "0"),
            (45.6789, "45.68"),
            (0.001234, "0.00"),
            (True, "True"),
            (False, "False"),
            (Decimal("123.456"), "123.46"),
        ],
    )
    def test_format_number_various_types(
        self, mock_results_with_timeslices, mock_user_config, value, expected
    ):
        """Test _format_number with various input types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            assert exporter._format_number(value) == expected


class TestTimesliceMetricsCsvExporterIntegration:
    """Integration tests for TimesliceMetricsCsvExporter."""

    @pytest.mark.asyncio
    async def test_export_creates_valid_csv_file(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify export creates a valid CSV file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                await exporter.export()

            # Verify file exists
            assert exporter._file_path.exists()

            # Verify it's valid CSV
            with open(exporter._file_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) > 1
            assert rows[0] == ["Timeslice", "Metric", "Unit", "Stat", "Value"]

    @pytest.mark.asyncio
    async def test_export_with_multiple_timeslices(self, mock_user_config):
        """Verify export with 10 timeslices creates correct row count."""
        # Create 10 timeslices with 2 metrics each (each with avg, min, max)
        timeslice_results = {
            i: [
                MetricResult(
                    tag="metric1",
                    header="Metric 1",
                    unit="ms",
                    avg=10.0,
                    min=5.0,
                    max=15.0,
                ),
                MetricResult(
                    tag="metric2",
                    header="Metric 2",
                    unit="ms",
                    avg=20.0,
                    min=10.0,
                    max=30.0,
                ),
            ]
            for i in range(10)
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                await exporter.export()

            with open(exporter._file_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Header + (10 timeslices * 2 metrics * 3 stats) = 1 + 60 = 61 rows
            assert len(rows) == 61

    @pytest.mark.asyncio
    async def test_export_empty_timeslice_data(self, mock_user_config):
        """Verify export with empty metrics creates header-only CSV."""
        timeslice_results = {
            0: [],  # Empty metric list
        }

        class MockResults:
            def __init__(self):
                self.timeslice_metric_results = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=MockResults(),
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsCsvExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {}

            with patch.object(
                mbe, "convert_all_metrics_to_display_units", mock_convert
            ):
                await exporter.export()

            with open(exporter._file_path) as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Should have only header
            assert len(rows) == 1
            assert rows[0] == ["Timeslice", "Metric", "Unit", "Stat", "Value"]
