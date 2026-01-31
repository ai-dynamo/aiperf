# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TimesliceMetricsJsonExporter."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.config import EndpointConfig, ServiceConfig, UserConfig
from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult
from aiperf.common.models.export_models import TimesliceCollectionExportData
from aiperf.exporters.exporter_config import ExporterConfig
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter
from aiperf.exporters.timeslice_metrics_json_exporter import (
    TimesliceMetricsJsonExporter,
)
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


class TestTimesliceMetricsJsonExporterInitialization:
    """Tests for TimesliceMetricsJsonExporter initialization."""

    def test_timeslice_json_exporter_initialization(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify _file_path is set to {base_filename}_timeslices.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            # Note: profile_export_json_file is already set by default config

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsJsonExporter(config)

            # Check that it has _timeslices suffix
            assert exporter._file_path.name.endswith("_timeslices.json")
            assert isinstance(exporter, MetricsJsonExporter)

    def test_timeslice_json_exporter_disabled_without_timeslice_data(
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
                TimesliceMetricsJsonExporter(config)

            assert "no timeslice metric results found" in str(exc_info.value)

    def test_timeslice_json_exporter_uses_base_filename(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify uses base filename from configured JSON path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)
            # The default profile_export_json_file should have a base name we can check

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsJsonExporter(config)

            # Verify the filename pattern: base_timeslices.json
            assert "_timeslices.json" in exporter._file_path.name
            assert exporter._file_path.parent == Path(temp_dir)


class TestTimesliceMetricsJsonExporterGetExportInfo:
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

            exporter = TimesliceMetricsJsonExporter(config)
            info = exporter.get_export_info()

            assert info.export_type == "Timeslice JSON Export"
            assert info.file_path == exporter._file_path


class TestTimesliceMetricsJsonExporterGenerateContent:
    """Tests for _generate_content() method."""

    def test_generate_content_creates_collection_structure(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify JSON has correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsJsonExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            data = json.loads(content)

            assert "timeslices" in data
            assert "input_config" in data
            assert isinstance(data["timeslices"], list)

    def test_generate_content_timeslices_have_index(self, mock_user_config):
        """Verify each timeslice object has timeslice_index field."""
        timeslice_results = {
            i: [MetricResult(tag="metric", header="Metric", unit="ms", avg=10.0)]
            for i in range(3)
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

            exporter = TimesliceMetricsJsonExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            data = json.loads(content)

            indices = [ts["timeslice_index"] for ts in data["timeslices"]]
            assert indices == [0, 1, 2]

    def test_generate_content_includes_metrics_dynamically(self, mock_user_config):
        """Verify JSON has fields for all metrics at timeslice level."""
        timeslice_results = {
            0: [
                MetricResult(
                    tag="time_to_first_token",
                    header="Time to First Token",
                    unit="ms",
                    avg=45.0,
                ),
                MetricResult(
                    tag="inter_token_latency",
                    header="Inter Token Latency",
                    unit="ms",
                    avg=5.0,
                ),
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

            exporter = TimesliceMetricsJsonExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            data = json.loads(content)

            timeslice_0 = data["timeslices"][0]
            assert "time_to_first_token" in timeslice_0
            assert "inter_token_latency" in timeslice_0

    def test_generate_content_uses_json_result_format(self, mock_user_config):
        """Verify uses JsonMetricResult format."""
        timeslice_results = {
            0: [
                MetricResult(
                    tag="metric",
                    header="Metric",
                    unit="ms",
                    avg=45.0,
                    min=10.0,
                    max=90.0,
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

            exporter = TimesliceMetricsJsonExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            data = json.loads(content)

            metric_data = data["timeslices"][0]["metric"]
            assert "unit" in metric_data
            assert "avg" in metric_data
            assert "min" in metric_data
            assert "max" in metric_data

    def test_generate_content_different_metrics_per_timeslice(self, mock_user_config):
        """Verify each timeslice can have different metrics."""
        timeslice_results = {
            0: [
                MetricResult(tag="metric_a", header="Metric A", unit="ms", avg=10.0),
                MetricResult(tag="metric_b", header="Metric B", unit="ms", avg=20.0),
            ],
            1: [
                MetricResult(tag="metric_b", header="Metric B", unit="ms", avg=25.0),
                MetricResult(tag="metric_c", header="Metric C", unit="ms", avg=30.0),
            ],
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

            exporter = TimesliceMetricsJsonExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                content = exporter._generate_content()

            data = json.loads(content)

            ts0 = data["timeslices"][0]
            ts1 = data["timeslices"][1]

            assert "metric_a" in ts0
            assert "metric_b" in ts0
            assert "metric_c" not in ts0

            assert "metric_a" not in ts1
            assert "metric_b" in ts1
            assert "metric_c" in ts1

    def test_generate_content_reuses_prepare_metrics_for_json(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify _prepare_metrics_for_json is called for each timeslice."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsJsonExporter(config)

            call_count = 0

            original_method = exporter._prepare_metrics_for_json

            def mock_prepare(metrics):
                nonlocal call_count
                call_count += 1
                return original_method(metrics)

            with patch.object(exporter, "_prepare_metrics_for_json", mock_prepare):
                import aiperf.exporters.metrics_base_exporter as mbe

                def mock_convert(metrics, reg):
                    return {m.tag: m for m in metrics}

                with (
                    patch.object(
                        mbe, "convert_all_metrics_to_display_units", mock_convert
                    ),
                    patch.object(exporter, "_should_export", return_value=True),
                ):
                    exporter._generate_content()

            # Should be called once for each timeslice (2 in fixture)
            assert call_count == 2


class TestTimesliceMetricsJsonExporterIntegration:
    """Integration tests for TimesliceMetricsJsonExporter."""

    @pytest.mark.asyncio
    async def test_export_creates_valid_json_file(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify export creates a valid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsJsonExporter(config)

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

            # Verify it's valid JSON
            with open(exporter._file_path) as f:
                data = json.load(f)

            assert "timeslices" in data
            assert "input_config" in data

    @pytest.mark.asyncio
    async def test_export_can_deserialize_to_pydantic_model(
        self, mock_results_with_timeslices, mock_user_config
    ):
        """Verify export can be deserialized to TimesliceCollectionExportData."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_user_config.output.artifact_directory = Path(temp_dir)

            config = ExporterConfig(
                results=mock_results_with_timeslices,
                user_config=mock_user_config,
                service_config=ServiceConfig(),
                telemetry_results=None,
            )

            exporter = TimesliceMetricsJsonExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                await exporter.export()

            # Read and deserialize
            with open(exporter._file_path) as f:
                content = f.read()

            # Should not raise exception
            TimesliceCollectionExportData.model_validate_json(content)

    @pytest.mark.asyncio
    async def test_export_with_many_timeslices(self, mock_user_config):
        """Verify export with 50 timeslices."""
        timeslice_results = {
            i: [MetricResult(tag="metric", header="Metric", unit="ms", avg=10.0 * i)]
            for i in range(50)
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

            exporter = TimesliceMetricsJsonExporter(config)

            import aiperf.exporters.metrics_base_exporter as mbe

            def mock_convert(metrics, reg):
                return {m.tag: m for m in metrics}

            with (
                patch.object(mbe, "convert_all_metrics_to_display_units", mock_convert),
                patch.object(exporter, "_should_export", return_value=True),
            ):
                await exporter.export()

            with open(exporter._file_path) as f:
                data = json.load(f)

            assert len(data["timeslices"]) == 50
