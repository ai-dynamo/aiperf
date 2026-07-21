# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for TimesliceMetricsJsonExporter."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.exceptions import DataExporterDisabled
from aiperf.common.models import MetricResult, TimesliceResult
from aiperf.common.models.export_models import TimesliceCollectionExportData
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.exporters.metrics_json_exporter import MetricsJsonExporter
from aiperf.exporters.timeslice_metrics_json_exporter import (
    TimesliceMetricsJsonExporter,
)
from aiperf.plugin.enums import EndpointType
from tests.unit.exporters.conftest import make_exporter_config


@pytest.fixture
def mock_cli_config():
    """Create a v1 CLIConfig that resolves to a valid v2 BenchmarkConfig."""
    return CLIConfig(
        model_names=["test-model"],
        endpoint_type=EndpointType.CHAT,
        custom_endpoint="/custom_endpoint",
    )


@pytest.fixture
def sample_timeslices():
    """Create sample timeslices.

    Shape: list[TimesliceResult] — chronological order, position == index.
    """
    return [
        TimesliceResult(
            start_ns=1_000_000_000,
            end_ns=2_000_000_000,
            metric_results=[
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
        ),
        TimesliceResult(
            start_ns=2_000_000_000,
            end_ns=3_000_000_000,
            metric_results=[
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
        ),
    ]


@pytest.fixture
def mock_results_with_timeslices(sample_timeslices):
    """Create mock results with timeslice data."""

    class MockResultsWithTimeslices:
        def __init__(self):
            self.timeslices = sample_timeslices
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
            self.timeslices = None
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
        self, mock_results_with_timeslices, mock_cli_config
    ):
        """Verify _file_path is set to {base_filename}_timeslices.json."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Note: profile_export_json_file is already set by default config

            config = make_exporter_config(
                results=mock_results_with_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            # Check that it has _timeslices suffix
            assert exporter._file_path.name.endswith("_timeslices.json")
            assert isinstance(exporter, MetricsJsonExporter)

    def test_timeslice_json_exporter_disabled_without_timeslice_data(
        self, mock_results_without_timeslices, mock_cli_config
    ):
        """Verify raises DataExporterDisabled when no timeslice data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=mock_results_without_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            with pytest.raises(DataExporterDisabled) as exc_info:
                TimesliceMetricsJsonExporter(config)

            assert "no timeslice metric results found" in str(exc_info.value)

    def test_timeslice_json_exporter_uses_base_filename(
        self, mock_results_with_timeslices, mock_cli_config
    ):
        """Verify uses base filename from configured JSON path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # The default profile_export_json_file should have a base name we can check

            config = make_exporter_config(
                results=mock_results_with_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            # Verify the filename pattern: base_timeslices.json
            assert "_timeslices.json" in exporter._file_path.name
            assert exporter._file_path.parent == Path(temp_dir)


class TestTimesliceMetricsJsonExporterGetExportInfo:
    """Tests for get_export_info() method."""

    def test_get_export_info_returns_correct_type(
        self, mock_results_with_timeslices, mock_cli_config
    ):
        """Verify export_type and file_path are correct."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=mock_results_with_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)
            info = exporter.get_export_info()

            assert info.export_type == "Timeslice JSON Export"
            assert info.file_path == exporter._file_path


class TestTimesliceMetricsJsonExporterGenerateContent:
    """Tests for _generate_content() method."""

    def test_generate_content_creates_collection_structure(
        self, mock_results_with_timeslices, mock_cli_config
    ):
        """Verify JSON has correct structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=mock_results_with_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            content = exporter._generate_content()

            data = json.loads(content)

            assert "timeslices" in data
            assert "input_config" in data
            assert isinstance(data["timeslices"], list)

    def test_generate_content_timeslices_have_index(self, mock_cli_config):
        """Verify the JSON timeslices array preserves chronological order.

        Slice ordering is now conveyed by position in the array — there's no
        explicit timeslice_index field (matches BaseTimeslice wire format).
        """
        timeslice_results = [
            TimesliceResult(
                start_ns=i * 1_000_000_000,
                end_ns=(i + 1) * 1_000_000_000,
                metric_results=[
                    MetricResult(
                        tag="metric",
                        header="Metric",
                        unit="ms",
                        avg=float(10 + i),
                    )
                ],
            )
            for i in range(3)
        ]

        class MockResults:
            def __init__(self):
                self.timeslices = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=MockResults(),
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            content = exporter._generate_content()

            data = json.loads(content)

            # No timeslice_index field; ordering comes from array position.
            assert len(data["timeslices"]) == 3
            for i, ts in enumerate(data["timeslices"]):
                assert "timeslice_index" not in ts
                assert ts["metric"]["avg"] == pytest.approx(10.0 + i)

    def test_generate_content_includes_metrics_dynamically(self, mock_cli_config):
        """Verify JSON has fields for all metrics at timeslice level."""
        timeslice_results = [
            TimesliceResult(
                start_ns=0,
                end_ns=1,
                metric_results=[
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
                ],
            )
        ]

        class MockResults:
            def __init__(self):
                self.timeslices = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=MockResults(),
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            content = exporter._generate_content()

            data = json.loads(content)

            timeslice_0 = data["timeslices"][0]
            assert "time_to_first_token" in timeslice_0
            assert "inter_token_latency" in timeslice_0

    def test_generate_content_uses_json_result_format(self, mock_cli_config):
        """Verify uses JsonMetricResult format."""
        timeslice_results = [
            TimesliceResult(
                start_ns=0,
                end_ns=1,
                metric_results=[
                    MetricResult(
                        tag="metric",
                        header="Metric",
                        unit="ms",
                        avg=45.0,
                        min=10.0,
                        max=90.0,
                    )
                ],
            )
        ]

        class MockResults:
            def __init__(self):
                self.timeslices = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=MockResults(),
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            content = exporter._generate_content()

            data = json.loads(content)

            metric_data = data["timeslices"][0]["metric"]
            assert "unit" in metric_data
            assert "avg" in metric_data
            assert "min" in metric_data
            assert "max" in metric_data

    def test_generate_content_different_metrics_per_timeslice(self, mock_cli_config):
        """Verify each timeslice can have different metrics."""
        timeslice_results = [
            TimesliceResult(
                start_ns=0,
                end_ns=1,
                metric_results=[
                    MetricResult(
                        tag="metric_a", header="Metric A", unit="ms", avg=10.0
                    ),
                    MetricResult(
                        tag="metric_b", header="Metric B", unit="ms", avg=20.0
                    ),
                ],
            ),
            TimesliceResult(
                start_ns=1,
                end_ns=2,
                metric_results=[
                    MetricResult(
                        tag="metric_b", header="Metric B", unit="ms", avg=25.0
                    ),
                    MetricResult(
                        tag="metric_c", header="Metric C", unit="ms", avg=30.0
                    ),
                ],
            ),
        ]

        class MockResults:
            def __init__(self):
                self.timeslices = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=MockResults(),
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

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
        self, mock_results_with_timeslices, mock_cli_config
    ):
        """Verify _prepare_metrics_for_json is called for each timeslice."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=mock_results_with_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            call_count = 0

            original_method = exporter._prepare_metrics_for_json

            def mock_prepare(metrics):
                nonlocal call_count
                call_count += 1
                return original_method(metrics)

            with patch.object(exporter, "_prepare_metrics_for_json", mock_prepare):
                exporter._generate_content()

            # Should be called once for each timeslice (2 in fixture)
            assert call_count == 2


class TestTimesliceMetricsJsonExporterIntegration:
    """Integration tests for TimesliceMetricsJsonExporter."""

    @pytest.mark.asyncio
    async def test_export_creates_valid_json_file(
        self, mock_results_with_timeslices, mock_cli_config
    ):
        """Verify export creates a valid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=mock_results_with_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

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
        self, mock_results_with_timeslices, mock_cli_config
    ):
        """Verify export can be deserialized to TimesliceCollectionExportData."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=mock_results_with_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            await exporter.export()

            # Read and deserialize
            with open(exporter._file_path) as f:
                content = f.read()

            # Should not raise exception
            TimesliceCollectionExportData.model_validate_json(content)

    @pytest.mark.asyncio
    async def test_export_with_many_timeslices(self, mock_cli_config):
        """Verify export with 50 timeslices."""
        timeslice_results = [
            TimesliceResult(
                start_ns=i * 1_000_000_000,
                end_ns=(i + 1) * 1_000_000_000,
                metric_results=[
                    MetricResult(tag="metric", header="Metric", unit="ms", avg=10.0 * i)
                ],
            )
            for i in range(50)
        ]

        class MockResults:
            def __init__(self):
                self.timeslices = timeslice_results
                self.records = []
                self.start_ns = None
                self.end_ns = None
                self.has_results = True
                self.was_cancelled = False
                self.error_summary = []

        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=MockResults(),
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)

            await exporter.export()

            with open(exporter._file_path) as f:
                data = json.load(f)

            assert len(data["timeslices"]) == 50

    def test_generate_content_includes_window_timestamps(
        self, mock_results_with_timeslices, mock_cli_config
    ):
        """Verify start_ns and end_ns appear in each timeslice JSON entry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_exporter_config(
                results=mock_results_with_timeslices,
                cli_config=mock_cli_config,
                artifact_directory=Path(temp_dir),
            )

            exporter = TimesliceMetricsJsonExporter(config)
            content = exporter._generate_content()

            data = json.loads(content)

            ts0 = data["timeslices"][0]
            ts1 = data["timeslices"][1]

            assert ts0["start_ns"] == 1_000_000_000
            assert ts0["end_ns"] == 2_000_000_000
            assert ts1["start_ns"] == 2_000_000_000
            assert ts1["end_ns"] == 3_000_000_000
            # is_complete=None should be omitted via exclude_none
            assert "is_complete" not in ts0
            assert "is_complete" not in ts1
