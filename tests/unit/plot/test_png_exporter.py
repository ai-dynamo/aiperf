# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the PNG Exporter classes.

This module tests the PNG export functionality, ensuring that plots are
correctly generated and saved as PNG files with proper metadata.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from aiperf.common.enums import PrometheusMetricType
from aiperf.common.models.record_models import MetricResult
from aiperf.plot.core.data_loader import DataLoader, RunData, RunMetadata
from aiperf.plot.core.data_preparation import (
    prepare_request_timeseries,
    validate_request_uniformity,
)
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import (
    DataSource,
    MetricSpec,
    PlotSpec,
    PlotType,
    Style,
    TimeSlicePlotSpec,
)
from aiperf.plot.exporters.png import MultiRunPNGExporter, SingleRunPNGExporter
from aiperf.plot.handlers.single_run_handlers import DualAxisHandler

# Check if Chrome is available for Kaleido PNG export
CHROME_AVAILABLE = (
    shutil.which("google-chrome") is not None or shutil.which("chromium") is not None
)
requires_chrome = pytest.mark.skipif(
    not CHROME_AVAILABLE, reason="Chrome/Chromium is required for PNG export tests"
)

# Module-level constants for plot specs (used in tests)
SINGLE_RUN_PLOT_SPECS = [
    PlotSpec(
        name="ttft_over_time",
        plot_type=PlotType.SCATTER,
        metrics=[
            MetricSpec(name="request_number", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(
                name="time_to_first_token", source=DataSource.REQUESTS, axis="y"
            ),
        ],
        title="TTFT Per Request Over Time",
        filename="ttft_over_time.png",
    ),
    PlotSpec(
        name="itl_over_time",
        plot_type=PlotType.SCATTER,
        metrics=[
            MetricSpec(name="request_number", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(
                name="inter_token_latency", source=DataSource.REQUESTS, axis="y"
            ),
        ],
        title="Inter-Token Latency Per Request Over Time",
        filename="itl_over_time.png",
    ),
    PlotSpec(
        name="latency_over_time",
        plot_type=PlotType.SCATTER_WITH_PERCENTILES,
        metrics=[
            MetricSpec(name="timestamp", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(name="request_latency", source=DataSource.REQUESTS, axis="y"),
        ],
        title="Request Latency Over Time with Percentiles",
        filename="latency_over_time.png",
    ),
    PlotSpec(
        name="dispersed_throughput_over_time",
        plot_type=PlotType.AREA,
        metrics=[
            MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(
                name="throughput_tokens_per_sec", source=DataSource.REQUESTS, axis="y"
            ),
        ],
        title="Dispersed Output Token Throughput Over Time",
        filename="dispersed_throughput_over_time.png",
    ),
]

TIMESLICE_PLOT_SPECS = [
    TimeSlicePlotSpec(
        name="timeslices_ttft",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
            MetricSpec(
                name="Time to First Token",
                source=DataSource.TIMESLICES,
                axis="y",
                stat="avg",
            ),
        ],
        title="Average Time to First Token Across Time Slices",
        filename="timeslices_ttft.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_itl",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
            MetricSpec(
                name="Inter Token Latency",
                source=DataSource.TIMESLICES,
                axis="y",
                stat="avg",
            ),
        ],
        title="Average Inter Token Latency Across Time Slices",
        filename="timeslices_itl.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_throughput",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
            MetricSpec(
                name="Request Throughput",
                source=DataSource.TIMESLICES,
                axis="y",
                stat="avg",
            ),
        ],
        title="Average Request Throughput Across Time Slices",
        filename="timeslices_throughput.png",
        use_slice_duration=True,
    ),
    TimeSlicePlotSpec(
        name="timeslices_latency",
        plot_type=PlotType.HISTOGRAM,
        metrics=[
            MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
            MetricSpec(
                name="Request Latency",
                source=DataSource.TIMESLICES,
                axis="y",
                stat="avg",
            ),
        ],
        title="Average Request Latency Across Time Slices",
        filename="timeslices_latency.png",
        use_slice_duration=True,
    ),
]

GPU_PLOT_SPECS = [
    PlotSpec(
        name="gpu_utilization_and_throughput_over_time",
        plot_type=PlotType.DUAL_AXIS,
        metrics=[
            MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
            MetricSpec(
                name="throughput_tokens_per_sec", source=DataSource.REQUESTS, axis="y"
            ),
            MetricSpec(
                name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
            ),
        ],
        title="Output Token Throughput with GPU Utilization",
        filename="gpu_utilization_and_throughput_over_time.png",
        primary_style=Style(mode="lines", line_shape="hv", fill=None),
        secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
        supplementary_col="active_requests",
    ),
]

MULTI_RUN_PLOT_SPECS = [
    PlotSpec(
        name="pareto_curve_throughput_per_gpu_vs_latency",
        plot_type=PlotType.PARETO,
        metrics=[
            MetricSpec(
                name="request_latency",
                source=DataSource.AGGREGATED,
                axis="x",
                stat="avg",
            ),
            MetricSpec(
                name="output_token_throughput_per_gpu",
                source=DataSource.AGGREGATED,
                axis="y",
                stat="avg",
            ),
        ],
        title="Pareto Curve: Token Throughput per GPU vs Latency",
        filename="pareto_curve_throughput_per_gpu_vs_latency.png",
        label_by=None,
        group_by=None,
    ),
    PlotSpec(
        name="ttft_vs_throughput",
        plot_type=PlotType.SCATTER_LINE,
        metrics=[
            MetricSpec(
                name="time_to_first_token",
                source=DataSource.AGGREGATED,
                axis="x",
                stat="p50",
            ),
            MetricSpec(
                name="request_throughput",
                source=DataSource.AGGREGATED,
                axis="y",
                stat="avg",
            ),
        ],
        title="TTFT vs Throughput",
        filename="ttft_vs_throughput.png",
        label_by=None,
        group_by=None,
    ),
    PlotSpec(
        name="pareto_curve_throughput_per_gpu_vs_interactivity",
        plot_type=PlotType.SCATTER_LINE,
        metrics=[
            MetricSpec(
                name="output_token_throughput_per_gpu",
                source=DataSource.AGGREGATED,
                axis="x",
                stat="avg",
            ),
            MetricSpec(
                name="output_token_throughput_per_user",
                source=DataSource.AGGREGATED,
                axis="y",
                stat="avg",
            ),
        ],
        title="Pareto Curve: Token Throughput per GPU vs Interactivity",
        filename="pareto_curve_throughput_per_gpu_vs_interactivity.png",
        label_by=["concurrency"],
        group_by=None,
    ),
]


@pytest.fixture
def multi_run_exporter(tmp_path):
    """Create a MultiRunPNGExporter instance for testing."""
    output_dir = tmp_path / "plots"
    return MultiRunPNGExporter(output_dir)


@pytest.fixture
def single_run_exporter(tmp_path):
    """Create a SingleRunPNGExporter instance for testing."""
    output_dir = tmp_path / "plots"
    return SingleRunPNGExporter(output_dir)


@pytest.fixture
def sample_plot_specs():
    """Create plot specs for single-run testing (matches original hardcoded specs)."""
    # Single-run plot specifications
    single_run_specs = [
        PlotSpec(
            name="ttft_over_time",
            plot_type=PlotType.SCATTER,
            metrics=[
                MetricSpec(name="request_number", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="time_to_first_token", source=DataSource.REQUESTS, axis="y"
                ),
            ],
            title="TTFT Per Request Over Time",
            filename="ttft_over_time.png",
        ),
        PlotSpec(
            name="itl_over_time",
            plot_type=PlotType.SCATTER,
            metrics=[
                MetricSpec(name="request_number", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="inter_token_latency", source=DataSource.REQUESTS, axis="y"
                ),
            ],
            title="Inter-Token Latency Per Request Over Time",
            filename="itl_over_time.png",
        ),
        PlotSpec(
            name="latency_over_time",
            plot_type=PlotType.SCATTER_WITH_PERCENTILES,
            metrics=[
                MetricSpec(name="timestamp", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="request_latency", source=DataSource.REQUESTS, axis="y"
                ),
            ],
            title="Request Latency Over Time with Percentiles",
            filename="latency_over_time.png",
        ),
        PlotSpec(
            name="dispersed_throughput_over_time",
            plot_type=PlotType.AREA,
            metrics=[
                MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="throughput_tokens_per_sec",
                    source=DataSource.REQUESTS,
                    axis="y",
                ),
            ],
            title="Dispersed Output Token Throughput Over Time",
            filename="dispersed_throughput_over_time.png",
        ),
    ]

    # Timeslice plot specifications
    timeslice_specs = [
        TimeSlicePlotSpec(
            name="timeslices_ttft",
            plot_type=PlotType.HISTOGRAM,
            metrics=[
                MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
                MetricSpec(
                    name="Time to First Token",
                    source=DataSource.TIMESLICES,
                    axis="y",
                    stat="avg",
                ),
            ],
            title="Average Time to First Token Across Time Slices",
            filename="timeslices_ttft.png",
            use_slice_duration=True,
        ),
        TimeSlicePlotSpec(
            name="timeslices_itl",
            plot_type=PlotType.HISTOGRAM,
            metrics=[
                MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
                MetricSpec(
                    name="Inter Token Latency",
                    source=DataSource.TIMESLICES,
                    axis="y",
                    stat="avg",
                ),
            ],
            title="Average Inter Token Latency Across Time Slices",
            filename="timeslices_itl.png",
            use_slice_duration=True,
        ),
        TimeSlicePlotSpec(
            name="timeslices_throughput",
            plot_type=PlotType.HISTOGRAM,
            metrics=[
                MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
                MetricSpec(
                    name="Request Throughput",
                    source=DataSource.TIMESLICES,
                    axis="y",
                    stat="avg",
                ),
            ],
            title="Average Request Throughput Across Time Slices",
            filename="timeslices_throughput.png",
            use_slice_duration=True,
        ),
        TimeSlicePlotSpec(
            name="timeslices_latency",
            plot_type=PlotType.HISTOGRAM,
            metrics=[
                MetricSpec(name="Timeslice", source=DataSource.TIMESLICES, axis="x"),
                MetricSpec(
                    name="Request Latency",
                    source=DataSource.TIMESLICES,
                    axis="y",
                    stat="avg",
                ),
            ],
            title="Average Request Latency Across Time Slices",
            filename="timeslices_latency.png",
            use_slice_duration=True,
        ),
    ]

    # GPU plot specifications
    gpu_specs = [
        PlotSpec(
            name="gpu_utilization_and_throughput_over_time",
            plot_type=PlotType.DUAL_AXIS,
            metrics=[
                MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="throughput_tokens_per_sec",
                    source=DataSource.REQUESTS,
                    axis="y",
                ),
                MetricSpec(
                    name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
                ),
            ],
            title="Output Token Throughput with GPU Utilization",
            filename="gpu_utilization_and_throughput_over_time.png",
            primary_style=Style(mode="lines", line_shape="hv", fill=None),
            secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
            supplementary_col="active_requests",
        ),
    ]

    return single_run_specs + timeslice_specs + gpu_specs


@pytest.fixture
def sample_multi_run_plot_specs():
    """Create plot specs for multi-run testing (matches original hardcoded specs)."""
    return [
        PlotSpec(
            name="pareto_curve_throughput_per_gpu_vs_latency",
            plot_type=PlotType.PARETO,
            metrics=[
                MetricSpec(
                    name="request_latency",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="avg",
                ),
                MetricSpec(
                    name="output_token_throughput_per_gpu",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            title="Pareto Curve: Token Throughput per GPU vs Latency",
            filename="pareto_curve_throughput_per_gpu_vs_latency.png",
            label_by=None,
            group_by=None,
        ),
        PlotSpec(
            name="ttft_vs_throughput",
            plot_type=PlotType.SCATTER_LINE,
            metrics=[
                MetricSpec(
                    name="time_to_first_token",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="p50",
                ),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            title="TTFT vs Throughput",
            filename="ttft_vs_throughput.png",
            label_by=None,
            group_by=None,
        ),
        PlotSpec(
            name="pareto_curve_throughput_per_gpu_vs_interactivity",
            plot_type=PlotType.SCATTER_LINE,
            metrics=[
                MetricSpec(
                    name="output_token_throughput_per_gpu",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="avg",
                ),
                MetricSpec(
                    name="output_token_throughput_per_user",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            title="Pareto Curve: Token Throughput per GPU vs Interactivity",
            filename="pareto_curve_throughput_per_gpu_vs_interactivity.png",
            label_by=["concurrency"],
            group_by=None,
        ),
    ]


@pytest.fixture
def sample_multi_run_data(tmp_path):
    """Create sample multi-run data for testing."""
    return [
        RunData(
            metadata=RunMetadata(
                run_name="run_001",
                run_path=tmp_path / "run_001",
                model="Qwen/Qwen3-0.6B",
                concurrency=1,
            ),
            requests=None,
            aggregated={
                "request_latency": {"p50": 100.0, "avg": 105.0, "unit": "ms"},
                "request_throughput": {"avg": 10.0, "unit": "req/s"},
                "time_to_first_token": {"p50": 45.0, "unit": "ms"},
                "inter_token_latency": {"p50": 18.0, "unit": "ms"},
                "output_token_throughput_per_user": {
                    "avg": 100.0,
                    "unit": "tokens/s/user",
                },
                "output_token_throughput_per_gpu": {
                    "avg": 50.0,
                    "unit": "tokens/s/gpu",
                },
            },
            timeslices=None,
            slice_duration=None,
        ),
        RunData(
            metadata=RunMetadata(
                run_name="run_002",
                run_path=tmp_path / "run_002",
                model="Qwen/Qwen3-0.6B",
                concurrency=4,
            ),
            requests=None,
            aggregated={
                "request_latency": {"p50": 150.0, "avg": 155.0, "unit": "ms"},
                "request_throughput": {"avg": 25.0, "unit": "req/s"},
                "time_to_first_token": {"p50": 55.0, "unit": "ms"},
                "inter_token_latency": {"p50": 20.0, "unit": "ms"},
                "output_token_throughput_per_user": {
                    "avg": 90.0,
                    "unit": "tokens/s/user",
                },
                "output_token_throughput_per_gpu": {
                    "avg": 120.0,
                    "unit": "tokens/s/gpu",
                },
            },
            timeslices=None,
            slice_duration=None,
        ),
    ]


@pytest.fixture
def sample_single_run_data(tmp_path):
    """Create sample single-run data for testing."""
    # Create per-request DataFrame with all required fields
    base_time = 1000000000000  # Base time in ns
    per_request_data = pd.DataFrame(
        {
            "request_start_ns": pd.to_datetime(
                [base_time + i * 500000000 for i in range(10)], unit="ns", utc=True
            ),
            "request_end_ns": pd.to_datetime(
                [base_time + i * 500000000 + 400000000 for i in range(10)],
                unit="ns",
                utc=True,
            ),
            "time_to_first_token": [45.0 + i * 2 for i in range(10)],
            "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
            "request_latency": [900.0 + i * 10 for i in range(10)],
            "output_sequence_length": [100 + i * 10 for i in range(10)],
            "input_sequence_length": [50] * 10,
        }
    )

    return RunData(
        metadata=RunMetadata(
            run_name="run_003",
            run_path=tmp_path / "run_003",
            model="Qwen/Qwen3-0.6B",
            concurrency=8,
        ),
        requests=per_request_data,
        aggregated={
            "request_latency": {"p50": 200.0, "avg": 205.0, "unit": "ms"},
            "request_throughput": {"avg": 35.0, "unit": "req/s"},
        },
        timeslices=None,
    )


@pytest.fixture
def sample_timeslice_data():
    """Create sample timeslice data for testing."""
    return pd.DataFrame(
        {
            "Timeslice": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "Metric": [
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
                "Time to First Token",
            ],
            "Unit": ["ms"] * 9,
            "Stat": ["avg", "min", "max", "avg", "min", "max", "avg", "min", "max"],
            "Value": [45.0, 30.0, 60.0, 47.0, 32.0, 65.0, 46.0, 31.0, 62.0],
        }
    )


@pytest.fixture
def sample_available_metrics():
    """Create sample available metrics dictionary."""
    return {
        "display_names": {
            "request_latency": "Request Latency",
            "request_throughput": "Request Throughput",
            "time_to_first_token": "Time to First Token",
            "inter_token_latency": "Inter Token Latency",
            "output_token_throughput_per_user": "Output Token Throughput per User",
        },
        "units": {
            "request_latency": "ms",
            "request_throughput": "req/s",
            "time_to_first_token": "ms",
            "inter_token_latency": "ms",
            "output_token_throughput_per_user": "tokens/s/user",
        },
    }


class TestMultiRunPNGExporter:
    """Tests for MultiRunPNGExporter class."""

    def test_initialization(self, multi_run_exporter):
        """Test that MultiRunPNGExporter can be instantiated."""
        assert isinstance(multi_run_exporter, MultiRunPNGExporter)
        assert isinstance(multi_run_exporter.output_dir, Path)

    @requires_chrome
    def test_export_multi_run_creates_files(
        self,
        multi_run_exporter,
        sample_multi_run_data,
        sample_available_metrics,
        sample_multi_run_plot_specs,
    ):
        """Test that multi-run export creates PNG files from provided specs."""
        generated_files = multi_run_exporter.export(
            sample_multi_run_data, sample_available_metrics, MULTI_RUN_PLOT_SPECS
        )

        # Should generate plots based on the specs provided (config-driven)
        assert len(generated_files) > 0

        # Check that files exist and are PNGs
        for file_path in generated_files:
            assert file_path.exists()
            assert file_path.suffix == ".png"

    @requires_chrome
    def test_export_multi_run_creates_expected_plots(
        self,
        multi_run_exporter,
        sample_multi_run_data,
        sample_available_metrics,
    ):
        """Test that plots matching the provided specs are created."""
        generated_files = multi_run_exporter.export(
            sample_multi_run_data, sample_available_metrics, MULTI_RUN_PLOT_SPECS
        )

        # Get filenames
        filenames = {f.name for f in generated_files}

        # Check that filenames match the specs provided (config-driven)
        spec_filenames = {spec.filename for spec in MULTI_RUN_PLOT_SPECS}
        assert filenames.issubset(spec_filenames), (
            f"Generated unexpected files: {filenames - spec_filenames}"
        )

    @requires_chrome
    def test_export_multi_run_creates_summary(
        self,
        multi_run_exporter,
        sample_multi_run_data,
        sample_available_metrics,
    ):
        """Test that summary file is created."""
        generated_files = multi_run_exporter.export(
            sample_multi_run_data, sample_available_metrics, MULTI_RUN_PLOT_SPECS
        )

        summary_path = multi_run_exporter.output_dir / "summary.txt"
        assert summary_path.exists()

        # Check summary content
        content = summary_path.read_text(encoding="utf-8")
        assert "AIPerf Plot Export Summary" in content
        assert f"Generated {len(generated_files)} plot" in content

    def test_runs_to_dataframe_with_metric_result_objects(
        self, multi_run_exporter, tmp_path
    ):
        """Test conversion with MetricResult objects instead of dicts."""
        runs = [
            RunData(
                metadata=RunMetadata(
                    run_name="run_001",
                    run_path=tmp_path / "run_001",
                    model="TestModel",
                    concurrency=4,
                ),
                requests=None,
                aggregated={
                    "request_latency": MetricResult(
                        tag="request_latency",
                        header="Request Latency",
                        unit="ms",
                        p50=100.0,
                        avg=105.0,
                    ),
                    "request_throughput": MetricResult(
                        tag="request_throughput",
                        header="Request Throughput",
                        unit="req/s",
                        avg=25.0,
                    ),
                },
                timeslices=None,
                slice_duration=None,
            )
        ]

        df = multi_run_exporter._runs_to_dataframe(
            runs, {"display_names": {}, "units": {}}
        )

        # Verify DataFrame structure
        assert len(df) == 1
        assert "model" in df.columns
        assert "concurrency" in df.columns
        assert "request_latency" in df.columns
        assert "request_throughput" in df.columns

        # Verify values were extracted correctly from MetricResult objects
        assert df["request_latency"].iloc[0] == 100.0  # p50 preferred
        assert df["request_throughput"].iloc[0] == 25.0  # avg fallback

    def test_runs_to_dataframe_with_mixed_types(self, multi_run_exporter, tmp_path):
        """Test conversion with both MetricResult objects and dicts."""
        runs = [
            RunData(
                metadata=RunMetadata(
                    run_name="run_001",
                    run_path=tmp_path / "run_001",
                    model="TestModel",
                    concurrency=4,
                ),
                requests=None,
                aggregated={
                    "request_latency": MetricResult(
                        tag="request_latency",
                        header="Request Latency",
                        unit="ms",
                        p50=100.0,
                    ),
                    "request_throughput": {
                        "avg": 25.0,
                        "unit": "req/s",
                    },  # Dict fallback
                },
                timeslices=None,
                slice_duration=None,
            )
        ]

        df = multi_run_exporter._runs_to_dataframe(
            runs, {"display_names": {}, "units": {}}
        )

        # Both MetricResult and dict should work
        assert df["request_latency"].iloc[0] == 100.0
        assert df["request_throughput"].iloc[0] == 25.0

    def test_runs_to_dataframe(
        self, multi_run_exporter, sample_multi_run_data, sample_available_metrics
    ):
        """Test conversion of runs to DataFrame."""
        df = multi_run_exporter._runs_to_dataframe(
            sample_multi_run_data, sample_available_metrics
        )

        # Check DataFrame structure
        assert len(df) == 2  # Two runs
        assert "model" in df.columns
        assert "concurrency" in df.columns
        assert "request_latency" in df.columns
        assert "request_throughput" in df.columns

        # Check values
        assert df["concurrency"].tolist() == [1, 4]
        assert df["request_latency"].tolist() == [100.0, 150.0]  # p50 values


class TestSingleRunPNGExporter:
    """Tests for SingleRunPNGExporter class."""

    def test_initialization(self, single_run_exporter):
        """Test that SingleRunPNGExporter can be instantiated."""
        assert isinstance(single_run_exporter, SingleRunPNGExporter)
        assert isinstance(single_run_exporter.output_dir, Path)

    @requires_chrome
    def test_export_single_run_creates_files(
        self,
        single_run_exporter,
        sample_single_run_data,
        sample_available_metrics,
        sample_plot_specs,
    ):
        """Test that single-run export creates PNG files."""
        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        generated_files = single_run_exporter.export(
            sample_single_run_data, sample_available_metrics, plot_specs
        )

        # Should generate plots based on available data and specs (config-driven)
        assert len(generated_files) > 0

        # Check that files exist and are PNGs
        for file_path in generated_files:
            assert file_path.exists()
            assert file_path.suffix == ".png"

    @requires_chrome
    def test_export_single_run_creates_expected_plots(
        self,
        single_run_exporter,
        sample_single_run_data,
        sample_available_metrics,
        sample_plot_specs,
    ):
        """Test that expected plot files are created for single run."""
        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        generated_files = single_run_exporter.export(
            sample_single_run_data, sample_available_metrics, plot_specs
        )

        # Get filenames
        filenames = {f.name for f in generated_files}

        # Check that filenames match specs that can be generated with available data
        spec_filenames = {spec.filename for spec in sample_plot_specs}
        assert filenames.issubset(spec_filenames), (
            f"Generated unexpected files: {filenames - spec_filenames}"
        )

    def test_export_single_run_with_no_per_request_data(
        self,
        single_run_exporter,
        sample_available_metrics,
        sample_plot_specs,
        tmp_path,
    ):
        """Test handling of single run with no per-request data."""
        run_data = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test",
                concurrency=1,
            ),
            requests=None,  # No per-request data
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        generated_files = single_run_exporter.export(
            run_data, sample_available_metrics, plot_specs
        )

        # Should return empty list when no data available
        assert len(generated_files) == 0

    def test_per_request_to_dataframe(self, sample_single_run_data):
        """Test conversion of per-request data to DataFrame."""
        df = prepare_request_timeseries(sample_single_run_data)

        # Check DataFrame structure
        assert len(df) == 10  # 10 requests
        assert "request_number" in df.columns
        assert "timestamp" in df.columns
        assert "time_to_first_token" in df.columns

        # Check timestamp normalization (should start from 0)
        assert df["timestamp"].min() == 0.0

    def test_get_metric_label(self, single_run_exporter, sample_available_metrics):
        """Test metric label formatting."""
        # With stat
        label = single_run_exporter._get_metric_label(
            "request_latency", "p50", sample_available_metrics
        )
        assert "Request Latency" in label
        assert "P50" in label
        assert "ms" in label

        # Without stat
        label = single_run_exporter._get_metric_label(
            "request_latency", None, sample_available_metrics
        )
        assert "Request Latency" in label
        assert "P50" not in label
        assert "ms" in label

    def test_get_metric_label_unknown_metric(self, single_run_exporter):
        """Test metric label for unknown metric."""
        label = single_run_exporter._get_metric_label(
            "unknown_metric", "p50", {"display_names": {}, "units": {}}
        )

        # Should use formatted metric tag as fallback
        assert "Unknown Metric" in label

    @requires_chrome
    def test_export_single_run_with_timeslice_data(
        self,
        single_run_exporter,
        sample_single_run_data,
        sample_timeslice_data,
        sample_available_metrics,
        sample_plot_specs,
        tmp_path,
    ):
        """Test that timeslice plots are generated when timeslice data is available."""
        # Add timeslice data to the run
        run_with_timeslices = RunData(
            metadata=sample_single_run_data.metadata,
            requests=sample_single_run_data.requests,
            aggregated=sample_single_run_data.aggregated,
            timeslices=sample_timeslice_data,
            slice_duration=10.0,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        generated_files = single_run_exporter.export(
            run_with_timeslices, sample_available_metrics, plot_specs
        )

        # Should generate more plots with timeslice data available
        assert len(generated_files) > 0

        # Check that at least one timeslice plot is in the generated files
        filenames = {f.name for f in generated_files}
        timeslice_plots = [f for f in filenames if "timeslices_" in f]
        assert len(timeslice_plots) > 0, (
            "Expected at least one timeslice plot to be generated"
        )

        # Validate that the timeslice plot was created successfully
        ttft_data = sample_timeslice_data[
            sample_timeslice_data["Metric"] == "Time to First Token"
        ]

        # Create a figure to verify basic structure
        plot_df = ttft_data[ttft_data["Stat"] == "avg"][["Timeslice", "Value"]]
        plot_df = plot_df.rename(columns={"Value": "avg"})
        fig = single_run_exporter.plot_generator.create_time_series_histogram(
            df=plot_df,
            x_col="Timeslice",
            y_col="avg",
            title="Test",
            x_label="Time (s)",
            y_label="TTFT (ms)",
            slice_duration=10.0,
        )

        # Verify basic plot structure
        assert fig.data is not None, "Figure should have data"
        assert len(fig.data) > 0, "Figure should have at least one trace"
        assert fig.layout.xaxis.title.text == "Time (s)"
        assert fig.layout.yaxis.title.text == "TTFT (ms)"

    @requires_chrome
    def test_timeslices_plot_handles_missing_data_gracefully(
        self,
        single_run_exporter,
        sample_single_run_data,
        sample_available_metrics,
        sample_plot_specs,
    ):
        """Test that missing timeslice data is handled gracefully."""
        # Run without timeslice data (None)
        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        generated_files = single_run_exporter.export(
            sample_single_run_data, sample_available_metrics, plot_specs
        )

        # Should generate plots, but no timeslice plots since data is missing
        assert len(generated_files) > 0

        filenames = {f.name for f in generated_files}
        # Should not generate timeslice plots when data is missing
        timeslice_plots = [f for f in filenames if "timeslices_" in f]
        assert len(timeslice_plots) == 0, (
            "Should not generate timeslice plots without data"
        )
        assert "timeslices_ttft.png" not in filenames

    @requires_chrome
    def test_uniform_requests_no_warning(
        self,
        single_run_exporter,
        tmp_path,
        sample_available_metrics,
        sample_plot_specs,
    ):
        """Test that uniform requests (identical ISL/OSL) show no warning."""
        per_request_data = pd.DataFrame(
            {
                "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
                "time_to_first_token": [45.0 + i * 2 for i in range(10)],
                "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
                "request_latency": [900.0 + i * 10 for i in range(10)],
                "input_sequence_length": [100] * 10,
                "output_sequence_length": [200] * 10,
            }
        )

        timeslice_data = pd.DataFrame(
            {
                "Timeslice": [0, 0, 1, 1],
                "Metric": ["Request Throughput", "Request Throughput"] * 2,
                "Unit": ["req/s"] * 4,
                "Stat": ["avg", "min", "avg", "min"],
                "Value": [10.0, 8.0, 12.0, 9.0],
            }
        )

        run_data = RunData(
            metadata=RunMetadata(
                run_name="uniform_run",
                run_path=tmp_path / "uniform_run",
                model="Test",
                concurrency=1,
            ),
            requests=per_request_data,
            aggregated={},
            timeslices=timeslice_data,
            slice_duration=10.0,
        )

        is_uniform, warning = validate_request_uniformity(run_data)
        assert is_uniform is True
        assert warning is None

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        generated_files = single_run_exporter.export(
            run_data, sample_available_metrics, plot_specs
        )
        throughput_plot = [
            f for f in generated_files if "timeslices_throughput" in f.name
        ]
        assert len(throughput_plot) == 1

        fig = single_run_exporter.plot_generator.create_time_series_histogram(
            df=timeslice_data[timeslice_data["Stat"] == "avg"][
                ["Timeslice", "Value"]
            ].rename(columns={"Value": "avg"}),
            x_col="Timeslice",
            y_col="avg",
            title="Request Throughput Across Time Slices",
            x_label="Time (s)",
            y_label="Request Throughput (req/s)",
            slice_duration=10.0,
            warning_text=warning,
        )

        if hasattr(fig.layout, "annotations") and fig.layout.annotations:
            for ann in fig.layout.annotations:
                assert "varying ISL/OSL" not in ann.text

    def test_non_uniform_isl_shows_warning(
        self,
        single_run_exporter,
        tmp_path,
        sample_available_metrics,
        sample_plot_specs,
    ):
        """Test that non-uniform ISL (varying input lengths) shows warning."""
        per_request_data = pd.DataFrame(
            {
                "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
                "time_to_first_token": [45.0 + i * 2 for i in range(10)],
                "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
                "request_latency": [900.0 + i * 10 for i in range(10)],
                "input_sequence_length": [
                    100,
                    200,
                    100,
                    300,
                    100,
                    200,
                    100,
                    300,
                    100,
                    200,
                ],
                "output_sequence_length": [200] * 10,
            }
        )

        timeslice_data = pd.DataFrame(
            {
                "Timeslice": [0, 0, 1, 1],
                "Metric": ["Request Throughput", "Request Throughput"] * 2,
                "Unit": ["req/s"] * 4,
                "Stat": ["avg", "min", "avg", "min"],
                "Value": [10.0, 8.0, 12.0, 9.0],
            }
        )

        run_data = RunData(
            metadata=RunMetadata(
                run_name="non_uniform_isl_run",
                run_path=tmp_path / "non_uniform_isl_run",
                model="Test",
                concurrency=1,
            ),
            requests=per_request_data,
            aggregated={},
            timeslices=timeslice_data,
            slice_duration=10.0,
        )

        is_uniform, warning = validate_request_uniformity(run_data)
        assert is_uniform is False
        assert warning is not None
        assert "varying ISL/OSL" in warning
        assert (
            "Req/sec throughput may not accurately represent workload capacity"
            in warning
        )

        fig = single_run_exporter.plot_generator.create_time_series_histogram(
            df=timeslice_data[timeslice_data["Stat"] == "avg"][
                ["Timeslice", "Value"]
            ].rename(columns={"Value": "avg"}),
            x_col="Timeslice",
            y_col="avg",
            title="Request Throughput Across Time Slices",
            x_label="Time (s)",
            y_label="Request Throughput (req/s)",
            slice_duration=10.0,
            warning_text=warning,
        )

        assert hasattr(fig.layout, "annotations")
        assert len(fig.layout.annotations) > 0

        warning_annotations = [
            ann for ann in fig.layout.annotations if "varying ISL/OSL" in ann.text
        ]
        assert len(warning_annotations) == 1

        warning_ann = warning_annotations[0]
        assert warning_ann.xref == "paper"
        assert warning_ann.yref == "paper"
        assert warning_ann.x == 0.5
        assert (
            warning_ann.y < 0
        )  # Warning is below x-axis (negative y in paper coordinates)
        assert warning_ann.y > -0.3  # But not too far below
        assert warning_ann.xanchor == "center"
        assert warning_ann.yanchor == "top"  # Top edge anchored to y position

    def test_non_uniform_osl_shows_warning(
        self,
        single_run_exporter,
        tmp_path,
        sample_available_metrics,
        sample_plot_specs,
    ):
        """Test that non-uniform OSL (varying output lengths) shows warning."""
        per_request_data = pd.DataFrame(
            {
                "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
                "time_to_first_token": [45.0 + i * 2 for i in range(10)],
                "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
                "request_latency": [900.0 + i * 10 for i in range(10)],
                "input_sequence_length": [100] * 10,
                "output_sequence_length": [
                    200,
                    300,
                    200,
                    400,
                    200,
                    300,
                    200,
                    400,
                    200,
                    300,
                ],
            }
        )

        run_data = RunData(
            metadata=RunMetadata(
                run_name="non_uniform_osl_run",
                run_path=tmp_path / "non_uniform_osl_run",
                model="Test",
                concurrency=1,
            ),
            requests=per_request_data,
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        is_uniform, warning = validate_request_uniformity(run_data)
        assert is_uniform is False
        assert warning is not None
        assert "varying ISL/OSL" in warning

    @requires_chrome
    def test_warning_only_on_throughput_plot(
        self,
        single_run_exporter,
        tmp_path,
        sample_available_metrics,
        sample_plot_specs,
    ):
        """Test that warning appears only on throughput plot, not other metrics."""
        per_request_data = pd.DataFrame(
            {
                "request_end_ns": [1000000000000 + i * 500000000 for i in range(10)],
                "time_to_first_token": [45.0 + i * 2 for i in range(10)],
                "inter_token_latency": [18.0 + i * 0.5 for i in range(10)],
                "request_latency": [900.0 + i * 10 for i in range(10)],
                "input_sequence_length": [
                    100,
                    200,
                    100,
                    200,
                    100,
                    200,
                    100,
                    200,
                    100,
                    200,
                ],
                "output_sequence_length": [200] * 10,
            }
        )

        timeslice_data = pd.DataFrame(
            {
                "Timeslice": [0, 0, 1, 1] * 4,
                "Metric": (
                    ["Time to First Token"] * 2
                    + ["Inter Token Latency"] * 2
                    + ["Request Throughput"] * 2
                    + ["Request Latency"] * 2
                )
                * 2,
                "Unit": ["ms"] * 4 + ["req/s"] * 4 + ["ms"] * 8,
                "Stat": ["avg", "min"] * 8,
                "Value": [45.0, 40.0, 18.0, 15.0, 10.0, 8.0, 900.0, 850.0] * 2,
            }
        )

        run_data = RunData(
            metadata=RunMetadata(
                run_name="multi_metric_run",
                run_path=tmp_path / "multi_metric_run",
                model="Test",
                concurrency=1,
            ),
            requests=per_request_data,
            aggregated={},
            timeslices=timeslice_data,
            slice_duration=10.0,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        generated_files = single_run_exporter.export(
            run_data, sample_available_metrics, plot_specs
        )

        assert len(generated_files) > 0

    def test_on_demand_loading_non_uniform_requests(
        self,
        single_run_exporter,
        tmp_path,
    ):
        """Test that warning works when loading ISL/OSL from disk (requests=None)."""
        run_path = tmp_path / "on_demand_test_run"
        run_path.mkdir()

        profile_data = []
        for i in range(10):
            profile_data.append(
                {
                    "metrics": {
                        "input_sequence_length": {"value": 100, "unit": "tokens"},
                        "output_sequence_length": {
                            "value": 200 + i * 50,
                            "unit": "tokens",
                        },
                    }
                }
            )

        with open(run_path / "profile_export.jsonl", "w") as f:
            for record in profile_data:
                f.write(json.dumps(record) + "\n")

        run_data = RunData(
            metadata=RunMetadata(
                run_name="on_demand_test_run",
                run_path=run_path,
                model="Test",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        is_uniform, warning = validate_request_uniformity(run_data)
        assert is_uniform is False
        assert warning is not None
        assert "varying ISL/OSL" in warning

    def test_on_demand_loading_uniform_requests(
        self,
        single_run_exporter,
        tmp_path,
    ):
        """Test that no warning appears when on-demand loading finds uniform requests."""
        run_path = tmp_path / "on_demand_uniform_run"
        run_path.mkdir()

        profile_data = []
        for _ in range(10):
            profile_data.append(
                {
                    "metrics": {
                        "input_sequence_length": {"value": 100, "unit": "tokens"},
                        "output_sequence_length": {"value": 200, "unit": "tokens"},
                    }
                }
            )

        with open(run_path / "profile_export.jsonl", "w") as f:
            for record in profile_data:
                f.write(json.dumps(record) + "\n")

        run_data = RunData(
            metadata=RunMetadata(
                run_name="on_demand_uniform_run",
                run_path=run_path,
                model="Test",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        is_uniform, warning = validate_request_uniformity(run_data)
        assert is_uniform is True
        assert warning is None

    def test_on_demand_loading_missing_file(
        self,
        single_run_exporter,
        tmp_path,
    ):
        """Test that missing profile_export.jsonl returns uniform (no warning)."""
        run_path = tmp_path / "missing_file_run"
        run_path.mkdir()

        run_data = RunData(
            metadata=RunMetadata(
                run_name="missing_file_run",
                run_path=run_path,
                model="Test",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
        )

        is_uniform, warning = validate_request_uniformity(run_data)
        assert is_uniform is True
        assert warning is None


class TestSharedExporterFunctionality:
    """Tests for shared functionality across both exporters."""

    def test_output_directory_created(
        self, tmp_path, sample_multi_run_data, sample_multi_run_plot_specs
    ):
        """Test that output directory is created if it doesn't exist."""
        output_dir = tmp_path / "new_directory" / "plots"
        assert not output_dir.exists()

        exporter = MultiRunPNGExporter(output_dir)
        exporter.export(
            sample_multi_run_data,
            {"display_names": {}, "units": {}},
            MULTI_RUN_PLOT_SPECS,
        )

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()

    @requires_chrome
    def test_export_handles_missing_metrics_gracefully(
        self, tmp_path, sample_available_metrics, sample_multi_run_plot_specs
    ):
        """Test that export handles missing metrics without crashing."""
        output_dir = tmp_path / "plots"
        exporter = MultiRunPNGExporter(output_dir)

        incomplete_data = [
            RunData(
                metadata=RunMetadata(
                    run_name="incomplete_run",
                    run_path=tmp_path / "incomplete_run",
                    model="Test",
                    concurrency=1,
                ),
                requests=None,
                aggregated={
                    "request_latency": {"p50": 100.0, "unit": "ms"},
                    # Missing other metrics
                },
                timeslices=None,
                slice_duration=None,
            )
        ]

        # Should not raise an exception
        generated_files = exporter.export(
            incomplete_data, sample_available_metrics, MULTI_RUN_PLOT_SPECS
        )

        # May generate fewer plots if metrics are missing
        assert isinstance(generated_files, list)


class TestSingleRunGPUPlots:
    """Tests for GPU plot generation in SingleRunPNGExporter."""

    @pytest.fixture
    def gpu_telemetry_df(self):
        """Create sample GPU telemetry DataFrame."""
        return pd.DataFrame(
            {
                "timestamp_s": [0.0, 1.0, 2.0, 3.0, 4.0],
                "gpu_index": [0, 0, 0, 0, 0],
                "gpu_uuid": ["GPU-0", "GPU-0", "GPU-0", "GPU-0", "GPU-0"],
                "gpu_utilization": [45.5, 67.2, 78.9, 82.3, 75.4],
                "gpu_memory_used": [2.5, 4.0, 5.5, 6.0, 5.0],
                "gpu_memory_free": [5.5, 4.0, 2.5, 2.0, 3.0],
                "gpu_power_usage": [50.0, 100.0, 150.0, 140.0, 120.0],
                "gpu_temperature": [60.0, 65.0, 70.0, 72.0, 68.0],
                "sm_clock_frequency": [1200.0, 1800.0, 2400.0, 2600.0, 2200.0],
                "memory_clock_frequency": [8000.0, 9000.0, 9500.0, 9500.0, 9200.0],
            }
        )

    @pytest.fixture
    def requests_df_for_gpu(self):
        """Create sample requests DataFrame for GPU throughput calculation."""
        return pd.DataFrame(
            {
                "request_start_ns": pd.to_datetime(
                    [
                        500_000_000,
                        800_000_000,
                        1_200_000_000,
                        1_800_000_000,
                        2_300_000_000,
                    ],
                    unit="ns",
                    utc=True,
                ),
                "request_end_ns": pd.to_datetime(
                    [
                        1_000_000_000,
                        1_500_000_000,
                        2_000_000_000,
                        2_500_000_000,
                        3_000_000_000,
                    ],
                    unit="ns",
                    utc=True,
                ),
                "time_to_first_token": [50, 75, 100, 80, 90],
                "output_sequence_length": [100, 150, 200, 180, 220],
            }
        )

    @requires_chrome
    def test_generate_gpu_plots_with_telemetry(
        self,
        tmp_path,
        sample_available_metrics,
        sample_plot_specs,
        gpu_telemetry_df,
        requests_df_for_gpu,
    ):
        """Test that GPU plots are generated when telemetry data is available."""
        exporter = SingleRunPNGExporter(output_dir=tmp_path)

        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test Model",
                concurrency=1,
            ),
            requests=requests_df_for_gpu,
            aggregated={},
            timeslices=None,
            slice_duration=None,
            gpu_telemetry=gpu_telemetry_df,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        all_files = exporter.export(run, sample_available_metrics, plot_specs)

        # Check that GPU files were generated
        gpu_files = [f for f in all_files if "gpu" in f.name]
        assert len(gpu_files) > 0
        for file_path in gpu_files:
            assert file_path.exists()
            assert file_path.suffix == ".png"

    def test_generate_gpu_plots_no_telemetry(
        self, tmp_path, sample_available_metrics, sample_plot_specs
    ):
        """Test that no GPU plots are generated when telemetry data is missing."""
        exporter = SingleRunPNGExporter(output_dir=tmp_path)

        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test Model",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
            gpu_telemetry=None,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        all_files = exporter.export(run, sample_available_metrics, plot_specs)
        gpu_files = [f for f in all_files if "gpu" in f.name]

        assert gpu_files == []

    def test_generate_gpu_plots_empty_telemetry(
        self, tmp_path, sample_available_metrics, sample_plot_specs
    ):
        """Test that no GPU plots are generated when telemetry DataFrame is empty."""
        exporter = SingleRunPNGExporter(output_dir=tmp_path)

        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test Model",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
            gpu_telemetry=pd.DataFrame(),
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        all_files = exporter.export(run, sample_available_metrics, plot_specs)
        gpu_files = [f for f in all_files if "gpu" in f.name]

        assert gpu_files == []

    @requires_chrome
    def test_generate_gpu_utilization_with_throughput(
        self,
        tmp_path,
        sample_available_metrics,
        sample_plot_specs,
        gpu_telemetry_df,
        requests_df_for_gpu,
    ):
        """Test GPU utilization with throughput overlay plot generation."""
        exporter = SingleRunPNGExporter(output_dir=tmp_path)

        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test Model",
                concurrency=1,
            ),
            requests=requests_df_for_gpu,
            aggregated={},
            timeslices=None,
            slice_duration=None,
            gpu_telemetry=gpu_telemetry_df,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        all_files = exporter.export(run, sample_available_metrics, plot_specs)
        gpu_files = [
            f
            for f in all_files
            if f.name == "gpu_utilization_and_throughput_over_time.png"
        ]

        assert len(gpu_files) == 1
        assert gpu_files[0].exists()

    def test_generate_gpu_utilization_no_requests(
        self, tmp_path, sample_available_metrics, sample_plot_specs, gpu_telemetry_df
    ):
        """Test GPU utilization plot when requests data is missing."""
        exporter = SingleRunPNGExporter(output_dir=tmp_path)

        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test Model",
                concurrency=1,
            ),
            requests=None,
            aggregated={},
            timeslices=None,
            slice_duration=None,
            gpu_telemetry=gpu_telemetry_df,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        all_files = exporter.export(run, sample_available_metrics, plot_specs)
        gpu_util_files = [
            f
            for f in all_files
            if f.name == "gpu_utilization_and_throughput_over_time.png"
        ]

        assert gpu_util_files == []

    @requires_chrome
    def test_generate_dispersed_throughput_over_time(
        self, tmp_path, sample_available_metrics, sample_plot_specs, requests_df_for_gpu
    ):
        """Test dispersed throughput over time plot generation."""
        exporter = SingleRunPNGExporter(output_dir=tmp_path)

        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test Model",
                concurrency=4,
            ),
            requests=requests_df_for_gpu,
            aggregated={},
            timeslices=None,
            slice_duration=None,
            gpu_telemetry=None,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        all_files = exporter.export(run, sample_available_metrics, plot_specs)
        throughput_files = [
            f for f in all_files if f.name == "dispersed_throughput_over_time.png"
        ]

        assert len(throughput_files) == 1
        assert throughput_files[0].exists()

    @requires_chrome
    def test_generate_gpu_plots_multi_gpu_aggregation(
        self, tmp_path, sample_available_metrics, sample_plot_specs, requests_df_for_gpu
    ):
        """Test that GPU plots aggregate data correctly across multiple GPUs."""
        exporter = SingleRunPNGExporter(output_dir=tmp_path)

        multi_gpu_df = pd.DataFrame(
            {
                "timestamp_s": [0.0, 0.0, 1.0, 1.0, 2.0, 2.0],
                "gpu_index": [0, 1, 0, 1, 0, 1],
                "gpu_uuid": ["GPU-0", "GPU-1", "GPU-0", "GPU-1", "GPU-0", "GPU-1"],
                "gpu_utilization": [45.0, 55.0, 65.0, 75.0, 70.0, 80.0],
                "gpu_memory_used": [2.0, 3.0, 4.0, 5.0, 4.5, 5.5],
                "gpu_memory_free": [6.0, 5.0, 4.0, 3.0, 3.5, 2.5],
                "gpu_power_usage": [50.0, 60.0, 100.0, 110.0, 120.0, 130.0],
                "gpu_temperature": [60.0, 62.0, 65.0, 67.0, 68.0, 70.0],
                "sm_clock_frequency": [1200.0, 1300.0, 1800.0, 1900.0, 2200.0, 2300.0],
                "memory_clock_frequency": [
                    8000.0,
                    8100.0,
                    9000.0,
                    9100.0,
                    9200.0,
                    9300.0,
                ],
            }
        )

        run = RunData(
            metadata=RunMetadata(
                run_name="test_run",
                run_path=tmp_path / "test_run",
                model="Test Model",
                concurrency=1,
            ),
            requests=requests_df_for_gpu,
            aggregated={},
            timeslices=None,
            slice_duration=None,
            gpu_telemetry=multi_gpu_df,
        )

        plot_specs = SINGLE_RUN_PLOT_SPECS + TIMESLICE_PLOT_SPECS + GPU_PLOT_SPECS
        all_files = exporter.export(run, sample_available_metrics, plot_specs)
        gpu_files = [f for f in all_files if "gpu" in f.name]

        assert len(gpu_files) > 0
        for file_path in gpu_files:
            assert file_path.exists()


class TestDualAxisHandler:
    """Tests for dual-axis handler."""

    @pytest.fixture
    def plot_generator(self):
        """Create a PlotGenerator instance for testing."""
        return PlotGenerator()

    @pytest.fixture
    def dual_axis_handler(self, plot_generator):
        """Create a DualAxisHandler instance for testing."""
        return DualAxisHandler(plot_generator=plot_generator)

    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance for testing."""
        return DataLoader()

    @pytest.fixture
    def available_metrics(self):
        """Sample available metrics dictionary."""
        return {
            "throughput_tokens_per_sec": {
                "display_name": "Output Tokens/sec",
                "unit": "",
            },
            "gpu_utilization": {
                "display_name": "GPU Utilization",
                "unit": "%",
            },
            "timestamp_s": {
                "display_name": "Time",
                "unit": "s",
            },
        }

    def test_original_gpu_plot_spec(
        self,
        dual_axis_handler,
        data_loader,
        available_metrics,
        single_run_dir,
    ):
        """Test that the original GPU utilization plot still works."""
        run_data = data_loader.load_run(single_run_dir)

        spec = PlotSpec(
            name="gpu_utilization_and_throughput_over_time",
            plot_type=PlotType.DUAL_AXIS,
            metrics=[
                MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="throughput_tokens_per_sec",
                    source=DataSource.REQUESTS,
                    axis="y",
                ),
                MetricSpec(
                    name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
                ),
            ],
            title="Output Token Throughput with GPU Utilization",
            primary_mode="lines",
            primary_line_shape="hv",
            primary_fill=None,
            secondary_mode="lines",
            secondary_line_shape=None,
            secondary_fill="tozeroy",
            supplementary_col="active_requests",
        )

        assert dual_axis_handler.can_handle(spec, run_data)
        fig = dual_axis_handler.create_plot(spec, run_data, available_metrics)
        assert fig is not None
        assert len(fig.data) > 0

    def test_custom_dual_axis_plot_different_name(
        self,
        dual_axis_handler,
        data_loader,
        available_metrics,
        single_run_dir,
    ):
        """Test that a dual-axis plot with a different name works (not hardcoded)."""
        run_data = data_loader.load_run(single_run_dir)

        spec = PlotSpec(
            name="my_custom_dual_axis_plot",
            plot_type=PlotType.DUAL_AXIS,
            metrics=[
                MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="throughput_tokens_per_sec",
                    source=DataSource.REQUESTS,
                    axis="y",
                ),
                MetricSpec(
                    name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
                ),
            ],
            title="Custom Dual-Axis Plot",
            primary_mode="lines",
            primary_line_shape="linear",
            primary_fill=None,
            secondary_mode="lines",
            secondary_line_shape=None,
            secondary_fill="tozeroy",
        )

        assert dual_axis_handler.can_handle(spec, run_data)
        fig = dual_axis_handler.create_plot(spec, run_data, available_metrics)
        assert fig is not None
        assert len(fig.data) > 0

    def test_custom_styling_parameters(
        self,
        dual_axis_handler,
        data_loader,
        available_metrics,
        single_run_dir,
    ):
        """Test that custom styling parameters are applied."""
        run_data = data_loader.load_run(single_run_dir)

        spec = PlotSpec(
            name="styled_dual_axis_plot",
            plot_type=PlotType.DUAL_AXIS,
            metrics=[
                MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="throughput_tokens_per_sec",
                    source=DataSource.REQUESTS,
                    axis="y",
                ),
                MetricSpec(
                    name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
                ),
            ],
            title="Custom Styled Plot",
            primary_mode="markers",
            primary_line_shape=None,
            primary_fill="tozeroy",
            secondary_mode="lines+markers",
            secondary_line_shape="spline",
            secondary_fill=None,
        )

        fig = dual_axis_handler.create_plot(spec, run_data, available_metrics)
        assert fig is not None
        assert len(fig.data) > 0

    def test_missing_x_metric_uses_default(
        self,
        dual_axis_handler,
        data_loader,
        available_metrics,
        single_run_dir,
    ):
        """Test that missing x metric defaults to timestamp_s."""
        run_data = data_loader.load_run(single_run_dir)

        spec = PlotSpec(
            name="no_x_metric_plot",
            plot_type=PlotType.DUAL_AXIS,
            metrics=[
                MetricSpec(
                    name="throughput_tokens_per_sec",
                    source=DataSource.REQUESTS,
                    axis="y",
                ),
                MetricSpec(
                    name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
                ),
            ],
            title="Plot Without X Metric",
        )

        fig = dual_axis_handler.create_plot(spec, run_data, available_metrics)
        assert fig is not None
        assert len(fig.data) > 0

    def test_can_handle_checks_gpu_telemetry(self, dual_axis_handler):
        """Test that can_handle properly checks for GPU telemetry data."""
        metadata = RunMetadata(
            run_name="test_run",
            run_path=Path("/tmp"),
        )

        run_data = RunData(
            metadata=metadata,
            requests=pd.DataFrame(),
            timeslices=pd.DataFrame(),
            aggregated={},
            gpu_telemetry=None,
            slice_duration=None,
        )

        spec = PlotSpec(
            name="test_plot",
            plot_type=PlotType.DUAL_AXIS,
            metrics=[
                MetricSpec(
                    name="throughput_tokens_per_sec",
                    source=DataSource.REQUESTS,
                    axis="y",
                ),
                MetricSpec(
                    name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
                ),
            ],
            title="Test Plot",
        )

        assert not dual_axis_handler.can_handle(spec, run_data)

    def test_empty_primary_data_raises_error(
        self,
        dual_axis_handler,
        data_loader,
        available_metrics,
        single_run_dir,
    ):
        """Test that empty primary data raises an appropriate error."""
        run_data = data_loader.load_run(single_run_dir)

        spec = PlotSpec(
            name="invalid_metric_plot",
            plot_type=PlotType.DUAL_AXIS,
            metrics=[
                MetricSpec(
                    name="nonexistent_metric", source=DataSource.REQUESTS, axis="y"
                ),
                MetricSpec(
                    name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
                ),
            ],
            title="Invalid Metric Plot",
        )

        with pytest.raises((ValueError, KeyError)):
            dual_axis_handler.create_plot(spec, run_data, available_metrics)

    def test_metric_prep_registry(self, dual_axis_handler):
        """Test that the metric preparation registry contains expected functions."""
        assert "throughput_tokens_per_sec" in dual_axis_handler.METRIC_PREP_FUNCTIONS
        assert "gpu_utilization" in dual_axis_handler.METRIC_PREP_FUNCTIONS

        assert callable(
            dual_axis_handler.METRIC_PREP_FUNCTIONS["throughput_tokens_per_sec"]
        )
        assert callable(dual_axis_handler.METRIC_PREP_FUNCTIONS["gpu_utilization"])

    def test_axis_labels_from_available_metrics(
        self, dual_axis_handler, data_loader, single_run_dir
    ):
        """Test that axis labels are derived from available_metrics."""
        run_data = data_loader.load_run(single_run_dir)

        custom_metrics = {
            "throughput_tokens_per_sec": {
                "display_name": "Custom Throughput Label",
                "unit": "tok/s",
            },
            "gpu_utilization": {
                "display_name": "Custom GPU Label",
                "unit": "percent",
            },
        }

        spec = PlotSpec(
            name="custom_labels_plot",
            plot_type=PlotType.DUAL_AXIS,
            metrics=[
                MetricSpec(name="timestamp_s", source=DataSource.REQUESTS, axis="x"),
                MetricSpec(
                    name="throughput_tokens_per_sec",
                    source=DataSource.REQUESTS,
                    axis="y",
                ),
                MetricSpec(
                    name="gpu_utilization", source=DataSource.GPU_TELEMETRY, axis="y2"
                ),
            ],
            title="Custom Labels Plot",
        )

        fig = dual_axis_handler.create_plot(spec, run_data, custom_metrics)
        assert fig is not None

        assert "Custom Throughput Label" in str(fig.layout.yaxis.title.text)
        assert "Custom GPU Label" in str(fig.layout.yaxis2.title.text)


class TestMultiRunServerMetricsAggregation:
    """Tests for server metrics aggregation in _runs_to_dataframe method."""

    @staticmethod
    def _make_run_with_server_metrics(
        tmp_path: Path, metric_name: str, series_data: dict
    ) -> list[RunData]:
        """Create a RunData with server_metrics_aggregated for testing."""
        return [
            RunData(
                metadata=RunMetadata(
                    run_name="run_001",
                    run_path=tmp_path / "run_001",
                    model="TestModel",
                    concurrency=4,
                ),
                requests=None,
                aggregated={},
                timeslices=None,
                slice_duration=None,
                server_metrics_aggregated={
                    metric_name: {"http://localhost:8000": {"labels": series_data}}
                },
            )
        ]

    @pytest.mark.parametrize(
        "metric_type,stats,expected",
        [
            (PrometheusMetricType.COUNTER, {"rate": 100.5}, 100.5),
            (PrometheusMetricType.GAUGE, {"avg": 42.5}, 42.5),
            (PrometheusMetricType.HISTOGRAM, {"avg": 150.0, "rate": 50.0}, 150.0),
            (PrometheusMetricType.UNKNOWN, {"avg": 25.5, "rate": 10.0}, 25.5),
            (None, {"avg": 30.0, "rate": 5.0}, 30.0),  # Missing type defaults to avg
        ],
        ids=["counter", "gauge", "histogram", "unknown", "missing_type"],
    )  # fmt: skip
    def test_metric_type_extracts_correct_stat(
        self, multi_run_exporter, tmp_path, metric_type, stats, expected
    ):
        """Test that each metric type extracts the correct stat value."""
        series_data = {"stats": stats}
        if metric_type is not None:
            series_data["type"] = metric_type
        runs = self._make_run_with_server_metrics(tmp_path, "test_metric", series_data)

        df = multi_run_exporter._runs_to_dataframe(runs, {})

        assert df["test_metric"].iloc[0] == expected

    @pytest.mark.parametrize(
        "metric_type,stats",
        [
            (PrometheusMetricType.COUNTER, {"rate": None, "avg": 50.0}),
            (PrometheusMetricType.GAUGE, {"avg": None, "rate": 10.0}),
        ],
        ids=["counter_none_rate", "gauge_none_avg"],
    )  # fmt: skip
    def test_none_stat_values_are_skipped(
        self, multi_run_exporter, tmp_path, metric_type, stats
    ):
        """Test that None values for the expected stat field are skipped."""
        series_data = {"type": metric_type, "stats": stats}
        runs = self._make_run_with_server_metrics(tmp_path, "test_metric", series_data)

        df = multi_run_exporter._runs_to_dataframe(runs, {})

        assert "test_metric" not in df.columns

    def test_gauge_averages_across_endpoints(self, multi_run_exporter, tmp_path):
        """Test that GAUGE metrics average values across multiple endpoints."""
        runs = [
            RunData(
                metadata=RunMetadata(
                    run_name="run_001",
                    run_path=tmp_path / "run_001",
                    model="TestModel",
                    concurrency=4,
                ),
                requests=None,
                aggregated={},
                timeslices=None,
                slice_duration=None,
                server_metrics_aggregated={
                    "gpu_util": {
                        "http://host1:8000": {
                            "gpu_0": {
                                "type": PrometheusMetricType.GAUGE,
                                "stats": {"avg": 80.0},
                            }
                        },
                        "http://host2:8000": {
                            "gpu_0": {
                                "type": PrometheusMetricType.GAUGE,
                                "stats": {"avg": 60.0},
                            }
                        },
                    }
                },
            )
        ]

        df = multi_run_exporter._runs_to_dataframe(runs, {})

        assert df["gpu_util"].iloc[0] == 70.0  # (80 + 60) / 2

    def test_counter_sums_across_endpoints(self, multi_run_exporter, tmp_path):
        """Test that COUNTER metrics sum rates across multiple endpoints."""
        runs = [
            RunData(
                metadata=RunMetadata(
                    run_name="run_001",
                    run_path=tmp_path / "run_001",
                    model="TestModel",
                    concurrency=4,
                ),
                requests=None,
                aggregated={},
                timeslices=None,
                slice_duration=None,
                server_metrics_aggregated={
                    "total_reqs": {
                        "http://host1:8000": {
                            "l1": {
                                "type": PrometheusMetricType.COUNTER,
                                "stats": {"rate": 50.0},
                            }
                        },
                        "http://host2:8000": {
                            "l1": {
                                "type": PrometheusMetricType.COUNTER,
                                "stats": {"rate": 30.0},
                            }
                        },
                    }
                },
            )
        ]

        df = multi_run_exporter._runs_to_dataframe(runs, {})

        assert df["total_reqs"].iloc[0] == 80.0  # 50 + 30

    def test_object_based_stats(self, multi_run_exporter, tmp_path):
        """Test that object-based stats (with attributes) work correctly."""
        series_data = {
            "type": PrometheusMetricType.GAUGE,
            "stats": SimpleNamespace(avg=45.0, rate=None),
        }
        runs = self._make_run_with_server_metrics(tmp_path, "obj_metric", series_data)

        df = multi_run_exporter._runs_to_dataframe(runs, {})

        assert df["obj_metric"].iloc[0] == 45.0

    def test_static_value_without_stats(self, multi_run_exporter, tmp_path):
        """Test that static values (stats=None) use the value field."""
        series_data = {"stats": None, "value": 999.0}
        runs = self._make_run_with_server_metrics(
            tmp_path, "static_metric", series_data
        )

        df = multi_run_exporter._runs_to_dataframe(runs, {})

        assert df["static_metric"].iloc[0] == 999.0


class TestPlotSpecListValidation:
    """Tests for PlotSpec validation with list-based label_by and group_by."""

    def test_label_by_single_element_list(self):
        """Test that label_by with a single-element list is converted to string."""
        spec = PlotSpec(
            name="test_plot",
            plot_type=PlotType.PARETO,
            metrics=[
                MetricSpec(
                    name="request_latency",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="avg",
                ),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            label_by=["concurrency"],
        )

        assert spec.label_by == "concurrency"

    def test_group_by_single_element_list(self):
        """Test that group_by with a single-element list is converted to string."""
        spec = PlotSpec(
            name="test_plot",
            plot_type=PlotType.PARETO,
            metrics=[
                MetricSpec(
                    name="request_latency",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="avg",
                ),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            group_by=["model"],
        )

        assert spec.group_by == "model"

    def test_label_by_multi_element_list_raises_error(self):
        """Test that label_by with multi-element list raises ValueError."""
        with pytest.raises(ValueError, match="Multi-column grouping is not supported"):
            PlotSpec(
                name="test_plot",
                plot_type=PlotType.PARETO,
                metrics=[
                    MetricSpec(
                        name="request_latency",
                        source=DataSource.AGGREGATED,
                        axis="x",
                        stat="avg",
                    ),
                    MetricSpec(
                        name="request_throughput",
                        source=DataSource.AGGREGATED,
                        axis="y",
                        stat="avg",
                    ),
                ],
                label_by=["model", "concurrency"],
            )

    def test_group_by_multi_element_list_raises_error(self):
        """Test that group_by with multi-element list raises ValueError."""
        with pytest.raises(ValueError, match="Multi-column grouping is not supported"):
            PlotSpec(
                name="test_plot",
                plot_type=PlotType.PARETO,
                metrics=[
                    MetricSpec(
                        name="request_latency",
                        source=DataSource.AGGREGATED,
                        axis="x",
                        stat="avg",
                    ),
                    MetricSpec(
                        name="request_throughput",
                        source=DataSource.AGGREGATED,
                        axis="y",
                        stat="avg",
                    ),
                ],
                group_by=["model", "batch_size"],
            )

    def test_label_by_and_group_by_multi_element_lists_raise_error(self):
        """Test that both label_by and group_by with multi-element lists raise ValueError."""
        with pytest.raises(ValueError, match="Multi-column grouping is not supported"):
            PlotSpec(
                name="test_plot",
                plot_type=PlotType.PARETO,
                metrics=[
                    MetricSpec(
                        name="request_latency",
                        source=DataSource.AGGREGATED,
                        axis="x",
                        stat="avg",
                    ),
                    MetricSpec(
                        name="request_throughput",
                        source=DataSource.AGGREGATED,
                        axis="y",
                        stat="avg",
                    ),
                ],
                label_by=["concurrency", "batch_size"],
                group_by=["model", "endpoint"],
            )

    def test_label_by_as_plain_string_accepted(self):
        """Test that plain strings are accepted for label_by."""
        spec = PlotSpec(
            name="test_plot",
            plot_type=PlotType.PARETO,
            metrics=[
                MetricSpec(
                    name="request_latency",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="avg",
                ),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            label_by="concurrency",
        )
        assert spec.label_by == "concurrency"

    def test_group_by_as_plain_string_accepted(self):
        """Test that plain strings are accepted for group_by."""
        spec = PlotSpec(
            name="test_plot",
            plot_type=PlotType.PARETO,
            metrics=[
                MetricSpec(
                    name="request_latency",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="avg",
                ),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            group_by="model",
        )
        assert spec.group_by == "model"

    def test_label_by_and_group_by_as_none(self):
        """Test that label_by and group_by can be None."""
        spec = PlotSpec(
            name="test_plot",
            plot_type=PlotType.PARETO,
            metrics=[
                MetricSpec(
                    name="request_latency",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="avg",
                ),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            label_by=None,
            group_by=None,
        )

        assert spec.label_by is None
        assert spec.group_by is None

    def test_empty_list_converts_to_none(self):
        """Test that empty lists are converted to None."""
        spec = PlotSpec(
            name="test_plot",
            plot_type=PlotType.PARETO,
            metrics=[
                MetricSpec(
                    name="request_latency",
                    source=DataSource.AGGREGATED,
                    axis="x",
                    stat="avg",
                ),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
            label_by=[],
        )

        assert spec.label_by is None
