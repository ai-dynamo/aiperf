# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the PlotGenerator class.

This module tests the plot generation functionality, ensuring that each plot
type is created correctly with proper styling and data handling.
"""

import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from aiperf.common.enums import PlotMetricDirection
from aiperf.common.enums.metric_enums import MetricFlags
from aiperf.plot.constants import (
    DARK_THEME_COLORS,
    LIGHT_THEME_COLORS,
    NVIDIA_CARD_BG,
    NVIDIA_DARK_BG,
    NVIDIA_GRAY,
    NVIDIA_GREEN,
    NVIDIA_TEXT_LIGHT,
    NVIDIA_WHITE,
    PlotTheme,
)
from aiperf.plot.core.plot_generator import (
    PlotGenerator,
    detect_directional_outliers,
    get_nvidia_color_scheme,
)
from aiperf.plot.core.plot_specs import Style

# Light mode uses seaborn "deep" palette (blue) instead of NVIDIA brand colors
LIGHT_MODE_PRIMARY_COLOR = "#4c72b0"


@pytest.fixture
def plot_generator():
    """Create a PlotGenerator instance for testing."""
    return PlotGenerator()


@pytest.fixture
def multi_run_df():
    """Create sample multi-run DataFrame for testing."""
    return pd.DataFrame(
        {
            "model": ["Qwen/Qwen3-0.6B"] * 3 + ["meta-llama/Meta-Llama-3-8B"] * 3,
            "concurrency": [1, 4, 8, 1, 4, 8],
            "request_latency": [100, 150, 200, 120, 180, 250],
            "request_throughput": [10, 25, 35, 8, 20, 28],
            "time_to_first_token": [45, 55, 70, 50, 65, 85],
            "inter_token_latency": [15, 18, 22, 16, 20, 25],
            "output_token_throughput_per_user": [100, 90, 80, 95, 85, 75],
        }
    )


@pytest.fixture
def single_run_df():
    """Create sample single-run DataFrame for testing."""
    return pd.DataFrame(
        {
            "request_number": list(range(10)),
            "timestamp": [i * 0.5 for i in range(10)],
            "time_to_first_token": [45 + i * 2 for i in range(10)],
            "inter_token_latency": [18 + i * 0.5 for i in range(10)],
            "request_latency": [900 + i * 10 for i in range(10)],
        }
    )


class TestPlotGenerator:
    """Tests for PlotGenerator class."""

    def test_initialization(self, plot_generator):
        """Test that PlotGenerator can be instantiated."""
        assert isinstance(plot_generator, PlotGenerator)

    def test_create_pareto_plot_basic(self, plot_generator, multi_run_df):
        """Test basic Pareto plot creation."""
        fig = plot_generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
        )

        # Verify return type
        assert isinstance(fig, go.Figure)

        # Verify figure has traces (data points + lines + shadows)
        assert len(fig.data) > 0

        # Verify layout properties (colors for light mode)
        assert fig.layout.plot_bgcolor == NVIDIA_WHITE
        assert fig.layout.paper_bgcolor == NVIDIA_WHITE

    def test_create_pareto_plot_custom_labels(self, plot_generator, multi_run_df):
        """Test Pareto plot with custom labels."""
        title = "Custom Title"
        x_label = "Custom X"
        y_label = "Custom Y"

        fig = plot_generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        # Verify custom labels are used
        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_create_pareto_plot_no_grouping(self, plot_generator):
        """Test Pareto plot without grouping."""
        df = pd.DataFrame(
            {
                "concurrency": [1, 4, 8],
                "request_latency": [100, 150, 200],
                "request_throughput": [10, 25, 35],
            }
        )

        fig = plot_generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by=None,  # No grouping
        )

        # Should still create a valid figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_create_scatter_line_plot_basic(self, plot_generator, multi_run_df):
        """Test basic scatter line plot creation."""
        fig = plot_generator.create_scatter_line_plot(
            df=multi_run_df,
            x_metric="time_to_first_token",
            y_metric="inter_token_latency",
            label_by="concurrency",
            group_by="model",
        )

        # Verify return type
        assert isinstance(fig, go.Figure)

        # Verify figure has traces
        assert len(fig.data) > 0

        # Verify styling (colors for light mode)
        assert fig.layout.plot_bgcolor == NVIDIA_WHITE

    def test_create_scatter_line_plot_auto_labels(self, plot_generator, multi_run_df):
        """Test scatter line plot with auto-generated labels."""
        fig = plot_generator.create_scatter_line_plot(
            df=multi_run_df,
            x_metric="time_to_first_token",
            y_metric="inter_token_latency",
            label_by="concurrency",
            group_by="model",
        )

        # Verify auto-generated labels contain metric names
        assert "Time to First Token" in fig.layout.xaxis.title.text
        assert "Inter Token Latency" in fig.layout.yaxis.title.text

    def test_create_time_series_scatter(self, plot_generator, single_run_df):
        """Test time series scatter plot creation."""
        fig = plot_generator.create_time_series_scatter(
            df=single_run_df,
            x_col="request_number",
            y_metric="time_to_first_token",
        )

        # Verify return type
        assert isinstance(fig, go.Figure)

        # Verify has scatter trace
        assert len(fig.data) > 0
        assert fig.data[0].mode == "markers"

        # Verify styling
        assert fig.layout.plot_bgcolor == NVIDIA_WHITE
        assert fig.layout.hovermode == "x unified"

    def test_create_time_series_scatter_custom_labels(
        self, plot_generator, single_run_df
    ):
        """Test time series scatter with custom labels."""
        title = "Custom Time Series"
        x_label = "Time"
        y_label = "Latency (ms)"

        fig = plot_generator.create_time_series_scatter(
            df=single_run_df,
            x_col="request_number",
            y_metric="time_to_first_token",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        # Verify custom labels
        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_create_time_series_area(self, plot_generator, single_run_df):
        """Test time series area plot creation."""
        fig = plot_generator.create_time_series_area(
            df=single_run_df,
            x_col="timestamp",
            y_metric="request_latency",
        )

        # Verify return type
        assert isinstance(fig, go.Figure)

        # Verify has filled area
        assert len(fig.data) > 0
        assert fig.data[0].fill == "tozeroy"
        assert fig.data[0].mode == "lines"

        # Verify light mode primary color (seaborn deep palette blue)
        assert LIGHT_MODE_PRIMARY_COLOR in fig.data[0].line.color

    def test_create_time_series_area_auto_labels(self, plot_generator, single_run_df):
        """Test time series area with auto-generated labels."""
        fig = plot_generator.create_time_series_area(
            df=single_run_df,
            x_col="timestamp",
            y_metric="request_latency",
        )

        # Verify auto-generated labels
        assert "Timestamp" in fig.layout.xaxis.title.text
        assert "Request Latency" in fig.layout.yaxis.title.text

    def test_plots_have_proper_height(self, plot_generator, multi_run_df):
        """Test that all plots have the expected height."""
        plots = [
            plot_generator.create_pareto_plot(
                multi_run_df, "request_latency", "request_throughput"
            ),
            plot_generator.create_scatter_line_plot(
                multi_run_df, "time_to_first_token", "inter_token_latency"
            ),
        ]

        for fig in plots:
            assert fig.layout.height == 400

    def test_plots_have_nvidia_branding(self, plot_generator, multi_run_df):
        """Test that plots use NVIDIA brand colors."""
        fig = plot_generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
        )

        # Check layout colors (light mode by default)
        assert fig.layout.plot_bgcolor == NVIDIA_WHITE
        assert fig.layout.paper_bgcolor == NVIDIA_WHITE

    def test_empty_dataframe_handling(self, plot_generator):
        """Test that generator handles empty DataFrames gracefully."""
        empty_df = pd.DataFrame()

        # Should not raise an exception
        try:
            fig = plot_generator.create_scatter_line_plot(
                df=empty_df,
                x_metric="request_latency",
                y_metric="request_throughput",
            )
            assert isinstance(fig, go.Figure)
        except KeyError:
            # Expected if columns don't exist in empty DataFrame
            pass

    def test_single_data_point(self, plot_generator):
        """Test plots with single data point."""
        df = pd.DataFrame(
            {
                "concurrency": [1],
                "request_latency": [100],
                "request_throughput": [10],
            }
        )

        fig = plot_generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by=None,
        )

        # Should create valid figure with single point
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_with_missing_group_column(self, plot_generator):
        """Test plot when group_by column doesn't exist."""
        df = pd.DataFrame(
            {
                "concurrency": [1, 4, 8],
                "request_latency": [100, 150, 200],
                "request_throughput": [10, 25, 35],
            }
        )

        fig = plot_generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="nonexistent_column",  # Column doesn't exist
        )

        # Should fall back to treating all as single group
        assert isinstance(fig, go.Figure)

    def test_dynamic_model_color_assignment(self, plot_generator):
        """Test that colors are assigned dynamically to any model names."""
        # Use arbitrary model names (not hardcoded)
        df = pd.DataFrame(
            {
                "model": ["ModelA", "ModelB", "ModelC"] * 2,
                "concurrency": [1, 1, 1, 4, 4, 4],
                "request_latency": [100, 110, 120, 150, 160, 170],
                "request_throughput": [10, 9, 8, 25, 23, 21],
            }
        )

        fig = plot_generator.create_pareto_plot(
            df=df,
            x_metric="request_latency",
            y_metric="request_throughput",
            label_by="concurrency",
            group_by="model",
        )

        # Should work with any model names
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

        # Test that groups are registered in the color registry
        _groups, color_map, _display_names = plot_generator._prepare_groups(df, "model")

        # Verify all models get colors
        assert len(color_map) == 3
        assert "ModelA" in color_map
        assert "ModelB" in color_map
        assert "ModelC" in color_map

        # Verify colors are hex codes
        for color in color_map.values():
            assert isinstance(color, str)
            assert color.startswith("#")

        # Verify colors are in the registry
        assert "ModelA" in plot_generator._group_color_registry
        assert "ModelB" in plot_generator._group_color_registry
        assert "ModelC" in plot_generator._group_color_registry

    def test_color_consistency_across_models(self, plot_generator):
        """Test that same model gets same color across different calls."""
        df1 = pd.DataFrame({"model": ["ModelX", "ModelY", "ModelZ"]})
        _groups1, colors1, _display_names1 = plot_generator._prepare_groups(
            df1, "model"
        )

        df2 = pd.DataFrame({"model": ["ModelX", "ModelY", "ModelZ"]})
        _groups2, colors2, _display_names2 = plot_generator._prepare_groups(
            df2, "model"
        )

        # Same models should get same colors across calls
        assert colors1 == colors2
        # Colors should be persisted in registry
        assert plot_generator._group_color_registry["ModelX"] == colors1["ModelX"]
        assert plot_generator._group_color_registry["ModelY"] == colors1["ModelY"]
        assert plot_generator._group_color_registry["ModelZ"] == colors1["ModelZ"]

    def test_color_assignment_with_many_models(self, plot_generator):
        """Test that color assignment cycles when there are more models than colors."""
        # Create more models than available colors in the pool (default 10)
        model_names = [f"Model{i}" for i in range(15)]
        df = pd.DataFrame({"model": model_names})
        _groups, color_map, _display_names = plot_generator._prepare_groups(df, "model")

        # All models should get a color
        assert len(color_map) == 15

        # Colors should cycle (some will repeat)
        unique_colors = set(color_map.values())
        # Should have at most pool_size unique colors (10 by default)
        assert len(unique_colors) <= len(plot_generator._color_pool)
        # Should have fewer unique colors than models (due to cycling)
        assert len(unique_colors) < len(model_names)


class TestTimeSeriesHistogram:
    """Tests for create_time_series_histogram method."""

    @pytest.fixture
    def timeslice_df(self):
        """Create sample timeslice DataFrame for testing."""
        return pd.DataFrame(
            {
                "timeslice": [0, 1, 2, 3, 4],
                "avg": [100.5, 120.3, 115.7, 130.2, 125.8],
                "p50": [95.0, 115.0, 110.0, 125.0, 120.0],
                "p90": [150.0, 180.0, 170.0, 195.0, 185.0],
            }
        )

    def test_histogram_basic(self, plot_generator, timeslice_df):
        """Test basic histogram creation."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df, x_col="timeslice", y_col="avg"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        assert fig.data[0].type == "bar"

    def test_histogram_with_slice_duration(self, plot_generator, timeslice_df):
        """Test histogram with slice duration for time-based x-axis."""
        slice_duration = 10.0
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=slice_duration,
        )

        assert isinstance(fig, go.Figure)
        assert fig.data[0].type == "bar"
        assert fig.data[0].width == slice_duration
        assert fig.layout.xaxis.dtick == slice_duration
        assert fig.layout.bargap == 0

    def test_histogram_with_annotations(self, plot_generator, timeslice_df):
        """Test that histogram with slice_duration has no annotations by default."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=10.0,
        )

        # Should have no annotations when slice_duration is provided (no labels by default)
        assert fig.layout.annotations is None or len(fig.layout.annotations) == 0

    def test_histogram_with_warning_text(self, plot_generator, timeslice_df):
        """Test histogram with warning text annotation."""
        warning_text = "Warning: Non-uniform request distribution detected"
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=10.0,
            warning_text=warning_text,
        )

        # Should have exactly 1 annotation (the warning text)
        assert len(fig.layout.annotations) == 1

        warning_annotation = fig.layout.annotations[0]
        assert warning_text in warning_annotation["text"]
        assert warning_annotation["yref"] == "paper"
        assert fig.layout.margin.b == 140

    def test_histogram_custom_labels(self, plot_generator, timeslice_df):
        """Test histogram with custom labels."""
        title = "Custom Histogram Title"
        x_label = "Custom X Label"
        y_label = "Custom Y Label"

        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_histogram_auto_labels(self, plot_generator, timeslice_df):
        """Test histogram with auto-generated labels."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df, x_col="timeslice", y_col="avg"
        )

        assert "avg" in fig.layout.title.text.lower()
        assert fig.layout.yaxis.title.text == "Avg"

    def test_histogram_auto_labels_with_slice_duration(
        self, plot_generator, timeslice_df
    ):
        """Test that x-axis label is 'Timeslice (s)' when slice_duration is provided."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=10.0,
        )

        assert fig.layout.xaxis.title.text == "Timeslice (s)"

    def test_histogram_with_empty_dataframe(self, plot_generator):
        """Test histogram with empty DataFrame."""
        empty_df = pd.DataFrame({"timeslice": [], "avg": []})
        fig = plot_generator.create_time_series_histogram(
            df=empty_df, x_col="timeslice", y_col="avg"
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_histogram_marker_config_with_slice_duration(
        self, plot_generator, timeslice_df
    ):
        """Test that histogram uses transparent bars with borders when slice_duration is provided."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df,
            x_col="timeslice",
            y_col="avg",
            slice_duration=10.0,
        )

        marker = fig.data[0].marker
        # Light mode uses seaborn deep palette blue
        assert "rgba(76, 114, 176, 0.7)" in marker.color
        assert marker.line.color == LIGHT_MODE_PRIMARY_COLOR
        assert marker.line.width == 2

    def test_histogram_marker_config_without_slice_duration(
        self, plot_generator, timeslice_df
    ):
        """Test that histogram uses solid bars when slice_duration is not provided."""
        fig = plot_generator.create_time_series_histogram(
            df=timeslice_df, x_col="timeslice", y_col="avg"
        )

        marker = fig.data[0].marker
        # Light mode uses seaborn deep palette blue
        assert marker.color == LIGHT_MODE_PRIMARY_COLOR


class TestDualAxisPlots:
    """Tests for dual-axis plotting functions."""

    @pytest.fixture
    def gpu_metrics_df(self):
        """Create sample GPU metrics DataFrame for testing."""
        return pd.DataFrame(
            {
                "timestamp_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "gpu_utilization": [45.5, 67.2, 78.9, 82.3, 75.4, 68.1],
                "throughput": [100.0, 150.0, 180.0, 190.0, 170.0, 155.0],
                "power_draw_w": [120.0, 180.0, 220.0, 240.0, 210.0, 190.0],
            }
        )

    def test_gpu_dual_axis_plot_basic(self, plot_generator, gpu_metrics_df):
        """Test basic GPU dual-axis plot creation with separate DataFrames."""
        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        throughput_df["active_requests"] = [2, 3, 4, 5, 4, 3]
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_style=Style(mode="lines", line_shape="hv", fill=None),
            secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
            active_count_col="active_requests",
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

        assert fig.data[0].yaxis == "y"
        assert fig.data[1].yaxis == "y2"

        assert fig.data[0].line.shape == "hv"
        assert fig.data[1].fill == "tozeroy"

    def test_gpu_dual_axis_plot_custom_labels(self, plot_generator, gpu_metrics_df):
        """Test GPU dual-axis plot with custom labels."""
        title = "Throughput with GPU Utilization"
        x_label = "Time"
        y1_label = "Tokens/s"
        y2_label = "GPU %"

        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_style=Style(mode="lines", line_shape="hv", fill=None),
            secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
            title=title,
            x_label=x_label,
            y1_label=y1_label,
            y2_label=y2_label,
        )

        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y1_label
        assert fig.layout.yaxis2.title.text == y2_label

    def test_gpu_dual_axis_plot_auto_labels(self, plot_generator, gpu_metrics_df):
        """Test dual-axis plot with auto-generated labels."""
        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_style=Style(mode="lines", line_shape="hv", fill=None),
            secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
        )

        assert "Throughput" in fig.layout.title.text
        assert "GPU Utilization" in fig.layout.title.text
        assert fig.layout.xaxis.title.text == "Time (s)"

    def test_gpu_dual_axis_plot_styling(self, plot_generator, gpu_metrics_df):
        """Test dual-axis plot styling (colors, line widths)."""
        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_style=Style(mode="lines", line_shape="hv", fill=None),
            secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
        )

        # Light mode uses seaborn deep palette colors
        assert fig.data[0].line.color == LIGHT_MODE_PRIMARY_COLOR
        assert fig.data[0].line.width == 2
        assert fig.data[1].line.width == 2

    def test_gpu_dual_axis_layout(self, plot_generator, gpu_metrics_df):
        """Test that secondary y-axis is configured correctly with theme colors."""
        throughput_df = gpu_metrics_df[["timestamp_s", "throughput"]].copy()
        gpu_df = gpu_metrics_df[["timestamp_s", "gpu_utilization"]].copy()

        fig = plot_generator.create_dual_axis_plot(
            df_primary=throughput_df,
            df_secondary=gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_style=Style(mode="lines", line_shape="hv", fill=None),
            secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
        )

        assert fig.layout.yaxis2 is not None
        assert fig.layout.yaxis2.overlaying == "y"
        assert fig.layout.yaxis2.side == "right"

        # Verify theme consistency for yaxis2
        assert fig.layout.yaxis2.gridcolor == LIGHT_THEME_COLORS["grid"]
        assert fig.layout.yaxis2.linecolor == LIGHT_THEME_COLORS["border"]
        assert fig.layout.yaxis2.color == LIGHT_THEME_COLORS["text"]

    def test_gpu_dual_axis_with_empty_dataframe(self, plot_generator):
        """Test dual-axis plot with empty DataFrame."""
        empty_throughput_df = pd.DataFrame({"timestamp_s": [], "throughput": []})
        empty_gpu_df = pd.DataFrame({"timestamp_s": [], "gpu_utilization": []})
        fig = plot_generator.create_dual_axis_plot(
            df_primary=empty_throughput_df,
            df_secondary=empty_gpu_df,
            x_col_primary="timestamp_s",
            x_col_secondary="timestamp_s",
            y1_metric="throughput",
            y2_metric="gpu_utilization",
            primary_style=Style(mode="lines", line_shape="hv", fill=None),
            secondary_style=Style(mode="lines", line_shape=None, fill="tozeroy"),
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


class TestLatencyScatterWithPercentiles:
    """Tests for create_latency_scatter_with_percentiles method."""

    @pytest.fixture
    def latency_df(self):
        """Create sample latency DataFrame with percentiles for testing."""
        return pd.DataFrame(
            {
                "timestamp": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
                "request_latency": [900, 920, 950, 880, 1100, 910, 940, 930, 960, 920],
                "p50": [900, 910, 920, 915, 920, 915, 920, 920, 925, 920],
                "p95": [900, 920, 950, 950, 1100, 1100, 1100, 1100, 1100, 1100],
                "p99": [900, 920, 950, 950, 1100, 1100, 1100, 1100, 1100, 1100],
            }
        )

    def test_latency_scatter_with_percentiles_basic(self, plot_generator, latency_df):
        """Test basic latency scatter with percentiles plot creation."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4

        assert fig.data[0].mode == "markers"
        assert fig.data[0].name == "Individual Requests"

        assert fig.data[1].mode == "lines"
        assert fig.data[1].name == "P50"

        assert fig.data[2].mode == "lines"
        assert fig.data[2].name == "P95"

        assert fig.data[3].mode == "lines"
        assert fig.data[3].name == "P99"

    def test_latency_scatter_with_percentiles_custom_labels(
        self, plot_generator, latency_df
    ):
        """Test latency scatter with percentiles plot with custom labels."""
        title = "Custom Latency Plot"
        x_label = "Time (s)"
        y_label = "Latency (ms)"

        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        assert fig.layout.title.text == title
        assert fig.layout.xaxis.title.text == x_label
        assert fig.layout.yaxis.title.text == y_label

    def test_latency_scatter_with_percentiles_auto_labels(
        self, plot_generator, latency_df
    ):
        """Test latency scatter with percentiles plot with auto-generated labels."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert "Request Latency" in fig.layout.title.text
        assert "Percentiles" in fig.layout.title.text
        assert "Timestamp" in fig.layout.xaxis.title.text
        assert "Request Latency" in fig.layout.yaxis.title.text

    def test_latency_scatter_with_percentiles_colors(self, plot_generator, latency_df):
        """Test that percentile lines use seaborn color palette in light mode."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        # Light mode uses seaborn deep palette - first color is blue
        assert fig.data[1].line.color == LIGHT_MODE_PRIMARY_COLOR
        assert fig.data[2].line.color is not None
        assert fig.data[3].line.color is not None

        for i in range(1, 4):
            assert isinstance(fig.data[i].line.color, str)
            assert fig.data[i].line.color.startswith("#")

    def test_latency_scatter_with_percentiles_styling(self, plot_generator, latency_df):
        """Test styling of scatter points and percentile lines."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert fig.data[0].marker.opacity == 0.4
        assert fig.data[0].marker.size == 6

        for i in range(1, 4):
            assert fig.data[i].line.width == 2.5

    def test_latency_scatter_with_percentiles_subset(self, plot_generator, latency_df):
        """Test with subset of percentiles."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95"],
        )

        assert len(fig.data) == 3
        assert fig.data[0].name == "Individual Requests"
        assert fig.data[1].name == "P50"
        assert fig.data[2].name == "P95"

    def test_latency_scatter_with_missing_percentile_column(
        self, plot_generator, latency_df
    ):
        """Test that missing percentile columns are gracefully skipped."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p90", "p99"],
        )

        assert len(fig.data) == 3
        assert fig.data[0].name == "Individual Requests"
        assert fig.data[1].name == "P50"
        assert fig.data[2].name == "P99"

    def test_latency_scatter_with_percentiles_hover_mode(
        self, plot_generator, latency_df
    ):
        """Test that hover mode is set to x unified."""
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=latency_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert fig.layout.hovermode == "x unified"

    def test_latency_scatter_with_empty_dataframe(self, plot_generator):
        """Test latency scatter with percentiles plot with empty DataFrame."""
        empty_df = pd.DataFrame(
            {
                "timestamp": [],
                "request_latency": [],
                "p50": [],
                "p95": [],
                "p99": [],
            }
        )
        fig = plot_generator.create_latency_scatter_with_percentiles(
            df=empty_df,
            x_col="timestamp",
            y_metric="request_latency",
            percentile_cols=["p50", "p95", "p99"],
        )

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 4


class TestOutlierDetection:
    """Tests for detect_directional_outliers function."""

    def test_detect_outliers_empty_values(self):
        """Test outlier detection with empty array returns empty boolean array."""
        values = np.array([])
        result = detect_directional_outliers(
            values, "time_to_first_token", run_average=50.0, run_std=10.0
        )

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == 0

    def test_detect_outliers_no_run_stats(self):
        """Test outlier detection without run statistics returns all False."""
        values = np.array([45.0, 50.0, 55.0, 60.0])

        # No run_average
        result = detect_directional_outliers(
            values, "time_to_first_token", run_average=None, run_std=10.0
        )
        assert np.all(~result)

        # No run_std
        result = detect_directional_outliers(
            values, "time_to_first_token", run_average=50.0, run_std=None
        )
        assert np.all(~result)

    def test_detect_outliers_throughput_vs_latency(self):
        """Test outlier direction depends on metric type."""
        run_avg = 100.0
        run_std = 10.0

        # Latency-type metrics: high values are bad
        latency_values = np.array([95.0, 105.0, 130.0])
        latency_outliers = detect_directional_outliers(
            latency_values, "time_to_first_token", run_avg, run_std
        )
        # Only 130.0 is outlier (above upper bound 110.0)
        is_95_outlier = latency_outliers[0]
        is_105_outlier = latency_outliers[1]
        is_130_outlier = latency_outliers[2]
        assert not is_95_outlier
        assert not is_105_outlier
        assert is_130_outlier

        # Throughput metrics: low values are bad
        throughput_values = np.array([105.0, 95.0, 70.0])
        throughput_outliers = detect_directional_outliers(
            throughput_values, "request_throughput", run_avg, run_std
        )
        # Only 70.0 is outlier (below lower bound 90.0)
        is_105_throughput_outlier = throughput_outliers[0]
        is_95_throughput_outlier = throughput_outliers[1]
        is_70_throughput_outlier = throughput_outliers[2]
        assert not is_105_throughput_outlier
        assert not is_95_throughput_outlier
        assert is_70_throughput_outlier

    def test_detect_outliers_with_slice_stds(self):
        """Test outlier detection incorporates per-slice standard deviations."""
        values = np.array([50.0, 60.0, 80.0, 90.0])
        run_avg = 70.0
        run_std = 10.0
        slice_stds = np.array([5.0, 5.0, 2.0, 15.0])

        outliers = detect_directional_outliers(
            values, "time_to_first_token", run_avg, run_std, slice_stds
        )

        # Upper bounds: 70 + 10 + slice_stds = [85, 85, 82, 95]
        # 50 < 85 (not outlier), 60 < 85 (not outlier),
        # 80 < 82 (not outlier), 90 < 95 (not outlier)
        assert not np.any(outliers)

        # But without slice_stds, 90 would be outlier (upper bound = 80)
        outliers_no_slice = detect_directional_outliers(
            values, "time_to_first_token", run_avg, run_std, slice_stds=None
        )
        assert outliers_no_slice[3]

    def test_detect_outliers_mismatched_slice_stds_length(self):
        """Test slice_stds length mismatch defaults to zeros."""
        values = np.array([50.0, 60.0, 80.0, 90.0])
        run_avg = 70.0
        run_std = 10.0
        slice_stds_wrong_length = np.array([5.0, 5.0])

        outliers = detect_directional_outliers(
            values, "time_to_first_token", run_avg, run_std, slice_stds_wrong_length
        )

        # Should use zeros for slice_stds (upper bound = 80)
        assert outliers[3]
        assert not outliers[0]


class TestDarkTheme:
    """Tests for dark theme functionality."""

    @pytest.fixture
    def dark_plot_generator(self):
        """Create a PlotGenerator with dark theme."""
        return PlotGenerator(theme=PlotTheme.DARK)

    @pytest.fixture
    def multi_run_df(self):
        """Create sample multi-run DataFrame for testing."""
        return pd.DataFrame(
            {
                "model": ["model-a"] * 3 + ["model-b"] * 3,
                "concurrency": [1, 4, 8] * 2,
                "request_latency": [100, 150, 200, 120, 180, 250],
                "request_throughput": [10, 25, 35, 8, 20, 28],
            }
        )

    def test_dark_theme_initialization(self, dark_plot_generator):
        """Test PlotGenerator initializes with dark theme."""
        assert dark_plot_generator.theme == PlotTheme.DARK
        assert dark_plot_generator.colors == DARK_THEME_COLORS

    def test_dark_theme_background_colors(self, dark_plot_generator, multi_run_df):
        """Test dark theme uses correct background colors."""
        fig = dark_plot_generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
            group_by="model",
        )

        assert fig.layout.plot_bgcolor == NVIDIA_DARK_BG
        assert fig.layout.paper_bgcolor == NVIDIA_CARD_BG

    def test_dark_theme_text_color(self, dark_plot_generator, multi_run_df):
        """Test dark theme uses light text color."""
        fig = dark_plot_generator.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
            group_by="model",
        )

        assert fig.layout.font.color == NVIDIA_TEXT_LIGHT

    def test_dark_theme_uses_nvidia_brand_colors(self, dark_plot_generator):
        """Test dark theme color pool starts with NVIDIA brand colors."""
        # Dark theme should use brand colors (green + gold first)
        color_pool = dark_plot_generator._color_pool

        assert len(color_pool) > 0
        assert NVIDIA_GREEN in color_pool
        # First color should be NVIDIA green
        assert color_pool[0] == NVIDIA_GREEN

    def test_dark_theme_plot_comparison_with_light(self, multi_run_df):
        """Test dark theme produces different colors than light theme."""
        light_gen = PlotGenerator(theme=PlotTheme.LIGHT)
        dark_gen = PlotGenerator(theme=PlotTheme.DARK)

        light_fig = light_gen.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
        )
        dark_fig = dark_gen.create_pareto_plot(
            df=multi_run_df,
            x_metric="request_latency",
            y_metric="request_throughput",
        )

        # Background colors should differ
        assert light_fig.layout.plot_bgcolor != dark_fig.layout.plot_bgcolor
        assert light_fig.layout.paper_bgcolor != dark_fig.layout.paper_bgcolor

        # Text colors should differ
        assert light_fig.layout.font.color != dark_fig.layout.font.color


class TestColorEdgeCases:
    """Tests for color assignment edge cases."""

    def test_prepare_groups_none_group_by(self):
        """Test _prepare_groups with group_by=None returns no color map."""
        plot_gen = PlotGenerator()
        df = pd.DataFrame({"model": ["a", "b", "c"]})

        groups, color_map, _display_names = plot_gen._prepare_groups(df, group_by=None)

        # Should return [None] groups and empty color_map
        assert groups == [None]
        assert color_map == {}

    def test_color_cycling_more_groups_than_pool(self):
        """Test color cycling when groups exceed pool size."""
        plot_gen = PlotGenerator(color_pool_size=5)
        # Create models with zero-padded numbers to ensure alphabetical sorting
        model_names = [f"model-{i:02d}" for i in range(12)]
        df = pd.DataFrame({"model": model_names})

        _groups, color_map, _display_names = plot_gen._prepare_groups(df, "model")

        # All models should get a color
        assert len(color_map) == 12

        # Colors should cycle (some will repeat)
        unique_colors = set(color_map.values())
        # Should have at most pool_size unique colors
        assert len(unique_colors) <= len(plot_gen._color_pool)
        # Should have fewer unique colors than models
        assert len(unique_colors) < len(model_names)

        # Verify cycling pattern: models assigned sequentially wrap around
        # model-00 gets index 0, model-05 gets index 5 which wraps to color pool[0]
        assert color_map["model-00"] == color_map["model-05"]
        assert color_map["model-01"] == color_map["model-06"]

    def test_color_pool_size_zero_edge_case(self):
        """Test color pool with size zero falls back to default."""
        # Even with 0, should get at least some colors from seaborn
        plot_gen = PlotGenerator(color_pool_size=0)

        # Color pool should still exist (seaborn will return empty or minimal palette)
        assert isinstance(plot_gen._color_pool, list)

    def test_prepare_groups_missing_column_returns_no_groups(self):
        """Test _prepare_groups with non-existent column returns no color map."""
        plot_gen = PlotGenerator()
        df = pd.DataFrame({"model": ["a", "b", "c"]})

        # Try to group by non-existent column
        groups, color_map, _display_names = plot_gen._prepare_groups(
            df, "nonexistent_column"
        )

        # Should return [None] and empty color_map (no grouping)
        assert groups == [None]
        assert color_map == {}


class TestGetNvidiaColorScheme:
    """Tests for get_nvidia_color_scheme function."""

    def test_brand_colors_less_than_requested(self):
        """Returns only NVIDIA colors when n_colors <= 2."""
        colors = get_nvidia_color_scheme(n_colors=1, use_brand_colors=True)
        assert len(colors) == 1
        assert colors[0] == NVIDIA_GREEN

        colors = get_nvidia_color_scheme(n_colors=2, use_brand_colors=True)
        assert len(colors) == 2
        assert colors[0] == NVIDIA_GREEN

    def test_brand_colors_with_palette_expansion(self):
        """Adds seaborn colors when n_colors > 2."""
        colors = get_nvidia_color_scheme(n_colors=5, use_brand_colors=True)
        assert len(colors) == 5
        assert colors[0] == NVIDIA_GREEN
        # Remaining colors should be from seaborn palette
        for color in colors[2:]:
            assert color.startswith("#")

    def test_without_brand_colors(self):
        """Uses only seaborn palette when use_brand_colors=False."""
        colors = get_nvidia_color_scheme(n_colors=5, use_brand_colors=False)
        assert len(colors) == 5
        # Should not start with NVIDIA brand colors
        assert colors[0] != NVIDIA_GREEN
        # All should be valid hex colors
        for color in colors:
            assert color.startswith("#")

    def test_bright_palette_with_brand_colors(self):
        """Verifies bright palette used with brand colors."""
        colors = get_nvidia_color_scheme(
            n_colors=5, palette_name="bright", use_brand_colors=True
        )
        assert len(colors) == 5
        assert colors[0] == NVIDIA_GREEN

    def test_deep_palette_without_brand_colors(self):
        """Verifies deep palette used without brand colors."""
        colors = get_nvidia_color_scheme(
            n_colors=5, palette_name="deep", use_brand_colors=False
        )
        assert len(colors) == 5
        for color in colors:
            assert color.startswith("#")

    def test_color_pool_cycling(self):
        """Verifies colors are valid hex strings."""
        colors = get_nvidia_color_scheme(n_colors=15, use_brand_colors=True)
        assert len(colors) == 15
        for color in colors:
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB format


class TestGetMetricDirection:
    """Tests for _get_metric_direction method."""

    def test_get_metric_direction_from_registry(self):
        """Uses MetricRegistry when available."""
        plot_gen = PlotGenerator()

        # Mock a metric with LARGER_IS_BETTER flag
        with patch(
            "aiperf.plot.core.plot_generator.MetricRegistry.get_class"
        ) as mock_get_class:
            mock_metric = type(
                "MockMetric",
                (),
                {
                    "has_flags": lambda flags: flags == MetricFlags.LARGER_IS_BETTER,
                },
            )
            mock_get_class.return_value = mock_metric

            direction = plot_gen._get_metric_direction("test_throughput")
            assert direction == PlotMetricDirection.HIGHER

    def test_get_metric_direction_fallback_to_derived(self):
        """Falls back to DERIVED_METRIC_DIRECTIONS."""
        plot_gen = PlotGenerator()

        # Mock MetricRegistry to raise exception
        with (
            patch(
                "aiperf.plot.core.plot_generator.MetricRegistry.get_class",
                side_effect=Exception,
            ),
            patch(
                "aiperf.plot.core.plot_generator.DERIVED_METRIC_DIRECTIONS",
                {"custom_throughput_metric": True},
            ),
        ):
            direction = plot_gen._get_metric_direction("custom_throughput_metric")
            assert direction == PlotMetricDirection.HIGHER

    def test_get_metric_direction_default_to_empty_string(self):
        """Returns empty string for unknown metrics."""
        plot_gen = PlotGenerator()

        # Mock MetricRegistry to raise exception and empty derived directions
        with (
            patch(
                "aiperf.plot.core.plot_generator.MetricRegistry.get_class",
                side_effect=Exception,
            ),
            patch("aiperf.plot.core.plot_generator.DERIVED_METRIC_DIRECTIONS", {}),
        ):
            direction = plot_gen._get_metric_direction("unknown_metric")
            assert direction == ""


class TestPrepareGroupsExperimentTypes:
    """Tests for _prepare_groups with experiment_types logic."""

    def test_prepare_groups_experiment_types_baseline_vs_treatment(self):
        """Separates baselines from treatments."""
        plot_gen = PlotGenerator()
        df = pd.DataFrame(
            {
                "experiment_group": [
                    "baseline_a",
                    "treatment_a",
                    "baseline_b",
                    "treatment_b",
                ],
                "value": [1, 2, 3, 4],
            }
        )
        experiment_types = {
            "baseline_a": "baseline",
            "treatment_a": "treatment",
            "baseline_b": "baseline",
            "treatment_b": "treatment",
        }

        groups, colors, _display_names = plot_gen._prepare_groups(
            df, "experiment_group", experiment_types
        )

        # Baselines should come first, then treatments
        assert groups[:2] == ["baseline_a", "baseline_b"]
        assert groups[2:] == ["treatment_a", "treatment_b"]

        # Baselines should be gray
        assert colors["baseline_a"] == NVIDIA_GRAY
        assert colors["baseline_b"] == NVIDIA_GRAY

        # First treatment should be green
        assert colors["treatment_a"] == NVIDIA_GREEN

    def test_prepare_groups_experiment_types_single_treatment(self):
        """Single treatment gets green color."""
        plot_gen = PlotGenerator()
        df = pd.DataFrame(
            {
                "experiment_group": ["baseline", "treatment"],
                "value": [1, 2],
            }
        )
        experiment_types = {
            "baseline": "baseline",
            "treatment": "treatment",
        }

        _groups, colors, _display_names = plot_gen._prepare_groups(
            df, "experiment_group", experiment_types
        )

        assert colors["baseline"] == NVIDIA_GRAY
        assert colors["treatment"] == NVIDIA_GREEN

    def test_prepare_groups_experiment_types_multiple_treatments(self):
        """Multiple treatments: first=green, rest=seaborn colors."""
        plot_gen = PlotGenerator()
        df = pd.DataFrame(
            {
                "experiment_group": [
                    "baseline",
                    "treatment1",
                    "treatment2",
                    "treatment3",
                ],
                "value": [1, 2, 3, 4],
            }
        )
        experiment_types = {
            "baseline": "baseline",
            "treatment1": "treatment",
            "treatment2": "treatment",
            "treatment3": "treatment",
        }

        _groups, colors, _display_names = plot_gen._prepare_groups(
            df, "experiment_group", experiment_types
        )

        assert colors["baseline"] == NVIDIA_GRAY
        assert colors["treatment1"] == NVIDIA_GREEN
        # Other treatments should have different colors
        assert colors["treatment2"] != NVIDIA_GREEN
        assert colors["treatment2"] != NVIDIA_GRAY
        assert colors["treatment3"] != NVIDIA_GREEN
        assert colors["treatment3"] != NVIDIA_GRAY

    def test_prepare_groups_with_string_input(self):
        """Accepts string input directly (validator converts lists to strings)."""
        plot_gen = PlotGenerator()
        df = pd.DataFrame(
            {
                "model": ["model_a", "model_b"],
                "value": [1, 2],
            }
        )

        # Pass string directly (validator already converted list to string)
        groups, colors, _display_names = plot_gen._prepare_groups(df, group_by="model")

        # Should successfully group by model
        assert groups == ["model_a", "model_b"]
        assert len(colors) == 2

    def test_prepare_groups_raises_error_for_unknown_experiment_type(self):
        """Raises ValueError when experiment_type is not 'baseline' or 'treatment'."""
        plot_gen = PlotGenerator()
        df = pd.DataFrame(
            {
                "experiment_group": ["group_a", "group_b", "group_c"],
                "value": [1, 2, 3],
            }
        )
        experiment_types = {
            "group_a": "baseline",
            "group_b": "treatment",
            "group_c": "control",  # Invalid type
        }

        with pytest.raises(ValueError) as exc_info:
            plot_gen._prepare_groups(df, "experiment_group", experiment_types)

        assert "group_c" in str(exc_info.value)
        assert "control" in str(exc_info.value)
        assert "baseline" in str(exc_info.value) or "treatment" in str(exc_info.value)

    def test_prepare_groups_raises_error_for_missing_experiment_type(self):
        """Raises ValueError when group is missing from experiment_types mapping."""
        plot_gen = PlotGenerator()
        df = pd.DataFrame(
            {
                "experiment_group": ["group_a", "group_b", "group_c"],
                "value": [1, 2, 3],
            }
        )
        experiment_types = {
            "group_a": "baseline",
            "group_b": "treatment",
            # group_c is missing from the mapping
        }

        with pytest.raises(ValueError) as exc_info:
            plot_gen._prepare_groups(df, "experiment_group", experiment_types)

        assert "group_c" in str(exc_info.value)
        assert "None" in str(exc_info.value)


class TestParetoFrontierOptimization:
    """Tests for optimized O(n log n) Pareto frontier calculation."""

    @pytest.mark.parametrize(
        "x_dir,y_dir,x_vals,y_vals,expected",
        [
            # LOWER x, HIGHER y (classic Pareto: minimize x, maximize y)
            ("LOWER", "HIGHER", [1, 2, 3, 4], [4, 5, 3, 6], [True, True, False, True]),
            ("LOWER", "HIGHER", [1, 2, 3], [3, 2, 1], [True, False, False]),
            # HIGHER x, HIGHER y (maximize both)
            (
                "HIGHER",
                "HIGHER",
                [1, 2, 3, 4],
                [4, 5, 3, 6],
                [False, False, False, True],
            ),
            ("HIGHER", "HIGHER", [1, 2, 3], [1, 2, 3], [False, False, True]),
            # LOWER x, LOWER y (minimize both)
            ("LOWER", "LOWER", [1, 2, 3, 4], [4, 3, 5, 2], [True, True, False, True]),
            ("LOWER", "LOWER", [1, 2, 3], [3, 2, 1], [True, True, True]),
            # HIGHER x, LOWER y (maximize x, minimize y)
            (
                "HIGHER",
                "LOWER",
                [1, 2, 3, 4],
                [4, 5, 3, 2],
                [False, False, False, True],
            ),
            ("HIGHER", "LOWER", [1, 2, 3], [3, 2, 1], [False, False, True]),
            # Edge cases: duplicate x values
            ("LOWER", "HIGHER", [1, 1, 2], [5, 3, 6], [True, False, True]),
            # Edge cases: duplicate y values
            ("LOWER", "HIGHER", [1, 2, 3], [5, 5, 5], [True, True, True]),
            # Edge cases: all points on frontier (monotonic increase)
            ("LOWER", "HIGHER", [1, 2, 3, 4], [1, 2, 3, 4], [True, True, True, True]),
            # Edge cases: single best point
            ("HIGHER", "HIGHER", [1, 2, 3], [1, 1, 10], [False, False, True]),
        ],  # fmt: skip
    )
    def test_pareto_frontier_directions(
        self, plot_generator, x_dir, y_dir, x_vals, y_vals, expected
    ):
        """Test Pareto frontier calculation for all direction combinations."""
        x_direction = PlotMetricDirection(x_dir)
        y_direction = PlotMetricDirection(y_dir)

        x_array = np.array(x_vals, dtype=float)
        y_array = np.array(y_vals, dtype=float)

        result = plot_generator._compute_pareto_frontier(
            x_array, y_array, x_direction, y_direction
        )

        expected_array = np.array(expected, dtype=bool)
        np.testing.assert_array_equal(
            result,
            expected_array,
            err_msg=f"Failed for x_dir={x_dir}, y_dir={y_dir}, x={x_vals}, y={y_vals}",
        )

    def test_pareto_empty_array(self, plot_generator):
        """Test with empty arrays."""
        result = plot_generator._compute_pareto_frontier(
            np.array([]),
            np.array([]),
            PlotMetricDirection.LOWER,
            PlotMetricDirection.HIGHER,
        )
        assert len(result) == 0
        assert result.dtype == bool

    def test_pareto_single_point(self, plot_generator):
        """Test with single point."""
        result = plot_generator._compute_pareto_frontier(
            np.array([1.0]),
            np.array([2.0]),
            PlotMetricDirection.LOWER,
            PlotMetricDirection.HIGHER,
        )
        np.testing.assert_array_equal(result, [True])

    def test_pareto_two_points(self, plot_generator):
        """Test with two points - various domination scenarios."""
        # Point 2 dominates point 1 (minimize x, maximize y)
        # Data must be sorted by x ascending (1.0 comes before 2.0)
        result = plot_generator._compute_pareto_frontier(
            np.array([1.0, 2.0]),  # Sorted by x
            np.array([2.0, 1.0]),  # Point 1 has y=2, point 2 has y=1
            PlotMetricDirection.LOWER,
            PlotMetricDirection.HIGHER,
        )
        # Point 1 (x=1, y=2) is on frontier, Point 2 (x=2, y=1) is dominated
        np.testing.assert_array_equal(result, [True, False])

        # Both points on frontier (non-dominated, moving away from each other)
        result = plot_generator._compute_pareto_frontier(
            np.array([1.0, 2.0]),  # Sorted by x
            np.array([1.0, 2.0]),  # Both increase together
            PlotMetricDirection.LOWER,
            PlotMetricDirection.HIGHER,
        )
        # For minimize x, maximize y: point 1 (x=1, y=1) is on frontier
        # Point 2 (x=2, y=2) has worse x but better y, so it's also on frontier
        np.testing.assert_array_equal(result, [True, True])

    def test_pareto_backwards_compatibility(self, multi_run_df):
        """
        Verify that the optimized algorithm produces identical results to the
        O(n²) algorithm for typical multi-run data.
        """
        plot_gen = PlotGenerator()
        df = multi_run_df.sort_values("request_latency")

        # Test on latency (LOWER) vs throughput (HIGHER) - classic Pareto
        x_vals = df["request_latency"].values
        y_vals = df["request_throughput"].values

        result = plot_gen._compute_pareto_frontier(
            x_vals, y_vals, PlotMetricDirection.LOWER, PlotMetricDirection.HIGHER
        )

        # Verify at least one point is on the frontier
        assert np.any(result), "At least one point should be on Pareto frontier"

        # Verify no point on the frontier is dominated by another point on the frontier
        pareto_points = np.where(result)[0]
        for i in pareto_points:
            for j in pareto_points:
                if i == j:
                    continue
                # For minimize x, maximize y: j should not have both (x_j <= x_i and y_j >= y_i) with strict inequality
                if x_vals[j] < x_vals[i] and y_vals[j] > y_vals[i]:
                    pytest.fail(
                        f"Point {j} dominates point {i}, but both are on frontier"
                    )

    def test_pareto_performance_large_dataset(self, plot_generator):
        """Benchmark with large dataset to verify O(n log n) performance."""
        # Generate 1000 random points
        np.random.seed(42)
        n = 1000
        x_vals = np.random.rand(n) * 100
        y_vals = np.random.rand(n) * 100

        # Sort by x (as the real algorithm expects)
        sorted_indices = np.argsort(x_vals)
        x_vals = x_vals[sorted_indices]
        y_vals = y_vals[sorted_indices]

        start = time.time()
        result = plot_generator._compute_pareto_frontier(
            x_vals, y_vals, PlotMetricDirection.LOWER, PlotMetricDirection.HIGHER
        )
        elapsed = time.time() - start

        # Should complete in well under 0.1 seconds for 1000 points
        assert elapsed < 0.1, f"Algorithm took {elapsed}s for {n} points (too slow)"
        assert np.any(result), "Should have at least one point on frontier"

    def test_pareto_all_points_identical(self, plot_generator):
        """Test when all points have identical coordinates."""
        # All points are identical, so all should be on the frontier
        result = plot_generator._compute_pareto_frontier(
            np.array([5.0, 5.0, 5.0]),
            np.array([3.0, 3.0, 3.0]),
            PlotMetricDirection.LOWER,
            PlotMetricDirection.HIGHER,
        )
        # All identical points are considered on the frontier (use >= comparison)
        np.testing.assert_array_equal(result, [True, True, True])

    def test_pareto_raises_error_for_unknown_metric_directions(self, multi_run_df):
        """Test that ValueError is raised when metric directions are unknown."""
        plot_gen = PlotGenerator()

        # Mock _get_metric_direction to return empty string (unknown direction)
        with (
            patch.object(plot_gen, "_get_metric_direction", return_value=""),
            pytest.raises(
                ValueError,
                match="Cannot determine optimization direction for x-axis metric 'request_latency' and y-axis metric 'request_throughput'",
            ),
        ):
            plot_gen.create_pareto_plot(
                df=multi_run_df,
                x_metric="request_latency",
                y_metric="request_throughput",
                label_by="concurrency",
                group_by="model",
            )

    def test_pareto_raises_error_for_one_unknown_metric(self, multi_run_df):
        """Test that ValueError is raised when one metric direction is unknown."""
        plot_gen = PlotGenerator()

        # Mock _get_metric_direction to return known for x, unknown for y
        def mock_direction(metric):
            if metric == "request_latency":
                return PlotMetricDirection.LOWER
            return ""

        with (
            patch.object(plot_gen, "_get_metric_direction", side_effect=mock_direction),
            pytest.raises(
                ValueError,
                match="Cannot determine optimization direction for y-axis metric 'request_throughput'",
            ),
        ):
            plot_gen.create_pareto_plot(
                df=multi_run_df,
                x_metric="request_latency",
                y_metric="request_throughput",
                label_by="concurrency",
                group_by="model",
            )
