# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for dashboard utility functions.

Tests for pure functions in aiperf.plot.dashboard.utils module.
"""

import math
from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go
import pytest
from dash import html

from aiperf.plot.constants import PlotTheme
from aiperf.plot.dashboard.utils import (
    _convert_to_numeric,
    add_run_idx_to_figure,
    create_plot_container_component,
    extract_metric_value,
    get_available_stats_for_metric,
    get_plot_title,
    get_single_run_metrics_with_stats,
    get_stat_options_for_single_run_metric,
    prepare_timeseries_dataframe,
    resolve_single_run_column_name,
    runs_to_dataframe,
)


class TestExtractMetricValue:
    """Tests for extract_metric_value function."""

    def test_extract_from_metric_result_object(self):
        """Test extracting value from MetricResult-like object with stats attribute."""
        mock_metric = MagicMock()
        mock_metric.stats.p50 = 42.5
        mock_metric.stats.avg = 45.0

        mock_run = MagicMock()
        mock_run.get_metric.return_value = mock_metric

        result = extract_metric_value(mock_run, "time_to_first_token", "p50")
        assert result == 42.5

        result = extract_metric_value(mock_run, "time_to_first_token", "avg")
        assert result == 45.0

    def test_extract_from_dict_metric(self):
        """Test extracting value from dict-format metric."""
        mock_run = MagicMock()
        mock_run.get_metric.return_value = {"p50": 100.0, "avg": 110.0, "p99": 200.0}

        result = extract_metric_value(mock_run, "request_latency", "p50")
        assert result == 100.0

        result = extract_metric_value(mock_run, "request_latency", "p99")
        assert result == 200.0

    def test_extract_returns_none_for_missing_metric(self):
        """Test that None is returned when metric doesn't exist."""
        mock_run = MagicMock()
        mock_run.get_metric.return_value = None

        result = extract_metric_value(mock_run, "nonexistent_metric", "p50")
        assert result is None

    def test_extract_returns_none_for_missing_stat(self):
        """Test that None is returned when stat doesn't exist on metric."""
        mock_metric = MagicMock(spec=["stats"])
        mock_metric.stats = MagicMock(spec=["p50"])
        mock_metric.stats.p50 = 42.5

        mock_run = MagicMock()
        mock_run.get_metric.return_value = mock_metric

        result = extract_metric_value(mock_run, "time_to_first_token", "p99")
        assert result is None

    def test_extract_from_dict_missing_stat(self):
        """Test extracting missing stat from dict returns None."""
        mock_run = MagicMock()
        mock_run.get_metric.return_value = {"p50": 100.0}

        result = extract_metric_value(mock_run, "metric", "p99")
        assert result is None

    def test_default_stat_is_p50(self):
        """Test that default stat is p50."""
        mock_run = MagicMock()
        mock_run.get_metric.return_value = {"p50": 50.0, "avg": 55.0}

        result = extract_metric_value(mock_run, "metric")
        assert result == 50.0


class TestCreatePlotContainerComponent:
    """Tests for create_plot_container_component function."""

    @pytest.fixture
    def sample_figure(self):
        """Create a sample Plotly figure for testing."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
        return fig

    def test_returns_html_div(self, sample_figure):
        """Test that function returns an html.Div component."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
        )
        assert isinstance(result, html.Div)

    def test_container_has_correct_id(self, sample_figure):
        """Test that container has correct pattern-matching ID."""
        result = create_plot_container_component(
            plot_id="my-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
        )
        assert result.id == {"type": "plot-container", "index": "my-plot"}

    def test_container_has_correct_class_for_half_size(self, sample_figure):
        """Test that half-size container has correct class."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
            size_class="half",
        )
        assert "size-half" in result.className

    def test_container_has_correct_class_for_full_size(self, sample_figure):
        """Test that full-size container has correct class."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
            size_class="full",
        )
        assert "size-full" in result.className

    def test_container_height_doubles_for_full_size(self, sample_figure):
        """Test that full-size container has doubled height."""
        half_result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
            size=400,
            size_class="half",
        )
        full_result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
            size=400,
            size_class="full",
        )
        assert "400px" in half_result.style["min-height"]
        assert "800px" in full_result.style["min-height"]

    def test_container_has_settings_button(self, sample_figure):
        """Test that container includes settings button."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
        )
        children_ids = [
            c.id for c in result.children if hasattr(c, "id") and isinstance(c.id, dict)
        ]
        settings_ids = [i for i in children_ids if i.get("type") == "settings-plot-btn"]
        assert len(settings_ids) == 1
        assert settings_ids[0]["index"] == "test-plot"

    def test_container_has_hide_button(self, sample_figure):
        """Test that container includes hide button."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
        )
        children_ids = [
            c.id for c in result.children if hasattr(c, "id") and isinstance(c.id, dict)
        ]
        hide_ids = [i for i in children_ids if i.get("type") == "hide-plot-btn-direct"]
        assert len(hide_ids) == 1

    def test_container_has_resize_handle_when_resizable(self, sample_figure):
        """Test that resize handle is included when resizable=True."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
            resizable=True,
        )
        children_ids = [
            c.id for c in result.children if hasattr(c, "id") and isinstance(c.id, dict)
        ]
        resize_ids = [i for i in children_ids if i.get("type") == "resize-handle"]
        assert len(resize_ids) == 1

    def test_container_no_resize_handle_when_not_resizable(self, sample_figure):
        """Test that resize handle is excluded when resizable=False."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
            resizable=False,
        )
        children_ids = [
            c.id for c in result.children if hasattr(c, "id") and isinstance(c.id, dict)
        ]
        resize_ids = [i for i in children_ids if i.get("type") == "resize-handle"]
        assert len(resize_ids) == 0

    def test_container_visible_by_default(self, sample_figure):
        """Test that container is visible by default."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
        )
        assert result.style["display"] == "block"

    def test_container_hidden_when_visible_false(self, sample_figure):
        """Test that container is hidden when visible=False."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=PlotTheme.DARK,
            visible=False,
        )
        assert result.style["display"] == "none"

    @pytest.mark.parametrize("theme", [PlotTheme.DARK, PlotTheme.LIGHT])
    def test_container_has_theme_appropriate_background(self, sample_figure, theme):
        """Test that container background matches theme."""
        result = create_plot_container_component(
            plot_id="test-plot",
            figure=sample_figure,
            theme=theme,
        )
        assert "background" in result.style


class TestGetPlotTitle:
    """Tests for get_plot_title function."""

    def test_returns_title_from_config(self):
        """Test that title is returned from plot_configs when available."""
        plot_configs = {
            "pareto": {"title": "Pareto Analysis", "x_metric": "throughput"},
            "latency": {"title": "Latency Distribution"},
        }
        result = get_plot_title("pareto", plot_configs)
        assert result == "Pareto Analysis"

    def test_returns_formatted_fallback_when_no_config(self):
        """Test that plot_id is formatted when not in config."""
        result = get_plot_title("my-custom-plot", None)
        assert result == "My Custom Plot"

    def test_returns_formatted_fallback_when_not_in_config(self):
        """Test that plot_id is formatted when config exists but plot not in it."""
        plot_configs = {"other": {"title": "Other Plot"}}
        result = get_plot_title("my-custom-plot", plot_configs)
        assert result == "My Custom Plot"

    def test_handles_hyphens_in_plot_id(self):
        """Test that hyphens are converted to spaces in fallback."""
        result = get_plot_title("time-to-first-token", None)
        assert result == "Time To First Token"


class TestConvertToNumeric:
    """Tests for _convert_to_numeric function."""

    def test_returns_none_for_none_input(self):
        """Test that None input returns None."""
        assert _convert_to_numeric(None) is None

    def test_returns_int_unchanged(self):
        """Test that int values are returned unchanged."""
        assert _convert_to_numeric(42) == 42
        assert isinstance(_convert_to_numeric(42), int)

    def test_returns_float_unchanged(self):
        """Test that float values are returned unchanged."""
        assert _convert_to_numeric(3.14) == 3.14

    def test_handles_nan(self):
        """Test that NaN is preserved."""
        result = _convert_to_numeric(float("nan"))
        assert math.isnan(result)

    def test_handles_inf(self):
        """Test that Inf is preserved."""
        result = _convert_to_numeric(float("inf"))
        assert math.isinf(result)

    def test_converts_string_int(self):
        """Test that string integers are converted to int."""
        result = _convert_to_numeric("42")
        assert result == 42
        assert isinstance(result, int)

    def test_converts_string_float(self):
        """Test that string floats are converted to float."""
        result = _convert_to_numeric("3.14")
        assert result == 3.14
        assert isinstance(result, float)

    def test_returns_none_for_empty_string(self):
        """Test that empty string returns None."""
        assert _convert_to_numeric("") is None
        assert _convert_to_numeric("   ") is None

    def test_returns_none_for_non_numeric_string(self):
        """Test that non-numeric strings return None."""
        assert _convert_to_numeric("abc") is None
        assert _convert_to_numeric("not a number") is None

    def test_strips_whitespace_from_string(self):
        """Test that whitespace is stripped from string input."""
        assert _convert_to_numeric("  42  ") == 42
        assert _convert_to_numeric("  3.14  ") == 3.14


class TestRunsToDataframe:
    """Tests for runs_to_dataframe function."""

    @pytest.fixture
    def mock_runs(self):
        """Create mock runs with metrics."""
        runs = []
        for i in range(3):
            run = MagicMock()
            run.metadata.model = f"model-{i}"
            run.metadata.concurrency = i + 1
            run.metadata.run_name = f"run_{i}"
            run.metadata.experiment_type = "baseline"
            run.metadata.experiment_group = "group_a"

            def make_get_metric(x_val, y_val):
                def get_metric(name):
                    if name == "throughput":
                        return {"p50": x_val, "avg": x_val + 10}
                    elif name == "latency":
                        return {"p50": y_val, "avg": y_val + 5}
                    return None

                return get_metric

            run.get_metric = make_get_metric(100 + i * 10, 50 + i * 5)
            runs.append(run)
        return runs

    def test_creates_dataframe_with_metrics(self, mock_runs):
        """Test that DataFrame is created with correct columns."""
        result = runs_to_dataframe(
            mock_runs,
            x_metric="throughput",
            x_stat="p50",
            y_metric="latency",
            y_stat="p50",
        )
        df = result["df"]
        assert len(df) == 3
        assert "throughput" in df.columns
        assert "latency" in df.columns
        assert "model" in df.columns
        assert "concurrency" in df.columns
        assert "run_idx" in df.columns
        assert "run_name" in df.columns

    def test_extracts_correct_stat_values(self, mock_runs):
        """Test that correct stat values are extracted."""
        result = runs_to_dataframe(
            mock_runs,
            x_metric="throughput",
            x_stat="p50",
            y_metric="latency",
            y_stat="p50",
        )
        df = result["df"]
        assert df.iloc[0]["throughput"] == 100
        assert df.iloc[0]["latency"] == 50

    def test_excludes_runs_with_missing_metrics(self):
        """Test that runs missing metrics are excluded."""
        runs = []
        for i in range(3):
            run = MagicMock()
            run.metadata.model = f"model-{i}"
            run.metadata.concurrency = i + 1
            run.metadata.run_name = f"run_{i}"
            run.metadata.experiment_type = "baseline"
            run.metadata.experiment_group = "group_a"

            if i == 1:  # Make middle run missing y metric
                run.get_metric = (
                    lambda name: {"p50": 100} if name == "throughput" else None
                )
            else:
                run.get_metric = lambda name: {"p50": 100}

            runs.append(run)

        result = runs_to_dataframe(
            runs,
            x_metric="throughput",
            x_stat="p50",
            y_metric="latency",
            y_stat="p50",
        )
        df = result["df"]
        assert len(df) == 2
        assert len(result["warnings"]) > 0

    def test_returns_empty_dataframe_when_no_valid_runs(self):
        """Test that empty DataFrame is returned when all runs are excluded."""
        run = MagicMock()
        run.metadata.model = "model"
        run.metadata.concurrency = 1
        run.metadata.run_name = "run"
        run.metadata.experiment_type = None
        run.metadata.experiment_group = None
        run.get_metric = lambda name: None

        result = runs_to_dataframe(
            [run],
            x_metric="throughput",
            x_stat="p50",
            y_metric="latency",
            y_stat="p50",
        )
        assert result["df"].empty

    def test_returns_actual_stats_used(self, mock_runs):
        """Test that actual stats used are returned."""
        result = runs_to_dataframe(
            mock_runs,
            x_metric="throughput",
            x_stat="avg",
            y_metric="latency",
            y_stat="p50",
        )
        assert result["x_stat_actual"] == "avg"
        assert result["y_stat_actual"] == "p50"


class TestGetAvailableStatsForMetric:
    """Tests for get_available_stats_for_metric function."""

    def test_returns_all_stats_for_empty_runs(self):
        """Test that all stats are returned when runs list is empty."""
        result = get_available_stats_for_metric([], "any_metric")
        # Should return ALL_STAT_KEYS
        assert "avg" in result
        assert "p50" in result

    def test_returns_value_for_concurrency_metric(self):
        """Test that 'value' is returned for concurrency metric."""
        runs = [MagicMock()]
        result = get_available_stats_for_metric(runs, "concurrency")
        assert result == ["value"]

    def test_extracts_stats_from_dict_metric(self):
        """Test that available stats are extracted from dict metric."""
        run = MagicMock()
        run.get_metric = lambda name: {"avg": 100, "p50": 95, "p99": 150, "unit": "ms"}
        result = get_available_stats_for_metric([run], "latency")
        assert "avg" in result
        assert "p50" in result
        assert "p99" in result
        assert "unit" not in result

    def test_extracts_stats_from_metric_result_object(self):
        """Test that available stats are extracted from MetricResult object."""
        metric = MagicMock()
        metric.avg = 100
        metric.p50 = 95
        metric.p99 = None  # Not available

        run = MagicMock()
        run.get_metric = lambda name: metric

        result = get_available_stats_for_metric([run], "latency")
        assert "avg" in result
        assert "p50" in result
        assert "p99" not in result

    def test_handles_derived_throughput_metric(self):
        """Test handling of output_token_throughput_per_user derived metric."""
        base_metric = MagicMock()
        base_metric.avg = 100
        base_metric.p50 = 95

        run = MagicMock()

        def get_metric(name):
            if name == "output_token_throughput":
                return base_metric
            return None

        run.get_metric = get_metric

        result = get_available_stats_for_metric(
            [run], "output_token_throughput_per_user"
        )
        assert "avg" in result


class TestPrepareTimeseriesDataframe:
    """Tests for prepare_timeseries_dataframe function."""

    def test_returns_df_unchanged_if_request_number_exists(self):
        """Test that DataFrame is returned unchanged if request_number column exists."""
        df = pd.DataFrame({"request_number": [1, 2, 3], "latency": [100, 110, 105]})
        result_df, x_col = prepare_timeseries_dataframe(df)
        assert x_col == "request_number"
        assert list(result_df["request_number"]) == [1, 2, 3]

    def test_creates_request_number_from_index(self):
        """Test that request_number is created from RangeIndex."""
        df = pd.DataFrame({"latency": [100, 110, 105]})
        result_df, x_col = prepare_timeseries_dataframe(df)
        assert x_col == "request_number"
        assert "request_number" in result_df.columns
        assert list(result_df["request_number"]) == [0, 1, 2]

    def test_uses_named_index_as_column(self):
        """Test that named index is converted to column."""
        df = pd.DataFrame({"latency": [100, 110, 105]})
        df.index = pd.Index([10, 20, 30], name="time")
        result_df, x_col = prepare_timeseries_dataframe(df)
        assert x_col == "time"
        assert "time" in result_df.columns


class TestGetSingleRunMetricsWithStats:
    """Tests for get_single_run_metrics_with_stats function."""

    def test_groups_compound_metrics_by_base_name(self):
        """Test that compound metrics are grouped by base name."""
        columns = [
            "inter_chunk_latency_avg",
            "inter_chunk_latency_p50",
            "inter_chunk_latency_p95",
            "request_latency",
        ]
        excluded = []

        options, metric_stats = get_single_run_metrics_with_stats(columns, excluded)

        # Should have inter_chunk_latency as compound metric
        assert "inter_chunk_latency" in metric_stats
        assert "avg" in metric_stats["inter_chunk_latency"]
        assert "p50" in metric_stats["inter_chunk_latency"]
        assert "p95" in metric_stats["inter_chunk_latency"]

        # request_latency is simple metric
        assert "request_latency" in metric_stats
        assert metric_stats["request_latency"] == ["avg"]

    def test_excludes_specified_columns(self):
        """Test that excluded columns are not included."""
        columns = ["latency", "timestamp", "session_id"]
        excluded = ["timestamp", "session_id"]

        options, metric_stats = get_single_run_metrics_with_stats(columns, excluded)

        assert "latency" in metric_stats
        assert "timestamp" not in metric_stats
        assert "session_id" not in metric_stats

    def test_returns_formatted_labels(self):
        """Test that option labels are properly formatted."""
        columns = ["time_to_first_token", "request_latency"]
        excluded = []

        options, _ = get_single_run_metrics_with_stats(columns, excluded)

        labels = [opt["label"] for opt in options]
        assert "Time to First Token" in labels
        assert "Request Latency" in labels


class TestGetStatOptionsForSingleRunMetric:
    """Tests for get_stat_options_for_single_run_metric function."""

    def test_returns_options_for_compound_metric(self):
        """Test that stat options are returned for compound metrics."""
        metric_stats = {"latency": ["avg", "p50", "p95"]}
        options = get_stat_options_for_single_run_metric("latency", metric_stats)

        values = [opt["value"] for opt in options]
        assert "avg" in values
        assert "p50" in values
        assert "p95" in values

    def test_returns_avg_for_simple_metric(self):
        """Test that only avg is returned for simple metrics."""
        metric_stats = {"throughput": ["avg"]}
        options = get_stat_options_for_single_run_metric("throughput", metric_stats)

        assert len(options) == 1
        assert options[0]["value"] == "avg"

    def test_returns_default_avg_for_unknown_metric(self):
        """Test that avg is returned for unknown metrics."""
        options = get_stat_options_for_single_run_metric("unknown", {})
        assert len(options) == 1
        assert options[0]["value"] == "avg"

    def test_options_have_correct_labels(self):
        """Test that stat options have correct labels."""
        metric_stats = {"latency": ["avg", "p50", "std"]}
        options = get_stat_options_for_single_run_metric("latency", metric_stats)

        labels_map = {opt["value"]: opt["label"] for opt in options}
        assert labels_map["avg"] == "Average"
        assert labels_map["p50"] == "p50 (Median)"
        assert labels_map["std"] == "Std Dev"


class TestResolveSingleRunColumnName:
    """Tests for resolve_single_run_column_name function."""

    def test_returns_metric_name_for_simple_metric(self):
        """Test that metric name is returned for simple metrics."""
        metric_stats = {"throughput": ["avg"]}
        result = resolve_single_run_column_name("throughput", "avg", metric_stats)
        assert result == "throughput"

    def test_returns_compound_name_for_compound_metric(self):
        """Test that compound name is returned for compound metrics."""
        metric_stats = {"latency": ["avg", "p50", "p95"]}
        result = resolve_single_run_column_name("latency", "p50", metric_stats)
        assert result == "latency_p50"

    def test_returns_first_stat_as_fallback(self):
        """Test that first available stat is used as fallback."""
        metric_stats = {"latency": ["p50", "p95"]}
        result = resolve_single_run_column_name("latency", None, metric_stats)
        assert result == "latency_p50"

    def test_returns_metric_name_for_unknown_metric(self):
        """Test that metric name is returned for unknown metrics."""
        result = resolve_single_run_column_name("unknown", "avg", {})
        assert result == "unknown"


class TestAddRunIdxToFigure:
    """Tests for add_run_idx_to_figure function."""

    def test_returns_figure_unchanged_if_no_run_idx(self):
        """Test that figure is returned unchanged if df has no run_idx."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
        df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})

        result = add_run_idx_to_figure(fig, df)
        assert result is fig

    def test_returns_figure_unchanged_if_df_empty(self):
        """Test that figure is returned unchanged if df is empty."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]))
        df = pd.DataFrame(columns=["run_idx", "x", "y"])

        result = add_run_idx_to_figure(fig, df)
        assert result is fig

    def test_adds_run_idx_to_customdata(self):
        """Test that run_idx is added to trace customdata."""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1.0, 2.0], y=[10.0, 20.0]))

        df = pd.DataFrame(
            {"x_metric": [1.0, 2.0], "y_metric": [10.0, 20.0], "run_idx": [0, 1]}
        )

        result = add_run_idx_to_figure(fig, df)

        # Check that customdata was added
        assert result.data[0].customdata is not None
        assert len(result.data[0].customdata) == 2
        assert result.data[0].customdata[0]["run_idx"] == 0
        assert result.data[0].customdata[1]["run_idx"] == 1

    def test_skips_traces_without_xy_data(self):
        """Test that traces without x/y data are skipped."""
        fig = go.Figure()
        # Add a trace that doesn't have x/y (like annotations)
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1)
        fig.add_trace(go.Scatter(x=[1.0], y=[10.0]))

        df = pd.DataFrame({"x_metric": [1.0], "y_metric": [10.0], "run_idx": [0]})

        result = add_run_idx_to_figure(fig, df)
        # Should not raise an error
        assert result is not None
