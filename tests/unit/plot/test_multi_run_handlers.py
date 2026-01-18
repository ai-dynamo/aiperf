# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for multi-run plot type handlers.

This module tests the handler classes that create comparison plots from multiple
profiling runs, including Pareto curves and scatter line plots.
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from aiperf.plot.constants import DEFAULT_PERCENTILE
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import DataSource, MetricSpec, PlotSpec, PlotType
from aiperf.plot.handlers.multi_run_handlers import (
    BaseMultiRunHandler,
    ParetoHandler,
    ScatterLineHandler,
)


@pytest.fixture
def sample_multi_run_dataframe():
    """
    Create a sample DataFrame with multi-run data for testing.

    Returns:
        DataFrame with experiment_group, experiment_type, group_display_name columns
    """
    return pd.DataFrame(
        {
            "run_name": ["run1", "run2", "run3"],
            "experiment_group": ["baseline", "treatment_a", "treatment_b"],
            "experiment_type": ["baseline", "treatment", "treatment"],
            "group_display_name": [
                "Baseline Model",
                "Treatment A",
                "Treatment B",
            ],
            "concurrency": [1, 2, 4],
            "request_latency": [100.0, 150.0, 200.0],
            "request_throughput": [10.0, 15.0, 20.0],
        }
    )


@pytest.fixture
def mock_plot_generator():
    """
    Create a mock PlotGenerator for testing handlers.

    Returns:
        MagicMock of PlotGenerator with create_pareto_plot and create_scatter_line_plot
    """
    generator = MagicMock(spec=PlotGenerator)
    generator.create_pareto_plot.return_value = MagicMock()
    generator.create_scatter_line_plot.return_value = MagicMock()
    return generator


@pytest.fixture
def sample_available_metrics():
    """
    Create a sample available_metrics dictionary for testing.

    Returns:
        Dict with metric display_names and units
    """
    return {
        "request_latency": {
            "display_name": "Request Latency",
            "unit": "ms",
        },
        "request_throughput": {
            "display_name": "Request Throughput",
            "unit": "req/s",
        },
        "time_to_first_token": {
            "display_name": "Time to First Token",
            "unit": "ms",
        },
    }


class TestBaseMultiRunHandler:
    """Tests for BaseMultiRunHandler base class functionality."""

    def test_get_metric_label_with_available_metrics(
        self, mock_plot_generator, sample_available_metrics
    ):
        """Test label formatting with display_name and unit from available_metrics."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        label = handler._get_metric_label(
            "request_latency", "p50", sample_available_metrics
        )
        assert label == "Request Latency (p50) (ms)"

    def test_get_metric_label_with_stat_filtering(
        self, mock_plot_generator, sample_available_metrics
    ):
        """Verify 'avg' and 'value' stats are filtered out from label."""
        handler = BaseMultiRunHandler(mock_plot_generator)

        label_avg = handler._get_metric_label(
            "request_latency", "avg", sample_available_metrics
        )
        assert label_avg == "Request Latency (ms)"

        label_value = handler._get_metric_label(
            "request_latency", "value", sample_available_metrics
        )
        assert label_value == "Request Latency (ms)"

    def test_get_metric_label_without_available_metrics(self, mock_plot_generator):
        """Fallback to metric_name when not in available_metrics."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        label = handler._get_metric_label("unknown_metric", "p50", {})
        assert label == "unknown_metric"

    def test_get_metric_label_with_stat_inclusion(
        self, mock_plot_generator, sample_available_metrics
    ):
        """Stats like 'p50', 'p99' are included in label."""
        handler = BaseMultiRunHandler(mock_plot_generator)

        label_p50 = handler._get_metric_label(
            "request_latency", "p50", sample_available_metrics
        )
        assert label_p50 == "Request Latency (p50) (ms)"

        label_p99 = handler._get_metric_label(
            "request_latency", "p99", sample_available_metrics
        )
        assert label_p99 == "Request Latency (p99) (ms)"

    def test_get_metric_label_without_unit(self, mock_plot_generator):
        """Test label formatting when metric has no unit."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        available_metrics = {
            "request_latency": {
                "display_name": "Request Latency",
            }
        }
        label = handler._get_metric_label("request_latency", "p50", available_metrics)
        assert label == "Request Latency (p50)"

    def test_extract_experiment_types_with_valid_data(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Extract experiment types from DataFrame."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        experiment_types = handler._extract_experiment_types(
            sample_multi_run_dataframe, "experiment_group"
        )

        assert experiment_types is not None
        assert experiment_types["baseline"] == "baseline"
        assert experiment_types["treatment_a"] == "treatment"
        assert experiment_types["treatment_b"] == "treatment"

    def test_extract_experiment_types_without_group_by(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Returns None when no group_by specified."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        experiment_types = handler._extract_experiment_types(
            sample_multi_run_dataframe, None
        )
        assert experiment_types is None

    def test_extract_experiment_types_missing_group_by_column(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Returns None when group_by column not in DataFrame."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        experiment_types = handler._extract_experiment_types(
            sample_multi_run_dataframe, "nonexistent_column"
        )
        assert experiment_types is None

    def test_extract_experiment_types_without_experiment_type_column(
        self, mock_plot_generator
    ):
        """Returns None when experiment_type column missing."""
        df = pd.DataFrame(
            {
                "experiment_group": ["baseline", "treatment_a"],
                "request_latency": [100.0, 150.0],
            }
        )
        handler = BaseMultiRunHandler(mock_plot_generator)
        experiment_types = handler._extract_experiment_types(df, "experiment_group")
        assert experiment_types is None

    def test_extract_experiment_types_deduplication(self, mock_plot_generator):
        """Takes first value when group has multiple experiment types."""
        df = pd.DataFrame(
            {
                "experiment_group": ["baseline", "baseline"],
                "experiment_type": ["baseline", "treatment"],
            }
        )
        handler = BaseMultiRunHandler(mock_plot_generator)
        experiment_types = handler._extract_experiment_types(df, "experiment_group")

        assert experiment_types is not None
        assert experiment_types["baseline"] == "baseline"

    def test_extract_group_display_names_with_valid_data(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Extract display names from DataFrame."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        display_names = handler._extract_group_display_names(
            sample_multi_run_dataframe, "experiment_group"
        )

        assert display_names is not None
        assert display_names["baseline"] == "Baseline Model"
        assert display_names["treatment_a"] == "Treatment A"
        assert display_names["treatment_b"] == "Treatment B"

    def test_extract_group_display_names_without_group_by(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Returns None when no group_by specified."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        display_names = handler._extract_group_display_names(
            sample_multi_run_dataframe, None
        )
        assert display_names is None

    def test_extract_group_display_names_missing_group_by_column(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Returns None when group_by column not in DataFrame."""
        handler = BaseMultiRunHandler(mock_plot_generator)
        display_names = handler._extract_group_display_names(
            sample_multi_run_dataframe, "nonexistent_column"
        )
        assert display_names is None

    def test_extract_group_display_names_without_column(self, mock_plot_generator):
        """Returns None when group_display_name column missing."""
        df = pd.DataFrame(
            {
                "experiment_group": ["baseline", "treatment_a"],
                "request_latency": [100.0, 150.0],
            }
        )
        handler = BaseMultiRunHandler(mock_plot_generator)
        display_names = handler._extract_group_display_names(df, "experiment_group")
        assert display_names is None

    def test_extract_group_display_names_returns_none_for_non_experiment_group(
        self, mock_plot_generator
    ):
        """Returns None when group_by is not 'experiment_group'.

        This is the key fix for the bug where legend showed run_name instead of
        model name. When grouping by 'model', the group_display_name column
        (which is derived from experiment_group/run_name) should NOT be used.
        Instead, the actual model value should be used as the legend label.
        """
        # Simulate a DataFrame with multiple runs of the same model
        # group_display_name is set to run_name (derived from experiment_group)
        df = pd.DataFrame(
            {
                "run_name": ["model_concurrency1", "model_concurrency50"],
                "model": [
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                ],
                "experiment_group": ["model_concurrency1", "model_concurrency50"],
                "group_display_name": ["model_concurrency1", "model_concurrency50"],
                "concurrency": [1, 50],
                "request_latency": [100.0, 150.0],
            }
        )
        handler = BaseMultiRunHandler(mock_plot_generator)

        # When grouping by 'model', should return None so the actual model
        # value is used as the legend label (not the run_name)
        display_names = handler._extract_group_display_names(df, "model")
        assert display_names is None

    def test_extract_group_display_names_returns_names_for_experiment_group(
        self, mock_plot_generator
    ):
        """Returns display names when group_by is 'experiment_group'.

        When explicitly grouping by experiment_group, the group_display_name
        column should be used for legend labels.
        """
        df = pd.DataFrame(
            {
                "run_name": ["model_concurrency1", "model_concurrency50"],
                "model": [
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                    "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
                ],
                "experiment_group": ["baseline", "treatment"],
                "group_display_name": ["Baseline Run", "Treatment Run"],
                "concurrency": [1, 50],
                "request_latency": [100.0, 150.0],
            }
        )
        handler = BaseMultiRunHandler(mock_plot_generator)

        # When grouping by 'experiment_group', should return display names
        display_names = handler._extract_group_display_names(df, "experiment_group")
        assert display_names is not None
        assert display_names["baseline"] == "Baseline Run"
        assert display_names["treatment"] == "Treatment Run"


class TestParetoHandler:
    """Tests for ParetoHandler class."""

    def test_can_handle_with_all_columns_present(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Returns True when all metrics available in DataFrame."""
        handler = ParetoHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_pareto",
            title="Test Pareto",
            plot_type=PlotType.PARETO,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="request_latency", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
        )
        assert handler.can_handle(spec, sample_multi_run_dataframe) is True

    def test_can_handle_with_missing_metric(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Returns False when metric column missing from DataFrame."""
        handler = ParetoHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_pareto",
            title="Test Pareto",
            plot_type=PlotType.PARETO,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="nonexistent_metric", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
        )
        assert handler.can_handle(spec, sample_multi_run_dataframe) is False

    def test_can_handle_with_concurrency_metric(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Special handling for 'concurrency' column - always available."""
        handler = ParetoHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_pareto",
            title="Test Pareto",
            plot_type=PlotType.PARETO,
            filename="test.png",
            metrics=[
                MetricSpec(name="concurrency", source=DataSource.AGGREGATED, axis="x"),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
        )
        assert handler.can_handle(spec, sample_multi_run_dataframe) is True

    def test_create_plot_with_concurrency_as_x_metric(
        self, mock_plot_generator, sample_multi_run_dataframe, sample_available_metrics
    ):
        """Uses 'Concurrency Level' label for concurrency metric."""
        handler = ParetoHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_pareto",
            title="Test Pareto",
            plot_type=PlotType.PARETO,
            filename="test.png",
            metrics=[
                MetricSpec(name="concurrency", source=DataSource.AGGREGATED, axis="x"),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
        )

        handler.create_plot(spec, sample_multi_run_dataframe, sample_available_metrics)

        mock_plot_generator.create_pareto_plot.assert_called_once()
        call_kwargs = mock_plot_generator.create_pareto_plot.call_args[1]
        assert call_kwargs["x_label"] == "Concurrency Level"
        assert call_kwargs["x_metric"] == "concurrency"

    def test_create_plot_with_regular_metrics(
        self, mock_plot_generator, sample_multi_run_dataframe, sample_available_metrics
    ):
        """Uses _get_metric_label for both axes with regular metrics."""
        handler = ParetoHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_pareto",
            title="Test Pareto",
            plot_type=PlotType.PARETO,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="request_latency", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
        )

        handler.create_plot(spec, sample_multi_run_dataframe, sample_available_metrics)

        mock_plot_generator.create_pareto_plot.assert_called_once()
        call_kwargs = mock_plot_generator.create_pareto_plot.call_args[1]
        assert call_kwargs["x_label"] == f"Request Latency ({DEFAULT_PERCENTILE}) (ms)"
        assert call_kwargs["y_label"] == "Request Throughput (req/s)"

    def test_create_plot_passes_experiment_types(
        self, mock_plot_generator, sample_multi_run_dataframe, sample_available_metrics
    ):
        """Verifies experiment_types passed to plot_generator."""
        handler = ParetoHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_pareto",
            title="Test Pareto",
            plot_type=PlotType.PARETO,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="request_latency", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
            group_by=["experiment_group"],
        )

        handler.create_plot(spec, sample_multi_run_dataframe, sample_available_metrics)

        mock_plot_generator.create_pareto_plot.assert_called_once()
        call_kwargs = mock_plot_generator.create_pareto_plot.call_args[1]
        assert call_kwargs["experiment_types"] is not None
        assert call_kwargs["experiment_types"]["baseline"] == "baseline"
        assert call_kwargs["experiment_types"]["treatment_a"] == "treatment"

    def test_create_plot_passes_display_names(
        self, mock_plot_generator, sample_multi_run_dataframe, sample_available_metrics
    ):
        """Verifies group_display_names passed to plot_generator."""
        handler = ParetoHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_pareto",
            title="Test Pareto",
            plot_type=PlotType.PARETO,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="request_latency", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
            group_by=["experiment_group"],
        )

        handler.create_plot(spec, sample_multi_run_dataframe, sample_available_metrics)

        mock_plot_generator.create_pareto_plot.assert_called_once()
        call_kwargs = mock_plot_generator.create_pareto_plot.call_args[1]
        assert call_kwargs["group_display_names"] is not None
        assert call_kwargs["group_display_names"]["baseline"] == "Baseline Model"
        assert call_kwargs["group_display_names"]["treatment_a"] == "Treatment A"


class TestScatterLineHandler:
    """Tests for ScatterLineHandler class."""

    def test_can_handle_with_all_columns_present(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Returns True when all metrics available in DataFrame."""
        handler = ScatterLineHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_scatter",
            title="Test Scatter Line",
            plot_type=PlotType.SCATTER_LINE,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="request_latency", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
        )
        assert handler.can_handle(spec, sample_multi_run_dataframe) is True

    def test_can_handle_with_missing_metric(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Returns False when metric column missing from DataFrame."""
        handler = ScatterLineHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_scatter",
            title="Test Scatter Line",
            plot_type=PlotType.SCATTER_LINE,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="nonexistent_metric", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
        )
        assert handler.can_handle(spec, sample_multi_run_dataframe) is False

    def test_can_handle_with_concurrency_metric(
        self, mock_plot_generator, sample_multi_run_dataframe
    ):
        """Special handling for 'concurrency' column - always available."""
        handler = ScatterLineHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_scatter",
            title="Test Scatter Line",
            plot_type=PlotType.SCATTER_LINE,
            filename="test.png",
            metrics=[
                MetricSpec(name="concurrency", source=DataSource.AGGREGATED, axis="x"),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
        )
        assert handler.can_handle(spec, sample_multi_run_dataframe) is True

    def test_create_plot_with_concurrency_as_x_metric(
        self, mock_plot_generator, sample_multi_run_dataframe, sample_available_metrics
    ):
        """Uses 'Concurrency Level' label for concurrency metric."""
        handler = ScatterLineHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_scatter",
            title="Test Scatter Line",
            plot_type=PlotType.SCATTER_LINE,
            filename="test.png",
            metrics=[
                MetricSpec(name="concurrency", source=DataSource.AGGREGATED, axis="x"),
                MetricSpec(
                    name="request_throughput",
                    source=DataSource.AGGREGATED,
                    axis="y",
                    stat="avg",
                ),
            ],
        )

        handler.create_plot(spec, sample_multi_run_dataframe, sample_available_metrics)

        mock_plot_generator.create_scatter_line_plot.assert_called_once()
        call_kwargs = mock_plot_generator.create_scatter_line_plot.call_args[1]
        assert call_kwargs["x_label"] == "Concurrency Level"
        assert call_kwargs["x_metric"] == "concurrency"

    def test_create_plot_with_regular_metrics(
        self, mock_plot_generator, sample_multi_run_dataframe, sample_available_metrics
    ):
        """Uses _get_metric_label for both axes with regular metrics."""
        handler = ScatterLineHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_scatter",
            title="Test Scatter Line",
            plot_type=PlotType.SCATTER_LINE,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="request_latency", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
        )

        handler.create_plot(spec, sample_multi_run_dataframe, sample_available_metrics)

        mock_plot_generator.create_scatter_line_plot.assert_called_once()
        call_kwargs = mock_plot_generator.create_scatter_line_plot.call_args[1]
        assert call_kwargs["x_label"] == f"Request Latency ({DEFAULT_PERCENTILE}) (ms)"
        assert call_kwargs["y_label"] == "Request Throughput (req/s)"

    def test_create_plot_passes_experiment_types(
        self, mock_plot_generator, sample_multi_run_dataframe, sample_available_metrics
    ):
        """Verifies experiment_types passed to plot_generator."""
        handler = ScatterLineHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_scatter",
            title="Test Scatter Line",
            plot_type=PlotType.SCATTER_LINE,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="request_latency", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
            group_by=["experiment_group"],
        )

        handler.create_plot(spec, sample_multi_run_dataframe, sample_available_metrics)

        mock_plot_generator.create_scatter_line_plot.assert_called_once()
        call_kwargs = mock_plot_generator.create_scatter_line_plot.call_args[1]
        assert call_kwargs["experiment_types"] is not None
        assert call_kwargs["experiment_types"]["baseline"] == "baseline"
        assert call_kwargs["experiment_types"]["treatment_a"] == "treatment"

    def test_create_plot_passes_display_names(
        self, mock_plot_generator, sample_multi_run_dataframe, sample_available_metrics
    ):
        """Verifies group_display_names passed to plot_generator."""
        handler = ScatterLineHandler(mock_plot_generator)
        spec = PlotSpec(
            name="test_scatter",
            title="Test Scatter Line",
            plot_type=PlotType.SCATTER_LINE,
            filename="test.png",
            metrics=[
                MetricSpec(
                    name="request_latency", source=DataSource.AGGREGATED, axis="x"
                ),
                MetricSpec(
                    name="request_throughput", source=DataSource.AGGREGATED, axis="y"
                ),
            ],
            group_by=["experiment_group"],
        )

        handler.create_plot(spec, sample_multi_run_dataframe, sample_available_metrics)

        mock_plot_generator.create_scatter_line_plot.assert_called_once()
        call_kwargs = mock_plot_generator.create_scatter_line_plot.call_args[1]
        assert call_kwargs["group_display_names"] is not None
        assert call_kwargs["group_display_names"]["baseline"] == "Baseline Model"
        assert call_kwargs["group_display_names"]["treatment_a"] == "Treatment A"
