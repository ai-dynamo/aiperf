# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-run plot type handlers.

Handlers for creating plots from single profiling run data.
"""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import (
    aggregate_gpu_telemetry,
    calculate_rolling_percentiles,
    calculate_throughput_events,
    prepare_request_timeseries,
    prepare_timeslice_metrics,
    validate_request_uniformity,
)
from aiperf.plot.core.plot_generator import PlotGenerator
from aiperf.plot.core.plot_specs import (
    DataSource,
    MetricSpec,
    PlotSpec,
    PlotType,
    TimeSlicePlotSpec,
)
from aiperf.plot.core.plot_type_handlers import PlotTypeHandlerFactory
from aiperf.plot.exceptions import DataLoadError, PlotGenerationError
from aiperf.plot.metric_names import get_all_metric_display_names


class BaseSingleRunHandler:
    """
    Base class for single-run plot handlers.

    Provides common functionality for data preparation and validation.
    """

    def __init__(self, plot_generator: PlotGenerator, logger=None) -> None:
        """
        Initialize the handler.

        Args:
            plot_generator: PlotGenerator instance for rendering plots
            logger: Optional logger instance
        """
        self.plot_generator = plot_generator
        self.logger = logger

    def _get_axis_label(self, metric_spec: MetricSpec, available_metrics: dict) -> str:
        """
        Get axis label for a metric.

        Args:
            metric_spec: MetricSpec object
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted axis label
        """
        if metric_spec.name == "request_number":
            return "Request Number"
        elif metric_spec.name == "timestamp":
            return "Time (seconds)"
        elif metric_spec.name == "timestamp_s":
            return "Time (s)"
        elif metric_spec.name == "Timeslice":
            return "Timeslice (s)"
        else:
            return self._get_metric_label(
                metric_spec.name, metric_spec.stat, available_metrics
            )

    def _get_metric_label(
        self, metric_name: str, stat: str | None, available_metrics: dict
    ) -> str:
        """
        Get formatted metric label.

        Args:
            metric_name: Name of the metric
            stat: Statistic (e.g., "avg", "p50")
            available_metrics: Dictionary with display_names and units

        Returns:
            Formatted metric label
        """
        if metric_name in available_metrics:
            display_name = available_metrics[metric_name].get(
                "display_name", metric_name
            )
            unit = available_metrics[metric_name].get("unit", "")
            if stat and stat not in ["avg", "value"]:
                display_name = f"{display_name} ({stat})"
            if unit:
                return f"{display_name} ({unit})"
            return display_name
        return metric_name

    def _prepare_data_for_source(
        self, source: DataSource, run: RunData
    ) -> pd.DataFrame:
        """
        Prepare data from a specific source.

        Args:
            source: Data source to prepare
            run: RunData object

        Returns:
            Prepared DataFrame
        """
        if source == DataSource.REQUESTS:
            return prepare_request_timeseries(run)
        elif source == DataSource.TIMESLICES:
            return run.timeslices
        elif source == DataSource.GPU_TELEMETRY:
            return run.gpu_telemetry
        else:
            raise PlotGenerationError(f"Unsupported data source: {source}")


@PlotTypeHandlerFactory.register(PlotType.SCATTER)
class ScatterHandler(BaseSingleRunHandler):
    """Handler for scatter plot type."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if scatter plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                data.requests is None or data.requests.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        df = self._prepare_data_for_source(x_metric.source, data)

        return self.plot_generator.create_time_series_scatter(
            df=df,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )


@PlotTypeHandlerFactory.register(PlotType.AREA)
class AreaHandler(BaseSingleRunHandler):
    """Handler for area plot type."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if area plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                data.requests is None or data.requests.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create an area plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Special handling for dispersed throughput due to nature of request throughput data
        if y_metric.name == "throughput_tokens_per_sec":
            df = prepare_request_timeseries(data)
            throughput_df = calculate_throughput_events(df)
        else:
            throughput_df = self._prepare_data_for_source(x_metric.source, data)

        return self.plot_generator.create_time_series_area(
            df=throughput_df,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )


@PlotTypeHandlerFactory.register(PlotType.TIMESLICE)
class TimeSliceHandler(BaseSingleRunHandler):
    """Handler for timeslice scatter plot type."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if timeslice plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.TIMESLICES and (
                data.timeslices is None or data.timeslices.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a timeslice scatter plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        stats_to_extract = ["avg", "std"]
        plot_df, unit = prepare_timeslice_metrics(data, y_metric.name, stats_to_extract)

        y_label = f"{y_metric.name} ({unit})" if unit else y_metric.name

        use_slice_duration = (
            isinstance(spec, TimeSlicePlotSpec) and spec.use_slice_duration
        )

        warning_message = None
        if "throughput" in spec.name.lower():
            _, warning_message = validate_request_uniformity(data, self.logger)

        # Extract average and std from aggregated stats by converting display name to metric tag
        average_value, average_label, average_std = (
            self._get_average_for_timeslice_metric(y_metric.name, data)
        )

        return self.plot_generator.create_timeslice_scatter(
            df=plot_df,
            x_col=x_metric.name,
            y_col=y_metric.stat,
            metric_name=y_metric.name,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=y_label,
            slice_duration=data.slice_duration if use_slice_duration else None,
            warning_text=warning_message,
            average_value=average_value,
            average_label=average_label,
            average_std=average_std,
            unit=unit,
        )

    def _get_average_for_timeslice_metric(
        self, metric_display_name: str, data: RunData
    ) -> tuple[float | None, str | None, float | None]:
        """
        Get average value and std for a timeslice metric from aggregated stats.

        Args:
            metric_display_name: Display name of the metric (e.g., "Time to First Token")
            data: RunData object containing aggregated stats

        Returns:
            Tuple of (average_value, formatted_label, std_value) or (None, None, None) if not found
        """

        display_to_tag = {v: k for k, v in get_all_metric_display_names().items()}
        metric_tag = display_to_tag.get(metric_display_name)
        if metric_tag is None:
            return None, None, None

        metric = data.get_metric(metric_tag)
        if not metric:
            return None, None, None

        avg = metric.avg if hasattr(metric, "avg") else metric.get("avg")
        unit = metric.unit if hasattr(metric, "unit") else metric.get("unit", "")
        std = metric.std if hasattr(metric, "std") else metric.get("std")

        if avg is None:
            return None, None, None

        label = f"Run Average: {avg:.2f}"
        if unit:
            label += f" {unit}"

        return avg, label, std


@PlotTypeHandlerFactory.register(PlotType.HISTOGRAM)
class HistogramHandler(BaseSingleRunHandler):
    """Handler for histogram/bar chart plots.

    Generates histogram/bar chart visualizations for timeslice data.
    """

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if histogram plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.TIMESLICES and (
                data.timeslices is None or data.timeslices.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a histogram/bar chart plot."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        stats_to_extract = ["avg", "std"]
        plot_df, unit = prepare_timeslice_metrics(data, y_metric.name, stats_to_extract)

        y_label = f"{y_metric.name} ({unit})" if unit else y_metric.name

        use_slice_duration = (
            isinstance(spec, TimeSlicePlotSpec) and spec.use_slice_duration
        )

        warning_message = None
        if "throughput" in spec.name.lower():
            _, warning_message = validate_request_uniformity(data, self.logger)

        # Extract average and std from aggregated stats
        average_value, average_label, average_std = (
            self._get_average_for_timeslice_metric(y_metric.name, data)
        )

        return self.plot_generator.create_time_series_histogram(
            df=plot_df,
            x_col=x_metric.name,
            y_col=y_metric.stat,
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=y_label,
            slice_duration=data.slice_duration if use_slice_duration else None,
            warning_text=warning_message,
            average_value=average_value,
            average_label=average_label,
            average_std=average_std,
        )

    def _get_average_for_timeslice_metric(
        self, metric_display_name: str, data: RunData
    ) -> tuple[float | None, str | None, float | None]:
        """Get average value and std for a timeslice metric from aggregated stats.

        Args:
            metric_display_name: Display name of the metric
            data: RunData object containing aggregated stats

        Returns:
            Tuple of (average_value, formatted_label, std_value) or (None, None, None)
        """
        from aiperf.plot.metric_names import get_all_metric_display_names

        display_to_tag = {v: k for k, v in get_all_metric_display_names().items()}
        metric_tag = display_to_tag.get(metric_display_name)
        if metric_tag is None:
            return None, None, None

        metric = data.get_metric(metric_tag)
        if not metric:
            return None, None, None

        avg = metric.avg if hasattr(metric, "avg") else metric.get("avg")
        unit = metric.unit if hasattr(metric, "unit") else metric.get("unit", "")
        std = metric.std if hasattr(metric, "std") else metric.get("std")

        if avg is None:
            return None, None, None

        label = f"Run Average: {avg:.2f}"
        if unit:
            label += f" {unit}"

        return avg, label, std


@PlotTypeHandlerFactory.register(PlotType.DUAL_AXIS)
class DualAxisHandler(BaseSingleRunHandler):
    """Handler for dual-axis plot type."""

    # Metric-specific data preparation functions
    METRIC_PREP_FUNCTIONS = {
        "throughput_tokens_per_sec": lambda self, data: calculate_throughput_events(
            prepare_request_timeseries(data)
        ),
        "gpu_utilization": lambda self, data: aggregate_gpu_telemetry(data),
    }

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if dual-axis plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.GPU_TELEMETRY and (
                data.gpu_telemetry is None or data.gpu_telemetry.empty
            ):
                return False
        return True

    def _prepare_metric_data(
        self, metric_name: str, source: DataSource, data: RunData
    ) -> pd.DataFrame:
        """
        Prepare data for a specific metric with optional special handling.

        Args:
            metric_name: Name of the metric
            source: Data source for the metric
            data: RunData object

        Returns:
            Prepared DataFrame
        """
        if metric_name in self.METRIC_PREP_FUNCTIONS:
            return self.METRIC_PREP_FUNCTIONS[metric_name](self, data)
        else:
            return self._prepare_data_for_source(source, data)

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a dual-axis plot."""
        x_metric = next((m for m in spec.metrics if m.axis == "x"), None)
        y1_metric = next(m for m in spec.metrics if m.axis == "y")
        y2_metric = next(m for m in spec.metrics if m.axis == "y2")

        df_primary = self._prepare_metric_data(y1_metric.name, y1_metric.source, data)
        df_secondary = self._prepare_metric_data(y2_metric.name, y2_metric.source, data)

        if df_primary.empty:
            raise DataLoadError(
                f"No data available for primary metric: {y1_metric.name}"
            )

        x_col = x_metric.name if x_metric else "timestamp_s"

        x_label = (
            self._get_axis_label(x_metric, available_metrics)
            if x_metric
            else "Time (s)"
        )
        y1_label = self._get_axis_label(y1_metric, available_metrics)
        y2_label = self._get_axis_label(y2_metric, available_metrics)

        return self.plot_generator.create_dual_axis_plot(
            df_primary=df_primary,
            df_secondary=df_secondary,
            x_col_primary=x_col,
            x_col_secondary=x_col,
            y1_metric=y1_metric.name,
            y2_metric=y2_metric.name,
            primary_style=spec.primary_style,
            secondary_style=spec.secondary_style,
            active_count_col=spec.supplementary_col,
            title=spec.title,
            x_label=x_label,
            y1_label=y1_label,
            y2_label=y2_label,
        )


@PlotTypeHandlerFactory.register(PlotType.SCATTER_WITH_PERCENTILES)
class ScatterWithPercentilesHandler(BaseSingleRunHandler):
    """Handler for scatter plot with percentile overlays."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if scatter with percentiles plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                data.requests is None or data.requests.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter plot with percentile overlays."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        df = self._prepare_data_for_source(x_metric.source, data)
        df_sorted = df.sort_values(x_metric.name).copy()

        df_sorted = calculate_rolling_percentiles(df_sorted, y_metric.name)

        return self.plot_generator.create_latency_scatter_with_percentiles(
            df=df_sorted,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            percentile_cols=["p50", "p95", "p99"],
            title=spec.title,
            x_label=self._get_axis_label(x_metric, available_metrics),
            y_label=self._get_axis_label(y_metric, available_metrics),
        )
