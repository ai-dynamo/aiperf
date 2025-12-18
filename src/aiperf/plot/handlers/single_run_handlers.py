# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-run plot type handlers.

Handlers for creating plots from single profiling run data.
"""

import logging

import orjson
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
from aiperf.plot.exceptions import (
    DataUnavailableError,
    PlotGenerationError,
)
from aiperf.plot.metric_names import get_all_metric_display_names, get_gpu_metric_unit
from aiperf.plot.utils import (
    create_series_legend_label,
    detect_server_metric_series,
    filter_server_metrics_dataframe,
    parse_server_metric_spec,
)
from aiperf.server_metrics.histogram_percentiles import compute_prometheus_percentiles

_logger = logging.getLogger(__name__)


def _is_single_stat_metric(metric) -> bool:
    """
    Check if metric only has 'avg' stat (no distribution stats like p50, std, etc.).

    Single-stat metrics are derived values (like throughput, count) where the aggregated
    "avg" is a calculated value (total/duration), not a statistical average of samples.

    Args:
        metric: MetricResult object or dict containing metric data

    Returns:
        True if metric only has 'avg' stat, False otherwise
    """
    distribution_stats = {
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "std",
        "min",
        "max",
    }

    # Check if any non-None distribution stat exists
    for stat in distribution_stats:
        if hasattr(metric, stat):
            val = getattr(metric, stat)
        elif isinstance(metric, dict):
            val = metric.get(stat)
        else:
            continue
        if val is not None:
            return False

    return True


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

    def _get_custom_or_default_label(
        self,
        custom_label: str | None,
        metric_spec: MetricSpec,
        available_metrics: dict,
    ) -> str:
        """
        Get custom axis label if provided, otherwise auto-generate.

        Args:
            custom_label: Custom label from PlotSpec (x_label or y_label)
            metric_spec: MetricSpec object for fallback generation
            available_metrics: Dictionary with display_names and units

        Returns:
            Custom label if provided, otherwise auto-generated label
        """
        if custom_label:
            return custom_label
        return self._get_axis_label(metric_spec, available_metrics)

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
        display_name = None
        unit = ""

        if "display_names" in available_metrics or "units" in available_metrics:
            display_name = available_metrics.get("display_names", {}).get(metric_name)
            unit = available_metrics.get("units", {}).get(metric_name, "")

        if not display_name and metric_name in available_metrics:
            display_name = available_metrics[metric_name].get(
                "display_name", metric_name
            )
            unit = available_metrics[metric_name].get("unit", "")

        if display_name:
            if stat and stat not in ["avg", "value"]:
                display_name = f"{display_name} ({stat})"
            if unit:
                return f"{display_name} ({unit})"
            return display_name

        # Fallback: Check if it's a GPU metric and get unit from GPU config
        display_name = metric_name.replace("_", " ").title()
        gpu_unit = get_gpu_metric_unit(metric_name)
        # Heuristic: metrics with "utilization" in the name are percentages
        if not gpu_unit and "utilization" in metric_name.lower():
            gpu_unit = "%"
        if stat and stat not in ["avg", "value"]:
            display_name = f"{display_name} ({stat})"
        if gpu_unit:
            return f"{display_name} ({gpu_unit})"
        return display_name

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
    """Handler for scatter plot type (supports REQUESTS and SERVER_METRICS sources)."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if scatter plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                data.requests is None or data.requests.empty
            ):
                return False
            if metric.source == DataSource.SERVER_METRICS and (
                data.server_metrics is None or data.server_metrics.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a scatter plot (supports REQUESTS and SERVER_METRICS sources)."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Handle SERVER_METRICS source
        if y_metric.source == DataSource.SERVER_METRICS:
            if data.server_metrics is None or data.server_metrics.empty:
                raise DataUnavailableError(
                    "Scatter plot cannot be generated: no server metrics data available.",
                    data_type="server_metrics",
                    hint="Server metrics data requires server_metrics collection to be enabled.",
                )

            # Parse metric name and apply filters
            metric_name, endpoint_filter, labels_filter = parse_server_metric_spec(
                y_metric.name
            )

            df, unit, metric_type = filter_server_metrics_dataframe(
                data.server_metrics, metric_name, endpoint_filter, labels_filter
            )

            y_label = self._get_custom_or_default_label(
                spec.y_label, y_metric, available_metrics
            )
            if not spec.y_label and unit:
                y_label = f"{metric_name} ({unit})"

            return self.plot_generator.create_time_series_scatter(
                df=df,
                x_col="timestamp_s",
                y_metric="value",
                title=spec.title or f"{metric_name} Raw Data Points Over Time",
                x_label=spec.x_label or "Time (s)",
                y_label=y_label,
            )

        # Handle REQUESTS source (existing logic)
        if data.requests is None or data.requests.empty:
            raise DataUnavailableError(
                "Scatter plot cannot be generated: no per-request data available.",
                data_type="requests",
                hint="Per-request data is generated during benchmark runs.",
            )

        df = self._prepare_data_for_source(x_metric.source, data)

        return self.plot_generator.create_time_series_scatter(
            df=df,
            x_col=x_metric.name,
            y_metric=y_metric.name,
            title=spec.title,
            x_label=self._get_custom_or_default_label(
                spec.x_label, x_metric, available_metrics
            ),
            y_label=self._get_custom_or_default_label(
                spec.y_label, y_metric, available_metrics
            ),
        )


@PlotTypeHandlerFactory.register(PlotType.AREA)
class AreaHandler(BaseSingleRunHandler):
    """Handler for area plot type (supports REQUESTS and SERVER_METRICS sources)."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if area plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.REQUESTS and (
                data.requests is None or data.requests.empty
            ):
                return False
            if metric.source == DataSource.SERVER_METRICS and (
                data.server_metrics is None or data.server_metrics.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create an area plot (supports REQUESTS and SERVER_METRICS sources)."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Handle SERVER_METRICS source
        if y_metric.source == DataSource.SERVER_METRICS:
            if data.server_metrics is None or data.server_metrics.empty:
                raise DataUnavailableError(
                    "Area plot cannot be generated: no server metrics data available.",
                    data_type="server_metrics",
                    hint="Server metrics data requires server_metrics collection to be enabled.",
                )
            # Prepare server metrics data
            throughput_df = self._prepare_server_metrics_for_area(y_metric.name, data)
        else:
            # Handle REQUESTS source (existing logic)
            if data.requests is None or data.requests.empty:
                raise DataUnavailableError(
                    "Area plot cannot be generated: no per-request data available.",
                    data_type="requests",
                    hint="Per-request data is generated during benchmark runs.",
                )

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
            x_label=self._get_custom_or_default_label(
                spec.x_label, x_metric, available_metrics
            ),
            y_label=self._get_custom_or_default_label(
                spec.y_label, y_metric, available_metrics
            ),
        )

    def _prepare_server_metrics_for_area(
        self, metric_name: str, data: RunData
    ) -> pd.DataFrame:
        """
        Prepare server metrics data for area plotting.

        Handles both single-series and multi-series scenarios. For multi-series
        (multiple endpoint/label combinations), aggregates values by timestamp
        to create a single merged time series for area fill.

        Args:
            metric_name: Server metric name (may include filters)
            data: RunData object

        Returns:
            DataFrame with timestamp_s and metric value column
        """
        # Parse and filter using shared utility
        base_metric, endpoint_filter, labels_filter = parse_server_metric_spec(
            metric_name
        )

        try:
            df, unit, metric_type = filter_server_metrics_dataframe(
                data.server_metrics, base_metric, endpoint_filter, labels_filter
            )
        except ValueError:
            # Return empty DataFrame if filtering fails
            return pd.DataFrame()

        if df.empty:
            return pd.DataFrame()

        # Detect series count
        series_list = detect_server_metric_series(df)

        # If multiple series, aggregate by timestamp
        if len(series_list) > 1:
            # Group by timestamp and sum/average values
            if metric_type == "COUNTER":
                # Sum rates for counters
                df_agg = df.groupby("timestamp_s")["value"].sum().reset_index()
            else:
                # Average for gauges/histograms
                df_agg = df.groupby("timestamp_s")["value"].mean().reset_index()

            df_agg[base_metric] = df_agg["value"]
            return df_agg[["timestamp_s", base_metric]].copy()

        # Single series - rename value column
        df[base_metric] = df["value"]
        return df[["timestamp_s", base_metric]].copy()


@PlotTypeHandlerFactory.register(PlotType.TIMESLICE)
class TimeSliceHandler(BaseSingleRunHandler):
    """Handler for timeslice scatter plot type (supports TIMESLICES and SERVER_METRICS sources)."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if timeslice plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.TIMESLICES and (
                data.timeslices is None or data.timeslices.empty
            ):
                return False
            if metric.source == DataSource.SERVER_METRICS and (
                data.server_metrics is None or data.server_metrics.empty
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a timeslice scatter plot (supports TIMESLICES and SERVER_METRICS sources)."""
        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        # Handle SERVER_METRICS source
        if y_metric.source == DataSource.SERVER_METRICS:
            return self._create_server_metrics_plot(
                spec, data, available_metrics, x_metric, y_metric
            )

        # Handle TIMESLICES source (existing logic)
        if data.timeslices is None or data.timeslices.empty:
            raise DataUnavailableError(
                "Timeslice plot cannot be generated: no timeslice data available.",
                data_type="timeslice",
                hint="Timeslice data requires running benchmarks with slice_duration configured.",
            )

        stats_to_extract = ["avg", "std"]
        plot_df, unit = prepare_timeslice_metrics(data, y_metric.name, stats_to_extract)

        default_y_label = f"{y_metric.name} ({unit})" if unit else y_metric.name
        y_label = spec.y_label or default_y_label

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
            x_label=spec.x_label or self._get_axis_label(x_metric, available_metrics),
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

        # Skip reference line for single-stat metrics (derived values like throughput, count)
        # These only have "avg" because they're calculated values (total/duration),
        # not per-request measurements with distributions
        if _is_single_stat_metric(metric):
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

    def _create_server_metrics_plot(
        self,
        spec: PlotSpec,
        data: RunData,
        available_metrics: dict,
        x_metric: MetricSpec,
        y_metric: MetricSpec,
    ) -> go.Figure:
        """
        Create a server metrics time series plot.

        Args:
            spec: Plot specification
            data: RunData with server_metrics DataFrame
            available_metrics: Available metrics metadata
            x_metric: X-axis metric specification
            y_metric: Y-axis metric specification (server metrics)

        Returns:
            Plotly Figure object

        Raises:
            DataUnavailableError: If server metrics data is not available
        """
        if data.server_metrics is None or data.server_metrics.empty:
            raise DataUnavailableError(
                "Server metrics plot cannot be generated: no server metrics data available.",
                data_type="server_metrics",
                hint="Server metrics data requires server_metrics collection to be enabled.",
            )

        # Parse metric name for optional endpoint/label filters (using shared utility)
        metric_name, endpoint_filter, labels_filter = parse_server_metric_spec(
            y_metric.name
        )

        # Filter DataFrame using shared utility
        try:
            df, unit, metric_type = filter_server_metrics_dataframe(
                data.server_metrics, metric_name, endpoint_filter, labels_filter
            )
        except ValueError as e:
            raise DataUnavailableError(
                str(e),
                data_type="server_metrics",
            ) from e

        # Detect if multiple series exist (different endpoint/label combinations)
        series_list = detect_server_metric_series(df)

        # If multiple series and no explicit filter, create multi-series plot
        if len(series_list) > 1 and endpoint_filter is None and labels_filter is None:
            return self._create_multi_series_server_metrics_plot(
                df, spec, metric_name, series_list, unit, available_metrics, x_metric
            )

        # Single series - use existing plot logic
        # Get aggregated stats for run average overlay
        avg_value, avg_label, avg_std = self._get_server_metric_average(
            data, metric_name, endpoint_filter, labels_filter
        )

        # Determine y column based on stat (default to "value")
        y_col = "value"

        default_y_label = f"{metric_name} ({unit})" if unit else metric_name
        y_label = spec.y_label or default_y_label

        return self.plot_generator.create_timeslice_scatter(
            df=df,
            x_col="timestamp_s",
            y_col=y_col,
            metric_name=metric_name,
            title=spec.title or f"{metric_name} Over Time",
            x_label=spec.x_label or "Time (s)",
            y_label=y_label,
            slice_duration=None,  # No windowing for server metrics
            average_value=avg_value,
            average_label=avg_label,
            average_std=avg_std,
            unit=unit,
        )

    def _create_multi_series_server_metrics_plot(
        self,
        df,
        spec: PlotSpec,
        metric_name: str,
        series_list: list[tuple[str, str]],
        unit: str,
        available_metrics: dict,
        x_metric: MetricSpec,
    ) -> go.Figure:
        """
        Create server metrics plot with multiple series (one trace per endpoint/label combo).

        Generates a plot with separate traces for each unique endpoint + label
        combination, with automatic legend entries showing what differentiates
        each series.

        Args:
            df: Filtered server metrics DataFrame (all series)
            spec: Plot specification
            metric_name: Base metric name
            series_list: List of (endpoint_url, labels_json) tuples
            unit: Metric unit for axis label
            available_metrics: Available metrics metadata
            x_metric: X-axis metric specification

        Returns:
            Plotly Figure with multiple traces
        """
        # Create figure manually with multiple traces
        fig = go.Figure()

        total_series = len(series_list)

        # Extract all labels for smart filtering
        all_series_labels = [
            orjson.loads(labels_json.encode()) if labels_json != "{}" else {}
            for _, labels_json in series_list
        ]

        # Add trace for each series
        for endpoint_url, labels_json in series_list:
            # Filter to this specific series
            series_df = df[
                (df["endpoint_url"] == endpoint_url)
                & (df["labels_json"] == labels_json)
            ].copy()

            if series_df.empty:
                continue

            # Sort by timestamp
            series_df = series_df.sort_values("timestamp_ns")

            # Create legend label (with smart filtering)
            trace_name = create_series_legend_label(
                metric_name, endpoint_url, labels_json, total_series, all_series_labels
            )

            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=series_df["timestamp_s"],
                    y=series_df["value"],
                    mode="lines+markers",
                    name=trace_name,
                    marker={"size": 6},
                    line={"width": 2},
                )
            )

        # Apply NVIDIA styling
        default_y_label = f"{metric_name} ({unit})" if unit else metric_name
        y_label = spec.y_label or default_y_label

        fig.update_layout(
            title=spec.title or f"{metric_name} Over Time (Multi-Series)",
            xaxis_title=spec.x_label or "Time (s)",
            yaxis_title=y_label,
            template="plotly_white",
            showlegend=True,
            legend={
                "orientation": "v",
                "yanchor": "top",
                "y": 1,
                "xanchor": "left",
                "x": 1.02,
            },
        )

        return fig

    def _get_server_metric_average(
        self,
        data: RunData,
        metric_name: str,
        endpoint: str | None,
        labels: dict | None,
    ) -> tuple[float | None, str | None, float | None]:
        """
        Get average value and std for a server metric from aggregated stats.

        Args:
            data: RunData object
            metric_name: Server metric name
            endpoint: Optional endpoint filter
            labels: Optional labels filter

        Returns:
            Tuple of (average_value, formatted_label, std_value)
        """
        if not data.server_metrics_aggregated:
            if self.logger:
                self.logger.debug(
                    "Server metrics aggregated stats not available. "
                    "Average line will not be displayed. "
                    "Ensure server_metrics_export.json exists alongside Parquet file."
                )
            return None, None, None

        if metric_name not in data.server_metrics_aggregated:
            if self.logger:
                available = list(data.server_metrics_aggregated.keys())[:5]
                self.logger.debug(
                    f"Metric '{metric_name}' not found in aggregated stats. "
                    f"Available metrics: {available}..."
                )
            return None, None, None

        metric_data = data.server_metrics_aggregated[metric_name]

        # If no endpoint specified, use first available endpoint
        if endpoint is None:
            if not metric_data:
                return None, None, None
            endpoint = next(iter(metric_data.keys()))

        if endpoint not in metric_data:
            return None, None, None

        labels_key = (
            orjson.dumps(labels, option=orjson.OPT_SORT_KEYS).decode()
            if labels
            else "{}"
        )

        if labels_key not in metric_data[endpoint]:
            return None, None, None

        series_data = metric_data[endpoint][labels_key]
        stats = series_data.get("stats")
        unit = series_data.get("unit", "")

        if stats is None:
            # Static value (no variation)
            return None, None, None

        # Extract avg and std based on metric type
        avg = None
        std = None

        if hasattr(stats, "avg"):
            avg = stats.avg
        elif isinstance(stats, dict):
            avg = stats.get("avg")

        if hasattr(stats, "std"):
            std = stats.std
        elif isinstance(stats, dict):
            std = stats.get("std")

        if avg is None:
            return None, None, None

        label = f"Run Average: {avg:.2f}"
        if unit:
            label += f" {unit}"

        return avg, label, std


@PlotTypeHandlerFactory.register(PlotType.HISTOGRAM)
class HistogramHandler(BaseSingleRunHandler):
    """Handler for histogram/bar chart plots.

    Supports two modes:
    - TIMESLICES: Time-windowed bar charts of client metrics
    - SERVER_METRICS: Prometheus histogram bucket distribution visualization
    """

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if histogram plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.TIMESLICES and (
                data.timeslices is None or data.timeslices.empty
            ):
                return False
            if metric.source == DataSource.SERVER_METRICS and (
                data.server_metrics_aggregated is None
                or not data.server_metrics_aggregated
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a histogram/bar chart plot.

        For TIMESLICES: Bar chart of metrics over time windows
        For SERVER_METRICS: Prometheus histogram bucket distribution
        """
        y_metric = next((m for m in spec.metrics if m.axis == "y"), None)
        if not y_metric:
            raise ValueError("Histogram plot requires a y-axis metric")

        # Handle SERVER_METRICS source (Prometheus histogram bucket visualization)
        if y_metric.source == DataSource.SERVER_METRICS:
            return self._create_server_metrics_bucket_histogram(
                y_metric, spec, data, available_metrics
            )

        # Handle TIMESLICES source (existing logic)
        if data.timeslices is None or data.timeslices.empty:
            raise DataUnavailableError(
                "Histogram plot cannot be generated: no timeslice data available.",
                data_type="timeslice",
                hint="Timeslice data requires running benchmarks with slice_duration configured.",
            )

        x_metric = next(m for m in spec.metrics if m.axis == "x")
        y_metric = next(m for m in spec.metrics if m.axis == "y")

        stats_to_extract = ["avg", "std"]
        plot_df, unit = prepare_timeslice_metrics(data, y_metric.name, stats_to_extract)

        default_y_label = f"{y_metric.name} ({unit})" if unit else y_metric.name
        y_label = spec.y_label or default_y_label

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
            x_label=spec.x_label or self._get_axis_label(x_metric, available_metrics),
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
        display_to_tag = {v: k for k, v in get_all_metric_display_names().items()}
        metric_tag = display_to_tag.get(metric_display_name)
        if metric_tag is None:
            return None, None, None

        metric = data.get_metric(metric_tag)
        if not metric:
            return None, None, None

        # Skip reference line for single-stat metrics (derived values like throughput, count)
        # These only have "avg" because they're calculated values (total/duration),
        # not per-request measurements with distributions
        if _is_single_stat_metric(metric):
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

    def _create_server_metrics_bucket_histogram(
        self,
        y_metric,
        spec: PlotSpec,
        data: RunData,
        available_metrics: dict,
    ) -> go.Figure:
        """Create Prometheus histogram bucket distribution visualization.

        Visualizes the bucket distribution from a Prometheus histogram metric,
        showing the actual bucket boundaries and observation counts.

        Args:
            y_metric: Y-axis metric spec
            spec: PlotSpec for the histogram
            data: RunData with server metrics
            available_metrics: Available metrics dict

        Returns:
            Plotly Figure with bucket distribution bar chart
        """
        metric_name, endpoint_filter, labels_filter = parse_server_metric_spec(
            y_metric.name
        )

        # Get metric data from aggregated stats
        if metric_name not in data.server_metrics_aggregated:
            available = list(data.server_metrics_aggregated.keys())[:10]
            raise DataUnavailableError(
                f"Metric '{metric_name}' not found in server metrics. "
                f"Available: {available}",
                data_type="server_metrics",
            )

        metric_data = data.server_metrics_aggregated[metric_name]

        # Get first endpoint if not specified
        if endpoint_filter is None:
            endpoint_filter = next(iter(metric_data.keys()))

        if endpoint_filter not in metric_data:
            raise DataUnavailableError(
                f"Endpoint '{endpoint_filter}' not found for metric '{metric_name}'",
                data_type="server_metrics",
            )

        labels_key = (
            orjson.dumps(labels_filter, option=orjson.OPT_SORT_KEYS).decode()
            if labels_filter
            else "{}"
        )

        if labels_key not in metric_data[endpoint_filter]:
            raise DataUnavailableError(
                f"Labels {labels_filter} not found for metric '{metric_name}'",
                data_type="server_metrics",
            )

        series_data = metric_data[endpoint_filter][labels_key]
        metric_type = series_data.get("type", "").upper()
        unit = series_data.get("unit", "")

        # Verify this is a histogram metric
        if metric_type != "HISTOGRAM":
            raise DataUnavailableError(
                f"Metric '{metric_name}' is type {metric_type}, not HISTOGRAM. "
                "Bucket distribution visualization requires HISTOGRAM metrics.",
                data_type="server_metrics",
            )

        # Get bucket data from series stats
        stats = series_data.get("stats")
        if not stats:
            raise DataUnavailableError(
                f"No statistics available for metric '{metric_name}'",
                data_type="server_metrics",
            )

        # Try to get buckets from stats (should be dict)
        buckets = None
        if isinstance(stats, dict):
            buckets = stats.get("buckets")
        elif hasattr(stats, "buckets"):
            buckets = stats.buckets

        if not buckets:
            raise DataUnavailableError(
                f"No bucket data available for histogram metric '{metric_name}'. "
                "Bucket distribution requires histogram metrics with bucket data.",
                data_type="server_metrics",
            )

        # Create bucket histogram
        y_label = self._get_custom_or_default_label(
            spec.y_label, y_metric, available_metrics
        )
        if not spec.y_label:
            y_label = "Observation Count"

        return self.plot_generator.create_bucket_histogram(
            buckets=buckets,
            metric_name=metric_name,
            title=spec.title or f"{metric_name} Distribution (Histogram Buckets)",
            x_label=spec.x_label or f"Bucket Upper Bound ({unit})"
            if unit
            else "Bucket Upper Bound",
            y_label=y_label,
            unit=unit,
        )


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
            if metric.source == DataSource.SERVER_METRICS and (
                data.server_metrics is None or data.server_metrics.empty
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
        elif source == DataSource.SERVER_METRICS:
            return self._prepare_server_metrics_data(metric_name, data)
        else:
            return self._prepare_data_for_source(source, data)

    def _prepare_server_metrics_data(
        self, metric_name: str, data: RunData
    ) -> pd.DataFrame:
        """
        Prepare server metrics data for dual-axis plotting.

        Args:
            metric_name: Server metric name (may include filters)
            data: RunData object

        Returns:
            DataFrame with timestamp_s and value columns
        """
        # Parse and filter using shared utility
        base_metric, endpoint_filter, labels_filter = parse_server_metric_spec(
            metric_name
        )

        try:
            df, unit, metric_type = filter_server_metrics_dataframe(
                data.server_metrics, base_metric, endpoint_filter, labels_filter
            )
        except ValueError:
            # Return empty DataFrame if filtering fails
            return pd.DataFrame()

        # Return DataFrame with required columns for dual-axis plot
        return df[["timestamp_s", "value"]].copy() if not df.empty else pd.DataFrame()

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create a dual-axis plot."""
        x_metric = next((m for m in spec.metrics if m.axis == "x"), None)
        y1_metric = next(m for m in spec.metrics if m.axis == "y")
        y2_metric = next(m for m in spec.metrics if m.axis == "y2")

        if y1_metric.source == DataSource.GPU_TELEMETRY and (
            data.gpu_telemetry is None or data.gpu_telemetry.empty
        ):
            raise DataUnavailableError(
                f"Dual-axis plot cannot be generated: no GPU telemetry data for {y1_metric.name}.",
                data_type="gpu_telemetry",
                hint="GPU telemetry requires DCGM to be configured during benchmark runs.",
            )
        if y2_metric.source == DataSource.GPU_TELEMETRY and (
            data.gpu_telemetry is None or data.gpu_telemetry.empty
        ):
            raise DataUnavailableError(
                f"Dual-axis plot cannot be generated: no GPU telemetry data for {y2_metric.name}.",
                data_type="gpu_telemetry",
                hint="GPU telemetry requires DCGM to be configured during benchmark runs.",
            )

        df_primary = self._prepare_metric_data(y1_metric.name, y1_metric.source, data)
        df_secondary = self._prepare_metric_data(y2_metric.name, y2_metric.source, data)

        if df_primary.empty:
            raise DataUnavailableError(
                f"Dual-axis plot cannot be generated: no data for {y1_metric.name}.",
                data_type=y1_metric.source.value if y1_metric.source else "unknown",
            )

        x_col = x_metric.name if x_metric else "timestamp_s"

        default_x_label = (
            self._get_axis_label(x_metric, available_metrics)
            if x_metric
            else "Time (s)"
        )
        x_label = spec.x_label or default_x_label
        y1_label = spec.y_label or self._get_axis_label(y1_metric, available_metrics)
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
        if data.requests is None or data.requests.empty:
            raise DataUnavailableError(
                "Scatter with percentiles plot cannot be generated: no per-request data available.",
                data_type="requests",
                hint="Per-request data is generated during benchmark runs.",
            )

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
            x_label=self._get_custom_or_default_label(
                spec.x_label, x_metric, available_metrics
            ),
            y_label=self._get_custom_or_default_label(
                spec.y_label, y_metric, available_metrics
            ),
        )


@PlotTypeHandlerFactory.register(PlotType.PERCENTILE_BANDS)
class PercentileBandsHandler(BaseSingleRunHandler):
    """Handler for percentile bands visualization over time.

    Renders time-series with p50 median line and p95/p99 shaded uncertainty bands.
    Perfect for SLA monitoring and latency stability analysis with server metrics.

    Supports:
    - HISTOGRAM metrics: Uses bucket data from timeslices to compute percentiles per window
    - GAUGE metrics: Shows min/avg/max bands (no percentiles available)
    """

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """Check if percentile bands plot can be generated."""
        for metric in spec.metrics:
            if metric.source == DataSource.SERVER_METRICS and (
                data.server_metrics_aggregated is None
                or not data.server_metrics_aggregated
            ):
                return False
        return True

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """Create percentile bands plot for server metrics.

        For HISTOGRAM metrics, computes p50/p95/p99 from timeslice bucket data.
        For GAUGE metrics, shows min/avg/max bands.
        """
        y_metric = next((m for m in spec.metrics if m.axis == "y"), None)
        if not y_metric:
            raise ValueError("Percentile bands plot requires a y-axis metric")

        metric_name = y_metric.name

        # Parse metric specification (handle endpoint/label filters)
        metric_name, endpoint_filter, labels_filter = parse_server_metric_spec(
            metric_name
        )

        # Get metric data from aggregated stats
        if metric_name not in data.server_metrics_aggregated:
            available = list(data.server_metrics_aggregated.keys())[:10]
            raise DataUnavailableError(
                f"Metric '{metric_name}' not found in server metrics. "
                f"Available: {available}",
                data_type="server_metrics",
            )

        metric_data = data.server_metrics_aggregated[metric_name]

        # Get first endpoint if not specified
        if endpoint_filter is None:
            endpoint_filter = next(iter(metric_data.keys()))

        if endpoint_filter not in metric_data:
            raise DataUnavailableError(
                f"Endpoint '{endpoint_filter}' not found for metric '{metric_name}'",
                data_type="server_metrics",
            )

        labels_key = (
            orjson.dumps(labels_filter, option=orjson.OPT_SORT_KEYS).decode()
            if labels_filter
            else "{}"
        )

        if labels_key not in metric_data[endpoint_filter]:
            raise DataUnavailableError(
                f"Labels {labels_filter} not found for metric '{metric_name}'",
                data_type="server_metrics",
            )

        series_data = metric_data[endpoint_filter][labels_key]
        metric_type = series_data.get("type", "").upper()
        unit = series_data.get("unit", "")
        timeslices = series_data.get("timeslices")

        if not timeslices:
            raise DataUnavailableError(
                f"No timeslice data available for metric '{metric_name}'. "
                "Percentile bands require timeslice data.",
                data_type="server_metrics",
            )

        # Build DataFrame from timeslices
        rows = []
        for ts in timeslices:
            timestamp_s = (ts.start_ns + ts.end_ns) / 2 / 1e9  # Midpoint
            row = {"timestamp_s": timestamp_s}

            if metric_type == "HISTOGRAM":
                # For histograms: compute percentiles from buckets if available
                if ts.buckets and ts.count > 0:
                    # Use standard Prometheus linear interpolation algorithm
                    try:
                        estimated = compute_prometheus_percentiles(
                            ts.buckets, total_count=ts.count
                        )
                        if estimated.p50_estimate is not None:
                            row["p50"] = estimated.p50_estimate
                            row["p95"] = estimated.p95_estimate
                            row["p99"] = estimated.p99_estimate
                        else:
                            _logger.warning(
                                "Percentile estimation returned None for metric '%s' "
                                "at timestamp %s, falling back to avg",
                                metric_name,
                                timestamp_s,
                            )
                            row["p50"] = ts.avg
                            row["p95"] = ts.avg
                            row["p99"] = ts.avg
                    except Exception as e:
                        _logger.warning(
                            "Failed to compute percentiles for metric '%s' "
                            "at timestamp %s: %r, falling back to avg",
                            metric_name,
                            timestamp_s,
                            e,
                        )
                        row["p50"] = ts.avg
                        row["p95"] = ts.avg
                        row["p99"] = ts.avg
                else:
                    # No buckets - use avg as approximation
                    row["p50"] = ts.avg
                    row["p95"] = ts.avg
                    row["p99"] = ts.avg
            elif metric_type == "GAUGE":
                # For gauges: use min/avg/max as bands
                row["p50"] = ts.avg
                row["p95"] = ts.max
                row["p99"] = ts.max  # Same as p95 for gauges
                row["p05"] = ts.min  # Add lower band
            elif metric_type == "COUNTER":
                # For counters: use rate as single line (no bands)
                row["p50"] = ts.rate
                row["p95"] = ts.rate
                row["p99"] = ts.rate
            else:
                continue

            rows.append(row)

        if not rows:
            raise DataUnavailableError(
                f"No percentile data could be computed for '{metric_name}'",
                data_type="server_metrics",
            )

        df = pd.DataFrame(rows)

        # Create plot using plot generator
        return self.plot_generator.create_percentile_bands(
            df=df,
            x_col="timestamp_s",
            percentile_cols=["p50", "p95", "p99"],
            lower_col="p05" if metric_type == "GAUGE" else None,
            metric_name=metric_name,
            metric_type=metric_type,
            title=spec.title or f"{metric_name} Percentile Bands Over Time",
            x_label=spec.x_label or "Time (s)",
            y_label=spec.y_label
            or (f"{metric_name} ({unit})" if unit else metric_name),
            unit=unit,
        )


@PlotTypeHandlerFactory.register(PlotType.REQUEST_TIMELINE)
class RequestTimelineHandler(BaseSingleRunHandler):
    """Handler for request timeline visualization with phase breakdown."""

    def can_handle(self, spec: PlotSpec, data: RunData) -> bool:
        """
        Check if request timeline plot can be generated.

        Args:
            spec: PlotSpec object
            data: RunData object

        Returns:
            True if required data is available
        """
        if data.requests is None or data.requests.empty:
            return False
        required_cols = ["request_start_ns", "request_end_ns", "time_to_first_token"]
        return all(col in data.requests.columns for col in required_cols)

    def create_plot(
        self, spec: PlotSpec, data: RunData, available_metrics: dict
    ) -> go.Figure:
        """
        Create request timeline plot with TTFT and generation phases.

        Args:
            spec: PlotSpec object
            data: RunData object
            available_metrics: Dictionary with display_names and units

        Returns:
            Plotly Figure object
        """
        if data.requests is None or data.requests.empty:
            raise DataUnavailableError(
                "Request timeline plot cannot be generated: no per-request data available.",
                data_type="requests",
                hint="Per-request data is generated during benchmark runs.",
            )

        required_cols = ["request_start_ns", "request_end_ns", "time_to_first_token"]
        missing_cols = [
            col for col in required_cols if col not in data.requests.columns
        ]
        if missing_cols:
            raise DataUnavailableError(
                f"Request timeline plot cannot be generated: missing columns {missing_cols}.",
                data_type="requests",
                hint="Request timing data may not have been captured during the benchmark.",
            )

        y_metric = next(m for m in spec.metrics if m.axis == "y")

        df = self._prepare_timeline_data(data, y_metric.name)

        if df.empty:
            raise DataUnavailableError(
                f"Request timeline plot cannot be generated: no valid data for {y_metric.name}.",
                data_type="requests",
                hint="After filtering, no valid timeline data remains.",
            )

        y_label = spec.y_label or self._get_axis_label(y_metric, available_metrics)
        x_label = spec.x_label or "Time (seconds)"

        return self.plot_generator.create_request_timeline(
            df=df,
            y_metric=y_metric.name,
            title=spec.title,
            x_label=x_label,
            y_label=y_label,
        )

    def _prepare_timeline_data(self, data: RunData, y_metric: str) -> pd.DataFrame:
        """
        Prepare timeline data with phase calculations.

        Args:
            data: RunData object with requests DataFrame
            y_metric: Name of the metric to plot on Y-axis

        Returns:
            DataFrame with columns: request_id, y_value, start_s, ttft_end_s, end_s
        """
        df = data.requests.copy()

        required_cols = [
            "request_start_ns",
            "request_end_ns",
            "time_to_first_token",
            y_metric,
        ]
        df = df.dropna(subset=required_cols)

        if df.empty:
            return pd.DataFrame()

        start_min = df["request_start_ns"].min()
        df["start_s"] = (df["request_start_ns"] - start_min) / 1e9
        df["end_s"] = (df["request_end_ns"] - start_min) / 1e9

        df["ttft_s"] = df["time_to_first_token"] / 1000.0
        df["ttft_end_s"] = df["start_s"] + df["ttft_s"]

        df["duration_s"] = df["end_s"] - df["start_s"]
        df["has_valid_phases"] = df["ttft_s"] <= df["duration_s"]

        invalid_count = (~df["has_valid_phases"]).sum()
        if invalid_count > 0 and self.logger:
            self.logger.warning(
                f"Filtered {invalid_count} requests where TTFT exceeds total duration"
            )

        df = df[df["has_valid_phases"]]

        df["request_id"] = range(len(df))
        df["y_value"] = df[y_metric]

        return df[["request_id", "y_value", "start_s", "ttft_end_s", "end_s"]]
