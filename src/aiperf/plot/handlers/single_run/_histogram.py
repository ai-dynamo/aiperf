# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Histogram plot handler for single-run data."""

import orjson
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import (
    prepare_timeslice_metrics,
    validate_request_uniformity,
)
from aiperf.plot.core.plot_specs import (
    DataSource,
    PlotSpec,
    TimeSlicePlotSpec,
)
from aiperf.plot.exceptions import DataUnavailableError
from aiperf.plot.handlers.single_run._base import (
    BaseSingleRunHandler,
    _is_single_stat_metric,
)
from aiperf.plot.metric_names import get_all_metric_display_names
from aiperf.plot.utils import parse_server_metric_spec


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
        """Get average value and std for a timeslice metric from aggregated stats."""
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
        """Create Prometheus histogram bucket distribution visualization."""
        metric_name, endpoint_filter, labels_filter = parse_server_metric_spec(
            y_metric.name
        )

        series_data = self._resolve_histogram_series(
            data, metric_name, endpoint_filter, labels_filter
        )
        buckets, unit = self._extract_histogram_buckets(series_data, metric_name)

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

    @staticmethod
    def _resolve_histogram_series(
        data: RunData,
        metric_name: str,
        endpoint_filter: str | None,
        labels_filter: dict | None,
    ) -> dict:
        """Look up aggregated series dict for a histogram metric.

        Raises DataUnavailableError when the metric/endpoint/labels are missing.
        """
        if metric_name not in data.server_metrics_aggregated:
            available = list(data.server_metrics_aggregated.keys())[:10]
            raise DataUnavailableError(
                f"Metric '{metric_name}' not found in server metrics. "
                f"Available: {available}",
                data_type="server_metrics",
            )

        metric_data = data.server_metrics_aggregated[metric_name]

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

        return metric_data[endpoint_filter][labels_key]

    @staticmethod
    def _extract_histogram_buckets(series_data: dict, metric_name: str):
        """Return ``(buckets, unit)`` from a histogram series, raising when missing."""
        metric_type = series_data.get("type", "").upper()
        unit = series_data.get("unit", "")

        if metric_type != "HISTOGRAM":
            raise DataUnavailableError(
                f"Metric '{metric_name}' is type {metric_type}, not HISTOGRAM. "
                "Bucket distribution visualization requires HISTOGRAM metrics.",
                data_type="server_metrics",
            )

        stats = series_data.get("stats")
        if not stats:
            raise DataUnavailableError(
                f"No statistics available for metric '{metric_name}'",
                data_type="server_metrics",
            )

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

        return buckets, unit
