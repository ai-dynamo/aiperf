# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Timeslice scatter plot handler for single-run data."""

import orjson
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import (
    prepare_timeslice_metrics,
    validate_request_uniformity,
)
from aiperf.plot.core.plot_specs import (
    DataSource,
    MetricSpec,
    PlotSpec,
    TimeSlicePlotSpec,
)
from aiperf.plot.exceptions import DataUnavailableError
from aiperf.plot.handlers.single_run._base import (
    BaseSingleRunHandler,
    _is_single_stat_metric,
)
from aiperf.plot.metric_names import get_all_metric_display_names
from aiperf.plot.utils import (
    create_series_legend_label,
    detect_server_metric_series,
    filter_server_metrics_dataframe,
    parse_server_metric_spec,
)


def _extract_stat(stats, name: str):
    """Return `stats.name` if attr exists, else dict lookup, else None."""
    if hasattr(stats, name):
        return getattr(stats, name)
    if isinstance(stats, dict):
        return stats.get(name)
    return None


def _format_run_average_label(avg: float, unit: str) -> str:
    """Format the 'Run Average: X.XX unit' overlay label."""
    label = f"Run Average: {avg:.2f}"
    if unit:
        label += f" {unit}"
    return label


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
                spec,
                data,
                available_metrics=available_metrics,
                x_metric=x_metric,
                y_metric=y_metric,
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

        return avg, _format_run_average_label(avg, unit), std

    def _create_server_metrics_plot(
        self,
        spec: PlotSpec,
        data: RunData,
        *,
        available_metrics: dict,
        x_metric: MetricSpec,
        y_metric: MetricSpec,
    ) -> go.Figure:
        """Create a server metrics time series plot.

        Raises DataUnavailableError if server metrics data is not available.
        """
        if data.server_metrics is None or data.server_metrics.empty:
            raise DataUnavailableError(
                "Server metrics plot cannot be generated: no server metrics data available.",
                data_type="server_metrics",
                hint="Server metrics data requires server_metrics collection to be enabled.",
            )

        metric_name, endpoint_filter, labels_filter = parse_server_metric_spec(
            y_metric.name
        )

        try:
            df, unit, metric_type = filter_server_metrics_dataframe(
                data.server_metrics, metric_name, endpoint_filter, labels_filter
            )
        except ValueError as e:
            raise DataUnavailableError(
                str(e),
                data_type="server_metrics",
            ) from e

        series_list = detect_server_metric_series(df)

        # If multiple series and no explicit filter, create multi-series plot
        if len(series_list) > 1 and endpoint_filter is None and labels_filter is None:
            return self._create_multi_series_server_metrics_plot(
                df,
                spec,
                metric_name=metric_name,
                series_list=series_list,
                unit=unit,
                available_metrics=available_metrics,
                x_metric=x_metric,
            )

        # Single series path
        avg_value, avg_label, avg_std = self._get_server_metric_average(
            data, metric_name, endpoint_filter, labels_filter
        )

        default_y_label = f"{metric_name} ({unit})" if unit else metric_name
        y_label = spec.y_label or default_y_label

        return self.plot_generator.create_timeslice_scatter(
            df=df,
            x_col="timestamp_s",
            y_col="value",
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
        *,
        metric_name: str,
        series_list: list[tuple[str, str]],
        unit: str,
        available_metrics: dict,
        x_metric: MetricSpec,
    ) -> go.Figure:
        """Create server metrics plot with multiple series (one trace per endpoint/label combo)."""
        fig = go.Figure()
        total_series = len(series_list)

        # Extract all labels for smart filtering
        all_series_labels = [
            orjson.loads(labels_json.encode()) if labels_json != "{}" else {}
            for _, labels_json in series_list
        ]

        for endpoint_url, labels_json in series_list:
            self._add_series_trace(
                fig,
                df,
                metric_name=metric_name,
                endpoint_url=endpoint_url,
                labels_json=labels_json,
                total_series=total_series,
                all_series_labels=all_series_labels,
            )

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

    @staticmethod
    def _add_series_trace(
        fig: go.Figure,
        df,
        *,
        metric_name: str,
        endpoint_url: str,
        labels_json: str,
        total_series: int,
        all_series_labels: list[dict],
    ) -> None:
        """Append a single endpoint/label series trace to ``fig``."""
        series_df = df[
            (df["endpoint_url"] == endpoint_url) & (df["labels_json"] == labels_json)
        ].copy()

        if series_df.empty:
            return

        series_df = series_df.sort_values("timestamp_ns")

        trace_name = create_series_legend_label(
            metric_name,
            endpoint_url=endpoint_url,
            labels_json=labels_json,
            total_series=total_series,
            all_series_labels=all_series_labels,
        )

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

    def _get_server_metric_average(
        self,
        data: RunData,
        metric_name: str,
        endpoint: str | None,
        labels: dict | None,
    ) -> tuple[float | None, str | None, float | None]:
        """Get average value and std for a server metric from aggregated stats."""
        series_data = self._lookup_series_data(data, metric_name, endpoint, labels)
        if series_data is None:
            return None, None, None

        stats = series_data.get("stats")
        unit = series_data.get("unit", "")

        if stats is None:
            # Static value (no variation)
            return None, None, None

        avg = _extract_stat(stats, "avg")
        std = _extract_stat(stats, "std")

        if avg is None:
            return None, None, None

        return avg, _format_run_average_label(avg, unit), std

    def _lookup_series_data(
        self,
        data: RunData,
        metric_name: str,
        endpoint: str | None,
        labels: dict | None,
    ) -> dict | None:
        """Resolve the aggregated-series dict for metric/endpoint/labels, or None."""
        if not data.server_metrics_aggregated:
            if self.logger:
                self.logger.debug(
                    "Server metrics aggregated stats not available. "
                    "Average line will not be displayed. "
                    "Ensure server_metrics_export.json exists alongside Parquet file."
                )
            return None

        if metric_name not in data.server_metrics_aggregated:
            if self.logger:
                available = list(data.server_metrics_aggregated.keys())[:5]
                self.logger.debug(
                    f"Metric '{metric_name}' not found in aggregated stats. "
                    f"Available metrics: {available}..."
                )
            return None

        metric_data = data.server_metrics_aggregated[metric_name]

        # If no endpoint specified, use first available endpoint
        if endpoint is None:
            if not metric_data:
                return None
            endpoint = next(iter(metric_data.keys()))

        if endpoint not in metric_data:
            return None

        labels_key = (
            orjson.dumps(labels, option=orjson.OPT_SORT_KEYS).decode()
            if labels
            else "{}"
        )

        if labels_key not in metric_data[endpoint]:
            return None

        return metric_data[endpoint][labels_key]
