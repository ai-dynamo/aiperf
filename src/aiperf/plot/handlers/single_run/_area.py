# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Area plot handler for single-run data."""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import (
    calculate_throughput_events,
    prepare_request_timeseries,
)
from aiperf.plot.core.plot_specs import DataSource, PlotSpec
from aiperf.plot.exceptions import DataUnavailableError
from aiperf.plot.handlers.single_run._base import BaseSingleRunHandler
from aiperf.plot.utils import (
    detect_server_metric_series,
    filter_server_metrics_dataframe,
    parse_server_metric_spec,
)


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
