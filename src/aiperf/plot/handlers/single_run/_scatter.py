# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scatter plot handler for single-run data."""

import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.plot_specs import DataSource, PlotSpec
from aiperf.plot.exceptions import DataUnavailableError
from aiperf.plot.handlers.single_run._base import BaseSingleRunHandler
from aiperf.plot.utils import (
    filter_server_metrics_dataframe,
    parse_server_metric_spec,
)


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
