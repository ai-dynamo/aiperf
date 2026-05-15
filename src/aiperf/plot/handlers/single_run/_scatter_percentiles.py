# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scatter plot with percentile overlays handler for single-run data."""

import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import calculate_rolling_percentiles
from aiperf.plot.core.plot_specs import DataSource, PlotSpec
from aiperf.plot.exceptions import DataUnavailableError
from aiperf.plot.handlers.single_run._base import BaseSingleRunHandler


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
