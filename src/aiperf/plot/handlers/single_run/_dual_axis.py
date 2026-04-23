# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dual-axis plot handler for single-run data."""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.data_preparation import (
    aggregate_gpu_telemetry,
    calculate_throughput_events,
    prepare_request_timeseries,
)
from aiperf.plot.core.plot_specs import DataSource, PlotSpec
from aiperf.plot.exceptions import DataUnavailableError
from aiperf.plot.handlers.single_run._base import BaseSingleRunHandler
from aiperf.plot.utils import (
    filter_server_metrics_dataframe,
    parse_server_metric_spec,
)


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
