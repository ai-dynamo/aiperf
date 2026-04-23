# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request timeline plot handler for single-run data."""

import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.plot_specs import PlotSpec
from aiperf.plot.exceptions import DataUnavailableError
from aiperf.plot.handlers.single_run._base import BaseSingleRunHandler


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
