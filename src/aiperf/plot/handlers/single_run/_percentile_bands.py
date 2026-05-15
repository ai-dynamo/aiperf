# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Percentile bands plot handler for server metrics."""

import logging

import orjson
import pandas as pd
import plotly.graph_objects as go

from aiperf.plot.core.data_loader import RunData
from aiperf.plot.core.plot_specs import DataSource, PlotSpec
from aiperf.plot.exceptions import DataUnavailableError
from aiperf.plot.handlers.single_run._base import BaseSingleRunHandler
from aiperf.plot.utils import parse_server_metric_spec
from aiperf.server_metrics.histogram_percentiles import compute_prometheus_percentiles

_logger = logging.getLogger(__name__)


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

        metric_name, endpoint_filter, labels_filter = parse_server_metric_spec(
            y_metric.name
        )

        series_data = self._resolve_series_data(
            data, metric_name, endpoint_filter, labels_filter
        )

        metric_type = series_data.get("type", "").upper()
        unit = series_data.get("unit", "")
        timeslices = series_data.get("timeslices")

        if not timeslices:
            raise DataUnavailableError(
                f"No timeslice data available for metric '{metric_name}'. "
                "Percentile bands require timeslice data.",
                data_type="server_metrics",
            )

        rows = self._build_percentile_rows(timeslices, metric_type, metric_name)

        if not rows:
            raise DataUnavailableError(
                f"No percentile data could be computed for '{metric_name}'",
                data_type="server_metrics",
            )

        df = pd.DataFrame(rows)

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

    @staticmethod
    def _resolve_series_data(
        data: RunData,
        metric_name: str,
        endpoint_filter: str | None,
        labels_filter: dict | None,
    ) -> dict:
        """Resolve aggregated series data for metric/endpoint/labels, or raise."""
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

    @classmethod
    def _build_percentile_rows(
        cls, timeslices, metric_type: str, metric_name: str
    ) -> list[dict]:
        """Build percentile row dicts from timeslice entries."""
        rows: list[dict] = []
        for ts in timeslices:
            timestamp_s = (ts.start_ns + ts.end_ns) / 2 / 1e9  # Midpoint
            row = cls._row_for_timeslice(ts, metric_type, metric_name, timestamp_s)
            if row is not None:
                rows.append(row)
        return rows

    @classmethod
    def _row_for_timeslice(
        cls, ts, metric_type: str, metric_name: str, timestamp_s: float
    ) -> dict | None:
        """Return a percentile row for a single timeslice, or None to skip."""
        row = {"timestamp_s": timestamp_s}

        if metric_type == "HISTOGRAM":
            cls._fill_histogram_row(row, ts, metric_name, timestamp_s)
            return row
        if metric_type == "GAUGE":
            row["p50"] = ts.avg
            row["p95"] = ts.max
            row["p99"] = ts.max  # Same as p95 for gauges
            row["p05"] = ts.min  # Add lower band
            return row
        if metric_type == "COUNTER":
            row["p50"] = ts.rate
            row["p95"] = ts.rate
            row["p99"] = ts.rate
            return row
        return None

    @staticmethod
    def _fill_histogram_row(
        row: dict, ts, metric_name: str, timestamp_s: float
    ) -> None:
        """Fill p50/p95/p99 entries on ``row`` for a histogram timeslice."""
        if not (ts.buckets and ts.count > 0):
            # No buckets - use avg as approximation
            row["p50"] = ts.avg
            row["p95"] = ts.avg
            row["p99"] = ts.avg
            return

        try:
            estimated = compute_prometheus_percentiles(ts.buckets, total_count=ts.count)
        except (ValueError, TypeError, ZeroDivisionError) as e:
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
            return

        if estimated.p50_estimate is None:
            _logger.warning(
                "Percentile estimation returned None for metric '%s' "
                "at timestamp %s, falling back to avg",
                metric_name,
                timestamp_s,
            )
            row["p50"] = ts.avg
            row["p95"] = ts.avg
            row["p99"] = ts.avg
            return

        row["p50"] = estimated.p50_estimate
        row["p95"] = estimated.p95_estimate
        row["p99"] = estimated.p99_estimate
