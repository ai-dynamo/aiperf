# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server metrics models — re-export facade.

The actual definitions live in two sibling modules:

- ``_server_metrics_records`` — ``TimeRangeFilter`` and the Prometheus record
  types (``MetricSample``, ``MetricFamily``, ``SlimRecord``, ``ServerMetricsRecord``).
- ``_server_metrics_export`` — the unified timeslice/series/stats/metric-data
  dataclasses, their factory functions, legacy aliases, and export container
  types (``ServerMetricsExportData``, ``ServerMetricsResults``, etc.).

This module preserves the original public import surface so callers can continue
using ``from aiperf.common.models.server_metrics_models import ...``.
"""

from aiperf.common.models._server_metrics_export import (
    BaseSeries,
    BaseServerMetricData,
    BaseTimeslice,
    CounterMetricData,
    CounterSeries,
    CounterStats,
    CounterTimeslice,
    GaugeMetricData,
    GaugeSeries,
    GaugeStats,
    GaugeTimeslice,
    HistogramMetricData,
    HistogramSeries,
    HistogramStats,
    HistogramTimeslice,
    ProcessServerMetricsResult,
    ServerMetricData,
    ServerMetricsEndpointInfo,
    ServerMetricsEndpointSummary,
    ServerMetricsExportData,
    ServerMetricsResults,
    ServerMetricsSummary,
    ServerSeries,
    ServerSeriesStats,
    ServerTimeslice,
)
from aiperf.common.models._server_metrics_records import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
    SlimRecord,
    TimeRangeFilter,
)

__all__ = [
    "BaseSeries",
    "BaseServerMetricData",
    "BaseTimeslice",
    "CounterMetricData",
    "CounterSeries",
    "CounterStats",
    "CounterTimeslice",
    "GaugeMetricData",
    "GaugeSeries",
    "GaugeStats",
    "GaugeTimeslice",
    "HistogramMetricData",
    "HistogramSeries",
    "HistogramStats",
    "HistogramTimeslice",
    "MetricFamily",
    "MetricSample",
    "ProcessServerMetricsResult",
    "ServerMetricData",
    "ServerMetricsEndpointInfo",
    "ServerMetricsEndpointSummary",
    "ServerMetricsExportData",
    "ServerMetricsRecord",
    "ServerMetricsResults",
    "ServerMetricsSummary",
    "ServerSeries",
    "ServerSeriesStats",
    "ServerTimeslice",
    "SlimRecord",
    "TimeRangeFilter",
]
