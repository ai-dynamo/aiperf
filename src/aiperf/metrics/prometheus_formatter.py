# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prometheus Exposition Format (PEF) formatter for AIPerf metrics.

Converts MetricResult objects to Prometheus-compatible text format for scraping.

Metric type mapping:
  Summary: Emitted when percentile stats (p1-p99) are present. Includes quantile
           labels and, when available, _sum and _count companion metrics.
  Gauge:   Emitted for scalar stats (avg, min, max, std, current). When no
           percentiles exist, count and sum are also emitted as gauges.

Naming format:
  Summary:  aiperf_{tag}_{unit}              e.g. aiperf_latency_ms
    _sum:   aiperf_{tag}_{unit}_sum          e.g. aiperf_latency_ms_sum
    _count: aiperf_{tag}_{unit}_count        e.g. aiperf_latency_ms_count
  Gauge:    aiperf_{tag}_{stat}_{unit}       e.g. aiperf_latency_avg_ms
  Info:     aiperf_info                      (benchmark metadata, always gauge)

Example output for MetricResult(tag="request_latency", unit="ms",
header="Request Latency", p50=95, p99=195, avg=100, count=500, sum=50000):

  # HELP aiperf_request_latency_ms Request Latency (in ms)
  # TYPE aiperf_request_latency_ms summary
  aiperf_request_latency_ms{quantile="0.5"} 95.0
  aiperf_request_latency_ms{quantile="0.99"} 195.0
  aiperf_request_latency_ms_sum 50000.0
  aiperf_request_latency_ms_count 500
  # HELP aiperf_request_latency_avg_ms Request Latency average (in ms)
  # TYPE aiperf_request_latency_avg_ms gauge
  aiperf_request_latency_avg_ms 100.0

We generate PEF text directly rather than using the prometheus_client library
because MetricResult is a pre-computed snapshot of statistics, not a live
collector. The prometheus_client registry model expects to own metric
lifecycle (create → observe → collect), which doesn't fit our use case of
formatting externally computed results on demand for a /metrics endpoint.
"""

from __future__ import annotations

import math
import re
from typing import Any, NamedTuple

from aiperf import __version__ as aiperf_version
from aiperf.common.models import MetricResult

# Type alias for info labels dict (values can be str or nested dicts e.g. 'config' key)
InfoLabels = dict[str, Any]

# Percentile field names mapped to Prometheus quantile label values
QUANTILE_STATS: dict[str, str] = {
    "p1": "0.01",
    "p5": "0.05",
    "p10": "0.1",
    "p25": "0.25",
    "p50": "0.5",
    "p75": "0.75",
    "p90": "0.9",
    "p95": "0.95",
    "p99": "0.99",
}

# Non-quantile stats exposed as individual gauges
GAUGE_STATS: dict[str, str] = {
    "avg": "average",
    "min": "minimum",
    "max": "maximum",
    "std": "standard deviation",
    "current": "current value",
}

# Mapping of strings to replace in unit display names
_UNIT_REPLACE_MAP = {
    "%": "percent",
    "/": "_per_",
    "tokens_per_sec": "tps",
    "__": "_",
}

# Regex for sanitizing metric names to Prometheus format
_METRIC_NAME_REGEX = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_metric_name(name: str) -> str:
    """Sanitize a metric name for Prometheus compatibility.

    Prometheus metric names must match [a-zA-Z_][a-zA-Z0-9_]*.
    Invalid characters are replaced with underscores.

    Args:
        name: The raw metric name/tag.

    Returns:
        A sanitized metric name valid for Prometheus.
    """
    sanitized = _METRIC_NAME_REGEX.sub("_", name.lower())
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized


def _escape_label_value(value: str) -> str:
    """Escape a label value per Prometheus spec (backslash, double-quote, newline)."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _escape_help_text(text: str) -> str:
    """Escape HELP docstring text per Prometheus spec (backslash, newline)."""
    return text.replace("\\", "\\\\").replace("\n", "\\n")


def _format_value(value: int | float) -> str:
    """Format a numeric value per Prometheus spec (handles NaN, +Inf, -Inf)."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
    return str(value)


def format_labels(labels: InfoLabels) -> str:
    """Format labels dict as Prometheus label string.

    Args:
        labels: Dict of label name to value.

    Returns:
        Formatted label string like {key1="value1",key2="value2"}.
    """
    if not labels:
        return ""
    label_pairs = [
        f'{key}="{_escape_label_value(str(value))}"' for key, value in labels.items()
    ]
    return "{" + ",".join(label_pairs) + "}"


def _format_info_metric(info_labels: InfoLabels) -> str:
    """Format the aiperf_info metric with benchmark metadata.

    Args:
        info_labels: Dict of label name to value for the info metric.

    Returns:
        Prometheus Exposition Format text for the info metric.
    """
    if not info_labels:
        return ""

    labels_str = format_labels({**info_labels, "version": aiperf_version or "unknown"})
    lines = [
        "# HELP aiperf_info AIPerf benchmark information",
        "# TYPE aiperf_info gauge",
        f"aiperf_info{labels_str} 1",
    ]
    return "\n".join(lines) + "\n"


def _sanitize_unit(unit: str) -> str:
    """Apply unit replacements and sanitize for a Prometheus-safe metric name suffix."""
    for old, new in _UNIT_REPLACE_MAP.items():
        unit = unit.replace(old, new)
    return sanitize_metric_name(unit)


def _extract_metric_labels(info_labels: InfoLabels | None) -> InfoLabels:
    """Extract metric-level labels, excluding config and version."""
    return {
        k: v for k, v in (info_labels or {}).items() if k not in ("config", "version")
    }


class MetricContext(NamedTuple):
    """Shared context for formatting a single MetricResult."""

    metric: MetricResult
    base_name: str
    unit_suffix: str
    metric_labels: InfoLabels
    base_labels_str: str

    @property
    def has_count(self) -> bool:
        return self.metric.count is not None

    @property
    def has_sum(self) -> bool:
        return self.metric.sum is not None

    @property
    def has_quantiles(self) -> bool:
        return any(
            getattr(self.metric, attr, None) is not None for attr in QUANTILE_STATS
        )


def _format_summary(ctx: MetricContext) -> list[str]:
    """Format a Summary block with quantile lines, _sum, and _count."""
    summary_name = f"{ctx.base_name}{ctx.unit_suffix}"

    quantile_lines = [
        f"{summary_name}{format_labels({**ctx.metric_labels, 'quantile': quantile})} {_format_value(value)}"
        for stat, quantile in QUANTILE_STATS.items()
        if (value := getattr(ctx.metric, stat, None)) is not None
    ]

    if not quantile_lines:
        return []

    help_text = _escape_help_text(f"{ctx.metric.header} (in {ctx.metric.unit})")
    lines = [
        f"# HELP {summary_name} {help_text}",
        f"# TYPE {summary_name} summary",
        *quantile_lines,
    ]
    if ctx.has_sum:
        sum_value = _format_value(ctx.metric.sum)
        lines.append(f"{summary_name}_sum{ctx.base_labels_str} {sum_value}")
    if ctx.has_count:
        count_value = _format_value(ctx.metric.count)
        lines.append(f"{summary_name}_count{ctx.base_labels_str} {count_value}")
    return lines


def _format_gauge(
    ctx: MetricContext, stat: str, display: str, value: int | float
) -> list[str]:
    """Format a single Gauge block."""
    name = f"{ctx.base_name}_{stat}{ctx.unit_suffix}"
    help_text = _escape_help_text(
        f"{ctx.metric.header} {display} (in {ctx.metric.unit})"
    )
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} gauge",
        f"{name}{ctx.base_labels_str} {_format_value(value)}",
    ]


def _format_gauges(ctx: MetricContext) -> list[str]:
    """Format individual Gauge blocks for scalar stats.

    When quantiles are present, count/sum are emitted as part of the Summary.
    When no quantiles exist, count/sum are emitted here as gauges instead.
    """
    lines: list[str] = []
    for stat, display in GAUGE_STATS.items():
        if (value := getattr(ctx.metric, stat, None)) is not None:
            lines.extend(_format_gauge(ctx, stat, display, value))

    if not ctx.has_quantiles:
        if ctx.has_count:
            lines.extend(
                _format_gauge(ctx, "count", "observation count", ctx.metric.count)
            )
        if ctx.has_sum:
            lines.extend(_format_gauge(ctx, "sum", "observation sum", ctx.metric.sum))

    return lines


def format_as_prometheus(
    metrics: list[MetricResult],
    info_labels: InfoLabels | None = None,
) -> str:
    """Convert MetricResult list to Prometheus Exposition Format text.

    Percentile stats are emitted as a Summary (with quantile labels and _count/_sum suffixes).
    Scalar stats (avg, min, max, std, current) are emitted as individual Gauges.

    Args:
        metrics: List of MetricResult objects from realtime metrics.
        info_labels: Optional dict of labels for the aiperf_info metric.
            Key labels (excluding 'config') are also added to all metrics.

    Returns:
        Prometheus Exposition Format text string.
    """
    lines: list[str] = []

    if info_labels:
        lines.append(_format_info_metric(info_labels))

    metric_labels = _extract_metric_labels(info_labels)
    base_labels_str = format_labels(metric_labels)

    for metric in metrics:
        ctx = MetricContext(
            metric=metric,
            base_name=f"aiperf_{sanitize_metric_name(metric.tag)}",
            unit_suffix=f"_{_sanitize_unit(metric.unit)}" if metric.unit else "",
            metric_labels=metric_labels,
            base_labels_str=base_labels_str,
        )

        if ctx.has_quantiles:
            lines.extend(_format_summary(ctx))

        lines.extend(_format_gauges(ctx))

    return "\n".join(lines) + "\n" if lines else ""
