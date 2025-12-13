# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metric display name resolution for plotting.

This module provides access to human-readable display names for all metrics
including standard metrics from MetricRegistry, GPU telemetry metrics, and
derived metrics.
"""

from collections.abc import Mapping

from aiperf.common.enums.metric_enums import MetricFlags, MetricType
from aiperf.gpu_telemetry.constants import GPU_TELEMETRY_METRICS_CONFIG
from aiperf.metrics.metric_registry import MetricRegistry

# Pre-compute all metric display names at module load time
_ALL_METRIC_NAMES: dict[str, str] = {
    # Standard metrics from MetricRegistry
    **{
        metric_class.tag: metric_class.header
        for metric_class in MetricRegistry.all_classes()
        if metric_class.header
    },
    # GPU telemetry metrics
    **{
        field_name: display_name
        for display_name, field_name, _ in GPU_TELEMETRY_METRICS_CONFIG
    },
    # Derived metrics calculated during data processing
    "output_token_throughput_per_gpu": "Output Token Throughput Per GPU",
}

# Pre-compute metric lists by data source at module load time
_AGGREGATED_METRICS: list[str] = MetricRegistry.tags_applicable_to(
    MetricFlags.NONE,
    MetricFlags.NONE,
    MetricType.RECORD,
    MetricType.DERIVED,
) + [
    # Add derived metrics calculated during data loading (not in MetricRegistry)
    "output_token_throughput_per_gpu",
]

_REQUEST_METRICS: list[str] = MetricRegistry.tags_applicable_to(
    MetricFlags.NONE,
    MetricFlags.NONE,
    MetricType.RECORD,
) + [
    "request_number",
    "timestamp",
    "timestamp_s",
    "request_start_ns",
    "request_end_ns",
    "throughput_tokens_per_sec",
    "active_requests",
]

_TIMESLICE_METRICS: list[str] = [
    "Timeslice",
    "Time to First Token",
    "Inter Token Latency",
    "Request Throughput",
    "Request Latency",
]

_GPU_METRICS: list[str] = [
    field_name for _, field_name, _ in GPU_TELEMETRY_METRICS_CONFIG
]

_GPU_METRIC_UNITS: dict[str, str] = {
    field_name: unit_enum.info.tag if hasattr(unit_enum, "info") else str(unit_enum)
    for _, field_name, unit_enum in GPU_TELEMETRY_METRICS_CONFIG
}


def get_all_metric_display_names() -> Mapping[str, str]:
    """
    Get display names for all metrics (standard + GPU telemetry + derived).

    Returns:
        Dictionary mapping metric tag/field to display name

    Examples:
        >>> names = get_all_metric_display_names()
        >>> names["time_to_first_token"]
        'Time to First Token'
        >>> names["gpu_power_usage"]
        'GPU Power Usage'
        >>> names["output_token_throughput_per_gpu"]
        'Output Token Throughput Per GPU'
    """
    return dict(_ALL_METRIC_NAMES)


def get_metric_display_name(metric_tag: str) -> str:
    """
    Get display name for a metric tag with fallback to title-cased tag.

    Args:
        metric_tag: The metric identifier (e.g., "time_to_first_token")

    Returns:
        Human-readable display name

    Examples:
        >>> get_metric_display_name("time_to_first_token")
        'Time to First Token'
        >>> get_metric_display_name("unknown_metric")
        'Unknown Metric'
    """
    return _ALL_METRIC_NAMES.get(metric_tag, metric_tag.replace("_", " ").title())


def get_aggregated_metrics() -> list[str]:
    """
    Get metrics available in aggregated statistics (RECORD + DERIVED types).

    These are metrics that have aggregated statistics like avg, min, max, std, and
    percentiles computed across all requests.

    Returns:
        List of metric tags available in aggregated data

    Examples:
        >>> metrics = get_aggregated_metrics()
        >>> 'request_latency' in metrics
        True
        >>> 'time_to_first_token' in metrics
        True
    """
    return _AGGREGATED_METRICS


def get_request_metrics() -> list[str]:
    """
    Get metrics available in per-request data (RECORD type + computed columns).

    These are metrics available in the requests DataFrame, including both metrics
    from MetricRegistry and computed columns added during data preparation.

    Returns:
        List of metric/column names available in requests data

    Examples:
        >>> metrics = get_request_metrics()
        >>> 'request_number' in metrics
        True
        >>> 'timestamp' in metrics
        True
        >>> 'request_latency' in metrics
        True
    """
    return _REQUEST_METRICS


def get_timeslice_metrics() -> list[str]:
    """
    Get display names of metrics available in timeslice data.

    These are the human-readable column names used in the timeslice CSV exports.

    Returns:
        List of display names for timeslice metrics

    Examples:
        >>> metrics = get_timeslice_metrics()
        >>> 'Timeslice' in metrics
        True
        >>> 'Time to First Token' in metrics
        True
    """
    return _TIMESLICE_METRICS


def get_gpu_metrics() -> list[str]:
    """
    Get field names of metrics available in GPU telemetry data.

    These are the field names used in GPU telemetry DataFrames.

    Returns:
        List of GPU telemetry metric field names

    Examples:
        >>> metrics = get_gpu_metrics()
        >>> 'gpu_utilization' in metrics
        True
        >>> 'gpu_memory_used' in metrics
        True
    """
    return _GPU_METRICS


def get_gpu_metric_unit(metric_name: str) -> str | None:
    """
    Get the unit string for a GPU telemetry metric.

    Args:
        metric_name: The GPU metric field name (e.g., "gpu_utilization")

    Returns:
        Unit string (e.g., "%", "W", "°C") or None if not a GPU metric

    Examples:
        >>> get_gpu_metric_unit("gpu_utilization")
        '%'
        >>> get_gpu_metric_unit("gpu_power_usage")
        'W'
        >>> get_gpu_metric_unit("unknown_metric")
        None
    """
    return _GPU_METRIC_UNITS.get(metric_name)


def get_metric_display_name_with_unit(metric_name: str) -> str:
    """
    Get display name for a metric with unit suffix if available.

    Args:
        metric_name: The metric identifier (e.g., "gpu_utilization")

    Returns:
        Human-readable display name with unit (e.g., "GPU Utilization (%)")

    Examples:
        >>> get_metric_display_name_with_unit("gpu_utilization")
        'GPU Utilization (%)'
        >>> get_metric_display_name_with_unit("memory_copy_utilization")
        'Memory Copy Utilization (%)'
        >>> get_metric_display_name_with_unit("request_latency")
        'Request Latency'
    """
    display_name = get_metric_display_name(metric_name)
    unit = get_gpu_metric_unit(metric_name)
    # Heuristic: metrics with "utilization" in the name are percentages
    if not unit and "utilization" in metric_name.lower():
        unit = "%"
    if unit:
        return f"{display_name} ({unit})"
    return display_name
