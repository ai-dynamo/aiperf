# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    "output_token_throughput_per_gpu": "Output Token Throughput Per GPU",  # nosec
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


def get_server_metrics(run_data) -> dict[str, dict[str, str]]:
    """
    Get available server metrics from RunData with metadata.

    Args:
        run_data: RunData object with server_metrics_aggregated

    Returns:
        Dictionary with structure:
        {
            "vllm:kv_cache_usage_perc": {
                "display_name": "KV Cache Usage",
                "unit": "percent",
                "type": "GAUGE",
                "description": "...",
                "endpoints": ["http://localhost:8081/metrics"],
                "has_labels": False
            },
            ...
        }

    Examples:
        >>> run_data = load_run(Path("artifacts/run1"))
        >>> metrics = get_server_metrics(run_data)
        >>> "vllm:kv_cache_usage_perc" in metrics
        True
        >>> metrics["vllm:kv_cache_usage_perc"]["unit"]
        'percent'
    """
    if (
        not hasattr(run_data, "server_metrics_aggregated")
        or not run_data.server_metrics_aggregated
    ):
        return {}

    metrics = {}
    for metric_name, endpoint_data in run_data.server_metrics_aggregated.items():
        # Get first endpoint to extract metadata
        if not endpoint_data:
            continue

        first_endpoint = next(iter(endpoint_data.values()))
        if not first_endpoint:
            continue

        first_series = next(iter(first_endpoint.values()))

        # Determine if metric has multiple endpoints or labels
        endpoints = list(endpoint_data.keys())
        has_labels = any(
            labels_key != "{}"
            for endpoint in endpoint_data.values()
            for labels_key in endpoint
        )

        metrics[metric_name] = {
            "display_name": _format_server_metric_name(metric_name),
            "unit": first_series.get("unit") or "",
            "type": first_series.get("type", ""),
            "description": first_series.get("description", ""),
            "endpoints": endpoints,
            "has_labels": has_labels,
        }

    return metrics


def _format_server_metric_name(metric_name: str) -> str:
    """
    Format server metric name for display with pretty prefix.

    Preserves and prettifies the server/namespace prefix (vLLM, SGLang, Dynamo, etc.)
    to help identify metric source when plotting multiple servers together.
    Strips unit suffixes and converts to title case.

    Args:
        metric_name: Raw metric name (e.g., "vllm:kv_cache_usage_perc")

    Returns:
        Formatted display name (e.g., "vLLM: KV Cache Usage")

    Examples:
        >>> _format_server_metric_name("vllm:kv_cache_usage_perc")
        'vLLM: KV Cache Usage'
        >>> _format_server_metric_name("sglang:time_to_first_token_seconds")
        'SGLang: Time to First Token'
        >>> _format_server_metric_name("nv_inference_request_duration_us")
        'Triton: Inference Request Duration'
        >>> _format_server_metric_name("nv_gpu_utilization")
        'Triton GPU: Utilization'
        >>> _format_server_metric_name("dynamo_frontend_requests_total")
        'Dynamo: Frontend Requests'
        >>> _format_server_metric_name("dynamo_component_requests{component='prefill'}")
        'Dynamo: Component Requests'
        >>> _format_server_metric_name("http_requests_total")
        'HTTP: Requests'
    """
    # Map of prefixes to pretty names (order matters - most specific first!)
    prefix_map = {
        # Inference servers (namespace convention)
        "vllm:": "vLLM: ",
        "sglang:": "SGLang: ",
        "trtllm:": "TensorRT-LLM: ",
        # Dynamo (all components use same prefix, labels differentiate)
        "dynamo_": "Dynamo: ",
        # Triton Inference Server (uses nv_ prefix)
        "nv_inference_": "Triton: ",
        "nv_gpu_": "Triton GPU: ",
        "nv_": "Triton: ",  # Fallback for other nv_ metrics
        # Generic HTTP/HTTPS
        "http_": "HTTP: ",
        "https_": "HTTPS: ",
        # NVIDIA/DCGM (more general)
        "nvidia_": "NVIDIA: ",
        "dcgm_": "DCGM: ",
        "gpu_": "GPU: ",
    }

    # Find and replace prefix
    pretty_prefix = ""
    name = metric_name
    for prefix, pretty in prefix_map.items():
        if name.startswith(prefix):
            pretty_prefix = pretty
            name = name[len(prefix) :]
            break

    # Remove common suffixes (unit-related)
    for suffix in [
        "_seconds",
        "_milliseconds",
        "_microseconds",
        "_nanoseconds",
        "_us",  # Triton uses _us for microseconds
        "_ms",  # Triton uses _ms for milliseconds
        "_ns",
        "_bytes",
        "_total",
        "_count",
        "_perc",
        "_percent",
        "_percentage",
    ]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break

    # Convert to title case
    formatted_name = name.replace("_", " ").title()

    # Combine prefix and name
    return f"{pretty_prefix}{formatted_name}" if pretty_prefix else formatted_name
