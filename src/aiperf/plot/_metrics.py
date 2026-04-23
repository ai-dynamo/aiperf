# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Metric shortcut parsing and data-source inference for plot configuration.

Keeps name/stat parsing, Prometheus-style server-metric detection, and
data-source resolution out of ``config.py`` so that module stays under the
ergonomics line limit.
"""

import difflib
import re

from aiperf.plot.constants import ALL_STAT_KEYS
from aiperf.plot.core.plot_specs import DataSource, MetricSpec
from aiperf.plot.metric_names import (
    get_aggregated_metrics,
    get_gpu_metrics,
    get_request_metrics,
    get_timeslice_metrics,
)

_PROMETHEUS_PREFIXES = (
    "vllm_",
    "sglang_",
    "trtllm_",
    "nv_inference_",  # Triton Inference Server (most specific)
    "nv_gpu_",  # Triton GPU metrics
    "nv_",  # Generic Triton/NVIDIA
    "http_",
    "https_",
    "dynamo_",
    "nvidia_",
    "dcgm_",
    "gpu_",
    "process_",
    "node_",
    "container_",
)

_PROMETHEUS_SUFFIXES = (
    "_total",
    "_count",
    "_sum",
    "_bucket",
    "_seconds",
    "_milliseconds",
    "_microseconds",
    "_us",  # Triton microseconds
    "_ms",  # Triton milliseconds
    "_ns",
    "_bytes",
)


def _detect_invalid_stat_pattern(metric_name: str) -> str | None:
    """
    Detect if metric name has an invalid stat-like suffix pattern.

    Args:
        metric_name: Full metric name

    Returns:
        The invalid stat suffix if detected (e.g., "p67"), None otherwise
    """
    if "_" not in metric_name:
        return None

    _, potential_stat = metric_name.rsplit("_", 1)

    if potential_stat in ["avg", "min", "max", "std"]:
        return None

    if (
        potential_stat.startswith("p")
        and potential_stat[1:].isdigit()
        and potential_stat not in ALL_STAT_KEYS
    ):
        return potential_stat

    return None


def parse_and_validate_metric_name(metric_name: str) -> tuple[str, str | None]:
    """
    Parse and validate metric name format.

    Supports two formats:
    1. {metric_name}_{stat} - e.g., "request_latency_p50"
    2. {metric_name} - e.g., "request_number"

    Args:
        metric_name: Metric shortcut name

    Returns:
        Tuple of (base_metric_name, stat) where stat is None if no suffix

    Raises:
        ValueError: If metric name has invalid stat suffix pattern
    """
    if "_" not in metric_name:
        return (metric_name, None)

    base_name, potential_stat = metric_name.rsplit("_", 1)

    if potential_stat in ALL_STAT_KEYS:
        return (base_name, potential_stat)

    invalid_stat = _detect_invalid_stat_pattern(metric_name)
    if invalid_stat:
        close_matches = difflib.get_close_matches(
            invalid_stat, ALL_STAT_KEYS, n=3, cutoff=0.6
        )

        error_msg = (
            f"Invalid stat suffix '{invalid_stat}' in metric '{metric_name}'.\n\n"
        )
        error_msg += "Valid stat suffixes are:\n"
        error_msg += f"  {', '.join(ALL_STAT_KEYS)}\n"

        if close_matches:
            error_msg += "\nDid you mean one of these?\n"
            for match in close_matches:
                error_msg += f"  - {base_name}_{match}\n"

        raise ValueError(error_msg)

    return (metric_name, None)


def is_server_metric(metric_name: str) -> bool:
    """
    Check if a metric name appears to be a server metric.

    This is a heuristic-based detection used during config parsing to determine
    the data source for metrics. The actual metric data comes from export files,
    so this is only used for automatic source inference in plot specifications.

    Server metrics typically follow Prometheus naming conventions:
    - Contains colon separator (e.g., "vllm:metric_name", "triton:metric")
    - Common prefixes: vllm, triton, http, dynamo, nvidia, nv
    - May include endpoint/label filters: metric[endpoint], metric{labels}

    Note: If you have custom Prometheus metrics that don't match these patterns,
    explicitly set `source: server_metrics` in your plot specification.

    Args:
        metric_name: Metric name to check

    Returns:
        True if likely a server metric, False otherwise
    """
    # Strip endpoint/label filters first
    base_name = re.sub(r"\[.*?\]|\{.*?\}", "", metric_name).strip()

    # Check for Prometheus namespace convention (most reliable indicator)
    # Format: namespace:metric_name (e.g., "vllm:kv_cache_usage")
    if ":" in base_name:
        return True

    # Check for common Prometheus/server metric prefixes
    # Includes standard patterns from vLLM, Triton, HTTP, DCGM, NVIDIA
    if any(base_name.startswith(prefix) for prefix in _PROMETHEUS_PREFIXES):
        return True

    # Check for common Prometheus suffixes (counter/gauge indicators)
    return any(base_name.endswith(suffix) for suffix in _PROMETHEUS_SUFFIXES)


def auto_detect_source(base_name: str, metric_value: str | dict) -> DataSource:
    """Infer the DataSource for a metric name via registered metric catalogs."""
    if base_name in get_aggregated_metrics():
        return DataSource.AGGREGATED
    if base_name in get_request_metrics():
        return DataSource.REQUESTS
    if base_name in get_timeslice_metrics():
        return DataSource.TIMESLICES
    if base_name in get_gpu_metrics():
        return DataSource.GPU_TELEMETRY
    if is_server_metric(base_name):
        # Server metrics (Prometheus-style names like "vllm:kv_cache_usage_perc")
        return DataSource.SERVER_METRICS
    all_known = (
        get_aggregated_metrics()
        + get_request_metrics()
        + get_timeslice_metrics()
        + get_gpu_metrics()
    )
    raise ValueError(
        f"Unknown metric: '{base_name}' (from shortcut '{metric_value}'). "
        f"Known metrics: {all_known}. For server metrics, use Prometheus-style names like 'vllm:metric_name'."
    )


def expand_metric_shortcut(
    metric_value: str | dict,
    axis: str,
    source_override: str | None = None,
    stat_override: str | None = None,
) -> MetricSpec:
    """
    Expand metric shortcut to full MetricSpec using dynamic pattern matching.

    Supports two formats:
    1. Dict format: {"metric": "request_latency", "stat": "avg"}
    2. String format (legacy): "request_latency_avg" or "request_number"

    Args:
        metric_value: Metric as dict with 'metric' and 'stat' keys, or string shortcut
        axis: Axis assignment ("x", "y", "y2")
        source_override: Override data source (for timeslice plots)
        stat_override: Override stat (for timeslice plots)

    Returns:
        MetricSpec object

    Raises:
        ValueError: If metric name or stat is not recognized
    """
    if isinstance(metric_value, dict):
        base_name = metric_value["metric"]
        stat = metric_value.get("stat")
        # Extract source from dict if present (overrides source_override)
        if "source" in metric_value and not source_override:
            source_override = metric_value["source"]
    else:
        base_name, stat = parse_and_validate_metric_name(metric_value)

    # If source is explicitly specified, use it and skip validation
    # This allows users to specify server metrics that don't match heuristic patterns
    if source_override:
        source = DataSource(source_override)
    else:
        source = auto_detect_source(base_name, metric_value)
    if stat_override:
        stat = stat_override

    return MetricSpec(name=base_name, source=source, axis=axis, stat=stat)
