# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from urllib.parse import urlparse

from aiperf.common.aiperf_logger import AIPerfLogger

_logger = AIPerfLogger(__name__)


def compute_histogram_delta(
    start_histogram: dict[str, float], end_histogram: dict[str, float]
) -> dict[str, float]:
    """Compute delta between two cumulative histogram snapshots.

    Prometheus histograms are cumulative counters. To get the delta
    for a time window, subtract the start histogram from the end histogram.

    Args:
        start_histogram: Histogram buckets at start of time window
        end_histogram: Histogram buckets at end of time window

    Returns:
        Delta histogram with bucket-by-bucket differences

    Raises:
        ValueError: If buckets don't match between start and end
    """
    if set(start_histogram.keys()) != set(end_histogram.keys()):
        raise ValueError(
            "Histogram bucket boundaries don't match between start and end"
        )

    delta: dict[str, float] = {}
    for le_str in end_histogram:
        delta_value = end_histogram[le_str] - start_histogram.get(le_str, 0)

        # Detect counter reset (server restart)
        if delta_value < 0:
            _logger.warning(
                f"Histogram counter decreased for bucket '{le_str}': "
                f"{start_histogram.get(le_str, 0)} -> {end_histogram[le_str]}. "
                f"Counter reset detected (likely server restart). Using end value."
            )
            delta[le_str] = end_histogram[le_str]
        else:
            delta[le_str] = delta_value

    return delta


def normalize_metrics_endpoint_url(url: str) -> str:
    """Ensure metrics endpoint URL ends with /metrics suffix.

    Works with Prometheus, DCGM, and other compatible endpoints.
    This utility is used by both TelemetryManager and ServerMetricsManager
    to ensure consistent URL formatting.

    Args:
        url: Base URL or full metrics URL (e.g., "http://localhost:9400" or
             "http://localhost:9400/metrics")

    Returns:
        URL ending with /metrics with trailing slashes removed
        (e.g., "http://localhost:9400/metrics")

    Examples:
        >>> normalize_metrics_endpoint_url("http://localhost:9400")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("http://localhost:9400/")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("http://localhost:9400/metrics")
        "http://localhost:9400/metrics"
    """
    url = url.rstrip("/")
    if not url.endswith("/metrics"):
        url = f"{url}/metrics"
    return url


def build_hostname_aware_prometheus_endpoints(
    inference_endpoint_url: str,
    default_ports: list[int],
) -> list[str]:
    """Build hostname-aware Prometheus/DCGM endpoint URLs based on inference endpoint.

    Extracts hostname and scheme from the inference endpoint URL and generates
    Prometheus-compatible URLs for the specified ports on the same hostname.
    This enables zero-config telemetry for distributed deployments.

    Args:
        inference_endpoint_url: The inference endpoint URL (e.g., http://myserver:8000/v1/chat)
        default_ports: List of ports to check on the same hostname (e.g., [9400, 9401])

    Returns:
        List of Prometheus endpoint URLs with /metrics suffix

    Examples:
        >>> build_hostname_aware_prometheus_endpoints("http://localhost:8000", [9400, 9401])
        ['http://localhost:9400/metrics', 'http://localhost:9401/metrics']
        >>> build_hostname_aware_prometheus_endpoints("http://gpu-server:8000", [8081, 6880])
        ['http://gpu-server:8081/metrics', 'http://gpu-server:6880/metrics']
    """
    parsed = urlparse(inference_endpoint_url)
    hostname = parsed.hostname or "localhost"
    scheme = parsed.scheme or "http"

    endpoints = []
    for port in default_ports:
        endpoint_url = f"{scheme}://{hostname}:{port}/metrics"
        endpoints.append(endpoint_url)

    return endpoints
