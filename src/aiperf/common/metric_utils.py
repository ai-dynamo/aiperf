# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


def normalize_metrics_endpoint_url(url: str) -> str:
    """Ensure metrics endpoint URL has a scheme and ends with /metrics suffix.

    Works with Prometheus, DCGM, and other compatible endpoints.
    This utility is used by both TelemetryManager and ServerMetricsManager
    to ensure consistent URL formatting. If the URL does not start with
    "http://" or "https://", "http://" is prepended.

    Args:
        url: Base URL or full metrics URL (e.g., "http://localhost:9400" or
             "localhost:9400/metrics")

    Returns:
        URL with an http/https scheme, ending with /metrics, with trailing
        slashes removed (e.g., "http://localhost:9400/metrics")

    Raises:
        ValueError: If URL is empty or whitespace-only

    Examples:
        >>> normalize_metrics_endpoint_url("http://localhost:9400")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("localhost:9400")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("http://localhost:9400/metrics")
        "http://localhost:9400/metrics"
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty or whitespace-only")

    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    url = url.rstrip("/")
    if not url.endswith("/metrics"):
        url = f"{url}/metrics"
    return url
