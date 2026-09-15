# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urlsplit, urlunsplit


def normalize_metrics_endpoint_url(
    url: str, *, preserve_explicit_path: bool = False
) -> str:
    """Ensure a metrics endpoint URL has a scheme and a usable path.

    Works with Prometheus, DCGM, and other compatible endpoints.
    This utility is used by both TelemetryManager and ServerMetricsManager
    to ensure consistent URL formatting. If the URL does not start with
    "http://" or "https://", "http://" is prepended.

    Args:
        url: Base URL or full metrics URL (e.g., "http://localhost:9400" or
             "localhost:9400/metrics").
        preserve_explicit_path: Keep a non-root path supplied by the user instead
            of appending ``/metrics``. Root and pathless URLs still receive the
            default suffix.

    Returns:
        URL with an http/https scheme and trailing slashes removed. The URL ends
        with ``/metrics`` unless an explicit path is preserved.

    Raises:
        ValueError: If URL is empty or whitespace-only

    Examples:
        >>> normalize_metrics_endpoint_url("http://localhost:9400")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("localhost:9400")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url("http://localhost:9400/metrics")
        "http://localhost:9400/metrics"
        >>> normalize_metrics_endpoint_url(
        ...     "https://localhost/prometheus", preserve_explicit_path=True
        ... )
        "https://localhost/prometheus"
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty or whitespace-only")

    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    parts = urlsplit(url)
    path = parts.path.rstrip("/")
    if not (preserve_explicit_path and path) and not path.endswith("/metrics"):
        path = f"{path}/metrics"
    return urlunsplit(parts._replace(path=path))
