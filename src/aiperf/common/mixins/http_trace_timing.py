# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HTTP trace timing and fetch-result dataclasses for metrics collectors.

Split out from ``base_metrics_collector_mixin`` to keep individual modules
under the ergonomics file-size limit. These dataclasses are re-exported from
``aiperf.common.mixins`` for backward compatibility.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class HttpTraceTiming:
    """Timing data captured from aiohttp TraceConfig for HTTP request lifecycle.

    Captures precise timestamps at key points in the HTTP request lifecycle using
    aiohttp's trace hooks. Combines wall clock (time.time_ns) and monotonic
    (time.perf_counter_ns) timestamps to enable both absolute timing and accurate
    duration measurements.

    The dual timestamp approach handles clock adjustments:
    - start_ns: Wall clock for absolute correlation with other system events
    - start_perf_ns/first_byte_perf_ns/end_perf_ns: Monotonic for accurate durations

    This enables accurate correlation between:
    - Client request timestamps (when requests were sent)
    - Server metric timestamps (when server generated the metrics)
    - Request latencies (how long requests took)

    Example:
        >>> # Captured automatically by aiohttp TraceConfig
        >>> timing = HttpTraceTiming(
        ...     start_ns=1_700_000_000_000_000_000,
        ...     start_perf_ns=100_000_000_000,
        ...     first_byte_perf_ns=100_050_000_000,  # +50ms
        ...     end_perf_ns=100_100_000_000  # +100ms total
        ... )
        >>> timing.latency_ns
        100_000_000  # 100ms
        >>> timing.first_byte_ns
        1_700_000_000_050_000_000  # Wall clock + 50ms
    """

    start_ns: int | None = None
    """Wall clock timestamp when request headers were sent (time.time_ns)."""

    start_perf_ns: int | None = None
    """Monotonic timestamp when request headers were sent (time.perf_counter_ns)."""

    first_byte_perf_ns: int | None = None
    """Monotonic timestamp when first response byte was received (TTFB)."""

    end_perf_ns: int | None = None
    """Monotonic timestamp when response was fully received."""

    @property
    def first_byte_ns(self) -> int | None:
        """Get wall clock timestamp of first byte received (best proxy for server snapshot time).

        Computes wall clock timestamp by adding TTFB offset to the request start
        wall clock time. This is the most accurate timestamp for when the server
        generated the metrics, as it represents when the server began sending data.

        Returns:
            Wall clock timestamp in nanoseconds (time.time_ns scale), or None if
            timing data is incomplete.
        """
        if any(
            attr is None
            for attr in [self.start_ns, self.start_perf_ns, self.first_byte_perf_ns]
        ):
            return None
        return self.start_ns + (self.first_byte_perf_ns - self.start_perf_ns)

    @property
    def latency_ns(self) -> int | None:
        """Get the total HTTP request latency in nanoseconds.

        Computes latency using monotonic timestamps (perf_counter_ns) to avoid
        issues with system clock adjustments during the request.

        Returns:
            Total latency from request start to response completion in nanoseconds,
            or None if timing data is incomplete.
        """
        if any(
            attr is None
            for attr in [self.start_ns, self.start_perf_ns, self.end_perf_ns]
        ):
            return None
        return self.end_perf_ns - self.start_perf_ns


@dataclass(frozen=True)
class FetchResult:
    """Result of fetching metrics from an HTTP endpoint with timing metadata.

    Encapsulates both the fetched content and timing information in a single
    immutable object. The is_duplicate flag enables efficient handling of
    unchanged metrics (common when scraping faster than server update rate).
    """

    text: str | None
    """Raw metrics text from the HTTP endpoint (Prometheus exposition format)."""

    trace_timing: HttpTraceTiming
    """Precise timing data captured via aiohttp TraceConfig hooks."""

    is_duplicate: bool = False
    """True if response content hash matches previous fetch, indicating unchanged metrics."""
