# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base mixin for async HTTP metrics data collectors.

This mixin provides common functionality for collecting metrics from HTTP endpoints,
used by both GPU telemetry and server metrics systems.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

import aiohttp

from aiperf.common.hooks import background_task, on_init, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.common.models import ErrorDetails
from aiperf.transports.aiohttp_client import create_tcp_connector


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

    Args:
        start_ns: Wall clock timestamp when request headers sent (time.time_ns)
        start_perf_ns: Monotonic timestamp when request headers sent (time.perf_counter_ns)
        first_byte_perf_ns: Monotonic timestamp when first response byte received (TTFB)
        end_perf_ns: Monotonic timestamp when response fully received

    Properties:
        first_byte_ns: Wall clock timestamp of first byte (start_ns + TTFB offset)
        latency_ns: Total request latency in nanoseconds (end - start)

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
    start_perf_ns: int | None = None
    first_byte_perf_ns: int | None = None
    end_perf_ns: int | None = None

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

    Args:
        text: Raw metrics text from HTTP endpoint (Prometheus exposition format)
        trace_timing: Precise timing data captured via aiohttp TraceConfig hooks
        is_duplicate: True if response content hash matches previous fetch,
                     indicating metrics haven't changed. Callers can skip parsing
                     when True to save CPU on repetitive data.
    """

    text: str | None
    trace_timing: HttpTraceTiming
    is_duplicate: bool = False


# Type variables for records returned by collectors
TRecord = TypeVar("TRecord")
TRecordCallback = TypeVar(
    "TRecordCallback", bound=Callable[[list[TRecord], str], Awaitable[None]]
)
TErrorCallback = TypeVar(
    "TErrorCallback", bound=Callable[[ErrorDetails, str], Awaitable[None]]
)


class BaseMetricsCollectorMixin(AIPerfLifecycleMixin, ABC, Generic[TRecord]):
    """Mixin providing async HTTP collection for metrics endpoints.

    Encapsulates the common pattern for periodically fetching metrics from
    HTTP endpoints, parsing them into typed records, and delivering them via
    async callbacks. Provides infrastructure for reliable metrics collection
    with error handling and lifecycle management.

    Features:
        - Managed aiohttp session with trace hooks for precise timing
        - Automatic reachability testing before collection starts
        - Background collection task with configurable interval
        - Response deduplication via content hashing
        - Callback-based delivery decouples collection from processing
        - Graceful error handling with ErrorDetails propagation

    Common patterns implemented:
        - HTTP session lifecycle tied to component lifecycle
        - Reachability testing with HEAD/GET fallback
        - Background collection loop with error recovery
        - Precise HTTP timing capture for correlation analysis

    Used by:
        - DCGMTelemetryCollector (DCGM metrics from GPU monitoring)
        - ServerMetricsDataCollector (Prometheus metrics from inference servers)

    Example:
        >>> class MyCollector(BaseMetricsCollectorMixin[MyRecord]):
        ...     async def _collect_and_process_metrics(self) -> None:
        ...         result = await self._fetch_metrics_text()
        ...         record = self._parse(result.text)
        ...         await self._send_records_via_callback([record])
        ...
        >>> collector = MyCollector(
        ...     endpoint_url="http://localhost:9400/metrics",
        ...     collection_interval=1.0,
        ...     reachability_timeout=5.0,
        ...     record_callback=my_async_callback
        ... )
        >>> await collector.initialize()
        >>> await collector.start()  # Begins background collection
    """

    def __init__(
        self,
        endpoint_url: str,
        collection_interval: float,
        reachability_timeout: float,
        record_callback: TRecordCallback | None = None,
        error_callback: TErrorCallback | None = None,
        **kwargs,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            endpoint_url: URL of the metrics endpoint
            collection_interval: Interval in seconds between collections
            reachability_timeout: Timeout in seconds for reachability checks
            record_callback: Optional callback to receive collected records
            error_callback: Optional callback to receive collection errors
            **kwargs: Additional arguments passed to super().__init__()
        """
        self._endpoint_url = endpoint_url
        self._collection_interval = collection_interval
        self._reachability_timeout = reachability_timeout
        self._record_callback = record_callback
        self._error_callback = error_callback
        self._connector: aiohttp.TCPConnector | None = None
        self._session: aiohttp.ClientSession | None = None
        # Storage for trace timing data (keyed by trace_request_ctx)
        self._trace_timing: dict[object, HttpTraceTiming] = {}
        # Hash of last response for deduplication
        self._last_response_hash: int | None = None
        super().__init__(**kwargs)

    @property
    def endpoint_url(self) -> str:
        """Get the metrics endpoint URL.

        Returns:
            Full URL of the metrics endpoint being scraped (e.g., "http://localhost:9400/metrics")
        """
        return self._endpoint_url

    @property
    def collection_interval(self) -> float:
        """Get the collection interval in seconds.

        Returns:
            Time between metric scrapes in seconds (e.g., 1.0 for 1 second interval)
        """
        return self._collection_interval

    @on_init
    async def _initialize_http_client(self) -> None:
        """Initialize the aiohttp client session with trace config.

        Called automatically during initialization phase.
        Creates an aiohttp ClientSession with appropriate timeout settings.
        Uses connect timeout only (no total timeout) to allow long-running scrapes.
        Configures TraceConfig to capture HTTP timing events for precise correlation.
        Uses create_tcp_connector to apply standard socket settings including IP version.
        """
        timeout = aiohttp.ClientTimeout(
            total=None,  # No total timeout for ongoing scrapes
            connect=self._reachability_timeout,  # Fast connection timeout only
        )
        trace_config = self._create_trace_config()
        self._connector = create_tcp_connector()
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            timeout=timeout,
            trace_configs=[trace_config],
        )

    def _create_trace_config(self) -> aiohttp.TraceConfig:
        """Create TraceConfig for HTTP timing capture.

        Captures:
        - start_ns: When HTTP request headers are sent (time.time_ns)
        - start_perf_ns: When HTTP request headers are sent (perf_counter_ns)
        - first_byte_perf_ns: Time-to-first-byte (TTFB) - best proxy for server snapshot time (perf_counter_ns)
        - end_perf_ns: When response is fully received (perf_counter_ns)

        Returns:
            Configured TraceConfig instance
        """
        trace_config = aiohttp.TraceConfig()
        trace_config.on_request_start.append(self._on_request_start)
        trace_config.on_response_chunk_received.append(self._on_response_chunk_received)
        return trace_config

    async def _on_request_start(
        self,
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceRequestStartParams,
    ) -> None:
        """Capture timestamp when HTTP request headers are sent.

        Called automatically by aiohttp TraceConfig when request begins.
        Captures both wall clock and monotonic timestamps for accurate
        correlation and duration measurement.

        Args:
            session: aiohttp client session (unused but required by trace signature)
            trace_config_ctx: Trace context containing request identifier
            params: Request parameters (unused but required by trace signature)
        """
        ctx = trace_config_ctx.trace_request_ctx
        if ctx is not None:
            start_perf_ns, start_ns = time.perf_counter_ns(), time.time_ns()
            self._trace_timing[ctx] = HttpTraceTiming(
                start_ns=start_ns,
                start_perf_ns=start_perf_ns,
            )

    async def _on_response_chunk_received(
        self,
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceResponseChunkReceivedParams,
    ) -> None:
        """Capture timestamp when first response byte is received (TTFB).

        Called automatically by aiohttp TraceConfig for each chunk received.
        Only captures the first chunk's timestamp (Time To First Byte), which
        is the best proxy for when the server actually generated the metrics.

        TTFB is more accurate than end timestamp because it represents when
        the server began responding, not when the client finished receiving
        all data (which can be delayed by slow networks).

        Args:
            session: aiohttp client session (unused but required by trace signature)
            trace_config_ctx: Trace context containing request identifier
            params: Chunk received parameters (unused but required by trace signature)
        """
        ctx = trace_config_ctx.trace_request_ctx
        if ctx is not None and ctx in self._trace_timing:
            timing = self._trace_timing[ctx]
            # Only capture first byte (first chunk)
            if timing.first_byte_perf_ns is None:
                timing.first_byte_perf_ns = time.perf_counter_ns()

    @on_stop
    async def _cleanup_http_client(self) -> None:
        """Clean up the aiohttp client session and connector.

        Called automatically during shutdown phase.
        """
        if self._session:
            await self._session.close()
            self._session = None
        if self._connector:
            await self._connector.close()
            self._connector = None

    async def is_url_reachable(self) -> bool:
        """Check if metrics endpoint is accessible before starting collection.

        Tests endpoint reachability using a two-phase approach:
        1. HEAD request (lightweight, doesn't fetch content)
        2. GET request fallback if HEAD not supported (some servers disable HEAD)

        Uses existing session if available (during lifecycle), otherwise creates
        a temporary session for pre-initialization testing. This allows reachability
        checks both before and after collector initialization.

        Returns:
            True if endpoint responds with HTTP 200 status, False for any error
            (connection refused, timeout, 4xx/5xx status, etc.)
        """
        if not self._endpoint_url:
            return False

        # Use existing session if available, otherwise create a temporary one
        if self._session:
            return await self._check_reachability_with_session(self._session)
        else:
            # Create a temporary session for reachability check with proper connector
            timeout = aiohttp.ClientTimeout(total=self._reachability_timeout)
            connector = create_tcp_connector()
            try:
                async with aiohttp.ClientSession(
                    connector=connector, timeout=timeout
                ) as temp_session:
                    return await self._check_reachability_with_session(temp_session)
            finally:
                await connector.close()

    async def _check_reachability_with_session(
        self, session: aiohttp.ClientSession
    ) -> bool:
        """Check reachability using a specific session.

        Args:
            session: aiohttp session to use for the check

        Returns:
            True if endpoint is reachable with HTTP 200
        """
        try:
            # Try HEAD first for efficiency
            async with session.head(
                self._endpoint_url, allow_redirects=False
            ) as response:
                if response.status == 200:
                    return True
            # Fall back to GET if HEAD is not supported
            async with session.get(self._endpoint_url) as response:
                return response.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    @background_task(immediate=True, interval=lambda self: self.collection_interval)
    async def _collect_metrics_loop(self) -> None:
        """Background task for collecting metrics at regular intervals.

        Runs continuously during collector's RUNNING state, triggering a metrics
        collection every collection_interval seconds. The @background_task decorator
        handles automatic lifecycle management and cancellation on stop.

        Uses execute_async (fire-and-forget) rather than await to enable:
        - Consistent scrape timing (next scrape starts on schedule, not after prev completes)
        - Concurrent scrapes when collection_interval < scrape latency
        - Non-blocking behavior (slow scrapes don't delay subsequent scrapes)

        This pattern is critical for accurate rate measurements on fast-changing
        metrics where scrape timing jitter would introduce measurement error.
        """
        self.execute_async(self.collect_and_process_metrics())

    async def collect_and_process_metrics(self) -> None:
        """Collect metrics from endpoint with error handling.

        Public wrapper around the abstract _collect_and_process_metrics method
        that subclasses implement. Provides centralized error handling:
        - Catches all exceptions from collection/parsing
        - Forwards errors to error_callback if configured
        - Logs errors if no callback configured
        - Prevents exceptions from crashing the background collection task

        This error handling enables collectors to recover from transient failures
        (network blips, server restarts) and continue collecting on the next interval.
        """
        try:
            await self._collect_and_process_metrics()
        except Exception as e:
            if self._error_callback:
                try:
                    await self._error_callback(
                        ErrorDetails.from_exception(e),
                        self.id,
                    )
                except Exception as callback_error:
                    self.error(f"Failed to send error via callback: {callback_error}")
            else:
                self.error(f"Metrics collection error: {e}")

    @abstractmethod
    async def _collect_and_process_metrics(self) -> None:
        """Collect metrics from endpoint and process them into records.

        Subclasses must implement this to:
        1. Fetch raw metrics data from the endpoint
        2. Parse data into record objects
        3. Send records via callback (if configured)
        """
        pass

    async def _fetch_metrics_text(self) -> FetchResult:
        """Fetch raw metrics text from the HTTP endpoint with trace timing.

        Uses aiohttp TraceConfig to capture precise HTTP timing:
        - start_ns: When request headers were sent
        - start_perf_ns: When request headers were sent (perf_counter_ns)
        - first_byte_perf_ns: Time-to-first-byte (TTFB) - best proxy for server snapshot time (perf_counter_ns)
        - end_perf_ns: When response was fully received (perf_counter_ns)

        Returns:
            FetchResult containing raw metrics text and trace timing data

        Raises:
            RuntimeError: If HTTP session is not initialized
            aiohttp.ClientError: If HTTP request fails
            asyncio.CancelledError: If collector is being stopped or session is closed
        """
        if self.stop_requested:
            raise asyncio.CancelledError

        session = self._session
        if not session:
            raise RuntimeError("HTTP session not initialized")

        trace_ctx = object()

        try:
            if session.closed:
                raise asyncio.CancelledError

            async with session.get(
                self._endpoint_url, trace_request_ctx=trace_ctx
            ) as response:
                response.raise_for_status()
                text = await response.text()

            timing = self._trace_timing.pop(trace_ctx, HttpTraceTiming())
            timing.end_perf_ns = time.perf_counter_ns()

            # Deduplicate using hash of response text
            # NOTE: Python's built-in hash function is not a secure hash, and is not stable, but it is fast
            #       and works for our use case where collectors run on the same process.
            response_hash = hash(text)
            is_duplicate = response_hash == self._last_response_hash
            self._last_response_hash = response_hash
            return FetchResult(
                text=text, trace_timing=timing, is_duplicate=is_duplicate
            )
        except (aiohttp.ClientConnectionError, RuntimeError) as e:
            if self.stop_requested or session.closed:
                raise asyncio.CancelledError from e
            raise
        finally:
            self._trace_timing.pop(trace_ctx, None)

    async def _send_records_via_callback(self, records: list[TRecord]) -> None:
        """Send records to the callback if configured.

        Helper method for subclasses to deliver parsed records. Handles the
        optional callback pattern and logs errors if callback execution fails.

        Silently no-ops if callback is not configured or records list is empty,
        allowing subclasses to call unconditionally.

        Args:
            records: List of typed records to send to the callback
        """
        if records and self._record_callback:
            try:
                await self._record_callback(records, self.id)
            except Exception as e:
                self.error(f"Failed to send records via callback: {e!r}", exc_info=True)
