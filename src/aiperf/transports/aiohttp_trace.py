# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""aiohttp TraceConfig factory for request lifecycle timing.

This module provides a factory function to create an aiohttp TraceConfig that captures
timing data throughout the HTTP request lifecycle.

Request Lifecycle (aiperf tracked events)
=========================================

NOTE: The official aiohttp docs diagram is misleading! on_request_end fires BEFORE
response body chunks are read, not after!

::

  on_request_start
  └── [acquire_connection]  (see Connection Acquiring below)
      └── on_request_headers_sent
          └── on_request_chunk_sent  (loops for each request body chunk)
              └── on_request_end  (response headers now received)
                  └── on_response_chunk_received  (loops as body is read)

    Any stage may raise exception → on_request_exception

Connection Acquiring
====================
::

  begin
  ├── on_connection_queued_start  (if pool exhausted, wait in queue)
  │   └── on_connection_queued_end
  │       └── (then either reuse or create below)
  │
  ├── on_connection_reuseconn  (reuse existing connection)
  │
  │
  └── on_connection_create_start  (create new connection)
      ├── [DNS resolution]
      ├── [socket connect]
      └── on_connection_create_end


DNS Resolution
==============

::

  [DNS resolution]
  ├── on_dns_cache_hit  (cached, skip lookup)
  │
  │
  └── on_dns_cache_miss  (not cached, do lookup)
      ├── on_dns_resolvehost_start
      └── on_dns_resolvehost_end

"""

import asyncio
from time import perf_counter_ns, time_ns

import aiohttp

from aiperf.common.models import AioHttpTraceData


def create_aiohttp_trace_config(
    trace_data: AioHttpTraceData,
    on_request_sent_event: asyncio.Event | None = None,
    expected_request_body_size: int | None = None,
) -> aiohttp.TraceConfig:
    """Create a TraceConfig for aiohttp that populates AioHttpTraceData with timestamps.

    Timestamps are captured using `time.perf_counter_ns()` for high-precision
    duration measurements. For HTTPS, TCP timing includes TLS handshake.

    See module docstring for lifecycle diagrams.

    Args:
        trace_data: The AioHttpTraceData instance to populate.
        on_request_sent_event: Optional asyncio.Event to set when the full request
            (headers + body) is sent. Useful for request cancellation timing.
        expected_request_body_size: Expected size of request body in bytes. If provided
            along with on_request_sent_event, the event will be triggered when this
            many bytes have been sent via on_request_chunk_sent.

    Returns:
        A TraceConfig to attach to an aiohttp ClientSession.
    """
    trace_config = aiohttp.TraceConfig()

    # Cache perf_counter_ns function to avoid attribute lookup from global scope
    _perf_counter_ns = perf_counter_ns

    start_perf_ns, start_time_ns = _perf_counter_ns(), time_ns()

    trace_data.reference_time_ns = start_time_ns
    trace_data.reference_perf_ns = start_perf_ns

    # Track bytes sent for body completion detection
    bytes_sent = 0

    # Track whether we're awaiting the first response chunk for efficient start time recording
    awaiting_first_chunk = True

    # Cache list.append method to avoid attribute lookup
    _append_request_chunk = trace_data.request_chunks.append
    _append_response_chunk = trace_data.response_chunks.append

    # ============================================================================
    # CONNECTION POOL EVENTS
    # ============================================================================

    async def on_connection_queued_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceConnectionQueuedStartParams,
    ) -> None:
        """Called when request starts waiting for an available connection from pool."""
        trace_data.connection_pool_wait_start_perf_ns = _perf_counter_ns()

    async def on_connection_queued_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceConnectionQueuedEndParams,
    ) -> None:
        """Called when request obtained an available connection from pool."""
        trace_data.connection_pool_wait_end_perf_ns = _perf_counter_ns()

    # ============================================================================
    # CONNECTION CREATION EVENTS
    # ============================================================================

    async def on_connection_create_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceConnectionCreateStartParams,
    ) -> None:
        """Called when creation of a new connection starts."""
        trace_data.tcp_connect_start_perf_ns = _perf_counter_ns()

    async def on_connection_create_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceConnectionCreateEndParams,
    ) -> None:
        """Called when creation of a new connection completes."""
        trace_data.tcp_connect_end_perf_ns = _perf_counter_ns()

    # ============================================================================
    # CONNECTION REUSE EVENT
    # ============================================================================

    async def on_connection_reuseconn(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceConnectionReuseconnParams,
    ) -> None:
        """Called when an existing connection is reused from pool."""
        trace_data.connection_reused_perf_ns = _perf_counter_ns()

    # ============================================================================
    # DNS RESOLUTION EVENTS
    # ============================================================================

    async def on_dns_resolvehost_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceDnsResolveHostStartParams,
    ) -> None:
        """Called when DNS resolution starts."""
        trace_data.dns_lookup_start_perf_ns = _perf_counter_ns()

    async def on_dns_resolvehost_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceDnsResolveHostEndParams,
    ) -> None:
        """Called when DNS resolution completes."""
        trace_data.dns_lookup_end_perf_ns = _perf_counter_ns()

    async def on_dns_cache_hit(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceDnsCacheHitParams,
    ) -> None:
        """Called when DNS cache hit occurs."""
        trace_data.dns_cache_hit_perf_ns = _perf_counter_ns()

    async def on_dns_cache_miss(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceDnsCacheMissParams,
    ) -> None:
        """Called when DNS cache miss occurs."""
        trace_data.dns_cache_miss_perf_ns = _perf_counter_ns()

    # ============================================================================
    # REQUEST EVENTS
    # ============================================================================

    async def on_request_start(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceRequestStartParams,
    ) -> None:
        """Called when HTTP request starts being sent."""
        trace_data.request_send_start_perf_ns = _perf_counter_ns()

    async def on_request_chunk_sent(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceRequestChunkSentParams,
    ) -> None:
        """Called when each request chunk is sent. Track total bytes sent and trigger event when request body is fully sent."""
        nonlocal bytes_sent
        perf_ns = _perf_counter_ns()
        chunk_size = len(params.chunk)
        _append_request_chunk((perf_ns, chunk_size))
        bytes_sent += chunk_size

        # Trigger event when request body is fully sent
        if (
            on_request_sent_event is not None
            and expected_request_body_size is not None
            and bytes_sent >= expected_request_body_size
        ):
            on_request_sent_event.set()

    async def on_request_end(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceRequestEndParams,
    ) -> None:
        """Called when HTTP request finishes being SENT and the response headers have been received.
        This is NOT the end of the full request/response cycle.


        Note: At this point, response headers have been received, so we capture
        response metadata here (status code and headers).
        """
        trace_data.response_headers_received_perf_ns = _perf_counter_ns()

        # Capture response metadata (status and headers are available at this point)
        trace_data.response_status_code = params.response.status
        trace_data.response_headers = dict(params.response.headers)

        # Capture connection socket info (works for both new and reused connections)
        if (conn := params.response.connection) and (transport := conn.transport):
            if sockname := transport.get_extra_info("sockname"):
                trace_data.local_ip = sockname[0]
                trace_data.local_port = sockname[1]

            if peername := transport.get_extra_info("peername"):
                trace_data.remote_ip = peername[0]
                trace_data.remote_port = peername[1]

    async def on_request_headers_sent(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceRequestHeadersSentParams,
    ) -> None:
        """Called when request headers finish being sent."""
        trace_data.request_headers_sent_perf_ns = _perf_counter_ns()
        trace_data.request_headers = dict(params.headers)

    # ============================================================================
    # RESPONSE EVENTS
    # ============================================================================

    async def on_response_chunk_received(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceResponseChunkReceivedParams,
    ) -> None:
        """Called when each response chunk is received.

        Note: Response metadata (status/headers) is captured in on_request_end,
        not here, since TraceResponseChunkReceivedParams doesn't include the response object.
        """
        nonlocal awaiting_first_chunk
        perf_ns = _perf_counter_ns()
        chunk_size = len(params.chunk)
        _append_response_chunk((perf_ns, chunk_size))

        # Track start/end timestamps for duration calculations
        # Use boolean flag instead of None check for ~6% performance improvement
        if awaiting_first_chunk:
            trace_data.response_receive_start_perf_ns = perf_ns
            awaiting_first_chunk = False
        trace_data.response_receive_end_perf_ns = perf_ns

    # ============================================================================
    # EXCEPTION EVENTS
    # ============================================================================

    async def on_request_exception(
        session: aiohttp.ClientSession,
        trace_config_ctx: aiohttp.tracing.SimpleNamespace,
        params: aiohttp.TraceRequestExceptionParams,
    ) -> None:
        """Called when exception occurs during request."""
        trace_data.error_timestamp_perf_ns = _perf_counter_ns()
        # Set the event on exception so callers waiting for request_sent don't hang
        if on_request_sent_event is not None:
            on_request_sent_event.set()

    # ============================================================================
    # REGISTER ALL CALLBACKS
    # ============================================================================

    # Connection pool events
    trace_config.on_connection_queued_start.append(on_connection_queued_start)
    trace_config.on_connection_queued_end.append(on_connection_queued_end)

    # Connection creation events
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)

    # Connection reuse event
    trace_config.on_connection_reuseconn.append(on_connection_reuseconn)

    # DNS resolution events
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_dns_cache_hit.append(on_dns_cache_hit)
    trace_config.on_dns_cache_miss.append(on_dns_cache_miss)

    # Request events
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_headers_sent.append(on_request_headers_sent)

    # Response events
    trace_config.on_response_chunk_received.append(on_response_chunk_received)

    # Exception events
    trace_config.on_request_exception.append(on_request_exception)

    return trace_config
