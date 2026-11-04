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
from dataclasses import dataclass
from time import perf_counter_ns, time_ns
from types import SimpleNamespace

import aiohttp

from aiperf.common.models import AioHttpTraceData
from aiperf.common.redact import redact_headers


@dataclass(slots=True)
class AioHttpRequestTrace:
    """Per-request trace state, carried through aiohttp as ``trace_request_ctx``.

    The handlers below are registered once on a shared ``TraceConfig`` and
    read this object from ``trace_config_ctx.trace_request_ctx``, so a request
    allocates one small state object instead of a ``TraceConfig`` with fifteen
    closures (which aiohttp ties into a reference cycle that workers, running
    with gc disabled, never free).
    """

    trace_data: AioHttpTraceData
    """The AioHttpTraceData instance to populate."""
    on_request_sent_event: asyncio.Event | None = None
    """Set when the full request (headers + body) is sent, or on exception."""
    expected_request_body_size: int | None = None
    """Body size in bytes at which ``on_request_sent_event`` fires."""
    collect_chunks: bool = False
    """Populate per-chunk (timestamp, size) lists, not just the aggregates."""
    bytes_sent: int = 0
    """Request bytes sent so far."""
    awaiting_first_chunk: bool = True
    """True until the first response chunk arrives."""


def start_request_trace(
    trace_data: AioHttpTraceData,
    on_request_sent_event: asyncio.Event | None = None,
    expected_request_body_size: int | None = None,
    collect_chunks: bool = False,
) -> AioHttpRequestTrace:
    """Stamp the reference timestamps on ``trace_data`` and return the request's trace state.

    Timestamps are captured using `time.perf_counter_ns()` for high-precision
    duration measurements. For HTTPS, TCP timing includes TLS handshake.
    See module docstring for lifecycle diagrams.
    """
    start_perf_ns, start_time_ns = perf_counter_ns(), time_ns()
    trace_data.reference_time_ns = start_time_ns
    trace_data.reference_perf_ns = start_perf_ns
    return AioHttpRequestTrace(
        trace_data=trace_data,
        on_request_sent_event=on_request_sent_event,
        expected_request_body_size=expected_request_body_size,
        collect_chunks=collect_chunks,
    )


def _state(trace_config_ctx: SimpleNamespace) -> AioHttpRequestTrace:
    return trace_config_ctx.trace_request_ctx


# ============================================================================
# CONNECTION POOL EVENTS
# ============================================================================


async def on_connection_queued_start(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceConnectionQueuedStartParams,
) -> None:
    """Called when request starts waiting for an available connection from pool."""
    _state(
        trace_config_ctx
    ).trace_data.connection_pool_wait_start_perf_ns = perf_counter_ns()


async def on_connection_queued_end(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceConnectionQueuedEndParams,
) -> None:
    """Called when request gets an available connection from pool."""
    _state(
        trace_config_ctx
    ).trace_data.connection_pool_wait_end_perf_ns = perf_counter_ns()


# ============================================================================
# TCP CONNECTION EVENTS
# ============================================================================


async def on_connection_create_start(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceConnectionCreateStartParams,
) -> None:
    """Called when TCP connection creation starts (includes TLS for HTTPS)."""
    _state(trace_config_ctx).trace_data.tcp_connect_start_perf_ns = perf_counter_ns()


async def on_connection_create_end(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceConnectionCreateEndParams,
) -> None:
    """Called when TCP connection creation ends (includes TLS for HTTPS)."""
    _state(trace_config_ctx).trace_data.tcp_connect_end_perf_ns = perf_counter_ns()


async def on_connection_reuseconn(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceConnectionReuseconnParams,
) -> None:
    """Called when an existing connection is reused from the pool."""
    _state(trace_config_ctx).trace_data.connection_reused_perf_ns = perf_counter_ns()


# ============================================================================
# DNS EVENTS
# ============================================================================


async def on_dns_resolvehost_start(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceDnsResolveHostStartParams,
) -> None:
    """Called when DNS resolution starts."""
    _state(trace_config_ctx).trace_data.dns_lookup_start_perf_ns = perf_counter_ns()


async def on_dns_resolvehost_end(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceDnsResolveHostEndParams,
) -> None:
    """Called when DNS resolution ends."""
    _state(trace_config_ctx).trace_data.dns_lookup_end_perf_ns = perf_counter_ns()


async def on_dns_cache_hit(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceDnsCacheHitParams,
) -> None:
    """Called when DNS cache is hit."""
    _state(trace_config_ctx).trace_data.dns_cache_hit_perf_ns = perf_counter_ns()


async def on_dns_cache_miss(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceDnsCacheMissParams,
) -> None:
    """Called when DNS cache is missed."""
    _state(trace_config_ctx).trace_data.dns_cache_miss_perf_ns = perf_counter_ns()


# ============================================================================
# REQUEST EVENTS
# ============================================================================


async def on_request_start(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceRequestStartParams,
) -> None:
    """Called when HTTP request starts being sent."""
    _state(trace_config_ctx).trace_data.request_send_start_perf_ns = perf_counter_ns()


async def on_request_chunk_sent(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceRequestChunkSentParams,
) -> None:
    """Called when each request chunk is sent. Track total bytes sent and trigger event when request body is fully sent."""
    state = _state(trace_config_ctx)
    trace_data = state.trace_data
    perf_ns = perf_counter_ns()
    chunk_size = len(params.chunk)
    trace_data.request_send_end_perf_ns = perf_ns
    trace_data.request_chunks_count += 1
    trace_data.request_bytes_total += chunk_size
    if state.collect_chunks:
        trace_data.request_chunks.append((perf_ns, chunk_size))
    state.bytes_sent += chunk_size
    # Trigger event when request body is fully sent
    if (
        state.on_request_sent_event is not None
        and state.expected_request_body_size is not None
        and state.bytes_sent >= state.expected_request_body_size
    ):
        state.on_request_sent_event.set()


async def on_request_end(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceRequestEndParams,
) -> None:
    """Called when HTTP request finishes being SENT and the response headers have been received.

    This is NOT the end of the full request/response cycle.

    Note: At this point, response headers have been received, so we capture
    response metadata here (status code and headers).
    """
    trace_data = _state(trace_config_ctx).trace_data
    trace_data.response_headers_received_perf_ns = perf_counter_ns()
    # Capture response metadata (status and headers are available at this point)
    trace_data.response_status_code = params.response.status
    trace_data.response_reason = params.response.reason
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
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceRequestHeadersSentParams,
) -> None:
    """Called when request headers finish being sent."""
    trace_data = _state(trace_config_ctx).trace_data
    trace_data.request_headers_sent_perf_ns = perf_counter_ns()
    trace_data.request_headers = redact_headers(dict(params.headers))


# ============================================================================
# RESPONSE EVENTS
# ============================================================================


async def on_response_chunk_received(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceResponseChunkReceivedParams,
) -> None:
    """Called when each response chunk is received.

    Note: Response metadata (status/headers) is captured in on_request_end,
    not here, since TraceResponseChunkReceivedParams doesn't include the response object.
    """
    state = _state(trace_config_ctx)
    trace_data = state.trace_data
    perf_ns = perf_counter_ns()
    chunk_size = len(params.chunk)
    trace_data.response_chunks_count += 1
    trace_data.response_bytes_total += chunk_size
    if state.collect_chunks:
        trace_data.response_chunks.append((perf_ns, chunk_size))
    # Track start/end timestamps for duration calculations
    if state.awaiting_first_chunk:
        trace_data.response_receive_start_perf_ns = perf_ns
        state.awaiting_first_chunk = False
    trace_data.response_receive_end_perf_ns = perf_ns


# ============================================================================
# EXCEPTION EVENTS
# ============================================================================


async def on_request_exception(
    session: aiohttp.ClientSession,
    trace_config_ctx: SimpleNamespace,
    params: aiohttp.TraceRequestExceptionParams,
) -> None:
    """Called when exception occurs during request."""
    state = _state(trace_config_ctx)
    state.trace_data.error_timestamp_perf_ns = perf_counter_ns()
    # Set the event on exception so callers waiting for request_sent don't hang
    if state.on_request_sent_event is not None:
        state.on_request_sent_event.set()


def create_aiohttp_trace_config() -> aiohttp.TraceConfig:
    """Create the shared TraceConfig; attach it to every ClientSession and pass
    :func:`start_request_trace`'s state as ``trace_request_ctx`` per request."""
    trace_config = aiohttp.TraceConfig()
    trace_config.on_connection_queued_start.append(on_connection_queued_start)
    trace_config.on_connection_queued_end.append(on_connection_queued_end)
    trace_config.on_connection_create_start.append(on_connection_create_start)
    trace_config.on_connection_create_end.append(on_connection_create_end)
    trace_config.on_connection_reuseconn.append(on_connection_reuseconn)
    trace_config.on_dns_resolvehost_start.append(on_dns_resolvehost_start)
    trace_config.on_dns_resolvehost_end.append(on_dns_resolvehost_end)
    trace_config.on_dns_cache_hit.append(on_dns_cache_hit)
    trace_config.on_dns_cache_miss.append(on_dns_cache_miss)
    trace_config.on_request_start.append(on_request_start)
    trace_config.on_request_chunk_sent.append(on_request_chunk_sent)
    trace_config.on_request_end.append(on_request_end)
    trace_config.on_request_headers_sent.append(on_request_headers_sent)
    trace_config.on_response_chunk_received.append(on_response_chunk_received)
    trace_config.on_request_exception.append(on_request_exception)
    return trace_config
