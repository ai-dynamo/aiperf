# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP trace-based metrics following k6 naming conventions.

These metrics extract timing data from the aiohttp trace system to provide
detailed HTTP request lifecycle measurements compatible with k6 load testing metrics.

Metric names match the computed properties in TraceDataExport/AioHttpTraceDataExport:
    - sending_ns, waiting_ns, receiving_ns, duration_ns (TraceDataExport)
    - blocked_ns, dns_lookup_ns, connecting_ns (AioHttpTraceDataExport)

k6 Reference: https://k6.io/docs/using-k6/metrics/reference/

Metrics (using k6 naming convention with http_req_ prefix):
    http_req_blocked: Time spent waiting for a free TCP connection from the pool
    http_req_connecting: Time spent establishing TCP connection (includes TLS for HTTPS)
    http_req_dns_lookup: Time spent on DNS resolution
    http_req_sending: Time spent sending the request
    http_req_waiting: Time to First Byte (TTFB) - server processing time
    http_req_receiving: Time spent receiving the response
    http_req_duration: Total request duration (sending + waiting + receiving)
    http_req_data_sent: Total bytes sent in the request
    http_req_data_received: Total bytes received in the response
    http_req_connection_reused: Whether the connection was reused from the pool (0 or 1)

Naming Convention:
    All metrics use http_req_ prefix to match k6 naming conventions.
"""

from typing import ClassVar

from aiperf.common.enums import (
    GenericMetricUnit,
    MetricFlags,
    MetricSizeUnit,
    MetricTimeUnit,
)
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import AioHttpTraceData, BaseTraceData, ParsedResponseRecord
from aiperf.metrics import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict


def _require_trace_data(record: ParsedResponseRecord) -> BaseTraceData:
    """Validate that trace data is available on the record and return it."""
    if record.request.trace_data is None:
        raise NoMetricValue("No trace data available on record")
    return record.request.trace_data


def _require_aiohttp_trace_data(record: ParsedResponseRecord) -> AioHttpTraceData:
    """Validate and return AioHttpTraceData from the record."""
    trace = _require_trace_data(record)
    if not isinstance(trace, AioHttpTraceData):
        raise NoMetricValue("Trace data is not AioHttpTraceData")
    return trace


# =============================================================================
# Connection Pool Metrics (AioHttp-specific)
# =============================================================================


class HttpBlockedMetric(BaseRecordMetric[int]):
    """Time spent blocked waiting for a free TCP connection slot from the pool.

    Matches: AioHttpTraceDataExport.blocked_ns computed property
    k6 equivalent: http_req_blocked
    HAR equivalent: blocked

    This metric measures the time a request spent waiting in the connection pool
    queue before a connection became available. High values indicate connection
    pool saturation.

    Formula:
        blocked = connection_pool_wait_end_perf_ns - connection_pool_wait_start_perf_ns

    Note: Returns 0 if no pool wait occurred (connection immediately available).
    Only available for AioHttpTraceData.
    """

    tag = "http_req_blocked"
    header = "HTTP Blocked"
    short_header = "Blocked"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_aiohttp_trace_data(record)

        # Same logic as AioHttpTraceDataExport.blocked_ns
        if trace.connection_pool_wait_start_perf_ns is None:
            return 0  # No pool wait = immediately available

        if trace.connection_pool_wait_end_perf_ns is None:
            raise NoMetricValue("Pool wait started but never completed")

        blocked = (
            trace.connection_pool_wait_end_perf_ns
            - trace.connection_pool_wait_start_perf_ns
        )
        if blocked < 0:
            raise ValueError("Connection pool wait end is before start")
        return blocked


class HttpConnectionReusedMetric(BaseRecordMetric[int]):
    """Whether the HTTP connection was reused from the connection pool.

    Returns 1 if a connection was reused, 0 if a new connection was established.

    This metric helps identify connection reuse patterns and indicates
    whether keep-alive connections are being effectively utilized.

    Only available for AioHttpTraceData.
    """

    tag = "http_req_connection_reused"
    header = "HTTP Connection Reused"
    short_header = "Conn Reused"
    unit = GenericMetricUnit.RATIO
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_aiohttp_trace_data(record)
        return 1 if trace.connection_reused_perf_ns is not None else 0


# =============================================================================
# Connection Establishment Metrics (AioHttp-specific)
# =============================================================================


class HttpConnectingMetric(BaseRecordMetric[int]):
    """Time spent establishing TCP connection to the remote host.

    Matches: AioHttpTraceDataExport.connecting_ns computed property
    k6 equivalent: http_req_connecting
    HAR equivalent: connect

    For HTTPS requests, this includes both TCP connection establishment
    and TLS handshake time (combined measurement from aiohttp).

    Formula:
        connecting = tcp_connect_end_perf_ns - tcp_connect_start_perf_ns

    Note: Returns 0 if connection was reused.
    Only available for AioHttpTraceData.
    """

    tag = "http_req_connecting"
    header = "HTTP Connecting"
    short_header = "Connecting"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_aiohttp_trace_data(record)

        # If connection was reused, no connection time
        if trace.connection_reused_perf_ns is not None:
            return 0

        # Same logic as AioHttpTraceDataExport.connecting_ns
        if trace.tcp_connect_start_perf_ns is None:
            raise NoMetricValue("No TCP connect start timestamp")
        if trace.tcp_connect_end_perf_ns is None:
            raise NoMetricValue("TCP connect started but never completed")

        connecting = trace.tcp_connect_end_perf_ns - trace.tcp_connect_start_perf_ns
        if connecting < 0:
            raise ValueError("TCP connect end is before start")
        return connecting


class HttpDnsLookupMetric(BaseRecordMetric[int]):
    """Time spent on DNS resolution.

    Matches: AioHttpTraceDataExport.dns_lookup_ns computed property
    k6 equivalent: http_req_looking_up
    HAR equivalent: dns

    This metric measures the time spent resolving the hostname to an IP address.

    Formula:
        dns_lookup = dns_lookup_end_perf_ns - dns_lookup_start_perf_ns

    Note: Returns 0 if DNS cache hit or connection was reused.
    Only available for AioHttpTraceData.
    """

    tag = "http_req_dns_lookup"
    header = "HTTP DNS Lookup"
    short_header = "DNS"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_aiohttp_trace_data(record)

        # DNS cache hit - no lookup time
        if trace.dns_cache_hit_perf_ns is not None:
            return 0

        # Connection reused - no DNS lookup needed
        if trace.connection_reused_perf_ns is not None:
            return 0

        # No DNS lookup occurred (e.g., IP address used directly)
        if trace.dns_lookup_start_perf_ns is None:
            return 0

        # Same logic as AioHttpTraceDataExport.dns_lookup_ns
        if trace.dns_lookup_end_perf_ns is None:
            raise NoMetricValue("DNS lookup started but never completed")

        dns_lookup = trace.dns_lookup_end_perf_ns - trace.dns_lookup_start_perf_ns
        if dns_lookup < 0:
            raise ValueError("DNS lookup end is before start")
        return dns_lookup


# =============================================================================
# Request/Response Phase Metrics (Base TraceData)
# =============================================================================


class HttpSendingMetric(BaseRecordMetric[int]):
    """Time spent sending data to the remote host.

    Matches: TraceDataExport.sending_ns computed property
    k6 equivalent: http_req_sending
    HAR equivalent: send

    This metric measures the time from when the request started being sent
    to when the full request (headers + body) was transmitted.

    Formula:
        sending = request_send_end_perf_ns - request_send_start_perf_ns
    """

    tag = "http_req_sending"
    header = "HTTP Sending"
    short_header = "Sending"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_trace_data(record)

        # Same logic as TraceDataExport.sending_ns
        if trace.request_send_start_perf_ns is None:
            raise NoMetricValue("No request send start timestamp")
        if trace.request_send_end_perf_ns is None:
            raise NoMetricValue("Request send started but never completed")

        sending = trace.request_send_end_perf_ns - trace.request_send_start_perf_ns
        if sending < 0:
            raise ValueError("Request send end is before start")
        return sending


class HttpWaitingMetric(BaseRecordMetric[int]):
    """Time to First Byte (TTFB) - time waiting for the server to respond.

    Matches: TraceDataExport.waiting_ns computed property
    k6 equivalent: http_req_waiting (also known as TTFB)
    HAR equivalent: wait

    This metric measures the time from when the request was fully sent
    to when the first byte of the response was received. This represents
    server processing time plus network latency.

    Note that this is not the same as the time to first token (TTFT),
    which is the time from request start to the first valid token received.
    The server may send non-token data first.

    Formula:
        waiting = response_chunks[0][0] - request_send_end_perf_ns
    """

    tag = "http_req_waiting"
    header = "HTTP Waiting (TTFB)"
    short_header = "TTFB"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_trace_data(record)

        # Same logic as TraceDataExport.waiting_ns
        if trace.request_send_end_perf_ns is None:
            raise NoMetricValue("No request send end timestamp")

        if not trace.response_chunks:
            raise NoMetricValue("No response chunks")

        first_response_ts = trace.response_chunks[0][0]
        waiting = first_response_ts - trace.request_send_end_perf_ns
        if waiting < 0:
            raise ValueError("First response is before request send end")
        return waiting


class HttpReceivingMetric(BaseRecordMetric[int]):
    """Time spent receiving response data from the remote host.

    Matches: TraceDataExport.receiving_ns computed property
    k6 equivalent: http_req_receiving
    HAR equivalent: receive

    This metric measures the time from when the first byte of the response
    was received to when the last byte was received.

    Formula:
        receiving = response_chunks[-1][0] - response_chunks[0][0]

    Note: Returns 0 if response was a single chunk.
    """

    tag = "http_req_receiving"
    header = "HTTP Receiving"
    short_header = "Receiving"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_trace_data(record)

        # Same logic as TraceDataExport.receiving_ns
        if not trace.response_chunks:
            raise NoMetricValue("No response chunks")

        # Single chunk - no transfer time
        if len(trace.response_chunks) == 1:
            return 0

        receiving = trace.response_chunks[-1][0] - trace.response_chunks[0][0]
        if receiving < 0:
            raise ValueError("Last response chunk is before first chunk")
        return receiving


# =============================================================================
# Total Duration Metric
# =============================================================================


class HttpDurationMetric(BaseRecordMetric[int]):
    """Total time for the HTTP request.

    Matches: TraceDataExport.duration_ns computed property
    k6 equivalent: http_req_duration
    HAR equivalent: time

    This is the total time from request start to response complete, equal to:
        duration = sending + waiting + receiving

    Formula:
        duration = response_receive_end_perf_ns - request_send_start_perf_ns

    Note: This uses trace-level timestamps for more accurate measurement
    than application-level request latency.
    """

    tag = "http_req_duration"
    header = "HTTP Duration"
    short_header = "HTTP Dur"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_trace_data(record)

        # Same logic as TraceDataExport.duration_ns
        if trace.request_send_start_perf_ns is None:
            raise NoMetricValue("No request send start timestamp")
        if trace.response_receive_end_perf_ns is None:
            raise NoMetricValue("No response receive end timestamp")

        duration = trace.response_receive_end_perf_ns - trace.request_send_start_perf_ns
        if duration < 0:
            raise ValueError("Response end is before request start")
        return duration


# =============================================================================
# Data Size Metrics
# =============================================================================


class HttpDataSentMetric(BaseRecordMetric[int]):
    """Total bytes sent in the HTTP request.

    k6 equivalent: data_sent (per request)

    This metric measures the total bytes written to the transport layer
    during the request (headers + body).
    """

    tag = "http_req_data_sent"
    header = "HTTP Data Sent"
    short_header = "Sent"
    unit = MetricSizeUnit.BYTES
    display_unit = MetricSizeUnit.KILOBYTES
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_trace_data(record)

        if not trace.request_chunks:
            return 0

        return sum(size for _, size in trace.request_chunks)


class HttpDataReceivedMetric(BaseRecordMetric[int]):
    """Total bytes received in the HTTP response.

    k6 equivalent: data_received (per request)

    This metric measures the total bytes read from the transport layer
    during the response (headers + body).
    """

    tag = "http_req_data_received"
    header = "HTTP Data Received"
    short_header = "Received"
    unit = MetricSizeUnit.BYTES
    display_unit = MetricSizeUnit.KILOBYTES
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_trace_data(record)

        if not trace.response_chunks:
            return 0

        return sum(size for _, size in trace.response_chunks)


# =============================================================================
# Request Phase Chunk Count Metrics (Internal/Debugging)
# =============================================================================


class HttpChunksSentMetric(BaseRecordMetric[int]):
    """Number of chunks sent during the request.

    This metric counts the number of transport-level write operations
    during the request. Useful for debugging chunked transfers.
    """

    tag = "http_req_chunks_sent"
    header = "HTTP Chunks Sent"
    short_header = "Chunks Sent"
    unit = GenericMetricUnit.COUNT
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_trace_data(record)
        return len(trace.request_chunks)


class HttpChunksReceivedMetric(BaseRecordMetric[int]):
    """Number of chunks received during the response.

    This metric counts the number of transport-level read operations
    during the response. Useful for debugging chunked/streaming responses.
    """

    tag = "http_req_chunks_received"
    header = "HTTP Chunks Received"
    short_header = "Chunks Recv"
    unit = GenericMetricUnit.COUNT
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        trace = _require_trace_data(record)
        return len(trace.response_chunks)


# =============================================================================
# Combined Connection Overhead Metric
# =============================================================================


class HttpConnectionOverheadMetric(BaseRecordMetric[int]):
    """Total connection overhead time (blocked + dns_lookup + connecting).

    This metric combines all pre-request overhead:
        overhead = blocked + dns_lookup + connecting

    Useful for identifying total connection establishment costs.
    Returns 0 if connection was reused with no pool wait.

    Only available for AioHttpTraceData.
    """

    tag = "http_req_connection_overhead"
    header = "HTTP Connection Overhead"
    short_header = "Conn Overhead"
    unit = MetricTimeUnit.NANOSECONDS
    display_unit = MetricTimeUnit.MILLISECONDS
    flags = MetricFlags.HTTP_TRACE_ONLY | MetricFlags.NO_CONSOLE
    required_metrics: ClassVar[set[str]] = {
        "http_req_blocked",
        "http_req_dns_lookup",
        "http_req_connecting",
    }

    def _parse_record(
        self,
        record: ParsedResponseRecord,
        record_metrics: MetricRecordDict,
    ) -> int:
        blocked = record_metrics.get("http_req_blocked", 0)
        dns_lookup = record_metrics.get("http_req_dns_lookup", 0)
        connecting = record_metrics.get("http_req_connecting", 0)
        return blocked + dns_lookup + connecting
