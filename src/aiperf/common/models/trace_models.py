# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from time import perf_counter_ns, time_ns
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, Field, computed_field

from aiperf.common.models.base_models import AIPerfBaseModel


class TraceDataExport(AIPerfBaseModel):
    """Export model with wall-clock timestamps following k6 and HAR conventions.

    All timestamps are converted from perf_counter to wall-clock time (time.time_ns())
    for correlation with logs, metadata, and cross-system analysis.

    Create from BaseTraceData using trace_data.to_export() method.

    Fields match BaseTraceData exactly, but with _perf_ns replaced by _ns (wall-clock).

    Timing Diagram
    ```
    request_send_start ──────────────────────────────────────────────────────►
            │                    │                    │                      │
            │◄── sending_ns ────►│◄── waiting_ns ────►│◄── receiving_ns ────►│
            │                    │              │     │                      │
        request start      request_send_end     first body chunk        last body chunk
                            (last chunk sent)   (response_chunks[0])
                                                │
                                                ├── response_headers_received_ns
    ```
    ```
    Request Lifecycle ──────────────────────────────────────────────────────────────────────────────►
        │              │              │                │                    │                       │
        │◄─ dns_ns ───►│◄ connect_ns ►│◄─ sending_ns ─►│◄──── waiting_ns ──►│◄──── receiving_ns ───►│
        │              │              │                │                 |  │                       │
    dns resolution   TCP+TLS      request send     request_send_end    first body chunk      last body chunk
    (cache miss)    handshake     (last chunk)     (ready for server)  (response starts)     (response complete)
        │              │                                                 │
        └─ dns_cache_hit_ns (skip lookup)                                └── response_headers_received_ns
                        │
                        └─ connection_reused_ns (skip TCP/TLS)
    ```

    k6 vs AIPerf vs HAR: Complete Metrics Equivalence

    Timing Metrics Comparison

    | Phase                | HAR                | k6                       | AIPerf                      |
    |----------------------|--------------------|--------------------------|-----------------------------|
    | Connection Pool Wait | blocked            | http_req_blocked         | http_req_blocked            |
    | DNS Resolution       | dns                | http_req_looking_up      | http_req_dns_lookup         |
    | TCP Handshake        | Part of connect    | http_req_connecting      | Part of http_req_connecting |
    | TLS/SSL Handshake    | ssl (+ in connect) | http_req_tls_handshaking | Part of http_req_connecting |
    | Request Send         | send               | http_req_sending         | http_req_sending            |
    | Server Wait (TTFB)   | wait               | http_req_waiting         | http_req_waiting            |
    | Response Receive     | receive            | http_req_receiving       | http_req_receiving          |
    | Total Duration       | time               | http_req_duration        | http_req_duration           |


    Computed Durations (k6/HAR compatible):
        - request_send_end_ns: True request completion (computed from last chunk)
        - sending_ns: Request send time (k6: http_req_sending, HAR: send)
        - waiting_ns: TTFB to first body byte (k6: http_req_waiting, HAR: wait)
        - receiving_ns: Response transfer time (k6: http_req_receiving, HAR: receive)
        - duration_ns: Total request duration (k6: http_req_duration, HAR: time)

    Note: All timestamps are in wall-clock time (time.time_ns()) for cross-system correlation.
    """

    # For auto-routed-model serialization and deserialization
    discriminator_field: ClassVar[str] = "trace_type"

    trace_type: str = Field(
        ...,
        description="The type of the trace. This is typically the name of the library used "
        "and must match the trace_type of the corresponding trace data model.",
    )

    # Enable computed fields in serialization
    model_config = ConfigDict(use_attribute_docstrings=True)

    # Request Send Phase (matches BaseTraceData field names)
    request_send_start_ns: int | None = Field(
        default=None,
        description="When the HTTP request started being sent (wall-clock time.time_ns()).",
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the request.",
    )
    request_headers_sent_ns: int | None = Field(
        default=None,
        description="When the request headers were sent to the server (wall-clock time.time_ns()).",
    )
    request_chunks: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Request chunks as (timestamp_ns, size_bytes) tuples. "
        "Transport-layer writes, not application messages.",
    )

    # Response Receive Phase (matches BaseTraceData field names)
    response_status_code: int | None = Field(
        default=None,
        description="The status code of the response.",
    )
    response_receive_start_ns: int | None = Field(
        default=None,
        description="When the response started being received from the server (wall-clock time.time_ns()).",
    )
    response_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the response.",
    )
    response_headers_received_ns: int | None = Field(
        default=None,
        description="When the response headers were received from the server (wall-clock time.time_ns()).",
    )
    response_chunks: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Response chunks as (timestamp_ns, size_bytes) tuples. "
        "Transport-layer reads, not application messages.",
    )
    response_receive_end_ns: int | None = Field(
        default=None,
        description="When the response finished being received from the server (wall-clock time.time_ns()).",
    )

    # Error Tracking
    error_timestamp_ns: int | None = Field(
        default=None,
        description="When an exception occurred during the request (wall-clock time.time_ns()).",
    )

    # Computed Properties
    @computed_field  # type: ignore[prop-decorator]
    @property
    def request_send_end_ns(self) -> int | None:
        """When the request body finished being sent (last chunk written to socket).

        Computed from the last request chunk timestamp. This is the true "request send
        complete" time - more accurate than response_headers_received_ns which includes
        network round-trip time.
        """
        return self.request_chunks[-1][0] if self.request_chunks else None

    # Computed Durations (k6/HAR compatible)
    @computed_field  # type: ignore[prop-decorator]
    @property
    def sending_ns(self) -> int | None:
        """Request send time (k6: http_req_sending, HAR: send)."""
        if self.request_send_start_ns and self.request_send_end_ns:
            return self.request_send_end_ns - self.request_send_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def waiting_ns(self) -> int | None:
        """TTFB (body) / server processing time (k6: http_req_waiting, HAR: wait)."""
        if self.request_send_end_ns and self.response_chunks:
            return self.response_chunks[0][0] - self.request_send_end_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def receiving_ns(self) -> int | None:
        """Response transfer time (k6: http_req_receiving, HAR: receive)."""
        if self.response_chunks:
            if len(self.response_chunks) > 1:
                return self.response_chunks[-1][0] - self.response_chunks[0][0]
            return 0  # Single chunk - no transfer time
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_ns(self) -> int | None:
        """Total request duration (k6: http_req_duration, HAR: time)."""
        if self.request_send_start_ns and self.response_receive_end_ns:
            return self.response_receive_end_ns - self.request_send_start_ns
        return None


class AioHttpTraceDataExport(TraceDataExport):
    """Export model for aiohttp with connection-level timing following k6/HAR conventions.

    Extends TraceDataExport with aiohttp-specific connection pool, DNS, TCP, and TLS timing.

    Fields match AioHttpTraceData exactly, but with _perf_ns replaced by _ns (wall-clock).

    Additional Computed Durations (k6/HAR compatible):
      - blocked_ns: Connection pool wait time (k6: http_req_blocked, HAR: blocked)
      - dns_lookup_ns: DNS lookup time (k6: http_req_looking_up, HAR: dns)
      - connecting_ns: TCP connection time including TLS (k6: http_req_connecting, HAR: connect)

    Note: Inherits all fields from TraceDataExport (request, response, error, durations).
          All timestamps are in wall-clock time (time.time_ns()).
    """

    trace_type: Literal["aiohttp"] = "aiohttp"

    # Connection Pool (matches AioHttpTraceData field names)
    connection_pool_wait_start_ns: int | None = Field(
        default=None,
        description="When the request started waiting for an available connection from the pool (wall-clock time.time_ns()).",
    )
    connection_pool_wait_end_ns: int | None = Field(
        default=None,
        description="When an available connection was obtained from the pool (wall-clock time.time_ns()).",
    )

    # TCP Connection (matches AioHttpTraceData field names)
    tcp_connect_start_ns: int | None = Field(
        default=None,
        description="When TCP connection establishment started (wall-clock time.time_ns()).",
    )
    tcp_connect_end_ns: int | None = Field(
        default=None,
        description="When TCP connection establishment completed (wall-clock time.time_ns()).",
    )

    # Connection Reuse (matches AioHttpTraceData field names)
    connection_reused_ns: int | None = Field(
        default=None,
        description="When an existing connection was reused from the pool (wall-clock time.time_ns()).",
    )

    # DNS Resolution (matches AioHttpTraceData field names)
    dns_cache_hit_ns: int | None = Field(
        default=None,
        description="When a DNS cache hit occurred (wall-clock time.time_ns()).",
    )
    dns_cache_miss_ns: int | None = Field(
        default=None,
        description="When a DNS cache miss occurred (wall-clock time.time_ns()).",
    )
    dns_lookup_start_ns: int | None = Field(
        default=None,
        description="When DNS resolution started for the hostname (wall-clock time.time_ns()).",
    )
    dns_lookup_end_ns: int | None = Field(
        default=None,
        description="When DNS resolution completed for the hostname (wall-clock time.time_ns()).",
    )

    # Connection Socket Info (matches AioHttpTraceData field names)
    local_ip: str | None = Field(
        default=None,
        description="Local IP address used for the connection.",
    )
    local_port: int | None = Field(
        default=None,
        description="Local (ephemeral) port used for the connection.",
    )
    remote_ip: str | None = Field(
        default=None,
        description="Remote IP address of the server (resolved from DNS).",
    )
    remote_port: int | None = Field(
        default=None,
        description="Remote port of the server.",
    )

    # Additional Computed Durations
    @computed_field  # type: ignore[prop-decorator]
    @property
    def blocked_ns(self) -> int | None:
        """Connection pool wait time (k6: http_req_blocked, HAR: blocked)."""
        if self.connection_pool_wait_start_ns and self.connection_pool_wait_end_ns:
            return self.connection_pool_wait_end_ns - self.connection_pool_wait_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dns_lookup_ns(self) -> int | None:
        """DNS lookup time (k6: http_req_looking_up, HAR: dns)."""
        if self.dns_lookup_start_ns and self.dns_lookup_end_ns:
            return self.dns_lookup_end_ns - self.dns_lookup_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def connecting_ns(self) -> int | None:
        """TCP connection time including TLS for HTTPS (k6: http_req_connecting, HAR: connect)."""
        if self.tcp_connect_start_ns and self.tcp_connect_end_ns:
            return self.tcp_connect_end_ns - self.tcp_connect_start_ns
        return None


class BaseTraceData(AIPerfBaseModel):
    """Base trace data model.

    Captures timing information for trace data lifecycle using perf_counter_ns().

    Fields organized by phase:

    Reference Timestamps (time synchronization):
      - reference_time_ns: Wall-clock sync point (time.time_ns)
      - reference_perf_ns: Performance counter sync point

    Request Send Phase:
      - request_send_start_perf_ns: Request send start
      - request_headers: Request headers dictionary
      - request_headers_sent_perf_ns: Request headers sent complete
      - request_chunks: List of (timestamp_perf_ns, size_bytes) tuples
      - request_send_end_perf_ns: Request send end (computed from last chunk)

    Response Receive Phase:
      - response_status_code: Response status code
      - response_receive_start_perf_ns: Response receive start (first body chunk)
      - response_headers: Response headers dictionary
      - response_headers_received_perf_ns: Response headers received (aiohttp on_request_end)
      - response_chunks: List of (timestamp_perf_ns, size_bytes) tuples
      - response_receive_end_perf_ns: Response receive end

    Error Tracking:
      - error_timestamp_perf_ns: Error timestamp (if any)

    Note: All *_perf_ns fields use time.perf_counter_ns() for high-precision measurements.
    """

    # For auto-routed-model serialization and deserialization
    discriminator_field: ClassVar[str] = "trace_type"

    trace_type: str = Field(
        ...,
        description="The type of the trace. This is typically the name of the library used",
        frozen=True,
    )

    # Reference Timestamps for converting between wall-clock and perf_counter time.
    reference_time_ns: int | None = Field(
        default=None,
        description="A reference timestamp in wall-clock time for helping with timing calculations (time.time_ns()).",
    )
    reference_perf_ns: int | None = Field(
        default=None,
        description="A reference perf timestamp for helping with timing calculations (time.perf_counter_ns()). "
        "This is the perf_counter_ns value when the reference_time_ns was set.",
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize the reference time_ns and perf_counter_ns timestamps."""
        if self.reference_time_ns is None or self.reference_perf_ns is None:
            # NOTE: perf_counter is slightly faster than time_ns, so we do it first for tighter coupling.
            #       We also do them as a single operation to avoid timing gaps between the two functions.
            self.reference_perf_ns, self.reference_time_ns = (
                perf_counter_ns(),
                time_ns(),
            )

    # Request Send Phase
    request_send_start_perf_ns: int | None = Field(
        default=None,
        description="When the HTTP request started being sent (perf_counter_ns).",
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the request.",
    )
    request_headers_sent_perf_ns: int | None = Field(
        default=None,
        description="When the request headers were sent to the server (perf_counter_ns).",
    )
    request_chunks: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Request chunks as (timestamp_perf_ns, size_bytes) tuples. "
        "Maps to aiohttp's on_request_chunk_sent events. These are transport-layer writes, not application messages.",
    )

    # Response Receive Phase
    response_status_code: int | None = Field(
        default=None,
        description="The status code of the response.",
    )
    response_receive_start_perf_ns: int | None = Field(
        default=None,
        description="When the response started being received from the server (perf_counter_ns).",
    )
    response_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the response.",
    )
    response_headers_received_perf_ns: int | None = Field(
        default=None,
        description="When the response headers were received from the server (perf_counter_ns).",
    )
    response_chunks: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Response chunks as (timestamp_perf_ns, size_bytes) tuples. "
        "Maps to aiohttp's on_response_chunk_received events. These are transport-layer reads, not application messages (SSE events).",
    )
    response_receive_end_perf_ns: int | None = Field(
        default=None,
        description="When the response finished being received from the server (perf_counter_ns).",
    )

    # Errors
    error_timestamp_perf_ns: int | None = Field(
        default=None,
        description="When an exception occurred during the request (perf_counter_ns). "
        "Maps to aiohttp's on_request_exception event.",
    )

    @property
    def request_send_end_perf_ns(self) -> int | None:
        """When the request body finished being sent (last chunk written to socket).

        Computed from the last on_request_chunk_sent timestamp. This is the true
        "request send complete" time - more accurate than response_headers_received_perf_ns
        which fires after response headers are received (includes network round-trip).

        Returns:
            Timestamp of last request chunk sent, or None if no chunks recorded.
        """
        return self.request_chunks[-1][0] if self.request_chunks else None

    def _convert_perf_to_wall(self, perf_ns: int | None) -> int | None:
        """Convert perf_counter timestamp to wall-clock timestamp.

        Args:
            perf_ns: Perf counter timestamp to convert

        Returns:
            Wall-clock timestamp or None if input is None
        """
        if perf_ns is None:
            return None
        if self.reference_time_ns is None or self.reference_perf_ns is None:
            raise ValueError(
                "Cannot convert without reference timestamps. "
                "Ensure reference_time_ns and reference_perf_ns are set."
            )
        return self.reference_time_ns + (perf_ns - self.reference_perf_ns)

    def to_export(self) -> TraceDataExport:
        """Convert to export model with wall-clock timestamps.

        Uses TraceDataExport's _model_lookup_table to auto-detect the correct export class
        based on trace_type (leverages the auto-routed-model infrastructure).

        Returns:
            TraceDataExport (or subclass) with all perf_counter timestamps converted to wall-clock time
        """
        # Auto-detect export class using the trace_type discriminator
        # This leverages the existing auto-routed-model infrastructure
        export_class = TraceDataExport._model_lookup_table.get(
            self.trace_type, TraceDataExport
        )

        # Get all fields from the model
        data = self.model_dump()

        # Convert all *_perf_ns fields to *_ns (wall-clock)
        export_data = {}
        for key, value in data.items():
            if key in ("reference_time_ns", "reference_perf_ns"):
                # Skip reference fields (not needed in export)
                continue
            elif key.endswith("_perf_ns"):
                # Smart conversion: just replace _perf_ns with _ns
                export_key = key.replace("_perf_ns", "_ns")

                # Convert timestamp field
                if isinstance(value, list):
                    # Convert list of timestamps
                    export_data[export_key] = [
                        self._convert_perf_to_wall(ts) for ts in value
                    ]
                else:
                    # Convert single timestamp
                    export_data[export_key] = self._convert_perf_to_wall(value)
            elif key in ("request_chunks", "response_chunks"):
                # Convert (timestamp_perf_ns, size_bytes) tuples - timestamp is first element
                export_data[key] = [
                    (self._convert_perf_to_wall(ts), size) for ts, size in value
                ]
            else:
                # Copy non-timestamp fields as-is
                export_data[key] = value

        return export_class(**export_data)


class AioHttpTraceData(BaseTraceData):
    """Comprehensive trace data for aiohttp requests using the tracing event system.

    Extends BaseTraceData with aiohttp-specific connection and network timing details.

    Additional fields organized by phase:

    Connection Pool Phase:
      - connection_pool_wait_start_perf_ns: Queue wait start for an available connection
      - connection_pool_wait_end_perf_ns: Queue wait end for an available connection

    Connection Reuse:
      - connection_reused_perf_ns: Existing connection reused timestamp

    DNS Resolution Phase:
      - dns_lookup_start_perf_ns: DNS resolution start
      - dns_lookup_end_perf_ns: DNS resolution complete
      - dns_cache_hit_perf_ns: DNS cache hit occurred
      - dns_cache_miss_perf_ns: DNS cache miss occurred

    TCP Connection Phase (only for new connections):
      - tcp_connect_start_perf_ns: TCP connection establishment start
      - tcp_connect_end_perf_ns: TCP connection established
        Note: For HTTPS, this includes both TCP and TLS handshake time combined.

    Note: Inherits all fields from BaseTraceData (reference, request, response, error).
          All timestamps use perf_counter_ns() for high-precision measurements.
    """

    trace_type: str = "aiohttp"

    # Connection Pool
    connection_pool_wait_start_perf_ns: int | None = Field(
        default=None,
        description="When the request started waiting for an available connection from the pool (perf_counter_ns). "
        "Maps to aiohttp's on_connection_queued_start event.",
    )
    connection_pool_wait_end_perf_ns: int | None = Field(
        default=None,
        description="When an available connection was obtained from the pool (perf_counter_ns). "
        "Maps to aiohttp's on_connection_queued_end event.",
    )

    # TCP Connection (only set if a new connection is created)
    tcp_connect_start_perf_ns: int | None = Field(
        default=None,
        description="When TCP connection establishment started (perf_counter_ns). "
        "Maps to aiohttp's on_connection_create_start event.",
    )
    tcp_connect_end_perf_ns: int | None = Field(
        default=None,
        description="When TCP connection establishment completed (perf_counter_ns). "
        "Maps to aiohttp's on_connection_create_end event.",
    )

    # Connection Reuse
    connection_reused_perf_ns: int | None = Field(
        default=None,
        description="When an existing connection was reused from the pool (perf_counter_ns). "
        "Maps to aiohttp's on_connection_reuseconn event.",
    )

    # DNS Resolution
    dns_lookup_start_perf_ns: int | None = Field(
        default=None,
        description="When DNS resolution started for the hostname (perf_counter_ns). "
        "Maps to aiohttp's on_dns_resolvehost_start event.",
    )
    dns_lookup_end_perf_ns: int | None = Field(
        default=None,
        description="When DNS resolution completed for the hostname (perf_counter_ns). "
        "Maps to aiohttp's on_dns_resolvehost_end event.",
    )
    dns_cache_hit_perf_ns: int | None = Field(
        default=None,
        description="When a DNS cache hit occurred (perf_counter_ns). "
        "Maps to aiohttp's on_dns_cache_hit event.",
    )
    dns_cache_miss_perf_ns: int | None = Field(
        default=None,
        description="When a DNS cache miss occurred (perf_counter_ns). "
        "Maps to aiohttp's on_dns_cache_miss event.",
    )

    # Connection Socket Info (captured in on_request_end, works for new + reused)
    local_ip: str | None = Field(
        default=None,
        description="Local IP address used for the connection.",
    )
    local_port: int | None = Field(
        default=None,
        description="Local (ephemeral) port used for the connection.",
    )
    remote_ip: str | None = Field(
        default=None,
        description="Remote IP address of the server (resolved from DNS).",
    )
    remote_port: int | None = Field(
        default=None,
        description="Remote port of the server.",
    )
