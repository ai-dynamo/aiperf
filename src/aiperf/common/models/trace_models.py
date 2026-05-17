# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from time import perf_counter_ns, time_ns
from typing import Any, ClassVar, Literal

import msgspec
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
                            (last chunk sent)   (response_receive_start)
                                                │
                                                ├── response_headers_received_ns
    ```
    ```
    Request Lifecycle ──────────────────────────────────────────────────────────────────────────────►
        │              │              │                │                    │                       │
        │◄dns_lookup_ns►│◄connecting_ns►│◄─ sending_ns ─►│◄──── waiting_ns ──►│◄──── receiving_ns ───►│
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
        "Only populated when --export-http-trace is enabled. "
        "Transport-layer writes, not application messages.",
    )
    request_send_end_ns: int | None = Field(
        default=None,
        description="When the request body finished being sent - last chunk written to socket (wall-clock time.time_ns()).",
    )
    request_chunks_count: int = Field(
        default=0,
        ge=0,
        description="Number of request chunks sent.",
    )
    request_bytes_total: int = Field(
        default=0,
        ge=0,
        description="Total bytes sent in request chunks.",
    )

    # Response Receive Phase (matches BaseTraceData field names)
    response_status_code: int | None = Field(
        default=None,
        description="The status code of the response.",
    )
    response_reason: str | None = Field(
        default=None,
        description="The HTTP status reason phrase (e.g., 'OK', 'Not Found').",
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
        "Only populated when --export-http-trace is enabled. "
        "Transport-layer reads, not application messages.",
    )
    response_chunks_count: int = Field(
        default=0,
        ge=0,
        description="Number of response chunks received.",
    )
    response_bytes_total: int = Field(
        default=0,
        ge=0,
        description="Total bytes received in response chunks.",
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
        if self.request_send_end_ns and self.response_receive_start_ns:
            return self.response_receive_start_ns - self.request_send_end_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def receiving_ns(self) -> int | None:
        """Response transfer time (k6: http_req_receiving, HAR: receive)."""
        if self.response_chunks_count == 0:
            return None
        if self.response_chunks_count == 1:
            return 0
        if self.response_receive_start_ns and self.response_receive_end_ns:
            return self.response_receive_end_ns - self.response_receive_start_ns
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
        ge=1,
        le=65535,
        description="Local (ephemeral) port used for the connection.",
    )
    remote_ip: str | None = Field(
        default=None,
        description="Remote IP address of the server (resolved from DNS).",
    )
    remote_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
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


class BaseTraceData(msgspec.Struct, kw_only=True, tag_field="trace_type", tag="base"):
    """Base trace data model — msgspec.Struct with tag-discriminated dispatch.

    Captures timing information for trace data lifecycle using perf_counter_ns().

    Subclasses set their own ``tag="..."`` (e.g. ``AioHttpTraceData`` uses
    ``tag="aiohttp"``); ``msgspec.json.Decoder(BaseTraceDataUnion)`` dispatches
    to the right subtype based on the ``trace_type`` field present on the
    JSON wire. The ``__post_init__`` hook seeds reference timestamps so the
    perf-to-wall conversion in ``to_export()`` works without callers
    pre-populating them.

    Fields organized by phase:

    Reference Timestamps (time synchronization):
      - reference_time_ns: Wall-clock sync point (time.time_ns)
      - reference_perf_ns: Performance counter sync point

    Request Send Phase:
      - request_send_start_perf_ns: Request send start
      - request_headers: Request headers dictionary
      - request_headers_sent_perf_ns: Request headers sent complete
      - request_chunks: List of (timestamp_perf_ns, size_bytes) tuples (only with --export-http-trace)
      - request_send_end_perf_ns: Request send end (last chunk written to socket)
      - request_chunks_count: Number of request chunks sent
      - request_bytes_total: Total bytes sent

    Response Receive Phase:
      - response_status_code: Response status code
      - response_receive_start_perf_ns: Response receive start (first body chunk)
      - response_headers: Response headers dictionary
      - response_headers_received_perf_ns: Response headers received (aiohttp on_request_end)
      - response_chunks: List of (timestamp_perf_ns, size_bytes) tuples (only with --export-http-trace)
      - response_chunks_count: Number of response chunks received
      - response_bytes_total: Total bytes received
      - response_receive_end_perf_ns: Response receive end

    Error Tracking:
      - error_timestamp_perf_ns: Error timestamp (if any)

    Note: All *_perf_ns fields use time.perf_counter_ns() for high-precision measurements.
    """

    # Reference Timestamps for converting between wall-clock and perf_counter time.
    reference_time_ns: int | None = None
    reference_perf_ns: int | None = None

    # Request Send Phase
    request_send_start_perf_ns: int | None = None
    request_headers: dict[str, str] | None = None
    request_headers_sent_perf_ns: int | None = None
    request_chunks: list[tuple[int, int]] = msgspec.field(default_factory=list)
    request_send_end_perf_ns: int | None = None
    request_chunks_count: int = 0
    request_bytes_total: int = 0

    # Response Receive Phase
    response_status_code: int | None = None
    response_reason: str | None = None
    response_receive_start_perf_ns: int | None = None
    response_headers: dict[str, str] | None = None
    response_headers_received_perf_ns: int | None = None
    response_chunks: list[tuple[int, int]] = msgspec.field(default_factory=list)
    response_chunks_count: int = 0
    response_bytes_total: int = 0
    response_receive_end_perf_ns: int | None = None

    # Errors
    error_timestamp_perf_ns: int | None = None

    def __post_init__(self) -> None:
        """Initialize the reference time_ns and perf_counter_ns timestamps."""
        if self.reference_time_ns is None or self.reference_perf_ns is None:
            # NOTE: perf_counter is slightly faster than time_ns, so we do it first for tighter coupling.
            #       We also do them as a single operation to avoid timing gaps between the two functions.
            self.reference_perf_ns, self.reference_time_ns = (
                perf_counter_ns(),
                time_ns(),
            )

    @property
    def trace_type(self) -> str:
        """Discriminator value used by the msgspec tagged union.

        Mirrors the previous Pydantic ``trace_type`` field for code that reads
        the discriminator off an instance. The underlying value is supplied by
        the msgspec ``tag`` on the Struct class.
        """
        return type(self).__struct_config__.tag  # type: ignore[no-any-return]

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

        # Get all fields from the Struct as a plain dict via msgspec.
        data = msgspec.to_builtins(self)
        # msgspec.to_builtins drops the tag field by default — re-add it so
        # downstream code that inspects ``trace_type`` still sees it.
        data["trace_type"] = self.trace_type

        # Convert all *_perf_ns fields to *_ns (wall-clock)
        export_data: dict[str, Any] = {}
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


class AioHttpTraceData(
    BaseTraceData, kw_only=True, tag_field="trace_type", tag="aiohttp"
):
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

    # Connection Pool
    connection_pool_wait_start_perf_ns: int | None = None
    connection_pool_wait_end_perf_ns: int | None = None

    # TCP Connection (only set if a new connection is created)
    tcp_connect_start_perf_ns: int | None = None
    tcp_connect_end_perf_ns: int | None = None

    # Connection Reuse
    connection_reused_perf_ns: int | None = None

    # DNS Resolution
    dns_lookup_start_perf_ns: int | None = None
    dns_lookup_end_perf_ns: int | None = None
    dns_cache_hit_perf_ns: int | None = None
    dns_cache_miss_perf_ns: int | None = None

    # Connection Socket Info (captured in on_request_end, works for new + reused)
    local_ip: str | None = None
    local_port: int | None = None
    remote_ip: str | None = None
    remote_port: int | None = None


# Tagged-union alias for ``BaseTraceData`` and its concrete subclasses.
# msgspec's ``Decoder(BaseTraceDataUnion)`` dispatches by the top-level
# ``trace_type`` field on the JSON wire. Defined at module load before any
# decoder is built so the dispatch is locked in before the records pipeline
# starts.
BaseTraceDataUnion = BaseTraceData | AioHttpTraceData

_TRACE_DATA_ENCODER = msgspec.json.Encoder()
_TRACE_DATA_DECODER = msgspec.json.Decoder(BaseTraceDataUnion)


def _trace_data_to_jsonable(value: BaseTraceDataUnion) -> dict[str, Any]:
    """Encode a ``BaseTraceDataUnion`` member to a plain JSON-safe dict.

    Used by Pydantic embedding sites whose ``trace_data`` field holds a
    msgspec.Struct value Pydantic itself cannot natively serialize.
    """
    import orjson

    return orjson.loads(_TRACE_DATA_ENCODER.encode(value))


def _trace_data_from_value(value: Any) -> BaseTraceDataUnion | None:
    """Decode a ``BaseTraceDataUnion`` member from a Pydantic-side value.

    Accepts already-typed Struct instances (Python-side construction), plain
    dicts (JSON wire load), bytes (raw wire bytes), and ``None``.
    """
    import orjson

    if value is None or isinstance(value, BaseTraceData):
        return value
    if isinstance(value, dict):
        return _TRACE_DATA_DECODER.decode(orjson.dumps(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _TRACE_DATA_DECODER.decode(bytes(value))
    raise TypeError(
        f"BaseTraceData must be a BaseTraceData Struct, dict, bytes, or None; "
        f"got {type(value).__name__}"
    )
