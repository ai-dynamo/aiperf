# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from time import perf_counter_ns, time_ns
from typing import Any, ClassVar, Literal

import msgspec
from msgspec import Struct, field
from pydantic import ConfigDict, Field, computed_field

from aiperf.common.models.base_models import AIPerfBaseModel


class TraceDataExport(AIPerfBaseModel):
    """Export model with wall-clock timestamps following k6 and HAR conventions.

    All timestamps are converted from perf_counter to wall-clock time (time.time_ns())
    for correlation with logs, metadata, and cross-system analysis.

    Create from BaseTraceData using trace_data.to_export() method.
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

    request_send_start_ns: int | None = Field(
        default=None, description="Request send start (wall-clock)."
    )
    request_headers: dict[str, str] | None = Field(
        default=None, description="Request headers."
    )
    request_headers_sent_ns: int | None = Field(
        default=None, description="Request headers sent (wall-clock)."
    )
    request_chunks: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Request chunks as (timestamp_ns, size_bytes) tuples.",
    )
    request_send_end_ns: int | None = Field(
        default=None, description="Request send end (wall-clock)."
    )
    request_chunks_count: int = Field(
        default=0, description="Number of request chunks sent."
    )
    request_bytes_total: int = Field(default=0, description="Total request bytes.")

    response_status_code: int | None = Field(
        default=None, description="Response status code."
    )
    response_reason: str | None = Field(
        default=None, description="Response status reason phrase."
    )
    response_receive_start_ns: int | None = Field(
        default=None, description="Response receive start (wall-clock)."
    )
    response_headers: dict[str, str] | None = Field(
        default=None, description="Response headers."
    )
    response_headers_received_ns: int | None = Field(
        default=None, description="Response headers received (wall-clock)."
    )
    response_chunks: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Response chunks as (timestamp_ns, size_bytes) tuples.",
    )
    response_chunks_count: int = Field(
        default=0, description="Number of response chunks received."
    )
    response_bytes_total: int = Field(default=0, description="Total response bytes.")
    response_receive_end_ns: int | None = Field(
        default=None, description="Response receive end (wall-clock)."
    )

    error_timestamp_ns: int | None = Field(
        default=None, description="Error timestamp (wall-clock)."
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sending_ns(self) -> int | None:
        """Request send time (k6: http_req_sending)."""
        if self.request_send_start_ns and self.request_send_end_ns:
            return self.request_send_end_ns - self.request_send_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def waiting_ns(self) -> int | None:
        """TTFB / server processing time (k6: http_req_waiting)."""
        if self.request_send_end_ns and self.response_receive_start_ns:
            return self.response_receive_start_ns - self.request_send_end_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def receiving_ns(self) -> int | None:
        """Response transfer time (k6: http_req_receiving)."""
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
        """Total request duration (k6: http_req_duration)."""
        if self.request_send_start_ns and self.response_receive_end_ns:
            return self.response_receive_end_ns - self.request_send_start_ns
        return None


class AioHttpTraceDataExport(TraceDataExport):
    """Export model for aiohttp with connection-level timing following k6/HAR conventions."""

    trace_type: Literal["aiohttp"] = "aiohttp"

    connection_pool_wait_start_ns: int | None = Field(
        default=None, description="Pool wait start (wall-clock)."
    )
    connection_pool_wait_end_ns: int | None = Field(
        default=None, description="Pool wait end (wall-clock)."
    )
    tcp_connect_start_ns: int | None = Field(
        default=None, description="TCP connect start (wall-clock)."
    )
    tcp_connect_end_ns: int | None = Field(
        default=None, description="TCP connect end (wall-clock)."
    )
    connection_reused_ns: int | None = Field(
        default=None, description="Connection reused (wall-clock)."
    )
    dns_cache_hit_ns: int | None = Field(
        default=None, description="DNS cache hit (wall-clock)."
    )
    dns_cache_miss_ns: int | None = Field(
        default=None, description="DNS cache miss (wall-clock)."
    )
    dns_lookup_start_ns: int | None = Field(
        default=None, description="DNS lookup start (wall-clock)."
    )
    dns_lookup_end_ns: int | None = Field(
        default=None, description="DNS lookup end (wall-clock)."
    )
    local_ip: str | None = Field(default=None, description="Local IP address.")
    local_port: int | None = Field(default=None, description="Local port.")
    remote_ip: str | None = Field(default=None, description="Remote IP address.")
    remote_port: int | None = Field(default=None, description="Remote port.")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def blocked_ns(self) -> int | None:
        """Connection pool wait time (k6: http_req_blocked)."""
        if self.connection_pool_wait_start_ns and self.connection_pool_wait_end_ns:
            return self.connection_pool_wait_end_ns - self.connection_pool_wait_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def dns_lookup_ns(self) -> int | None:
        """DNS lookup time (k6: http_req_looking_up)."""
        if self.dns_lookup_start_ns and self.dns_lookup_end_ns:
            return self.dns_lookup_end_ns - self.dns_lookup_start_ns
        return None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def connecting_ns(self) -> int | None:
        """TCP connection time (k6: http_req_connecting)."""
        if self.tcp_connect_start_ns and self.tcp_connect_end_ns:
            return self.tcp_connect_end_ns - self.tcp_connect_start_ns
        return None


# Re-export mapping from trace_type -> export class, used by to_export().
_EXPORT_LOOKUP: dict[str, type[TraceDataExport]] = {
    "aiohttp": AioHttpTraceDataExport,
}


class BaseTraceData(
    Struct,
    kw_only=True,
    omit_defaults=True,
    tag_field="__struct_tag",
    tag="base",
):
    """Base trace data captured via perf_counter_ns().

    Native msgspec Struct so instances can be embedded directly in msgpack
    wire envelopes without a dict round-trip. Polymorphic decoding is driven
    by the msgspec tag field; the separate ``trace_type`` field is a
    free-form label used by the export layer (TraceDataExport subclasses).
    """

    trace_type: str = "base"
    """Free-form label identifying the trace source (e.g. 'aiohttp', 'httpcore')."""

    reference_time_ns: int | None = None
    """Wall-clock reference for converting perf timestamps (time.time_ns())."""
    reference_perf_ns: int | None = None
    """Perf counter reference paired with reference_time_ns."""

    # Request phase
    request_send_start_perf_ns: int | None = None
    request_headers: dict[str, str] | None = None
    request_headers_sent_perf_ns: int | None = None
    request_chunks: list[tuple[int, int]] = field(default_factory=list)
    request_send_end_perf_ns: int | None = None
    request_chunks_count: int = 0
    request_bytes_total: int = 0

    # Response phase
    response_status_code: int | None = None
    response_reason: str | None = None
    response_receive_start_perf_ns: int | None = None
    response_headers: dict[str, str] | None = None
    response_headers_received_perf_ns: int | None = None
    response_chunks: list[tuple[int, int]] = field(default_factory=list)
    response_chunks_count: int = 0
    response_bytes_total: int = 0
    response_receive_end_perf_ns: int | None = None

    # Errors
    error_timestamp_perf_ns: int | None = None

    def __post_init__(self) -> None:
        """Auto-initialize reference timestamps in a single tight pair for tight coupling."""
        if self.reference_time_ns is None or self.reference_perf_ns is None:
            perf, wall = perf_counter_ns(), time_ns()
            self.reference_perf_ns = perf
            self.reference_time_ns = wall

    def _convert_perf_to_wall(self, perf_ns: int | None) -> int | None:
        """Convert a perf_counter timestamp to wall-clock time using the stored reference."""
        if perf_ns is None:
            return None
        if self.reference_time_ns is None or self.reference_perf_ns is None:
            raise ValueError(
                "Cannot convert without reference timestamps. "
                "Ensure reference_time_ns and reference_perf_ns are set."
            )
        return self.reference_time_ns + (perf_ns - self.reference_perf_ns)

    def model_dump(
        self,
        *,
        exclude_none: bool = False,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-safe dict of the trace data (compat with prior Pydantic API)."""
        data: dict[str, Any] = {}
        for name in self.__struct_fields__:
            value = getattr(self, name)
            if exclude_none and value is None:
                continue
            data[name] = value
        return data

    @classmethod
    def from_json(
        cls, json_or_dict: str | bytes | bytearray | dict[str, Any]
    ) -> BaseTraceData:
        """Rehydrate a trace-data instance from its dict/JSON representation."""
        if isinstance(json_or_dict, bytes | bytearray | str):
            data = msgspec.json.decode(json_or_dict)
        else:
            data = dict(json_or_dict)
        trace_type = data.get("trace_type")
        target: type[BaseTraceData] = (
            AioHttpTraceData if trace_type == "aiohttp" else cls
        )
        # msgspec.convert handles any residual type coercion while accepting plain dicts.
        return msgspec.convert(data, target, strict=False)

    def to_export(self) -> TraceDataExport:
        """Convert to an export model with wall-clock timestamps."""
        export_class = _EXPORT_LOOKUP.get(self.trace_type, TraceDataExport)

        export_data: dict[str, Any] = {}
        for name in self.__struct_fields__:
            if name in ("reference_time_ns", "reference_perf_ns"):
                continue
            value = getattr(self, name)
            if name.endswith("_perf_ns"):
                export_key = name.replace("_perf_ns", "_ns")
                export_data[export_key] = self._convert_perf_to_wall(value)
            elif name in ("request_chunks", "response_chunks"):
                export_data[name] = [
                    (self._convert_perf_to_wall(ts), size) for ts, size in value
                ]
            else:
                export_data[name] = value

        return export_class(**export_data)


class AioHttpTraceData(
    BaseTraceData,
    kw_only=True,
    omit_defaults=True,
    tag="aiohttp",
):
    """Trace data for aiohttp requests extending BaseTraceData with connection-level timing."""

    trace_type: str = "aiohttp"

    # Connection pool
    connection_pool_wait_start_perf_ns: int | None = None
    connection_pool_wait_end_perf_ns: int | None = None

    # TCP connection
    tcp_connect_start_perf_ns: int | None = None
    tcp_connect_end_perf_ns: int | None = None

    # Connection reuse
    connection_reused_perf_ns: int | None = None

    # DNS resolution
    dns_lookup_start_perf_ns: int | None = None
    dns_lookup_end_perf_ns: int | None = None
    dns_cache_hit_perf_ns: int | None = None
    dns_cache_miss_perf_ns: int | None = None

    # Socket info
    local_ip: str | None = None
    local_port: int | None = None
    remote_ip: str | None = None
    remote_port: int | None = None


# Type alias for unions that carry either concrete trace-data struct on the wire.
TraceDataT = BaseTraceData | AioHttpTraceData
