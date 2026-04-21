# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msgspec wire envelope for server-metrics records.

The RECORDS channel uses a typed msgspec decoder (see
``aiperf.common.channel_codecs.RECORDS_CODEC``). The wire envelope carries a
native ``ServerMetricsRecord`` msgspec struct directly — no JSON dict
round-tripping.
"""

from __future__ import annotations

from msgspec import Struct

from aiperf.common.enums import MessageType
from aiperf.common.metric_records_wire import (
    WireErrorDetails,
    _error_to_wire,
    _wire_to_error,
)
from aiperf.common.models import ErrorDetails, ServerMetricsRecord


class ServerMetricsRecordWireMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="smr",
):
    """Wire envelope for a single server-metrics record on the RECORDS channel."""

    message_type: MessageType = MessageType.SERVER_METRICS_RECORD
    service_id: str
    collector_id: str
    record: ServerMetricsRecord | None = None
    error: WireErrorDetails | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.record is not None


def build_server_metrics_record_wire_message(
    *,
    service_id: str,
    collector_id: str,
    record: ServerMetricsRecord | None,
    error: ErrorDetails | None = None,
) -> ServerMetricsRecordWireMessage:
    """Build a wire envelope for a server-metrics record."""
    return ServerMetricsRecordWireMessage(
        service_id=service_id,
        collector_id=collector_id,
        record=record,
        error=_error_to_wire(error),
    )


def server_metrics_record_from_wire(
    wire: ServerMetricsRecordWireMessage,
) -> ServerMetricsRecord | None:
    """Return the native server-metrics record from its wire envelope."""
    return wire.record


def server_metrics_error_from_wire(
    wire: ServerMetricsRecordWireMessage,
) -> ErrorDetails | None:
    """Rehydrate the optional ErrorDetails from the wire envelope."""
    return _wire_to_error(wire.error)
