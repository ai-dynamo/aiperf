# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msgspec wire envelope for server-metrics records.

The RECORDS channel uses a typed msgspec decoder (see
``aiperf.common.channel_codecs.RECORDS_CODEC``). To share that channel the
server-metrics path builds a msgspec-native envelope at push time and
rehydrates the Pydantic ``ServerMetricsRecord`` in the pull handler.

The nested ``ServerMetricsRecord`` model tree (MetricFamily -> MetricSample
with histogram buckets) is deep, so the wire envelope carries the record as a
JSON-safe ``dict`` rather than mirroring every Pydantic type as a separate
msgspec Struct.
"""

from __future__ import annotations

from typing import Any

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
    record: dict[str, Any] | None = None
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
        record=(
            record.model_dump(exclude_none=True, mode="json")
            if record is not None
            else None
        ),
        error=_error_to_wire(error),
    )


def server_metrics_record_from_wire(
    wire: ServerMetricsRecordWireMessage,
) -> ServerMetricsRecord | None:
    """Rehydrate the pydantic ServerMetricsRecord from its wire envelope."""
    if wire.record is None:
        return None
    return ServerMetricsRecord.model_validate(wire.record)


def server_metrics_error_from_wire(
    wire: ServerMetricsRecordWireMessage,
) -> ErrorDetails | None:
    """Rehydrate the optional ErrorDetails from the wire envelope."""
    return _wire_to_error(wire.error)
