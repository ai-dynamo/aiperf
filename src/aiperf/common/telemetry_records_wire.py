# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msgspec wire envelope for GPU-telemetry records.

The RECORDS channel is decoded via a typed msgspec decoder (see
``aiperf.common.channel_codecs.RECORDS_CODEC``). ``TelemetryRecord`` is a
native msgspec.Struct, so this envelope carries records directly without any
intermediate dict round-trip or Pydantic rehydration step.
"""

from __future__ import annotations

from msgspec import Struct

from aiperf.common.enums import MessageType
from aiperf.common.metric_records_wire import (
    WireErrorDetails,
    _error_to_wire,
    _wire_to_error,
)
from aiperf.common.models import ErrorDetails, TelemetryRecord


class TelemetryRecordsWireMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="tlr",
):
    """Wire envelope carrying a batch of telemetry records on the RECORDS channel."""

    message_type: MessageType = MessageType.TELEMETRY_RECORDS
    service_id: str
    collector_id: str
    dcgm_url: str
    records: tuple[TelemetryRecord, ...] = ()
    error: WireErrorDetails | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and len(self.records) > 0


def build_telemetry_records_wire_message(
    *,
    service_id: str,
    collector_id: str,
    dcgm_url: str,
    records: list[TelemetryRecord],
    error: ErrorDetails | None = None,
) -> TelemetryRecordsWireMessage:
    """Build a wire envelope for a batch of telemetry records."""
    return TelemetryRecordsWireMessage(
        service_id=service_id,
        collector_id=collector_id,
        dcgm_url=dcgm_url,
        records=tuple(records),
        error=_error_to_wire(error),
    )


def telemetry_records_from_wire(
    wire: TelemetryRecordsWireMessage,
) -> list[TelemetryRecord]:
    """Return the native telemetry record list from its wire envelope."""
    return list(wire.records)


def telemetry_error_from_wire(
    wire: TelemetryRecordsWireMessage,
) -> ErrorDetails | None:
    """Rehydrate the optional ErrorDetails from the wire envelope."""
    return _wire_to_error(wire.error)
