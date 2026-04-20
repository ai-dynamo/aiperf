# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""msgspec wire envelope for GPU-telemetry records.

The RECORDS channel is decoded via a typed msgspec decoder (see
``aiperf.common.channel_codecs.RECORDS_CODEC``). To share that channel the
telemetry path builds a msgspec-native wire struct at push time and rehydrates
the Pydantic ``TelemetryRecord`` instances in the pull handler.
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
from aiperf.common.models.telemetry_models import TelemetryMetrics


class TelemetryRecordWireData(Struct, frozen=True, kw_only=True, omit_defaults=True):
    """msgspec-native mirror of ``TelemetryRecord`` fields for wire transport."""

    gpu_index: int
    gpu_uuid: str
    gpu_model_name: str
    timestamp_ns: int
    dcgm_url: str
    telemetry_data: dict[str, float]
    pci_bus_id: str | None = None
    device: str | None = None
    hostname: str | None = None
    namespace: str | None = None
    pod_name: str | None = None


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
    records: tuple[TelemetryRecordWireData, ...] = ()
    error: WireErrorDetails | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and len(self.records) > 0


def _record_to_wire(record: TelemetryRecord) -> TelemetryRecordWireData:
    data = record.telemetry_data.model_dump(exclude_none=True, mode="json")
    return TelemetryRecordWireData(
        gpu_index=record.gpu_index,
        gpu_uuid=record.gpu_uuid,
        gpu_model_name=record.gpu_model_name,
        pci_bus_id=record.pci_bus_id,
        device=record.device,
        hostname=record.hostname,
        namespace=record.namespace,
        pod_name=record.pod_name,
        timestamp_ns=record.timestamp_ns,
        dcgm_url=record.dcgm_url,
        telemetry_data={k: float(v) for k, v in data.items() if v is not None},
    )


def _wire_to_record(wire: TelemetryRecordWireData) -> TelemetryRecord:
    return TelemetryRecord(
        gpu_index=wire.gpu_index,
        gpu_uuid=wire.gpu_uuid,
        gpu_model_name=wire.gpu_model_name,
        pci_bus_id=wire.pci_bus_id,
        device=wire.device,
        hostname=wire.hostname,
        namespace=wire.namespace,
        pod_name=wire.pod_name,
        timestamp_ns=wire.timestamp_ns,
        dcgm_url=wire.dcgm_url,
        telemetry_data=TelemetryMetrics.model_validate(wire.telemetry_data),
    )


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
        records=tuple(_record_to_wire(r) for r in records),
        error=_error_to_wire(error),
    )


def telemetry_records_from_wire(
    wire: TelemetryRecordsWireMessage,
) -> list[TelemetryRecord]:
    """Rehydrate the pydantic TelemetryRecord list from its wire envelope."""
    return [_wire_to_record(r) for r in wire.records]


def telemetry_error_from_wire(
    wire: TelemetryRecordsWireMessage,
) -> ErrorDetails | None:
    """Rehydrate the optional ErrorDetails from the wire envelope."""
    return _wire_to_error(wire.error)
