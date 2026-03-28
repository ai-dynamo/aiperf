# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Msgspec structs for the metric-record processing path."""

from __future__ import annotations

from typing import Any

import orjson
from msgspec import Struct

from aiperf.common.enums import CreditPhase, MessageType, MetricValueTypeT
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.trace_models import BaseTraceData
from aiperf.common.types import MetricTagT


def _json_safe(value: Any) -> Any:
    """Convert dynamic values to a JSON-safe representation."""
    if value is None:
        return None
    return orjson.loads(orjson.dumps(value))


class WireErrorDetails(Struct, frozen=True, kw_only=True, omit_defaults=True):
    code: int | None = None
    type: str | None = None
    message: str
    cause: str | None = None
    cause_chain: tuple[str, ...] | None = None
    details: Any | None = None


class MetricRecordMetadata(Struct, frozen=True, kw_only=True, omit_defaults=True):
    request_num: int | None = None
    session_num: int
    x_request_id: str | None = None
    x_correlation_id: str | None = None
    conversation_id: str | None = None
    turn_index: int | None = None
    credit_issued_ns: int | None = None
    credit_received_ns: int | None = None
    request_start_ns: int
    request_ack_ns: int | None = None
    request_end_ns: int
    worker_id: str
    record_processor_id: str
    benchmark_phase: CreditPhase
    was_cancelled: bool = False
    cancellation_time_ns: int | None = None
    clock_offset_ns: int | None = None

    def model_dump(self) -> dict[str, Any]:
        """Compatibility helper for code that flattens metadata into dicts."""
        return {
            "request_num": self.request_num,
            "session_num": self.session_num,
            "x_request_id": self.x_request_id,
            "x_correlation_id": self.x_correlation_id,
            "conversation_id": self.conversation_id,
            "turn_index": self.turn_index,
            "credit_issued_ns": self.credit_issued_ns,
            "credit_received_ns": self.credit_received_ns,
            "request_start_ns": self.request_start_ns,
            "request_ack_ns": self.request_ack_ns,
            "request_end_ns": self.request_end_ns,
            "worker_id": self.worker_id,
            "record_processor_id": self.record_processor_id,
            "benchmark_phase": self.benchmark_phase,
            "was_cancelled": self.was_cancelled,
            "cancellation_time_ns": self.cancellation_time_ns,
            "clock_offset_ns": self.clock_offset_ns,
        }


class MetricRecordsData(Struct, frozen=True, kw_only=True, omit_defaults=True):
    metadata: MetricRecordMetadata
    metrics: dict[MetricTagT, Any]
    trace_data: BaseTraceData | None = None
    error: ErrorDetails | None = None

    @property
    def valid(self) -> bool:
        return self.error is None

    def to_data(self) -> MetricRecordsData:
        return self


class MetricRecordsWireMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="mr",
):
    message_type: MessageType = MessageType.METRIC_RECORDS
    service_id: str
    metadata: MetricRecordMetadata
    metrics: dict[MetricTagT, Any]
    trace_data: dict[str, Any] | None = None
    error: WireErrorDetails | None = None


class MetricRecordsBatchWireMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="mrb",
):
    message_type: MessageType = MessageType.METRIC_RECORDS
    service_id: str
    records: list[MetricRecordsData]


def _error_to_wire(error: ErrorDetails | None) -> WireErrorDetails | None:
    if error is None:
        return None
    cause_chain = tuple(error.cause_chain) if error.cause_chain else None
    return WireErrorDetails(
        code=error.code,
        type=error.type,
        message=error.message,
        cause=error.cause,
        cause_chain=cause_chain,
        details=_json_safe(error.details),
    )


def _wire_to_error(error: WireErrorDetails | None) -> ErrorDetails | None:
    if error is None:
        return None
    return ErrorDetails(
        code=error.code,
        type=error.type,
        message=error.message,
        cause=error.cause,
        cause_chain=list(error.cause_chain) if error.cause_chain else None,
        details=error.details,
    )


def metric_record_metadata_from_model(
    metadata: Any,
) -> MetricRecordMetadata:
    if isinstance(metadata, MetricRecordMetadata):
        return metadata
    if isinstance(metadata, dict):
        return MetricRecordMetadata(**metadata)
    return MetricRecordMetadata(
        request_num=metadata.request_num,
        session_num=metadata.session_num,
        x_request_id=metadata.x_request_id,
        x_correlation_id=metadata.x_correlation_id,
        conversation_id=metadata.conversation_id,
        turn_index=metadata.turn_index,
        credit_issued_ns=metadata.credit_issued_ns,
        credit_received_ns=metadata.credit_received_ns,
        request_start_ns=metadata.request_start_ns,
        request_ack_ns=metadata.request_ack_ns,
        request_end_ns=metadata.request_end_ns,
        worker_id=metadata.worker_id,
        record_processor_id=metadata.record_processor_id,
        benchmark_phase=metadata.benchmark_phase,
        was_cancelled=metadata.was_cancelled,
        cancellation_time_ns=metadata.cancellation_time_ns,
        clock_offset_ns=metadata.clock_offset_ns,
    )


def build_metric_records_data(
    *,
    metadata: Any,
    metrics: dict[MetricTagT, MetricValueTypeT | Any],
    trace_data: BaseTraceData | None,
    error: ErrorDetails | None,
) -> MetricRecordsData:
    return MetricRecordsData(
        metadata=metric_record_metadata_from_model(metadata),
        metrics=metrics,
        trace_data=trace_data,
        error=error,
    )


def build_metric_records_wire_message(
    *,
    service_id: str,
    metadata: Any,
    metrics: dict[MetricTagT, MetricValueTypeT | Any],
    trace_data: BaseTraceData | None,
    error: ErrorDetails | None,
) -> MetricRecordsWireMessage:
    metadata_struct = metric_record_metadata_from_model(metadata)
    return MetricRecordsWireMessage(
        service_id=service_id,
        metadata=metadata_struct,
        metrics=metrics,
        trace_data=trace_data.model_dump(exclude_none=True, mode="json")
        if trace_data is not None
        else None,
        error=_error_to_wire(error),
    )


def build_metric_records_batch_wire_message(
    *,
    service_id: str,
    records: list[MetricRecordsData],
) -> MetricRecordsBatchWireMessage:
    return MetricRecordsBatchWireMessage(service_id=service_id, records=records)


def wire_message_to_record_data(message: MetricRecordsWireMessage) -> MetricRecordsData:
    return MetricRecordsData(
        metadata=message.metadata,
        metrics=message.metrics,
        trace_data=BaseTraceData.from_json(message.trace_data)
        if message.trace_data is not None
        else None,
        error=_wire_to_error(message.error),
    )
