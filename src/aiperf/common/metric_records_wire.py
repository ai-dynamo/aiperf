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
    """Wire-format error details for the metric-record processing path."""

    code: int | None = None
    """HTTP status code or application error code, when available."""

    type: str | None = None
    """Error type classifier string."""

    message: str
    """Human-readable error message."""

    cause: str | None = None
    """Root cause description, when available."""

    cause_chain: tuple[str, ...] | None = None
    """Chain of exception causes for nested errors."""

    details: Any | None = None
    """Arbitrary JSON-safe error detail payload."""


class MetricRecordMetadata(Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Metadata associated with a single metric record for request tracking."""

    request_num: int | None = None
    """Sequential request number (0-based credit index within the phase)."""

    session_num: int
    """Sequential session/conversation number (0-based)."""

    x_request_id: str | None = None
    """Unique request identifier (X-Request-ID header)."""

    x_correlation_id: str | None = None
    """Conversation instance identifier for sticky routing (X-Correlation-ID header)."""

    conversation_id: str | None = None
    """Template conversation ID from the dataset."""

    turn_index: int | None = None
    """Index of the turn in the conversation (0-based)."""

    credit_issued_ns: int | None = None
    """Wall clock timestamp when the credit was issued."""

    credit_received_ns: int | None = None
    """Wall clock timestamp when the credit was received by the worker."""

    request_start_ns: int
    """Performance counter timestamp at request start."""

    request_ack_ns: int | None = None
    """Performance counter timestamp when the server acknowledged the request."""

    request_end_ns: int
    """Performance counter timestamp at request end."""

    worker_id: str
    """Worker service identifier that processed this request."""

    record_processor_id: str
    """Record processor service identifier that computed the metrics."""

    benchmark_phase: CreditPhase
    """Credit phase during which this request was executed."""

    was_cancelled: bool = False
    """Whether this request was cancelled before completion."""

    cancellation_time_ns: int | None = None
    """Performance counter timestamp when cancellation was triggered."""

    clock_offset_ns: int | None = None
    """Estimated clock offset in nanoseconds for cross-process time alignment."""

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
    """Computed metric record for a single inference request."""

    metadata: MetricRecordMetadata
    """Request tracking metadata."""

    metrics: dict[MetricTagT, Any]
    """Computed metric values keyed by metric tag."""

    trace_data: BaseTraceData | None = None
    """Plugin-specific trace data, when available."""

    error: ErrorDetails | None = None
    """Error details if the request failed."""

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
    """Wire envelope for a single metric record on the RP->RecordsManager channel."""

    message_type: MessageType = MessageType.METRIC_RECORDS
    """Message type discriminator."""

    service_id: str
    """Record processor service identifier that produced this record."""

    metadata: MetricRecordMetadata
    """Request tracking metadata."""

    metrics: dict[MetricTagT, Any]
    """Computed metric values keyed by metric tag."""

    trace_data: dict[str, Any] | None = None
    """JSON-safe trace data dict for plugin-specific trace fields."""

    error: WireErrorDetails | None = None
    """Wire-format error details if the request failed."""


class MetricRecordsBatchWireMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="mrb",
):
    """Wire envelope for a batch of metric records on the RP->RecordsManager channel."""

    message_type: MessageType = MessageType.METRIC_RECORDS
    """Message type discriminator."""

    service_id: str
    """Record processor service identifier that produced this batch."""

    records: list[MetricRecordsData]
    """Batch of computed metric records."""


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
