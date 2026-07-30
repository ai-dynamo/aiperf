# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import Field, SerializeAsAny, field_validator

from aiperf.common.enums import MessageType, MetricValueTypeT
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails, RecordData, RequestRecord
from aiperf.common.models.record_models import MetricRecordMetadata, MetricResult
from aiperf.common.models.spec_decode_models import SpecDecodeAcceptanceRecord
from aiperf.common.models.trace_models import BaseTraceData
from aiperf.common.types import MessageTypeT, MetricTagT


class InferenceResultsMessage(BaseServiceMessage):
    """Message for a inference results."""

    message_type: MessageTypeT = MessageType.INFERENCE_RESULTS

    record: SerializeAsAny[RequestRecord] = Field(
        ..., description="The inference results record"
    )


class MetricRecordsData(RecordData):
    """Incoming data from the record processor service to combine metric records for the profile run."""

    record_type: Literal["metric_records"] = Field(
        default="metric_records",
        description="Serialized discriminator routing this record to the "
        "metric_records channel for wire reconstruction.",
    )

    metadata: MetricRecordMetadata = Field(
        ..., description="The metadata of the request record."
    )
    metrics: dict[MetricTagT, MetricValueTypeT] = Field(
        ..., description="The combined metric records for this inference request."
    )
    trace_data: SerializeAsAny[BaseTraceData] | None = Field(
        default=None,
        description="Comprehensive trace data captured via a trace config. "
        "Includes detailed timing for connection establishment, DNS resolution, request/response events, etc. "
        "The type of the trace data is determined by the transport and library used.",
    )
    spec_decode_acceptance: SpecDecodeAcceptanceRecord | None = Field(
        default=None,
        description="Engine-neutral per-request speculative-decoding acceptance "
        "record, carried across the ZMQ boundary so the records manager can pool "
        "its histogram and the records-trace exporter can emit it per request. "
        "None when spec decode is off or the request had no verify steps.",
    )
    error: ErrorDetails | None = Field(
        default=None, description="The error details if the request failed."
    )

    @field_validator("trace_data", mode="before")
    @classmethod
    def route_trace_data(cls, v: Any) -> BaseTraceData | None:
        """Route nested trace_data to correct subclass based on trace_type discriminator."""
        if isinstance(v, dict):
            return BaseTraceData.from_json(v)
        return v

    @property
    def valid(self) -> bool:
        """Whether the request was valid."""
        return self.error is None


class RecordsMessage(BaseServiceMessage):
    """Per-request envelope from the record processor service to the records manager.

    One ``RecordsMessage`` is pushed for every inference record (the credit
    lockstep contract). It carries the request metadata plus the flattened list
    of typed records produced for that request. Each record self-identifies via
    its own serialized ``record_type`` discriminator, so the records manager
    dispatches generically without inspecting the message for a record type.
    """

    message_type: MessageTypeT = MessageType.RECORDS

    metadata: MetricRecordMetadata = Field(
        ..., description="The metadata of the request record; drives the lockstep."
    )
    records: list[SerializeAsAny[RecordData]] = Field(
        default_factory=list,
        description="The typed records produced for this request. Each record "
        "self-identifies via its own serialized record_type discriminator, so the "
        "concrete subclass is reconstructed on the receiving side of the ZMQ boundary.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The request-level error details if the request failed.",
    )

    @field_validator("records", mode="before")
    @classmethod
    def route_records(cls, v: Any) -> Any:
        """Reconstruct each dict record into its concrete RecordData subclass.

        On the wire the records arrive as plain dicts; route each one via
        ``RecordData.from_json`` (using the serialized ``record_type``
        discriminator) so the records manager receives concrete typed records,
        not bare dicts. Already-constructed model instances pass through.
        """
        if not isinstance(v, list):
            return v
        return [
            RecordData.from_json(item) if isinstance(item, dict) else item for item in v
        ]

    @property
    def valid(self) -> bool:
        """Whether the request was valid."""
        return self.error is None


class RealtimeMetricsMessage(BaseServiceMessage):
    """Message from the records manager to show real-time metrics for the profile run."""

    message_type: MessageTypeT = MessageType.REALTIME_METRICS

    metrics: list[MetricResult] = Field(
        ..., description="The current real-time metrics."
    )
