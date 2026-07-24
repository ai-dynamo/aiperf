# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ConfigDict, Field, SerializeAsAny, TypeAdapter, field_validator

from aiperf.common.enums import MessageType, MetricValueTypeT
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    BaseResponseData,
    EmbeddingResponseData,
    ErrorDetails,
    ImageResponseData,
    ImageRetrievalResponseData,
    ParsedResponse,
    RAGSources,
    RankingsResponseData,
    ReasoningResponseData,
    RecordData,
    RequestRecord,
    TextResponseData,
    ToolCallResponseData,
    VideoResponseData,
)
from aiperf.common.models.record_models import MetricRecordMetadata, MetricResult
from aiperf.common.models.trace_models import BaseTraceData
from aiperf.common.types import MessageTypeT, MetricTagT

ParsedResponseDataType = Literal[
    "base",
    "embedding",
    "image",
    "image_retrieval",
    "rankings",
    "reasoning",
    "text",
    "tool_call",
    "video",
]
_RESPONSE_DATA_TYPE_BY_CLASS: dict[type[BaseResponseData], ParsedResponseDataType] = {
    BaseResponseData: "base",
    EmbeddingResponseData: "embedding",
    ImageResponseData: "image",
    ImageRetrievalResponseData: "image_retrieval",
    RankingsResponseData: "rankings",
    ReasoningResponseData: "reasoning",
    TextResponseData: "text",
    ToolCallResponseData: "tool_call",
    VideoResponseData: "video",
}
_RESPONSE_DATA_ADAPTER_BY_TYPE = {
    data_type: TypeAdapter(data_class)
    for data_class, data_type in _RESPONSE_DATA_TYPE_BY_CLASS.items()
}


@dataclass(slots=True)
class ParsedResponsePayload:
    """Compact built-in parsed response representation for internal IPC."""

    __pydantic_config__ = ConfigDict(extra="forbid")

    perf_ns: int
    """Performance timestamp of the parsed response in nanoseconds."""

    data_type: ParsedResponseDataType | None = None
    """Built-in response data discriminator, or None for usage-only responses."""

    data: Any | None = None
    """Built-in parsed response data."""

    usage: dict[str, Any] | None = None
    """Server-reported usage associated with this response."""

    sources: RAGSources | None = None
    """RAG sources associated with this response."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional response metadata."""

    @classmethod
    def from_parsed_response(
        cls, response: ParsedResponse
    ) -> ParsedResponsePayload | None:
        """Encode a parsed response when its concrete data type is built in."""
        data = response.data
        data_type = None
        if data is not None:
            data_type = _RESPONSE_DATA_TYPE_BY_CLASS.get(type(data))
            if data_type is None:
                return None

        # Encoding is terminal: the worker does not mutate parsed responses after
        # publication, so sharing these references avoids hot-path defensive copies.
        return cls(
            perf_ns=response.perf_ns,
            data_type=data_type,
            data=data,
            usage=response.usage,
            sources=response.sources,
            metadata=response.metadata,
        )

    def to_parsed_response(self) -> ParsedResponse:
        """Reconstruct the worker-produced parsed response."""
        data = self.data
        if self.data_type is not None and isinstance(data, dict):
            data = _RESPONSE_DATA_ADAPTER_BY_TYPE[self.data_type].validate_python(data)
        return ParsedResponse(
            perf_ns=self.perf_ns,
            data=data,
            usage=self.usage,
            sources=self.sources,
            metadata=self.metadata,
        )


def encode_parsed_responses(
    responses: list[ParsedResponse],
) -> list[ParsedResponsePayload] | None:
    """Encode built-in responses atomically, falling back on any custom type."""
    payloads: list[ParsedResponsePayload] = []
    for response in responses:
        payload = ParsedResponsePayload.from_parsed_response(response)
        if payload is None:
            return None
        payloads.append(payload)
    return payloads


class InferenceResultsMessage(BaseServiceMessage):
    """Message for a inference results."""

    message_type: MessageTypeT = MessageType.INFERENCE_RESULTS

    record: SerializeAsAny[RequestRecord] = Field(
        ..., description="The inference results record"
    )
    parsed_responses: list[ParsedResponsePayload] | None = Field(
        default=None,
        description="Worker-produced parsed responses for compact internal processing.",
    )
    last_response_perf_ns: int | None = Field(
        default=None,
        gt=0,
        description="Performance timestamp of the final raw response.",
    )
    raw_response_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of raw responses received before optional compaction.",
    )
    responses_compacted: bool = Field(
        default=False,
        description="Whether the worker validated and compacted the raw responses.",
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
