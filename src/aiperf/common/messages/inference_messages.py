# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field, SerializeAsAny, field_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import MessageType, MetricValueTypeT
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails, RequestRecord
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.record_models import MetricRecordMetadata, MetricResult
from aiperf.common.models.trace_models import BaseTraceData
from aiperf.common.types import MessageTypeT, MetricTagT

_logger = AIPerfLogger(__name__)


class InferenceResultsMessage(BaseServiceMessage):
    """Message for a inference results."""

    message_type: MessageTypeT = MessageType.INFERENCE_RESULTS

    record: SerializeAsAny[RequestRecord] = Field(
        ..., description="The inference results record"
    )


class MetricRecordsData(AIPerfBaseModel):
    """Incoming data from the record processor service to combine metric records for the profile run."""

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


class MetricRecordsMessage(BaseServiceMessage):
    """Message from the result parser to the records manager to notify it
    of the metric records for a single request."""

    message_type: MessageTypeT = MessageType.METRIC_RECORDS

    metadata: MetricRecordMetadata = Field(
        ..., description="The metadata of the request record."
    )
    results: list[dict[MetricTagT, MetricValueTypeT]] = Field(
        ..., description="The record processor metric results"
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

    def to_data(self) -> MetricRecordsData:
        """Convert the metric records message to a MetricRecordsData for processing by the records manager."""
        metrics = {}
        for result in self.results:
            for tag, value in result.items():
                if tag in metrics:
                    _logger.warning(
                        f"Duplicate metric tag '{tag}' found in results. "
                        f"Overwriting previous value {metrics[tag]} with {value}."
                    )
                metrics[tag] = value

        return MetricRecordsData(
            metadata=self.metadata,
            metrics=metrics,
            trace_data=self.trace_data,
            error=self.error,
        )


class RealtimeMetricsMessage(BaseServiceMessage):
    """Message from the records manager to show real-time metrics for the profile run."""

    message_type: MessageTypeT = MessageType.REALTIME_METRICS

    metrics: list[MetricResult] = Field(
        ..., description="The current real-time metrics."
    )
