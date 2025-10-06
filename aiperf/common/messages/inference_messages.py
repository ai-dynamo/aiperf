# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import (
    Field,
    SerializeAsAny,
)

from aiperf.common.enums import (
    CreditPhase,
    MessageType,
)
from aiperf.common.enums.metric_enums import MetricValueTypeT
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails, RequestRecord
from aiperf.common.models.record_models import MetricResult
from aiperf.common.types import MessageTypeT, MetricTagT


class InferenceResultsMessage(BaseServiceMessage):
    """Message for a inference results."""

    message_type: MessageTypeT = MessageType.INFERENCE_RESULTS

    record: SerializeAsAny[RequestRecord] = Field(
        ..., description="The inference results record"
    )


class MetricRecordsMessage(BaseServiceMessage):
    """Message from the result parser to the records manager to notify it
    of the metric records for a single request."""

    message_type: MessageTypeT = MessageType.METRIC_RECORDS

    timestamp_ns: int = Field(
        ..., description="The wall clock timestamp of the request in nanoseconds."
    )
    x_request_id: str | None = Field(
        default=None, description="The X-Request-ID header of the request."
    )
    x_correlation_id: str | None = Field(
        default=None, description="The X-Correlation-ID header of the request."
    )
    worker_id: str = Field(
        ..., description="The ID of the worker that processed the request."
    )
    credit_phase: CreditPhase = Field(
        ..., description="The credit phase of the request."
    )
    results: list[dict[MetricTagT, MetricValueTypeT]] = Field(
        ..., description="The record processor results"
    )
    error: ErrorDetails | None = Field(
        default=None, description="The error details if the request failed."
    )

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
