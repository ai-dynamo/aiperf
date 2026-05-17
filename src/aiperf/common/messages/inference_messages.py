# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import msgspec

from aiperf.common.enums import MessageType
from aiperf.common.messages.wire_codec import register_msgspec_message
from aiperf.common.models import ErrorDetails, RequestRecord
from aiperf.common.models.record_models import MetricRecordMetadata, MetricResult
from aiperf.common.models.trace_models import BaseTraceDataUnion
from aiperf.common.types import MetricTagT

# msgspec rejects ``int | float | list[float] | list[int]`` (multiple list-like
# branches in one union). ``MetricRecordsData.metrics`` and
# ``MetricRecordsMessage.metrics`` therefore carry the runtime
# ``MetricValueTypeT`` shape via plain ``object``; consumers downcast as needed.
MetricValuePayload = object


class InferenceResultsMessage(
    msgspec.Struct,
    kw_only=True,
    tag=str(MessageType.INFERENCE_RESULTS),
    tag_field="message_type",
    omit_defaults=True,
):
    """Message for inference results.

    Wire-only ``msgspec.Struct`` (not part of the Pydantic ``Message`` MRO);
    the wire codec dispatches encode/decode by family. ``tag_field`` pins the
    ``message_type`` discriminator on the wire so the codec can route inbound
    bytes back to this class; the property below exposes the same value to
    in-process consumers that read ``message.message_type`` uniformly across
    Pydantic and msgspec messages.
    """

    service_id: str
    record: RequestRecord
    request_ns: int | None = None
    request_id: str | None = None

    @property
    def message_type(self) -> str:
        return type(self).__struct_config__.tag  # type: ignore[no-any-return]


register_msgspec_message(MessageType.INFERENCE_RESULTS, InferenceResultsMessage)


class MetricRecordsData(msgspec.Struct, kw_only=True, omit_defaults=True):
    """Incoming data from the record processor service to combine metric records for the profile run.

    Wire-compatible ``msgspec.Struct``. Constructed from ``MetricRecordsMessage``
    on the records-manager side via ``MetricRecordsMessage.to_data()``.
    """

    metadata: MetricRecordMetadata
    """The metadata of the request record."""

    metrics: dict[MetricTagT, MetricValuePayload]
    """The combined metric records for this inference request."""

    trace_data: BaseTraceDataUnion | None = None
    """Comprehensive trace data captured via a trace config. Includes detailed
    timing for connection establishment, DNS resolution, request/response
    events, etc. The type of the trace data is determined by the transport
    and library used."""

    error: ErrorDetails | None = None
    """The error details if the request failed."""

    @property
    def valid(self) -> bool:
        """Whether the request was valid."""
        return self.error is None


class MetricRecordsMessage(
    msgspec.Struct,
    kw_only=True,
    tag=str(MessageType.METRIC_RECORDS),
    tag_field="message_type",
    omit_defaults=True,
):
    """Message from the result parser to the records manager to notify it
    of the metric records for a single request.

    Wire-only ``msgspec.Struct`` (not part of the Pydantic ``Message`` MRO);
    the wire codec dispatches encode/decode by family.
    """

    metadata: MetricRecordMetadata
    """The metadata of the request record."""

    metrics: dict[MetricTagT, MetricValuePayload]
    """The merged record processor metric results."""

    service_id: str | None = None
    """ID of the service sending the message."""

    request_ns: int | None = None
    request_id: str | None = None

    trace_data: BaseTraceDataUnion | None = None
    """Comprehensive trace data captured via a trace config."""

    error: ErrorDetails | None = None
    """The error details if the request failed."""

    @property
    def message_type(self) -> str:
        return type(self).__struct_config__.tag  # type: ignore[no-any-return]

    @property
    def valid(self) -> bool:
        """Whether the request was valid."""
        return self.error is None

    def to_data(self) -> MetricRecordsData:
        """Convert the metric records message to MetricRecordsData for processing."""
        return MetricRecordsData(
            metadata=self.metadata,
            metrics=self.metrics,
            trace_data=self.trace_data,
            error=self.error,
        )


register_msgspec_message(MessageType.METRIC_RECORDS, MetricRecordsMessage)


class RealtimeMetricsMessage(
    msgspec.Struct,
    kw_only=True,
    tag=str(MessageType.REALTIME_METRICS),
    tag_field="message_type",
    omit_defaults=True,
):
    """Message from the records manager to show real-time metrics for the profile run.

    Wire-only ``msgspec.Struct`` (not part of the Pydantic ``Message`` MRO);
    the wire codec dispatches encode/decode by family.
    """

    metrics: list[MetricResult]
    """The current real-time metrics."""

    service_id: str | None = None
    """ID of the service sending the message."""

    request_ns: int | None = None
    request_id: str | None = None

    @property
    def message_type(self) -> str:
        return type(self).__struct_config__.tag  # type: ignore[no-any-return]


register_msgspec_message(MessageType.REALTIME_METRICS, RealtimeMetricsMessage)
