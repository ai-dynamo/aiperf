# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import MessageType, MetricValueTypeT
from aiperf.common.messages.service_messages import BaseServiceMessage
# Re-export the wire-layer payload type so existing call-sites that imported
# ``MetricRecordsData`` from this module keep working after the wholesale
# msgspec port. The canonical home is ``aiperf.common.metric_records_wire``.
from aiperf.common.metric_records_wire import (  # noqa: F401
    MetricRecordsData,
)
from aiperf.common.models import ErrorDetails, RequestRecord
from aiperf.common.models.record_models import MetricRecordMetadata, MetricResult
from aiperf.common.models.trace_models import BaseTraceData
from aiperf.common.types import MetricTagT


class InferenceResultsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.INFERENCE_RESULTS.value
):
    """Single inference result record."""

    record: RequestRecord


class RealtimeMetricsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.REALTIME_METRICS.value
):
    """Real-time metrics summary."""

    metrics: list[MetricResult]


class MetricRecordsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.METRIC_RECORDS.value
):
    """Per-request metric records emitted by the record-processor to the records-manager.

    Carries the metadata, the list of per-metric values, optional trace data, and
    error details if the request failed. Compatibility shim retained while the
    record pipeline still routes through the Message bus; the native msgspec
    wire layer (``metric_records_wire``) is the underlying transport.
    """

    metadata: MetricRecordMetadata
    results: list[dict[MetricTagT, MetricValueTypeT]]
    trace_data: BaseTraceData | None = None
    error: ErrorDetails | None = None
