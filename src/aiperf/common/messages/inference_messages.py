# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage

# Re-export the wire-layer payload type so existing call-sites that imported
# ``MetricRecordsData`` from this module keep working after the wholesale
# msgspec port. The canonical home is ``aiperf.common.metric_records_wire``.
from aiperf.common.metric_records_wire import (  # noqa: F401
    MetricRecordsData,
    _error_to_wire,
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
    results: list[dict[MetricTagT, Any]]
    trace_data: BaseTraceData | None = None
    error: ErrorDetails | None = None

    def to_data(self) -> MetricRecordsData:
        """Project the on-bus message into the wire-layer payload struct used
        by post-processors that consume ``MetricRecordsData`` directly.

        ``MetricRecordsMessage.results`` is a list of per-call dicts (one per
        result record produced by the parser); ``MetricRecordsData.metrics``
        is a flat dict keyed by metric tag. Merge by taking the last-write
        value when a tag repeats (downstream order is irrelevant for the
        aggregator paths).
        """
        merged: dict = {}
        for d in self.results:
            merged.update(d)
        return MetricRecordsData(
            metadata=self.metadata,
            metrics=merged,
            trace_data=self.trace_data,
            error=_error_to_wire(self.error),
        )
