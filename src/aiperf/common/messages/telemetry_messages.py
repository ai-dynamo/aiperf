# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import msgspec

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    ErrorDetails,
    MetricResult,
    ProcessTelemetryResult,
    TelemetryRecord,
)


class ProcessTelemetryResultMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.PROCESS_TELEMETRY_RESULT.value
):
    """Processed telemetry results envelope."""

    telemetry_result: ProcessTelemetryResult


class TelemetryStatusMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.TELEMETRY_STATUS.value
):
    """Telemetry availability report."""

    enabled: bool
    reason: str | None = None
    endpoints_configured: list[str] = msgspec.field(default_factory=list)
    endpoints_reachable: list[str] = msgspec.field(default_factory=list)


class RealtimeTelemetryMetricsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.REALTIME_TELEMETRY_METRICS.value
):
    """Real-time GPU telemetry metrics."""

    metrics: list[MetricResult]


class TelemetryRecordsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.TELEMETRY_RECORDS.value
):
    """Telemetry data collector -> records manager batch of GPU samples."""

    collector_id: str
    dcgm_url: str
    records: list[TelemetryRecord]
    error: ErrorDetails | None = None
