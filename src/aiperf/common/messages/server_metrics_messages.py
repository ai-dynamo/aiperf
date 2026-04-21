# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import msgspec

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models.server_metrics_models import (
    ProcessServerMetricsResult,
    ServerMetricsEndpointSummary,
)


class ServerMetricsStatusMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.SERVER_METRICS_STATUS.value
):
    """Server-metrics availability report."""

    enabled: bool
    reason: str | None = None
    endpoints_configured: list[str] = msgspec.field(default_factory=list)
    endpoints_reachable: list[str] = msgspec.field(default_factory=list)


class ProcessServerMetricsResultMessage(
    BaseServiceMessage,
    kw_only=True,
    tag=MessageType.PROCESS_SERVER_METRICS_RESULT.value,
):
    """Processed server-metrics results envelope."""

    server_metrics_result: ProcessServerMetricsResult


class RealtimeServerMetricsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.REALTIME_SERVER_METRICS.value
):
    """Real-time per-endpoint server metrics."""

    endpoint_summaries: dict[str, ServerMetricsEndpointSummary]
