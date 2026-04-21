# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models.server_metrics_models import (
    ProcessServerMetricsResult,
    ServerMetricsEndpointSummary,
)
from aiperf.common.types import MessageTypeT


class ServerMetricsStatusMessage(BaseServiceMessage):
    """Message from ServerMetricsManager to SystemController indicating server metrics availability."""

    message_type: MessageTypeT = MessageType.SERVER_METRICS_STATUS

    enabled: bool = Field(
        description="Whether server metrics collection is enabled and will produce results"
    )
    reason: str | None = Field(
        default=None,
        description="Reason why server metrics is disabled (if enabled=False)",
    )
    endpoints_configured: list[str] = Field(
        default_factory=list,
        description="List of Prometheus endpoint URLs configured",
    )
    endpoints_reachable: list[str] = Field(
        default_factory=list,
        description="List of Prometheus endpoint URLs that were reachable and will provide data",
    )


class ProcessServerMetricsResultMessage(BaseServiceMessage):
    """Message containing processed server metrics results - mirrors ProcessTelemetryResultMessage."""

    message_type: MessageTypeT = MessageType.PROCESS_SERVER_METRICS_RESULT

    server_metrics_result: ProcessServerMetricsResult = Field(
        description="The processed server metrics results"
    )


class RealtimeServerMetricsMessage(BaseServiceMessage):
    """Message from the records manager to show real-time server metrics.

    Uses the same structure as the JSON export format with endpoint summaries
    containing metrics keyed by name with type-specific stats.
    """

    message_type: MessageTypeT = MessageType.REALTIME_SERVER_METRICS

    endpoint_summaries: dict[str, ServerMetricsEndpointSummary] = Field(
        ...,
        description="Per-endpoint server metrics summaries with computed statistics.",
    )
