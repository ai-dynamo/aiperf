# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.finite import FiniteFloat
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
    """Real-time per-endpoint server metrics fan-out.

    Published by the ServerMetricsManager on every scrape cycle so the
    ``/api/server-metrics`` router can serve a live view without waiting for
    end-of-run aggregation. Carries only summaries, not raw samples, to keep
    the per-cycle message size bounded.
    """

    message_type: MessageTypeT = MessageType.REALTIME_SERVER_METRICS

    # Redeclared with ge=0 so the numeric-bounds invariant in
    # tests/unit/property/test_finite_invariants.py sees a bounded timestamp.
    request_ns: int | None = Field(
        default=None,
        ge=0,
        description="Timestamp of the request in nanoseconds",
    )

    endpoint_summaries: dict[str, ServerMetricsEndpointSummary] = Field(
        default_factory=dict,
        description="Latest metrics summary per Prometheus endpoint URL",
    )
    snapshot: dict[str, FiniteFloat] = Field(
        default_factory=dict,
        description="Bounded scalar snapshot for realtime console rendering.",
    )
