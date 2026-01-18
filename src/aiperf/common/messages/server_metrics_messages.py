# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import ErrorDetails, ServerMetricsRecord
from aiperf.common.models.server_metrics_models import ProcessServerMetricsResult
from aiperf.common.types import MessageTypeT


class ServerMetricsRecordMessage(BaseServiceMessage):
    """Message from the server metrics data collector to the records manager.

    Contains a single server metrics record with all metric samples from one
    Prometheus endpoint scrape.
    """

    message_type: MessageTypeT = MessageType.SERVER_METRICS_RECORD

    collector_id: str = Field(
        description="The ID of the server metrics data collector that collected the records"
    )
    record: ServerMetricsRecord | None = Field(
        default=None,
        description="The server metrics record",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the server metrics record collection failed.",
    )

    @property
    def valid(self) -> bool:
        """Whether server metrics collection succeeded."""
        return self.error is None and self.record is not None


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
