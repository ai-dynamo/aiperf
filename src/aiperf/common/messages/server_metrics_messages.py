# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    ErrorDetails,
    ProcessServerMetricsResult,
    ServerMetricsRecord,
)
from aiperf.common.types import MessageTypeT


class ServerMetricsRecordsMessage(BaseServiceMessage):
    """Message from the server metrics data collector to the records manager to notify it
    of the server metrics records for a batch of server samples.

    Contains full server metrics records with all metadata.
    """

    message_type: MessageTypeT = MessageType.SERVER_METRICS_RECORDS

    collector_id: str = Field(
        description="The ID of the server metrics data collector that collected the records"
    )
    records: list[ServerMetricsRecord] = Field(
        default_factory=list,
        description="The server metrics records",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the server metrics record collection failed.",
    )

    @property
    def valid(self) -> bool:
        """Whether server metrics collection succeeded (empty response is valid)."""
        return self.error is None

    @property
    def has_data(self) -> bool:
        """Whether any metrics were collected."""
        return len(self.records) > 0


class ProcessServerMetricsResultMessage(BaseServiceMessage):
    """Message containing processed server metrics results - mirrors ProcessRecordsResultMessage."""

    message_type: MessageTypeT = MessageType.PROCESS_SERVER_METRICS_RESULT

    server_metrics_result: ProcessServerMetricsResult = Field(
        description="The processed server metrics results"
    )


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
