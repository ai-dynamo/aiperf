# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field

from aiperf.common.enums import MessageType
from aiperf.common.messages.base_messages import RequiresRequestNSMixin
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import (
    PhaseRecordsStats,
    ServerMetricsResults,
    TelemetryExportData,
    WorkerProcessingStats,
)
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults
from aiperf.common.types import MessageTypeT


class RecordsProcessingStatsMessage(BaseServiceMessage):
    """Message for processing stats. Sent by the RecordsManager to report the stats of the profile run.
    This contains the stats for a single credit phase only."""

    message_type: MessageTypeT = MessageType.PROCESSING_STATS

    processing_stats: PhaseRecordsStats = Field(
        ..., description="The stats for the credit phase"
    )
    worker_stats: dict[str, WorkerProcessingStats] = Field(
        default_factory=dict,
        description="The stats for each worker how many requests were processed and how many errors were "
        "encountered, keyed by worker service_id",
    )


class ProfileResultsMessage(BaseServiceMessage):
    """Message for profile results."""

    message_type: MessageTypeT = MessageType.PROFILE_RESULTS

    profile_results: ProfileResults = Field(..., description="The profile results")


class AllRecordsReceivedMessage(BaseServiceMessage, RequiresRequestNSMixin):
    """This is sent by the RecordsManager to signal that all parsed records have been received, and the final processing stats are available."""

    message_type: MessageTypeT = MessageType.ALL_RECORDS_RECEIVED
    final_processing_stats: PhaseRecordsStats = Field(
        ..., description="The final processing stats for the profile run"
    )


class ProcessRecordsResultMessage(BaseServiceMessage):
    """Message for process records result."""

    message_type: MessageTypeT = MessageType.PROCESS_RECORDS_RESULT

    results: ProcessRecordsResult = Field(..., description="The process records result")


class ProcessAllResultsMessage(BaseServiceMessage):
    """Unified message carrying all accumulator results from RecordsManager to SystemController.

    The ``exported_artifacts`` map is typed as ``Any`` to keep this foundation
    module out of the ``aiperf.exporters`` import graph; producers/consumers
    cast to the concrete types they own (``dict[str, FileExportInfo]``).
    """

    message_type: MessageTypeT = MessageType.PROCESS_ALL_RESULTS

    request_ns: int | None = Field(
        default=None,
        ge=0,
        description="Timestamp of the request in nanoseconds",
    )
    results: ProcessRecordsResult = Field(
        ...,
        description="Per-record metric results aggregated by the MetricsAccumulator",
    )
    telemetry_results: TelemetryExportData | None = Field(
        default=None,
        description="Aggregated GPU telemetry summary, or None when telemetry was disabled",
    )
    server_metrics_results: ServerMetricsResults | None = Field(
        default=None,
        description="Aggregated server-side Prometheus metrics, or None when server metrics were disabled",
    )
    exported_artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of exporter-name to FileExportInfo for files written during this run "
        "(typed Any-valued to avoid pulling exporter types into the foundation graph)",
    )
