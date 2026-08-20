# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic import Field

from aiperf.common.enums import MessageType, SystemState
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


class BenchmarkCompleteMessage(BaseServiceMessage):
    """Benchmark completion signal.

    Published by the SystemController after all result artifacts have been
    exported, so external consumers (the API results router, the operator)
    only report "complete" once every file is safe to fetch.
    """

    message_type: MessageTypeT = MessageType.BENCHMARK_COMPLETE

    request_ns: int | None = Field(
        default=None,
        ge=0,
        description="Timestamp of the message in nanoseconds",
    )

    was_cancelled: bool = Field(
        default=False,
        description="True when the benchmark was cancelled before completing normally",
    )


class SystemStateChangedMessage(BaseServiceMessage):
    """Published by the SystemController whenever its outer-lifecycle
    ``SystemState`` advances (e.g. CONFIGURING -> READY -> PROFILING ->
    PROCESSING -> STOPPING -> SHUTDOWN).

    Subscribers (notably the ProgressRouter that fronts ``/api/progress``)
    mirror the new value so external tooling -- operator, dashboard,
    ``kubectl get aiperfjob`` -- can observe the controller's view of where
    the run is, distinct from the operator's outer ``phase`` field.
    """

    message_type: MessageTypeT = MessageType.SYSTEM_STATE_CHANGED

    request_ns: int | None = Field(
        default=None,
        ge=0,
        description="Timestamp of the message in nanoseconds",
    )

    state: SystemState = Field(
        ..., description="The new outer-lifecycle state of the SystemController"
    )


class ResultsExportedMessage(BaseServiceMessage):
    """Signals that all result artifacts have been written to disk.

    Published by the SystemController after ``ExporterManager.export_data()``
    completes and (in K8s mode) after ``write_ready_marker(...)`` is on disk.
    The operator gates ``JobProgress.is_complete`` on this signal: for
    sub-second benchmarks the existing ``is_requests_complete &&
    is_records_complete`` check flips True before the controller has finished
    writing, so the kopf-timer monitor can otherwise claim completion and
    fetch a partial artifact set.
    """

    message_type: MessageTypeT = MessageType.RESULTS_EXPORTED

    request_ns: int | None = Field(
        default=None,
        ge=0,
        description="Timestamp of the message in nanoseconds",
    )

    was_cancelled: bool = Field(
        default=False,
        description="True when the benchmark was cancelled before completing normally",
    )
