# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import TYPE_CHECKING, Any

import msgspec

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import PhaseRecordsStats, WorkerProcessingStats
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults
from aiperf.common.models.server_metrics_models import ServerMetricsResults

if TYPE_CHECKING:
    pass


class RecordsProcessingStatsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.PROCESSING_STATS.value
):
    """Per-phase processing stats from the RecordsManager."""

    processing_stats: PhaseRecordsStats
    worker_stats: dict[str, WorkerProcessingStats] = msgspec.field(default_factory=dict)


class ProfileResultsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.PROFILE_RESULTS.value
):
    """Final profile results."""

    profile_results: ProfileResults


class AllRecordsReceivedMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.ALL_RECORDS_RECEIVED.value
):
    """All parsed records received; final stats available."""

    final_processing_stats: PhaseRecordsStats
    request_ns: int = msgspec.field(default_factory=time.time_ns)  # type: ignore[assignment]


class ProcessRecordsResultMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.PROCESS_RECORDS_RESULT.value
):
    """Record-processor batch result."""

    results: ProcessRecordsResult


class BenchmarkCompleteMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.BENCHMARK_COMPLETE.value
):
    """Benchmark completion signal."""

    was_cancelled: bool = False


class ResultsExportedMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.RESULTS_EXPORTED.value
):
    """Signals that all result artifacts have been written to disk.

    Published by the SystemController after ``ExporterManager.export_data()``
    completes and (in K8s mode) after ``write_ready_marker(...)`` is on disk.
    The operator gates ``JobProgress.is_complete`` on this signal: for
    sub-second benchmarks the existing ``is_requests_complete &&
    is_records_complete`` check flips True before the controller has finished
    writing, so the kopf-timer monitor can otherwise claim completion and
    fetch a partial artifact set. Without this gate, the operator races the
    controller's exporter and surfaces ``Phase.Failed``.
    """

    was_cancelled: bool = False


class ProcessAllResultsMessage(
    BaseServiceMessage, kw_only=True, tag=MessageType.PROCESS_ALL_RESULTS.value
):
    """Unified message carrying all accumulator results from RecordsManager to SystemController.

    The optional summary payloads (steady-state, energy efficiency) and the
    ``exported_artifacts`` map are typed as ``Any`` to keep this foundation
    module out of the ``aiperf.analysis`` / ``aiperf.exporters`` /
    ``aiperf.post_processors`` import graph; producers/consumers cast to the
    concrete types they own (``SteadyStateSummary``,
    ``EnergyEfficiencySummary``, ``dict[str, FileExportInfo]``).
    """

    results: ProcessRecordsResult
    telemetry_results: TelemetryExportData | None = None
    server_metrics_results: ServerMetricsResults | None = None
    steady_state_results: Any = None
    energy_efficiency_results: Any = None
    exported_artifacts: dict[str, Any] = msgspec.field(default_factory=dict)
