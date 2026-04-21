# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

import msgspec

from aiperf.common.enums import MessageType
from aiperf.common.messages.service_messages import BaseServiceMessage
from aiperf.common.models import PhaseRecordsStats, WorkerProcessingStats
from aiperf.common.models.record_models import ProcessRecordsResult, ProfileResults


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
