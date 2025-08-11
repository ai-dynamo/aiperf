# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Models for tracking the progress of the benchmark suite."""

import time
from typing import Protocol

from pydantic import Field

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums.timing_enums import CreditPhase
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.credit_models import CreditPhaseStats, PhaseProcessingStats
from aiperf.common.models.health_models import ProcessHealth
from aiperf.common.models.worker_models import WorkerTaskStats


class StatsProtocol(Protocol):
    """Protocol for stats."""

    progress_percent: float | None
    total_expected_requests: int | None
    update_ns: int | None
    per_second: float | None
    eta: float | None
    start_ns: int | None

    @property
    def finished(self) -> int: ...


class ComputedStats(AIPerfBaseModel):
    """Computed info for a phase (can be used for requests or records)."""

    per_second: float | None = Field(default=None, description="The average per second")
    eta: float | None = Field(
        default=None, description="The estimated time for completion"
    )
    update_ns: int | None = Field(
        default=None, description="The time of the last update"
    )


@implements_protocol(StatsProtocol)
class RequestsStats(ComputedStats, CreditPhaseStats):
    """Stats for the requests. Based on the TimingManager data."""

    @property
    def finished(self) -> int:
        """Get the number of finished requests."""
        return self.completed

    @property
    def elapsed_time(self) -> float | None:
        """Get the elapsed time."""
        if self.start_ns is None:
            return None
        return (time.time_ns() - self.start_ns) / NANOS_PER_SECOND


@implements_protocol(StatsProtocol)
class RecordsStats(ComputedStats, PhaseProcessingStats):
    """Stats for the records. Based on the RecordsManager data."""

    start_ns: int | None = Field(
        default=None,
        description="The start time of the requests in nanoseconds.",
    )

    @property
    def finished(self) -> int:
        """Get the number of finished records."""
        return self.processed + self.errors

    @property
    def progress_percent(self) -> float | None:
        """Get the progress percent."""
        if not self.total_expected_requests:
            return None
        return (self.total_records / self.total_expected_requests) * 100


class WorkerStats(AIPerfBaseModel):
    """Stats for a worker."""

    tasks: WorkerTaskStats = Field(
        default_factory=WorkerTaskStats,
        description="The task stats for the worker as reported by the Workers (total, completed, failed)",
    )
    processing: PhaseProcessingStats = Field(
        default_factory=PhaseProcessingStats,
        description="The processing stats for the worker as reported by the RecordsManager (processed, errors)",
    )
    health: ProcessHealth | None = Field(
        default=None,
        description="The health of the worker as reported by the Workers",
    )
    update_ns: int | None = Field(
        default=None,
        description="The last time the worker was updated in nanoseconds",
    )


class CreditPhaseProgress(AIPerfBaseModel):
    """Full state of the credit phase progress, including the progress of the phase, the processing stats, and the worker stats."""

    requests: RequestsStats = Field(
        ...,
        description="The progress stats for the requests as reported by the TimingManager",
    )
    records: RecordsStats = Field(
        ...,
        description="The progress stats for the records as reported by the RecordsManager",
    )

    @property
    def start_ns(self) -> int:
        """Get the start time."""
        # NOTE: This will always be set, because ProgressTracker will always set the start_ns when the phase starts.
        return self.requests.start_ns  # type: ignore

    @property
    def phase(self) -> CreditPhase:
        """Get the phase."""
        return self.requests.type

    @property
    def elapsed_time(self) -> float:
        """Get the elapsed time."""
        return (time.time_ns() - self.start_ns) / NANOS_PER_SECOND
