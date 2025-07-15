# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from pydantic import Field

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.credit_models import (
    CreditPhaseStats,
    PhaseProcessingStats,
)
from aiperf.common.enums import BenchmarkSuiteType, CreditPhase, MessageType
from aiperf.common.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseStartMessage,
    Message,
    ProfileResultsMessage,
    RecordsProcessingStatsMessage,
    WorkerHealthMessage,
)
from aiperf.common.pydantic_utils import AIPerfBaseModel
from aiperf.common.worker_models import WorkerPhaseTaskStats

logger = AIPerfLogger(__name__)


class CreditPhaseComputedStats(AIPerfBaseModel):
    """Contains additional stats for a credit phase based on computed values."""

    # Computed stats based on the TimingManager
    requests_per_second: float | None = Field(
        default=None, description="The average requests per second"
    )
    requests_eta: float | None = Field(
        default=None, description="The estimated time for all requests to be completed"
    )
    requests_update_ns: int | None = Field(
        default=None, description="The time of the last request update"
    )

    # Computed stats based on the RecordsManager
    records_per_second: float | None = Field(
        default=None, description="The average records processed per second"
    )
    records_eta: float | None = Field(
        default=None, description="The estimated time for all records to be processed"
    )
    records_update_ns: int | None = Field(
        default=None, description="The time of the last record update"
    )


class FullCreditPhaseProgress(
    CreditPhaseStats, PhaseProcessingStats, CreditPhaseComputedStats
):
    """Full state of the credit phase progress, including the progress of the phase, the processing stats, and the worker stats."""

    last_record_update_ns: int | None = Field(
        default=None,
        description="The last time the records stats were updated in nanoseconds",
    )
    worker_processing_stats: dict[str, PhaseProcessingStats] = Field(
        default_factory=dict,
        description="The processing stats for each worker as reported by the RecordsManager (processed, errors)",
    )
    last_request_update_ns: int | None = Field(
        default=None,
        description="The last time the requests stats were updated in nanoseconds",
    )
    worker_request_stats: dict[str, WorkerPhaseTaskStats] = Field(
        default_factory=dict,
        description="The request stats for each worker as reported by the Workers (total, completed, failed)",
    )

    @property
    def elapsed_time(self) -> float | None:
        """Get the elapsed time."""
        if not self.start_ns or not self.last_request_update_ns:
            return None
        return (self.last_request_update_ns - self.start_ns) / NANOS_PER_SECOND


class ProfileRunProgress(AIPerfBaseModel):
    """State of the profile run progress, including the progress of each credit phase, the processing stats, and the worker stats.

    The progress of each credit phase is tracked by from the TimingManager.
    The processing stats and worker processing stats are tracked from the RecordsManager.
    The worker task stats are tracked from the Workers.
    """

    # TODO: implement the profile_id
    profile_id: str | None = Field(default=None, description="The ID of the profile")
    start_ns: int | None = Field(
        default=None, description="The start time of the profile run in nanoseconds"
    )
    end_ns: int | None = Field(
        default=None, description="The end time of the profile run in nanoseconds"
    )
    last_update_ns: int | None = Field(
        default=None,
        description="The last time the progress was updated in nanoseconds",
    )
    active_phase: CreditPhase | None = Field(
        default=None, description="The active credit phase"
    )
    phases: dict[CreditPhase, FullCreditPhaseProgress] = Field(
        default_factory=dict,
        description="The full credit stats for each credit phase as reported by the TimingManager and RecordsManager.",
    )
    profile_results: ProfileResultsMessage | None = Field(
        default=None, description="The profile results"
    )
    was_cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled early",
    )

    @property
    def is_started(self) -> bool:
        """Check if the profile run is started."""
        return any(phase.is_started for phase in self.phases.values())

    @property
    def is_complete(self) -> bool:
        """Check if the profile run is complete."""
        if not self.phases:
            return False
        return all(phase.is_complete for phase in self.phases.values())

    @property
    def total_expected_requests(self) -> int | None:
        """Get the total number of requests."""
        if not self.phases:
            return None
        return sum(
            phase.total_requests
            for phase in self.phases.values()
            if phase.total_requests is not None
        )

    @property
    def requests_completed(self) -> int | None:
        """Get the number of requests completed."""
        if not self.phases:
            return None
        return sum(
            phase.completed
            for phase in self.phases.values()
            if phase.completed is not None
        )

    @property
    def requests_processed(self) -> int | None:
        """Get the number of requests processed."""
        if not self.phases:
            return None
        return sum(
            phase.processed
            for phase in self.phases.values()
            if phase.processed is not None
        )

    @property
    def request_errors(self) -> int | None:
        """Get the number of requests with errors."""
        if not self.phases:
            return None
        return sum(
            phase.errors for phase in self.phases.values() if phase.errors is not None
        )

    @property
    def requests_per_second(self) -> float | None:
        """Get the requests per second."""
        if not self.active_phase:
            return None
        if not self.phases:
            return None
        return self.phases[self.active_phase].requests_per_second

    @property
    def requests_eta(self) -> float | None:
        """Get the requests eta."""
        if not self.active_phase:
            return None
        if not self.phases:
            return None
        return self.phases[self.active_phase].requests_eta

    @property
    def processed_per_second(self) -> float | None:
        """Get the processed per second."""
        if not self.active_phase:
            return None
        if not self.phases:
            return None
        return self.phases[self.active_phase].records_per_second

    @property
    def processing_eta(self) -> float | None:
        """Get the processed eta."""
        if not self.active_phase:
            return None
        if not self.phases:
            return None
        return self.phases[self.active_phase].records_eta

    @property
    def elapsed_time(self) -> float | None:
        """Get the elapsed time."""
        if not self.start_ns or not self.last_update_ns:
            return None
        return (self.last_update_ns - self.start_ns) / NANOS_PER_SECOND

    @property
    def eta(self) -> float | None:
        """Get the eta."""
        if not self.requests_eta or not self.processing_eta:
            return None
        return max(self.requests_eta, self.processing_eta)

    def on_message(self, message: Message):
        """Update the progress from a message."""
        _message_mappings = {
            MessageType.CREDIT_PHASE_START: self.on_credit_phase_start,
            MessageType.CREDIT_PHASE_PROGRESS: self.on_credit_phase_progress,
            MessageType.CREDIT_PHASE_COMPLETE: self.on_credit_phase_complete,
            MessageType.PROCESSING_STATS: self.on_phase_processing_stats,
            MessageType.WORKER_HEALTH: self.on_worker_health,
            MessageType.PROFILE_RESULTS: self.on_profile_results,
        }

        if message.message_type in _message_mappings:
            _message_mappings[message.message_type](message)
        else:
            logger.debug(
                lambda: f"ProfileRunProgress: Received unsupported message type: {message.message_type}"
            )

    def on_credit_phase_progress(self, message: CreditPhaseProgressMessage):
        """Update the progress from a credit phase progress message."""
        if message.phase not in self.phases:
            logger.debug(
                lambda: f"ProfileRunProgress: Received credit phase progress message for unknown phase: {message.phase}"
            )
            return

        self.phases[message.phase].sent = message.sent
        self.phases[message.phase].completed = message.completed
        self.update_requests_stats(self.phases[message.phase], message.request_ns)

    def on_credit_phase_start(self, message: CreditPhaseStartMessage):
        """Update the progress from a credit phase start message."""
        self.active_phase = message.phase
        self.phases[message.phase] = FullCreditPhaseProgress(
            type=message.phase,
            start_ns=message.start_ns,
            total_requests=message.total_requests,
            expected_duration_ns=message.expected_duration_ns,
        )
        self.update_requests_stats(self.phases[message.phase], message.start_ns)

    def on_credit_phase_complete(self, message: CreditPhaseCompleteMessage):
        """Update the progress from a credit phase complete message."""
        self.phases[message.phase].end_ns = message.end_ns
        # Mark all credits as completed
        self.phases[message.phase].completed = self.phases[message.phase].sent
        self.update_requests_stats(self.phases[message.phase], message.request_ns)

    def on_phase_processing_stats(self, message: RecordsProcessingStatsMessage):
        """Update the progress from a phase processing stats message."""
        self.phases[message.phase].processed = message.processing_stats.processed
        self.phases[message.phase].errors = message.processing_stats.errors
        self.phases[message.phase].last_record_update_ns = time.time_ns()

        for worker_id, worker_stats in message.worker_stats.items():
            self.phases[message.phase].worker_processing_stats[worker_id] = worker_stats

        self.update_records_stats(
            message.phase,
            message.request_ns,
            message.processing_stats,
        )

    def on_worker_health(self, message: WorkerHealthMessage):
        """Update the progress from a worker health message."""
        worker_id = message.service_id
        for phase, stats in message.task_stats.items():
            self.phases[phase].worker_request_stats[worker_id] = stats

    def on_profile_results(self, message: ProfileResultsMessage):
        """Update the progress from a profile results message."""
        self.profile_results = message

    def update_requests_stats(self, phase: CreditPhaseStats, request_ns: int):
        """Update the requests stats based on the TimingManager stats."""
        self.last_update_ns = request_ns or time.time_ns()
        self.phases[phase.type].requests_update_ns = request_ns
        logger.debug(
            lambda: f"Updating requests stats for phase {phase.type} {request_ns} {phase.start_ns} {time.time_ns()}"
        )

        if (
            phase.start_ns is not None
            and (diff_ns := (request_ns or time.time_ns()) - phase.start_ns) > 0
        ):
            dur_sec = diff_ns / NANOS_PER_SECOND
            self.phases[phase.type].requests_per_second = phase.completed / dur_sec
            if pct := phase.progress_percent:
                pct_remaining = 100 - pct
                self.phases[phase.type].requests_eta = pct_remaining / (pct / dur_sec)
            else:
                self.phases[phase.type].requests_eta = None
        else:
            self.phases[phase.type].requests_per_second = None
            self.phases[phase.type].requests_eta = None

    def update_records_stats(
        self, phase: CreditPhase, request_ns: int, stats: PhaseProcessingStats
    ):
        """Update the records stats based on the RecordsManager stats."""
        self.phases[phase].records_update_ns = request_ns

        # Check if the phase exists in phases before accessing it
        if phase in self.phases and (
            self.phases[phase].start_ns is not None
            and (diff_ns := request_ns - self.phases[phase].start_ns) > 0  # pyright: ignore
        ):
            dur_sec = diff_ns / NANOS_PER_SECOND
            self.phases[phase].records_per_second = stats.processed / dur_sec
            if (
                self.phases[phase].records_per_second
                and self.phases[phase].total_requests is not None
            ):
                self.phases[phase].records_eta = (
                    self.phases[phase].total_requests - stats.processed  # pyright: ignore
                ) / self.phases[phase].records_per_second
            else:
                self.phases[phase].records_eta = None
        else:
            self.phases[phase].records_per_second = None
            self.phases[phase].records_eta = None


# class SweepRunProgress(AIPerfBaseModel):
#     """State of the sweep run progress. TBD"""

#     sweep_id: str = Field(..., description="The ID of the sweep")


class BenchmarkSuiteProgress(AIPerfBaseModel):
    """State of the benchmark suite progress."""

    type: BenchmarkSuiteType = Field(..., description="The type of benchmark suite")
    start_ns: int | None = Field(
        default=None, description="The start time of the benchmark suite in nanoseconds"
    )
    end_ns: int | None = Field(
        default=None, description="The end time of the benchmark suite in nanoseconds"
    )
    profile_runs: list[ProfileRunProgress] = Field(
        default_factory=list, description="The state of the profile runs in the suite"
    )
    current_profile_run: ProfileRunProgress | None = Field(
        default_factory=ProfileRunProgress,
        description="The current profile run progress",
    )
