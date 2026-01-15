# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from collections import defaultdict

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import MetricRecordsData
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import (
    CreditPhaseStats,
    PhaseRecordsStats,
    WorkerProcessingStats,
)


class CreditPhaseRecordsTracker(AIPerfLoggerMixin):
    """Credit Phase Records Tracker. This is used to track the progress of a credit phase, as well
    as provide atomic operations for incrementing the processed and error counts.

    Thread Safety:
        The increment_* methods guarantee no partial updates.
        Safe for asyncio concurrent access without locks because:
            1. Updates use atomic incrementation (no in-place mutation)
            2. No await points between read and write in atomic methods
        3. Asyncio event loop serializes all operations

    Key Methods:
        - increment_processed(): Atomically increment the processed count
        - increment_errors(): Atomically increment the errors count
        - create_stats(): Create a new immutable RecordsPhaseStats object for the phase (for use in messages).
        - mark_started(): Mark the phase as started (set the start_ns).
        - mark_processing_complete(): Mark the phase as processing complete (set the processing_end_ns).
    """

    def __init__(self, phase: CreditPhase, **kwargs) -> None:
        super().__init__(**kwargs)
        # Must be set by the caller
        self._phase: CreditPhase = phase
        self._total_expected_requests: int | None = None

        # Timestamp fields
        self._start_ns: int | None = None
        self._sent_end_ns: int | None = None
        self._requests_end_ns: int | None = None
        # Records processing timestamp fields
        self._records_end_ns: int | None = None

        # Progress fields
        self._success_records: int = 0
        self._error_records: int = 0

        # Final count fields
        self._final_requests_completed: int | None = None
        self._final_requests_cancelled: int | None = None
        self._final_request_errors: int | None = None

        # Timeout/cancel fields
        self._timeout_triggered: bool = False
        self._grace_period_timeout_triggered: bool = False
        self._was_cancelled: bool = False

        # Completion fields
        self._sent_all_records_received: bool = False

        # Worker fields
        self._worker_stats: dict[str, WorkerProcessingStats] = defaultdict(
            WorkerProcessingStats
        )

    @property
    def phase(self) -> CreditPhase:
        """The phase of the credit phase tracker."""
        return self._phase

    @property
    def total_records(self) -> int:
        """The total number of records processed, errored, or filtered out."""
        return self._success_records + self._error_records

    @property
    def is_active(self) -> bool:
        """Check if the phase is active."""
        return self._start_ns is not None and self._records_end_ns is None

    def create_stats(self) -> PhaseRecordsStats:
        """Create a new immutable RecordsPhaseStats object for the phase (for use in messages)."""
        return PhaseRecordsStats(
            phase=self._phase,
            start_ns=self._start_ns,
            sent_end_ns=self._sent_end_ns,
            requests_end_ns=self._requests_end_ns,
            records_end_ns=self._records_end_ns,
            total_expected_requests=self._total_expected_requests,
            success_records=self._success_records,
            error_records=self._error_records,
            final_requests_completed=self._final_requests_completed,
            final_requests_cancelled=self._final_requests_cancelled,
            final_request_errors=self._final_request_errors,
            timeout_triggered=self._timeout_triggered,
            grace_period_timeout_triggered=self._grace_period_timeout_triggered,
            was_cancelled=self._was_cancelled,
        )

    def update_from_credit_phase_stats(self, credit_stats: CreditPhaseStats) -> None:
        """Update the phase info."""
        self._start_ns = credit_stats.start_ns
        self._sent_end_ns = credit_stats.sent_end_ns
        self._requests_end_ns = credit_stats.requests_end_ns
        self._total_expected_requests = credit_stats.total_expected_requests
        self._final_requests_completed = credit_stats.final_requests_completed
        self._final_requests_cancelled = credit_stats.final_requests_cancelled
        self._final_request_errors = credit_stats.final_request_errors
        self._timeout_triggered = credit_stats.timeout_triggered
        self._grace_period_timeout_triggered = (
            credit_stats.grace_period_timeout_triggered
        )
        self._was_cancelled = credit_stats.was_cancelled

    def increment_success_records(self) -> None:
        """Increment the success records count."""
        self._success_records += 1

    def increment_error_records(self) -> None:
        """Increment the error records count."""
        self._error_records += 1

    def increment_worker_success_records(self, worker_id: str) -> None:
        """Increment the success records count for a worker."""
        self._worker_stats[worker_id].success_records += 1

    def increment_worker_error_records(self, worker_id: str) -> None:
        """Increment the error records count for a worker."""
        self._worker_stats[worker_id].error_records += 1

    def check_and_set_all_records_received(self) -> bool:
        """Check if all records have been received and set the flag if so.
        Returns:
            True if all records have been received and the flag was not already set, False otherwise.
        """
        if self._sent_all_records_received:
            return False

        all_records_received = self._final_requests_completed is not None and (
            self._success_records + self._error_records
            >= self._final_requests_completed
        )
        if all_records_received:
            self._records_end_ns = time.time_ns()
            self._sent_all_records_received = True

        return all_records_received


class RecordsTracker:
    """Records Tracker. This is used to track the progress of the records phases.

    Fields:
        phase: The type of credit phase
        total_expected_requests: The total number of expected requests to process. If None, the phase is not request count based.
    """

    def __init__(self) -> None:
        self._phase_trackers: dict[CreditPhase, CreditPhaseRecordsTracker] = {}

    def _get_phase_tracker(self, phase: CreditPhase) -> CreditPhaseRecordsTracker:
        """Get the phase tracker."""
        if phase not in self._phase_trackers:
            self._phase_trackers[phase] = CreditPhaseRecordsTracker(phase)
        return self._phase_trackers[phase]

    def create_overall_worker_stats(self) -> dict[str, WorkerProcessingStats]:
        """Create a new dictionary of WorkerProcessingStats objects for ALL phases."""
        all_worker_stats = defaultdict(WorkerProcessingStats)
        for phase_tracker in self._phase_trackers.values():
            for worker_id, worker_stats in phase_tracker._worker_stats.items():
                all_worker_stats[
                    worker_id
                ].success_records += worker_stats.success_records
                all_worker_stats[worker_id].error_records += worker_stats.error_records
        return dict(all_worker_stats)

    def create_stats_for_phase(self, phase: CreditPhase) -> PhaseRecordsStats:
        """Create a new immutable RecordsPhaseStats object for the phase (for use in messages)."""
        phase_tracker = self._get_phase_tracker(phase)
        return phase_tracker.create_stats()

    def update_phase_info(self, credit_phase_stats: CreditPhaseStats) -> None:
        """Update the phase tracker."""
        phase_tracker = self._get_phase_tracker(credit_phase_stats.phase)
        phase_tracker.update_from_credit_phase_stats(credit_phase_stats)

    def update_from_record_data(self, record_data: MetricRecordsData) -> None:
        """Update the phase tracker from a record data."""
        phase_tracker = self._get_phase_tracker(record_data.metadata.benchmark_phase)
        if record_data.valid:
            phase_tracker.increment_success_records()
            phase_tracker.increment_worker_success_records(
                record_data.metadata.worker_id
            )
        else:
            phase_tracker.increment_error_records()
            phase_tracker.increment_worker_error_records(record_data.metadata.worker_id)

    def was_phase_cancelled(self, phase: CreditPhase) -> bool:
        """Check if the phase was cancelled."""
        phase_tracker = self._get_phase_tracker(phase)
        return phase_tracker._was_cancelled

    def mark_phase_cancelled(self, phase: CreditPhase) -> None:
        """Mark a phase as cancelled (e.g., from ProfileCancelCommand).

        This should be called when the cancel command is received to ensure
        the cancelled state is tracked even before the CreditPhaseCompleteMessage
        arrives with the updated stats.
        """
        phase_tracker = self._get_phase_tracker(phase)
        phase_tracker._was_cancelled = True

    def check_and_set_all_records_received_for_phase(self, phase: CreditPhase) -> bool:
        """Check if all records have been received and set the flag if so."""
        phase_tracker = self._get_phase_tracker(phase)
        return phase_tracker.check_and_set_all_records_received()

    def create_active_phase_stats_list(self) -> list[PhaseRecordsStats]:
        """Get the active phase stats."""
        return [
            pt.create_stats() for pt in self._phase_trackers.values() if pt.is_active
        ]
