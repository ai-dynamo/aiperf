# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from dataclasses import dataclass
from typing import ClassVar

import msgspec
from pydantic import ConfigDict

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase


@dataclass(slots=True, kw_only=True, frozen=True)
class BasePhaseStats:
    """Base model for phase stats. Tracks credit-phase progress.

    Slotted dataclass — shared type for msgspec envelopes
    (``CreditPhaseStartMessage.stats`` etc.) and Pydantic
    (``JobProgress.phases`` via ``CombinedPhaseStats``). Self-contained,
    so converting to dataclass needs no further cascade.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    phase: CreditPhase
    exclude_from_results: bool = False

    # Timestamp fields (None until the phase reaches that state)
    start_ns: int | None = None
    sent_end_ns: int | None = None
    requests_end_ns: int | None = None

    # Expectation / stop-condition fields (None when that condition is not used)
    total_expected_requests: int | None = None
    expected_duration_sec: float | None = None
    expected_num_sessions: int | None = None
    expected_grace_period_sec: float | None = None

    # Final count fields (None until the phase completes)
    final_requests_sent: int | None = None
    final_requests_completed: int | None = None
    final_requests_cancelled: int | None = None
    final_request_errors: int | None = None
    final_sent_sessions: int | None = None
    final_completed_sessions: int | None = None
    final_cancelled_sessions: int | None = None

    # Timeout / cancellation fields
    timeout_triggered: bool = False
    grace_period_timeout_triggered: bool = False
    was_cancelled: bool = False

    @property
    def is_started(self) -> bool:
        return self.start_ns is not None

    @property
    def is_sending_complete(self) -> bool:
        return self.sent_end_ns is not None

    @property
    def is_requests_complete(self) -> bool:
        return self.requests_end_ns is not None


@dataclass(slots=True, kw_only=True, frozen=True)
class CreditPhaseStats(BasePhaseStats):
    """Immutable phase-credit progress snapshot published per tick."""

    # Credit progress fields
    requests_sent: int = 0
    requests_completed: int = 0
    requests_cancelled: int = 0
    request_errors: int = 0
    sent_sessions: int = 0
    completed_sessions: int = 0
    cancelled_sessions: int = 0
    total_session_turns: int = 0

    @property
    def in_flight_sessions(self) -> int:
        """Sessions started but not yet finished (no final turn returned)."""
        return self.sent_sessions - self.completed_sessions - self.cancelled_sessions

    @property
    def in_flight_requests(self) -> int:
        """Calculate the number of in-flight requests (sent but not completed).

        NOTE: This can also be seen as the current actual "concurrency" value for the phase
        """
        return self.requests_sent - self.requests_completed - self.requests_cancelled

    @property
    def requests_elapsed_time(self) -> float:
        """Get the elapsed time."""
        if self.start_ns is None:
            return 0.0
        if self.requests_end_ns is not None:
            return (self.requests_end_ns - self.start_ns) / NANOS_PER_SECOND
        return (time.time_ns() - self.start_ns) / NANOS_PER_SECOND

    @property
    def requests_error_percent(self) -> float:
        """The error percentage of the requests completed."""
        if self.final_requests_completed is not None:
            if self.final_requests_completed == 0:
                return 0.0
            return (self.final_request_errors / self.final_requests_completed) * 100

        if self.requests_completed == 0:
            return 0.0
        return (self.request_errors / self.requests_completed) * 100

    @property
    def requests_progress_percent(self) -> float | None:
        """The progress percentage of the requests completed."""

        if self.start_ns is None:
            return None

        if self.is_requests_complete:
            return 100

        percentages = []
        pct_complete, pct_time_elapsed = 0, 0
        if self.total_expected_requests:
            pct_complete = (
                self.requests_completed / self.total_expected_requests
            ) * 100
            percentages.append(pct_complete)
        if self.expected_duration_sec:
            elapsed_ns = time.time_ns() - self.start_ns
            expected_duration_ns = self.expected_duration_sec * NANOS_PER_SECOND
            pct_time_elapsed = (elapsed_ns / expected_duration_ns) * 100
            percentages.append(pct_time_elapsed)
        if self.expected_num_sessions:
            pct_sessions_complete = (
                self.completed_sessions / self.expected_num_sessions
            ) * 100
            percentages.append(pct_sessions_complete)

        if not percentages:
            return None

        # Return the highest percentage, because the first condition met
        # will win when multiple conditions exist. Cap at 100%.
        return min(max(percentages), 100)


@dataclass(slots=True, kw_only=True, frozen=True)
class PhaseRecordsStats(BasePhaseStats):
    """Immutable phase-records progress snapshot."""

    # Timestamp fields
    records_end_ns: int | None = None

    # Progress fields
    success_records: int = 0
    error_records: int = 0

    @property
    def total_records(self) -> int:
        """The total number of records processed (success + errors)."""
        return self.success_records + self.error_records

    @property
    def records_elapsed_time(self) -> float:
        """Get the elapsed time."""
        if self.start_ns is None:
            return 0.0
        if self.records_end_ns is not None:
            return (self.records_end_ns - self.start_ns) / NANOS_PER_SECOND
        return (time.time_ns() - self.start_ns) / NANOS_PER_SECOND

    @property
    def records_error_percent(self) -> float:
        """The error percentage of the records processed."""
        if self.total_records == 0:
            return 0.0
        return (self.error_records / self.total_records) * 100

    @property
    def records_progress_percent(self) -> float | None:
        """The progress percent of the records processed."""
        if self.final_requests_completed:
            return (self.total_records / self.final_requests_completed) * 100

        if self.total_expected_requests:
            return (self.total_records / self.total_expected_requests) * 100

        return None

    @property
    def is_records_complete(self) -> bool:
        return self.records_end_ns is not None


class ProcessingStats(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Per-worker record-processing counters.

    Mutable accumulator used by the RecordsTracker to tally success/error
    counts in place — hence no ``frozen=True``. Stays ``msgspec.Struct``
    because it's nested in ``WorkerStats`` (which is still
    ``msgspec.Struct`` + mixin pending a broader cascade).
    """

    processed: int = 0
    errors: int = 0

    @property
    def total_records(self) -> int:
        """The total number of records (processed + errors)."""
        return self.processed + self.errors
