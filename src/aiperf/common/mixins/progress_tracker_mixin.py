# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import msgspec

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import (
    ProfileResultsMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import CreditPhaseStats, PhaseRecordsStats
from aiperf.common.models.base_models import PydanticStructMixin
from aiperf.common.models.credit_models import BasePhaseStats
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun

_logger = AIPerfLogger(__name__)


class CombinedPhaseStats(
    PydanticStructMixin,
    BasePhaseStats,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
):
    """Combined progress for a single phase: requests + records + computed rates.

    msgspec.Struct forbids multiple-inheritance of structs, so the request and
    records fields are flattened here rather than inherited from both
    CreditPhaseStats and PhaseRecordsStats.

    Retains ``PydanticStructMixin`` because ``JobProgress.phases`` (Pydantic,
    a FastAPI operator API response) embeds this struct.
    """

    # Credit progress fields (mirror CreditPhaseStats)
    requests_sent: int = 0
    requests_completed: int = 0
    requests_cancelled: int = 0
    request_errors: int = 0
    sent_sessions: int = 0
    completed_sessions: int = 0
    cancelled_sessions: int = 0
    total_session_turns: int = 0

    # Records progress fields (mirror PhaseRecordsStats)
    records_end_ns: int | None = None
    success_records: int = 0
    error_records: int = 0

    # Computed fields
    requests_per_second: float | None = None
    records_per_second: float | None = None
    requests_eta_sec: float | None = None
    records_eta_sec: float | None = None

    # Timestamp fields
    last_update_ns: int | None = None

    # --- Properties mirrored from CreditPhaseStats ---

    @property
    def in_flight_sessions(self) -> int:
        """Sessions started but not yet finished (no final turn returned)."""
        return self.sent_sessions - self.completed_sessions - self.cancelled_sessions

    @property
    def in_flight_requests(self) -> int:
        """Number of in-flight requests (sent but not completed)."""
        return self.requests_sent - self.requests_completed - self.requests_cancelled

    @property
    def requests_elapsed_time(self) -> float:
        """Get the elapsed time for requests."""
        if self.start_ns is None:
            return 0.0
        if self.requests_end_ns is not None:
            return (self.requests_end_ns - self.start_ns) / NANOS_PER_SECOND
        return (time.time_ns() - self.start_ns) / NANOS_PER_SECOND

    @property
    def requests_error_percent(self) -> float:
        """Error percentage of the requests completed."""
        if self.final_requests_completed is not None:
            if self.final_requests_completed == 0:
                return 0.0
            return (self.final_request_errors / self.final_requests_completed) * 100

        if self.requests_completed == 0:
            return 0.0
        return (self.request_errors / self.requests_completed) * 100

    @property
    def requests_progress_percent(self) -> float | None:
        """Progress percentage of the requests completed."""
        if self.start_ns is None:
            return None

        if self.is_requests_complete:
            return 100

        percentages = []
        if self.total_expected_requests:
            percentages.append(
                (self.requests_completed / self.total_expected_requests) * 100
            )
        if self.expected_duration_sec:
            elapsed_ns = time.time_ns() - self.start_ns
            expected_duration_ns = self.expected_duration_sec * NANOS_PER_SECOND
            percentages.append((elapsed_ns / expected_duration_ns) * 100)
        if self.expected_num_sessions:
            percentages.append(
                (self.completed_sessions / self.expected_num_sessions) * 100
            )

        if not percentages:
            return None

        return min(max(percentages), 100)

    # --- Properties mirrored from PhaseRecordsStats ---

    @property
    def total_records(self) -> int:
        """Total number of records processed (success + errors)."""
        return self.success_records + self.error_records

    @property
    def records_elapsed_time(self) -> float:
        """Get the elapsed time for records."""
        if self.start_ns is None:
            return 0.0
        if self.records_end_ns is not None:
            return (self.records_end_ns - self.start_ns) / NANOS_PER_SECOND
        return (time.time_ns() - self.start_ns) / NANOS_PER_SECOND

    @property
    def records_error_percent(self) -> float:
        """Error percentage of the records processed."""
        if self.total_records == 0:
            return 0.0
        return (self.error_records / self.total_records) * 100

    @property
    def records_progress_percent(self) -> float | None:
        """Progress percent of the records processed."""
        if self.final_requests_completed:
            return (self.total_records / self.final_requests_completed) * 100

        if self.total_expected_requests:
            return (self.total_records / self.total_expected_requests) * 100

        return None

    @property
    def is_records_complete(self) -> bool:
        return self.records_end_ns is not None


class ProgressTracker:
    """Progress tracker for the benchmark suite."""

    def __init__(self):
        self._phases: dict[CreditPhase, CombinedPhaseStats] = {}
        self._last_update_ns: int | None = None

    def _get_phase_progress(self, phase: CreditPhase) -> CombinedPhaseStats:
        """Get or create the combined phase stats for a phase."""
        if phase not in self._phases:
            self._phases[phase] = CombinedPhaseStats(phase=phase)
        return self._phases[phase]

    def _update_phase_progress(
        self,
        *,
        stats: CreditPhaseStats | PhaseRecordsStats,
        last_update_ns: int,
        finished: int,
        prefix: str,
    ) -> CombinedPhaseStats:
        """Update the combined phase stats with new progress data."""
        self._last_update_ns = last_update_ns

        pct = getattr(stats, f"{prefix}_progress_percent")

        _logger.debug(
            lambda: f"Updating {prefix} stats for phase '{stats.phase.title()}': progress_percent: {pct}, finished: {finished}"
        )

        if not pct or finished == 0:
            per_second = None
            eta_sec = None
        else:
            dur_ns = last_update_ns - (stats.start_ns or time.time_ns())
            dur_sec = dur_ns / NANOS_PER_SECOND
            # amount finished per second
            per_second = finished / dur_sec
            # (progress % remaining) / (progress % per second)
            eta_sec = (100 - pct) / (pct / dur_sec)

        updates = msgspec.structs.asdict(stats)
        updates["last_update_ns"] = last_update_ns
        updates[f"{prefix}_per_second"] = per_second
        updates[f"{prefix}_eta_sec"] = eta_sec

        current = self._get_phase_progress(stats.phase)
        self._phases[stats.phase] = msgspec.structs.replace(current, **updates)
        return self._phases[stats.phase]

    def update_requests_stats(self, stats: CreditPhaseStats) -> CombinedPhaseStats:
        """Update the requests stats for a phase."""
        return self._update_phase_progress(
            stats=stats,
            last_update_ns=time.time_ns(),
            finished=stats.requests_completed,
            prefix="requests",
        )

    def update_records_stats(self, stats: PhaseRecordsStats) -> CombinedPhaseStats:
        """Update the records stats for a phase."""
        return self._update_phase_progress(
            stats=stats,
            last_update_ns=time.time_ns(),
            finished=stats.total_records,
            prefix="records",
        )

    @property
    def last_update_ns(self) -> int | None:
        """Get the last update time."""
        return self._last_update_ns


@provides_hooks(
    AIPerfHook.ON_RECORDS_PROGRESS,
    AIPerfHook.ON_PHASE_PROGRESS,
)
class ProgressTrackerMixin(MessageBusClientMixin):
    """A progress tracker that tracks the progress of the entire benchmark suite."""

    def __init__(self, run: BenchmarkRun, **kwargs):
        super().__init__(run=run, **kwargs)
        self._progress_tracker = ProgressTracker()

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(self, message: CreditPhaseStartMessage):
        """Update the progress from a credit phase start message."""
        progress = self._progress_tracker.update_requests_stats(message.stats)
        await self._update_requests_stats(progress, message.stats.start_ns)
        await self._update_records_stats(progress, message.request_ns)

    @on_message(MessageType.CREDIT_PHASE_PROGRESS)
    async def _on_credit_phase_progress(self, message: CreditPhaseProgressMessage):
        """Update the progress from a credit phase progress message."""
        progress = self._progress_tracker.update_requests_stats(message.stats)
        await self._update_requests_stats(progress, message.stats.start_ns)

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ):
        """Update the progress from a credit phase sending complete message."""
        progress = self._progress_tracker.update_requests_stats(message.stats)
        await self._update_requests_stats(progress, message.stats.start_ns)

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(self, message: CreditPhaseCompleteMessage):
        """Update the progress from a credit phase complete message."""
        progress = self._progress_tracker.update_requests_stats(message.stats)
        await self._update_requests_stats(progress, message.stats.start_ns)
        await self._update_records_stats(progress, message.request_ns)

    @on_message(MessageType.PROCESSING_STATS)
    async def _on_phase_processing_stats(self, message: RecordsProcessingStatsMessage):
        """Update the progress from a phase processing stats message."""
        progress = self._progress_tracker.update_records_stats(message.processing_stats)
        await self._update_records_stats(progress, message.request_ns)

    @on_message(MessageType.PROFILE_RESULTS)
    async def _on_profile_results(self, message: ProfileResultsMessage):
        """Update the progress from a profile results message."""
        self.profile_results = message

    async def _update_requests_stats(
        self,
        phase_progress: CombinedPhaseStats,
        request_ns: int | None,
    ):
        """Update the requests stats based on the TimingManager stats."""
        await self.run_hooks(
            AIPerfHook.ON_PHASE_PROGRESS,
            phase_stats=phase_progress,
        )

    async def _update_records_stats(
        self, phase_progress: CombinedPhaseStats, request_ns: int | None
    ):
        """Update the records stats based on the RecordsManager stats."""
        if self.is_debug_enabled:
            self.debug(
                f"Updating records stats for phase '{phase_progress.phase.title()}': "
                f"processed: {phase_progress.success_records}, errors: {phase_progress.error_records}"
            )

        await self.run_hooks(
            AIPerfHook.ON_RECORDS_PROGRESS, records_stats=phase_progress
        )
