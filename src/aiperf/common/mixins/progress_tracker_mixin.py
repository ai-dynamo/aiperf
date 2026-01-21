# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from pydantic import ConfigDict, Field

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.config import ServiceConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import (
    ProfileResultsMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models import CreditPhaseStats, PhaseRecordsStats
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
)

_logger = AIPerfLogger(__name__)


class CombinedPhaseStats(CreditPhaseStats, PhaseRecordsStats):
    """Combined progress for a single phase for requests and records."""

    model_config = ConfigDict(frozen=True)

    # Computed fields
    requests_per_second: float | None = Field(
        default=None, description="The number of requests processed per second."
    )
    records_per_second: float | None = Field(
        default=None, description="The number of records processed per second."
    )
    requests_eta_sec: float | None = Field(
        default=None,
        description="The estimated time remaining to complete the requests in the phase in seconds.",
    )
    records_eta_sec: float | None = Field(
        default=None,
        description="The estimated time remaining to complete the records in the phase in seconds.",
    )

    # Timestamp fields
    last_update_ns: int | None = Field(
        default=None,
        ge=0,
        description="The last update time in nanoseconds (time.time_ns()).",
    )


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

        updates = stats.model_dump()
        updates["last_update_ns"] = last_update_ns
        updates[f"{prefix}_per_second"] = per_second
        updates[f"{prefix}_eta_sec"] = eta_sec

        current = self._get_phase_progress(stats.phase)
        self._phases[stats.phase] = current.model_copy(update=updates)
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
    AIPerfHook.ON_PROFILING_PROGRESS,
    AIPerfHook.ON_WARMUP_PROGRESS,
)
class ProgressTrackerMixin(MessageBusClientMixin):
    """A progress tracker that tracks the progress of the entire benchmark suite."""

    def __init__(self, service_config: ServiceConfig, **kwargs):
        super().__init__(service_config=service_config, **kwargs)
        self._progress_tracker = ProgressTracker()

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(self, message: CreditPhaseStartMessage):
        """Update the progress from a credit phase start message."""
        progress = self._progress_tracker.update_requests_stats(message.stats)
        await self._update_requests_stats(
            message.stats.phase, progress, message.stats.start_ns
        )
        await self._update_records_stats(progress, message.request_ns)

    @on_message(MessageType.CREDIT_PHASE_PROGRESS)
    async def _on_credit_phase_progress(self, message: CreditPhaseProgressMessage):
        """Update the progress from a credit phase progress message."""
        progress = self._progress_tracker.update_requests_stats(message.stats)
        await self._update_requests_stats(
            message.stats.phase, progress, message.stats.start_ns
        )

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ):
        """Update the progress from a credit phase sending complete message."""
        progress = self._progress_tracker.update_requests_stats(message.stats)
        await self._update_requests_stats(
            message.stats.phase, progress, message.stats.start_ns
        )

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(self, message: CreditPhaseCompleteMessage):
        """Update the progress from a credit phase complete message."""
        progress = self._progress_tracker.update_requests_stats(message.stats)
        await self._update_requests_stats(
            message.stats.phase, progress, message.stats.start_ns
        )
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
        phase: CreditPhase,
        phase_progress: CombinedPhaseStats,
        request_ns: int | None,
    ):
        """Update the requests stats based on the TimingManager stats."""
        if phase == CreditPhase.WARMUP:
            await self.run_hooks(
                AIPerfHook.ON_WARMUP_PROGRESS,
                warmup_stats=phase_progress,
            )
        elif phase == CreditPhase.PROFILING:
            await self.run_hooks(
                AIPerfHook.ON_PROFILING_PROGRESS,
                profiling_stats=phase_progress,
            )
        else:
            self.warning(f"Unsupported phase: {phase}")

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
