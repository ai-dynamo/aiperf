# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from aiperf.common.config import ServiceConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase, MessageType
from aiperf.common.hooks import AIPerfHook, on_message, provides_hooks
from aiperf.common.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseProgressMessage,
    CreditPhaseSendingCompleteMessage,
    CreditPhaseStartMessage,
    ProfileResultsMessage,
    RecordsProcessingStatsMessage,
)
from aiperf.common.mixins.message_bus_mixin import MessageBusClientMixin
from aiperf.common.models.progress_models import (
    FullPhaseProgress,
    RecordsStats,
    RequestsStats,
    StatsProtocol,
    WorkerStats,
)


@provides_hooks(
    AIPerfHook.ON_RECORDS_PROGRESS,
    AIPerfHook.ON_PROFILING_PROGRESS,
    AIPerfHook.ON_WARMUP_PROGRESS,
)
class ProgressTrackerMixin(MessageBusClientMixin):
    """A progress tracker that tracks the progress of the entire benchmark suite."""

    def __init__(self, service_config: ServiceConfig, **kwargs):
        super().__init__(service_config=service_config, **kwargs)
        self._phase_progress_map: dict[CreditPhase, FullPhaseProgress] = {}
        self._active_phase: CreditPhase | None = None
        self._phase_progress_lock = asyncio.Lock()
        self._workers_stats: dict[str, WorkerStats] = {}
        self._workers_stats_lock = asyncio.Lock()

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(self, message: CreditPhaseStartMessage):
        """Update the progress from a credit phase start message."""
        async with self._phase_progress_lock:
            if message.phase in self._phase_progress_map:
                self.warning(f"Phase stats already started for {message.phase}")
                return
            self._active_phase = message.phase
            phase_progress = FullPhaseProgress(
                requests=RequestsStats(
                    type=message.phase,
                    start_ns=message.start_ns,
                    # Only one of the below would be set
                    total_expected_requests=message.total_expected_requests,
                    expected_duration_sec=message.expected_duration_sec,
                ),
                records=RecordsStats(
                    # May potentially not be set if the phase is not request count based
                    total_expected_requests=message.total_expected_requests,
                    start_ns=message.start_ns,
                ),
            )
            self._phase_progress_map[message.phase] = phase_progress
            await self._update_requests_stats(
                message.phase, phase_progress, message.start_ns
            )
            if message.phase == CreditPhase.PROFILING:
                await self._update_records_stats(phase_progress, message.start_ns)

    @on_message(MessageType.CREDIT_PHASE_PROGRESS)
    async def _on_credit_phase_progress(self, message: CreditPhaseProgressMessage):
        """Update the progress from a credit phase progress message."""
        async with self.phase_progress_context(message.phase) as phase_progress:
            phase_progress.requests.sent = message.sent
            phase_progress.requests.completed = message.completed
            await self._update_requests_stats(
                message.phase, phase_progress, message.request_ns
            )

    @on_message(MessageType.CREDIT_PHASE_SENDING_COMPLETE)
    async def _on_credit_phase_sending_complete(
        self, message: CreditPhaseSendingCompleteMessage
    ):
        """Update the progress from a credit phase sending complete message."""
        async with self.phase_progress_context(message.phase) as phase_progress:
            phase_progress.requests.sent_end_ns = message.sent_end_ns
            phase_progress.requests.sent = message.sent
            await self._update_requests_stats(
                message.phase, phase_progress, message.request_ns
            )

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(self, message: CreditPhaseCompleteMessage):
        """Update the progress from a credit phase complete message."""
        async with self.phase_progress_context(message.phase) as phase_progress:
            phase_progress.requests.end_ns = message.end_ns
            # Just in case we did not get a progress report for the last credit (timing issues due to network)
            phase_progress.requests.completed = phase_progress.requests.sent
            await self._update_requests_stats(
                message.phase, phase_progress, message.request_ns
            )

    @on_message(MessageType.PROCESSING_STATS)
    async def _on_phase_processing_stats(self, message: RecordsProcessingStatsMessage):
        """Update the progress from a phase processing stats message."""
        async with self.phase_progress_context(CreditPhase.PROFILING) as phase_progress:
            phase_progress.records.processed = message.processing_stats.processed
            phase_progress.records.errors = message.processing_stats.errors
            phase_progress.records.last_update_ns = time.time_ns()

            for worker_id, processing_stats in message.worker_stats.items():
                async with self._workers_stats_lock:
                    if worker_id not in self._workers_stats:
                        self._workers_stats[worker_id] = WorkerStats(
                            worker_id=worker_id
                        )
                    self._workers_stats[worker_id].processing_stats = processing_stats

            await self._update_records_stats(phase_progress, message.request_ns)

    @on_message(MessageType.PROFILE_RESULTS)
    async def _on_profile_results(self, message: ProfileResultsMessage):
        """Update the progress from a profile results message."""
        self.profile_results = message

    async def _update_requests_stats(
        self,
        phase: CreditPhase,
        phase_progress: FullPhaseProgress,
        request_ns: int | None,
    ):
        """Update the requests stats based on the TimingManager stats."""
        if self.is_debug_enabled:
            self.debug(
                f"Updating requests stats for phase '{phase.title()}': sent: {phase_progress.requests.sent}, "
                f"completed: {phase_progress.requests.completed}, total_expected: {phase_progress.requests.total_expected_requests}"
            )
        self._update_computed_stats(request_ns, phase_progress.requests)

        if phase == CreditPhase.WARMUP:
            await self.run_hooks(
                AIPerfHook.ON_WARMUP_PROGRESS,
                warmup_stats=phase_progress.requests,
            )
        elif phase == CreditPhase.PROFILING:
            await self.run_hooks(
                AIPerfHook.ON_PROFILING_PROGRESS,
                profiling_stats=phase_progress.requests,
            )
        else:
            raise ValueError(f"Invalid phase: {phase}")

    async def _update_records_stats(
        self, phase_progress: FullPhaseProgress, request_ns: int | None
    ):
        """Update the records stats based on the RecordsManager stats."""
        if self.is_debug_enabled:
            self.debug(
                f"Updating records stats for phase '{phase_progress.phase.title()}': "
                f"processed: {phase_progress.records.processed}, errors: {phase_progress.records.errors}"
            )

        self._update_computed_stats(request_ns, phase_progress.records)
        await self.run_hooks(
            AIPerfHook.ON_RECORDS_PROGRESS, records_stats=phase_progress.records
        )

    def _update_computed_stats(
        self,
        request_ns: int | None,
        stats: StatsProtocol,
    ):
        """Update the computed stats based on incoming data.

        Args:
            request_ns: The time of the last request (or time.time_ns() if not set)
            stats: The stats to update
            finished: The number of finished requests or records
            progress_percent: The progress percent of the requests or records
            start_ns: The start time of the requests
        """
        dur_ns = (request_ns or time.time_ns()) - (stats.start_ns or time.time_ns())
        stats.last_update_ns = request_ns or time.time_ns()
        if dur_ns <= 0:
            stats.per_second = None
            stats.eta = None
            return

        dur_sec = dur_ns / NANOS_PER_SECOND
        stats.per_second = stats.finished / dur_sec
        if stats.progress_percent:
            # (% remaining) / (% per second)
            stats.eta = (100 - stats.progress_percent) / (
                stats.progress_percent / dur_sec
            )
        else:
            stats.eta = None

    @asynccontextmanager
    async def phase_progress_context(
        self, phase: CreditPhase
    ) -> AsyncGenerator[FullPhaseProgress, None]:
        """Context manager for safely accessing phase progress info with warning."""
        async with self._phase_progress_lock:
            phase_progress = self._phase_progress_map.get(phase)
            if phase_progress is None:
                self.warning(
                    f"Phase '{phase.title()}' not found in current profile run"
                )
                return
            yield phase_progress
