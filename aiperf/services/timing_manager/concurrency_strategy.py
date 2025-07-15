# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

from aiperf.common.credit_models import CreditPhaseStats
from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.mixins import AIPerfLoggerMixin, AsyncTaskManagerMixin
from aiperf.services.timing_manager.config import TimingManagerConfig
from aiperf.services.timing_manager.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditManagerProtocol,
)


class ConcurrencyStrategy(
    CreditIssuingStrategy, AsyncTaskManagerMixin, AIPerfLoggerMixin
):
    """Class for concurrency credit issuing strategy."""

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__(config=config, credit_manager=credit_manager)

        if not config.request_count or config.request_count < 1:
            # TODO: Add support for alternate completion triggers vs request count (eg. time based)
            raise InvalidStateError("Request count must be at least 1")

        # if the concurrency is larger than the total number of requests, it does not matter
        # as it is simply an upper bound that will never be reached
        self._concurrency = config.concurrency
        self._ramp_up_time = config.concurrency_ramp_up_time
        self._semaphore = asyncio.Semaphore(value=self._concurrency)

        self._setup_phases()

    def _setup_phases(self) -> None:
        """Setup the phases for the strategy."""
        # Add the warmup phase if applicable
        if self.config.warmup_request_count > 0:
            self.phases.append(CreditPhase.WARMUP)
            self.phase_stats[CreditPhase.WARMUP] = CreditPhaseStats(
                type=CreditPhase.WARMUP, total_requests=self.config.warmup_request_count
            )

        # Add the profiling phase
        self.phases.append(CreditPhase.PROFILING)
        self.phase_stats[CreditPhase.PROFILING] = CreditPhaseStats(
            type=CreditPhase.PROFILING, total_requests=self.config.request_count
        )

        self.active_phase_idx = 0
        self.active_phase: CreditPhaseStats = self.phase_stats[
            self.phases[self.active_phase_idx]
        ]

        self.info(
            lambda: f"TM: Concurrency Strategy initialized with {len(self.phases)} phases: {self.phases}"
        )

    async def start(self) -> None:
        """Start the credit issuing strategy. This will launch the progress reporting loop, the
        warmup phase (if applicable), and the profiling phase, all in the background."""

        # Start the progress reporting loop in the background
        self.execute_async(self._progress_report_loop())

        # Execute the phases in the background
        self.execute_async(self._execute_phases())

    async def _execute_time_based_phase(self, phase: CreditPhaseStats) -> None:
        start_ns: int = phase.start_ns  # type: ignore[assignment]
        expected_duration_ns: int = phase.expected_duration_ns  # type: ignore[assignment]

        while time.time_ns() - start_ns < expected_duration_ns:
            await self._semaphore.acquire()

            if time.time_ns() - start_ns >= expected_duration_ns:
                # If the time has expired while we were waiting for the semaphore,
                # do not send the credit
                return

            self.execute_async(
                self.credit_manager.drop_credit(
                    credit_phase=phase.type,
                    conversation_id=None,
                    credit_drop_ns=None,
                )
            )
            phase.sent += 1

    async def _execute_request_count_based_phase(self, phase: CreditPhaseStats) -> None:
        total: int = phase.total_requests  # type: ignore[assignment]

        while phase.sent < total:
            await self._semaphore.acquire()
            self.execute_async(
                self.credit_manager.drop_credit(
                    credit_phase=phase.type,
                    conversation_id=None,
                    credit_drop_ns=None,
                )
            )
            phase.sent += 1

    async def _execute_phases(self) -> None:
        for phase in self.phases:
            stats = CreditPhaseStats(
                type=phase,
                start_ns=time.time_ns(),
                total_requests=self.phase_stats[phase].total_requests,
                expected_duration_ns=self.phase_stats[phase].expected_duration_ns,
            )
            self.execute_async(
                self.credit_manager.publish_phase_start(
                    phase,
                    stats.start_ns,
                    stats.total_requests,
                    stats.expected_duration_ns,
                )
            )

            if stats.is_time_based:
                await self._execute_time_based_phase(stats)
            elif stats.total_requests and stats.total_requests > 0:
                await self._execute_request_count_based_phase(stats)
            else:
                raise InvalidStateError(
                    "Phase must have either a valid total or expected_duration_ns set"
                )

            # We have sent all the credits for this phase. We must continue to the next
            # phase even though not all the credits have been returned. This is because
            # we do not want a gap in the credit issuing.
            self.execute_async(
                self.credit_manager.publish_phase_sending_complete(
                    phase, time.time_ns()
                )
            )

    async def on_credit_return(self, message: CreditReturnMessage) -> None:
        """Process a credit return message."""

        # Release the semaphore, then call the superclass to handle the credit return
        self._semaphore.release()
        self.trace(lambda: f"TM: Released semaphore: {self._semaphore}")
        await super().on_credit_return(message)
