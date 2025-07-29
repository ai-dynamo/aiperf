# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import TimingMode
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.models import CreditPhaseStats
from aiperf.services.timing_manager.config import TimingManagerConfig
from aiperf.services.timing_manager.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
    CreditManagerProtocol,
)


@CreditIssuingStrategyFactory.register(TimingMode.CONCURRENCY)
class ConcurrencyStrategy(CreditIssuingStrategy):
    """Class for concurrency credit issuing strategy."""

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__(config=config, credit_manager=credit_manager)

        # If the concurrency is larger than the total number of requests, it does not matter
        # as it is simply an upper bound that will never be reached
        self._semaphore = asyncio.Semaphore(value=config.concurrency)

    async def _execute_single_phase(self, phase_stats: CreditPhaseStats) -> None:
        """Execute a single credit phase. This will not return until the phase sending is complete."""
        if phase_stats.is_time_based:
            await self._execute_time_based_phase(phase_stats)
        elif phase_stats.is_request_count_based:
            await self._execute_request_count_based_phase(phase_stats)
        else:
            raise InvalidStateError(
                "Phase must have either a valid total or expected_duration_ns set"
            )

    async def _execute_time_based_phase(self, phase_stats: CreditPhaseStats) -> None:
        """Execute a time-based phase."""

        # Start the internal loop in a task so that we can cancel it when the time expires
        time_task = asyncio.create_task(
            self._execute_time_based_phase_internal(phase_stats)
        )

        # Calculate how long until the phase expires
        sleep_time_sec = (
            (phase_stats.start_ns / NANOS_PER_SECOND)  # type: ignore
            + phase_stats.expected_duration_sec
            - time.time()
        )
        self.trace(
            lambda: f"Time-based phase will expire in {sleep_time_sec} seconds: {phase_stats}"
        )

        # Sleep until the phase expires, and then cancel the task
        await asyncio.sleep(sleep_time_sec)
        time_task.cancel()
        self.debug(lambda: f"Time-based phase execution expired: {phase_stats}")
        # Note, not awaiting the task here as we do not want to block moving to the next phase

    async def _execute_time_based_phase_internal(
        self, phase_stats: CreditPhaseStats
    ) -> None:
        """Execute a the internal loop for a time-based phase. This will be called within a task and cancelled when the time expires."""

        self.trace(
            lambda: f"_execute_time_based_phase_internal loop entered: {phase_stats}"
        )

        # This will loop until the task is cancelled
        while True:
            try:
                # Acquire the semaphore. Once we hit the concurrency limit, this will block until a credit is returned
                await self._semaphore.acquire()
                self.execute_async(
                    self.credit_manager.drop_credit(
                        credit_phase=phase_stats.type,
                    )
                )
                phase_stats.sent += 1
            except asyncio.CancelledError:
                self.trace(
                    lambda: f"_execute_time_based_phase_internal loop exited: {phase_stats}"
                )
                self.debug("Time-based phase execution expired")
                break

    async def _execute_request_count_based_phase(
        self, phase_stats: CreditPhaseStats
    ) -> None:
        self.trace(
            lambda: f"_execute_request_count_based_phase loop entered: {phase_stats}"
        )

        total: int = phase_stats.total_expected_requests  # type: ignore

        while phase_stats.sent < total:
            await self._semaphore.acquire()
            self.execute_async(
                self.credit_manager.drop_credit(
                    credit_phase=phase_stats.type,
                )
            )
            phase_stats.sent += 1

        self.trace(
            lambda: f"_execute_request_count_based_phase loop exited: {phase_stats}"
        )

    async def _on_credit_return(self, message: CreditReturnMessage) -> None:
        """Process a credit return message."""

        # Release the semaphore to allow another credit to be issued,
        # then call the superclass to handle the credit return like normal
        self._semaphore.release()
        self.trace(lambda: f"Credit return released semaphore: {self._semaphore}")
        await super()._on_credit_return(message)
