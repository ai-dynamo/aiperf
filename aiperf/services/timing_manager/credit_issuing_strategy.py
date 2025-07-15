# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from abc import ABC, abstractmethod

from aiperf.common.credit_models import CreditPhaseStats
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.mixins import AIPerfLoggerMixin, AsyncTaskManagerMixin
from aiperf.services.timing_manager.config import TimingManagerConfig
from aiperf.services.timing_manager.credit_manager import CreditManagerProtocol


class CreditIssuingStrategy(AsyncTaskManagerMixin, AIPerfLoggerMixin, ABC):
    """
    Base class for credit issuing strategies.
    """

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__()
        self.config = config
        self.credit_manager = credit_manager

        # The stats for each phase, keyed by phase type
        self.phase_stats: dict[CreditPhase, CreditPhaseStats] = {}
        # The phases to run, in order
        self.phases: list[CreditPhase] = []

    async def start(self) -> None:
        """Start the credit issuing strategy. This will launch the progress reporting loop, the
        warmup phase (if applicable), and the profiling phase, all in the background."""

        # Start the progress reporting loop in the background
        self.execute_async(self._progress_report_loop())

        # Execute the phases in the background
        self.execute_async(self._execute_phases())

    @abstractmethod
    async def _execute_phases(self) -> None:
        """Execute the phases in the background."""
        raise NotImplementedError("Subclasses must implement this method")

    async def stop(self) -> None:
        """Stop the credit issuing strategy."""
        await self.cancel_all_tasks()

    async def on_credit_return(self, message: CreditReturnMessage) -> None:
        """This is called by the credit manager when a credit is returned. It can be
        overridden in subclasses to handle the credit return."""
        phase_stats = self.phase_stats[message.phase]
        phase_stats.completed += 1

        if (
            # If we have sent all the credits, check if this is the last one to be returned
            phase_stats.is_sending_complete
            and phase_stats.completed >= phase_stats.total_requests  # type: ignore[operator]
        ):
            phase_stats.end_ns = time.time_ns()
            self.info(lambda: f"TM: Phase completed: {phase_stats}")

            self.execute_async(
                self.credit_manager.publish_phase_complete(
                    message.phase, phase_stats.end_ns
                )
            )

            if self.all_phases_complete():
                self.execute_async(self.credit_manager.publish_credits_complete())

    async def _progress_report_loop(self) -> None:
        """Report the progress at a fixed interval."""
        self.debug("TM: Starting progress reporting loop")
        while not self.all_phases_complete():
            await asyncio.sleep(1)  # TODO: Make this configurable
            for phase, stats in self.phase_stats.items():
                try:
                    await self.credit_manager.publish_progress(
                        phase, stats.sent, stats.completed
                    )
                except Exception as e:
                    self.error(f"TM: Error publishing progress: {e}")
                except asyncio.CancelledError:
                    self.debug("TM: Progress reporting loop cancelled")
                    return

        self.debug("TM: All credits completed, stopping progress reporting loop")

    def all_phases_complete(self) -> bool:
        """Check if all phases are complete."""
        return all(phase.is_complete for phase in self.phase_stats.values())
