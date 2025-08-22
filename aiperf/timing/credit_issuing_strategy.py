# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from abc import ABC, abstractmethod

from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.exceptions import ConfigurationError
from aiperf.common.factories import AIPerfFactory
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.mixins import TaskManagerMixin
from aiperf.common.models import CreditPhaseConfig, CreditPhaseStats
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.credit_manager import CreditManagerProtocol


class CreditIssuingStrategy(TaskManagerMixin, ABC):
    """
    Base class for credit issuing strategies.
    """

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__()
        self.config = config
        self.credit_manager = credit_manager

        # This event is set when all phases are complete
        self.all_phases_complete_event = asyncio.Event()

        # This event is set when a single phase is complete
        self.phase_complete_event = asyncio.Event()

        # The running stats for each phase, keyed by phase type.
        self.phase_stats: dict[CreditPhase, CreditPhaseStats] = {}

        # The phases to run including their configuration, in order of execution.
        self.ordered_phase_configs: list[CreditPhaseConfig] = []

        self._setup_phase_configs()
        self._validate_phase_configs()

    def _setup_phase_configs(self) -> None:
        """Setup the phases for the strategy. This can be overridden in subclasses to modify the phases."""
        self._setup_warmup_phase_config()
        self._setup_profiling_phase_config()
        self.info(
            lambda: f"Credit issuing strategy {self.__class__.__name__} initialized with {len(self.ordered_phase_configs)} "
            f"phase(s): {self.ordered_phase_configs}"
        )

    def _setup_warmup_phase_config(self) -> None:
        """Setup the warmup phase. This can be overridden in subclasses to modify the warmup phase."""
        if self.config.warmup_request_count > 0:
            self.ordered_phase_configs.append(
                CreditPhaseConfig(
                    type=CreditPhase.WARMUP,
                    total_expected_requests=self.config.warmup_request_count,
                )
            )

    def _setup_profiling_phase_config(self) -> None:
        """Setup the profiling phase. This can be overridden in subclasses to modify the profiling phase."""
        self.ordered_phase_configs.append(
            CreditPhaseConfig(
                type=CreditPhase.PROFILING,
                total_expected_requests=self.config.request_count,
            )
        )

    def _validate_phase_configs(self) -> None:
        """Validate the phase configs."""
        for phase_config in self.ordered_phase_configs:
            if not phase_config.is_valid:
                raise ConfigurationError(
                    f"Phase {phase_config.type} is not valid. It must have either a valid total_expected_requests or expected_duration_sec set"
                )

    async def start(self) -> None:
        """Start the credit issuing strategy. This will launch the progress reporting loop, the
        warmup phase (if applicable), and the profiling phase, all in the background."""
        self.debug(
            lambda: f"Starting credit issuing strategy {self.__class__.__name__}"
        )
        self.all_phases_complete_event.clear()

        # Start the progress reporting loop in the background
        self.execute_async(self._progress_report_loop())

        # Execute the phases in the background
        self.execute_async(self._execute_phases())

        self.debug(
            lambda: f"Waiting for all credit phases to complete for {self.__class__.__name__}"
        )
        # Wait for all phases to complete before returning
        await self.all_phases_complete_event.wait()
        self.debug(lambda: f"All credit phases completed for {self.__class__.__name__}")

    async def _execute_phases(self) -> None:
        """Execute the all of the credit phases sequentially. This can be overridden in subclasses to modify the execution of the phases."""
        for phase_config in self.ordered_phase_configs:
            self.phase_complete_event.clear()

            phase_stats = CreditPhaseStats.from_phase_config(phase_config)
            phase_stats.start_ns = time.time_ns()
            self.phase_stats[phase_config.type] = phase_stats

            self.execute_async(
                self.credit_manager.publish_phase_start(
                    phase_config.type,
                    phase_stats.start_ns,
                    # Only one of the below will be set, this is already validated in the strategy
                    phase_config.total_expected_requests,
                    phase_config.expected_duration_sec,
                )
            )

            # This is implemented in subclasses
            await self._execute_single_phase(phase_stats)

            # We have sent all the credits for this phase, but we still will need to wait for the credits to be returned
            phase_stats.sent_end_ns = time.time_ns()
            self.execute_async(
                self.credit_manager.publish_phase_sending_complete(
                    phase_config.type, phase_stats.sent_end_ns, phase_stats.sent
                )
            )

            # Wait for the credits to be returned before continuing to the next phase
            await self.phase_complete_event.wait()

    @abstractmethod
    async def _execute_single_phase(self, phase_stats: CreditPhaseStats) -> None:
        """Execute a single phase. Should not return until the phase sending is complete. Must be implemented in subclasses."""
        raise NotImplementedError("Subclasses must implement this method")

    async def stop(self) -> None:
        """Stop the credit issuing strategy."""
        await self.cancel_all_tasks()

    async def _on_credit_return(self, message: CreditReturnMessage) -> None:
        """This is called by the credit manager when a credit is returned. It can be
        overridden in subclasses to handle the credit return."""
        if message.phase not in self.phase_stats:
            self.warning(
                f"Credit return message received for phase {message.phase} but no phase stats found"
            )
            return

        phase_stats = self.phase_stats[message.phase]
        phase_stats.completed += 1

        if (
            # If we have sent all the credits, check if this is the last one to be returned
            phase_stats.is_sending_complete
            and phase_stats.completed >= phase_stats.total_expected_requests  # type: ignore[operator]
        ):
            phase_stats.end_ns = time.time_ns()
            self.notice(f"Phase completed: {phase_stats}")

            self.execute_async(
                self.credit_manager.publish_phase_complete(
                    message.phase, phase_stats.completed, phase_stats.end_ns
                )
            )

            self.phase_complete_event.set()

            if phase_stats.type == CreditPhase.PROFILING:
                self.execute_async(self.credit_manager.publish_credits_complete())
                self.all_phases_complete_event.set()

            # We don't need to keep track of the phase stats anymore
            self.phase_stats.pop(message.phase)

    async def _progress_report_loop(self) -> None:
        """Report the progress at a fixed interval."""
        self.debug("Starting progress reporting loop")
        while not self.all_phases_complete_event.is_set():
            await asyncio.sleep(self.config.progress_report_interval_sec)

            for phase, stats in self.phase_stats.items():
                try:
                    await self.credit_manager.publish_progress(
                        phase, stats.sent, stats.completed
                    )
                except Exception as e:
                    self.error(f"Error publishing credit progress: {e}")
                except asyncio.CancelledError:
                    self.debug("Credit progress reporting loop cancelled")
                    return

        self.debug("All credits completed, stopping credit progress reporting loop")


class CreditIssuingStrategyFactory(AIPerfFactory[TimingMode, CreditIssuingStrategy]):
    """Factory for creating credit issuing strategies based on the timing mode."""
