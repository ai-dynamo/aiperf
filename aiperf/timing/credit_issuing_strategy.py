# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from abc import ABC, abstractmethod

from aiperf.common.constants import (
    DEFAULT_CREDIT_PROGRESS_REPORT_INTERVAL,
    NANOS_PER_SECOND,
)
from aiperf.common.enums import CreditPhase, TimingMode
from aiperf.common.exceptions import ConfigurationError
from aiperf.common.factories import AIPerfFactory
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.mixins import TaskManagerMixin
from aiperf.common.models import CreditPhaseConfig, CreditPhaseStats
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.credit_manager import CreditManagerProtocol
from aiperf.timing.request_cancellation_strategy import RequestCancellationStrategy


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

        self.cancellation_strategy = RequestCancellationStrategy(config)

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
        if self.config.benchmark_duration is not None:
            self.debug(
                f"Setting up duration-based profiling phase: expected_duration_sec={self.config.benchmark_duration}"
            )
            self.ordered_phase_configs.append(
                CreditPhaseConfig(
                    type=CreditPhase.PROFILING,
                    expected_duration_sec=self.config.benchmark_duration,
                )
            )
        else:
            self.debug(
                f"Setting up count-based profiling phase: total_expected_requests={self.config.request_count}"
            )
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
            await self._wait_for_phase_completion(phase_stats)

    async def _wait_for_phase_completion(self, phase_stats: CreditPhaseStats) -> None:
        """Wait for a phase to complete, with timeout for time-based phases."""
        if phase_stats.is_time_based:
            # For time-based phases, calculate how much time is left from the original duration
            elapsed_ns = time.time_ns() - phase_stats.start_ns
            elapsed_sec = elapsed_ns / NANOS_PER_SECOND
            remaining_sec = max(0, phase_stats.expected_duration_sec - elapsed_sec)

            grace_period = self.config.benchmark_grace_period
            total_timeout = remaining_sec + grace_period

            if grace_period > 0 and remaining_sec <= 0:
                self.info(
                    f"Benchmark duration elapsed for {phase_stats.type} phase, entering {grace_period}s grace period"
                )

                # Check if phase is already complete before starting grace period wait
                if phase_stats.in_flight == 0:
                    self.info(
                        f"Phase {phase_stats.type} has no in-flight requests, skipping grace period"
                    )
                    await self._force_phase_completion(
                        phase_stats, grace_period_timeout=False
                    )
                    return

            # Wait for either phase completion or timeout
            try:
                await asyncio.wait_for(
                    self.phase_complete_event.wait(), timeout=total_timeout
                )
                # Phase completed naturally
                return
            except asyncio.TimeoutError:
                # Total timeout elapsed, force completion
                if grace_period > 0 and remaining_sec <= 0:
                    self.info(
                        f"Grace period timeout elapsed for {phase_stats.type} phase"
                    )
                    await self._force_phase_completion(
                        phase_stats, grace_period_timeout=True
                    )
                else:
                    self.info(
                        f"Total timeout ({phase_stats.expected_duration_sec}s + {grace_period}s grace) elapsed for {phase_stats.type} phase"
                    )
                    await self._force_phase_completion(
                        phase_stats, grace_period_timeout=True
                    )
        else:
            # For request-count-based phases, wait indefinitely
            await self.phase_complete_event.wait()

    async def _wait_for_grace_period_completion(
        self, phase_stats: CreditPhaseStats
    ) -> None:
        """Wait for grace period completion after benchmark duration ends."""
        grace_period = self.config.benchmark_grace_period

        if grace_period <= 0:
            self.info(
                f"No grace period configured, forcing completion of {phase_stats.type} phase immediately"
            )
            await self._force_phase_completion(phase_stats, grace_period_timeout=False)
            return

        if self.phase_complete_event.is_set():
            self.info(
                f"Phase {phase_stats.type} already completed, no grace period needed"
            )
            return

        self.info(
            f"Entering grace period of {grace_period}s for {phase_stats.type} phase with {phase_stats.in_flight} in-flight requests"
        )

        try:
            await asyncio.wait_for(
                self.phase_complete_event.wait(), timeout=grace_period
            )
            self.info(
                f"All responses received during grace period for {phase_stats.type} phase"
            )
        except asyncio.TimeoutError:
            self.info(
                f"Grace period of {grace_period}s elapsed for {phase_stats.type} phase, forcing completion with {phase_stats.in_flight} remaining in-flight requests"
            )
            await self._force_phase_completion(phase_stats, grace_period_timeout=True)

    async def _force_phase_completion(
        self, phase_stats: CreditPhaseStats, grace_period_timeout: bool = False
    ) -> None:
        """Force completion of a phase when the duration has elapsed."""
        # Defensive check: ensure this phase is listed as an active phase.
        # In normal operation, this should be true, but it guards against edge
        # cases like duplicate timeout events or race conditions during shutdown.
        if phase_stats.type in self.phase_stats:
            phase_stats.end_ns = time.time_ns()
            self.notice(
                f"Phase force-completed due to grace period timeout: {phase_stats}"
            )

            self.execute_async(
                self.credit_manager.publish_phase_complete(
                    phase_stats.type,
                    phase_stats.completed,
                    phase_stats.end_ns,
                    timeout_triggered=True,
                )
            )

            self.phase_complete_event.set()

            if phase_stats.type == CreditPhase.PROFILING:
                await self.credit_manager.publish_credits_complete()
                self.all_phases_complete_event.set()

            self.phase_stats.pop(phase_stats.type)

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
            self.debug(
                f"Credit return message received for phase {message.phase} but no phase stats found"
            )
            return

        phase_stats = self.phase_stats[message.phase]
        phase_stats.completed += 1

        # Check if this phase is complete
        is_phase_complete = False
        if phase_stats.is_sending_complete:
            if phase_stats.is_request_count_based:
                # Request-count-based: complete when all requests are returned
                is_phase_complete = (
                    phase_stats.completed >= phase_stats.total_expected_requests
                )  # type: ignore[operator]
            else:
                # Time-based: complete when all in-flight requests complete before the timeout.
                # Duration timeout is handled separately with force_phase_completion.
                is_phase_complete = phase_stats.in_flight == 0

        if is_phase_complete:
            phase_stats.end_ns = time.time_ns()
            self.notice(f"Phase completed: {phase_stats}")

            self.execute_async(
                self.credit_manager.publish_phase_complete(
                    message.phase, phase_stats.completed, phase_stats.end_ns
                )
            )

            self.phase_complete_event.set()

            if phase_stats.type == CreditPhase.PROFILING:
                await self.credit_manager.publish_credits_complete()
                self.all_phases_complete_event.set()

            # We don't need to keep track of the phase stats anymore
            self.phase_stats.pop(message.phase)

    async def _progress_report_loop(self) -> None:
        """Report the progress at a fixed interval."""
        self.debug("Starting progress reporting loop")
        while not self.all_phases_complete_event.is_set():
            await asyncio.sleep(DEFAULT_CREDIT_PROGRESS_REPORT_INTERVAL)

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
