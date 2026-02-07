# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase runner for credit phase lifecycle management.

Coordinates phase execution: create components → start → wait for sends → wait for returns → complete.
Owns the LoopScheduler and all per-phase components (lifecycle, progress, stop_checker, credit_issuer).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.mixins import TaskManagerMixin
from aiperf.credit.issuer import CreditIssuer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TimingMode
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker
from aiperf.timing.ramping import RampConfig, Ramper, RampType
from aiperf.timing.strategies.core import RateSettableProtocol
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.common.models import CreditPhaseStats
    from aiperf.credit.callback_handler import CreditCallbackHandler
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.phase.publisher import PhasePublisher
    from aiperf.timing.request_cancellation import RequestCancellationSimulator
    from aiperf.timing.strategies.core import TimingStrategyProtocol


class PhaseRunner(TaskManagerMixin):
    """Executes credit phases with full lifecycle management.

    Creates all per-phase components lazily during run():
    - LoopScheduler (SINGLE owner - key architectural decision)
    - PhaseLifecycle (state machine)
    - PhaseProgressTracker (wraps counter + events)
    - StopConditionChecker (evaluates stop conditions)
    - CreditIssuer (issues credits with concurrency control)

    Lifecycle:
        1. Create components
        2. Register phase with callback handler
        3. Setup timing strategy with injected dependencies
        4. Start phase (mark started, publish)
        5. Execute timing strategy (with timeout)
        6. Wait for returns (with grace period)
        7. Complete phase (mark complete, publish)
        8. Cleanup (cancel scheduler, stop rampers)

    Component Ownership Diagram:
        PhaseRunner (owns)
            ├── LoopScheduler
            ├── PhaseLifecycle
            ├── PhaseProgressTracker
            │       └── CreditCounter (owned by tracker)
            ├── StopConditionChecker (reads lifecycle + counter)
            └── CreditIssuer (uses stop_checker, progress, concurrency, router)
    """

    def __init__(
        self,
        *,
        config: CreditPhaseConfig,
        conversation_source: ConversationSource,
        phase_publisher: PhasePublisher,
        credit_router: CreditRouterProtocol,
        concurrency_manager: ConcurrencyManager,
        cancellation_policy: RequestCancellationSimulator,
        callback_handler: CreditCallbackHandler,
        url_selection_strategy: URLSelectionStrategyProtocol | None = None,
        **kwargs,
    ) -> None:
        """Initialize phase runner.

        Args:
            config: Phase configuration (phase enum, stop conditions, concurrency limits).
            conversation_source: Source for conversation data (shared across phases).
            phase_publisher: Publishes phase lifecycle events to message bus.
            credit_router: Routes credits to workers (for cancel_all_credits on timeout).
            concurrency_manager: Manages session and prefill concurrency slots.
            cancellation_policy: Determines credit cancellation delays.
            callback_handler: Handles credit returns and TTFT events.
            url_selection_strategy: Optional URL selection strategy for multi-URL
                load balancing. Passed to CreditIssuer.
        """
        super().__init__(**kwargs)
        self._config = config
        self._conversation_source = conversation_source

        # For FIXED_SCHEDULE mode, use actual dataset size instead of config values.
        # Config values may reflect pre-filtered file size, but dataset_metadata
        # reflects the actual filtered dataset after start/end offset filtering.
        metadata = conversation_source.dataset_metadata
        if config.timing_mode == TimingMode.FIXED_SCHEDULE and metadata:
            self._config = config.model_copy(
                update={
                    "total_expected_requests": metadata.total_turn_count,
                    "expected_num_sessions": len(metadata.conversations),
                }
            )
        self._phase_publisher = phase_publisher
        self._credit_router = credit_router
        self._concurrency_manager = concurrency_manager
        self._cancellation_policy = cancellation_policy
        self._callback_handler = callback_handler
        self._on_phase_complete: Callable[[], None] | None = None

        # Per-phase components - order matters
        self._scheduler = LoopScheduler()
        self._lifecycle = PhaseLifecycle(self._config)
        self._progress = PhaseProgressTracker(self._config)
        self._stop_checker = StopConditionChecker(
            config=self._config,
            lifecycle=self._lifecycle,
            counter=self._progress.counter,
        )
        self._credit_issuer = CreditIssuer(
            phase=self._config.phase,
            stop_checker=self._stop_checker,
            progress=self._progress,
            concurrency_manager=self._concurrency_manager,
            credit_router=self._credit_router,
            cancellation_policy=self._cancellation_policy,
            lifecycle=self._lifecycle,
            url_selection_strategy=url_selection_strategy,
        )

        # Execution state
        self._execution_task: asyncio.Task | None = None
        self._progress_task: asyncio.Task | None = None
        self._return_wait_task: asyncio.Task | None = None
        self._was_cancelled = False
        self._rampers: list[Ramper] = []

    @property
    def phase(self) -> CreditPhase:
        """Phase enum (WARMUP or PROFILING)."""
        return self._config.phase

    def set_phase_complete_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to invoke when phase fully completes.

        Used for seamless phases to notify the orchestrator when the background
        return wait task finishes, allowing cleanup of the runner from active list.
        """
        self._on_phase_complete = callback

    def cancel(self) -> None:
        """Cancel the phase runner (external cancellation like Ctrl+C)."""
        self._was_cancelled = True
        self._lifecycle.cancel()
        if self._execution_task:
            self._execution_task.cancel()
        if self._progress_task:
            self._progress_task.cancel()
        if self._return_wait_task:
            self._return_wait_task.cancel()
        for ramper in self._rampers:
            ramper.stop()
        self._scheduler.cancel_all()

    def _on_return_wait_complete(self, task: asyncio.Task) -> None:
        """Handle completion of background return wait task (seamless mode).

        Called when _return_wait_task finishes. Cancels progress reporting and
        notifies the orchestrator via on_phase_complete callback.
        """
        if self._progress_task:
            self._progress_task.cancel()

        if self._on_phase_complete:
            self._on_phase_complete()

    async def run(
        self,
        is_final_phase: bool,
    ) -> CreditPhaseStats:
        """Execute phase with full lifecycle management.

        Lifecycle: register callback handler → setup strategy → configure rampers →
        start phase → execute timing strategy → wait for sends → wait for returns →
        complete phase → cleanup (cancel scheduler, stop rampers).

        Args:
            is_final_phase: True if this is the last phase. Non-final seamless phases
                spawn background return-wait task; final phases wait synchronously.

        Returns:
            CreditPhaseStats snapshot of final phase state.
        """
        StrategyClass = plugins.get_class(
            PluginType.TIMING_STRATEGY, self._config.timing_mode
        )
        strategy: TimingStrategyProtocol = StrategyClass(
            config=self._config,
            conversation_source=self._conversation_source,
            scheduler=self._scheduler,
            stop_checker=self._stop_checker,
            credit_issuer=self._credit_issuer,
            lifecycle=self._lifecycle,
        )

        try:
            # Register phase with callback handler (BEFORE any credits are sent)
            self._callback_handler.register_phase(
                phase=self._config.phase,
                progress=self._progress,
                lifecycle=self._lifecycle,
                stop_checker=self._stop_checker,
                strategy=strategy,
            )

            self._concurrency_manager.configure_for_phase(
                self._config.phase,
                self._config.concurrency,
                self._config.prefill_concurrency,
            )

            await strategy.setup_phase()

            self._create_rampers(strategy)

            self._lifecycle.start()
            stats = self._progress.create_stats(self._lifecycle)
            self.notice(self._format_phase_started(stats))
            await self._phase_publisher.publish_phase_start(self._config, stats)

            self._progress_task = self.execute_async(self._progress_report_loop())

            # Start rampers BEFORE execution to ensure concurrency limits are applied
            # from the start. Otherwise, credits could be issued at full concurrency
            # before the ramper sets the initial (lower) limit.
            for ramper in self._rampers:
                ramper.start()

            self._execution_task = self.execute_async(strategy.execute_phase())

            await self._wait_for_sending_complete()

            if self._was_cancelled:
                if not self._lifecycle.is_complete:
                    self._lifecycle.mark_complete(grace_period_triggered=True)
                    self._progress.freeze_completed_counts()
                self._progress.all_credits_returned_event.set()
                return self._progress.create_stats(self._lifecycle)

            # 11. Seamless mode: phase flows into next without waiting for returns
            #     Progress task continues in background until phase complete
            if self._config.seamless and not is_final_phase:
                self._return_wait_task = self.execute_async(
                    self._wait_for_returning_complete()
                )
                self._return_wait_task.add_done_callback(self._on_return_wait_complete)
            else:
                await self._wait_for_returning_complete()
                self._progress_task.cancel()

            for ramper in self._rampers:
                ramper.stop()
            self._scheduler.cancel_all()

            return self._progress.create_stats(self._lifecycle)

        except Exception as e:
            # TODO: This can be improved a bit by having a better way to notify other services
            # and the system controller of a failure in the benchmark.
            # If there is an error while setting up or executing the phase,
            # we need to flush it through the lifecycle to ensure the other services
            # are notified that the phase has ended, and the benchmark does not hang forever.
            self.error(f"Error executing phase {self._config.phase.title}: {e!r}")
            if not self._was_cancelled:
                self.cancel()

            if not self._lifecycle.is_started:
                self._lifecycle.start()
                stats = self._progress.create_stats(self._lifecycle)
                await self._phase_publisher.publish_phase_start(self._config, stats)

            if not self._lifecycle.is_sending_complete:
                self._lifecycle.mark_sending_complete(timeout_triggered=False)
                self._progress.freeze_sent_counts()
                self._progress.all_credits_sent_event.set()
                stats = self._progress.create_stats(self._lifecycle)
                await self._phase_publisher.publish_phase_sending_complete(stats)

            if not self._lifecycle.is_complete:
                self._lifecycle.mark_complete(grace_period_triggered=False)
                self._progress.freeze_completed_counts()
                self._progress.all_credits_returned_event.set()
                stats = self._progress.create_stats(self._lifecycle)
                await self._phase_publisher.publish_phase_complete(stats)

            raise e

    def _create_rampers(self, strategy: TimingStrategyProtocol) -> None:
        """Create rampers for concurrency and rate if ramp durations are configured.

        Concurrency rampers use stepped mode (discrete integer steps), starting at 1.
        Rate rampers use continuous mode (smooth float interpolation), starting at a
        rate proportional to target (to avoid issues when target < 1 QPS).
        """
        self._rampers = []
        config = self._config

        # Session concurrency ramper (stepped mode)
        if config.concurrency_ramp_duration_sec and config.concurrency:
            self.info(
                f"Starting session concurrency ramp: 1 → {config.concurrency} "
                f"over {config.concurrency_ramp_duration_sec}s"
            )
            ramp_config = RampConfig(
                ramp_type=RampType.LINEAR,
                start=1,
                target=config.concurrency,
                duration_sec=config.concurrency_ramp_duration_sec,
            )

            def setter(limit: float) -> None:
                return self._concurrency_manager.set_session_limit(
                    config.phase, int(limit)
                )

            self._rampers.append(Ramper(setter=setter, config=ramp_config))

        # Prefill concurrency ramper (stepped mode)
        if config.prefill_concurrency_ramp_duration_sec and config.prefill_concurrency:
            self.info(
                f"Starting prefill concurrency ramp: 1 → {config.prefill_concurrency} "
                f"over {config.prefill_concurrency_ramp_duration_sec}s"
            )
            ramp_config = RampConfig(
                ramp_type=RampType.LINEAR,
                start=1,
                target=config.prefill_concurrency,
                duration_sec=config.prefill_concurrency_ramp_duration_sec,
            )

            def setter(limit: float) -> None:
                return self._concurrency_manager.set_prefill_limit(
                    config.phase, int(limit)
                )

            self._rampers.append(Ramper(setter=setter, config=ramp_config))

        # Request rate ramper (continuous mode via update_interval)
        if config.request_rate_ramp_duration_sec and config.request_rate:
            # Start at one linear increment (proportional to target, not fixed 1 QPS).
            # This avoids awkward cases where target < 1 QPS would actually increase.
            update_interval = Environment.TIMING.RATE_RAMP_UPDATE_INTERVAL
            start_rate = config.request_rate * (
                update_interval / config.request_rate_ramp_duration_sec
            )
            self.info(
                f"Starting request rate ramp: {start_rate:.2f} → {config.request_rate} QPS "
                f"over {config.request_rate_ramp_duration_sec}s"
            )
            ramp_config = RampConfig(
                ramp_type=RampType.LINEAR,
                start=start_rate,
                target=config.request_rate,
                duration_sec=config.request_rate_ramp_duration_sec,
                update_interval=update_interval,
            )
            if isinstance(strategy, RateSettableProtocol):
                self._rampers.append(
                    Ramper(setter=strategy.set_request_rate, config=ramp_config)
                )
            else:
                self.warning(
                    f"Strategy {strategy.__class__.__name__} does not implement RateSettableProtocol. "
                    "Request rate will be fixed at the target value."
                )

    def _format_phase_started(self, stats: CreditPhaseStats) -> str:
        """Format a concise log message for phase start."""
        parts = [f"Phase {stats.phase} started"]
        targets = []
        if stats.total_expected_requests:
            targets.append(f"{stats.total_expected_requests:,} requests")
        if stats.expected_duration_sec:
            targets.append(f"{stats.expected_duration_sec:.1f}s duration")
        if stats.expected_num_sessions:
            targets.append(f"{stats.expected_num_sessions:,} sessions")
        if targets:
            parts.append(f"target: {', '.join(targets)}")
        return " | ".join(parts)

    def _format_phase_sending_complete(self, stats: CreditPhaseStats) -> str:
        """Format a concise log message for phase sending complete."""
        parts = [f"Phase {stats.phase} sending complete"]
        parts.append(
            f"sent={stats.requests_sent:,}, "
            f"completed={stats.requests_completed:,}, "
            f"in_flight={stats.in_flight_requests:,}"
        )
        if stats.sent_sessions > 0:
            parts.append(
                f"sessions: sent={stats.sent_sessions:,}, "
                f"completed={stats.completed_sessions:,}"
            )
        if stats.timeout_triggered:
            parts.append("timeout_triggered=True")
        return " | ".join(parts)

    def _format_phase_complete(self, stats: CreditPhaseStats) -> str:
        """Format a concise log message for phase complete."""
        parts = [f"Phase {stats.phase} complete"]
        parts.append(
            f"completed={stats.final_requests_completed:,}, "
            f"cancelled={stats.final_requests_cancelled:,}, "
            f"errors={stats.final_request_errors:,}"
        )
        if stats.final_sent_sessions and stats.final_sent_sessions > 0:
            parts.append(
                f"sessions: completed={stats.final_completed_sessions:,}, "
                f"cancelled={stats.final_cancelled_sessions:,}"
            )
        elapsed = stats.requests_elapsed_time
        parts.append(f"elapsed={elapsed:.2f}s")
        if stats.grace_period_timeout_triggered:
            parts.append("grace_period_timeout=True")
        if stats.was_cancelled:
            parts.append("was_cancelled=True")
        return " | ".join(parts)

    async def _wait_for_sending_complete(self) -> None:
        """Wait for phase to send all credits (with timeout).

        Uses lifecycle.time_left_in_seconds() for timeout duration.
        On timeout or completion, cancels pending scheduled requests,
        freezes sent counts, and marks sending complete.
        """
        timed_out = False
        try:
            timeout = self._lifecycle.time_left_in_seconds()
            timed_out = await self._wait_for_event_with_timeout(
                name=f"{self._config.phase} phase sending",
                event=self._progress.all_credits_sent_event,
                timeout=timeout,
                task_to_cancel=self._execution_task,
                set_event_on_timeout=True,
            )
        except Exception as e:
            self.error(
                f"Error waiting for phase {self._config.phase} to send all credits: {e!r}"
            )
        finally:
            if not self._lifecycle.is_sending_complete:
                self._lifecycle.mark_sending_complete(timeout_triggered=timed_out)
                self._progress.freeze_sent_counts()
                self._scheduler.cancel_all_pending()
                self._progress.all_credits_sent_event.set()

            stats = self._progress.create_stats(self._lifecycle)
            self.notice(self._format_phase_sending_complete(stats))
            await self._phase_publisher.publish_progress(stats)
            await self._phase_publisher.publish_phase_sending_complete(stats)

    async def _wait_for_returning_complete(self) -> None:
        """Wait for all credits to return (with grace period).

        Multi-stage process on timeout:
        1. Initial wait with grace period timeout
        2. If timed out: cancel_all_credits() via credit router
        3. Wait for cancelled credits to drain (CANCEL_DRAIN_TIMEOUT)
        4. If drain times out: release stuck concurrency slots and force completion
        """
        timed_out = False
        try:
            if self._progress.check_all_returned_or_cancelled():
                self.info(
                    "All credits already returned. Setting all_credits_returned_event."
                )
                self._progress.all_credits_returned_event.set()
                return

            timeout = self._lifecycle.time_left_in_seconds(include_grace_period=True)
            timed_out = await self._wait_for_event_with_timeout(
                name=f"{self._config.phase} phase credits returned",
                event=self._progress.all_credits_returned_event,
                timeout=timeout,
                task_to_cancel=None,
                set_event_on_timeout=False,
            )
            if timed_out:
                stats = self._progress.create_stats(self._lifecycle)
                self.warning(
                    f"Phase {self._config.phase} timed out, cancelling all "
                    f"credits. Stats: sent={stats.requests_sent}, "
                    f"completed={stats.requests_completed}, "
                    f"cancelled={stats.requests_cancelled}, "
                    f"in_flight={stats.in_flight_requests}"
                )
                await self._credit_router.cancel_all_credits()
                stats = self._progress.create_stats(self._lifecycle)
                need = (
                    stats.final_requests_sent
                    - stats.requests_completed
                    - stats.requests_cancelled
                )
                self.info(
                    f"Waiting for all cancelled credits to be returned for "
                    f"phase {self._config.phase}. Need {need} more credits."
                )
                # Wait with timeout to avoid hanging indefinitely
                drain_timeout = Environment.TIMING.CANCEL_DRAIN_TIMEOUT
                try:
                    await asyncio.wait_for(
                        self._progress.all_credits_returned_event.wait(),
                        timeout=drain_timeout,
                    )
                    self.info(
                        f"All cancelled credits returned for phase {self._config.phase}"
                    )
                except asyncio.TimeoutError:
                    self.error(
                        f"Timeout waiting {drain_timeout}s for cancelled credits to return. "
                        f"Some credits may be stuck. Forcing phase completion."
                    )
                    # Release slots for sessions/requests that will never return.
                    self._release_stuck_slots()

                    if not self._lifecycle.is_complete:
                        self._lifecycle.mark_complete(grace_period_triggered=True)
                        self._progress.freeze_completed_counts()
                    self._progress.all_credits_returned_event.set()
        finally:
            if not self._lifecycle.is_complete:
                self._lifecycle.mark_complete(grace_period_triggered=timed_out)
                self._progress.freeze_completed_counts()
            stats = self._progress.create_stats(self._lifecycle)
            self.notice(self._format_phase_complete(stats))
            await self._phase_publisher.publish_progress(stats)
            await self._phase_publisher.publish_phase_complete(stats)

    def _release_stuck_slots(self) -> None:
        """Release concurrency slots for credits that will never return."""
        session_released, prefill_released = (
            self._concurrency_manager.release_stuck_slots(self._config.phase)
        )
        if session_released or prefill_released:
            self.warning(
                f"Released stuck slots for phase {self._config.phase}: "
                f"session={session_released}, prefill={prefill_released}"
            )

    async def _wait_for_event_with_timeout(
        self,
        *,
        name: str,
        event: asyncio.Event,
        timeout: float | None,
        task_to_cancel: asyncio.Task | None,
        set_event_on_timeout: bool = False,
    ) -> bool:
        """Wait for event with optional timeout.

        Args:
            name: The name of the event to wait for.
            event: The event to wait for.
            timeout: The timeout in seconds.
                If None, the event will be waited for indefinitely.
                If timeout is <= 0, returns immediately with timeout.
            task_to_cancel: The optional task to cancel when the timeout occurs.
            set_event_on_timeout: If True, the event will also be set when the timeout occurs.

        Returns:
            True if the event timed out, False if the event was set before timeout.
        """
        if timeout is None:
            self.debug(lambda: f"Waiting for event '{name}' indefinitely")
            await event.wait()
            return False

        def _on_timeout() -> bool:
            self.info(f"Timeout of {timeout}s elapsed for event '{name}'")
            if set_event_on_timeout:
                event.set()
            if task_to_cancel:
                task_to_cancel.cancel()
            return True

        if timeout <= 0:
            self.debug(lambda: f"Timeout already elapsed for event '{name}'")
            return _on_timeout()

        try:
            self.info(f"Waiting for event '{name}' with timeout of {timeout}s")
            await asyncio.wait_for(event.wait(), timeout=timeout)
            self.debug(lambda: f"Event '{name}' set before timeout of {timeout}s")
            return False

        except asyncio.TimeoutError:
            return _on_timeout()

        except Exception as e:
            self.error(f"Error waiting for event '{name}' with timeout: {e!r}")
            raise

    async def _progress_report_loop(self) -> None:
        """Publish phase progress stats at regular intervals.

        Runs as a background task until the phase is complete.
        Publishes progress at CREDIT_PROGRESS_REPORT_INTERVAL intervals.
        """
        self.debug(f"Starting progress reporting loop for phase {self._config.phase}")
        try:
            while True:
                try:
                    stats = self._progress.create_stats(self._lifecycle)
                    await self._phase_publisher.publish_progress(stats)
                except Exception as e:
                    self.error(
                        f"Error publishing progress for phase {self._config.phase}: {e!r}"
                    )
                await asyncio.sleep(Environment.SERVICE.CREDIT_PROGRESS_REPORT_INTERVAL)
        except asyncio.CancelledError:
            self.debug(
                f"Progress reporting loop cancelled for phase {self._config.phase}"
            )
            raise
