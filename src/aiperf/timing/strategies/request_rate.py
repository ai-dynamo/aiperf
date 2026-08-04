# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rate-based timing strategy for credit issuance."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from aiperf.common.constants import IS_LINUX, MILLIS_PER_SECOND, NANOS_PER_SECOND
from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.timing.intervals import IntervalGeneratorConfig

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.timing.branch_orchestrator import BranchOrchestrator
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.high_res_timer import ThreadPacer, TimerFdPacer
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker


class RequestRateStrategy(AIPerfLoggerMixin):
    """Issues credits at a target average rate with configurable arrival patterns.

    The arrival pattern (Constant, Poisson, Gamma, ConcurrencyBurst) determines
    inter-arrival time distribution. Rate is the average; actual intervals vary
    except for Constant which is deterministic.

    Subsequent turns have priority over new sessions to prevent starvation:
    multi-turn conversations hold session slots, so completing them frees slots
    faster than starting new ones.

    Terminology:
        - Credit: permission token to send one request (turn)
        - Session: a multi-turn conversation holding a concurrency slot
        - Turn: a single request/response in a conversation
        - Continuation turn: next turn of an in-progress session, queued after
          the previous turn completes (has priority over new sessions)
        - Rate interval: time between credit issuances (from arrival pattern)

    Flow::

        ┌──► wait for next rate interval ─┐
        │                                 │
        │                                 ▼
        │                 ┌───────────────────────────────┐
        │                 │   queued continuation turn?   │
        │                 └───────────────┬───────────────┘
        │                         no      │      yes
        │                 ┌───────────────┴───────────────┐
        │                 ▼                               ▼
        │     ┌───────────────────────┐       ┌───────────────────────┐
        │     │   start new session   │       │    issue next turn    │
        │     │                       │       │     (has priority)    │
        │     └───────────┬───────────┘       └───────────┬───────────┘
        │                 │                               │
        │                 └───────────────┬───────────────┘
        └─────────────────────────────────┘
                                          │ send credit
                          ────────────────┼────────────────
                                          ▼
                              ┌───────────────────────┐
                              │    worker (async)     │
                              └───────────┬───────────┘
                                          │ return credit
                                          ▼
                                    is final turn?
                                    no │      │ yes
                                ┌──────┘      └──────┐
                                ▼                    ▼
                       ┌─────────────────┐        (done)
                       │ queue next turn │
                       └────────┬────────┘
                                │
                                └───► back to continuation queue

    """

    def __init__(
        self,
        *,
        config: CreditPhaseConfig,
        conversation_source: ConversationSource,
        scheduler: LoopScheduler,
        stop_checker: StopConditionChecker,
        credit_issuer: CreditIssuer,
        lifecycle: PhaseLifecycle,
        branch_orchestrator: BranchOrchestrator | None = None,
        **kwargs,
    ):
        """Initialize rate timing strategy with all dependencies."""
        super().__init__(logger_name="RateTiming")
        self._config = config
        self._conversation_source: ConversationSource = conversation_source
        self._scheduler = scheduler
        self._stop_checker = stop_checker
        self._credit_issuer = credit_issuer
        self._lifecycle = lifecycle
        self._branch_orchestrator = branch_orchestrator

        # Queue for subsequent turns (turn_index > 0) waiting to be issued.
        # Populated by handle_credit_return when workers complete turns.
        # Drained by execute_phase at each rate interval (priority over new sessions).
        self._continuation_turns: asyncio.Queue[TurnToSend] = asyncio.Queue()

        interval_config = IntervalGeneratorConfig.from_phase_config(self._config)
        self.info(
            f"Creating interval generator: pattern={interval_config.arrival_pattern}, "
            f"rate={interval_config.request_rate}, smoothness={interval_config.arrival_smoothness}"
        )
        GeneratorClass = plugins.get_class(
            PluginType.ARRIVAL_PATTERN, interval_config.arrival_pattern
        )
        self._rate_generator = GeneratorClass(interval_config)
        self._rate_update_event = asyncio.Event()
        # High-resolution pacer for sub-millisecond sleep precision; created
        # lazily in execute_phase (requires a running event loop), closed on exit.
        self._pacer: TimerFdPacer | ThreadPacer | None = None

    async def setup_phase(self) -> None:
        """Setup the phase."""
        pass  # Already setup in __init__

    def _create_high_res_pacer(self) -> TimerFdPacer | ThreadPacer | None:
        """Create the best available high-resolution pacer, else None.

        Selection order: Linux timerfd (kernel hrtimer, in-loop fd) ->
        dedicated sleep thread (all platforms) -> event-loop timers (None).
        """
        if not Environment.TIMING.HIGH_RES_TIMER:
            return None
        if IS_LINUX:
            try:
                from aiperf.timing.high_res_timer import TimerFdPacer

                pacer = TimerFdPacer()
                self.info("Rate loop pacing via timerfd (kernel hrtimer)")
                return pacer
            except OSError as e:
                self.warning(f"timerfd unavailable ({e}); trying pacing thread")
        try:
            from aiperf.timing.high_res_timer import ThreadPacer

            pacer = ThreadPacer()
            self.info("Rate loop pacing via dedicated sleep thread")
            return pacer
        except (OSError, RuntimeError) as e:
            self.warning(
                f"high-res pacing unavailable ({e}); falling back to event-loop timers"
            )
            return None

    def _reschedule_for_rate_update(self, next_target_perf: float) -> float:
        """Clear a rate update and calculate its earliest eligible target."""
        self._rate_update_event.clear()
        return min(
            next_target_perf,
            time.perf_counter() + self._rate_generator.next_interval(),
        )

    async def _wait_for_pacer_or_rate_update(self, deadline_perf_s: float) -> bool:
        """Wait for a high-resolution deadline or a request-rate update."""
        if self._pacer is None:
            return False

        pacer_task = asyncio.create_task(self._pacer.sleep_until(deadline_perf_s))
        rate_update_task = asyncio.create_task(self._rate_update_event.wait())
        try:
            done, _ = await asyncio.wait(
                {pacer_task, rate_update_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in (pacer_task, rate_update_task):
                if not task.done():
                    task.cancel()
            # Shielded so a second cancellation of the enclosing task (eager
            # cancellation, or an external shutdown racing a first cancel)
            # cannot interrupt the gather and leave the pacer's tick wait
            # pending. The outer CancelledError is re-raised after cleanup.
            await asyncio.shield(
                asyncio.gather(pacer_task, rate_update_task, return_exceptions=True)
            )

        if pacer_task in done:
            pacer_task.result()
        return rate_update_task in done

    async def _wait_until_next_target(
        self, next_target_perf: float, now: float
    ) -> bool:
        """Wait until the target and return whether a rate update won."""
        sleep_duration = next_target_perf - now
        if sleep_duration <= 0:
            await yield_to_event_loop()
            return False

        if (
            self._pacer is not None
            and sleep_duration <= Environment.TIMING.RATE_RAMP_UPDATE_INTERVAL
        ):
            try:
                return await self._wait_for_pacer_or_rate_update(next_target_perf)
            except OSError as e:
                self.warning(
                    f"high-res pacing failed ({e}); falling back to event-loop timers"
                )
                self._pacer.close()
                self._pacer = None
                return await self._wait_until_next_target(
                    next_target_perf, time.perf_counter()
                )

        try:
            await asyncio.wait_for(
                self._rate_update_event.wait(), timeout=sleep_duration
            )
        except asyncio.TimeoutError:  # noqa: UP041 - distinct on Python 3.10
            return False
        return True

    async def execute_phase(self) -> None:
        """Execute request rate main loop until stop condition reached.

        Uses absolute scheduling: we track cumulative target times rather than
        sleeping for relative intervals. This prevents drift accumulation over
        many iterations (relative sleeps compound small timing errors).

        When falling behind, we re-anchor to "now" only once the backlog
        exceeds a bounded catch-up window (Environment.TIMING.MAX_CATCHUP_SECONDS)
        rather than on every oversleep. Event-loop timers oversleep sub-ms waits
        (~1ms under uvloop/libuv); unconditionally re-anchoring forfeits schedule
        on every tick and silently under-delivers high request rates. Genuine
        stalls beyond the window still re-anchor to avoid a burst storm.

        A high-resolution pacer (Environment.TIMING.HIGH_RES_TIMER) sleeps to
        absolute deadlines with ~50us precision (Linux timerfd, else a dedicated
        sleep thread), bypassing the event-loop timer wheel entirely.
        """
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        perf_start = self._lifecycle.started_at_perf_ns / NANOS_PER_SECOND
        next_target_perf = perf_start + self._rate_generator.next_interval()

        # The first turn of the next new session. Cached to avoid wasting samples from shuffle/sequential samplers.
        next_new_session_turn = self._conversation_source.next().build_first_turn()

        self._pacer = self._create_high_res_pacer()
        try:
            while True:
                now = time.perf_counter()

                # Behind schedule: re-anchor to now only past the bounded catch-up
                # window, so sub-ms oversleeps stay on the original schedule.
                if next_target_perf < now - Environment.TIMING.MAX_CATCHUP_SECONDS:
                    next_target_perf = now

                if self._rate_update_event.is_set():
                    next_target_perf = self._reschedule_for_rate_update(
                        next_target_perf
                    )
                    continue

                if await self._wait_until_next_target(next_target_perf, now):
                    next_target_perf = self._reschedule_for_rate_update(
                        next_target_perf
                    )
                    continue

                # Schedule next interval BEFORE issuing credit. This way, variable
                # credit issuance latency doesn't affect the timing of the next interval.
                next_target_perf += self._rate_generator.next_interval()

                # Priority 1: Queued continuation turns from completed previous turns.
                # These already hold session slots, so we just need prefill slots.
                if not self._continuation_turns.empty():
                    should_continue = await self._credit_issuer.issue_credit(
                        self._continuation_turns.get_nowait()
                    )
                    if not should_continue:
                        return

                # Priority 2: Start new session if allowed and slots available.
                # try_issue_credit returns None if no slot (skip interval), False if
                # stop condition reached (exit loop), True if issued successfully.
                elif self._stop_checker.can_start_new_session():
                    result = await self._credit_issuer.try_issue_credit(
                        next_new_session_turn
                    )
                    match result:
                        case True:  # Successfully issued credit
                            # Re-sample the next new turn for the next interval.
                            next_new_session_turn = (
                                self._conversation_source.next().build_first_turn()
                            )
                        case False:  # Stop condition reached
                            self.debug(
                                "Exiting: stop condition reached after try_issue_credit"
                            )
                            return
                        case None:  # No slot available, retry later
                            # Always yield to event loop to allow callbacks to run.
                            # This is especially critical to prevent deadlock in CONCURRENCY_BURST mode (0 interval).
                            await yield_to_event_loop()

                # Priority 3: No more sessions to start and queue is empty.
                # Check if we're done sending entirely.
                elif not self._stop_checker.can_send_any_turn():
                    return
                else:
                    # Can still send turns but queue is empty and can't start new
                    # sessions (session limit reached). Skip this interval and wait for
                    # continuation turns to arrive from callbacks.
                    # Always yield to event loop to allow callbacks to run.
                    # This is especially critical to prevent deadlock in CONCURRENCY_BURST mode (0 interval).
                    await yield_to_event_loop()
        finally:
            if self._pacer is not None:
                self._pacer.close()
                self._pacer = None

    async def handle_credit_return(
        self, credit: Credit, *, error: str | None = None
    ) -> None:
        """Queue the next turn of this conversation for the main loop.

        ``error`` is accepted for protocol parity (the callback handler passes
        the worker error string for context-overflow early termination in
        AgenticReplayStrategy); RequestRateStrategy ignores it.

        Called by CreditCallbackHandler when a worker completes a turn.
        If not the final turn, queues the next turn for the main rate loop
        to issue at the next available interval.

        The delay_ms from turn metadata (if present) is honored before queuing,
        simulating user "think time" between turns in a conversation.

        DAG sub-agent children (turns carrying ``parent_correlation_id``) are
        dispatched directly here rather than queued: their continuation turns
        arrive after the phase has been marked sending-complete for root
        sampling, so the main rate loop may have already exited. Direct
        dispatch avoids that race and keeps the DAG tree flowing.
        """
        if credit.is_final_turn:
            return

        meta = self._conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit, meta)

        if credit.agent_depth > 0:
            if meta.delay_ms is not None:
                self._scheduler.schedule_later(
                    meta.delay_ms / MILLIS_PER_SECOND,
                    self._issue_child_continuation_or_release(turn, credit),
                )
            else:
                await self._issue_child_continuation_or_release(turn, credit)
            return

        # Honor think-time delay from dataset metadata before queuing
        if meta.delay_ms is not None:
            self._scheduler.schedule_later(
                meta.delay_ms / MILLIS_PER_SECOND,
                self._continuation_turns.put(turn),
            )
        else:
            self._continuation_turns.put_nowait(turn)

    async def _issue_child_continuation_or_release(
        self, turn: TurnToSend, child_returning_credit: Credit
    ) -> None:
        """Issue a child continuation turn, releasing it from join tracking on refusal.

        ``CreditCallbackHandler`` already gates non-final child returns behind
        ``can_send_child_turn`` and routes stop-blocked ones to
        ``BranchOrchestrator.on_child_stopped`` directly. This wrapper covers
        the residual race where the cap is reached between that gate check
        and this issuance (e.g. a delay-deferred continuation arrives after
        the cap fires, or another in-flight credit pushes ``requests_sent``
        past the cap during a yield window).

        ``child_returning_credit`` is the just-returned credit whose
        continuation we're about to issue — its ``x_correlation_id`` is
        the *child*'s correlation id (the one tracked in
        ``BranchOrchestrator._child_to_parent``).

        We use ``dispatch_child_turn`` rather than ``issue_credit`` because
        the latter's False return is overloaded — it conflates "gate refused,
        credit NOT on wire" with "credit issued, was the final one". Calling
        ``on_child_stopped`` in the latter case prematurely drains a child
        whose return is still in flight, leading to a deadlock when that
        return arrives at an empty join. ``dispatch_child_turn`` returns
        True iff the credit was actually sent on the wire.
        """
        if (
            not await self._credit_issuer.dispatch_child_turn(turn)
            and self._branch_orchestrator is not None
        ):
            try:
                await self._branch_orchestrator.on_child_stopped(
                    child_returning_credit.x_correlation_id
                )
            except Exception:
                self.exception(
                    f"on_child_stopped failed for x_correlation_id="
                    f"{child_returning_credit.x_correlation_id}"
                )

    def set_request_rate(self, new_rate: float) -> None:
        """Update the request rate dynamically.

        Args:
            new_rate: New request rate (requests per second, must be > 0).
        """
        if new_rate <= 0:
            raise ValueError(f"Rate must be > 0, got {new_rate}")
        if getattr(self._rate_generator, "rate", None) == new_rate:
            return
        self._rate_generator.set_rate(new_rate)
        self._rate_update_event.set()
