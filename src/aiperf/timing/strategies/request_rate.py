# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rate-based timing strategy for credit issuance."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from aiperf.common.constants import MILLIS_PER_SECOND, NANOS_PER_SECOND
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.timing.intervals import IntervalGeneratorConfig

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
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
        branch_orchestrator: Any = None,
        **kwargs,
    ):
        """Initialize rate timing strategy with all dependencies.

        ``branch_orchestrator`` is the DAG ``BranchOrchestrator`` when running
        a DAG benchmark; ``None`` for non-DAG runs. When non-final DAG child
        credits return, the strategy routes their continuation through
        ``credit_issuer.dispatch_child_turn`` (bypassing the rate-loop
        continuation queue, since children share their parent's session
        slot); if the cap rejects the dispatch, the strategy notifies
        ``branch_orchestrator.on_child_stopped`` so the orchestrator releases
        the child's x_correlation_id.
        """
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

    async def setup_phase(self) -> None:
        """Setup the phase."""
        pass  # Already setup in __init__

    async def execute_phase(self) -> None:
        """Execute request rate main loop until stop condition reached.

        Uses absolute scheduling: we track cumulative target times rather than
        sleeping for relative intervals. This prevents drift accumulation over
        many iterations (relative sleeps compound small timing errors).

        When falling behind (credit issuance took longer than the interval),
        we reset to "now" rather than trying to catch up. This prioritizes
        maintaining throughput (preventing bursts) over preserving the exact
        arrival distribution.
        """
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        perf_start = self._lifecycle.started_at_perf_ns / NANOS_PER_SECOND
        next_target_perf = perf_start + self._rate_generator.next_interval()

        # The first turn of the next new session. Cached to avoid wasting samples from shuffle/sequential samplers.
        next_new_session_turn = self._conversation_source.next().build_first_turn()

        while True:
            next_target_perf = await self._wait_for_next_interval(next_target_perf)
            # Schedule next interval BEFORE issuing credit. This way, variable
            # credit issuance latency doesn't affect the timing of the next interval.
            next_target_perf += self._rate_generator.next_interval()

            done, next_new_session_turn = await self._issue_next_credit(
                next_new_session_turn
            )
            if done:
                return

    async def _wait_for_next_interval(self, next_target_perf: float) -> float:
        """Sleep until the next target perf time, returning the (possibly reset) target.

        Resets the target to `now` if we're behind schedule to avoid catch-up bursts.
        """
        now = time.perf_counter()

        # Behind schedule: reset to now instead of sending a burst to catch up.
        # This sacrifices inter-arrival distribution accuracy for stable throughput.
        if next_target_perf < now:
            next_target_perf = now

        sleep_duration = next_target_perf - now
        if sleep_duration > 0:
            await asyncio.sleep(sleep_duration)
        else:
            # CRITICAL: Always yield to event loop to allow callbacks to run.
            # Without this, CONCURRENCY_BURST mode (0 interval) busy-loops and
            # starves credit return callbacks, causing deadlock.
            await yield_to_event_loop()
        return next_target_perf

    async def _issue_next_credit(
        self, next_new_session_turn: TurnToSend
    ) -> tuple[bool, TurnToSend]:
        """Issue one credit for this interval by priority: continuation, new session, or skip.

        Returns (done, next_new_session_turn). `done=True` signals the main loop to exit.
        """
        # Priority 1: Queued continuation turns from completed previous turns.
        # These already hold session slots, so we just need prefill slots.
        if not self._continuation_turns.empty():
            should_continue = await self._credit_issuer.issue_credit(
                self._continuation_turns.get_nowait()
            )
            return (not should_continue, next_new_session_turn)

        # Priority 2: Start new session if allowed and slots available.
        if self._stop_checker.can_start_new_session():
            return await self._try_start_new_session(next_new_session_turn)

        # Priority 3: No more sessions to start and queue is empty.
        # Check if we're done sending entirely.
        if not self._stop_checker.can_send_any_turn():
            return (True, next_new_session_turn)

        # Can still send turns but queue is empty and can't start new
        # sessions (session limit reached). Skip this interval and wait for
        # continuation turns to arrive from callbacks.
        # Always yield to event loop to allow callbacks to run.
        # This is especially critical to prevent deadlock in CONCURRENCY_BURST mode (0 interval).
        await yield_to_event_loop()
        return (False, next_new_session_turn)

    async def _try_start_new_session(
        self, next_new_session_turn: TurnToSend
    ) -> tuple[bool, TurnToSend]:
        """Attempt to issue a credit for a new session.

        try_issue_credit returns None if no slot (skip interval), False if
        stop condition reached (exit loop), True if issued successfully.
        """
        result = await self._credit_issuer.try_issue_credit(next_new_session_turn)
        match result:
            case True:  # Successfully issued credit
                # Re-sample the next new turn for the next interval.
                return (
                    False,
                    self._conversation_source.next().build_first_turn(),
                )
            case False:  # Stop condition reached
                self.debug("Exiting: stop condition reached after try_issue_credit")
                return (True, next_new_session_turn)
            case _:  # None: no slot available, retry later
                # Always yield to event loop to allow callbacks to run.
                # This is especially critical to prevent deadlock in CONCURRENCY_BURST mode (0 interval).
                await yield_to_event_loop()
                return (False, next_new_session_turn)

    async def handle_credit_return(self, credit: Credit) -> None:
        """Queue the next turn of this conversation for the main loop.

        Called by CreditCallbackHandler when a worker completes a turn.
        If not the final turn, queues the next turn for the main rate loop
        to issue at the next available interval.

        The delay_ms from turn metadata (if present) is honored before queuing,
        simulating user "think time" between turns in a conversation.

        DAG children (``agent_depth > 0``) bypass the rate-loop continuation
        queue: they already hold their parent's session slot, so their
        continuations go through ``credit_issuer.dispatch_child_turn`` directly
        (and notify the BranchOrchestrator when the cap blocks them).
        """
        if credit.is_final_turn:
            return

        meta = self._conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit, meta)

        if credit.agent_depth > 0:
            # DAG child: route continuation directly through the issuer's DAG
            # path (bypasses the rate-loop continuation queue). Honor think-time
            # delay so a delayed child turn waits before dispatching.
            if meta.delay_ms is not None:
                # Run the delay inside a tracked *running* task rather than a
                # pending rate-loop timer. A pending timer is close()-dropped by
                # ``LoopScheduler.cancel_all_pending()`` when the phase reaches
                # sending-complete, which would silently lose this child turn:
                # neither the dispatch nor the on_child_stopped release would
                # run, so the parent's SPAWN_JOIN gate would never drain. A
                # running task survives that cull and instead fires (or releases
                # the child on cancellation) during the return-wait window.
                self._scheduler.execute_async(
                    self._delayed_child_continuation(
                        meta.delay_ms / MILLIS_PER_SECOND, turn, credit
                    )
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
        self, turn: TurnToSend, credit: Credit
    ) -> None:
        """Dispatch a DAG-child continuation turn, releasing the child on cap refusal.

        Children share their parent's session slot, so they don't go through
        the rate-loop continuation queue. They use the issuer's
        ``dispatch_child_turn`` path which respects the prefill cap and the
        DAG-child stop gate.

        Returns:
            None. When the dispatch is refused (cap reached or stop), the
            child's x_correlation_id is forwarded to
            ``branch_orchestrator.on_child_stopped`` so the orchestrator can
            release any join gates waiting on this child. If no orchestrator
            is wired (defensive: shouldn't happen in DAG runs), the refusal
            is logged and swallowed.
        """
        dispatched = await self._credit_issuer.dispatch_child_turn(turn)
        if dispatched:
            return
        if self._branch_orchestrator is None:
            self.debug(
                f"DAG child continuation refused by issuer for "
                f"x_correlation_id={credit.x_correlation_id} but no "
                f"branch_orchestrator is wired (non-DAG path)"
            )
            return
        try:
            await self._branch_orchestrator.on_child_stopped(credit.x_correlation_id)
        except Exception as exc:  # noqa: BLE001 - orchestrator error boundary; never propagate to the rate loop
            self.error(
                f"BranchOrchestrator.on_child_stopped raised for "
                f"x_correlation_id={credit.x_correlation_id}: {exc!r}"
            )

    async def _delayed_child_continuation(
        self, delay_sec: float, turn: TurnToSend, credit: Credit
    ) -> None:
        """Wait out a DAG child's think-time delay, then dispatch-or-release it.

        Runs as a tracked *running* task (via ``scheduler.execute_async``) so it
        is not close()-dropped by ``cancel_all_pending`` at sending-complete. If
        the phase force-cancels this task before the delay elapses, the child is
        still released through ``branch_orchestrator.on_child_stopped`` so the
        parent's join gate drains instead of hanging.
        """
        try:
            await asyncio.sleep(delay_sec)
        except asyncio.CancelledError:
            await self._release_child_on_cancel(credit)
            raise
        await self._issue_child_continuation_or_release(turn, credit)

    async def _release_child_on_cancel(self, credit: Credit) -> None:
        """Release a DAG child whose delayed continuation was cancelled.

        Mirrors the cap-refusal release in ``_issue_child_continuation_or_release``
        so a cancelled delay still drains the parent's join gate. Swallows
        orchestrator errors; never re-raises (the caller re-raises CancelledError).
        """
        if self._branch_orchestrator is None:
            return
        try:
            await self._branch_orchestrator.on_child_stopped(credit.x_correlation_id)
        except Exception as exc:  # noqa: BLE001 - orchestrator error boundary; never mask the cancellation
            self.error(
                f"BranchOrchestrator.on_child_stopped raised during cancel for "
                f"x_correlation_id={credit.x_correlation_id}: {exc!r}"
            )

    async def handle_session_ended(self, credit: Credit) -> None:
        """No strategy-local cleanup is needed when a session ends."""

    def set_request_rate(self, new_rate: float) -> None:
        """Update the request rate dynamically.

        Args:
            new_rate: New request rate (requests per second, must be > 0).
        """
        if new_rate <= 0:
            raise ValueError(f"Rate must be > 0, got {new_rate}")
        self._rate_generator.set_rate(new_rate)
