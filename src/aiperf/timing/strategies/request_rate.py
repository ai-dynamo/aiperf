# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rate-based timing strategy for credit issuance."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

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

    async def handle_credit_return(self, credit: Credit) -> None:
        """Queue the next turn of this conversation for the main loop.

        Called by CreditCallbackHandler when a worker completes a turn.
        If not the final turn, queues the next turn for the main rate loop
        to issue at the next available interval.

        The delay_ms from turn metadata (if present) is honored before queuing,
        simulating user "think time" between turns in a conversation.
        """
        if credit.is_final_turn:
            return

        meta = self._conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit)

        # Honor think-time delay from dataset metadata before queuing
        if meta.delay_ms is not None:
            self._scheduler.schedule_later(
                meta.delay_ms / MILLIS_PER_SECOND,
                self._continuation_turns.put(turn),
            )
        else:
            self._continuation_turns.put_nowait(turn)

    def set_request_rate(self, new_rate: float) -> None:
        """Update the request rate dynamically.

        Args:
            new_rate: New request rate (requests per second, must be > 0).
        """
        if new_rate <= 0:
            raise ValueError(f"Rate must be > 0, got {new_rate}")
        self._rate_generator.set_rate(new_rate)
