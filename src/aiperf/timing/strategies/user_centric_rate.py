# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-centric rate timing strategy for KV cache benchmarking.

Simulates a realistic multi-turn chat scenario where at t=0 there is already a steady-state
of users at varying stages of their session. Over time, new users join and old users leave.

Maintains consistent timing between turns for each user (`turn_gap`), simulating real multi-turn chat.
This timing directly affects KV cache hit rates: gaps that are too short keep caches
artificially warm, while gaps that are too long allow cache entries to be evicted before reuse.

Virtual History & Start Order
-----------------------------
Simulates steady-state from t=0 by distributing users across the "session lifetime"
(the time from a user's first turn to their last, measured in gaps = session_turns - 1).

Each user is assigned a virtual "age" representing how far through their session they are:
- User 1 (oldest): virtually done - all turns completed before t=0, replaced immediately
- User N (youngest): just started - most turns remaining

The user who just finished (User 1) is replaced by a fresh user who fires first at t=0.
Other users fire in staggered order based on their position in the session lifetime.
This creates immediate user churn rather than waiting for the first natural completions.

Example: 15 users, 20 turns, 1.0 QPS
-------------------------------------------
 User | Turns | Time | Turn Visualization
-------------------------------------------
    1 |     - |    - | (All turns completed before t=0) ← User 1 is "virtually done"
   16 |    20 |   0s | ████████████████████ ← New user at t=0 with all turns remaining
    5 |     6 |   1s | ██████
    9 |    11 |   2s | ███████████
   13 |    16 |   3s | ████████████████
    2 |     2 |   4s | ██
    6 |     7 |   5s | ███████
   10 |    12 |   6s | ████████████
   14 |    17 |   7s | █████████████████
    3 |     3 |   8s | ███
    7 |     8 |   9s | ████████
   11 |    13 |  10s | █████████████
   15 |    18 |  11s | ██████████████████
    4 |     4 |  12s | ████
    8 |     9 |  13s | █████████
   12 |    14 |  14s | ██████████████

New User Spawn Timing
------------
The first new user is spawned at t=0, in order to replace user 1, who already
finished all their turns before t=0.

After that, new users are spawned throughout the benchmark, specified by the following formula:
`next_spawn_time = prev_spawn_time + (max_turns * turn_gap)`

This ensures that a new user is spawned at the correct time to maintain the turn_gap.
Note that this is an absolute schedule and will not be affected by response times.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
import time
from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource, SampledSession
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker


def _find_alternate_spacing_step(n: int) -> int:
    """Find a step that produces unique positions when iterating 0 to n-1.

    Returns the smallest integer > 1 that is coprime with n, or 1 if n <= 2
    (since no valid coprime step exists for n <= 2).
    """
    for step in range(2, n):
        if gcd(n, step) == 1:
            return step
    return 1


@dataclass(slots=True)
class User:
    """Per-user state for enforcing turn_gap timing.

    Each user needs independent timing - their next turn is scheduled based on
    THEIR last send time, not a global clock.

    Attributes:
        user_id: Unique identifier for this user.
        sampled: The conversation session (prompts/responses) for this user.
        next_send_time: When this user should send their next request (perf_counter).
        max_turns: How many turns this user can send. Virtual history users have
            reduced max_turns (they've "already completed" some before t=0).
        order: Position in the initial stagger sequence (0 = fires first).
    """

    user_id: int
    sampled: SampledSession
    next_send_time: float = 0.0
    max_turns: int = 0
    order: int = 0

    @property
    def x_correlation_id(self) -> str:
        return self.sampled.x_correlation_id

    def build_first_turn(self) -> TurnToSend:
        return self.sampled.build_first_turn(max_turns=self.max_turns)


class UserCentricStrategy(AIPerfLoggerMixin):
    """User-centric timing strategy for KV cache benchmarking with realistic multi-user patterns."""

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
        """Initialize user-centric timing strategy with all dependencies."""
        super().__init__(logger_name="UserCentricTiming", **kwargs)
        self._config = config
        self._conversation_source = conversation_source
        self._scheduler = scheduler
        self._stop_checker = stop_checker
        self._credit_issuer = credit_issuer
        self._lifecycle = lifecycle

        self._num_users = self._config.num_users
        self._request_rate = self._config.request_rate

        if self._num_users is None or self._num_users <= 0:
            raise ValueError(
                "num_users must be set and non-zero for user-centric rate mode"
            )
        if self._request_rate is None or self._request_rate <= 0:
            raise ValueError(
                "request_rate must be set and positive for user-centric rate mode"
            )

        # Stagger is the smallest gap between any 2 users' first turns.
        # Request rate is requests/second, whereas stagger is seconds/request.
        self._stagger = 1 / self._request_rate

        # Computed in setup_phase
        self._turn_gap: float = 0.0
        self._target_num_users = self._num_users
        self._adaptive_target_enabled = False
        self._session_to_user: dict[str, User] = {}
        self._initial_users: list[User] = []
        self._spawn_queue: list[tuple[float, int, User | None]] | None = None
        self._spawn_seq = itertools.count()
        self._arrivals_stopped = False
        self._next_user_id: int = 1
        self._retired_user_cancellations = 0

    def _generate_next_user(
        self,
        target_perf_sec: float | None = None,
        max_turns: int | None = None,
        order: int | None = None,
    ) -> User:
        """Generate next user and add to session_to_user mapping.

        Creates user with sequential user_id, samples conversation (x_correlation_id
        set to user_id string), and configures timing/turn limits.
        """
        user_id = self._next_user_id
        self._next_user_id += 1
        sampled = self._conversation_source.next(x_correlation_id=str(user_id))

        user = User(
            user_id=user_id,
            sampled=sampled,
            next_send_time=target_perf_sec or 0.0,
            max_turns=max_turns or len(sampled.metadata.turns),
            order=order or 0,
        )
        self._session_to_user[user.x_correlation_id] = user
        return user

    async def setup_phase(self) -> None:
        """Pre-generate num_users initial users with timing and virtual history.

        Instead of all users starting fresh at t=0, we simulate steady-state by pretending
        each user has been active for some time. This creates immediate user churn
        (some users finishing soon, others just started) rather than waiting for the
        first completions. This is critical for KV cache benchmarking where we want
        realistic cache pressure from the first second.
        """
        num_users = self._num_users
        # We allow varying turn counts per conversation, so we use the average across the whole dataset.
        session_turns = round(
            self._conversation_source.dataset_metadata.average_turn_count
        )
        self._recompute_turn_gap(num_users)

        # Session lifetime = time from first to last turn, measured in gaps between turns.
        # Floor at 1 ensures spacing even for single-turn sessions.
        session_lifetime = max(1, session_turns - 1)

        # When num_users and session_lifetime share a common factor, the virtual history
        # formula produces duplicate positions. Use alternate spacing to ensure
        # each user gets a unique position.
        use_alternate_spacing = gcd(num_users, session_lifetime) > 1
        if use_alternate_spacing:
            spacing_step = _find_alternate_spacing_step(num_users)
            self.debug(
                f"Using alternate spacing: gcd({num_users}, {session_lifetime}) > 1, "
                f"step={spacing_step}"
            )

        for i in range(num_users):
            # Users with high virtual age have already finished most of their turns before t=0.
            # This creates the steady-state mix:
            # some users almost done, some mid-session, some just started.
            virtual_age = (num_users - i) * session_lifetime
            # Spread the distribution of turns across the users evenly based on their virtual_age.
            session_age = virtual_age // num_users
            turns_to_send = session_lifetime - session_age

            if turns_to_send <= 0:
                # User has virtually completed all their turns before t=0.
                # Still increment the next user id to ensure user ids are assigned in order.
                self._next_user_id += 1
                continue

            # Assign each user their starting order (0 = fires first, N-1 = fires last)
            # This spreads out users with similar turns_to_send such that users naturally
            # start and finish at varying times throughout the benchmark.
            if use_alternate_spacing:
                slot_index = (i * spacing_step) % num_users
            else:
                slot_index = virtual_age % num_users
            starting_order = num_users - slot_index

            # Generate the user regardless of whether they have turns to send to ensure
            # user ids are assigned in order.
            user = self._generate_next_user(
                max_turns=turns_to_send, order=starting_order
            )
            self._initial_users.append(user)

        # Always spawn a new user at t=0 with all turns remaining to replace the
        # first user that is "virtually done" (all turns completed).
        self._initial_users.append(self._generate_next_user(order=0))

    def _recompute_turn_gap(self, num_users: int) -> None:
        # num_users firing once per turn_gap gives: qps = num_users / turn_gap.
        self._turn_gap = num_users / self._request_rate

    @property
    def target_users(self) -> int:
        return self._target_num_users

    def set_target_users(self, value: int) -> None:
        if value <= 0:
            raise ValueError("target users must be positive")
        old_target = self._target_num_users
        self._adaptive_target_enabled = True
        self._target_num_users = value
        self._num_users = value
        self._recompute_turn_gap(value)
        if self._spawn_queue is None or value <= old_target:
            return
        now = time.perf_counter()
        for slot in range(value - old_target):
            self._push_arrival(self._spawn_queue, now + (slot * self._stagger), None)

    def user_control_snapshot(self) -> dict[str, int]:
        active = len(self._session_to_user)
        retiring = max(0, active - self._target_num_users)
        return {
            "target_value": self._target_num_users,
            "actual_value": active,
            "active_users": active,
            "retiring_users": retiring,
            "cancelled": self._retired_user_cancellations,
        }

    def _active_user_count(self) -> int:
        return len(self._session_to_user)

    def _should_spawn_user(self) -> bool:
        if not self._adaptive_target_enabled:
            return True
        return self._active_user_count() < self._target_num_users

    def _push_arrival(
        self,
        spawn_queue: list[tuple[float, int, User | None]],
        send_sec: float,
        user: User | None,
    ) -> None:
        """Push one arrival slot. ``user=None`` means generate the
        replacement user when the slot pops. ``seq`` breaks ties so heap
        entries never compare User objects."""
        heapq.heappush(spawn_queue, (send_sec, next(self._spawn_seq), user))

    def _defer_next_spawn(
        self, spawn_queue: list[tuple[float, int, User | None]]
    ) -> None:
        self._push_arrival(spawn_queue, time.perf_counter() + self._stagger, None)

    def _schedule_replacement_user(
        self,
        spawn_queue: list[tuple[float, int, User | None]],
        spawn_sec: float,
        user: User,
    ) -> None:
        if not self._should_spawn_replacement():
            return
        next_spawn_sec = spawn_sec + (user.max_turns * self._turn_gap)
        self._push_arrival(spawn_queue, next_spawn_sec, None)

    def _should_spawn_replacement(self) -> bool:
        if not self._adaptive_target_enabled:
            return True
        return self._active_user_count() <= self._target_num_users

    async def execute_phase(self) -> None:
        """Execute the user-centric rate phase.

        All first-turn arrivals (the pre-generated initial population and
        replacement users alike) flow through one clamped arrival queue; see
        the queue comment below. Subsequent spawn times are derived from
        stagger math: spawn + max_turns * turn_gap.
        This ensures all stagger slots remain active and maintains turn_gap spacing.

        Uses virtual history to simulate steady-state from t=0 with precise stagger spacing.
        """
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        self.info(
            f"User-centric mode: "
            f"qps={self._request_rate}, "
            f"{self._num_users} users, "
            f"session_turns={round(self._conversation_source.dataset_metadata.average_turn_count)}, "
            f"stagger={self._stagger:.3f}s, "
            f"turn_gap={self._turn_gap:.3f}s"
        )

        # Priority queue (heapq) of future arrival slots in seconds. It is
        # populated up front with one slot per initial user, and over the
        # benchmark duration each spawned user derives its replacement slot
        # from stagger math (spawn + max_turns * turn_gap). This maintains a
        # steady spawn rate and QPS. Note that this is still an "open-loop"
        # strategy: the replacement slot comes from the planned time
        # regardless of whether the previous user completed on time. The only
        # exception is if `--concurrency` is set (slot acquisition blocks).
        #
        # Every slot fires at max(planned, previous_arrival_fire + stagger):
        # stagger is the documented SMALLEST gap between any two users' first
        # turns, so the clamp applies to on-plan and overdue slots alike -- a
        # late fire must not let the next on-plan slot close to a sub-stagger
        # gap. Without the clamp, an event-loop stall longer than one planned
        # gap makes a backlog of overdue slots pop back-to-back and burst
        # new-user arrivals -- exactly what the stagger exists to prevent, and
        # the mechanism behind the collapsed-stagger CI flake (initial
        # pre-armed timers bursting, and replacements popping overdue while
        # the initial grid was still incomplete). The absolute grid is kept
        # exactly when the loop keeps up; under persistent lateness the grid
        # slides (spacing is preserved, punctuality is not recoverable).
        spawn_queue: list[tuple[float, int, User | None]] = []
        self._spawn_queue = spawn_queue
        self._arrivals_stopped = False

        base_perf_sec = self._lifecycle.started_at_perf_sec
        for user in sorted(self._initial_users, key=lambda u: u.order):
            # Send time is based on starting order (0 = first, N-1 = last)
            user.next_send_time = base_perf_sec + (user.order * self._stagger)
            self._push_arrival(spawn_queue, user.next_send_time, user)

        # Anchored at slot fire time, before issuance: slot backpressure and
        # issuance latency never shift subsequent slot targets. Issuance
        # itself is fire-and-track (one task per arrival via the scheduler,
        # which captures per-issue exceptions and cancels them at teardown),
        # so one blocked or excepting issuance neither stalls nor aborts the
        # remaining grid; measured capture times still reflect issuance
        # backpressure when slots block, which is the intended concurrency
        # behavior.
        last_slot_perf = base_perf_sec - self._stagger
        while True:
            if self._arrivals_stopped:
                return
            if not spawn_queue:
                await asyncio.sleep(0.1)
                continue

            spawn_sec, _, queued_user = heapq.heappop(spawn_queue)
            target_sec = max(spawn_sec, last_slot_perf + self._stagger)
            await asyncio.sleep(max(0.0, target_sec - time.perf_counter()))
            # Issuance is fire-and-track, so a refusal lands one loop
            # iteration late; re-check after the sleep so a stopped phase
            # fires at most the slots already in flight.
            if self._arrivals_stopped:
                return
            last_slot_perf = time.perf_counter()

            if queued_user is None:
                if not self._should_spawn_user():
                    self._defer_next_spawn(spawn_queue)
                    continue
                user = self._generate_next_user(spawn_sec)
            else:
                user = queued_user

            self._scheduler.execute_async(self._issue_arrival(user, spawn_sec))

    async def _issue_arrival(self, user: User, spawn_sec: float) -> None:
        """Issue one arrival's first turn and derive its replacement slot.

        Runs as a scheduler-tracked task (fire-and-track from execute_phase).
        A refusal (stop condition reached / sending stopped) flips
        ``_arrivals_stopped`` so the pop loop exits instead of spinning on a
        queue that no longer grows; exceptions propagate to the scheduler's
        exception handler without disturbing the other arrivals.
        """
        should_continue = await self._credit_issuer.issue_credit(
            user.build_first_turn()
        )
        if not should_continue:
            self._arrivals_stopped = True
            return

        # Derive next spawn time based on estimated time to completion.
        # Adaptive scale-down drains excess users by suppressing replacements.
        assert self._spawn_queue is not None
        self._schedule_replacement_user(self._spawn_queue, spawn_sec, user)

    async def handle_credit_return(
        self,
        credit: Credit,
        *,
        error: str | None = None,
    ) -> None:
        """Handle credit return: dispatch next turn.

        ``error`` is accepted for protocol parity and ignored here.

        Schedules next turn at `max(now, user.next_send_time + turn_gap)`.
        This maintains ideal pacing when responses arrive on time, but if the
        response is late, the max() re-aligns to current time (sends immediately).
        """
        if credit.is_final_turn:
            # User finished all their turns. New users continue spawning in execute_phase.
            self._session_to_user.pop(credit.x_correlation_id, None)
            # user_ids are never reused, so the retained cache-bust marker for
            # this correlation id can never be looked up again.
            self._conversation_source.release_marker_for_correlation_id(
                credit.x_correlation_id
            )
            return

        current_sec = time.perf_counter()
        user = self._session_to_user.get(credit.x_correlation_id)
        if user is None:
            raise ValueError(
                f"User not found for x_correlation_id: {credit.x_correlation_id}"
            )
        # Pass next-turn metadata so has_forks rides onto the continuation turn
        # (the sticky router defers parent-entry eviction until DAG children
        # drain); dropping it premature-evicts a fork-bearing parent's turns.
        meta = self._conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit, meta)

        # If the next turn time already passed, the max() will
        # re-align their schedule to account for the delay.
        user.next_send_time = max(current_sec, user.next_send_time + self._turn_gap)
        self._scheduler.schedule_at_perf_sec(
            user.next_send_time,
            self._credit_issuer.issue_credit(turn),
        )
