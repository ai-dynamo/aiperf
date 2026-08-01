# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-centric rate timing strategy for KV cache benchmarking.

Simulates a realistic multi-turn chat scenario where at t=0 there is already a steady-state
of users at varying stages of their session. Over time, new users join and old users leave.

Maintains consistent timing between turns for each user (`turn_gap`), simulating real multi-turn chat.
This timing directly affects KV cache hit rates: gaps that are too short keep caches
artificially warm, while gaps that are too long allow cache entries to be evicted before reuse.

By default the turn gap is the deterministic constant `num_users / request_rate` seconds.
With `--user-centric-gap-distribution lognormal|weibull`, each turn gap is instead drawn
per turn from the named distribution whose mean is pinned to `num_users / request_rate`
and whose median is user-supplied to control skew. Each user draws from an independent,
deterministically seeded stream, so gap sequences are reproducible under `--random-seed`.
New-user spawn scheduling keeps using the mean gap.

Sampled Gaps Do Not Preserve The Realized Request Rate
------------------------------------------------------
Pinning the mean pins the *distribution* mean, not the realized aggregate request rate.
The next send time is `max(now, next_send_time + gap)` (see `handle_credit_return`), so
the clamp only ever pushes a send later: a gap shorter than the in-flight service time is
truncated away, while a longer-than-mean gap is applied in full. The realized inter-send
interval is `max(service_time, gap)`, whose expectation is at least the pinned mean and
strictly above it whenever a response can outlast its gap - which is why the shortfall
exists with fixed gaps too. Because `max(service_time, ...)` is convex, swapping a fixed
gap for a random one of the same mean raises that expectation further, and heavier skew
raises it more: more of the distribution's mass falls below the service time and is
clamped away. Do not read `--request-rate` as a guaranteed realized rate in this mode -
measure the achieved rate from the run's own output.

Heavy Skew Produces Near-Zero Gaps
----------------------------------
Neither params model is constructed with `min`/`max` rejection bounds here, so nothing
floors a sampled gap. For `weibull`, the shape solved from mean/median drops below 1 once
`mean / median > 1 / ln(2) ~= 1.443`; below shape 1 the density is unbounded at zero and a
non-trivial share of draws are effectively back-to-back turns. Measured with this repo's
own solver and sampler (`aiperf.common.distributions`), 2,000,000 draws per row:

  mean / median      | mean/median | weibull shape | draws under 1 ms
  2.000 s / 0.500 s  | 4.000       | 0.5079        | 2.93%
  1.667 s / 1.200 s  | 1.389       | 1.0514        | 0.041%
  1.667 s / 1.300 s  | 1.282       | 1.1907        | 0.014%

Lognormal is far less exposed: 0.0089% of draws under 1 ms at the same 2.0 s / 0.5 s
setting. Keep `mean / median` below about 1.44 if near-zero gaps would distort the
workload under test.

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
import time
from dataclasses import dataclass
from math import gcd
from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.distributions import (
    LognormalParams,
    WeibullParams,
    sample_lognormal,
    sample_weibull,
)
from aiperf.common.enums import UserCentricGapDistribution
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.credit.structs import Credit, TurnToSend

if TYPE_CHECKING:
    from numpy.random import Generator

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
        gap_rng: Per-user NumPy generator for sampled turn gaps, seeded
            deterministically from user_id. None in fixed gap mode.
    """

    user_id: int
    sampled: SampledSession
    next_send_time: float = 0.0
    max_turns: int = 0
    order: int = 0
    gap_rng: Generator | None = None

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

        self._gap_distribution = self._config.user_centric_gap_distribution
        self._gap_median = self._config.user_centric_gap_median
        if (
            self._gap_distribution != UserCentricGapDistribution.FIXED
            and self._gap_median is None
        ):
            raise ValueError(
                "user_centric_gap_median must be set when "
                "user_centric_gap_distribution is lognormal or weibull"
            )

        # Stagger is the smallest gap between any 2 users' first turns.
        # Request rate is requests/second, whereas stagger is seconds/request.
        self._stagger = 1 / self._request_rate

        # Computed in setup_phase
        self._turn_gap: float = 0.0
        self._gap_params: LognormalParams | WeibullParams | None = None
        self._target_num_users = self._num_users
        self._adaptive_target_enabled = False
        self._session_to_user: dict[str, User] = {}
        self._initial_users: list[User] = []
        self._spawn_queue: list[float] | None = None
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
            gap_rng=self._derive_gap_rng(user_id),
        )
        self._session_to_user[user.x_correlation_id] = user
        return user

    def _derive_gap_rng(self, user_id: int) -> Generator | None:
        """Derive a per-user gap RNG so each user's gap stream is reproducible
        under --random-seed regardless of how users' credit returns interleave."""
        if self._gap_distribution == UserCentricGapDistribution.FIXED:
            return None
        return rng.derive(
            f"timing.user_centric_rate.turn_gap.user.{user_id}"
        ).numpy_generator

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
        turn_gap = num_users / self._request_rate
        # Solve first: a rejected gap must leave the previous turn gap and its
        # params still describing the same distribution.
        self._gap_params = self._solve_gap_params(turn_gap)
        self._turn_gap = turn_gap

    def _solve_gap_params(
        self, turn_gap: float
    ) -> LognormalParams | WeibullParams | None:
        """Solve the sampled gap distribution with its mean pinned to *turn_gap*.

        Takes the gap as an argument rather than reading `self._turn_gap` so it can
        never observe half-updated state.

        Pinning the mean pins the distribution mean only; the realized aggregate
        request rate is lower and degrades with skew (see the module docstring).

        Neither params model is given `min`/`max` bounds here, so heavy skew is
        unfloored. For weibull, a mean/median ratio above `1 / ln(2) ~= 1.443` drives
        the solved shape below 1 and yields near-zero gaps: at mean 2.0 s /
        median 0.5 s the shape is 0.5079 and 2.93% of draws land under 1 ms
        (2,000,000-draw sample); at mean 1.667 s / median 1.2 s the shape is 1.0514
        and 0.041% land under 1 ms.
        """
        if self._gap_distribution == UserCentricGapDistribution.FIXED:
            return None
        if self._gap_median is None or self._gap_median >= turn_gap:
            raise ValueError(
                f"user_centric_gap_median ({self._gap_median}) must be strictly "
                "less than the mean turn gap of num_users / request_rate = "
                f"{turn_gap} seconds; lognormal and weibull turn-gap "
                "sampling supports right-skewed gaps only (median < mean)"
            )
        if self._gap_distribution == UserCentricGapDistribution.WEIBULL:
            return WeibullParams(
                distribution="weibull", mean=turn_gap, median=self._gap_median
            )
        return LognormalParams(mean=turn_gap, median=self._gap_median)

    def _next_turn_gap(self, user: User) -> float:
        """Return the gap in seconds to apply before this user's next turn."""
        if self._gap_params is None or user.gap_rng is None:
            return self._turn_gap
        if isinstance(self._gap_params, WeibullParams):
            return float(sample_weibull(self._gap_params, user.gap_rng, size=1)[0])
        return float(sample_lognormal(self._gap_params, user.gap_rng, size=1)[0])

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
            heapq.heappush(self._spawn_queue, now + (slot * self._stagger))

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

    def _defer_next_spawn(self, spawn_queue: list[float]) -> None:
        heapq.heappush(spawn_queue, time.perf_counter() + self._stagger)

    def _schedule_replacement_user(
        self, spawn_queue: list[float], spawn_sec: float, user: User
    ) -> None:
        if not self._should_spawn_replacement():
            return
        next_spawn_sec = spawn_sec + (user.max_turns * self._turn_gap)
        heapq.heappush(spawn_queue, next_spawn_sec)

    def _should_spawn_replacement(self) -> bool:
        if not self._adaptive_target_enabled:
            return True
        return self._active_user_count() <= self._target_num_users

    async def execute_phase(self) -> None:
        """Execute the user-centric rate phase.

        Pre-generated users are scheduled asynchronously (fire-and-forget).
        Subsequent spawn times are derived from stagger math: spawn + max_turns * turn_gap.
        This ensures all stagger slots remain active and maintains turn_gap spacing.

        Uses virtual history to simulate steady-state from t=0 with precise stagger spacing.
        """
        if self._lifecycle.started_at_perf_ns is None:
            raise RuntimeError("started_at_perf_ns is not set in the lifecycle")

        gap_note = (
            f", gap_distribution={self._gap_distribution}, "
            f"gap_median={self._gap_median}s"
            if self._gap_distribution != UserCentricGapDistribution.FIXED
            else ""
        )
        self.info(
            f"User-centric mode: "
            f"qps={self._request_rate}, "
            f"{self._num_users} users, "
            f"session_turns={round(self._conversation_source.dataset_metadata.average_turn_count)}, "
            f"stagger={self._stagger:.3f}s, "
            f"turn_gap={self._turn_gap:.3f}s{gap_note}"
        )

        # Priority queue (heapq) of future spawn times in seconds (derived from stagger math)
        # This will be initially populated by 1 spawn user per initial user.
        # Then, over the benchmark duration, as a new user popped off the queue and spawned,
        # a new user will be added to the queue based on target completion time of spawned user.
        #
        # This maintains a steady spawn rate.
        # Note that this is still an "open-loop" strategy because the replacement
        # spawn user will spawn at the specified time regardless of whether the previous
        # spawn user completed on time. The only exception is if `--concurrency` is set.
        spawn_queue: list[float] = []
        self._spawn_queue = spawn_queue

        # Schedule initial users and derive the initial spawn times.
        # This is what creates the initial "steady-state" of the benchmark.
        for user in self._initial_users:
            # Send time is based on starting order (0 = first, N-1 = last)
            user.next_send_time = self._lifecycle.started_at_perf_sec + (
                user.order * self._stagger
            )
            self._scheduler.schedule_at_perf_sec(
                user.next_send_time,
                self._credit_issuer.issue_credit(user.build_first_turn()),
            )
            # Derive next spawn time based on estimated time to completion.
            # Always uses the mean turn gap, keeping the spawn cadence an absolute
            # schedule that per-turn draws do not perturb.
            next_spawn_sec = user.next_send_time + (user.max_turns * self._turn_gap)
            heapq.heappush(spawn_queue, next_spawn_sec)

        # Continuously spawn new users at discrete intervals to maintain the target QPS.
        while True:
            if not spawn_queue:
                await asyncio.sleep(0.1)
                continue

            spawn_sec = heapq.heappop(spawn_queue)
            await asyncio.sleep(max(0.0, spawn_sec - time.perf_counter()))

            if not self._should_spawn_user():
                self._defer_next_spawn(spawn_queue)
                continue

            user = self._generate_next_user(spawn_sec)
            turn = user.build_first_turn()
            should_continue = await self._credit_issuer.issue_credit(turn)
            if not should_continue:
                return

            # Derive next spawn time based on estimated time to completion.
            # Adaptive scale-down drains excess users by suppressing replacements.
            self._schedule_replacement_user(spawn_queue, spawn_sec, user)

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
        user.next_send_time = max(
            current_sec, user.next_send_time + self._next_turn_gap(user)
        )
        self._scheduler.schedule_at_perf_sec(
            user.next_send_time,
            self._credit_issuer.issue_credit(turn),
        )
