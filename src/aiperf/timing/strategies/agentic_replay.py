# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgenticReplayStrategy - trajectory-driven trace replay timing strategy.

Phase-aware timing strategy for the ``agentic_replay`` timing mode (spec §4.2).

Each trajectory is a wall-clock snapshot of a trace at a sampled instant t*
(25-75% through the trace's recorded duration). Every stream (root + each
subagent chain) splits at t*: turns before t* are history, turns at/after t*
are profiled.

WARMUP: for every session active (mid-flight) at t*, replay its last request
before t* (turn ``next_turn_index - 1``) as a full-prefix request, priming
the server cache to the stream's state at t*. This includes parents gated on
a child join (they sent turn n-1 before t* and resume at the join turn during
PROFILING, so n-1 primes that turn). Streams whose first request is at/after
t* (``next_turn_index == 0``) have nothing to warm. The phase exits via the
standard ``SendingCompleteStopCondition`` plus ``grace_period_sec=inf``
semantics already in CreditPhaseConfig (the warmup barrier).

Warmup-failure accumulation: terminal failures (``credit_return.error`` or
``credit_return.cancelled``) on a WARMUP credit's final turn are routed by
``CreditCallbackHandler`` into ``record_warmup_failure(trace_id)``. At
WARMUP teardown, ``PhaseRunner`` calls ``report_warmup_failures()`` which
raises ``TrajectoryWarmupFailedError`` if any failures were recorded. This
aborts PROFILING so steady-state metrics aren't silently biased by a
degraded trajectory pool.

PROFILING: each stream resumes at its first turn at/after t*
(``next_turn_index``). Dispatch times are normalized so the trajectory's
earliest post-t* request fires at profiling-time 0 and all other requests
preserve their recorded relative offsets; subsequent turns honor trace
inter-turn ``delay_ms`` (already clamped upstream in the loader). Gated
parents fire their join turn when blocking children complete. When a root
session reaches its final turn, its trace_id is recycled FIFO-style and a
fresh session (starting at turn 0) is spawned from the next trace_id in the
queue.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import Counter
from typing import TYPE_CHECKING

from msgspec.structs import replace as _struct_replace

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.common.scenario.context_overflow import is_context_overflow_response
from aiperf.credit.structs import TurnToSend
from aiperf.timing.conversation_source import SampledSession
from aiperf.timing.strategies.cache_bust import build_cache_bust_marker
from aiperf.timing.trajectory_source import (
    Trajectory,
    TrajectorySnapshot,
    TrajectorySource,
)

if TYPE_CHECKING:
    from aiperf.common.config import UserConfig
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.structs import Credit
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.stop_conditions import StopConditionChecker


class AgenticReplayStrategy(AIPerfLoggerMixin):
    """Phase-aware trajectory-driven trace replay timing strategy.

    Constructed fresh per phase by ``PhaseRunner``. Trajectory state survives
    the WARMUP -> PROFILING boundary because ``TrajectorySource`` is
    constructed once at TimingManager level and shared across phases.
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
        user_config: UserConfig | None = None,
        branch_orchestrator=None,
        **kwargs,
    ) -> None:
        super().__init__(logger_name="AgenticReplayTiming")

        if config.phase not in (CreditPhase.WARMUP, CreditPhase.PROFILING):
            raise ValueError(
                "AgenticReplayStrategy requires phase WARMUP or PROFILING, "
                f"got {config.phase!r}"
            )
        if not isinstance(conversation_source, TrajectorySource):
            raise TypeError(
                "AgenticReplayStrategy requires TrajectorySource (got "
                f"{type(conversation_source).__name__}). Construct it once at "
                "TimingManager level and inject into both phase strategies."
            )

        self.config = config
        self.conversation_source: TrajectorySource = conversation_source
        self.scheduler = scheduler
        self.stop_checker = stop_checker
        self.credit_issuer = credit_issuer
        self.lifecycle = lifecycle
        self.branch_orchestrator = branch_orchestrator

        self._recycle_queue: asyncio.Queue[str] | None = None
        # Keyed on x_correlation_id (not trace_id): the guard's intent is to
        # catch the same final turn firing handle_credit_return twice — a
        # per-session property. trace_id-keying spuriously tripped when two
        # wrap-filled lanes finished the same trace_id with distinct
        # correlation_ids.
        self._in_flight_recycled: set[str] = set()
        # Trace_ids whose session is currently dispatched (any turn in flight
        # or scheduled). Used by ``_spawn_from_recycle_or_id`` to skip
        # popping a trace whose every lane is already alive — prevents over-
        # subscribing a trace_id, which would otherwise be possible when the
        # initial recycle queue spans the full pool (trajectories appear in
        # the queue while their sessions are still running at PROFILING start).
        # Multiset (Counter) rather than a set because wrap-fill can place
        # multiple lanes on the same trace_id: skip only when every lane for
        # this trace is busy. Collapses to set-style semantics when every
        # value in _lanes_per_trace is 1.
        self._active_traces: Counter[str] = Counter()
        # Lane multiplicity per trace_id, frozen at strategy init from the
        # trajectory list. _pop_next_eligible_trace skips only when every
        # lane for a trace is busy (count >= capacity).
        self._lanes_per_trace: Counter[str] = Counter(
            t.conversation_id for t in conversation_source.trajectories
        )
        self._failed_warmup_traces: list[str] = []

        # Cache-bust state. WARMUP and PROFILING construct distinct strategy
        # instances (PhaseRunner builds a fresh AgenticReplayStrategy per
        # phase), while the shared TrajectorySource keeps each sampled lane's
        # x_correlation_id stable across the phase boundary AND carries the
        # marker ledger across it. A session continuing into PROFILING reuses
        # the exact marker minted for it during WARMUP (see
        # ``_mint_marker_for_session``), so warmup turn k_i and profile turn
        # k_i+1 share the same marker within the continued session - the
        # KV-cache lineage warmup is meant to prime is preserved by identity,
        # not by replaying mint order. New sessions draw from the shared
        # ``recycle_pass`` counter, which never restarts, so a recycled
        # session's digest can never collide with a warmed one.
        ledger = conversation_source.cache_bust_ledger
        self._recycle_pass: dict[str, int] = ledger.recycle_pass
        self._session_marker: dict[str, str | None] = ledger.session_marker
        self._correlation_to_lane: dict[str, int] = {}
        self._cache_bust_target: CacheBustTarget = (
            user_config.input.prompt.cache_bust.target
            if user_config is not None
            else CacheBustTarget.NONE
        )
        self._benchmark_id: str = (
            user_config.benchmark_id if user_config is not None else "unknown"
        )
        # When True, PROFILING preserves each trajectory's t*->first-request
        # idle gap instead of anchoring the earliest request at profiling-time
        # 0. See _dispatch_snapshot_for_profiling and the env-var docstring.
        self._preserve_trajectory_start_gap: bool = (
            Environment.TIMING.PRESERVE_TRAJECTORY_START_GAP
        )

        # Wrap-fill + cache_bust=NONE produces byte-identical traffic across
        # shared-trace lanes. agentx-mvp auto-locks cache_bust=first_turn_prefix
        # so this never fires there; ad-hoc agentic-replay with cache_bust
        # explicitly off gets a loud heads-up.
        wrap_fill_active = any(count > 1 for count in self._lanes_per_trace.values())
        if wrap_fill_active and self._cache_bust_target == CacheBustTarget.NONE:
            self.warning(
                "Wrap-fill active (%d distinct trace_ids fanned across %d "
                "lanes) with cache_bust.target=NONE: per-lane traffic will "
                "be byte-identical. Set cache_bust.target=first_turn_prefix "
                "(or another non-NONE target) for distinct shared-trace "
                "replays.",
                len(self._lanes_per_trace),
                sum(self._lanes_per_trace.values()),
            )

    async def setup_phase(self) -> None:
        """Phase-specific async setup.

        WARMUP: nothing - trajectories already built by TrajectorySource at
        TimingManager construction time.

        PROFILING: build the FIFO recycle queue with the FULL set of loader
        trace_ids (including trajectory ids), and pre-register every live
        trajectory lane in ``_active_traces``. Trajectories run live at
        PROFILING start (resumed at k_i+1); pre-registering them here -
        rather than lane-by-lane during dispatch - means a lane that
        recycles immediately at startup can never pop a trace whose own
        lane simply hasn't dispatched yet (a duplicate concurrent session).
        """
        if self.config.phase == CreditPhase.PROFILING:
            if not self.conversation_source.trajectories:
                raise RuntimeError(
                    "AgenticReplayStrategy PROFILING setup: trajectories empty. "
                    "WARMUP must complete with at least one trajectory before "
                    "PROFILING can start. Check loader output and warmup failures."
                )
            self._active_traces.update(self._lanes_per_trace)
            self._recycle_queue = asyncio.Queue()
            # Recycle pool spans the FULL dataset, not (full - trajectories).
            # Trajectories run live at PROFILING start (resumed at k_i+1) and
            # are pushed to the queue tail when their session ends; including
            # them in the initial pool means recycled lanes draw from the
            # full diversity of dataset_metadata.conversations rather than
            # being capped at (pool_size - concurrency) distinct trace_ids.
            trajectory_ids = {
                trajectory.conversation_id
                for trajectory in self.conversation_source.trajectories
            }
            for conv in self.conversation_source.dataset_metadata.conversations:
                if getattr(conv, "is_root", True):
                    self._recycle_queue.put_nowait(conv.conversation_id)
            self.info(
                f"PROFILING setup: trajectories={len(trajectory_ids)} traces, "
                f"recycle_queue={self._recycle_queue.qsize()} traces (full pool)"
            )

    async def execute_phase(self) -> None:
        """Dispatch initial credits for the phase."""
        if self.config.phase == CreditPhase.WARMUP:
            await self._execute_warmup()
        else:
            await self._execute_profiling()

    async def _execute_warmup(self) -> None:
        """Dispatch one warmup credit for every ready trajectory state."""
        warmup_total_count = self.conversation_source.warmup_credit_count
        self.info(
            f"WARMUP execute: dispatching {warmup_total_count} trajectory credits"
        )
        for lane, trajectory in enumerate(self.conversation_source.trajectories):
            if trajectory.snapshot is None:
                session = self.conversation_source.session_for(trajectory)
                dispatch_index = trajectory.start_turn_index
                self._correlation_to_lane[session.x_correlation_id] = lane
                self._active_traces[trajectory.conversation_id] += 1
                self._mint_marker_for_session(
                    session.x_correlation_id, trajectory.conversation_id, lane
                )
                turn = self._build_turn_for_session(session, dispatch_index)
                await self.credit_issuer.issue_credit(turn)
                continue

            states = self._get_snapshot(trajectory).states
            for state in states:
                warm_index = state.warmup_turn_index
                if warm_index is None:
                    # First request is at/after t*: the server had not seen
                    # this stream at the snapshot instant, so there is nothing
                    # to warm. PROFILING dispatches its turn 0 at the
                    # normalized offset.
                    continue
                # Every session active (mid-flight) at t* is warmed, INCLUDING
                # a parent gated on a child join: it sent turn n-1 before t*
                # and is waiting to send the join turn n during PROFILING, so
                # warming n-1 primes that join turn's prefix (it would cold-
                # miss otherwise). Warmup is a separate phase, so this does not
                # disturb the profiling-side join machinery.
                session = self.conversation_source.session_for_state(state)
                self._correlation_to_lane[session.x_correlation_id] = lane
                self._mint_marker_for_session(
                    session.x_correlation_id, state.conversation_id, lane
                )
                turn = self._build_turn_for_session(session, warm_index)
                await self.credit_issuer.issue_credit(turn)
        # Trajectory dispatch complete; signal the phase that no more credits
        # will be issued. SendingCompleteStopCondition watches this flag and
        # fires once all in-flight credits return (the warmup barrier).
        # Normally redundant with the phase's count-based path: PhaseRunner
        # re-anchors ``total_expected_requests`` to the actual trajectory count
        # at __init__, so ``CreditCounter.is_final_credit`` flips on the last
        # dispatched credit and ``CreditIssuer`` already fires
        # ``all_credits_sent_event`` + freezes counts. Kept as a guarded fallback
        # for defense-in-depth; the ``is_sending_complete`` guard avoids the
        # double-transition ValueError when the count path won the race.
        if not self.lifecycle.is_sending_complete:
            self.lifecycle.mark_sending_complete()

    async def _execute_profiling(self) -> None:
        """Resume each trajectory at ``k_i + 1`` to seed the steady state.

        All trajectories are dispatched concurrently so the full concurrency
        target is reached as fast as slot limits allow, rather than
        serializing over N credit round-trips. Subsequent turns and
        recycle-pool sessions are dispatched from handle_credit_return.
        """
        self.info(
            f"PROFILING execute: resuming {len(self.conversation_source.trajectories)} "
            f"trajectory sessions"
        )
        # return_exceptions=True keeps ownership of every lane until it
        # settles: a bare gather would re-raise the first failure while the
        # sibling coroutines keep issuing credits into a failing phase,
        # unreachable by the phase runner's cancellation.
        results = await asyncio.gather(
            *(
                self._dispatch_one_profiling_trajectory(trajectory, lane)
                for lane, trajectory in enumerate(self.conversation_source.trajectories)
            ),
            return_exceptions=True,
        )
        first_error: BaseException | None = None
        for lane, result in enumerate(results):
            if not isinstance(result, BaseException):
                continue
            trace_id = self.conversation_source.trajectories[lane].conversation_id
            self.error(
                f"PROFILING dispatch failed for lane {lane} "
                f"(trace_id={trace_id!r}): {result!r}"
            )
            if first_error is None:
                first_error = result
        if first_error is not None:
            raise first_error

    async def _dispatch_one_profiling_trajectory(
        self, trajectory: Trajectory, lane: int
    ) -> None:
        """Dispatch one lane's initial PROFILING credit (run under gather)."""
        if trajectory.snapshot is not None:
            await self._dispatch_snapshot_for_profiling(trajectory, lane)
            return

        session = self.conversation_source.session_for(trajectory)
        self._correlation_to_lane[session.x_correlation_id] = lane
        # The lane's trace was pre-registered in _active_traces by setup_phase.
        self._mint_marker_for_session(
            session.x_correlation_id, trajectory.conversation_id, lane
        )
        resume_index = trajectory.start_turn_index + 1
        num_turns = len(session.metadata.turns)

        if resume_index >= num_turns:
            # Trajectory's k_i was already the last turn (rare: happens
            # only for very short traces). Skip directly to recycle.
            self.debug(
                lambda: (
                    f"Trajectory {trajectory.conversation_id} "
                    f"k_i={trajectory.start_turn_index} >= last turn "
                    f"(n={num_turns}); recycling immediately"
                )
            )
            await self._spawn_from_recycle_or_id(
                trajectory.conversation_id,
                finished_correlation_id=session.x_correlation_id,
            )
            return

        turn = self._build_turn_for_session(session, resume_index)
        await self.credit_issuer.issue_credit(turn)

    async def handle_credit_return(
        self, credit: Credit, *, error: str | None = None
    ) -> None:
        """Dispatch next turn or recycle on session completion.

        WARMUP returns are no-ops at the strategy level; phase termination is
        handled by ``SendingCompleteStopCondition`` + grace period. Terminal
        WARMUP failures are routed by ``CreditCallbackHandler`` directly into
        ``record_warmup_failure`` and surfaced at WARMUP teardown.

        PROFILING: if not the final turn, dispatch the next turn honoring
        trace ``delay_ms``. If the final turn just completed, recycle the
        trace_id and spawn a fresh session from the next queued trace_id.

        Context-overflow short-circuit: when a non-final turn returns with an
        error body matching the AgentX context-overflow allowlist, recycle the
        trajectory immediately instead of dispatching subsequent turns. Once a
        trajectory has blown past the model's context limit, every later turn's
        cumulative prompt will too — continuing to dispatch them just wastes
        compute and inflates the run's overflow rate. This mirrors the
        kv-cache-tester behavior of marking the user "truncated" on the first
        context-length error and removing them from the active pool.

        DAG-child final turns short-circuit: child terminal completion is
        owned by ``BranchOrchestrator`` (the callback handler invokes
        ``on_child_leaf_reached`` / ``on_child_errored`` before the strategy).
        The strategy must not push child conversation_ids into the recycle
        pool — they're not root pool entries, and they repeat across recycle
        passes of the parent, which would trip the double-recycle guard the
        second time the parent re-runs.
        """
        if self.config.phase == CreditPhase.WARMUP:
            return

        terminal_overflow = (
            not credit.is_final_turn
            and error is not None
            and is_context_overflow_response(body=error)
        )

        if credit.agent_depth > 0:
            if not credit.is_final_turn and not terminal_overflow:
                await self._dispatch_next_turn(credit)
                return
            if terminal_overflow and self.branch_orchestrator is not None:
                await self.branch_orchestrator.on_child_stopped(credit.x_correlation_id)
            self._session_marker.pop(credit.x_correlation_id, None)
            self._correlation_to_lane.pop(credit.x_correlation_id, None)
            return

        if not credit.is_final_turn and not terminal_overflow:
            await self._dispatch_next_turn(credit)
            return

        if terminal_overflow:
            self.info(
                lambda: (
                    f"Terminating trajectory {credit.conversation_id} early at "
                    f"turn {credit.turn_index}/{credit.num_turns - 1}: "
                    f"context-overflow error from server"
                )
            )

        await self._spawn_from_recycle_or_id(
            credit.conversation_id,
            finished_correlation_id=credit.x_correlation_id,
        )

    async def _dispatch_next_turn(self, credit: Credit) -> None:
        """Issue the next turn of an in-progress session, honoring delay_ms."""
        next_meta = self.conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit, next_meta)

        if next_meta.delay_ms is not None and next_meta.delay_ms > 0:
            self.scheduler.schedule_later(
                next_meta.delay_ms / MILLIS_PER_SECOND,
                self.credit_issuer.issue_credit(turn),
            )
        else:
            await self.credit_issuer.issue_credit(turn)

    async def _spawn_from_recycle_or_id(
        self,
        finished_trace_id: str,
        *,
        finished_correlation_id: str,
    ) -> None:
        """Push finished trace_id to recycle tail, spawn fresh session from head.

        If the queue is empty (small dataset), the just-finished trace_id is
        reused immediately because we put then get on the same queue.

        Skipped when the phase has already entered cooldown (stop condition
        fired): in-flight credits returning during cooldown must not re-pop a
        fresh trace from the queue. Cooldown is for finishing, not starting.

        The initial recycle queue spans the full dataset pool (including
        trajectory trace_ids whose sessions are running live at PROFILING
        start; every live lane is pre-registered in ``_active_traces`` by
        ``setup_phase``). The pop loop skips trace_ids in ``_active_traces``
        and re-enqueues them to avoid duplicate concurrent sessions.
        """
        # Prune unconditionally so every early-return path leaves dicts clean.
        self._session_marker.pop(finished_correlation_id, None)
        self._active_traces[finished_trace_id] -= 1
        if self._active_traces[finished_trace_id] <= 0:
            del self._active_traces[finished_trace_id]

        lane = self._release_lane_for(finished_correlation_id, finished_trace_id)

        if self._recycle_queue is None:
            return

        # Double-recycle guard. Raise rather than gate on __debug__ — `python -O`
        # would otherwise let the duplicate-final-turn corruption escape silently.
        if finished_correlation_id in self._in_flight_recycled:
            raise RuntimeError(
                f"Double recycle of correlation_id {finished_correlation_id!r} "
                f"(trace_id={finished_trace_id!r}) - handle_credit_return "
                "invoked twice for the same final turn"
            )
        self._in_flight_recycled.add(finished_correlation_id)

        # Re-enqueue BEFORE the cooldown check so an in-flight credit returning
        # during cooldown can't drop the trace_id from the recycle pool.
        self._recycle_queue.put_nowait(finished_trace_id)

        if not self.stop_checker.can_start_new_session():
            return

        next_trace_id = self._pop_next_eligible_trace()
        if next_trace_id is None:
            return

        session = self._build_session_for_trace(next_trace_id)
        if session is None or not session.metadata.turns:
            # Unspawnable right now (missing metadata / zero turns): re-enqueue
            # so the recycle pool's eligible set is conserved. Dropping it here
            # silently erodes pool diversity for the rest of the phase (the
            # finished trace was already re-enqueued above).
            self._recycle_queue.put_nowait(next_trace_id)
            return

        self._correlation_to_lane[session.x_correlation_id] = lane
        self._active_traces[next_trace_id] += 1
        self._mint_marker_for_session(session.x_correlation_id, next_trace_id, lane)

        turn = self._build_turn_for_session(session, 0)
        await self.credit_issuer.issue_credit(turn)

    async def _dispatch_snapshot_for_profiling(
        self, trajectory: Trajectory, lane: int
    ) -> None:
        """Resume one trajectory's streams for PROFILING.

        Each stream profiles from turn ``next_turn_index`` (the first turn at
        or after t*; its predecessor, if any, was primed during WARMUP).

        Dispatch anchoring depends on ``PRESERVE_TRAJECTORY_START_GAP``:
        by default the trajectory's earliest post-t* request is anchored at
        profiling-time 0 (subtracting T0, the min offset) so all lanes burst
        at once; when the flag is set, T0 is 0 so each stream waits out its
        recorded offset from t* (the leading idle gap is preserved). Relative
        timing among the trajectory's streams and turns is identical either
        way -- only the per-lane start offset differs.

        Gated parents (``waiting_on_children``) are not dispatched here; their
        join is seeded with the orchestrator and their gated turn fires when
        the blocking children complete during PROFILING. No stream completes
        during WARMUP (warmup only ever sends a non-terminal turn), so there
        is no warmup-continuation or terminal-root recycle step.
        """
        snapshot = self._get_snapshot(trajectory)
        for state in snapshot.states:
            self._correlation_to_lane[state.x_correlation_id] = lane
            self._mint_marker_for_session(
                state.x_correlation_id, state.conversation_id, lane
            )

        if self.branch_orchestrator is not None:
            self.branch_orchestrator.seed_snapshot(
                snapshot.states,
                cache_bust_markers=self._session_marker,
            )

        dispatchable = [s for s in snapshot.states if not s.waiting_on_children]
        # T0 anchors the trajectory's earliest post-t* request at profiling-
        # time 0; with PRESERVE_TRAJECTORY_START_GAP the anchor is t* itself
        # (T0 = 0), so the leading t*->first-request idle gap is preserved.
        if self._preserve_trajectory_start_gap:
            t0_offset_ms = 0.0
        else:
            t0_offset_ms = (
                min(s.next_dispatch_offset_ms for s in dispatchable)
                if dispatchable
                else 0.0
            )
        for state in dispatchable:
            session = self.conversation_source.session_for_state(state)
            turn = self._build_turn_for_session(session, state.next_turn_index)
            delay_s = (state.next_dispatch_offset_ms - t0_offset_ms) / MILLIS_PER_SECOND
            if delay_s > 0:
                self.scheduler.schedule_later(
                    delay_s,
                    self.credit_issuer.issue_credit(turn),
                )
            else:
                await self.credit_issuer.issue_credit(turn)

    def _get_snapshot(self, trajectory: Trajectory) -> TrajectorySnapshot:
        """Return the persistent sampled snapshot for a trajectory lane.

        ``TrajectorySource`` constructs each timestamped lane once and is
        shared across WARMUP and PROFILING. Reusing that realized graph keeps
        every continuing root and subagent on the same ``X-Session-ID`` across
        the phase boundary.
        """
        assert trajectory.snapshot is not None
        return trajectory.snapshot

    def _release_lane_for(
        self, finished_correlation_id: str, finished_trace_id: str
    ) -> int:
        """Pop and return the lane for a finished correlation_id.

        Missing entry means upstream bookkeeping was violated; log loudly and
        fall back to lane 0 so recycle still progresses. Silent skip would
        wedge the queue head.
        """
        if finished_correlation_id not in self._correlation_to_lane:
            self.warning(
                lambda: (
                    f"Recycle: finished_correlation_id={finished_correlation_id!r} "
                    f"missing from _correlation_to_lane; bookkeeping invariant "
                    f"violated. Falling back to lane 0 for trace_id={finished_trace_id!r}."
                )
            )
            return 0
        return self._correlation_to_lane.pop(finished_correlation_id)

    def _pop_next_eligible_trace(self) -> str | None:
        """Pop next queued trace_id whose session isn't currently active.

        Bounded by initial qsize so we never busy-loop in the degenerate
        small-pool case where every queued trace_id has a live session.
        """
        if self._recycle_queue is None:
            return None
        scan_budget = self._recycle_queue.qsize()
        while scan_budget > 0:
            scan_budget -= 1
            try:
                candidate = self._recycle_queue.get_nowait()
            except asyncio.QueueEmpty:
                return None
            lane_cap = self._lanes_per_trace.get(candidate, 1) or 1
            if self._active_traces[candidate] >= lane_cap:
                self._recycle_queue.put_nowait(candidate)
                continue
            return candidate
        return None

    def _build_session_for_trace(self, trace_id: str) -> SampledSession | None:
        """Build a fresh SampledSession for a recycled trace_id starting at turn 0."""
        metadata_lookup = self.conversation_source._metadata_lookup
        meta = metadata_lookup.get(trace_id)
        if meta is None:
            self.warning(
                f"Recycled trace_id {trace_id!r} missing from metadata lookup; "
                "skipping spawn"
            )
            return None
        return SampledSession(
            conversation_id=trace_id,
            metadata=meta,
            x_correlation_id=str(uuid.uuid4()),
            start_turn_index=0,
        )

    def _build_turn_for_session(
        self, session: SampledSession, turn_index: int
    ) -> TurnToSend:
        """Build a TurnToSend for the given session at the given turn index."""
        base = session.build_turn_at_index(turn_index)
        marker = self._session_marker.get(session.x_correlation_id)
        if marker is None and self._cache_bust_target == CacheBustTarget.NONE:
            return base
        return _struct_replace(
            base,
            cache_bust_marker=marker,
            cache_bust_target=self._cache_bust_target,
        )

    def _mint_marker_for_session(
        self, x_correlation_id: str, trace_id: str, trajectory_index: int
    ) -> str | None:
        """Mint (or reuse) and store a per-session cache-bust marker.

        Returns None when the feature is disabled (target=NONE), in which
        case the session map records None so callers can unconditionally
        look it up. Increments _recycle_pass[trace_id] each time a new
        session is minted for the same trace_id, so digest rotates across
        recycles.

        Both ``_session_marker`` and ``_recycle_pass`` live on the shared
        ``TrajectorySource`` ledger, surviving the WARMUP -> PROFILING
        boundary (strategies are constructed fresh per phase). A session
        whose x_correlation_id was already minted - a continuing lane
        resuming at k_i+1 - keeps its WARMUP marker verbatim instead of
        re-minting, so a continued session's digest can never rotate at the
        phase boundary regardless of mint order. The pass counter never
        restarts, so fresh sessions (recycles, parents unblocked after t*)
        can never collide with a warmed digest.
        """
        if x_correlation_id in self._session_marker:
            return self._session_marker[x_correlation_id]
        if self._cache_bust_target == CacheBustTarget.NONE:
            self._session_marker[x_correlation_id] = None
            return None
        new_pass = self._recycle_pass.get(trace_id, -1) + 1
        self._recycle_pass[trace_id] = new_pass
        marker = build_cache_bust_marker(
            self._benchmark_id,
            new_pass,
            trajectory_index,
            trace_id,
            target=self._cache_bust_target,
        )
        self._session_marker[x_correlation_id] = marker
        return marker

    def record_warmup_failure(self, trace_id: str) -> None:
        """Accumulate a terminal warmup credit failure for later reporting.

        Invoked by ``CreditCallbackHandler`` on every WARMUP credit return
        whose final turn carried an error or cancellation. Per-trajectory
        attribution stays alongside the trajectory list itself.
        """
        self._failed_warmup_traces.append(trace_id)

    def report_warmup_failures(self) -> None:
        """Raise TrajectoryWarmupFailedError if any warmup credits failed terminally.

        Called by ``PhaseRunner`` at WARMUP teardown. PROFILING must not start
        with a degraded set of trajectories - mixing successful and failed
        warmup traces would silently bias steady-state metrics.
        """
        if self._failed_warmup_traces:
            raise TrajectoryWarmupFailedError(self._failed_warmup_traces)
