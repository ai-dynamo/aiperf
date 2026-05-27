# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgenticReplayStrategy - trajectory-driven trace replay timing strategy.

Phase-aware timing strategy for the ``agentic_replay`` timing mode (spec §4.2).

WARMUP: dispatch one credit per trajectory at that trajectory's sampled turn
index ``k_i``. The phase exits via the standard ``SendingCompleteStopCondition``
plus ``grace_period_sec=inf`` semantics already in CreditPhaseConfig (the
warmup barrier).

Warmup-failure accumulation: terminal failures (``credit_return.error`` or
``credit_return.cancelled``) on a WARMUP credit's final turn are routed by
``CreditCallbackHandler`` into ``record_warmup_failure(trace_id)``. At
WARMUP teardown, ``PhaseRunner`` calls ``report_warmup_failures()`` which
raises ``TrajectoryWarmupFailedError`` if any failures were recorded. This
aborts PROFILING so steady-state metrics aren't silently biased by a
degraded trajectory pool.

PROFILING: each trajectory resumes at ``k_i + 1``; subsequent turns honor
trace inter-turn ``delay_ms`` (already clamped upstream in the loader). When
a session reaches its final turn, its trace_id is recycled FIFO-style and a
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
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.common.scenario.context_overflow import is_context_overflow_response
from aiperf.credit.structs import TurnToSend
from aiperf.timing.conversation_source import SampledSession
from aiperf.timing.strategies.cache_bust import build_cache_bust_marker
from aiperf.timing.trajectory_source import (
    ConversationState,
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
        self._warmup_completed_count: int = 0
        self._warmup_total_count: int = 0
        # Track which x_correlation_ids correspond to trajectories in WARMUP
        # so that terminal failures can be attributed to a trace_id.
        self._warmup_correlation_to_trace: dict[str, str] = {}
        # Per-trajectory (k_i, num_turns) recorded at warmup dispatch so the
        # warmup-completion log line can show the actual start position and
        # how far into the trace the trajectory began.
        self._warmup_correlation_to_start_info: dict[str, tuple[int, int]] = {}

        # Cache-bust state. WARMUP and PROFILING construct distinct strategy
        # instances (PhaseRunner builds a fresh AgenticReplayStrategy per
        # phase) and ``session_for(...)`` mints a new uuid per call, so the
        # two phases use different ``x_correlation_id``s for the same
        # trajectory. The MARKER text, however, is spec-required to be
        # warmup-coherent: the digest is computed from
        # ``(benchmark_id, recycle_pass, trajectory_index, trace_id)`` —
        # phase-agnostic — so warmup turn k_i and profile turn k_i+1 get
        # the same marker even though they belong to different sessions.
        # That preserves the KV-cache lineage warmup is meant to prime.
        # trajectory_index is stable per "lane" (slot in the trajectory list)
        # and reused on recycle, so the digest changes only across recycle
        # passes for a given trace_id.
        self._recycle_pass: dict[str, int] = {}
        self._session_marker: dict[str, str | None] = {}
        self._correlation_to_lane: dict[str, int] = {}
        self._cache_bust_target: CacheBustTarget = (
            user_config.input.prompt.cache_bust.target
            if user_config is not None
            else CacheBustTarget.NONE
        )
        self._benchmark_id: str = (
            user_config.benchmark_id if user_config is not None else "unknown"
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
        trace_ids (including trajectory ids). Trajectories run live at
        PROFILING start (resumed at k_i+1); the pop loop in
        ``_spawn_from_recycle_or_id`` skips trace_ids whose session is
        currently active so we never spawn a duplicate concurrent session.
        """
        if self.config.phase == CreditPhase.PROFILING:
            if not self.conversation_source.trajectories:
                raise RuntimeError(
                    "AgenticReplayStrategy PROFILING setup: trajectories empty. "
                    "WARMUP must complete with at least one trajectory before "
                    "PROFILING can start. Check loader output and warmup failures."
                )
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
        self._warmup_total_count = self.conversation_source.warmup_credit_count
        self.info(
            f"WARMUP execute: dispatching {self._warmup_total_count} trajectory credits"
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
                self._record_warmup_turn(turn.x_correlation_id, session, dispatch_index)
                await self.credit_issuer.issue_credit(turn)
                continue

            states = self._materialize_snapshot(trajectory).states
            for state in states:
                if state.waiting_on_children:
                    continue
                session = self.conversation_source.session_for_state(state)
                self._correlation_to_lane[session.x_correlation_id] = lane
                self._mint_marker_for_session(
                    session.x_correlation_id, state.conversation_id, lane
                )
                turn = self._build_turn_for_session(session, state.next_turn_index)
                self._record_warmup_turn(
                    turn.x_correlation_id, session, state.next_turn_index
                )
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

        Subsequent turns and recycle-pool sessions are dispatched from
        handle_credit_return.
        """
        self.info(
            f"PROFILING execute: resuming {len(self.conversation_source.trajectories)} "
            f"trajectory sessions"
        )
        for lane, trajectory in enumerate(self.conversation_source.trajectories):
            if trajectory.snapshot is not None:
                await self._dispatch_snapshot_for_profiling(trajectory, lane)
                continue

            session = self.conversation_source.session_for(trajectory)
            self._correlation_to_lane[session.x_correlation_id] = lane
            self._active_traces[trajectory.conversation_id] += 1
            self._mint_marker_for_session(
                session.x_correlation_id, trajectory.conversation_id, lane
            )
            resume_index = trajectory.start_turn_index + 1
            num_turns = len(session.metadata.turns)

            if resume_index >= num_turns:
                # Trajectory's k_i was already the last turn (rare: happens
                # only for very short traces). Skip directly to recycle.
                self.debug(
                    lambda cid=trajectory.conversation_id,
                    k=trajectory.start_turn_index,
                    n=num_turns: f"Trajectory {cid} k_i={k} >= last turn (n={n}); recycling immediately"
                )
                await self._spawn_from_recycle_or_id(
                    trajectory.conversation_id,
                    finished_correlation_id=session.x_correlation_id,
                )
                continue

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
            self._warmup_completed_count += 1
            cid = credit.x_correlation_id
            lane = self._correlation_to_lane.get(cid, -1)
            trace_id = self._warmup_correlation_to_trace.get(cid, "?")
            start_info = self._warmup_correlation_to_start_info.get(cid)
            if start_info is not None:
                k_i, n_turns = start_info
                pct = (k_i / n_turns * 100.0) if n_turns > 0 else 0.0
                start_desc = f"start_turn={k_i}/{n_turns} ({pct:.0f}% through trace)"
            else:
                start_desc = "start_turn=?/?"
            status = "error" if error is not None else "ok"
            self.info(
                lambda c=self._warmup_completed_count,
                t=self._warmup_total_count,
                s=status,
                ln=lane,
                tid=trace_id,
                sd=start_desc: (
                    f"WARMUP {c}/{t} returned [{s}] (lane={ln}, trace_id={tid}, {sd})"
                )
            )
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
                await self.branch_orchestrator.on_child_stopped(
                    credit.x_correlation_id
                )
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
        start). The pop loop skips trace_ids in ``_active_traces`` and
        re-enqueues them to avoid duplicate concurrent sessions.
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
            return

        self._correlation_to_lane[session.x_correlation_id] = lane
        self._active_traces[next_trace_id] += 1
        self._mint_marker_for_session(session.x_correlation_id, next_trace_id, lane)

        turn = self._build_turn_for_session(session, 0)
        await self.credit_issuer.issue_credit(turn)

    def _record_warmup_turn(
        self,
        x_correlation_id: str,
        session: SampledSession,
        turn_index: int,
    ) -> None:
        self._warmup_correlation_to_trace[x_correlation_id] = session.conversation_id
        self._warmup_correlation_to_start_info[x_correlation_id] = (
            turn_index,
            len(session.metadata.turns),
        )

    async def _dispatch_snapshot_for_profiling(
        self, trajectory: Trajectory, lane: int
    ) -> None:
        snapshot = self._materialize_snapshot(trajectory)
        for state in snapshot.states:
            self._correlation_to_lane[state.x_correlation_id] = lane
            if state.agent_depth == 0:
                self._active_traces[state.conversation_id] += 1
            self._mint_marker_for_session(
                state.x_correlation_id, state.conversation_id, lane
            )

        if self.branch_orchestrator is not None:
            self.branch_orchestrator.seed_snapshot(
                snapshot.states,
                cache_bust_markers=self._session_marker,
            )

        for state in snapshot.states:
            if state.waiting_on_children:
                continue

            session = self.conversation_source.session_for_state(state)
            turn = self._build_turn_for_session(session, state.next_turn_index)
            delay_s = state.next_dispatch_offset_ms / MILLIS_PER_SECOND
            if delay_s > 0:
                self.scheduler.schedule_later(
                    delay_s,
                    self.credit_issuer.issue_credit(turn),
                )
            else:
                await self.credit_issuer.issue_credit(turn)

    def _materialize_snapshot(self, trajectory: Trajectory) -> TrajectorySnapshot:
        assert trajectory.snapshot is not None
        corr_map = {
            state.x_correlation_id: str(uuid.uuid4())
            for state in trajectory.snapshot.states
        }
        states: list[ConversationState] = []
        for state in trajectory.snapshot.states:
            parent_corr = (
                corr_map.get(state.parent_correlation_id)
                if state.parent_correlation_id is not None
                else None
            )
            states.append(
                ConversationState(
                    conversation_id=state.conversation_id,
                    x_correlation_id=corr_map[state.x_correlation_id],
                    next_turn_index=state.next_turn_index,
                    next_dispatch_offset_ms=state.next_dispatch_offset_ms,
                    agent_depth=state.agent_depth,
                    parent_correlation_id=parent_corr,
                    waiting_on_children=state.waiting_on_children,
                    join_target_turn_index=state.join_target_turn_index,
                    branch_id=state.branch_id,
                    branch_mode=state.branch_mode,
                )
            )
        return TrajectorySnapshot(
            t_star_ms=trajectory.snapshot.t_star_ms,
            states=tuple(states),
        )

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
        """Mint and store a per-session cache-bust marker.

        Returns None when the feature is disabled (target=NONE), in which
        case the session map records None so callers can unconditionally
        look it up. Increments _recycle_pass[trace_id] each time a new
        session is minted for the same trace_id, so digest rotates across
        recycles within a single phase.

        The strategy is constructed FRESH for each phase (per the
        TimingStrategyProtocol contract; PhaseRunner builds a new instance for
        WARMUP and another for PROFILING). Both phases start with empty
        ``_recycle_pass``, so the first mint for a given trace_id in PROFILING
        produces ``pass=0`` — matching WARMUP's pass=0 digest for the same
        (trace_id, lane) pair. Note that WARMUP and PROFILING use *different*
        x_correlation_ids for the same trajectory (``session_for(...)`` mints a
        fresh uuid per call), so the marker is not literally reused across the
        boundary; rather, the digest *value* coincides because (benchmark_id,
        pass=0, trajectory_index, trace_id) does.
        """
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
