# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgenticReplayStrategy - trajectory-driven trace replay timing strategy.

Phase-aware timing strategy for the ``agentic_replay`` timing mode (spec §4.2).

Each trajectory is a wall-clock snapshot of a trace at a sampled instant t*
(25-75% through the trace's recorded duration). Every stream (root + each
subagent chain) splits at t*: turns before t* are history, turns at/after t*
are profiled.

WARMUP: for every session active (mid-flight) at t*, replay its last request
before t* (turn ``next_turn_index - 1``) as a session start. The chat prefix
is rebuilt worker-side by ``UserSession.advance_turn``, and what it reproduces
is context-mode dependent: under ``DELTAS_WITH_RESPONSES`` (the weka default)
it back-seeds the earlier turns so the warmup request carries the full prefix
and primes the server cache to the stream's state at t*; under
``DELTAS_WITHOUT_RESPONSES`` the prior turns are not seeded (live responses
aren't captured yet), so only that turn's delta is sent and the t* prefix is
not reproduced. This includes parents gated on a child join (they sent turn
n-1 before t* and resume at the join turn during PROFILING, so n-1 warms that
turn). Streams whose first request is at/after t* (``next_turn_index == 0``)
have nothing to warm. By default the priming
requests are SPREAD -- aligned globally on t* so every trajectory's t* lands
at the warmup end (see ``_execute_warmup``); ``--burst-phase-starts`` fires
them all at once instead. The phase exits via the standard
``SendingCompleteStopCondition`` plus ``grace_period_sec=inf`` semantics
already in CreditPhaseConfig (count-driven: every turn-n-1 must return).

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
from collections import Counter, deque
from collections.abc import Iterable
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
from aiperf.timing.trajectory_source import (
    Trajectory,
    TrajectorySnapshot,
    TrajectorySource,
    _as_timestamp_ms,
)

_WARMUP_MAX_TOKENS = 1

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
        session_tree_registry=None,
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
        # Per-tree session-slot ledger (agentic replay PROFILING only). When
        # present, a lane's session slot is held until its whole TREE drains
        # (root + every descendant), and recycle of the freed lane is driven by
        # the registry's drain callback (``_on_tree_drained``) rather than fired
        # synchronously on the root's final turn. None -> legacy per-root-credit
        # release + the rootless lane-credit / ``_rootless_lane_outstanding``
        # path below.
        self._session_tree_registry = session_tree_registry

        # Double-recycle guard, keyed on x_correlation_id (not trace_id): the
        # guard's intent is to catch the same final turn firing
        # handle_credit_return twice — a per-session property. trace_id-keying
        # spuriously tripped when two wrap-filled lanes finished the same
        # trace_id with distinct correlation_ids.
        self._in_flight_recycled: set[str] = set()
        # FIFO eviction order for _in_flight_recycled. The guard retains a
        # recycled correlation_id so a duplicate final-turn return (which would
        # double-recycle) still raises; unbounded that is one entry per recycled
        # session for the whole PROFILING phase. Cap the retained window (oldest
        # evicted first) so memory is bounded while the window still spans far
        # more than any realistic duplicate-delivery gap.
        self._recycle_guard_order: deque[str] = deque()
        self._recycle_guard_max_window = Environment.AGENTX.RECYCLE_GUARD_MAX_WINDOW
        # Lane multiplicity per trace_id, frozen at strategy init from the
        # trajectory list; used only for the wrap-fill cache-bust warning below.
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
        self._cache_bust_ledger = ledger
        self._recycle_pass: dict[str, int] = ledger.recycle_pass
        self._session_marker: dict[str, str | None] = ledger.session_marker
        self._correlation_to_lane: dict[str, int] = {}
        # Rootless lanes (root's turns all before t*; only background subagents
        # remain at PROFILING start) hold a lane credit in place of a root
        # credit. Track outstanding background children per lane so the credit
        # is released and the lane recycled into a fresh root once they drain,
        # instead of the lane going dark for the rest of the phase.
        self._rootless_lane_outstanding: dict[int, int] = {}
        self._cache_bust_target: CacheBustTarget = (
            user_config.input.prompt.cache_bust.target
            if user_config is not None
            else CacheBustTarget.NONE
        )
        self._benchmark_id: str = (
            user_config.benchmark_id if user_config is not None else "unknown"
        )
        # ``--burst-phase-starts`` (loadgen.burst_phase_starts). Default False:
        # the WARMUP and PROFILING phase starts are SPREAD by each request's
        # recorded offset from t* (warmup globally t*-aligned, profiling
        # preserving each lane's leading gap). When True, both phase starts
        # collapse into synchronized bursts. Governs ONLY the two phase-start
        # dispatch patterns; the rest of replay timing is faithful regardless.
        # ``is True`` guards MagicMock/None test configs -> default (spread).
        loadgen = getattr(user_config, "loadgen", None)
        self._burst_phase_starts: bool = (
            getattr(loadgen, "burst_phase_starts", False) is True
        )
        # Idle-gap cap (ms) for the t* boundary the load-time warp can't see (t*
        # is the sampling instant, not a request). Consumed two ways:
        #   - WARMUP: clamps each warmup lead so priming doesn't start hours
        #     early (``_capped_warmup_lead_ms``); priming spacing is meaningless.
        #   - PROFILING: a single uniform shift caps the leading idle (t* ->
        #     earliest stream) while preserving recorded inter-stream spacing
        #     (``_leading_idle_shift_ms``); a per-stream clamp would collapse
        #     every idle subagent onto t=cap.
        # None when no cap is set (raw faithful timing -- the user's choice).
        idle_cap_s = getattr(loadgen, "trace_idle_gap_cap_seconds", None)
        self._phase_offset_cap_ms: float | None = (
            idle_cap_s * MILLIS_PER_SECOND
            if isinstance(idle_cap_s, int | float)
            else None
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

    @property
    def _has_tree_registry(self) -> bool:
        """True when per-tree session-slot accounting is engaged (PROFILING)."""
        return (
            self._session_tree_registry is not None
            and self.config.phase == CreditPhase.PROFILING
        )

    def _lane_root_corr(self, snapshot: TrajectorySnapshot) -> str | None:
        """Tree-root id shared by every stream of a snapshot lane.

        All states of one snapshot lane carry the same ``root_correlation_id``
        (the snapshot's synthetic parent_corr -- the root state's own id when a
        root is present, or the shared parent of the background subagents when
        rootless). Falls back to a state's x_correlation_id for snapshots built
        without the field (older fixtures)."""
        return next(
            (s.root_correlation_id or s.x_correlation_id for s in snapshot.states),
            None,
        )

    def _on_tree_drained(self, root_corr: str, phase: CreditPhase) -> None:
        """Registry drain callback: a session tree fully drained and freed its
        slot, so recycle its lane into a fresh root.

        Fired synchronously from the registry when the last of a tree's root +
        descendants completes. Maps the tree root to its lane and schedules the
        (async) recycle on the loop scheduler so it runs after the current
        return finishes and is cancelled cleanly at phase teardown. The slot was
        already released by the registry, so the recycled root's acquire keeps
        occupancy at exactly the configured concurrency.
        """
        lane = self._correlation_to_lane.pop(root_corr, None)
        self._session_marker.pop(root_corr, None)
        if lane is None:
            self.warning(
                lambda: (
                    f"Tree {root_corr!r} drained but has no lane mapping; cannot "
                    "recycle (bookkeeping invariant violated)."
                )
            )
            return
        self.scheduler.schedule_later(0.0, self._dispatch_recycled_on_lane(lane))

    async def setup_phase(self) -> None:
        """Phase-specific async setup.

        WARMUP: nothing - trajectories already built by TrajectorySource at
        TimingManager construction time.

        PROFILING: register the per-tree drain callback (when engaged) so a
        drained lane recycles into a fresh root. Recycle otherwise draws the
        next root directly from the shared dataset sampler
        (``TrajectorySource.next_recycle_conversation_id``), so it honours the
        dataset's ``sampling_strategy`` and reuses every root about equally --
        no strategy-side recycle queue to keep in sync.
        """
        if self._has_tree_registry:
            self._session_tree_registry.set_drain_callback(self._on_tree_drained)
        if self.config.phase == CreditPhase.PROFILING:
            if not self.conversation_source.trajectories:
                raise RuntimeError(
                    "AgenticReplayStrategy PROFILING setup: trajectories empty. "
                    "WARMUP must complete with at least one trajectory before "
                    "PROFILING can start. Check loader output and warmup failures."
                )
            self.info(
                f"PROFILING setup: {len(self.conversation_source.trajectories)} "
                "trajectory lanes; recycle draws roots from the dataset sampler"
            )
            rootless, gated = self._lane_credit_lane_counts()
            if rootless or gated:
                self.info(
                    f"PROFILING: {rootless} rootless + {gated} gated-parent lanes "
                    f"of {len(self.conversation_source.trajectories)} dispatch no "
                    f"root credit at start and hold a lane credit instead (so they "
                    f"still count toward concurrency); rootless lanes recycle into a "
                    f"fresh root once their background subagents drain"
                )

    async def execute_phase(self) -> None:
        """Dispatch initial credits for the phase."""
        if self.config.phase == CreditPhase.WARMUP:
            await self._execute_warmup()
        else:
            await self._execute_profiling()

    def _capped_warmup_lead_ms(self, lead_ms: float) -> float:
        """Clamp a WARMUP lead (t* - the warmed turn) to the idle-gap cap.

        Warmup is a cache-priming pass, not faithful replay: warming a stream
        that last fired hours before t* that far ahead would make the warmup
        phase itself take hours. Clamping each lead bounds the priming window
        to the cap; the relative timing among warmup requests carries no replay
        meaning (each primes its own session, all converge at t*), so an
        independent per-lead clamp is fine here. No-op when no cap is set.

        PROFILING dispatch offsets are NOT clamped this way -- see
        :meth:`_leading_idle_shift_ms`.
        """
        if self._phase_offset_cap_ms is not None:
            return min(lead_ms, self._phase_offset_cap_ms)
        return lead_ms

    def _leading_idle_shift_ms(self, offsets: Iterable[float]) -> float:
        """Excess to subtract UNIFORMLY from every PROFILING dispatch offset so
        a trajectory's leading idle (t* -> its earliest post-t* request) is
        capped to the idle-gap cap.

        The idle-gap warp (applied at load time across ALL of a trace's request
        timestamps) already bounds every inter-request gap to the cap, so the
        dispatch offsets carry faithful relative spacing. The ONE gap the warp
        cannot see is t* -> first request: t* is the trajectory sampling
        instant, not a request. A single uniform shift caps that leading gap
        while keeping every stream's recorded offset relative to the others
        intact -- unlike a per-stream ``min(offset, cap)`` clamp, which collapses
        every offset above the cap onto it, firing every idle subagent at the
        same instant.

        Returns 0 when no cap is set or the leading idle is already within it.
        """
        if self._phase_offset_cap_ms is None:
            return 0.0
        present = [o for o in offsets if o is not None]
        if not present:
            return 0.0
        return max(0.0, min(present) - self._phase_offset_cap_ms)

    async def _execute_warmup(self) -> None:
        """Warm turn n-1 of every session active (mid-flight) at t*.

        Pass 1 prepares a warmup credit for every active session and records
        its lead delta (how long before that trajectory's t* the warmed
        request fired). Pass 2 dispatches them. Lane registration and marker
        minting happen synchronously in pass 1 regardless of dispatch timing.

        Every session mid-flight at t* is warmed at its last request before t*,
        priming the server cache to its t* state. This INCLUDES a parent gated
        on a child join: it sent turn n-1 before t* and resumes at the join
        turn n during PROFILING, so warming n-1 primes that join turn's prefix.
        Streams whose first request is at/after t* (``warmup_turn_index is
        None``) have nothing to warm.

        Dispatch timing:
          - Default (spread): aligned GLOBALLY across all trajectories so every
            trajectory's t* lands at the same moment (the warmup end). A
            request that fired ``lead`` ms before its t* dispatches at
            ``max_lead - lead`` (max_lead over ALL warmup requests): the one
            furthest before its t* fires at warmup-time 0, requests closer to
            their t* fire later, total spread = ``max_lead - min_lead``.
            ``mark_sending_complete`` is skipped (the deferred dispatches must
            not be refused early); the count path drives completion once the
            last scheduled dispatch fires, and warmup has no duration timeout
            to cancel the spread.
          - ``--burst-phase-starts``: all warmup credits fire at once.
        """
        warmup_total_count = self.conversation_source.warmup_credit_count
        self.info(
            f"WARMUP execute: dispatching {warmup_total_count} trajectory credits"
        )
        spread = not self._burst_phase_starts

        # Pass 1: register lane + mint marker + build the turn for every active
        # session. ``lead_ms`` = how long before that trajectory's t* the warmed
        # request fired (None for timestamp-less lanes / missing timestamps).
        prepared: list[tuple[TurnToSend, float | None]] = []
        for lane, trajectory in enumerate(self.conversation_source.trajectories):
            if trajectory.snapshot is None:
                session = self.conversation_source.session_for(trajectory)
                self._correlation_to_lane[session.x_correlation_id] = lane
                self._mint_marker_for_session(
                    session.effective_root_correlation_id,
                    trajectory.conversation_id,
                    lane,
                )
                turn = self._build_turn_for_session(
                    session, trajectory.start_turn_index
                )
                prepared.append((turn, None))
                continue

            t_star_ms = trajectory.snapshot.t_star_ms
            for state in trajectory.snapshot.states:
                warm_index = state.warmup_turn_index
                if warm_index is None:
                    continue
                session = self.conversation_source.session_for_state(state)
                self._correlation_to_lane[session.x_correlation_id] = lane
                self._mint_marker_for_session(
                    session.effective_root_correlation_id, state.conversation_id, lane
                )
                turn = self._build_turn_for_session(session, warm_index)
                lead_ms: float | None = None
                if spread:
                    meta = self.conversation_source.get_metadata(state.conversation_id)
                    warm_ts = _as_timestamp_ms(
                        getattr(meta.turns[warm_index], "timestamp_ms", None)
                    )
                    if warm_ts is not None:
                        lead_ms = self._capped_warmup_lead_ms(t_star_ms - warm_ts)
                prepared.append((turn, lead_ms))

        # Pass 2: dispatch.
        if not spread:
            self.info(
                "WARMUP burst: dispatching all warmup requests at once (spread 0.0s)"
            )
            for turn, _ in prepared:
                await self.credit_issuer.issue_credit(turn)
            if not self.lifecycle.is_sending_complete:
                self.lifecycle.mark_sending_complete()
            return

        # Global t*-alignment: a request that fired ``lead`` before its t*
        # dispatches at ``max_lead - lead`` so every trajectory's t* lands at
        # the same instant (warmup-time ``max_lead``); the furthest-before-t*
        # request fires at 0. Do NOT mark sending complete -- the count path
        # finalizes once the last scheduled dispatch fires.
        leads = [d for _, d in prepared if d is not None]
        max_lead_ms = max(leads, default=0.0)
        spread_s = (max_lead_ms - min(leads, default=0.0)) / MILLIS_PER_SECOND
        self.info(
            f"WARMUP spread: {spread_s:.1f}s ramp aligning {len(leads)} request(s) "
            f"on t* (earliest fires at 0, last at {spread_s:.1f}s)"
        )
        for turn, lead_ms in prepared:
            offset_s = (
                (max_lead_ms - lead_ms) / MILLIS_PER_SECOND
                if lead_ms is not None
                else 0.0
            )
            if offset_s > 0:
                self.scheduler.schedule_later(
                    offset_s, self.credit_issuer.issue_credit(turn)
                )
            else:
                await self.credit_issuer.issue_credit(turn)

    async def _execute_profiling(self) -> None:
        """Resume each trajectory at ``k_i + 1`` to seed the steady state.

        All trajectories are dispatched concurrently so the full concurrency
        target is reached as fast as slot limits allow, rather than
        serializing over N credit round-trips. Subsequent turns and
        recycle-pool sessions are dispatched from handle_credit_return.
        """
        spread_s = self._profiling_spread_seconds()
        mode = "burst" if self._burst_phase_starts else "spread"
        self.info(
            f"PROFILING execute: resuming {len(self.conversation_source.trajectories)} "
            f"trajectory sessions ({mode}; first-request spread {spread_s:.1f}s)"
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

    def _profiling_spread_seconds(self) -> float:
        """Window over which each trajectory's FIRST request fires, in seconds.

        Per trajectory the first request is its earliest dispatchable stream
        (``min`` of the lane offsets after the leading-idle shift; t0=0 spread,
        or t0=lane-min burst). This returns max-minus-min of those per-trajectory
        first-request offsets -- the ramp-in window. Later streams in a
        trajectory (subagents that spawn further out at their own faithful
        offsets) are excluded: including them would report the full per-
        trajectory replay span, not how long the trajectories take to start.
        Logged once at phase start; 0.0 when there is nothing to spread.
        """
        first_offsets: list[float] = []
        for trajectory in self.conversation_source.trajectories:
            if trajectory.snapshot is None:
                continue
            dispatchable = [
                s for s in trajectory.snapshot.states if not s.waiting_on_children
            ]
            if not dispatchable:
                continue
            leading_shift_ms = self._leading_idle_shift_ms(
                s.next_dispatch_offset_ms for s in dispatchable
            )
            lane_offsets = [
                s.next_dispatch_offset_ms - leading_shift_ms for s in dispatchable
            ]
            lane_t0 = min(lane_offsets) if self._burst_phase_starts else 0.0
            first_offsets.append(min(lane_offsets) - lane_t0)
        if not first_offsets:
            return 0.0
        return (max(first_offsets) - min(first_offsets)) / MILLIS_PER_SECOND

    async def _dispatch_one_profiling_trajectory(
        self, trajectory: Trajectory, lane: int
    ) -> None:
        """Dispatch one lane's initial PROFILING credit (run under gather)."""
        if trajectory.snapshot is not None:
            await self._dispatch_snapshot_for_profiling(trajectory, lane)
            return

        session = self.conversation_source.session_for(trajectory)
        self._correlation_to_lane[session.x_correlation_id] = lane
        self._mint_marker_for_session(
            session.effective_root_correlation_id, trajectory.conversation_id, lane
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
            lane = self._correlation_to_lane.pop(credit.x_correlation_id, None)
            # Under the registry the orchestrator's descendant-done hook drives
            # tree drain + lane recycle (see _on_tree_drained); the legacy path
            # tracks rootless drain here instead.
            if (
                not self._has_tree_registry
                and lane is not None
                and lane in self._rootless_lane_outstanding
            ):
                await self._on_rootless_child_done(lane)
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

        # Under the registry, the root's slot is held until its WHOLE tree
        # drains: the callback handler already called registry.on_root_terminal
        # (after intercept), and recycle of the freed lane is driven by the
        # registry drain callback (_on_tree_drained) once every descendant has
        # also finished. Recycling here would start a fresh root while this
        # tree's background subagents are still running -> concurrency overshoot.
        if self._has_tree_registry:
            return

        await self._spawn_from_recycle_or_id(
            credit.conversation_id,
            finished_correlation_id=credit.x_correlation_id,
        )

    async def _dispatch_next_turn(self, credit: Credit) -> None:
        """Issue the next turn of an in-progress session, honoring delay_ms.

        DAG child continuations (``agent_depth > 0``) go through the single
        child-issuance chokepoint (``_issue_child_continuation_or_drain``) so a
        ``--request-count`` cap refusal is routed to ``on_child_stopped`` (drain
        the parent join) instead of being silently swallowed by the discarded
        ``issue_credit`` return — including on the delayed (``delay_ms``) path,
        where the refusal would otherwise fire long after the callback handler
        decided the child could proceed. Root continuations keep ``issue_credit``.
        """
        next_meta = self.conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit, next_meta)
        is_child = turn.agent_depth > 0

        coro = (
            self._issue_child_continuation_or_drain(turn)
            if is_child
            else self.credit_issuer.issue_credit(turn)
        )
        if next_meta.delay_ms is not None and next_meta.delay_ms > 0:
            self.scheduler.schedule_later(next_meta.delay_ms / MILLIS_PER_SECOND, coro)
        else:
            await coro

    async def _issue_child_continuation_or_drain(self, turn: TurnToSend) -> None:
        """Single child-issuance chokepoint (dataflow-inspired IssuanceAuthority).

        Dispatch a DAG child continuation via ``dispatch_child_turn`` (which
        returns a clean True-iff-on-wire, avoiding ``issue_credit``'s overloaded
        False) and, on ANY refusal (e.g. the ``--request-count`` wire cap), notify
        ``BranchOrchestrator.on_child_stopped`` so the parent's join drains
        deterministically rather than deadlocking on a child whose remaining
        turns will never be issued. Centralizing here means no dispatch site can
        "forget" to drain on refusal.
        """
        on_wire = await self.credit_issuer.dispatch_child_turn(turn)
        if not on_wire and self.branch_orchestrator is not None:
            await self.branch_orchestrator.on_child_stopped(turn.x_correlation_id)

    async def _spawn_from_recycle_or_id(
        self,
        finished_trace_id: str,
        *,
        finished_correlation_id: str,
    ) -> None:
        """Recycle a finished root: release its lane and start a fresh session.

        The next root is drawn from the dataset sampler (see
        ``_dispatch_recycled_on_lane``), so the just-finished trace_id no longer
        needs re-enqueuing -- the sampler will hand it back in its own rotation.
        Skipped when the phase has entered cooldown (in-flight returns must not
        start new sessions); the double-recycle guard still fires so a duplicate
        final-turn return can't spawn twice.
        """
        # Double-recycle guard. Raise rather than gate on __debug__ — `python -O`
        # would otherwise let the duplicate-final-turn corruption escape silently.
        if finished_correlation_id in self._in_flight_recycled:
            raise RuntimeError(
                f"Double recycle of correlation_id {finished_correlation_id!r} "
                f"(trace_id={finished_trace_id!r}) - handle_credit_return "
                "invoked twice for the same final turn"
            )
        self._in_flight_recycled.add(finished_correlation_id)
        # Bound the guard set: evict the oldest retained correlation_ids once the
        # window is full so a long, high-throughput run does not accumulate one
        # entry per recycled session forever.
        self._recycle_guard_order.append(finished_correlation_id)
        while len(self._recycle_guard_order) > self._recycle_guard_max_window:
            self._in_flight_recycled.discard(self._recycle_guard_order.popleft())

        # Prune so every early-return path leaves dicts clean.
        self._session_marker.pop(finished_correlation_id, None)
        lane = self._release_lane_for(finished_correlation_id, finished_trace_id)
        await self._dispatch_recycled_on_lane(lane)

    async def _dispatch_recycled_on_lane(self, lane: int) -> None:
        """Draw the next root from the dataset sampler and dispatch its turn-0
        session onto ``lane``.

        Shared by ``_spawn_from_recycle_or_id`` (a finished root recycling) and
        ``_on_rootless_child_done`` (a drained rootless lane recycling into its
        first real root). The recycled session restarts at turn 0 with a fresh
        x_correlation_id and a freshly minted cache-bust marker (nothing
        warmed). No-op during cooldown -- in-flight returns must not start new
        sessions -- or when the sampler yields no spawnable root.
        """
        if not self.stop_checker.can_start_new_session():
            return

        next_trace_id = self.conversation_source.next_recycle_conversation_id()
        if next_trace_id is None:
            return

        session = self._build_session_for_trace(next_trace_id)
        if session is None or not session.metadata.turns:
            return

        self._correlation_to_lane[session.x_correlation_id] = lane
        self._mint_marker_for_session(
            session.effective_root_correlation_id, next_trace_id, lane
        )

        turn = self._build_turn_for_session(session, 0)
        await self.credit_issuer.issue_credit(turn)

    async def _on_rootless_child_done(self, lane: int) -> None:
        """Account a rootless lane's background child completing.

        Until the last background child drains, the lane keeps its lane credit
        (it is still doing background work). When the final one finishes,
        release the credit and recycle the lane into a fresh turn-0 root so it
        keeps contributing load for the rest of the phase instead of going dark
        (the leak this whole path fixes). The fresh root acquires its own
        session slot via ``issue_credit``; releasing the lane credit first keeps
        the slot budget balanced.
        """
        remaining = self._rootless_lane_outstanding.get(lane, 0) - 1
        if remaining > 0:
            self._rootless_lane_outstanding[lane] = remaining
            return

        self._rootless_lane_outstanding.pop(lane, None)
        self.credit_issuer.release_lane_credit()
        await self._dispatch_recycled_on_lane(lane)

    async def _dispatch_snapshot_for_profiling(
        self, trajectory: Trajectory, lane: int
    ) -> None:
        """Resume one trajectory's streams for PROFILING.

        Each stream profiles from turn ``next_turn_index`` (the first turn at
        or after t*; its predecessor, if any, was primed during WARMUP).

        Dispatch anchoring depends on ``--burst-phase-starts``: by default
        (spread) ``t0_offset = 0`` so each stream waits out its recorded offset
        from t* -- the leading t*->first-request gap is preserved and lanes
        ramp in. With ``--burst-phase-starts`` the trajectory's earliest
        post-t* request is anchored at profiling-time 0 (subtracting T0, the
        min offset) so the lane bursts at once. Relative timing among the
        trajectory's streams and turns is identical either way -- only the
        per-lane start offset differs.

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
                state.root_correlation_id or state.x_correlation_id,
                state.conversation_id,
                lane,
            )

        if self.branch_orchestrator is not None:
            self.branch_orchestrator.seed_snapshot(
                snapshot.states,
                cache_bust_markers=self._session_marker,
            )

        dispatchable = [s for s in snapshot.states if not s.waiting_on_children]
        # A lane needs its own session credit when it dispatches no
        # slot-acquiring depth-0 root credit at PROFILING start. Two cases:
        #   - rootless: the root's turns are all before t*, so the snapshot has
        #     no root state at all -- only its background ::fa:/::aux: subagents.
        #   - gated parent: a root state exists but waits on a child join, so it
        #     is excluded from ``dispatchable`` and resumes only when its
        #     children complete.
        # Without a lane credit such a lane holds no session slot: rootless
        # silently drops below --concurrency, and a gated parent over-releases
        # the limiter when its join's final turn later fires. Acquire one slot
        # for the LANE itself; its subagents/sidecars still acquire none.
        has_dispatchable_root = any(
            s.conversation_id == trajectory.conversation_id
            and not s.waiting_on_children
            for s in snapshot.states
        )
        has_root_state = any(
            s.conversation_id == trajectory.conversation_id for s in snapshot.states
        )
        if not has_dispatchable_root and dispatchable:
            lane_root_corr = self._lane_root_corr(snapshot)
            # The lane credit IS this tree's session slot. root_pending=True for
            # a gated parent (its root credit will still run the join turn and
            # reach a terminal turn); False for a truly rootless lane (no root
            # credit ever -- it drains on its background subagents alone). Under
            # the registry the slot is released and the lane recycled when the
            # tree drains; the legacy path uses _rootless_lane_outstanding.
            if self._has_tree_registry and lane_root_corr is not None:
                self._correlation_to_lane[lane_root_corr] = lane
            await self.credit_issuer.acquire_lane_credit(
                lane_root_corr, root_pending=has_root_state
            )
            if not self._has_tree_registry and not has_root_state:
                self._rootless_lane_outstanding[lane] = len(dispatchable)
        # Cap the leading idle (t* -> the trajectory's earliest post-t* request)
        # by shifting EVERY stream left by the same excess, so an idle-at-t*
        # trajectory ramps in within the cap instead of going dead for the whole
        # run -- while preserving the recorded spacing among streams. A per-stream
        # min(offset, cap) clamp would instead collapse every idle subagent onto
        # t=cap. The idle-gap warp already bounded inter-request gaps at load
        # time; this fixes the one gap it cannot see (t* is not a request).
        # Spread (default): t0 = 0, each lane fires at its (shifted) offset from
        # t*. Burst (--burst-phase-starts): t0 = the lane's min offset, anchoring
        # the earliest post-t* request at profiling-0.
        leading_shift_ms = self._leading_idle_shift_ms(
            s.next_dispatch_offset_ms for s in dispatchable
        )
        offset_by_corr = {
            s.x_correlation_id: s.next_dispatch_offset_ms - leading_shift_ms
            for s in dispatchable
        }
        if self._burst_phase_starts and offset_by_corr:
            t0_offset_ms = min(offset_by_corr.values())
        else:
            t0_offset_ms = 0.0
        for state in dispatchable:
            session = self.conversation_source.session_for_state(state)
            turn = self._build_turn_for_session(session, state.next_turn_index)
            delay_s = (
                offset_by_corr[state.x_correlation_id] - t0_offset_ms
            ) / MILLIS_PER_SECOND
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

    def _lane_credit_lane_counts(self) -> tuple[int, int]:
        """Count PROFILING lanes that dispatch no root credit at start and so
        hold a lane credit: ``(rootless, gated_parent)``.

        Rootless = the snapshot has no root state (the root's turns are all
        before t*); gated = a root state exists but waits on a child join. Only
        lanes with at least one dispatchable stream are counted (an empty lane
        dispatches nothing and takes no credit). Mirrors the dispatch-time
        condition in ``_dispatch_snapshot_for_profiling``; used for the setup
        log so an under-target run is diagnosable.
        """
        rootless = gated = 0
        for trajectory in self.conversation_source.trajectories:
            snapshot = trajectory.snapshot
            if snapshot is None:
                continue
            states = snapshot.states
            if not any(not s.waiting_on_children for s in states):
                continue
            if any(
                s.conversation_id == trajectory.conversation_id
                and not s.waiting_on_children
                for s in states
            ):
                continue
            if any(s.conversation_id == trajectory.conversation_id for s in states):
                gated += 1
            else:
                rootless += 1
        return rootless, gated

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
        marker = self._session_marker.get(session.effective_root_correlation_id)
        updates: dict[str, object] = {}
        if self.config.phase == CreditPhase.WARMUP:
            updates["max_tokens_override"] = _WARMUP_MAX_TOKENS
        if marker is not None or self._cache_bust_target != CacheBustTarget.NONE:
            updates["cache_bust_marker"] = marker
            updates["cache_bust_target"] = self._cache_bust_target
        if not updates:
            return base
        return _struct_replace(base, **updates)

    def _mint_marker_for_session(
        self, root_correlation_id: str, conversation_id: str, trajectory_index: int
    ) -> str | None:
        """Mint (or reuse) the cache-bust marker for a session's trajectory TREE.

        Keyed by ``root_correlation_id`` (not the session's own id), so the
        depth-0 root and every descendant (subagents, flat agents) of one tree
        resolve a single shared marker — the tree is one prefix-cache domain.
        The digest is taken on the base trace id (``conversation_id`` stripped of
        any ``::sa:``/``::fa:`` suffix) and the tree lane, so whichever member
        resolves first mints the same value; the rest reuse it.

        Returns None when the feature is disabled (target=NONE), recording the
        None so callers can look it up unconditionally. ``_recycle_pass`` bumps
        once per fresh tree, so the digest rotates across recycles.

        The ledger survives the WARMUP -> PROFILING boundary (strategies are
        constructed fresh per phase), so a tree continuing across the boundary
        keeps its marker (idempotent reuse) while fresh trees draw a new pass.
        """
        from aiperf.timing.strategies.cache_bust import resolve_tree_marker

        return resolve_tree_marker(
            self._cache_bust_ledger,
            root_correlation_id,
            benchmark_id=self._benchmark_id,
            trajectory_index=trajectory_index,
            conversation_id=conversation_id,
            target=self._cache_bust_target,
        )

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
