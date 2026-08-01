# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgenticReplayStrategy - trajectory-driven trace replay timing strategy.

Phase-aware timing strategy for the ``agentic_replay`` timing mode (spec §4.2).

Each trajectory is a wall-clock snapshot of a trace at a sampled instant t*
(by default 25-75% through the trace's recorded duration; the range is
configurable via ``trajectory_start_{min,max}_ratio`` and the
inferencex-agentx-mvp scenario widens it to 0-100%). Every stream (root + each
subagent chain) splits at t*: turns before t* are history, turns at/after t*
are profiled.

WARMUP: for every session active (mid-flight) at t*, replay its last request
before t* (turn ``next_turn_index - 1``) as a session start. The chat prefix
is rebuilt worker-side by ``UserSession.advance_turn``, and what it reproduces
is context-mode dependent: under ``DELTAS_WITH_RESPONSES`` (the weka mode)
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
(``next_turn_index``). Default dispatch subtracts one phase-wide minimum from
every stream's recorded offset: the earliest eligible request fires at
profiling-time 0 while all cross-trajectory spacing and ordering are preserved.
``--burst-phase-starts`` instead collapses each lane's first eligible request
to profiling-time 0. Accelerated cache warmup synthesizes a new replay boundary
at the warmup handoff and carries each live stream's full next-turn delay into
profiling; time spent waiting for the global warmup barrier does not consume
that delay. Subsequent turns honor trace inter-turn
``delay_ms`` from the original trace timeline. Gated parents fire their
join turn when blocking children complete. When a root session reaches its
final turn AND its whole tree (root + every descendant subagent) has drained,
its lane recycles: a fresh session (starting at turn 0) is spawned from the
next root drawn from the shared dataset sampler
(``TrajectorySource.next_recycle_conversation_id``), honoring the configured
``sampling_strategy`` -- there is no strategy-side FIFO recycle queue.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter, deque
from typing import TYPE_CHECKING

from msgspec.structs import replace as _struct_replace

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.common.scenario.context_overflow import is_context_overflow_response
from aiperf.credit.dispatch import ChildDispatchResult, TurnAdmission
from aiperf.credit.structs import TurnToSend
from aiperf.timing.conversation_source import SampledSession
from aiperf.timing.replay_dependencies import ReplayResumeBoundary
from aiperf.timing.trajectory_source import (
    ConversationState,
    Trajectory,
    TrajectorySnapshot,
    TrajectorySource,
    _as_timestamp_ms,
)

_WARMUP_MAX_TOKENS = 1
_IDLE_WATCHDOG_EPSILON_SECONDS = 1e-6

if TYPE_CHECKING:
    from aiperf.common.loop_scheduler import LoopScheduler
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.credit.issuer import CreditIssuer
    from aiperf.credit.structs import Credit
    from aiperf.timing.branch_orchestrator import BranchOrchestrator
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.session_tree import SessionTreeRegistry


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
        run: BenchmarkRun | None = None,
        branch_orchestrator: BranchOrchestrator | None = None,
        session_tree_registry: SessionTreeRegistry | None = None,
        progress: PhaseProgressTracker | None = None,
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
        self._progress = progress
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
        cache_warmup_duration = getattr(
            config, "agentic_cache_warmup_duration_sec", None
        )
        self._cache_warmup_duration: float | None = (
            float(cache_warmup_duration)
            if isinstance(cache_warmup_duration, int | float)
            else None
        )
        cache_warmup_requests_per_lane = getattr(
            config, "warmup_requests_per_lane", None
        )
        self._cache_warmup_requests_per_lane: int | None = (
            int(cache_warmup_requests_per_lane)
            if isinstance(cache_warmup_requests_per_lane, int)
            else None
        )
        self._cache_warmup_requests_by_lane: Counter[int] = Counter()
        self._cache_warmup_request_budget_reached = False
        self._baseline_warmup_admitted = 0
        self._quota_handoff_turns: dict[tuple[str, str, int], TurnToSend] = {}
        self._baseline_warmup_returns: dict[str, Credit] = {}
        self._baseline_correlations: set[str] = set()
        self._baseline_warmup_turns: set[tuple[str, int]] = set()
        self._accelerated_warmup_started = False
        self._handoff_credits: dict[str, Credit] = {}
        self._root_to_lane: dict[str, int] = {}
        # Accelerated warmup removes idle delays. Keep each tree's sampled t*
        # so handoff can restore every stream to one flattened dataset clock;
        # otherwise pending child turn-0 requests all look due at offset zero.
        self._replay_origin_ms_by_root: dict[str, float] = {}

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
            run.cfg.get_cache_bust_target() if run is not None else CacheBustTarget.NONE
        )
        self._benchmark_id: str = run.benchmark_id if run is not None else "unknown"
        # ``--burst-phase-starts`` (BasePhaseConfig.burst_phase_starts on the
        # profiling phase). Default False: WARMUP is globally t*-aligned and
        # PROFILING subtracts one global minimum so its earliest request starts
        # immediately while every other request keeps its relative offset.
        # When True, each lane subtracts its own minimum and the phase starts as
        # a synchronized burst. Governs ONLY the two phase-start dispatch
        # patterns; the rest of replay timing is faithful regardless. ``is
        # True`` guards MagicMock/None test configs -> default (spread).
        profiling_phase = None
        if run is not None:
            phases = run.cfg.get_profiling_phases()
            profiling_phase = phases[0] if phases else None
        self._burst_phase_starts: bool = (
            getattr(profiling_phase, "burst_phase_starts", False) is True
        )
        system_idle_cap_s = getattr(
            profiling_phase, "system_idle_gap_cap_seconds", None
        )
        self._system_idle_gap_cap_seconds: float | None = (
            float(system_idle_cap_s)
            if isinstance(system_idle_cap_s, int | float)
            else None
        )
        self._system_idle_jump_count = 0
        self._system_idle_seconds_skipped = 0.0
        self._system_idle_started_at: float | None = None
        self._system_idle_watchdog: asyncio.TimerHandle | None = None
        if self._system_idle_gap_cap_seconds is not None:
            self.scheduler.set_drain_observer(self.enforce_system_idle_cap)

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
    def _cache_warmup_enabled(self) -> bool:
        """Whether either accelerated cache-pressure warmup mode is active."""
        return (
            self._cache_warmup_duration is not None
            or self._cache_warmup_requests_per_lane is not None
        )

    @property
    def _has_tree_registry(self) -> bool:
        """True when per-tree session-slot accounting is engaged.

        PROFILING always engages it; an accelerated cache-pressure WARMUP also
        engages it because it opens trees and spawns descendants during WARMUP.
        """
        return self._session_tree_registry is not None and (
            self.config.phase == CreditPhase.PROFILING or self._cache_warmup_enabled
        )

    @property
    def wants_returns_after_sending_complete(self) -> bool:
        """Pressure warmup returns must be observed to build the handoff state."""
        return self.config.phase == CreditPhase.WARMUP and self._cache_warmup_enabled

    @property
    def allows_pending_branch_handoff_after_sending_complete(self) -> bool:
        """Pressure warmup preserves paused DAG branches for profiling handoff."""
        return self.wants_returns_after_sending_complete

    def _lane_root_corr(self, snapshot: TrajectorySnapshot) -> str | None:
        """Tree-root id shared by every stream of a snapshot lane.

        All states of one snapshot lane carry the same ``root_correlation_id``
        (the snapshot's synthetic parent_corr -- the root state's own id when a
        root is present, or the shared parent of the background subagents when
        rootless). Falls back to a state's x_correlation_id for snapshots built
        without the field (older fixtures)."""
        return (
            next(
                (
                    state.root_correlation_id
                    for state in snapshot.states
                    if state.root_correlation_id is not None
                ),
                None,
            )
            or next(
                (
                    state.x_correlation_id
                    for state in snapshot.states
                    if state.agent_depth == 0
                ),
                None,
            )
            or next(
                (
                    state.parent_correlation_id
                    for state in snapshot.states
                    if state.parent_correlation_id is not None
                ),
                None,
            )
            or next(
                (state.x_correlation_id for state in snapshot.states),
                None,
            )
        )

    def _seed_trajectory_replay_prefix(self, trajectory: Trajectory) -> None:
        """Seed the exact completed history before a resumed phase dispatches."""
        if trajectory.snapshot is None:
            root_correlation_id = trajectory.x_correlation_id
            boundaries = (
                ReplayResumeBoundary(
                    trajectory.conversation_id, trajectory.start_turn_index + 1
                ),
            )
        else:
            root_correlation_id = self._lane_root_corr(trajectory.snapshot)
            boundaries = trajectory.snapshot.replay_resume_boundaries
        if root_correlation_id is not None:
            self.credit_issuer.replay_gate.seed_completed_prefixes(
                root_correlation_id, boundaries
            )

    def _on_tree_drained(self, root_corr: str, phase: CreditPhase | int) -> None:
        """Registry drain callback: a session tree fully drained and freed its
        slot, so recycle its lane into a fresh root.

        Fired synchronously from the registry when the last of a tree's root +
        descendants completes. Maps the tree root to its lane and schedules the
        (async) recycle on the loop scheduler so it runs after the current
        return finishes and is cancelled cleanly at phase teardown. The slot was
        already released by the registry, so the recycled root's acquire keeps
        occupancy at exactly the configured concurrency.
        """
        self.credit_issuer.replay_gate.close_root(root_corr)
        if self.branch_orchestrator is not None:
            self.branch_orchestrator.close_replay_root(root_corr)
        lane = self._correlation_to_lane.pop(root_corr, None)
        self._session_marker.pop(root_corr, None)
        self._root_to_lane.pop(root_corr, None)
        self._replay_origin_ms_by_root.pop(root_corr, None)
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
        if (
            self.config.phase == CreditPhase.WARMUP
            and self._cache_warmup_requests_per_lane is not None
        ):
            self.credit_issuer.set_turn_admission(self._admit_cache_warmup_turn)
        if self.config.phase == CreditPhase.PROFILING:
            for trajectory in self.conversation_source.trajectories:
                self._seed_trajectory_replay_prefix(trajectory)
            self.credit_issuer.replay_gate.activate()
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

    def _cache_warmup_lane(self, turn_or_credit: TurnToSend | Credit) -> int:
        """Resolve a warmup turn or credit to its stable trajectory lane."""
        lane = self._root_to_lane.get(turn_or_credit.effective_root_correlation_id)
        if lane is None:
            lane = self._correlation_to_lane.get(turn_or_credit.x_correlation_id)
        if lane is None:
            raise RuntimeError(
                "Agentic cache warmup could not resolve a request to a "
                f"trajectory lane: correlation_id={turn_or_credit.x_correlation_id!r}, "
                "root_correlation_id="
                f"{turn_or_credit.effective_root_correlation_id!r}"
            )
        return lane

    def _admit_cache_warmup_turn(self, turn: TurnToSend) -> TurnAdmission:
        """Admit mandatory primers, then enforce the additional per-lane quota.

        Snapshot reconstruction can require multiple primers on one lane when
        a root and one or more subagents are all live at ``t*``. Those primers
        are the first warmup stage and do not consume the configured
        cache-pressure quota. The quota applies only to traffic replayed after
        all mandatory primers return.
        """
        assert self._cache_warmup_requests_per_lane is not None
        lane = self._cache_warmup_lane(turn)
        is_baseline = (
            turn.x_correlation_id,
            turn.turn_index,
        ) in self._baseline_warmup_turns
        if is_baseline:
            self._baseline_warmup_admitted += 1
            return TurnAdmission.ADMIT
        if (
            self._cache_warmup_requests_by_lane[lane]
            >= self._cache_warmup_requests_per_lane
        ):
            state_key = (
                turn.conversation_id,
                turn.x_correlation_id,
                turn.turn_index,
            )
            self._quota_handoff_turns[state_key] = turn
            return TurnAdmission.DEFER
        self._cache_warmup_requests_by_lane[lane] += 1
        if not self._cache_warmup_request_budget_reached and all(
            self._cache_warmup_requests_by_lane[lane_index]
            >= self._cache_warmup_requests_per_lane
            for lane_index in range(len(self.conversation_source.trajectories))
        ):
            self._cache_warmup_request_budget_reached = True
            self.credit_issuer.replay_gate.pause_releases()
            self.info(
                "WARMUP cache pressure request budget reached: "
                f"{self._cache_warmup_requests_per_lane} additional requests "
                "on each of "
                f"{len(self.conversation_source.trajectories)} lanes; "
                "draining requests"
            )
        return TurnAdmission.ADMIT

    async def execute_phase(self) -> None:
        """Dispatch initial credits for the phase."""
        if self.config.phase == CreditPhase.WARMUP:
            await self._execute_warmup()
        else:
            await self._execute_profiling()
        self.enforce_system_idle_cap()

    def enforce_system_idle_cap(self, in_flight_requests: int | None = None) -> None:
        """Bound true system-idle time without changing an individual trace."""
        if self._system_idle_gap_cap_seconds is None:
            return
        # Accelerated warmup owns a scheduler timer that marks its duration
        # cutoff. That control timer is not a pending request and must not be
        # pulled forward by the replay idle guard.
        if self._accelerated_warmup_started:
            return
        # Credit returns start a new idle interval. Scheduler-drain callbacks
        # recheck the same interval after a scheduled turn is barrier-retained.
        in_flight_requests, follows_request_return = (
            self._resolve_in_flight_request_count(in_flight_requests)
        )
        if in_flight_requests is None:
            return
        if in_flight_requests > 0:
            self._system_idle_started_at = None
            self._cancel_system_idle_watchdog()
            return
        now = time.monotonic()
        if follows_request_return or self._system_idle_started_at is None:
            self._system_idle_started_at = now

        idle_elapsed = max(0.0, now - self._system_idle_started_at)
        remaining_idle_budget = max(
            0.0, self._system_idle_gap_cap_seconds - idle_elapsed
        )
        if remaining_idle_budget <= _IDLE_WATCHDOG_EPSILON_SECONDS:
            remaining_idle_budget = 0.0
        if remaining_idle_budget > 0:
            self._arm_system_idle_watchdog(remaining_idle_budget)

        # A just-fired replay callback may still be transitioning into a real
        # request. Give it the remainder of the idle budget to do so, but do
        # not let a control-plane coroutine extend true request-idle time past
        # the configured cap. The watchdog re-enters here at the deadline and
        # progress.in_flight remains the authoritative wire-activity signal.
        if self.scheduler.running_count > 0 and remaining_idle_budget > 0:
            return

        shifted = self.scheduler.cap_pending_delay(remaining_idle_budget)
        if shifted <= 0:
            return
        self._system_idle_jump_count += 1
        self._system_idle_seconds_skipped += shifted
        self.debug(
            lambda: (
                "Global system-idle cap advanced all pending replay timers by "
                f"{shifted:.3f}s"
            )
        )

    def _resolve_in_flight_request_count(
        self, supplied_count: int | None
    ) -> tuple[int | None, bool]:
        """Resolve wire activity and whether the observation is a return."""
        if supplied_count is not None:
            return supplied_count, True
        if self._progress is None:
            return None, False
        return self._progress.in_flight, False

    def _arm_system_idle_watchdog(self, delay_seconds: float) -> None:
        """Guarantee an idle-cap recheck even when no scheduler task drains.

        Request-return and scheduler-drain callbacks remain the fast path. The
        independent control timer closes the gap where one of those callbacks
        observes a transient running scheduler coroutine and no later state
        transition re-enters the cap logic. It deliberately lives outside the
        replay scheduler so ``cap_pending_delay`` cannot advance its own guard.
        """
        if self._system_idle_watchdog is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Some direct unit/API callers exercise the synchronous fast path
            # without an event loop. They still get the immediate schedule
            # shift below; only the independent future recheck is unavailable.
            return
        self._system_idle_watchdog = loop.call_later(
            max(0.0, delay_seconds),
            self._run_system_idle_watchdog,
        )

    def _run_system_idle_watchdog(self) -> None:
        """Clear the one-shot handle and re-evaluate actual request idleness."""
        self._system_idle_watchdog = None
        self.enforce_system_idle_cap()

    def _cancel_system_idle_watchdog(self) -> None:
        """Cancel the independent idle guard when wire activity resumes."""
        if self._system_idle_watchdog is None:
            return
        self._system_idle_watchdog.cancel()
        self._system_idle_watchdog = None

    def _capped_warmup_lead_ms(self, lead_ms: float) -> float:
        """Clamp a WARMUP lead to the global system-idle guard.

        Warmup is a cache-priming pass, not faithful replay: warming a stream
        that last fired hours before t* that far ahead would make the warmup
        phase itself take hours. The per-trajectory trace cap is deliberately
        absent here: it is a profiling runtime watchdog, not a phase-start
        schedule transform.
        """
        if self._system_idle_gap_cap_seconds is None:
            return lead_ms
        return min(
            lead_ms,
            self._system_idle_gap_cap_seconds * MILLIS_PER_SECOND,
        )

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
                self._baseline_correlations.add(turn.x_correlation_id)
                self._baseline_warmup_turns.add(
                    (turn.x_correlation_id, turn.turn_index)
                )
                self._root_to_lane[turn.effective_root_correlation_id] = lane
                continue

            t_star_ms = trajectory.snapshot.t_star_ms
            root_correlation_id = self._lane_root_corr(trajectory.snapshot)
            if root_correlation_id is not None:
                self._replay_origin_ms_by_root[root_correlation_id] = t_star_ms
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
                self._baseline_correlations.add(turn.x_correlation_id)
                self._baseline_warmup_turns.add(
                    (turn.x_correlation_id, turn.turn_index)
                )
                self._root_to_lane[turn.effective_root_correlation_id] = lane

        if self._cache_warmup_requests_per_lane is not None:
            self.info(
                "WARMUP count budget: "
                f"{self._cache_warmup_requests_per_lane} additional requests "
                f"per lane after {len(prepared)} mandatory snapshot primers"
            )

        # Nothing to warm: every lane's first request is at/after t* (no turn
        # precedes t*), so no warmup credit will dispatch. The count path that
        # normally drives completion is triggered by credit dispatch/return, so
        # with zero credits it would never fire and the warmup phase would hang
        # waiting on a barrier sized to concurrency. Finalize immediately.
        if not prepared:
            if self._cache_warmup_enabled:
                # No baseline priming to wait for; jump straight to the
                # accelerated cache-pressure substage.
                await self._start_accelerated_warmup()
                return
            self.info("WARMUP execute: no requests precede t*; nothing to warm")
            self.credit_issuer.signal_sending_complete()
            return

        # Pass 2: dispatch.
        if not spread:
            self.info(
                "WARMUP burst: dispatching all warmup requests at once (spread 0.0s)"
            )
            for turn, _ in prepared:
                await self.credit_issuer.issue_credit(turn)
            await self._finish_initial_warmup_dispatch()
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

    async def _finish_initial_warmup_dispatch(self) -> None:
        """Mark sending complete for burst warmup with no cache-pressure stage.

        When either cache-pressure mode is set, the accelerated substage is
        driven by baseline credit returns (``_handle_warmup_return``), so there
        is nothing to finalize here.
        """
        if not self._cache_warmup_enabled and not self.lifecycle.is_sending_complete:
            self.lifecycle.mark_sending_complete()

    async def _start_accelerated_warmup(self) -> None:
        """Continue the sampled trajectories under compressed warmup traffic."""
        if self._accelerated_warmup_started:
            return
        assert self._cache_warmup_enabled
        # Baseline primers return before accelerated replay is active. A parent
        # already blocked on a child join may never return again in WARMUP, so
        # seed its last nonterminal credit into the profiling handoff now.
        for credit in self._baseline_warmup_returns.values():
            if credit.is_final_turn:
                continue
            self._handoff_credits[credit.x_correlation_id] = credit
        self._accelerated_warmup_started = True
        self.credit_issuer.set_max_tokens_override(_WARMUP_MAX_TOKENS)
        for trajectory in self.conversation_source.trajectories:
            self._seed_trajectory_replay_prefix(trajectory)
        self.credit_issuer.replay_gate.activate()
        if self._cache_warmup_duration is not None:
            limit = f"for {self._cache_warmup_duration:.1f}s"
        else:
            limit = (
                f"until each lane reaches "
                f"{self._cache_warmup_requests_per_lane} additional requests"
            )
        self.info(
            "WARMUP cache pressure: replaying live trajectories "
            f"{limit} with zero idle delay and max_tokens={_WARMUP_MAX_TOKENS}"
        )
        if self.branch_orchestrator is not None:
            self.branch_orchestrator.start_accelerated_warmup()
        if self._cache_warmup_duration is not None:
            self.scheduler.schedule_later(
                self._cache_warmup_duration,
                self._finish_accelerated_warmup(),
            )
        results = await asyncio.gather(
            *(
                self._dispatch_accelerated_trajectory(trajectory, lane)
                for lane, trajectory in enumerate(self.conversation_source.trajectories)
            ),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise errors[0]

    async def _finish_accelerated_warmup(self) -> None:
        """Stop new pressure traffic and let all issued requests drain."""
        self.info("WARMUP cache pressure duration reached; draining requests")
        self.credit_issuer.replay_gate.pause_releases()
        self.credit_issuer.mark_sending_complete()

    async def _dispatch_accelerated_trajectory(
        self, trajectory: Trajectory, lane: int
    ) -> None:
        """Dispatch one lane from its post-snapshot state without idle delays."""
        if trajectory.snapshot is None:
            session = self.conversation_source.session_for(trajectory)
            resume_index = trajectory.start_turn_index + 1
            turn = self._build_turn_for_session(session, resume_index)
            turn = _struct_replace(turn, is_session_start=False)
            self._correlation_to_lane[turn.x_correlation_id] = lane
            self._root_to_lane[turn.effective_root_correlation_id] = lane
            await self.credit_issuer.issue_credit(turn)
            return

        snapshot = trajectory.snapshot
        root_correlation_id = self._lane_root_corr(snapshot)
        if root_correlation_id is not None:
            self._replay_origin_ms_by_root[root_correlation_id] = snapshot.t_star_ms
        for state in snapshot.states:
            self._correlation_to_lane[state.x_correlation_id] = lane
            self._root_to_lane[state.root_correlation_id or state.x_correlation_id] = (
                lane
            )
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

        dispatchable = [
            state for state in snapshot.states if not state.waiting_on_children
        ]
        has_baseline_root = any(
            state.agent_depth == 0
            and state.x_correlation_id in self._baseline_correlations
            for state in snapshot.states
        )
        has_dispatchable_root = any(state.agent_depth == 0 for state in dispatchable)
        if dispatchable and not has_baseline_root and not has_dispatchable_root:
            root_correlation_id = self._lane_root_corr(snapshot)
            if root_correlation_id is not None:
                self._correlation_to_lane[root_correlation_id] = lane
            await self.credit_issuer.acquire_lane_credit(
                root_correlation_id,
                root_pending=any(state.agent_depth == 0 for state in snapshot.states),
            )

        for state in dispatchable:
            session = self.conversation_source.session_for_state(state)
            turn = self._build_turn_for_session(session, state.next_turn_index)
            turn = _struct_replace(
                turn,
                is_session_start=(
                    state.agent_depth == 0
                    and state.x_correlation_id not in self._baseline_correlations
                ),
            )
            await self.credit_issuer.issue_credit(turn)

    def observe_credit_return(self, credit: Credit) -> None:
        """Track the next live turn for the warmup-to-profile handoff."""
        self.credit_issuer.replay_gate.complete(credit)
        if not self._accelerated_warmup_started:
            return
        root_correlation_id = credit.effective_root_correlation_id
        lane = self._root_to_lane.get(root_correlation_id)
        if lane is None:
            lane = self._correlation_to_lane.get(credit.x_correlation_id)
        if lane is not None:
            self._root_to_lane[root_correlation_id] = lane
            self._correlation_to_lane[credit.x_correlation_id] = lane
        if credit.is_final_turn:
            self._handoff_credits.pop(credit.x_correlation_id, None)
        else:
            self._handoff_credits[credit.x_correlation_id] = credit

    async def _handle_accelerated_warmup_return(self, credit: Credit) -> None:
        """Issue the next compressed turn or recycle a completed tree."""
        if credit.is_final_turn:
            if credit.agent_depth == 0 and not self._has_tree_registry:
                await self._spawn_from_recycle_or_id(
                    credit.conversation_id,
                    finished_correlation_id=credit.x_correlation_id,
                )
            return

        next_meta = self.conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit, next_meta)
        turn = _struct_replace(turn, max_tokens_override=_WARMUP_MAX_TOKENS)
        if turn.agent_depth > 0:
            await self._issue_child_continuation_or_drain(turn)
        else:
            await self.credit_issuer.issue_credit(turn)

    async def _issue_child_continuation_or_drain(self, turn: TurnToSend) -> None:
        """Dispatch a DAG child continuation, draining terminal refusals.

        On a terminal refusal, notify the orchestrator so the parent's join
        drains deterministically instead of deadlocking on a child whose
        remaining turns will never be issued. A deferred accelerated-warmup
        turn is different: the remaining child and its active join are
        persisted for profiling, so marking the child stopped here would
        release the parent prematurely.
        """
        result = ChildDispatchResult.normalize(
            await self.credit_issuer.dispatch_child_turn(turn)
        )
        if (
            result is ChildDispatchResult.REJECTED
            and self.branch_orchestrator is not None
            and not self.allows_pending_branch_handoff_after_sending_complete
        ):
            await self.branch_orchestrator.on_child_stopped(turn.x_correlation_id)

    async def finalize_phase(self) -> None:
        """Persist the drained accelerated-warmup DAG for profiling."""
        if self._system_idle_gap_cap_seconds is not None:
            self._cancel_system_idle_watchdog()
            self.scheduler.set_drain_observer(None)
            self.info(
                "Global system-idle cap summary: "
                f"limit={self._system_idle_gap_cap_seconds:g}s, "
                f"jumps={self._system_idle_jump_count}, "
                f"skipped={self._system_idle_seconds_skipped:.3f}s"
            )
        if not self._accelerated_warmup_started:
            return
        states_by_lane = self._build_handoff_states()
        boundaries_by_lane = {
            lane: self._build_handoff_replay_boundaries(states)
            for lane, states in states_by_lane.items()
        }
        self.conversation_source.trajectories = self._build_handoff_trajectories(
            states_by_lane, boundaries_by_lane
        )
        self.info(
            "WARMUP cache pressure handoff: persisted "
            f"{sum(len(states) for states in states_by_lane.values())} live streams"
        )

    def _build_handoff_states(self) -> dict[int, list[ConversationState]]:
        """Convert drained credits and join annotations into lane states."""
        blocked, child_annotations = self._handoff_annotations()
        states_by_lane: dict[int, list[ConversationState]] = {
            lane: [] for lane in range(len(self.conversation_source.trajectories))
        }
        seen_states = self._add_returned_handoff_states(
            states_by_lane,
            blocked=blocked,
            child_annotations=child_annotations,
        )
        self._add_pending_handoff_states(
            states_by_lane, seen_states, child_annotations=child_annotations
        )
        return states_by_lane

    def _handoff_annotations(
        self,
    ) -> tuple[dict[str, int], dict[str, list[tuple[str | None, int | None]]]]:
        if self.branch_orchestrator is None:
            return {}, {}
        return self.branch_orchestrator.snapshot_annotations()

    @staticmethod
    def _child_join_from_annotations(
        child_annotations: dict[str, list[tuple[str | None, int | None]]],
        correlation_id: str,
    ) -> tuple[str | None, int | None, tuple[tuple[str, int], ...]]:
        """Unpack primary join fields plus full multi-gate memberships."""
        memberships = [
            (branch_id, gated_idx)
            for branch_id, gated_idx in child_annotations.get(correlation_id, [])
            if branch_id is not None and gated_idx is not None
        ]
        if not memberships:
            return None, None, ()
        branch_id, join_target = memberships[0]
        return branch_id, join_target, tuple(memberships)

    def _add_returned_handoff_states(
        self,
        states_by_lane: dict[int, list[ConversationState]],
        *,
        blocked: dict[str, int],
        child_annotations: dict[str, list[tuple[str | None, int | None]]],
    ) -> set[tuple[str, str, int]]:
        seen_states: set[tuple[str, str, int]] = set()
        for credit in self._handoff_credits.values():
            handoff = self._returned_credit_handoff_state(
                credit,
                blocked=blocked,
                child_annotations=child_annotations,
            )
            if handoff is None:
                continue
            lane, state = handoff
            seen_states.add(
                (state.conversation_id, state.x_correlation_id, state.next_turn_index)
            )
            states_by_lane[lane].append(state)
        return seen_states

    def _returned_credit_handoff_state(
        self,
        credit: Credit,
        *,
        blocked: dict[str, int],
        child_annotations: dict[str, list[tuple[str | None, int | None]]],
    ) -> tuple[int, ConversationState] | None:
        lane = self._root_to_lane.get(credit.effective_root_correlation_id)
        if lane is None or credit.turn_index + 1 >= credit.num_turns:
            return None
        branch_id, join_target, memberships = self._child_join_from_annotations(
            child_annotations, credit.x_correlation_id
        )
        return lane, ConversationState(
            conversation_id=credit.conversation_id,
            x_correlation_id=credit.x_correlation_id,
            next_turn_index=credit.turn_index + 1,
            next_dispatch_offset_ms=self._handoff_replay_offset_ms(
                credit.effective_root_correlation_id,
                credit.conversation_id,
                credit.turn_index + 1,
            ),
            agent_depth=credit.agent_depth,
            parent_correlation_id=credit.parent_correlation_id,
            root_correlation_id=credit.root_correlation_id,
            waiting_on_children=credit.x_correlation_id in blocked,
            join_target_turn_index=blocked.get(credit.x_correlation_id, join_target),
            branch_id=branch_id,
            join_gate_memberships=memberships,
            branch_mode=credit.branch_mode,
        )

    def _add_pending_handoff_states(
        self,
        states_by_lane: dict[int, list[ConversationState]],
        seen_states: set[tuple[str, str, int]],
        *,
        child_annotations: dict[str, list[tuple[str | None, int | None]]],
    ) -> None:
        for root_correlation_id, turns in self._pending_handoff_turns_by_root().items():
            for turn in turns:
                handoff = self._pending_turn_handoff_state(
                    root_correlation_id,
                    turn,
                    seen_states,
                    child_annotations=child_annotations,
                )
                if handoff is None:
                    continue
                lane, state_key, state = handoff
                seen_states.add(state_key)
                states_by_lane[lane].append(state)

    def _pending_handoff_turns_by_root(self) -> dict[str, tuple[TurnToSend, ...]]:
        pending_by_root_getter = getattr(
            self.credit_issuer.replay_gate, "pending_turns_by_root", None
        )
        if callable(pending_by_root_getter):
            pending_by_root = dict(pending_by_root_getter())
        else:
            pending_by_root = {}
        for turn in self._quota_handoff_turns.values():
            root_correlation_id = turn.effective_root_correlation_id
            pending_by_root.setdefault(root_correlation_id, ())
            pending_by_root[root_correlation_id] += (turn,)
        pending_turns_getter = getattr(
            self.credit_issuer.replay_gate, "pending_turns", None
        )
        if callable(pending_turns_getter):
            for root_correlation_id in self._root_to_lane:
                pending_by_root.setdefault(
                    root_correlation_id,
                    tuple(pending_turns_getter(root_correlation_id)),
                )
        return pending_by_root

    def _pending_turn_handoff_state(
        self,
        root_correlation_id: str,
        turn: TurnToSend,
        seen_states: set[tuple[str, str, int]],
        *,
        child_annotations: dict[str, list[tuple[str | None, int | None]]],
    ) -> tuple[int, tuple[str, str, int], ConversationState] | None:
        lane = self._handoff_lane_for_turn(root_correlation_id, turn)
        state_key = (turn.conversation_id, turn.x_correlation_id, turn.turn_index)
        if (
            lane is None
            or state_key in seen_states
            or turn.turn_index >= turn.num_turns
        ):
            return None
        self._root_to_lane[turn.effective_root_correlation_id] = lane
        branch_id, join_target, memberships = self._child_join_from_annotations(
            child_annotations, turn.x_correlation_id
        )
        state = ConversationState(
            conversation_id=turn.conversation_id,
            x_correlation_id=turn.x_correlation_id,
            next_turn_index=turn.turn_index,
            next_dispatch_offset_ms=self._handoff_replay_offset_ms(
                turn.effective_root_correlation_id,
                turn.conversation_id,
                turn.turn_index,
            ),
            agent_depth=turn.agent_depth,
            parent_correlation_id=turn.parent_correlation_id,
            root_correlation_id=turn.root_correlation_id,
            waiting_on_children=False,
            join_target_turn_index=join_target,
            branch_id=branch_id,
            join_gate_memberships=memberships,
            branch_mode=turn.branch_mode,
        )
        return lane, state_key, state

    def _handoff_lane_for_turn(
        self, root_correlation_id: str, turn: TurnToSend
    ) -> int | None:
        lane = self._root_to_lane.get(root_correlation_id)
        if lane is not None:
            return lane
        lane = self._root_to_lane.get(turn.effective_root_correlation_id)
        if lane is not None:
            return lane
        if turn.parent_correlation_id is not None:
            lane = self._correlation_to_lane.get(turn.parent_correlation_id)
            if lane is not None:
                return lane
        return self._correlation_to_lane.get(turn.x_correlation_id)

    def _handoff_delay_ms(self, conversation_id: str, next_turn_index: int) -> float:
        """Recorded end-to-start delay carried across the warmup barrier."""
        try:
            metadata = self.conversation_source.get_metadata(conversation_id)
        except KeyError:
            return 0.0
        if next_turn_index < 0 or next_turn_index >= len(metadata.turns):
            return 0.0

        next_meta = metadata.turns[next_turn_index]
        delay_ms = _as_timestamp_ms(getattr(next_meta, "delay_ms", None))
        if delay_ms is not None:
            return max(0.0, delay_ms)
        if next_turn_index == 0:
            return 0.0

        previous_meta = metadata.turns[next_turn_index - 1]
        previous_ts_ms = _as_timestamp_ms(getattr(previous_meta, "timestamp_ms", None))
        next_ts_ms = _as_timestamp_ms(getattr(next_meta, "timestamp_ms", None))
        if previous_ts_ms is None or next_ts_ms is None:
            return 0.0

        previous_api_ms = _as_timestamp_ms(getattr(previous_meta, "api_time_ms", None))
        previous_duration_ms = max(0.0, previous_api_ms or 0.0)
        return max(0.0, next_ts_ms - previous_ts_ms - previous_duration_ms)

    def _handoff_replay_offset_ms(
        self,
        root_correlation_id: str,
        conversation_id: str,
        next_turn_index: int,
    ) -> float:
        """Place a surviving stream back on its tree's shared replay clock.

        Accelerated warmup compresses runtime waits, so elapsed wall time is
        not a usable profiling deadline.  Timestamped AgentX datasets already
        provide a common clock for every stream in a tree.  Subtracting the
        tree's sampled replay origin preserves the original flattened order
        and spacing across roots, children, sidecars, and gated joins.  The
        profiling phase later subtracts one phase-wide minimum so the earliest
        request starts immediately without turning the rest into a burst.

        Timestamp-less datasets have no shared clock, so retain the legacy
        per-stream end-to-start delay as the best available fallback.
        """
        replay_origin_ms = self._replay_origin_ms_by_root.get(root_correlation_id)
        if replay_origin_ms is not None:
            try:
                metadata = self.conversation_source.get_metadata(conversation_id)
            except KeyError:
                metadata = None
            if metadata is not None and 0 <= next_turn_index < len(metadata.turns):
                next_timestamp_ms = _as_timestamp_ms(
                    getattr(metadata.turns[next_turn_index], "timestamp_ms", None)
                )
                if next_timestamp_ms is not None:
                    return max(0.0, next_timestamp_ms - replay_origin_ms)
        return self._handoff_delay_ms(conversation_id, next_turn_index)

    def _build_handoff_replay_boundaries(
        self, states: list[ConversationState]
    ) -> tuple[ReplayResumeBoundary, ...]:
        """Merge live stream positions with terminal history from warmup."""
        next_turn_by_conversation = {
            state.conversation_id: state.next_turn_index
            for state in states
            if state.next_turn_index > 0
        }
        if states:
            root_correlation_id = self._lane_root_corr(
                TrajectorySnapshot(t_star_ms=0.0, states=tuple(states))
            )
            if root_correlation_id is not None:
                for boundary in self.credit_issuer.replay_gate.completed_prefixes(
                    root_correlation_id
                ):
                    next_turn_by_conversation[boundary.conversation_id] = max(
                        next_turn_by_conversation.get(boundary.conversation_id, 0),
                        boundary.next_turn_index,
                    )
        return tuple(
            ReplayResumeBoundary(conversation_id, next_turn_index)
            for conversation_id, next_turn_index in sorted(
                next_turn_by_conversation.items()
            )
        )

    def _build_handoff_trajectories(
        self,
        states_by_lane: dict[int, list[ConversationState]],
        boundaries_by_lane: dict[int, tuple[ReplayResumeBoundary, ...]],
    ) -> list[Trajectory]:
        """Build the shared trajectory list consumed by profiling."""
        rebuilt: list[Trajectory] = []
        for lane, previous in enumerate(self.conversation_source.trajectories):
            states = states_by_lane[lane]
            boundaries = boundaries_by_lane[lane]
            if not states:
                trace_id = self.conversation_source.next_recycle_conversation_id()
                if trace_id is None:
                    rebuilt.append(previous)
                    continue
                correlation_id = str(uuid.uuid4())
                states = [
                    ConversationState(
                        conversation_id=trace_id,
                        x_correlation_id=correlation_id,
                        next_turn_index=0,
                    )
                ]
                boundaries = ()
            root_state = next(
                (state for state in states if state.agent_depth == 0), None
            )
            root_trace_id = (
                root_state.conversation_id
                if root_state is not None
                else previous.conversation_id
            )
            rebuilt.append(
                Trajectory(
                    conversation_id=root_trace_id,
                    start_turn_index=(
                        root_state.next_turn_index if root_state is not None else 0
                    ),
                    snapshot=TrajectorySnapshot(
                        t_star_ms=0.0,
                        states=tuple(
                            sorted(
                                states,
                                key=lambda state: (
                                    state.agent_depth,
                                    state.x_correlation_id,
                                ),
                            )
                        ),
                        replay_resume_boundaries=boundaries,
                    ),
                    x_correlation_id=(
                        root_state.x_correlation_id
                        if root_state is not None
                        else previous.x_correlation_id
                    ),
                )
            )
        return rebuilt

    async def _execute_profiling(self) -> None:
        """Resume each trajectory at ``k_i + 1`` to seed the steady state.

        All trajectories are dispatched concurrently so the full concurrency
        target is reached as fast as slot limits allow, rather than
        serializing over N credit round-trips. Subsequent turns and
        recycle-pool sessions are dispatched from handle_credit_return.
        """
        phase_t0_offset_ms = self._profiling_phase_t0_offset_ms()
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
                self._dispatch_one_profiling_trajectory(
                    trajectory, lane, phase_t0_offset_ms
                )
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
        (t0=0 spread, or t0=lane-min burst). This returns max-minus-min of those per-trajectory
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
            lane_offsets = [s.next_dispatch_offset_ms for s in dispatchable]
            lane_t0 = min(lane_offsets) if self._burst_phase_starts else 0.0
            first_offsets.append(min(lane_offsets) - lane_t0)
        if not first_offsets:
            return 0.0
        return (max(first_offsets) - min(first_offsets)) / MILLIS_PER_SECOND

    def _profiling_phase_t0_offset_ms(self) -> float:
        """Global spread anchor that starts the earliest request at phase zero."""
        if self._burst_phase_starts:
            return 0.0
        offsets: list[float] = []
        for trajectory in self.conversation_source.trajectories:
            if trajectory.snapshot is None:
                return 0.0
            offsets.extend(
                state.next_dispatch_offset_ms
                for state in trajectory.snapshot.states
                if not state.waiting_on_children
            )
        return min(offsets, default=0.0)

    async def _dispatch_one_profiling_trajectory(
        self, trajectory: Trajectory, lane: int, phase_t0_offset_ms: float
    ) -> None:
        """Dispatch one lane's initial PROFILING credit (run under gather)."""
        if trajectory.snapshot is not None:
            await self._dispatch_snapshot_for_profiling(
                trajectory, lane, phase_t0_offset_ms
            )
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
            await self._handle_warmup_return(credit)
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

    async def _handle_warmup_return(self, credit: Credit) -> None:
        """Advance baseline warmup into the optional cache-pressure stage."""
        if not self._cache_warmup_enabled:
            return
        if self._accelerated_warmup_started:
            await self._handle_accelerated_warmup_return(credit)
            return
        self._baseline_warmup_returns[credit.x_correlation_id] = credit
        if (
            len(self._baseline_warmup_returns)
            >= self.conversation_source.warmup_credit_count
        ):
            await self._start_accelerated_warmup()

    async def _dispatch_next_turn(self, credit: Credit) -> None:
        """Issue the next turn of an in-progress session, honoring delay_ms.

        DAG child continuations (``agent_depth > 0``) go through the single
        child-issuance chokepoint (``_issue_child_continuation_or_drain``) so a
        terminal refusal is routed to ``on_child_stopped`` (drain the parent
        join) instead of being silently swallowed by the discarded
        ``issue_credit`` return. A warmup cutoff is preserved for profiling
        handoff instead. This applies equally to delayed continuations, whose
        refusal may happen after the callback handler decided the child could
        proceed. Root continuations keep ``issue_credit``.
        """
        next_meta = self.conversation_source.get_next_turn_metadata(credit)
        turn = TurnToSend.from_previous_credit(credit, next_meta)

        coro = (
            self._issue_child_continuation_or_drain(turn)
            if turn.agent_depth > 0
            else self.credit_issuer.issue_credit(turn)
        )
        if next_meta.delay_ms is not None and next_meta.delay_ms > 0:
            self.scheduler.schedule_later(
                next_meta.delay_ms / MILLIS_PER_SECOND,
                coro,
                group_id=credit.effective_root_correlation_id,
            )
        else:
            await coro

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
        self._root_to_lane.pop(finished_correlation_id, None)
        self._replay_origin_ms_by_root.pop(finished_correlation_id, None)
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
        self._root_to_lane[session.effective_root_correlation_id] = lane
        if self.config.phase == CreditPhase.WARMUP:
            first_timestamp_ms = _as_timestamp_ms(
                getattr(session.metadata.turns[0], "timestamp_ms", None)
            )
            if first_timestamp_ms is not None:
                self._replay_origin_ms_by_root[
                    session.effective_root_correlation_id
                ] = first_timestamp_ms
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
        self, trajectory: Trajectory, lane: int, phase_t0_offset_ms: float
    ) -> None:
        """Resume one trajectory's streams for PROFILING.

        Each stream profiles from turn ``next_turn_index`` (the first turn at
        or after t*; its predecessor, if any, was primed during WARMUP).

        Dispatch anchoring depends on ``--burst-phase-starts``. By default one
        phase-wide minimum is subtracted from every stream: the earliest
        request starts at profiling-time 0 and all cross-trajectory spacing is
        preserved. With ``--burst-phase-starts`` each trajectory subtracts its
        own minimum, so every lane starts at profiling-time 0. Relative timing
        within each trajectory is identical either way.

        Gated parents (``waiting_on_children``) are not dispatched here; their
        join is seeded with the orchestrator and their gated turn fires at the
        later of its recorded replay deadline and the blocking-child completion
        frontier. No stream completes during WARMUP (warmup only ever sends a
        non-terminal turn), so there is no warmup-continuation or terminal-root
        recycle step.
        """
        snapshot = self._get_snapshot(trajectory)
        for state in snapshot.states:
            self._correlation_to_lane[state.x_correlation_id] = lane
            self._mint_marker_for_session(
                state.root_correlation_id or state.x_correlation_id,
                state.conversation_id,
                lane,
            )

        dispatchable = [s for s in snapshot.states if not s.waiting_on_children]
        # Compute one replay timeline for dispatchable streams and gated
        # parents alike. A parent's join deadline uses the same optional burst
        # anchor as every other request in its trajectory; only its child gate
        # is additional.
        offset_by_corr = {
            s.x_correlation_id: s.next_dispatch_offset_ms for s in snapshot.states
        }
        if self._burst_phase_starts and dispatchable:
            t0_offset_ms = min(
                offset_by_corr[state.x_correlation_id] for state in dispatchable
            )
        else:
            t0_offset_ms = phase_t0_offset_ms

        if self.branch_orchestrator is not None:
            self.branch_orchestrator.seed_snapshot(
                snapshot.states,
                cache_bust_markers=self._session_marker,
                join_release_delays_ms={
                    state.x_correlation_id: max(
                        0.0,
                        offset_by_corr[state.x_correlation_id] - t0_offset_ms,
                    )
                    for state in snapshot.states
                    if state.waiting_on_children
                },
            )

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
            # A gated parent's turn 0 was before t*, so it never dispatches a
            # session-start root credit -- yet its join turn reaches a terminal
            # turn that bumps completed_sessions. Pass its remaining turn count
            # (num_turns - gated_turn_index) so acquire_lane_credit counts it in
            # sent_sessions / total_session_turns and in_flight_sessions stays
            # non-negative. Rootless lanes (no root state) reach no terminal turn.
            gated_session_turns = 0
            if has_root_state:
                gated_state = next(
                    (
                        s
                        for s in snapshot.states
                        if s.conversation_id == trajectory.conversation_id
                        and s.waiting_on_children
                    ),
                    None,
                )
                if gated_state is not None:
                    gated_session = self.conversation_source.session_for_state(
                        gated_state
                    )
                    gated_session_turns = max(
                        0,
                        len(gated_session.metadata.turns) - gated_state.next_turn_index,
                    )
            await self.credit_issuer.acquire_lane_credit(
                lane_root_corr,
                root_pending=has_root_state,
                session_turns=gated_session_turns,
            )
            if not self._has_tree_registry and not has_root_state:
                self._rootless_lane_outstanding[lane] = len(dispatchable)
        # Spread (default): every lane shares one phase-wide T0, preserving
        # cross-trajectory offsets. Burst: each lane uses its own minimum.
        for state in dispatchable:
            session = self.conversation_source.session_for_state(state)
            turn = self._build_turn_for_session(session, state.next_turn_index)
            if state.agent_depth == 0:
                turn = _struct_replace(turn, is_session_start=True)
            delay_s = (
                offset_by_corr[state.x_correlation_id] - t0_offset_ms
            ) / MILLIS_PER_SECOND
            if delay_s > 0:
                self.scheduler.schedule_later(
                    delay_s,
                    self.credit_issuer.issue_credit(turn),
                    group_id=turn.effective_root_correlation_id,
                )
            else:
                await self.credit_issuer.issue_credit(turn)

        root_correlation_id = self._lane_root_corr(snapshot)
        if root_correlation_id is not None:
            self.credit_issuer.replay_gate.observe_idle_root(root_correlation_id)

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
