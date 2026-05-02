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
from typing import TYPE_CHECKING

from msgspec.structs import replace as _struct_replace

from aiperf.common.constants import MILLIS_PER_SECOND
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.scenario.base import TrajectoryWarmupFailedError
from aiperf.credit.structs import TurnToSend
from aiperf.timing.conversation_source import SampledSession
from aiperf.timing.strategies.cache_bust import build_cache_bust_marker
from aiperf.timing.trajectory_source import TrajectorySource

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

        self._recycle_queue: asyncio.Queue[str] | None = None
        self._in_flight_recycled: set[str] = set()
        self._failed_warmup_traces: list[str] = []
        # Track which x_correlation_ids correspond to trajectories in WARMUP
        # so that terminal failures can be attributed to a trace_id.
        self._warmup_correlation_to_trace: dict[str, str] = {}

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

    async def setup_phase(self) -> None:
        """Phase-specific async setup.

        WARMUP: nothing - trajectories already built by TrajectorySource at
        TimingManager construction time.

        PROFILING: build the FIFO recycle queue with all loader trace_ids minus
        the trajectories.
        """
        if self.config.phase == CreditPhase.PROFILING:
            if not self.conversation_source.trajectories:
                raise RuntimeError(
                    "AgenticReplayStrategy PROFILING setup: trajectories empty. "
                    "WARMUP must complete with at least one trajectory before "
                    "PROFILING can start. Check loader output and warmup failures."
                )
            self._recycle_queue = asyncio.Queue()
            trajectory_ids = {
                trajectory.conversation_id
                for trajectory in self.conversation_source.trajectories
            }
            for conv in self.conversation_source.dataset_metadata.conversations:
                trace_id = conv.conversation_id
                if trace_id not in trajectory_ids:
                    self._recycle_queue.put_nowait(trace_id)
            self.info(
                f"PROFILING setup: trajectories={len(trajectory_ids)} traces, "
                f"recycle_queue={self._recycle_queue.qsize()} traces"
            )

    async def execute_phase(self) -> None:
        """Dispatch initial credits for the phase."""
        if self.config.phase == CreditPhase.WARMUP:
            await self._execute_warmup()
        else:
            await self._execute_profiling()

    async def _execute_warmup(self) -> None:
        """Dispatch one credit per trajectory at turn ``k_i``."""
        self.info(
            f"WARMUP execute: dispatching {len(self.conversation_source.trajectories)} "
            f"trajectory credits"
        )
        for lane, trajectory in enumerate(self.conversation_source.trajectories):
            session = self.conversation_source.session_for(trajectory)
            self._correlation_to_lane[session.x_correlation_id] = lane
            self._mint_marker_for_session(
                session.x_correlation_id, trajectory.conversation_id, lane
            )
            turn = self._build_turn_for_session(session, trajectory.start_turn_index)
            self._warmup_correlation_to_trace[turn.x_correlation_id] = (
                trajectory.conversation_id
            )
            await self.credit_issuer.issue_credit(turn)
        # Trajectory dispatch complete; signal the phase that no more credits
        # will be issued. SendingCompleteStopCondition watches this flag and
        # fires once all in-flight credits return (the warmup barrier). Belt-
        # and-suspenders alongside total_expected_requests=loadgen.concurrency
        # in the warmup CreditPhaseConfig: handles the pool_size < concurrency
        # case where the count target would never be reached.
        self.lifecycle.mark_sending_complete()

    async def _execute_profiling(self) -> None:
        """Resume each trajectory at ``k_i + 1`` to seed the steady state.

        Subsequent turns and recycle-pool sessions are dispatched from
        handle_credit_return.
        """
        self.info(
            f"PROFILING execute: resuming {len(self.conversation_source.trajectories)} "
            f"trajectory sessions at k_i + 1"
        )
        for lane, trajectory in enumerate(self.conversation_source.trajectories):
            session = self.conversation_source.session_for(trajectory)
            self._correlation_to_lane[session.x_correlation_id] = lane
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

    async def handle_credit_return(self, credit: Credit) -> None:
        """Dispatch next turn or recycle on session completion.

        WARMUP returns are no-ops at the strategy level; phase termination is
        handled by ``SendingCompleteStopCondition`` + grace period. Terminal
        WARMUP failures are routed by ``CreditCallbackHandler`` directly into
        ``record_warmup_failure`` and surfaced at WARMUP teardown.

        PROFILING: if not the final turn, dispatch the next turn honoring
        trace ``delay_ms``. If the final turn just completed, recycle the
        trace_id and spawn a fresh session from the next queued trace_id.
        """
        if self.config.phase == CreditPhase.WARMUP:
            return

        if not credit.is_final_turn:
            await self._dispatch_next_turn(credit)
            return

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
        """
        # Drop the finished session's marker bookkeeping unconditionally so
        # every early-return path below leaves the dicts pruned.
        self._session_marker.pop(finished_correlation_id, None)

        # Lane bookkeeping: the correlation_id should be present. A missing
        # entry means the per-session bookkeeping invariant has already been
        # violated upstream (e.g. credit returning twice, or the session was
        # never registered on dispatch). Log a warning so the misuse is loud,
        # but fall back to lane 0 so the recycle still progresses (the
        # alternative — silent skip — leaves the queue head wedged and breaks
        # the test contract that recycle is unconditional on final-turn
        # return).
        if finished_correlation_id not in self._correlation_to_lane:
            self.warning(
                lambda: (
                    f"Recycle: finished_correlation_id={finished_correlation_id!r} "
                    f"missing from _correlation_to_lane; bookkeeping invariant "
                    f"violated. Falling back to lane 0 for trace_id={finished_trace_id!r}."
                )
            )
            lane = 0
        else:
            lane = self._correlation_to_lane.pop(finished_correlation_id)

        if self._recycle_queue is None:
            return

        # Unconditional double-recycle guard. Previously gated on __debug__,
        # which `python -O` strips, silently allowing the duplicate-final-turn
        # corruption to escape into production. Raise loudly here — this is a
        # programming error, not a runtime condition we can recover from.
        if finished_trace_id in self._in_flight_recycled:
            raise RuntimeError(
                f"Double recycle of trace_id {finished_trace_id!r} - "
                "handle_credit_return invoked twice for the same final turn"
            )
        self._in_flight_recycled.add(finished_trace_id)

        # Re-enqueue the finished trace_id BEFORE checking cooldown so that an
        # in-flight credit returning during cooldown does not permanently drop
        # the trace_id from the recycle pool. Cooldown gates *starting* a new
        # session; preserving FIFO recycle state is independent of that.
        self._recycle_queue.put_nowait(finished_trace_id)

        if not self.stop_checker.can_start_new_session():
            return

        try:
            next_trace_id = self._recycle_queue.get_nowait()
        except asyncio.QueueEmpty:
            return

        self._in_flight_recycled.discard(next_trace_id)

        session = self._build_session_for_trace(next_trace_id)
        if session is None or not session.metadata.turns:
            return

        self._correlation_to_lane[session.x_correlation_id] = lane
        self._mint_marker_for_session(session.x_correlation_id, next_trace_id, lane)

        turn = self._build_turn_for_session(session, 0)
        await self.credit_issuer.issue_credit(turn)

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
