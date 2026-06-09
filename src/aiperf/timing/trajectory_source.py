# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory conversation source for the AgenticReplay timing strategy.

Builds a fixed set of trajectories at construction time so trajectory state
survives the WARMUP -> PROFILING boundary. Timestamped traces are sampled as
wall-clock snapshots: choose a ``t*`` inside the configured percent range,
then reconstruct the conversations that are alive at that instant (root and
subagents). Legacy timestamp-less datasets fall back to the original
``(trace_id, start_turn_index)`` split.

"Trajectory" matches the aa-agent-perf vocabulary and standard agentic-AI / RL
terminology for one rollout-style sequence of turns. Avoids conflating with
aiperf's existing ``User`` class in ``user_centric_rate.py``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from dataclasses import dataclass, field

import numpy as np

from aiperf.common.enums import (
    ConversationBranchMode,
    ConversationContextMode,
    PrerequisiteKind,
)
from aiperf.common.models import DatasetMetadata
from aiperf.common.scenario.base import EmptyTracePoolError
from aiperf.dataset.protocols import DatasetSamplingStrategyProtocol
from aiperf.timing.conversation_source import ConversationSource, SampledSession

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ConversationState:
    """One live conversation in a wall-clock trajectory snapshot."""

    conversation_id: str
    x_correlation_id: str
    next_turn_index: int
    next_dispatch_offset_ms: float = 0.0
    agent_depth: int = 0
    parent_correlation_id: str | None = None
    waiting_on_children: bool = False
    join_target_turn_index: int | None = None
    branch_id: str | None = None
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK


@dataclass(slots=True, frozen=True)
class TrajectorySnapshot:
    """Wall-clock state for one sampled root trace."""

    t_star_ms: float
    states: tuple[ConversationState, ...]


@dataclass(slots=True, frozen=True)
class Trajectory:
    """One sampled replay lane.

    ``snapshot`` is set for timestamped traces. ``start_turn_index`` remains
    available for compatibility with timestamp-less datasets and older tests.
    ``x_correlation_id`` is the persistent session identity used by legacy
    timestamp-less trajectories across the WARMUP -> PROFILING boundary.
    Timestamped snapshots store the equivalent realized identity graph on each
    ``ConversationState``.
    """

    conversation_id: str
    start_turn_index: int
    snapshot: TrajectorySnapshot | None = None
    x_correlation_id: str = field(
        default_factory=lambda: str(uuid.uuid4()), compare=False
    )


@dataclass(slots=True, frozen=True)
class _BranchRuntime:
    branch_id: str
    child_conversation_ids: tuple[str, ...]
    mode: ConversationBranchMode
    is_background: bool
    start_timestamp_ms: float | None
    join_turn_index: int | None
    spawning_turn_index: int | None


def _seed_for_trace(base_seed: int, trace_id: str) -> int:
    """Derive a per-trace RNG seed by hashing trace_id with the base seed.

    Per-trajectory k_i values must be deterministic given base_seed but
    uncorrelated across traces. Salting with trace_id via SHA-256 avoids
    linear correlation.
    """
    h = hashlib.sha256(f"{base_seed}:{trace_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def _seed_for_trace_lane(base_seed: int, trace_id: str, lane_index: int) -> int:
    """Derive a per-(trace, lane) RNG seed by hashing ``trace_id`` and lane index.

    Wrap-fill lanes share a ``conversation_id`` but must produce different
    ``start_turn_index`` values; salting the digest with ``lane_index``
    decorrelates them while keeping the choice deterministic in ``base_seed``.
    """
    h = hashlib.sha256(f"{base_seed}:{trace_id}:{lane_index}".encode()).digest()
    return int.from_bytes(h[:8], "big")


class TrajectorySource(ConversationSource):
    """ConversationSource that samples a fixed set of trajectories with a randomized
    per-trajectory start position drawn from [start_min_ratio, start_max_ratio] of
    each trace's total turn count.

    Constructed once at TimingManager level (not per-phase) so trajectory
    state survives the WARMUP -> PROFILING boundary.
    """

    def __init__(
        self,
        *,
        dataset_metadata: DatasetMetadata,
        dataset_sampler: DatasetSamplingStrategyProtocol,
        concurrency: int,
        random_seed: int,
        start_min_ratio: float = 0.0,
        start_max_ratio: float = 0.7,
    ) -> None:
        super().__init__(
            dataset_metadata=dataset_metadata, dataset_sampler=dataset_sampler
        )

        if not dataset_metadata.conversations:
            raise EmptyTracePoolError(
                "Loader produced 0 traces; trajectories cannot be built."
            )

        if start_min_ratio > start_max_ratio:
            raise ValueError(
                f"start_min_ratio ({start_min_ratio}) must be <= "
                f"start_max_ratio ({start_max_ratio})."
            )

        self._random_seed = random_seed
        self._start_min_ratio = start_min_ratio
        self._start_max_ratio = start_max_ratio
        pool_size = sum(
            1
            for conv in dataset_metadata.conversations
            if getattr(conv, "is_root", True) is not False
        )
        self._concurrency = concurrency
        self._pool_size = pool_size
        self._children_by_parent: dict[str, set[str]] = self._build_child_index()
        self._warned_live_delta_snapshot = False
        # Build distinct trajectories up to the user-requested concurrency.
        # If the pool or its usable subset (after dropping traces too short
        # to split into warmup+profile turns) is smaller than concurrency,
        # ``_wrap_fill_lanes`` below cycles through the distinct trajectories
        # with fresh per-lane ``start_turn_index`` salts so the run still
        # honours ``--concurrency`` instead of silently capping effective load.
        self._target_size = concurrency
        distinct: list[Trajectory] = self._build_trajectories()

        if not distinct:
            raise EmptyTracePoolError(
                "Trajectories empty after skipping invalid traces; pool exhausted."
            )

        self.trajectories: list[Trajectory] = list(distinct)
        if len(self.trajectories) < concurrency:
            extras = self._wrap_fill_lanes(distinct, concurrency - len(distinct))
            self.trajectories.extend(extras)
            _logger.info(
                "Trajectory reuse: %d distinct trajectories fanned out to %d "
                "lanes (avg %.1f lanes per trace). Cache-bust marker keeps "
                "per-lane traffic distinct when cache_bust.target != NONE.",
                len(distinct),
                concurrency,
                concurrency / len(distinct),
            )

        self._log_trajectory_summary()

    def _log_trajectory_summary(self) -> None:
        """Log a one-block table of every trajectory's start position.

        Format::

            TrajectorySource: built 14 trajectories from 949 traces
              range cfg=[0.25, 0.75]  observed pct: min=27% median=51% max=72%
                lane=00  start_turn= 6/24 (25%)  trace_id=abc123
                lane=01  start_turn=15/22 (68%)  trace_id=def456
                ...

        Emitted once at construction. Lets you sanity-check the configured
        start-range produced sensible per-trajectory positions before any
        request fires, without needing to wait for warmup-completion lines
        or correlate per-credit return logs.
        """
        rows: list[str] = []
        pcts: list[float] = []
        has_snapshots = False
        # Sort by lane (insertion order = lane assignment in dispatch loops)
        # so the table reads in the same order it'll be dispatched.
        for lane, trajectory in enumerate(self.trajectories):
            meta = self._metadata_lookup.get(trajectory.conversation_id)
            n_turns = len(meta.turns) if meta is not None else 0
            k_i = trajectory.start_turn_index
            turn_pct = (k_i / n_turns * 100.0) if n_turns > 0 else 0.0
            if trajectory.snapshot is not None:
                has_snapshots = True
                pct = self._trajectory_snapshot_pct(trajectory)
                ready = sum(
                    1
                    for state in trajectory.snapshot.states
                    if not state.waiting_on_children
                )
                rows.append(
                    f"    lane={lane:02d}  sample_time={pct:>3.0f}%  "
                    f"root_next={k_i:>3d}/{n_turns:<3d} "
                    f"({turn_pct:>3.0f}% turns)  "
                    f"live={len(trajectory.snapshot.states)} ready={ready}  "
                    f"trace_id={trajectory.conversation_id}"
                )
                pcts.append(pct)
                continue

            pct = turn_pct
            pcts.append(pct)
            rows.append(
                f"    lane={lane:02d}  start_turn={k_i:>3d}/{n_turns:<3d} "
                f"({pct:>3.0f}%)  trace_id={trajectory.conversation_id}"
            )

        if pcts:
            pcts_sorted = sorted(pcts)
            mid = len(pcts_sorted) // 2
            if len(pcts_sorted) % 2 == 0:
                median = (pcts_sorted[mid - 1] + pcts_sorted[mid]) / 2
            else:
                median = pcts_sorted[mid]
            obs_line = (
                f"  range cfg=[{self._start_min_ratio:.2f}, "
                f"{self._start_max_ratio:.2f}]  observed "
                f"{'sample' if has_snapshots else 'turn'} pct: "
                f"min={min(pcts):>3.0f}% median={median:>3.0f}% "
                f"max={max(pcts):>3.0f}%"
            )
        else:
            obs_line = (
                f"  range cfg=[{self._start_min_ratio:.2f}, "
                f"{self._start_max_ratio:.2f}]  (no trajectories built)"
            )

        body = "\n".join(rows)
        _logger.info(
            "TrajectorySource: built %d trajectories from %d traces\n%s\n%s",
            len(self.trajectories),
            self._pool_size,
            obs_line,
            body,
        )

    @property
    def warmup_credit_count(self) -> int:
        """Number of ready snapshot conversations warmup will dispatch."""
        total = 0
        for trajectory in self.trajectories:
            if trajectory.snapshot is None:
                total += 1
            else:
                total += sum(
                    1
                    for state in trajectory.snapshot.states
                    if not state.waiting_on_children
                )
        return total

    def _build_trajectories(self) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        seen: set[str] = set()
        attempts = 0
        max_attempts = len(self._metadata_lookup) * 2

        while len(trajectories) < self._target_size and attempts < max_attempts:
            attempts += 1
            try:
                cid = self._dataset_sampler.next_conversation_id()
            except StopIteration:
                break
            if cid in seen:
                continue
            seen.add(cid)
            meta = self._metadata_lookup.get(cid)
            if meta is None or not meta.turns:
                _logger.warning(
                    "Skipping trace %r at trajectory selection: %d turns.",
                    cid,
                    0 if meta is None else len(meta.turns),
                )
                continue
            timestamped = self._build_timestamped_trajectory(cid)
            if timestamped is not None:
                trajectories.append(timestamped)
                continue

            n = len(meta.turns)
            # Require at least one PROFILING turn after WARMUP. For n<=1
            # there is no profile turn at all, so reject. For n==2 only
            # k_i=0 leaves a profile turn (turn 1). For n>=3 sample uniformly
            # from [int(start_min_ratio * n), int(start_max_ratio * n)] but
            # cap at n-2 so k_i+1 < n always holds (avoids the immediate-
            # recycle pathology where PROFILING resume index == num_turns
            # and the trajectory dies on its first credit). The lower bound
            # is also clamped to n-2 in case start_min_ratio * n exceeds it.
            if n <= 1:
                _logger.warning(
                    "Skipping trace %r at trajectory selection: %d turns "
                    "(need >= 2 for warmup+profile split).",
                    cid,
                    n,
                )
                continue
            rng = np.random.default_rng(_seed_for_trace(self._random_seed, cid))
            if n == 2:
                candidates = [0]
            else:
                k_min = min(int(self._start_min_ratio * n), n - 2)
                k_max = min(int(self._start_max_ratio * n), n - 2)
                if k_min > k_max:
                    k_min = k_max
                candidates = list(range(k_min, k_max + 1))

            candidates = [
                k
                for k in candidates
                if self._trajectory_start_is_sendable(meta, k)
                and self._trajectory_start_is_sendable(meta, k + 1)
            ]
            if not candidates:
                _logger.warning(
                    "Skipping trace %r at trajectory selection: no valid "
                    "warmup/profile start pair in configured range.",
                    cid,
                )
                continue
            k_i = int(rng.choice(candidates))
            trajectories.append(Trajectory(conversation_id=cid, start_turn_index=k_i))

        return trajectories

    def _trajectory_snapshot_pct(self, trajectory: Trajectory) -> float:
        if trajectory.snapshot is None:
            meta = self._metadata_lookup.get(trajectory.conversation_id)
            n_turns = len(meta.turns) if meta is not None else 0
            return trajectory.start_turn_index / n_turns * 100.0 if n_turns > 0 else 0.0
        bounds = self._trace_time_bounds(trajectory.conversation_id)
        if bounds is None:
            return 0.0
        start_ms, end_ms = bounds
        duration_ms = end_ms - start_ms
        if duration_ms <= 0:
            return 0.0
        return (trajectory.snapshot.t_star_ms - start_ms) / duration_ms * 100.0

    @staticmethod
    def _trajectory_start_is_sendable(meta, turn_index: int) -> bool:
        """Return whether ``turn_index`` can be the first request of a session.

        Agentic replay starts a fresh session at ``k_i`` during WARMUP and at
        ``k_i + 1`` during PROFILING. In live-assistant mode the loader emits
        user-only deltas, so a mid-trace turn whose delta contains only the
        original assistant segment has ``raw_messages=[]``. That turn is valid
        after prior live responses have been accumulated, but invalid as the
        first request of a fresh OpenAI chat session: vLLM/Kimi rejects an
        empty ``messages`` array in the chat template. Metadata stores only
        ``raw_messages_count`` instead of duplicating full OpenAI payloads.
        """
        if turn_index < 0 or turn_index >= len(meta.turns):
            return False
        turn = meta.turns[turn_index]

        raw_messages_count = getattr(turn, "raw_messages_count", None)
        if isinstance(raw_messages_count, int):
            if raw_messages_count > 0:
                return True
            return bool(meta.system_message or meta.user_context_message)

        # Backwards compatibility for old metadata and lightweight tests.
        raw_messages = getattr(turn, "raw_messages", None)
        if raw_messages is None:
            return True
        if raw_messages:
            return True
        return bool(meta.system_message or meta.user_context_message)

    def _wrap_fill_lanes(
        self, distinct: list[Trajectory], extra_count: int
    ) -> list[Trajectory]:
        """Return ``extra_count`` additional trajectories cycling through ``distinct``.

        Each wrap-filled lane reuses a source ``conversation_id`` but gets a
        fresh ``start_turn_index`` sampled with a per-(trace, absolute-lane-index)
        RNG seed. ``absolute_lane_index`` is ``len(distinct) + i`` where ``i``
        is the position within the extra block, so seeds are unique even when
        two extras share the same source ``conversation_id``.
        """
        extras: list[Trajectory] = []
        base_count = len(distinct)
        for i in range(extra_count):
            source = distinct[i % base_count]
            lane_index = base_count + i
            if source.snapshot is not None:
                timestamped = self._build_timestamped_trajectory(
                    source.conversation_id, lane_index=lane_index
                )
                if timestamped is not None:
                    extras.append(timestamped)
                continue
            meta = self._metadata_lookup[source.conversation_id]
            n = len(meta.turns)
            rng = np.random.default_rng(
                _seed_for_trace_lane(
                    self._random_seed, source.conversation_id, lane_index
                )
            )
            if n == 2:
                candidates = [0]
            else:
                k_min = min(int(self._start_min_ratio * n), n - 2)
                k_max = min(int(self._start_max_ratio * n), n - 2)
                if k_min > k_max:
                    k_min = k_max
                candidates = list(range(k_min, k_max + 1))
            candidates = [
                k
                for k in candidates
                if self._trajectory_start_is_sendable(meta, k)
                and self._trajectory_start_is_sendable(meta, k + 1)
            ]
            if not candidates:
                _logger.warning(
                    "Skipping wrap-fill lane for trace %r: no valid "
                    "warmup/profile start pair in configured range.",
                    source.conversation_id,
                )
                continue
            k_i = int(rng.choice(candidates))
            extras.append(
                Trajectory(conversation_id=source.conversation_id, start_turn_index=k_i)
            )
        return extras

    def _build_child_index(self) -> dict[str, set[str]]:
        children_by_parent: dict[str, set[str]] = {}
        for meta in self._metadata_lookup.values():
            for branch in getattr(meta, "branches", []) or []:
                if branch.child_conversation_ids:
                    children_by_parent.setdefault(meta.conversation_id, set()).update(
                        branch.child_conversation_ids
                    )
        return children_by_parent

    def _collect_trace_conversation_ids(self, root_id: str) -> set[str]:
        """Return root + recursively reachable child conversation ids."""
        seen: set[str] = set()
        stack = [root_id]
        while stack:
            cid = stack.pop()
            if cid in seen:
                continue
            seen.add(cid)
            stack.extend(self._children_by_parent.get(cid, ()))
        return seen

    def _build_timestamped_trajectory(
        self, root_id: str, lane_index: int | None = None
    ) -> Trajectory | None:
        bounds = self._trace_time_bounds(root_id)
        if bounds is None:
            return None

        start_ms, end_ms = bounds
        duration_ms = end_ms - start_ms
        seed = (
            _seed_for_trace(self._random_seed, root_id)
            if lane_index is None
            else _seed_for_trace_lane(self._random_seed, root_id, lane_index)
        )
        rng = np.random.default_rng(seed)
        lo = start_ms + self._start_min_ratio * duration_ms
        hi = start_ms + self._start_max_ratio * duration_ms
        if hi < lo:
            hi = lo
        t_star_ms = float(lo if hi == lo else rng.uniform(lo, hi))

        snapshot = self._snapshot_for(root_id, t_star_ms)
        if snapshot is None:
            return None
        self._warn_if_live_delta_snapshot_needs_prior_responses(root_id, snapshot)

        root_state = next(
            (state for state in snapshot.states if state.conversation_id == root_id),
            None,
        )
        start_turn_index = root_state.next_turn_index if root_state is not None else 0
        return Trajectory(
            conversation_id=root_id,
            start_turn_index=start_turn_index,
            snapshot=snapshot,
        )

    def _trace_time_bounds(self, root_id: str) -> tuple[float, float] | None:
        timestamps: list[float] = []
        for cid in self._collect_trace_conversation_ids(root_id):
            meta = self._metadata_lookup.get(cid)
            if meta is None:
                continue
            for turn in meta.turns:
                t_ms = _as_timestamp_ms(getattr(turn, "timestamp_ms", None))
                if t_ms is not None:
                    timestamps.append(t_ms)
        if not timestamps:
            return None
        return min(timestamps), max(timestamps)

    def _warn_if_live_delta_snapshot_needs_prior_responses(
        self, root_id: str, snapshot: TrajectorySnapshot
    ) -> None:
        if self._warned_live_delta_snapshot:
            return
        if (
            self.dataset_metadata.default_context_mode
            != ConversationContextMode.DELTAS_WITHOUT_RESPONSES
        ):
            return
        if not any(state.next_turn_index > 0 for state in snapshot.states):
            return
        self._warned_live_delta_snapshot = True
        _logger.warning(
            "Agentic replay snapshot for trace %r starts at one or more "
            "non-zero turn indices while the dataset uses "
            "DELTAS_WITHOUT_RESPONSES. Earlier live assistant responses are "
            "not available to bootstrap skipped turns; replay can start "
            "in-flight subagents, but prompt fidelity for those skipped-prefix "
            "sessions remains limited unless the dataset provides responses.",
            root_id,
        )

    def _snapshot_for(
        self, root_id: str, t_star_ms: float
    ) -> TrajectorySnapshot | None:
        root_meta = self._metadata_lookup[root_id]
        parent_corr = str(uuid.uuid4())
        root_next_idx = _next_turn_index_at_or_after(root_meta, t_star_ms)

        states: list[ConversationState] = []
        child_states: list[ConversationState] = []
        pending_join_targets: set[int] = set()
        branch_runtimes = self._branch_runtimes(root_meta)

        for runtime in branch_runtimes:
            start_ts = runtime.start_timestamp_ms
            if start_ts is not None and t_star_ms < start_ts:
                spawn_ts = (
                    _turn_timestamp_ms(root_meta, runtime.spawning_turn_index)
                    if runtime.spawning_turn_index is not None
                    else None
                )
                spawn_turn_not_completed = (
                    runtime.spawning_turn_index is None
                    or spawn_ts is None
                    or t_star_ms < spawn_ts
                    or (
                        root_next_idx is not None
                        and root_next_idx <= runtime.spawning_turn_index
                    )
                )
                if spawn_turn_not_completed:
                    continue

            branch_child_states: list[ConversationState] = []
            for child_cid in runtime.child_conversation_ids:
                child_meta = self._metadata_lookup.get(child_cid)
                if child_meta is None:
                    continue
                child_next_idx = _next_turn_index_at_or_after(child_meta, t_star_ms)
                if child_next_idx is None:
                    continue
                child_ts = _turn_timestamp_ms(child_meta, child_next_idx)
                # If the branch lacks an explicit start timestamp, use the
                # child's first request as a conservative spawn boundary.
                if start_ts is None:
                    first_ts = _turn_timestamp_ms(child_meta, 0)
                    if first_ts is not None and t_star_ms < first_ts:
                        continue
                branch_child_states.append(
                    ConversationState(
                        conversation_id=child_cid,
                        x_correlation_id=str(uuid.uuid4()),
                        next_turn_index=child_next_idx,
                        next_dispatch_offset_ms=_offset_ms(child_ts, t_star_ms),
                        agent_depth=getattr(child_meta, "agent_depth", 1) or 1,
                        parent_correlation_id=parent_corr,
                        waiting_on_children=False,
                        join_target_turn_index=runtime.join_turn_index,
                        branch_id=runtime.branch_id,
                        branch_mode=runtime.mode,
                    )
                )

            child_states.extend(branch_child_states)
            if branch_child_states and runtime.join_turn_index is not None:
                pending_join_targets.add(runtime.join_turn_index)

        root_state: ConversationState | None = None
        if root_next_idx is not None:
            root_ts = _turn_timestamp_ms(root_meta, root_next_idx)
            waiting = root_next_idx in pending_join_targets
            root_state = ConversationState(
                conversation_id=root_id,
                x_correlation_id=parent_corr,
                next_turn_index=root_next_idx,
                next_dispatch_offset_ms=_offset_ms(root_ts, t_star_ms),
                agent_depth=getattr(root_meta, "agent_depth", 0),
                parent_correlation_id=None,
                waiting_on_children=waiting,
                join_target_turn_index=root_next_idx if waiting else None,
                branch_id=None,
                branch_mode=ConversationBranchMode.FORK,
            )
            states.append(root_state)

        # If children are active but the root's next timestamp is absent, keep
        # the children. This can happen for terminal background subagents.
        states.extend(child_states)
        if not states:
            return None
        if not any(not state.waiting_on_children for state in states):
            return None
        return TrajectorySnapshot(t_star_ms=t_star_ms, states=tuple(states))

    def _branch_runtimes(self, parent_meta) -> list[_BranchRuntime]:
        join_by_branch: dict[str, int] = {}
        spawn_by_branch: dict[str, int] = {}
        for turn_idx, turn in enumerate(parent_meta.turns):
            for branch_id in getattr(turn, "branch_ids", []) or []:
                spawn_by_branch.setdefault(branch_id, turn_idx)
            for prereq in getattr(turn, "prerequisites", []) or []:
                if (
                    prereq.kind == PrerequisiteKind.SPAWN_JOIN
                    and prereq.branch_id is not None
                    and prereq.branch_id not in join_by_branch
                ):
                    join_by_branch[prereq.branch_id] = turn_idx

        runtimes: list[_BranchRuntime] = []
        for branch in getattr(parent_meta, "branches", []) or []:
            start_ts = _as_timestamp_ms(getattr(branch, "start_timestamp_ms", None))
            if start_ts is None:
                child_starts = [
                    ts
                    for child_id in branch.child_conversation_ids
                    if (child_meta := self._metadata_lookup.get(child_id)) is not None
                    if (ts := _turn_timestamp_ms(child_meta, 0)) is not None
                ]
                if child_starts:
                    start_ts = min(child_starts)
            runtimes.append(
                _BranchRuntime(
                    branch_id=branch.branch_id,
                    child_conversation_ids=tuple(branch.child_conversation_ids),
                    mode=branch.mode,
                    is_background=branch.is_background,
                    start_timestamp_ms=start_ts,
                    join_turn_index=None
                    if branch.is_background
                    else join_by_branch.get(branch.branch_id),
                    spawning_turn_index=spawn_by_branch.get(branch.branch_id),
                )
            )
        return runtimes

    def session_for(
        self,
        trajectory: Trajectory,
        x_correlation_id: str | None = None,
    ) -> SampledSession:
        """Build a SampledSession for a trajectory with start_turn_index pre-set."""
        meta = self._metadata_lookup[trajectory.conversation_id]
        return SampledSession(
            conversation_id=trajectory.conversation_id,
            metadata=meta,
            x_correlation_id=x_correlation_id or trajectory.x_correlation_id,
            start_turn_index=trajectory.start_turn_index,
        )

    def session_for_state(self, state: ConversationState) -> SampledSession:
        """Build a SampledSession for one live snapshot conversation state."""
        meta = self._metadata_lookup[state.conversation_id]
        return SampledSession(
            conversation_id=state.conversation_id,
            metadata=meta,
            x_correlation_id=state.x_correlation_id,
            agent_depth=state.agent_depth,
            parent_correlation_id=state.parent_correlation_id,
            branch_mode=state.branch_mode,
            start_turn_index=state.next_turn_index,
        )


def _as_timestamp_ms(value: object) -> float | None:
    if isinstance(value, int | float):
        v = float(value)
        # Reject non-finite timestamps (NaN / +-inf). A malformed loader value
        # would otherwise poison min()/max() in _trace_time_bounds and make
        # rng.uniform(lo, hi) raise OverflowError out of __init__. Treating it
        # as absent lets the trace use its remaining finite timestamps (or
        # fall back to the timestamp-less split).
        return v if math.isfinite(v) else None
    return None


def _turn_timestamp_ms(meta, turn_index: int) -> float | None:
    if turn_index < 0 or turn_index >= len(meta.turns):
        return None
    return _as_timestamp_ms(getattr(meta.turns[turn_index], "timestamp_ms", None))


def _next_turn_index_at_or_after(meta, t_star_ms: float) -> int | None:
    for idx, turn in enumerate(meta.turns):
        t_ms = _as_timestamp_ms(getattr(turn, "timestamp_ms", None))
        if t_ms is not None and t_ms >= t_star_ms:
            return idx
    return None


def _offset_ms(timestamp_ms: float | None, t_star_ms: float) -> float:
    if timestamp_ms is None:
        return 0.0
    return max(0.0, timestamp_ms - t_star_ms)
