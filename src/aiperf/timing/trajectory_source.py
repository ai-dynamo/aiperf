# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory conversation source for the AgenticReplay timing strategy.

Builds a fixed set of trajectories (each a (trace_id, start_turn_index) pair)
at construction time so trajectory state survives the WARMUP -> PROFILING
boundary. The WARMUP strategy reads each trajectory and dispatches turn k_i
for it; PROFILING resumes from k_i + 1 and feeds recycled trace_ids through
the standard ``next()`` path.

"Trajectory" matches the aa-agent-perf vocabulary and standard agentic-AI / RL
terminology for one rollout-style sequence of turns. Avoids conflating with
aiperf's existing ``User`` class in ``user_centric_rate.py``.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass

import numpy as np

from aiperf.common.models import DatasetMetadata
from aiperf.common.scenario.base import EmptyTracePoolError
from aiperf.dataset.protocols import DatasetSamplingStrategyProtocol
from aiperf.timing.conversation_source import ConversationSource, SampledSession

_logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class Trajectory:
    """One trajectory: (trace_id, sampled start turn index k_i)."""

    conversation_id: str
    start_turn_index: int


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
        pool_size = len(dataset_metadata.conversations)
        self._concurrency = concurrency
        self._pool_size = pool_size
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
        # Sort by lane (insertion order = lane assignment in dispatch loops)
        # so the table reads in the same order it'll be dispatched.
        for lane, trajectory in enumerate(self.trajectories):
            meta = self._metadata_lookup.get(trajectory.conversation_id)
            n_turns = len(meta.turns) if meta is not None else 0
            k_i = trajectory.start_turn_index
            pct = (k_i / n_turns * 100.0) if n_turns > 0 else 0.0
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
                f"{self._start_max_ratio:.2f}]  observed pct: "
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
        if raw_messages_count is not None:
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
            x_correlation_id=x_correlation_id or str(uuid.uuid4()),
            start_turn_index=trajectory.start_turn_index,
        )
