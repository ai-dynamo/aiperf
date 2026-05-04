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
from aiperf.common.scenario.base import (
    EmptyTracePoolError,
    InsufficientTrajectoriesError,
)
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


class TrajectorySource(ConversationSource):
    """ConversationSource that samples a fixed set of trajectories with randomized 0-70% start.

    Constructed once at TimingManager level (not per-phase) so trajectory
    state survives the WARMUP -> PROFILING boundary.
    """

    def __init__(
        self,
        dataset_metadata: DatasetMetadata,
        dataset_sampler: DatasetSamplingStrategyProtocol,
        concurrency: int,
        random_seed: int,
    ) -> None:
        super().__init__(
            dataset_metadata=dataset_metadata, dataset_sampler=dataset_sampler
        )

        if not dataset_metadata.conversations:
            raise EmptyTracePoolError(
                "Loader produced 0 traces; trajectories cannot be built."
            )

        self._random_seed = random_seed
        pool_size = len(dataset_metadata.conversations)
        self._concurrency = concurrency
        self._pool_size = pool_size
        # Build trajectories up to the user-requested concurrency. If we can't
        # fill that many lanes from the pool (either pool < concurrency, or
        # too many traces are too short to split into warmup+profiling), the
        # post-build check below rejects the run instead of silently capping
        # effective load below --concurrency.
        self._target_size = concurrency
        self.trajectories: list[Trajectory] = self._build_trajectories()

        if not self.trajectories:
            raise EmptyTracePoolError(
                "Trajectories empty after skipping invalid traces; pool exhausted."
            )

        if len(self.trajectories) < concurrency:
            raise InsufficientTrajectoriesError(
                concurrency=concurrency,
                usable_trajectories=len(self.trajectories),
                pool_size=pool_size,
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
            # k_i=0 leaves a profile turn (turn 1). For n>=3 keep the
            # 0..int(0.7*n) sample but cap at n-2 so k_i+1 < n always holds
            # (avoids the immediate-recycle pathology where PROFILING resume
            # index == num_turns and the trajectory dies on its first credit).
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
                k_i = 0
            else:
                k_max = min(int(0.7 * n), n - 2)
                k_i = int(rng.integers(low=0, high=k_max + 1))
            trajectories.append(Trajectory(conversation_id=cid, start_turn_index=k_i))

        return trajectories

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
