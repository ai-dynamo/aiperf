# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphIRConversationSource — per-trace t* selection.

The schedule-plane source the ``GraphIRReplayStrategy`` consults to decide, per
weka trace, the wall-clock snapshot instant ``t*`` at/after which nodes are
profiled (pre-``t*`` firings are cache-priming warmup history).

t* selection mirrors agentx ``TrajectorySource`` (``timing/trajectory_source.py``):
a per-trace ``t*`` is drawn uniformly from
``[start_min_ratio, start_max_ratio] * trace_duration`` with a deterministic,
LANE-salted RNG (``sha256(f"{seed}:{trace_id}:{lane_index}")``), so the same
``(seed, ratios, lane_index)`` reproduce the exact same t* on every run and
across both parse planes. AgentX production ALWAYS uses the lane-salted seed
(``_seed_for_trace_lane``, ``trajectory_source.py:410``); each replayed trace
INSTANCE is a lane, and the single-pass non-recycling case is lane ``0``. The
default ratios ``[0.0, 0.0]`` give ``t*=0`` == full native replay: no warmup
history, every node profiled (the working profiling path is unchanged).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from aiperf.dataset.graph.models import ParsedGraph, TraceRecord
from aiperf.graph.analysis import trace_duration_us

__all__ = ["GraphIRConversationSource", "GraphTrace"]


@dataclass(slots=True, frozen=True)
class GraphTrace:
    """One weka trace's snapshot plan: the sampled snapshot instant ``t*``.

    Attributes:
        trace_id: The trace's root id.
        parsed_graph: The shared ``ParsedGraph`` the trace belongs to (passed
            through to the executor; t* reconstruction, when enabled, rewrites it
            per trace via the trie snapshot chop).
        t_star_us: The sampled snapshot instant in microseconds, kept as a float
            (agentx parity: ``trajectory_source.py`` keeps t* a float ms -- no
            integer truncation). ``0`` means full native replay (no warmup
            history).
    """

    trace_id: str
    parsed_graph: ParsedGraph
    t_star_us: float


class GraphIRConversationSource:
    """Yields a :class:`GraphTrace` (t* + node partition) per weka trace.

    Holds the shared ``ParsedGraph`` and the per-build node-ordinal catalog, and
    samples a deterministic per-trace ``t*`` from the configured ratio window.
    """

    def __init__(
        self,
        *,
        parsed: ParsedGraph,
        start_min_ratio: float = 0.0,
        start_max_ratio: float = 0.0,
        random_seed: int = 0,
        lane_index: int = 0,
    ) -> None:
        """Initialize the source.

        Args:
            parsed: The built ``ParsedGraph`` whose ``traces`` this source plans.
            start_min_ratio: Lower bound (fraction of trace duration) of the t*
                sampling window. Default ``0.0``. A ratio of ``0.0`` selects
                full native replay (t*=0, no warmup).
            start_max_ratio: Upper bound of the t* window. Default ``0.0``.
                Must be >= start_min_ratio.
            random_seed: Base seed for per-trace t* sampling; salted with the
                trace id AND ``lane_index`` so traces are uncorrelated yet
                reproducible.
            lane_index: The replay-lane index this source plans for. AgentX seeds
                t* per ``(trace_id, lane_index)`` so the same trace recurring
                across lanes decorrelates. The single-pass non-recycling case is
                lane ``0`` (the default), matching AgentX's first/only lane.
        """
        if start_min_ratio > start_max_ratio:
            raise ValueError(
                f"start_min_ratio ({start_min_ratio}) must be <= "
                f"start_max_ratio ({start_max_ratio})"
            )
        self._parsed = parsed
        self._start_min_ratio = start_min_ratio
        self._start_max_ratio = start_max_ratio
        self._random_seed = random_seed
        self._lane_index = lane_index

    def iter_traces(self) -> Iterator[GraphTrace]:
        """Yield one :class:`GraphTrace` per trace in the parsed graph."""
        for trace in self._parsed.traces:
            yield self._plan_trace(trace)

    def _plan_trace(self, trace: TraceRecord) -> GraphTrace:
        """Sample ``t*`` for ``trace``."""
        t_star_us = self._sample_t_star(trace)
        return GraphTrace(
            trace_id=trace.id,
            parsed_graph=self._parsed,
            t_star_us=max(t_star_us, 0.0),
        )

    def _sample_t_star(self, trace: TraceRecord) -> float:
        """Draw a deterministic per-trace ``t*`` in microseconds.

        Agentx parity: ``t* = uniform(lo, hi)`` over
        ``[start_min_ratio, start_max_ratio] * trace_duration``, with a
        LANE-salted RNG (``_seed_for_trace_lane`` -- AgentX production always
        passes a concrete lane, ``trajectory_source.py:410``). ``lo == hi``
        (e.g. the default ``[0, 0]`` window or a zero-duration trace) collapses
        to that exact instant with no draw. The draw is kept as a ``float``
        (no integer-microsecond truncation) to match AgentX, which keeps t* a
        float ms.
        """
        duration_us = trace_duration_us(self._parsed, trace)
        if duration_us <= 0:
            return 0.0
        lo = self._start_min_ratio * duration_us
        hi = self._start_max_ratio * duration_us
        if hi <= lo:
            return float(lo)
        rng = np.random.default_rng(
            _seed_for_trace_lane(self._random_seed, trace.id, self._lane_index)
        )
        return float(rng.uniform(lo, hi))


def _seed_for_trace_lane(base_seed: int, trace_id: str, lane_index: int) -> int:
    """Derive a per-trace-per-lane RNG seed (agentx ``_seed_for_trace_lane``).

    Agentx ``TrajectorySource._seed_for_trace_lane`` parity
    (``trajectory_source.py:149``): per-(trace, lane) t* values must be
    deterministic given ``base_seed`` yet decorrelated across both traces and
    lanes, so we SHA-256 ``f"{base_seed}:{trace_id}:{lane_index}"`` and take the
    low 8 bytes. AgentX production ALWAYS uses this lane variant
    (``_build_trajectory_for_lane`` passes a concrete lane, TS:410); the plain
    no-lane ``_seed_for_trace`` is unreachable in production.
    """
    digest = hashlib.sha256(f"{base_seed}:{trace_id}:{lane_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")
