# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extended-warmup handoff payload -- the WARMUP -> PROFILING re-cut channel.

The graph-native counterpart of AgentX v1.0's ``finalize_phase`` trajectory
replacement (ported from the retired agentic-replay plane): the WARMUP ``GraphIRReplayStrategy``
builds one :class:`GraphWarmupHandoff` at ``teardown_phase`` (after every
warmup credit return has landed) and stashes it on the SHARED
``GraphPhaseChannel``; ``PhaseRunner._build_graph_ir_strategy`` pops it
(consume-once) into the PROFILING strategy, which resumes each lane at its
recorded execution frontier via ``chop_trie_at_frontier``.

Everything deterministic (the per-(trace, lane) t* plan) is NOT carried here
-- both phases re-derive it from the seeded sampler. The handoff carries only
what determinism cannot reproduce: which template each lane was mid-flight on
at drain, which nodes actually executed, and when their returns landed.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["GraphWarmupHandoff", "LaneHandoff"]


@dataclass(slots=True, frozen=True)
class LaneHandoff:
    """One lane's live-at-drain execution state."""

    # Template trace id the lane was mid-flight on at drain (may differ from
    # the lane's pass-0 template when the pressure stage recycled the lane).
    template_trace_id: str
    # The live pressure instance id at drain (e.g. ``t-1#0.p2``). The profiling
    # resume reuses it verbatim as the resumed instance's id so the per-instance
    # cache-bust marker (digest of ``credit.trace_id``; see
    # ``build_trace_instance_marker``) is continuous across the handoff and the
    # KV built during pressure transfers instead of cold-prefilling behind a
    # fresh ``.0`` marker.
    instance_id: str
    # The instance's t* (lane-salted plan for the pressure pass-0 instance;
    # 0.0 for recycled full-replay instances). Pre-t* nodes are warmup history
    # and are dropped by the frontier chop alongside the executed set.
    t_star_us: float
    # Node ids of the live instance that dispatched AND returned during
    # warmup/pressure -- the server holds their KV; profiling must not refire.
    executed_node_ids: frozenset[str]
    # Monotonic return wall times (microseconds, strategy ledger clock) used
    # to compute residual re-root delays; includes the lane's boundary-priming
    # returns merged in for pass-0 instances.
    return_wall_us: dict[str, float]


@dataclass(slots=True, frozen=True)
class GraphWarmupHandoff:
    """The full warmup -> profiling handoff, one entry per live lane."""

    # Lanes live at drain. Lanes absent here (template completed exactly at
    # drain, or lane index beyond the pressure lane count) resume the normal
    # pass-0 t* path in profiling.
    lanes: dict[int, LaneHandoff]
    # Drain-end instant on the same monotonic clock as the return walls,
    # stamped at warmup teardown (after all returns landed).
    drain_end_wall_us: float
    # Next corpus draw index after the pressure stage's last recycle draw.
    # Profiling's BOUNDED recycle loop continues the wrap from here so freed
    # lanes don't re-serve templates the pressure stage just replayed (agentx
    # shares ONE sampler across pressure / handoff / profiling draws). Single-
    # pass profiling (no stop conditions) deliberately ignores it -- full-corpus
    # coverage takes precedence over cursor continuity there.
    corpus_cursor: int
    # Number of pressure lanes (0..K-1) the warmup ran. A lane below this
    # count with NO entry in ``lanes`` completed at drain: profiling must
    # fresh-start it (next cursor template, full t*=0 replay) instead of
    # re-running a t* resume the pressure stage already executed -- agentx
    # ``_build_handoff_trajectories`` empty-lane parity.
    pressure_lane_count: int
