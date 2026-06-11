# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial unit tests for ``TrajectorySource`` snapshot sampling.

Probes accounting and numeric-edge anomalies NOT already covered by
``test_trajectory_source.py``, ``test_trajectory_source_extended_adversarial.py``,
or ``test_trajectory_source_wrap_fill.py``:

- Non-finite timestamps (NaN / +inf) poisoning ``_trace_time_bounds`` ->
  ``rng.uniform`` (CONFIRMED crash, xfail).
- Non-monotonic / duplicate timestamps and ``_next_turn_index_at_or_after``
  picking the first-at-or-after turn (characterization).
- ``start_max_ratio == 1.0`` snapshot putting the root on its final turn with
  no profiling turn after warmup (asymmetry vs the timestamp-less ``n-2`` cap).
- ``pool_size`` (root-only) vs the build loop accepting any sampled id
  (characterization).
- Background-only subagent snapshot (``root_next_idx is None``) defaulting the
  trajectory ``start_turn_index`` to 0 despite no root state.
- Empty ``child_conversation_ids`` on a joined branch -> root not marked
  waiting (vacuous join, characterization).
- Missing child metadata silently dropped from the snapshot.
- ``warmup_credit_count`` matching exactly what ``_execute_warmup`` would
  dispatch across mixed trajectory types.
- ``start_min_ratio``/``start_max_ratio`` boundary validation.
"""

from __future__ import annotations

import math

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.trajectory_source import (
    TrajectorySource,
    _next_turn_index_at_or_after,
)

# =============================================================================
# Helpers
# =============================================================================


class _Sampler:
    """Stub sampler that hands out ids in order, then raises StopIteration."""

    def __init__(self, ids: list[str]) -> None:
        self._ids = list(ids)
        self._i = 0

    def next_conversation_id(self) -> str:
        if self._i >= len(self._ids):
            raise StopIteration
        cid = self._ids[self._i]
        self._i += 1
        return cid


def _dataset(convs: list[ConversationMetadata]) -> DatasetMetadata:
    return DatasetMetadata(
        conversations=convs,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _ts_conv(cid: str, timestamps: list[float | None]) -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=cid,
        turns=[TurnMetadata(timestamp_ms=t) for t in timestamps],
    )


def _build(
    convs: list[ConversationMetadata],
    order: list[str],
    *,
    concurrency: int = 1,
    seed: int = 42,
    start_min_ratio: float = 0.0,
    start_max_ratio: float = 0.7,
) -> TrajectorySource:
    return TrajectorySource(
        dataset_metadata=_dataset(convs),
        dataset_sampler=_Sampler(order),
        concurrency=concurrency,
        random_seed=seed,
        start_min_ratio=start_min_ratio,
        start_max_ratio=start_max_ratio,
    )


# =============================================================================
# Regression: non-finite timestamps must not crash construction
# (fixed: _as_timestamp_ms rejects NaN/+-inf so they are treated as absent)
# =============================================================================


def test_nan_timestamp_in_root_does_not_crash_construction() -> None:
    """A single NaN turn timestamp must not crash TrajectorySource construction.

    The loader can emit a malformed timestamp; the source should skip the
    trace or fall back to the timestamp-less path, never propagate an
    unhandled numpy OverflowError out of ``__init__``.
    """
    convs = [_ts_conv("c", [float("nan"), 1000.0, 2000.0])]
    src = _build(convs, ["c"], start_min_ratio=0.0, start_max_ratio=0.5)
    assert len(src.trajectories) == 1


def test_positive_inf_timestamp_in_root_does_not_crash_construction() -> None:
    """A +inf turn timestamp must not crash construction."""
    convs = [_ts_conv("c", [0.0, float("inf"), 2000.0])]
    src = _build(convs, ["c"], start_min_ratio=0.0, start_max_ratio=0.5)
    assert len(src.trajectories) == 1


# =============================================================================
# _next_turn_index_at_or_after: non-monotonic / duplicate timestamps
# =============================================================================


def test_next_turn_index_non_monotonic_returns_first_at_or_after_not_earliest() -> None:
    """With out-of-order timestamps, the resume index is the first turn whose
    timestamp is >= t_star in *list order*, NOT the chronologically-next turn.

    Turn 1 (ts=50) is chronologically before t_star=60 but turn 0 (ts=100) is
    returned because it is the first in list order with ts >= 60. A snapshot
    taken at t_star=60 thus 'resumes' at turn 0 and silently re-sends turn 0
    while the chronologically-past turn 1 is skipped. Characterization of the
    list-order scan in ``_next_turn_index_at_or_after``.
    """
    meta = _ts_conv("c", [100.0, 50.0, 200.0])
    assert _next_turn_index_at_or_after(meta, 60.0) == 0
    # After turn 0's timestamp, the next at-or-after is turn 2 (ts=200),
    # skipping turn 1 (ts=50) entirely.
    assert _next_turn_index_at_or_after(meta, 150.0) == 2


def test_next_turn_index_duplicate_timestamps_returns_earliest_index() -> None:
    """All-equal timestamps: t_star at the shared value resolves to index 0.

    Every turn shares ts=1000. Sampling exactly at 1000 returns the first
    turn; one millisecond later nothing is at-or-after, so None. This means a
    snapshot 'inside' such a flat trace always rewinds to turn 0.
    """
    meta = _ts_conv("c", [1000.0, 1000.0, 1000.0, 1000.0])
    assert _next_turn_index_at_or_after(meta, 1000.0) == 0
    assert _next_turn_index_at_or_after(meta, 1000.1) is None


def test_duplicate_timestamp_snapshot_collapses_all_offsets_to_zero() -> None:
    """A flat-timestamp trace yields a snapshot whose root resumes at turn 0
    with a zero dispatch offset (all turns share the sample instant)."""
    convs = [_ts_conv("c", [1000.0, 1000.0, 1000.0])]
    # min==max==1000 -> duration 0 -> t_star == 1000 regardless of ratio.
    src = _build(convs, ["c"], start_min_ratio=0.3, start_max_ratio=0.6)
    traj = src.trajectories[0]
    assert traj.snapshot is not None
    assert traj.snapshot.t_star_ms == pytest.approx(1000.0)
    root = traj.snapshot.states[0]
    assert root.conversation_id == "c"
    assert root.next_turn_index == 0
    assert root.next_dispatch_offset_ms == pytest.approx(0.0)


# =============================================================================
# start_max_ratio == 1.0: timestamped snapshot can land root on its final turn
# =============================================================================


def test_timestamped_snapshot_at_ratio_one_places_root_on_final_turn() -> None:
    """At ratio 1.0 a timestamped snapshot resumes the root at ``n-1``.

    The timestamp-less build path caps k_i at ``n-2`` (trajectory_source.py:335)
    precisely to guarantee a profiling turn at ``k_i + 1`` exists. The
    timestamped snapshot path applies NO such cap: at start ratio 1.0 the root
    lands on its last turn (next_turn_index == n-1), so profiling continuation
    (resume_index == n) has no turn to send. The continuation handler drops
    such a state defensively (agentic_replay.py:523), so the lane contributes a
    warmup credit but zero profiling credits -- a silently shortened lane.
    Characterization of this timestamped/timestamp-less asymmetry.
    """
    convs = [_ts_conv("c", [0.0, 1000.0, 2000.0, 3000.0, 4000.0])]
    src = _build(convs, ["c"], start_min_ratio=1.0, start_max_ratio=1.0)
    traj = src.trajectories[0]
    assert traj.snapshot is not None
    n = 5
    root = next(s for s in traj.snapshot.states if s.conversation_id == "c")
    assert root.next_turn_index == n - 1  # final turn -> no k_i+1 profile turn
    # Snapshot pct sits at 100% (t_star == last timestamp).
    assert src._trajectory_snapshot_pct(traj) == pytest.approx(100.0)


def test_timestampless_path_caps_k_i_at_n_minus_two() -> None:
    """Contrast: the timestamp-less path caps k_i at n-2 so k_i+1 < n holds."""
    convs = [
        ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata() for _ in range(5)],  # no timestamps
        )
    ]
    src = _build(convs, ["c"], start_min_ratio=1.0, start_max_ratio=1.0)
    traj = src.trajectories[0]
    assert traj.snapshot is None
    assert traj.start_turn_index == 5 - 2  # n-2, leaving a profile turn at n-1


# =============================================================================
# pool_size (root-only) vs build loop (accepts whatever the sampler returns)
# =============================================================================


def test_pool_size_counts_roots_only_but_build_accepts_non_root_sampled_id() -> None:
    """``_pool_size`` counts only roots, yet ``_build_trajectories`` builds a
    trajectory on whatever id the sampler returns -- including a non-root child.

    There is no ``is_root`` filter inside the build loop, so a sampler that
    yields a child id will produce a trajectory rooted at a DAG child. The
    reported pool size (1) then disagrees with the realized trajectory set.
    Characterization: the source trusts the sampler to return roots only.
    """
    convs = [
        ConversationMetadata(
            conversation_id="root",
            turns=[TurnMetadata() for _ in range(3)],
            is_root=True,
        ),
        ConversationMetadata(
            conversation_id="child",
            turns=[TurnMetadata() for _ in range(3)],
            is_root=False,
            agent_depth=1,
            parent_conversation_id="root",
        ),
    ]
    src = _build(convs, ["child", "root"], concurrency=1)
    assert src._pool_size == 1  # only the root counts toward pool_size
    assert src.trajectories[0].conversation_id == "child"  # built on the child


# =============================================================================
# Background-only subagent snapshot: root_next_idx is None
# =============================================================================


def test_background_only_snapshot_keeps_child_but_start_index_defaults_to_zero() -> (
    None
):
    """When the root has no turn at-or-after t_star (``root_next_idx is None``)
    but a background subagent is still live, the snapshot keeps the child and
    the Trajectory.start_turn_index falls back to 0 despite there being NO root
    state. Warmup dispatches via ``session_for_state`` (one per ready snapshot
    state), so the bogus ``start_turn_index`` is inert -- but it is still a
    misleading sentinel. Characterization of trajectory_source.py:513.
    """
    branch_id = "trace:spawn:bg"
    convs = [
        ConversationMetadata(
            conversation_id="trace",
            turns=[
                TurnMetadata(timestamp_ms=0.0),
                TurnMetadata(timestamp_ms=100.0, branch_ids=[branch_id]),
            ],
            branches=[
                ConversationBranchInfo(
                    branch_id=branch_id,
                    child_conversation_ids=["trace::bg"],
                    mode=ConversationBranchMode.SPAWN,
                    is_background=True,
                    start_timestamp_ms=50.0,
                )
            ],
        ),
        ConversationMetadata(
            conversation_id="trace::bg",
            turns=[
                TurnMetadata(timestamp_ms=50.0),
                TurnMetadata(timestamp_ms=5000.0),
            ],
            is_root=False,
            agent_depth=1,
            parent_conversation_id="trace",
        ),
    ]
    # bounds 0..5000; t_star ~200 (ratio 0.04) -> root has no turn >= 200.
    src = _build(convs, ["trace"], start_min_ratio=0.04, start_max_ratio=0.04)
    traj = src.trajectories[0]
    assert traj.snapshot is not None
    cids = {s.conversation_id for s in traj.snapshot.states}
    assert cids == {"trace::bg"}  # only the background child survives
    assert traj.start_turn_index == 0  # sentinel default, no root state
    assert src.warmup_credit_count == 1  # the one ready child state


# =============================================================================
# Empty child_conversation_ids on a joined branch: root not marked waiting
# =============================================================================


def test_join_branch_with_empty_children_leaves_root_not_waiting() -> None:
    """A SPAWN_JOIN prereq whose branch declares zero children does NOT block
    the root: ``pending_join_targets`` is only populated when the branch
    produced child states. With an empty child list the join is vacuous, the
    root proceeds (waiting_on_children False) and is counted as warmup-ready.
    Characterization of the branch-child-gated join accounting.
    """
    branch_id = "trace:spawn:agent_0"
    convs = [
        ConversationMetadata(
            conversation_id="trace",
            turns=[
                TurnMetadata(timestamp_ms=0.0),
                TurnMetadata(timestamp_ms=12000.0, branch_ids=[branch_id]),
                TurnMetadata(
                    timestamp_ms=20000.0,
                    prerequisites=[
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN, branch_id=branch_id
                        )
                    ],
                ),
            ],
            branches=[
                ConversationBranchInfo(
                    branch_id=branch_id,
                    child_conversation_ids=[],  # no children
                    mode=ConversationBranchMode.SPAWN,
                    start_timestamp_ms=13000.0,
                )
            ],
        )
    ]
    src = _build(convs, ["trace"], start_min_ratio=0.675, start_max_ratio=0.675)
    traj = src.trajectories[0]
    assert traj.snapshot is not None
    root = next(s for s in traj.snapshot.states if s.conversation_id == "trace")
    assert root.next_turn_index == 2
    assert root.waiting_on_children is False
    assert src.warmup_credit_count == 1


# =============================================================================
# Missing child metadata is silently dropped from the snapshot
# =============================================================================


def test_branch_child_missing_metadata_is_dropped_root_survives() -> None:
    """A branch referencing a child id with no metadata entry drops that child
    silently (trajectory_source.py:591-593); the root snapshot is still built.

    Probes the filtered/missing-parent-or-child edge: a dangling
    child_conversation_id must not crash and must not invent a phantom state.
    """
    branch_id = "trace:spawn:ghost"
    convs = [
        ConversationMetadata(
            conversation_id="trace",
            turns=[
                TurnMetadata(timestamp_ms=0.0),
                TurnMetadata(timestamp_ms=10000.0, branch_ids=[branch_id]),
                TurnMetadata(timestamp_ms=20000.0),
            ],
            branches=[
                ConversationBranchInfo(
                    branch_id=branch_id,
                    child_conversation_ids=["trace::ghost"],  # no metadata for this id
                    mode=ConversationBranchMode.SPAWN,
                    start_timestamp_ms=11000.0,
                )
            ],
        )
    ]
    src = _build(convs, ["trace"], start_min_ratio=0.5, start_max_ratio=0.5)
    traj = src.trajectories[0]
    assert traj.snapshot is not None
    cids = {s.conversation_id for s in traj.snapshot.states}
    assert cids == {"trace"}  # ghost child dropped, no phantom state


# =============================================================================
# warmup_credit_count must equal exactly what _execute_warmup would dispatch
# =============================================================================


def test_warmup_credit_count_matches_dispatchable_states_mixed_trajectories() -> None:
    """``warmup_credit_count`` must equal one-per-snapshotless-trajectory plus
    one-per-non-waiting snapshot state -- exactly what ``_execute_warmup``
    iterates (agentic_replay.py:223-246). Verified against a mix of a timestamp
    -less lane and a timestamped lane with a gated (waiting) root + ready child.
    """
    branch_id = "trace:spawn:agent_0"
    convs = [
        # timestamp-less lane -> contributes exactly 1
        ConversationMetadata(
            conversation_id="plain",
            turns=[TurnMetadata() for _ in range(5)],
        ),
        # timestamped lane: root gated on join (waiting) + one ready child
        ConversationMetadata(
            conversation_id="trace",
            turns=[
                TurnMetadata(timestamp_ms=0.0),
                TurnMetadata(timestamp_ms=12000.0, branch_ids=[branch_id]),
                TurnMetadata(
                    timestamp_ms=20000.0,
                    prerequisites=[
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN, branch_id=branch_id
                        )
                    ],
                ),
            ],
            branches=[
                ConversationBranchInfo(
                    branch_id=branch_id,
                    child_conversation_ids=["trace::sa:agent_0"],
                    mode=ConversationBranchMode.SPAWN,
                    start_timestamp_ms=13000.0,
                )
            ],
        ),
        ConversationMetadata(
            conversation_id="trace::sa:agent_0",
            turns=[
                TurnMetadata(timestamp_ms=13000.0),
                TurnMetadata(timestamp_ms=14000.0),
                TurnMetadata(timestamp_ms=17000.0),
            ],
            is_root=False,
            agent_depth=1,
            parent_conversation_id="trace",
        ),
    ]
    src = _build(
        convs,
        ["plain", "trace"],
        concurrency=2,
        start_min_ratio=0.675,
        start_max_ratio=0.675,
    )

    # Re-derive the dispatch count exactly as _execute_warmup would.
    dispatchable = 0
    for traj in src.trajectories:
        if traj.snapshot is None:
            dispatchable += 1
        else:
            dispatchable += sum(
                1 for st in traj.snapshot.states if not st.waiting_on_children
            )
    assert src.warmup_credit_count == dispatchable
    # plain(1) + trace child(1); trace root is waiting -> not counted.
    assert dispatchable == 2


# =============================================================================
# start ratio boundary validation
# =============================================================================


def test_equal_start_ratios_accepted_strictly_greater_rejected() -> None:
    """``start_min_ratio == start_max_ratio`` is accepted (deterministic t_star)
    while ``start_min_ratio > start_max_ratio`` raises ValueError."""
    convs = [_ts_conv("c", [0.0, 1000.0, 2000.0, 3000.0])]
    # equal is fine
    src = _build(convs, ["c"], start_min_ratio=0.5, start_max_ratio=0.5)
    assert len(src.trajectories) == 1

    with pytest.raises(ValueError, match="must be <="):
        _build(
            [_ts_conv("c", [0.0, 1000.0, 2000.0, 3000.0])],
            ["c"],
            start_min_ratio=0.8,
            start_max_ratio=0.3,
        )


def test_negative_start_min_ratio_does_not_produce_negative_k_i() -> None:
    """A negative ``start_min_ratio`` widens the candidate range into negative
    indices, but ``_trajectory_start_is_sendable`` rejects ``turn_index < 0``
    so no negative ``start_turn_index`` can be selected. Defensive: probes that
    the sendability guard backstops the unvalidated lower ratio bound.
    """
    convs = [
        ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata() for _ in range(10)],
        )
    ]
    # Sweep seeds; negative candidates are in range(int(-0.5*10), ...) but must
    # never survive the sendability filter.
    for seed in range(40):
        src = TrajectorySource(
            dataset_metadata=_dataset(
                [ConversationMetadata(conversation_id="c", turns=convs[0].turns)]
            ),
            dataset_sampler=_Sampler(["c"]),
            concurrency=1,
            random_seed=seed,
            start_min_ratio=-0.5,
            start_max_ratio=0.7,
        )
        k = src.trajectories[0].start_turn_index
        assert k >= 0, f"negative k_i {k} at seed {seed}"


def test_trace_time_bounds_none_when_no_timestamps_falls_back_to_turn_split() -> None:
    """A trace with all-None timestamps has no time bounds, so the timestamped
    path is skipped and the trajectory uses the ``start_turn_index`` split
    (snapshot is None). Guards the timestamp-less fall-through.
    """
    convs = [
        ConversationMetadata(
            conversation_id="c",
            turns=[TurnMetadata(timestamp_ms=None) for _ in range(6)],
        )
    ]
    src = _build(convs, ["c"], start_min_ratio=0.0, start_max_ratio=0.5)
    traj = src.trajectories[0]
    assert traj.snapshot is None
    assert 0 <= traj.start_turn_index <= 6 - 2
    # sanity: the helper genuinely sees no finite timestamps
    assert not any(
        isinstance(getattr(t, "timestamp_ms", None), int | float)
        and not math.isnan(t.timestamp_ms)
        for t in convs[0].turns
    )
