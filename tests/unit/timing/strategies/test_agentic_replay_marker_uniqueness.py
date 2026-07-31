# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy-level marker-uniqueness coverage for AgenticReplayStrategy."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import Trajectory, TrajectorySource
from tests.unit.timing.strategies._shared_helpers import _make_dataset

_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


# Harness (mirrors test_agentic_replay.py)


def _build_real_trajectory_source(
    num_traces: int,
    turns_per_trace: int,
    trajectories: list[Trajectory],
) -> TrajectorySource:
    ds = _make_dataset(num_traces, turns_per_trace)
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src._random_seed = 0
    src._target_size = len(trajectories)
    src.trajectories = list(trajectories)
    return src


def _make_run(*, target: CacheBustTarget, benchmark_id: str = "bench-uniqueness"):
    """Build a v2 ``BenchmarkRun`` exposing the values the strategy reads."""
    from aiperf.config import BenchmarkConfig, BenchmarkRun

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "type": "completions",
                "urls": ["http://localhost:8000/v1"],
                "streaming": False,
            },
            "datasets": [
                {
                    "name": "default",
                    "type": "synthetic",
                    "prompts": {"cache_bust": {"target": target}},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "concurrency": 1,
                    "requests": 1,
                }
            ],
        }
    )
    return BenchmarkRun(
        benchmark_id=benchmark_id,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )


def _make_strategy(
    *,
    phase: CreditPhase,
    trajectories: list[Trajectory],
    num_traces: int,
    turns_per_trace: int = 4,
    run: object | None = None,
) -> AgenticReplayStrategy:
    src = _build_real_trajectory_source(num_traces, turns_per_trace, trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(trajectories)
    return AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=AsyncMock(),
        lifecycle=MagicMock(),
        run=run,
    )


def _extract_rid(marker: str | None) -> str | None:
    if marker is None:
        return None
    m = _RID_RE.search(marker)
    return m.group(0) if m else None


# Tests


def test_mint_produces_unique_markers_across_many_recycles():
    """20 lanes warmup + 50 recycles per trace => 1020 unique markers."""
    num_lanes = 20
    num_recycles = 50
    run = _make_run(target=CacheBustTarget.SYSTEM_PREFIX)

    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0)
        for i in range(num_lanes)
    ]
    strategy = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=num_lanes,
        run=run,
    )

    rids: list[str] = []

    # (a) WARMUP-equivalent mint per lane.
    for lane in range(num_lanes):
        marker = strategy._mint_marker_for_session(
            root_correlation_id=f"warmup_{lane}",
            conversation_id=f"trace_{lane}",
            trajectory_index=lane,
        )
        rid = _extract_rid(marker)
        assert rid is not None
        rids.append(rid)

    # (b) Recycle each trace 50 times into the same lane.
    for lane in range(num_lanes):
        for recycle in range(num_recycles):
            marker = strategy._mint_marker_for_session(
                root_correlation_id=f"recycle_{lane}_{recycle}",
                conversation_id=f"trace_{lane}",
                trajectory_index=lane,
            )
            rid = _extract_rid(marker)
            assert rid is not None
            rids.append(rid)

    expected = num_lanes + num_lanes * num_recycles  # 20 + 1000
    assert len(rids) == expected
    assert len(set(rids)) == expected, (
        f"Expected {expected} distinct rids across {num_lanes} lanes x "
        f"({num_recycles} recycles + 1 warmup); got {len(set(rids))} "
        f"({expected - len(set(rids))} collisions)"
    )


def test_recycle_continuity_within_trace_after_trace_id_addition():
    """Same trace, same lane, 100 sequential recycles -> 100 distinct rids."""
    run = _make_run(target=CacheBustTarget.SYSTEM_PREFIX)
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=1,
        run=run,
    )

    rids: list[str] = []
    for i in range(100):
        marker = strategy._mint_marker_for_session(
            root_correlation_id=f"x_{i}",
            conversation_id="trace_0",
            trajectory_index=0,
        )
        rid = _extract_rid(marker)
        assert rid is not None
        rids.append(rid)

    assert len(set(rids)) == 100, (
        f"Same trace + lane + 100 recycle passes must rotate digest each pass; "
        f"got {len(set(rids))} distinct"
    )


def test_warmup_marker_matches_first_profile_marker_after_fix():
    """Intra-session continuity invariant survives the trace_id addition."""
    run = _make_run(target=CacheBustTarget.SYSTEM_PREFIX)
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=2)]

    warmup_strategy = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        num_traces=3,
        turns_per_trace=5,
        run=run,
    )
    warmup_marker = warmup_strategy._mint_marker_for_session(
        root_correlation_id="xcorr-warmup",
        conversation_id="trace_0",
        trajectory_index=0,
    )

    profile_strategy = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=3,
        turns_per_trace=5,
        run=run,
    )
    profile_marker = profile_strategy._mint_marker_for_session(
        root_correlation_id="xcorr-profile",
        conversation_id="trace_0",
        trajectory_index=0,
    )

    assert _extract_rid(warmup_marker) is not None
    assert _extract_rid(warmup_marker) == _extract_rid(profile_marker), (
        "Same (benchmark_id, recycle_pass=0, lane=0, trace_id) across phases "
        "must yield the same rid -- continuity is the contract."
    )


def test_target_none_no_minting_at_scale():
    """At target=NONE, 1000 mint calls yield no real markers and bounded state."""
    run = _make_run(target=CacheBustTarget.NONE)
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=1,
        run=run,
    )

    for i in range(1000):
        result = strategy._mint_marker_for_session(
            root_correlation_id=f"x_{i}",
            conversation_id=f"trace_{i % 10}",
            trajectory_index=i % 5,
        )
        assert result is None

    # _session_marker carries one None entry per xcorr, never a real digest.
    assert len(strategy._session_marker) == 1000
    assert all(v is None for v in strategy._session_marker.values())
    # _recycle_pass is bounded at 0 (no dict writes under NONE).
    assert strategy._recycle_pass == {}


def test_descendant_turn_build_reads_marker_by_root():
    """Regression (found on a live run): the turn-builder must read the cache-bust marker by the session's TREE ROOT (root_correlation_id), not its own x_correlation_id. A descendant session whose root marker is in the ledger must carry that marker -- not None."""
    from aiperf.timing.conversation_source import SampledSession

    run = _make_run(target=CacheBustTarget.FIRST_TURN_PREFIX)
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=1,
        turns_per_trace=4,
        run=run,
    )
    root_marker = "[rid:deadbeefcafe]\n\n"
    strategy._session_marker["ROOT-CORR"] = root_marker
    meta = strategy.conversation_source._metadata_lookup["trace_0"]
    child = SampledSession(
        conversation_id="trace_0",
        metadata=meta,
        x_correlation_id="child-xyz",
        agent_depth=1,
        root_correlation_id="ROOT-CORR",
    )

    turn = strategy._build_turn_for_session(child, 0)
    assert turn.cache_bust_marker == root_marker, (
        "descendant turn must carry the tree-root marker, "
        f"got {turn.cache_bust_marker!r}"
    )
