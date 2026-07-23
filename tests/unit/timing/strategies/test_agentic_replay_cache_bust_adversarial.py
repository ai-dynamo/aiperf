# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial coverage for AgenticReplayStrategy cache-bust state lifecycle."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode, CreditPhase
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import Trajectory, TrajectorySource
from tests.unit.timing.strategies._shared_helpers import _make_dataset, _make_run

# Helpers (mirror test_agentic_replay.py)


def _build_real_trajectory_source(
    num_traces: int,
    turns_per_trace: int,
    trajectories: list[Trajectory],
) -> TrajectorySource:
    ds = _make_dataset(num_traces, turns_per_trace)
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = ds
    _roots = [
        c.conversation_id
        for c in src._dataset_metadata.conversations
        if getattr(c, "is_root", True)
    ]
    src._dataset_sampler = SequentialSampler(_roots) if _roots else MagicMock()
    src._pool_size = len(_roots)
    src._metadata_lookup = {c.conversation_id: c for c in ds.conversations}
    src._random_seed = 0
    src._target_size = len(trajectories)
    src.trajectories = list(trajectories)
    return src


def _make_strategy(
    *,
    phase: CreditPhase,
    trajectories: list[Trajectory],
    num_traces: int = 5,
    turns_per_trace: int = 4,
    issuer: AsyncMock | None = None,
    scheduler: MagicMock | None = None,
    run: object | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock, TrajectorySource]:
    src = _build_real_trajectory_source(num_traces, turns_per_trace, trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(trajectories)
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        run=run,
    )
    return strategy, issuer, scheduler, src


def _make_credit(
    *,
    conversation_id: str,
    x_correlation_id: str = "xcorr",
    turn_index: int,
    num_turns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
    cache_bust_marker: str | None = None,
    cache_bust_target: CacheBustTarget = CacheBustTarget.NONE,
    parent_correlation_id: str | None = None,
    agent_depth: int = 0,
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.FORK,
        cache_bust_marker=cache_bust_marker,
        cache_bust_target=cache_bust_target,
        parent_correlation_id=parent_correlation_id,
        agent_depth=agent_depth,
    )


# Cache-bust disabled (user_config is None)


def test_cache_bust_disabled_when_user_config_is_none():
    """No user_config -> target defaults to NONE and benchmark_id to "unknown". Construction stays cheap (no marker minting at __init__)."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    strategy, *_ = _make_strategy(
        phase=CreditPhase.WARMUP,
        trajectories=trajectories,
        run=None,
    )

    assert strategy._cache_bust_target == CacheBustTarget.NONE
    assert strategy._benchmark_id == "unknown"
    # No sessions seeded yet -> marker dict is empty.
    assert strategy._session_marker == {}


# _recycle_pass dict bounded by pool size


@pytest.mark.asyncio
async def test_recycle_pass_dict_grows_only_to_pool_size():
    """Recycling N traces twice each must NOT inflate _recycle_pass beyond the pool size — the dict is keyed by trace_id, not by recycle event."""
    n = 3
    trajectories = [
        Trajectory(conversation_id=f"trace_{i}", start_turn_index=0) for i in range(n)
    ]
    run = _make_run(target=CacheBustTarget.SYSTEM_PREFIX)

    issued_turns: list = []

    async def capture(turn):
        issued_turns.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture

    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=n,  # all trajectories consume the pool
        turns_per_trace=2,
        issuer=issuer,
        run=run,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()

    # Each trace ends -> recycled via the sampler. Drive two full passes.
    # Only finalize turns we have not yet finalized: every recycle spawns a
    # NEW credit with a fresh correlation_id, and the double-recycle guard
    # (Task 5: keyed on correlation_id) raises if we replay an already-final
    # correlation_id.
    finalized: set[str] = set()
    for _round in range(2):
        pending = [t for t in issued_turns if t.x_correlation_id not in finalized]
        for turn in pending:
            final_credit = _make_credit(
                conversation_id=turn.conversation_id,
                x_correlation_id=turn.x_correlation_id,
                turn_index=turn.num_turns - 1,
                num_turns=turn.num_turns,
            )
            await strategy.handle_credit_return(final_credit)
            finalized.add(turn.x_correlation_id)

    # _recycle_pass entries are bounded by the trace pool (one entry per
    # trace_id), regardless of how many recycle events fired.
    assert len(strategy._recycle_pass) <= n
    assert set(strategy._recycle_pass.keys()) <= {f"trace_{i}" for i in range(n)}


# Marker dict pruned on queue-empty recycle (complement to stop-checker test)


@pytest.mark.asyncio
async def test_session_marker_dict_pruned_on_recycle():
    """``_spawn_from_recycle_or_id`` prunes the finished session's bookkeeping up front, before any later branch (cooldown, empty pool) can short-circuit."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    run = _make_run(target=CacheBustTarget.SYSTEM_PREFIX)
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        turns_per_trace=2,
        run=run,
    )
    await strategy.setup_phase()

    # Seed in-flight bookkeeping for a finished session.
    finished_corr = "xcorr-finished"
    strategy._correlation_to_lane[finished_corr] = 0
    strategy._session_marker[finished_corr] = "[rid:dummy]"

    await strategy._spawn_from_recycle_or_id(
        "trace_0", finished_correlation_id=finished_corr
    )

    # Pruning fires before the queue-None early return.
    assert finished_corr not in strategy._session_marker
    assert finished_corr not in strategy._correlation_to_lane


@pytest.mark.asyncio
async def test_session_marker_dict_pruned_on_metadata_miss_recycle():
    """If ``_build_session_for_trace`` cannot resolve the next trace (metadata missing in the lookup) the spawn returns early. The finished session's bookkeeping must still be pruned because the prune happens up front in ``_spawn_from_recycle_or_id``, before any later branch can short-circuit."""
    trajectories = [Trajectory(conversation_id="trace_0", start_turn_index=0)]
    run = _make_run(target=CacheBustTarget.SYSTEM_PREFIX)
    strategy, issuer, _, src = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=trajectories,
        num_traces=2,
        turns_per_trace=2,
        run=run,
    )
    await strategy.setup_phase()

    # Force a metadata-lookup miss for every recycled trace_id.
    src._metadata_lookup = {}

    finished_corr = "xcorr-finished"
    strategy._correlation_to_lane[finished_corr] = 0
    strategy._session_marker[finished_corr] = "[rid:dummy]"

    issuer.issue_credit.reset_mock()
    await strategy._spawn_from_recycle_or_id(
        "trace_0", finished_correlation_id=finished_corr
    )

    # No new credit dispatched (metadata miss returns early after pop).
    assert issuer.issue_credit.await_count == 0
    # But pruning fired before the early return.
    assert finished_corr not in strategy._session_marker
    assert finished_corr not in strategy._correlation_to_lane


# from_previous_credit cache-bust propagation


def test_marker_propagates_through_from_previous_credit_within_session():
    """``TurnToSend.from_previous_credit`` carries cache_bust_marker / cache_bust_target verbatim from the previous credit to the next-turn descriptor — this is the strategy-side seam that keeps the same marker on every turn of a session."""
    credit = _make_credit(
        conversation_id="trace_0",
        x_correlation_id="xc-0",
        turn_index=0,
        num_turns=3,
        cache_bust_marker="[rid:abcdef012345]\n\n",
        cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
    )

    next_turn = TurnToSend.from_previous_credit(credit)

    assert next_turn.cache_bust_marker == "[rid:abcdef012345]\n\n"
    assert next_turn.cache_bust_target == CacheBustTarget.SYSTEM_PREFIX
    assert next_turn.turn_index == 1
    assert next_turn.x_correlation_id == "xc-0"


def test_subagent_fork_inherits_parent_marker_via_from_previous_credit():
    """A DAG fork is constructed from a parent credit through the same ``from_previous_credit`` seam: the child credit's marker matches the parent's marker, and ``parent_correlation_id`` is preserved."""
    parent = _make_credit(
        conversation_id="trace_0",
        x_correlation_id="xc-parent",
        turn_index=2,
        num_turns=4,
        cache_bust_marker="[rid:parent_marker]\n\n",
        cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
        parent_correlation_id="xc-grandparent",
        agent_depth=1,
    )

    fork = TurnToSend.from_previous_credit(parent)

    assert fork.cache_bust_marker == parent.cache_bust_marker
    assert fork.cache_bust_target == parent.cache_bust_target
    assert fork.parent_correlation_id == "xc-grandparent"
    assert fork.agent_depth == 1
