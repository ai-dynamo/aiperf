# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial unit tests for AgenticReplayStrategy.

These probe accounting and edge-case anomalies NOT covered by the existing
agentic-replay test suites (base, context-overflow, warmup-failure). Each test
probes exactly one thing:

    * recycle-pool trace LEAK when a popped trace yields no spawnable session
      (missing metadata / empty turns) - the popped trace is silently dropped
      while the finished trace is re-enqueued (CONFIRMED BUG, xfail).
    * recycle-pool trace LEAK for an empty-turns trace (same root cause, xfail).
    * _pop_next_eligible_trace busy-loop bound + FIFO order preservation when
      every queued trace is active (characterization).
    * _active_traces self-heal: decrement of an unknown/zero-count trace does not
      leak a negative count (characterization).
    * double-recycle guard raises RuntimeError on a duplicate final-turn
      (characterization - already partly covered, but exercises the post-spawn
      lane-fallback interaction).
    * _release_lane_for missing-entry fallback to lane 0 keeps recycle
      progressing (characterization).
    * num_turns / metadata mismatch: a non-final credit whose num_turns
      overstates real turns raises ValueError from get_next_turn_metadata
      (characterization of the fragility).
    * context-overflow short-circuit on a non-final CHILD turn (agent_depth>0)
      routes to BranchOrchestrator.on_child_stopped and prunes bookkeeping
      (characterization).
    * child overflow cleanup with branch_orchestrator=None still prunes dicts
      (characterization).
    * setup_phase recycle pool excludes non-root (DAG child) conversations
      (characterization).
    * wrap-fill + cache_bust=NONE emits the byte-identical-traffic warning;
      non-NONE suppresses it (characterization).
    * snapshot terminal-root immediate-recycle rotates the cache-bust marker
      pass for the FRESH session while the warmed terminal root keeps pass=0
      (characterization).
"""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import (
    CacheBustTarget,
    ConversationBranchMode,
    CreditPhase,
)
from aiperf.common.models import (
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    ConversationState,
    Trajectory,
    TrajectorySnapshot,
    TrajectorySource,
)

# =============================================================================
# Helpers (mirrors the sibling adversarial suites for parity)
# =============================================================================

_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


def _make_dataset(num_traces: int, turns_per_trace: int) -> DatasetMetadata:
    convs = []
    for i in range(num_traces):
        turns = [
            TurnMetadata(timestamp_ms=None, delay_ms=None)
            for _ in range(turns_per_trace)
        ]
        convs.append(ConversationMetadata(conversation_id=f"trace_{i}", turns=turns))
    return DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )


def _build_real_trajectory_source(
    *,
    dataset: DatasetMetadata,
    trajectories: list[Trajectory],
) -> TrajectorySource:
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = dataset
    src._dataset_sampler = MagicMock()
    src._metadata_lookup = {c.conversation_id: c for c in dataset.conversations}
    src._random_seed = 0
    src._target_size = len(trajectories)
    src.trajectories = list(trajectories)
    return src


def _make_strategy(
    *,
    phase: CreditPhase,
    trajectories: list[Trajectory],
    dataset: DatasetMetadata,
    issuer: AsyncMock | None = None,
    scheduler: MagicMock | None = None,
    stop_checker: MagicMock | None = None,
    user_config: object | None = None,
    branch_orchestrator: object | None = None,
) -> tuple[AgenticReplayStrategy, AsyncMock, MagicMock, MagicMock]:
    src = _build_real_trajectory_source(dataset=dataset, trajectories=trajectories)
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = max(1, len(trajectories))
    issuer = issuer if issuer is not None else AsyncMock()
    scheduler = scheduler if scheduler is not None else MagicMock()
    if stop_checker is None:
        stop_checker = MagicMock()
        stop_checker.can_start_new_session.return_value = True
    strategy = AgenticReplayStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        user_config=user_config,
        branch_orchestrator=branch_orchestrator,
    )
    return strategy, issuer, scheduler, stop_checker


def _make_credit(
    *,
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
    x_correlation_id: str = "xcorr",
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK,
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        agent_depth=agent_depth,
        parent_correlation_id=parent_correlation_id,
        branch_mode=branch_mode,
    )


def _make_user_config(*, target: CacheBustTarget) -> SimpleNamespace:
    return SimpleNamespace(
        input=SimpleNamespace(
            prompt=SimpleNamespace(cache_bust=SimpleNamespace(target=target))
        ),
        benchmark_id="bench-fixed",
    )


def _rid(marker: str | None) -> str | None:
    if marker is None:
        return None
    m = _RID_RE.search(marker)
    return m.group(0) if m else None


def _drain(queue: asyncio.Queue[str]) -> list[str]:
    items: list[str] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


# =============================================================================
# CONFIRMED BUG: recycle-pool trace LEAK on unspawnable popped trace
# =============================================================================


@pytest.mark.asyncio
async def test_recycle_pop_missing_metadata_trace_not_dropped_from_pool() -> None:
    """A popped trace whose session cannot be built must NOT vanish from the pool.

    Invariant: the recycle queue's set of eligible trace_ids is conserved across
    a recycle attempt (a trace temporarily unspawnable should remain available),
    just as the finished trace is re-enqueued. Otherwise a single degenerate
    trace silently erodes pool diversity for the rest of the phase.
    """
    ds = _make_dataset(num_traces=2, turns_per_trace=2)
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
        dataset=ds,
    )
    await strategy.setup_phase()

    # Replace the recycle queue so only the broken trace_1 is eligible to pop,
    # and keep trace_0 perpetually active (cap 2) so the pop loop skips it.
    _drain(strategy._recycle_queue)
    del strategy.conversation_source._metadata_lookup["trace_1"]
    strategy._recycle_queue.put_nowait("trace_1")
    strategy._active_traces["trace_0"] = 2
    strategy._lanes_per_trace["trace_0"] = 2
    strategy._correlation_to_lane["finished"] = 0

    await strategy._spawn_from_recycle_or_id(
        "trace_0", finished_correlation_id="finished"
    )

    remaining = _drain(strategy._recycle_queue)
    # trace_1 was popped, could not spawn, and must still be in the pool.
    assert "trace_1" in remaining, (
        f"unspawnable popped trace was permanently dropped; remaining={remaining}"
    )
    # No new credit issued (nothing spawnable this round).
    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_recycle_pop_empty_turns_trace_not_dropped_from_pool() -> None:
    """A popped trace with zero turns must not be silently dropped from the pool."""
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0", turns=[TurnMetadata(), TurnMetadata()]
            ),
            # Zero-turn trace: build_turn would be unspawnable.
            ConversationMetadata(conversation_id="trace_empty", turns=[]),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
        dataset=ds,
    )
    await strategy.setup_phase()
    _drain(strategy._recycle_queue)
    strategy._recycle_queue.put_nowait("trace_empty")
    strategy._active_traces["trace_0"] = 2
    strategy._lanes_per_trace["trace_0"] = 2
    strategy._correlation_to_lane["finished"] = 0

    await strategy._spawn_from_recycle_or_id(
        "trace_0", finished_correlation_id="finished"
    )

    remaining = _drain(strategy._recycle_queue)
    assert "trace_empty" in remaining, (
        f"zero-turn popped trace permanently dropped; remaining={remaining}"
    )


# =============================================================================
# Characterization: _pop_next_eligible_trace bound + FIFO order
# =============================================================================


@pytest.mark.asyncio
async def test_pop_next_eligible_all_active_returns_none_and_preserves_queue() -> None:
    """When every queued trace is at lane capacity, the pop loop terminates.

    Bounded by the initial qsize (no busy-loop). All skipped traces are
    re-enqueued in FIFO order, so the queue is returned intact.
    """
    ds = _make_dataset(num_traces=3, turns_per_trace=2)
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
        dataset=ds,
    )
    await strategy.setup_phase()
    for tid in ("trace_0", "trace_1", "trace_2"):
        strategy._active_traces[tid] = 1  # default lane cap is 1

    result = strategy._pop_next_eligible_trace()

    assert result is None
    # Order and size preserved (FIFO rotation of skipped traces).
    assert _drain(strategy._recycle_queue) == ["trace_0", "trace_1", "trace_2"]


# =============================================================================
# Characterization: _active_traces self-heal on unknown / zero-count trace
# =============================================================================


@pytest.mark.asyncio
async def test_spawn_unknown_trace_does_not_leak_negative_active_count() -> None:
    """Decrementing _active_traces for a trace that was never tracked self-heals.

    Counter[missing] reads 0, -=1 yields -1, then the <=0 guard deletes the key,
    so no spurious negative count survives. The missing lane entry falls back to
    lane 0, letting recycle progress against the head of the real pool.
    """
    ds = _make_dataset(num_traces=2, turns_per_trace=2)
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
        dataset=ds,
    )
    await strategy.setup_phase()

    await strategy._spawn_from_recycle_or_id(
        "never_tracked", finished_correlation_id="never_seen"
    )

    assert "never_tracked" not in strategy._active_traces
    assert all(count >= 0 for count in strategy._active_traces.values())


# =============================================================================
# Characterization: double-recycle guard + lane-0 fallback interaction
# =============================================================================


@pytest.mark.asyncio
async def test_duplicate_final_turn_for_same_correlation_raises_runtime_error() -> None:
    """Firing handle_credit_return twice for the same final turn raises.

    The guard keys on x_correlation_id, so even though the first recycle popped
    the finished correlation_id out of _correlation_to_lane (and the second call
    falls back to lane 0), the duplicate is still caught by _in_flight_recycled.
    """
    ds = _make_dataset(num_traces=3, turns_per_trace=2)
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
        dataset=ds,
    )
    await strategy.setup_phase()
    strategy._correlation_to_lane["xc1"] = 0
    strategy._active_traces["trace_0"] += 1

    final = _make_credit(
        conversation_id="trace_0", turn_index=1, num_turns=2, x_correlation_id="xc1"
    )
    await strategy.handle_credit_return(final)

    with pytest.raises(RuntimeError, match="Double recycle"):
        await strategy.handle_credit_return(final)


# =============================================================================
# Characterization: num_turns / metadata mismatch fragility
# =============================================================================


@pytest.mark.asyncio
async def test_non_final_credit_overstating_num_turns_raises_value_error() -> None:
    """A non-final credit whose num_turns exceeds the real turn count blows up.

    The strategy trusts credit.num_turns for the is_final_turn decision but
    fetches the next turn from metadata. If they disagree (turn_index+1 is out
    of metadata range while is_final_turn is False), get_next_turn_metadata
    raises an uncaught ValueError. Documents the lack of defensive validation;
    in production credit.num_turns is sourced from the same metadata so this is
    a 'garbage in' fragility rather than a normal-flow bug.
    """
    ds = _make_dataset(num_traces=2, turns_per_trace=3)
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
        dataset=ds,
    )
    await strategy.setup_phase()
    issuer.issue_credit.reset_mock()

    # turn_index=2, num_turns=5 -> not final; but metadata only has 3 turns.
    bogus = _make_credit(conversation_id="trace_0", turn_index=2, num_turns=5)
    with pytest.raises(ValueError, match="No turn 3"):
        await strategy.handle_credit_return(bogus)


# =============================================================================
# Characterization: context-overflow short-circuit on a CHILD turn
# =============================================================================


@pytest.mark.asyncio
async def test_child_non_final_overflow_routes_to_orchestrator_and_prunes() -> None:
    """A non-final CHILD turn with context-overflow stops the child via the
    BranchOrchestrator and prunes its bookkeeping, without recycling (children
    are never root-pool members)."""
    ds = _make_dataset(num_traces=2, turns_per_trace=3)
    branch_orchestrator = MagicMock()
    branch_orchestrator.on_child_stopped = AsyncMock()
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
        dataset=ds,
        branch_orchestrator=branch_orchestrator,
    )
    await strategy.setup_phase()
    strategy._session_marker["child-corr"] = "marker"
    strategy._correlation_to_lane["child-corr"] = 4
    issuer.issue_credit.reset_mock()

    child = _make_credit(
        conversation_id="trace_0::sa:0",
        turn_index=1,  # non-final of 3
        num_turns=3,
        x_correlation_id="child-corr",
        agent_depth=1,
        parent_correlation_id="parent",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    await strategy.handle_credit_return(
        child, error="This model's maximum context length is 131072 tokens"
    )

    branch_orchestrator.on_child_stopped.assert_awaited_once_with("child-corr")
    assert "child-corr" not in strategy._session_marker
    assert "child-corr" not in strategy._correlation_to_lane
    # No recycle: the child trace_id must not enter the root recycle pool.
    assert issuer.issue_credit.await_count == 0


@pytest.mark.asyncio
async def test_child_overflow_with_no_orchestrator_still_prunes_bookkeeping() -> None:
    """branch_orchestrator=None: child overflow short-circuit must still prune
    the marker/lane dicts (no AttributeError from a None orchestrator)."""
    ds = _make_dataset(num_traces=2, turns_per_trace=3)
    strategy, issuer, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="trace_0", start_turn_index=0)],
        dataset=ds,
        branch_orchestrator=None,
    )
    await strategy.setup_phase()
    strategy._session_marker["child-corr"] = "marker"
    strategy._correlation_to_lane["child-corr"] = 4

    child = _make_credit(
        conversation_id="trace_0::sa:0",
        turn_index=1,
        num_turns=3,
        x_correlation_id="child-corr",
        agent_depth=1,
        parent_correlation_id="parent",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    await strategy.handle_credit_return(
        child, error="This model's maximum context length is 131072 tokens"
    )

    assert "child-corr" not in strategy._session_marker
    assert "child-corr" not in strategy._correlation_to_lane


# =============================================================================
# Characterization: recycle pool excludes non-root (DAG child) conversations
# =============================================================================


@pytest.mark.asyncio
async def test_setup_recycle_pool_excludes_non_root_children() -> None:
    """The PROFILING recycle pool draws only is_root conversations.

    DAG child conversations (is_root=False) must never enter the recycle queue;
    they are reachable only via their parent's branches, and spawning a fresh
    root session from a child trace_id would replay a partial context.
    """
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="root_0", turns=[TurnMetadata(), TurnMetadata()]
            ),
            ConversationMetadata(
                conversation_id="root_0::sa",
                turns=[TurnMetadata(), TurnMetadata()],
                is_root=False,
                agent_depth=1,
                parent_conversation_id="root_0",
            ),
            ConversationMetadata(
                conversation_id="root_1", turns=[TurnMetadata(), TurnMetadata()]
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[Trajectory(conversation_id="root_0", start_turn_index=0)],
        dataset=ds,
    )
    await strategy.setup_phase()

    queued = _drain(strategy._recycle_queue)
    assert queued == ["root_0", "root_1"]
    assert "root_0::sa" not in queued


# =============================================================================
# Characterization: wrap-fill + cache_bust=NONE warning coherence
# =============================================================================


def test_wrap_fill_with_cache_bust_none_warns_about_identical_traffic() -> None:
    """Wrap-fill (>1 lanes per trace) + cache_bust=NONE warns; non-NONE doesn't.

    Byte-identical per-lane traffic across shared-trace lanes is a real
    measurement hazard, so the constructor emits a loud heads-up only when the
    feature is off. A non-NONE target makes per-lane traffic distinct, so the
    warning must be suppressed.
    """
    ds = _make_dataset(num_traces=1, turns_per_trace=3)
    wrap_fill = [
        Trajectory(conversation_id="trace_0", start_turn_index=0),
        Trajectory(conversation_id="trace_0", start_turn_index=1),
    ]

    none_calls: list[tuple] = []

    class SpyNone(AgenticReplayStrategy):
        def warning(self, *args, **kwargs) -> None:  # type: ignore[override]
            none_calls.append(args)

    SpyNone(
        config=SimpleNamespace(phase=CreditPhase.WARMUP, concurrency=2),
        conversation_source=_build_real_trajectory_source(
            dataset=ds, trajectories=wrap_fill
        ),
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=AsyncMock(),
        lifecycle=MagicMock(),
        user_config=_make_user_config(target=CacheBustTarget.NONE),
    )
    assert len(none_calls) == 1, "wrap-fill + cache_bust=NONE must warn exactly once"

    nonnone_calls: list[tuple] = []

    class SpyNonNone(AgenticReplayStrategy):
        def warning(self, *args, **kwargs) -> None:  # type: ignore[override]
            nonnone_calls.append(args)

    SpyNonNone(
        config=SimpleNamespace(phase=CreditPhase.WARMUP, concurrency=2),
        conversation_source=_build_real_trajectory_source(
            dataset=ds, trajectories=wrap_fill
        ),
        scheduler=MagicMock(),
        stop_checker=MagicMock(),
        credit_issuer=AsyncMock(),
        lifecycle=MagicMock(),
        user_config=_make_user_config(target=CacheBustTarget.SYSTEM_PREFIX),
    )
    assert nonnone_calls == [], "non-NONE target must suppress the wrap-fill warning"


# =============================================================================
# Characterization: snapshot terminal-root immediate recycle rotates marker
# =============================================================================


@pytest.mark.asyncio
async def test_snapshot_terminal_root_recycle_rotates_marker_for_fresh_session() -> (
    None
):
    """A snapshot whose warmed root is already at its last turn recycles
    immediately in PROFILING; the FRESH recycled session gets a rotated marker
    (recycle_pass advances), while the terminal root retained its warmup pass=0.
    """
    ds = DatasetMetadata(
        conversations=[
            ConversationMetadata(
                conversation_id="trace_0", turns=[TurnMetadata(timestamp_ms=0.0)]
            ),
        ],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    root_state = ConversationState(
        conversation_id="trace_0",
        x_correlation_id="warmed-root",
        next_turn_index=0,
    )
    trajectory = Trajectory(
        conversation_id="trace_0",
        start_turn_index=0,
        snapshot=TrajectorySnapshot(t_star_ms=0.0, states=(root_state,)),
    )
    issued: list[Credit] = []

    async def capture(turn):
        issued.append(turn)
        return True

    issuer = AsyncMock()
    issuer.issue_credit.side_effect = capture
    strategy, _, _, _ = _make_strategy(
        phase=CreditPhase.PROFILING,
        trajectories=[trajectory],
        dataset=ds,
        issuer=issuer,
        user_config=_make_user_config(target=CacheBustTarget.SYSTEM_PREFIX),
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    assert len(issued) == 1
    fresh = issued[0]
    # The dispatched session is the recycled one (fresh uuid, not the warmed id).
    assert fresh.x_correlation_id != "warmed-root"
    # recycle_pass advanced past the warmed root's pass=0.
    assert strategy._recycle_pass["trace_0"] == 1
    assert _rid(fresh.cache_bust_marker) is not None
