# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pathological / adversarial unit tests for AgenticReplayStrategy."""

from __future__ import annotations

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
from aiperf.dataset.dataset_samplers import SequentialSampler
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import (
    ConversationState,
    Trajectory,
    TrajectorySnapshot,
    TrajectorySource,
)
from tests.unit.timing.strategies._shared_helpers import _make_dataset, _make_run

# Helpers (mirrors the sibling adversarial suites for parity)

_RID_RE = re.compile(r"\[rid:[0-9a-f]{12}\]")


def _build_real_trajectory_source(
    *,
    dataset: DatasetMetadata,
    trajectories: list[Trajectory],
) -> TrajectorySource:
    src = TrajectorySource.__new__(TrajectorySource)
    src._dataset_metadata = dataset
    _roots = [
        c.conversation_id
        for c in src._dataset_metadata.conversations
        if getattr(c, "is_root", True)
    ]
    src._dataset_sampler = SequentialSampler(_roots) if _roots else MagicMock()
    src._pool_size = len(_roots)
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
    run: object | None = None,
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
        run=run,
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


def _rid(marker: str | None) -> str | None:
    if marker is None:
        return None
    m = _RID_RE.search(marker)
    return m.group(0) if m else None


# Characterization: double-recycle guard + lane-0 fallback interaction


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

    final = _make_credit(
        conversation_id="trace_0", turn_index=1, num_turns=2, x_correlation_id="xc1"
    )
    await strategy.handle_credit_return(final)

    with pytest.raises(RuntimeError, match="Double recycle"):
        await strategy.handle_credit_return(final)


# Characterization: num_turns / metadata mismatch fragility


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


# Characterization: context-overflow short-circuit on a CHILD turn


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


# Characterization: recycle pool excludes non-root (DAG child) conversations


@pytest.mark.asyncio
async def test_recycle_excludes_non_root_children() -> None:
    """Recycle draws only is_root conversations from the sampler.

    DAG child conversations (is_root=False) must never be spawned as fresh
    roots; they are reachable only via their parent's branches, and a fresh
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

    seen = {
        strategy.conversation_source.next_recycle_conversation_id() for _ in range(6)
    }
    assert seen == {"root_0", "root_1"}
    assert "root_0::sa" not in seen


# Characterization: wrap-fill + cache_bust=NONE warning coherence


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
        run=_make_run(target=CacheBustTarget.NONE),
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
        run=_make_run(target=CacheBustTarget.SYSTEM_PREFIX),
    )
    assert nonnone_calls == [], "non-NONE target must suppress the wrap-fill warning"


# Characterization: snapshot terminal-root immediate recycle rotates marker


@pytest.mark.asyncio
async def test_snapshot_single_turn_root_profiles_own_turn_zero_with_marker() -> None:
    """A single-turn root sampled at t* == its turn-0 timestamp (n == 0) has
    nothing to warm, so PROFILING measures its own turn 0 rather than
    recycling at startup. The dispatched credit keeps the snapshot's own
    correlation id and carries a minted cache-bust marker; recycle (and its
    marker rotation) happens only later on the turn's completion.
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
        x_correlation_id="snap-root",
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
        run=_make_run(target=CacheBustTarget.SYSTEM_PREFIX),
    )

    await strategy.setup_phase()
    await strategy.execute_phase()

    assert len(issued) == 1
    profiled = issued[0]
    # The snapshot's own session profiles turn 0 (no startup recycle).
    assert profiled.x_correlation_id == "snap-root"
    assert profiled.turn_index == 0
    # No recycle yet -> pass counter has not advanced for a fresh session.
    assert strategy._recycle_pass.get("trace_0", 0) == 0
    assert _rid(profiled.cache_bust_marker) is not None
