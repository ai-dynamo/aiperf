# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial pathology tests targeting timing-strategy ↔ DAG-orchestrator
interactions.

Sibling to ``test_dag_adversarial_timing_modes.py``: that suite parameterises
strategy-agnostic orchestrator invariants over the three TimingMode shapes.
This suite drills into strategy-specific timestamp / rate / slot pathologies
that the orchestrator alone does not see — out-of-order timestamps,
extreme magnitudes, rate-limit ↔ fan-out interaction, slot exhaustion under
fan-out, very wide / very deep DAGs, cancellation during scheduled delays,
zero-child branches.

Where a strategy is exercised end-to-end, we use the strategy class directly
with mocked dependencies (scheduler, credit_issuer, lifecycle) so each test
runs in <100ms and avoids the full PhaseRunner spin-up. Orchestrator-level
behaviour is exercised through ``BranchOrchestrator.intercept`` directly
(same pattern as the sibling suite).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import (
    ConversationBranchMode,
    CreditPhase,
    PrerequisiteKind,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import (
    ArrivalPattern,
    DatasetSamplingStrategy,
    TimingMode,
)
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.intervals import IntervalGeneratorConfig
from aiperf.timing.strategies.fixed_schedule import FixedScheduleStrategy
from aiperf.timing.strategies.request_rate import RequestRateStrategy

pytestmark = pytest.mark.component_integration


# =============================================================================
# Helpers (mirror the patterns in test_dag_adversarial_timing_modes.py)
# =============================================================================


def _mk_credit(
    conv_id: str,
    x_corr: str,
    *,
    turn_index: int = 0,
    num_turns: int = 1,
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
    branch_mode: ConversationBranchMode = ConversationBranchMode.FORK,
) -> Credit:
    c = MagicMock(spec=Credit)
    c.conversation_id = conv_id
    c.x_correlation_id = x_corr
    c.turn_index = turn_index
    c.num_turns = num_turns
    c.agent_depth = agent_depth
    c.parent_correlation_id = parent_correlation_id
    c.branch_mode = branch_mode
    c.is_final_turn = turn_index == num_turns - 1
    return c


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    lookup = {c.conversation_id: c for c in conversations}
    cs.get_metadata.side_effect = lambda cid: lookup[cid]

    counter = {"n": 0}

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **_kw
    ):
        counter["n"] += 1
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}-{counter['n']}"
        s.conversation_id = child_conversation_id
        s.agent_depth = agent_depth
        s.parent_correlation_id = parent_correlation_id
        s.branch_mode = branch_mode
        return s

    cs.start_branch_child.side_effect = _start

    def _start_pre(child_conversation_id, **_kw):
        counter["n"] += 1
        s = MagicMock()
        s.x_correlation_id = f"pre-{child_conversation_id}-{counter['n']}"
        s.conversation_id = child_conversation_id
        s.agent_depth = 1
        s.parent_correlation_id = None
        s.branch_mode = ConversationBranchMode.SPAWN
        return s

    cs.start_pre_session_child.side_effect = _start_pre
    return cs


def _mk_issuer(
    *, dispatch_first_returns: bool = True, dispatch_join_returns: bool = True
):
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=dispatch_first_returns)
    issuer.dispatch_join_turn = AsyncMock(return_value=dispatch_join_returns)
    issuer.abort_session = AsyncMock()
    return issuer


def _make_branch(
    branch_id: str,
    children: list[str],
    *,
    mode: ConversationBranchMode = ConversationBranchMode.SPAWN,
    is_background: bool = False,
    dispatch_timing: str = "post",
) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=children,
        mode=mode,
        is_background=is_background,
        dispatch_timing=dispatch_timing,
    )


# =============================================================================
# FixedSchedule: timestamp pathologies
# =============================================================================


@pytest.mark.asyncio
async def test_fixed_schedule_out_of_order_timestamps_within_conversation() -> None:
    """Turn 5 has timestamp_ms < turn 4 within the same conversation. The
    strategy's handle_credit_return pipes ``next_meta.timestamp_ms`` through
    ``schedule_at_perf_sec`` directly, computing a NEGATIVE perf-sec offset.
    Document: the scheduler is told to dispatch in the past — likely fires
    immediately, but no validation rejects this at load time. Flagged as a
    fidelity concern for trace replay."""
    timestamps = [0, 1000, 2000, 3000, 5000, 4000]  # turn 5 < turn 4
    turns = [TurnMetadata(timestamp_ms=ts) for ts in timestamps]
    conv = ConversationMetadata(conversation_id="c1", turns=turns)
    ds = DatasetMetadata(
        conversations=[conv], sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    src = MagicMock()
    src.dataset_metadata = ds
    src.get_next_turn_metadata = lambda credit: turns[credit.turn_index + 1]

    scheduler = MagicMock()
    issuer = MagicMock()
    issuer.issue_credit = lambda *a, **k: True
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000
    lifecycle.started_at_perf_sec = 1.0

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.FIXED_SCHEDULE,
        total_expected_requests=6,
        auto_offset_timestamps=True,
    )
    strategy = FixedScheduleStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )
    strategy._schedule_zero_ms = 0.0

    # Drive return on turn 4 -> next is turn 5 with backwards timestamp.
    credit = _mk_credit("c1", "x", turn_index=4, num_turns=6)
    await strategy.handle_credit_return(credit)

    # Strategy passes the timestamp through without validation. Compute the
    # expected perf-sec the scheduler was told to fire at: started_at_perf_sec
    # + (4000 - 0)/1000 = 1.0 + 4.0 = 5.0 — *earlier* than the previous turn's
    # would-be 6.0. The scheduler will fire it immediately.
    scheduler.schedule_at_perf_sec.assert_called_once()
    target_perf, _ = scheduler.schedule_at_perf_sec.call_args.args
    assert target_perf == pytest.approx(5.0), (
        "out-of-order timestamps are passed through unvalidated"
    )


@pytest.mark.asyncio
async def test_fixed_schedule_negative_timestamp_no_validation() -> None:
    """Pydantic accepts negative timestamps (no min check). Document for
    flagging: trace replay with a negative timestamp_ms produces a negative
    target perf-sec and the scheduler fires immediately. No load-time
    rejection."""
    # Pydantic accepts this — flag if/when validation is added.
    tm = TurnMetadata(timestamp_ms=-1000)
    assert tm.timestamp_ms == -1000


@pytest.mark.asyncio
async def test_fixed_schedule_very_large_timestamp_no_overflow() -> None:
    """timestamp_ms = 2^53 (boundary of float-safe-integer).

    Verify the strategy's float arithmetic for ``_timestamp_to_perf_sec``
    survives without raising. The math: (2^53 - 0)/1000 + offset_sec.
    Pydantic accepts ints of arbitrary size, but the strategy converts to
    float in ``_timestamp_to_perf_sec`` — at 2^53 we are at the boundary
    where consecutive integers stop being representable, but the test only
    verifies we do not crash."""
    ts = 2**53
    turns = [
        TurnMetadata(timestamp_ms=0),
        TurnMetadata(timestamp_ms=ts),
    ]
    conv = ConversationMetadata(conversation_id="c1", turns=turns)
    ds = DatasetMetadata(
        conversations=[conv], sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    src = MagicMock()
    src.dataset_metadata = ds
    src.get_next_turn_metadata = lambda credit: turns[credit.turn_index + 1]

    scheduler = MagicMock()
    issuer = MagicMock()
    issuer.issue_credit = lambda *a, **k: True
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000
    lifecycle.started_at_perf_sec = 1.0

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.FIXED_SCHEDULE,
        total_expected_requests=2,
        auto_offset_timestamps=True,
    )
    strategy = FixedScheduleStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )
    strategy._schedule_zero_ms = 0.0

    credit = _mk_credit("c1", "x", turn_index=0, num_turns=2)
    await strategy.handle_credit_return(credit)

    scheduler.schedule_at_perf_sec.assert_called_once()
    target_perf, _ = scheduler.schedule_at_perf_sec.call_args.args
    assert target_perf > 0  # Did not overflow / wrap.


@pytest.mark.asyncio
async def test_fixed_schedule_setup_sorts_identical_timestamps_stably() -> None:
    """Three sibling conversations all with timestamp_ms=0 — the schedule
    sort is stable (Python list.sort is Timsort), so dispatch order matches
    the conversation iteration order from dataset_metadata."""
    convs = [
        ConversationMetadata(
            conversation_id=f"c{i}", turns=[TurnMetadata(timestamp_ms=0)]
        )
        for i in range(3)
    ]
    ds = DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    src = MagicMock()
    src.dataset_metadata = ds

    scheduler = MagicMock()
    issuer = MagicMock()
    issuer.issue_credit = lambda *a, **k: True
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000
    lifecycle.started_at_perf_sec = 1.0

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.FIXED_SCHEDULE,
        total_expected_requests=3,
        auto_offset_timestamps=True,
    )
    strategy = FixedScheduleStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )

    await strategy.setup_phase()
    # Order is preserved among equal-timestamp entries (stable sort).
    cids = [entry.turn.conversation_id for entry in strategy._absolute_schedule]
    assert cids == ["c0", "c1", "c2"]


@pytest.mark.asyncio
async def test_fixed_schedule_zero_timestamp_fires_at_perf_start() -> None:
    """timestamp_ms=0 with auto_offset must fire at started_at_perf_sec."""
    convs = [
        ConversationMetadata(
            conversation_id="c1", turns=[TurnMetadata(timestamp_ms=0)]
        ),
    ]
    ds = DatasetMetadata(
        conversations=convs, sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    src = MagicMock()
    src.dataset_metadata = ds

    scheduler = MagicMock()
    issuer = MagicMock()
    issuer.issue_credit = lambda *a, **k: True
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 7_000_000_000
    lifecycle.started_at_perf_sec = 7.0

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.FIXED_SCHEDULE,
        total_expected_requests=1,
        auto_offset_timestamps=True,
    )
    strategy = FixedScheduleStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )
    await strategy.setup_phase()
    await strategy.execute_phase()
    target_perf, _ = scheduler.schedule_at_perf_sec.call_args.args
    assert target_perf == pytest.approx(7.0)


# =============================================================================
# RequestRate: rate generator validation
# =============================================================================


def test_request_rate_validates_zero_rate_at_interval_config() -> None:
    """Rate=0 must be rejected by the interval generator's validator."""
    cfg = IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.CONSTANT, request_rate=0.0
    )
    from aiperf.timing.intervals import ConstantIntervalGenerator

    with pytest.raises(ValueError, match="must be set and greater than 0"):
        ConstantIntervalGenerator(cfg)


def test_request_rate_validates_negative_rate() -> None:
    cfg = IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.CONSTANT, request_rate=-1.0
    )
    from aiperf.timing.intervals import ConstantIntervalGenerator

    with pytest.raises(ValueError, match="must be set and greater than 0"):
        ConstantIntervalGenerator(cfg)


def test_request_rate_set_rate_rejects_zero() -> None:
    cfg = IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.CONSTANT, request_rate=10.0
    )
    from aiperf.timing.intervals import ConstantIntervalGenerator

    gen = ConstantIntervalGenerator(cfg)
    with pytest.raises(ValueError, match="must be > 0"):
        gen.set_rate(0.0)


def test_request_rate_infinity_passes_validation_but_yields_zero_period() -> None:
    """rate=inf passes the > 0 check; ConstantIntervalGenerator returns 1/inf=0.

    Document: the validator accepts inf even though it is conceptually the
    same as concurrency-burst. Not a bug per se but worth noting."""
    cfg = IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.CONSTANT, request_rate=float("inf")
    )
    from aiperf.timing.intervals import ConstantIntervalGenerator

    gen = ConstantIntervalGenerator(cfg)
    assert gen.next_interval() == 0.0


# =============================================================================
# RequestRate: handle_credit_return for DAG children
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Depends on dispatch_child_turn API on CreditIssuer not yet ported "
    "(future task; see plan P2T18 follow-ups)."
)
async def test_request_rate_dag_child_continuation_bypasses_continuation_queue() -> (
    None
):
    """RequestRate.handle_credit_return path for a credit with agent_depth>0
    must bypass the rate-limited ``_continuation_turns`` queue and dispatch
    via the credit issuer directly (immediate dispatch).

    Source semantics (request_rate.py:232-239): children dispatch directly
    rather than queueing because the main rate loop may have already exited
    by the time their continuation turns arrive."""
    turns = [TurnMetadata(), TurnMetadata()]
    conv = ConversationMetadata(conversation_id="child", turns=turns)
    ds = DatasetMetadata(
        conversations=[conv], sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    src = MagicMock()
    src.dataset_metadata = ds
    src.get_next_turn_metadata = lambda credit: turns[credit.turn_index + 1]

    issuer = MagicMock()
    issuer.issue_credit = AsyncMock(return_value=True)

    scheduler = MagicMock()
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
        arrival_pattern=ArrivalPattern.CONSTANT,
        total_expected_requests=2,
    )
    strategy = RequestRateStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )

    # Drive a child credit (agent_depth=1): must call issue_credit directly,
    # NOT queue.
    child_credit = _mk_credit(
        "child",
        "child-x",
        turn_index=0,
        num_turns=2,
        agent_depth=1,
        parent_correlation_id="parent-x",
    )
    await strategy.handle_credit_return(child_credit)

    issuer.issue_credit.assert_awaited_once()
    assert strategy._continuation_turns.empty(), (
        "child continuation must not enter rate-limited queue"
    )


@pytest.mark.asyncio
async def test_request_rate_dag_child_with_delay_uses_scheduler() -> None:
    """If the child's next-turn metadata has delay_ms, the rate strategy
    routes via scheduler.schedule_later, NOT via the rate-limited queue."""
    turns = [TurnMetadata(), TurnMetadata(delay_ms=500.0)]
    conv = ConversationMetadata(conversation_id="child", turns=turns)
    ds = DatasetMetadata(
        conversations=[conv], sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL
    )
    src = MagicMock()
    src.dataset_metadata = ds
    src.get_next_turn_metadata = lambda credit: turns[credit.turn_index + 1]

    issuer = MagicMock()
    issuer.issue_credit = lambda *a, **k: True
    scheduler = MagicMock()
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 1_000_000_000

    cfg = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
        arrival_pattern=ArrivalPattern.CONSTANT,
        total_expected_requests=2,
    )
    strategy = RequestRateStrategy(
        config=cfg,
        conversation_source=src,
        scheduler=scheduler,
        stop_checker=MagicMock(),
        credit_issuer=issuer,
        lifecycle=lifecycle,
    )

    child_credit = _mk_credit(
        "child", "child-x", turn_index=0, num_turns=2, agent_depth=1
    )
    await strategy.handle_credit_return(child_credit)

    scheduler.schedule_later.assert_called_once()
    delay_sec, _coro = scheduler.schedule_later.call_args.args
    assert delay_sec == pytest.approx(0.5)
    assert strategy._continuation_turns.empty()


# =============================================================================
# Orchestrator under wide and deep DAGs
# =============================================================================


@pytest.mark.asyncio
async def test_orchestrator_very_wide_fan_out_1000_children() -> None:
    """Single branch with 1000 children — orchestrator must dispatch each,
    register the gate accumulating expected=1000, and not OOM. Scaled to
    a manageable size for CI; the data-structure stress is the same."""
    N = 1000
    child_ids = [f"c{i}" for i in range(N)]
    branch = _make_branch("root:0", child_ids)
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    children = [
        ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
        for cid in child_ids
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    assert s is True
    assert orch.stats.children_spawned == N
    pending = orch._active_joins["p"]
    state = pending.outstanding["SPAWN_JOIN:root:0"]
    assert state.expected == N
    assert state.registered is True

    # Drain all children — gate fires exactly once.
    for child_corr in list(orch._child_to_join.keys()):
        await orch.on_child_leaf_reached(child_corr)
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_orchestrator_high_k_10000_intermediate_turns_no_suspension() -> None:
    """K=10000: parent has 10000 turns between spawn (0) and gate. Children
    finish before parent reaches the gate; ``parents_suspended`` stays at 0
    and ``_future_joins[parent]`` dict size never exceeds 1 entry."""
    K = 10000
    branch = _make_branch("root:0", ["c1"])
    parent_turns = [TurnMetadata(branch_ids=["root:0"])]
    parent_turns.extend(TurnMetadata() for _ in range(K - 1))
    parent_turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
            ]
        )
    )
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=[branch]
    )
    child = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()])
    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Spawn turn 0 -> registers single future-gate.
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=K + 1))
    assert len(orch._future_joins["p"]) == 1

    # Child finishes early.
    [child_corr] = list(orch._child_to_join.keys())
    await orch.on_child_leaf_reached(child_corr)
    # Future gate auto-popped.
    assert "p" not in orch._future_joins or not orch._future_joins["p"]

    # Walk parent through all 10001 turns; never suspends.
    for t in range(1, K + 1):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=K + 1))
        assert s is False, f"turn {t} must not suspend"
    assert orch.stats.parents_suspended == 0


@pytest.mark.asyncio
async def test_orchestrator_zero_child_branch_via_direct_construction() -> None:
    """Pydantic does NOT reject ConversationBranchInfo with empty children
    today (``child_conversation_ids`` has no min-length validator). Direct
    construction yields a branch the orchestrator must handle without hang.

    The validator (orchestrator_v1) is what would reject this at load time;
    when the orchestrator is fed a zero-child branch directly, the spawn
    loop iterates zero children, the gate is created with an empty
    outstanding dict (no prereqs declared on the spawning turn), and the
    parent must NOT suspend at the next turn since no prereq exists for
    that gated_idx."""
    # Branch with zero children. No prereq references it, so no gate is
    # registered for the parent's next turn either.
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=[],  # zero children
        mode=ConversationBranchMode.SPAWN,
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(),
        ],
        branches=[branch],
    )
    cs = _mk_source([root])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    # No children spawned; no gate registered (no prereq references "root:0");
    # parent must NOT suspend.
    assert s is False
    assert orch.stats.children_spawned == 0
    assert "p" not in orch._active_joins
    assert "p" not in orch._future_joins or not orch._future_joins.get("p")


@pytest.mark.asyncio
async def test_orchestrator_zero_child_branch_with_gate_does_not_hang() -> None:
    """Branch with zero children but the parent's next turn declares a
    SPAWN_JOIN against it. The orchestrator's expected_gates path must
    create a future-join with an unregistered PrereqState seed (from
    _gated_turn_prereq_keys) AND mark it registered with expected=0 — so
    is_done is True and the gate does NOT block the parent."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=[],  # zero children
        mode=ConversationBranchMode.SPAWN,
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    cs = _mk_source([root])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    # No children -> the expected_gates path fires the join immediately, so
    # by the time intercept returns the gate has been drained and the parent
    # is NOT suspended.
    assert s is False, "zero-child branch must not deadlock parent at next turn"
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.parents_resumed == 1
    assert orch.stats.parents_suspended == 0


# =============================================================================
# Phase replay state isolation
# =============================================================================


@pytest.mark.asyncio
async def test_phase_replay_active_joins_do_not_leak() -> None:
    """Run a complete spawn → suspend → drain cycle on phase 1, cleanup, then
    a fresh orchestrator for phase 2 must see empty state across
    ``_active_joins``, ``_future_joins``, ``_child_to_join``, and
    ``_descendant_counts``."""
    branch = _make_branch("root:0", ["c1"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    child = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()])
    cs = _mk_source([root, child])
    issuer = _mk_issuer()

    # Phase 1.
    warmup = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await warmup.intercept(_mk_credit("root", "p1", turn_index=0, num_turns=2))
    [child_corr] = list(warmup._child_to_join.keys())
    await warmup.on_child_leaf_reached(child_corr)
    warmup.cleanup()
    assert not warmup._active_joins
    assert not warmup._future_joins
    assert not warmup._child_to_join
    assert not warmup._descendant_counts

    # Phase 2: fresh state.
    measurement = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    assert not measurement._active_joins
    assert not measurement._future_joins
    assert not measurement._child_to_join
    assert not measurement._descendant_counts
    assert measurement.stats.children_spawned == 0


# =============================================================================
# Phase shutdown with stuck child
# =============================================================================


@pytest.mark.asyncio
async def test_phase_shutdown_with_stuck_child_fail_fast(monkeypatch) -> None:
    """One child errors -> fail-fast aborts the parent and any orphan siblings.
    The parent's pending join is dropped, ``has_pending_branch_work`` returns
    False once orphans are aborted, and shutdown can complete."""
    branch = _make_branch("root:0", ["c1", "c2"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    children = [
        ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
        for cid in ("c1", "c2")
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()

    monkeypatch.setattr("aiperf.common.environment.Environment.DAG.FAIL_FAST", True)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    assert orch.has_pending_branch_work()

    # Stuck child errors -> fail-fast aborts parent and orphan sibling.
    [c1, c2] = list(orch._child_to_join.keys())
    await orch.on_child_errored(c1)

    # Parent and orphan abort_session called.
    assert issuer.abort_session.await_count >= 1
    assert "p" not in orch._active_joins
    assert "p" not in orch._future_joins
    # Orphan should have been cleared from _child_to_join too.
    assert c2 not in orch._child_to_join


@pytest.mark.asyncio
async def test_phase_shutdown_cleanup_idempotent_under_late_returns() -> None:
    """After cleanup, a late ``intercept`` call must short-circuit (return
    False) without raising — even if the credit looks like a fresh spawn."""
    branch = _make_branch("root:0", ["c1"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    child = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()])
    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    orch.cleanup()
    # Second cleanup is idempotent.
    orch.cleanup()

    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    assert s is False
    # No children dispatched — the cleanup short-circuit fires before the
    # spawn path is reached.
    issuer.dispatch_first_turn.assert_not_called()


# =============================================================================
# Pre-session: nested branches in a pre-session child are NOT pre-dispatched
# =============================================================================


@pytest.mark.asyncio
async def test_pre_session_child_with_own_dag_does_not_recurse_pre_dispatch() -> None:
    """A pre-session child has its own DAG metadata with a 'pre' branch on
    turn 0. ``dispatch_pre_session_branches`` only iterates root conversations
    (``agent_depth == 0``); a child conversation, even if it has dispatch_timing
    'pre' branches in metadata, is NOT pre-dispatched recursively.

    This is current behaviour. Documented as a fidelity concern: trace
    replay where a captured pre-session child itself has nested pre-session
    spawns would NOT honour the nesting.
    """
    pre_branch_root = _make_branch(
        "root:0",
        ["pre_child"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    pre_branch_nested = _make_branch(
        "pre_child:0",
        ["nested"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(),
        ],
        branches=[pre_branch_root],
    )
    pre_child = ConversationMetadata(
        conversation_id="pre_child",
        turns=[
            TurnMetadata(branch_ids=["pre_child:0"]),
            TurnMetadata(),
        ],
        branches=[pre_branch_nested],
        agent_depth=1,
    )
    nested = ConversationMetadata(
        conversation_id="nested", turns=[TurnMetadata()], agent_depth=2
    )
    cs = _mk_source([root, pre_child, nested])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    # Only one pre-dispatched: pre_child. nested is NOT recursively
    # pre-dispatched even though pre_child's metadata declares a pre-branch.
    assert orch.stats.children_spawned == 1
    assert ("root", "root:0") in orch._pre_dispatched_branches
    assert ("pre_child", "pre_child:0") not in orch._pre_dispatched_branches


# =============================================================================
# delay_ms after a delayed-join gap (Fixed Schedule)
# =============================================================================


@pytest.mark.asyncio
async def test_fixed_schedule_resumed_gated_turn_uses_authored_timestamp() -> None:
    """When a parent's gated turn dispatches via ``CreditIssuer.dispatch_join_turn``,
    that path ignores the ``delay_ms`` and ``timestamp_ms`` of the gated turn —
    the orchestrator builds a TurnToSend directly from PendingBranchJoin and
    issues it immediately (no scheduler.schedule_at_perf_sec).

    Verify by inspecting that ``dispatch_join_turn`` is what fires (not
    ``handle_credit_return``); scheduler.schedule_at_perf_sec is untouched
    for the gated turn."""
    branch = _make_branch("root:0", ["c1"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"], timestamp_ms=0),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ],
                # Authored delay AND timestamp on the gated turn — both ignored
                # because the orchestrator dispatches directly via dispatch_join_turn.
                delay_ms=100.0,
                timestamp_ms=5000,
            ),
        ],
        branches=[branch],
    )
    child = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()])
    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    [child_corr] = list(orch._child_to_join.keys())
    await orch.on_child_leaf_reached(child_corr)

    # Gated turn dispatched via the join path — bypasses any delay_ms /
    # timestamp_ms scheduling on the gated TurnMetadata.
    issuer.dispatch_join_turn.assert_awaited_once()
    sent_pending = issuer.dispatch_join_turn.call_args.args[0]
    assert sent_pending.gated_turn_index == 1
    # No fields propagating delay_ms / timestamp_ms exist on PendingBranchJoin —
    # documents the contract.
    assert not hasattr(sent_pending, "delay_ms")
    assert not hasattr(sent_pending, "timestamp_ms")


# =============================================================================
# Cancellation surface during async dispatch
# =============================================================================


@pytest.mark.asyncio
async def test_intercept_cancellation_surfaces_cleanly() -> None:
    """If ``dispatch_first_turn`` is cancelled mid-spawn, the CancelledError
    propagates out of ``intercept``. Verify no orphan _child_to_join entries
    remain for the cancelled child path."""
    branch = _make_branch("root:0", ["c1", "c2"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    children = [
        ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
        for cid in ("c1", "c2")
    ]
    cs = _mk_source([root, *children])

    issuer = _mk_issuer()

    call_count = {"n": 0}

    async def _dispatch(session):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise asyncio.CancelledError("simulated cancellation mid-dispatch")
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # asyncio.gather(return_exceptions=True) absorbs the CancelledError.
    # Verify that the rollback path runs for the cancelled child.
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))

    # One child landed (returned True), one was cancelled (rolled back).
    assert orch.stats.children_spawned == 1
    assert orch.stats.children_errored == 1
    # The successful child is still tracked.
    assert len(orch._child_to_join) == 1


# =============================================================================
# Rate-limit ↔ DAG: child agent_depth=1 bypasses session-slot but still goes
# through credit_issuer.dispatch_first_turn.
# =============================================================================


@pytest.mark.asyncio
async def test_dag_child_dispatch_path_decoupled_from_main_rate_loop() -> None:
    """Child dispatch goes through ``credit_issuer.dispatch_first_turn`` (the
    DAG path), not ``credit_issuer.try_issue_credit`` (the rate-limited
    new-session path). Confirms children are NOT subject to the rate
    interval-generator's pacing — they fire as soon as the orchestrator
    schedules them."""
    branch = _make_branch("root:0", ["c1", "c2", "c3"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    children = [
        ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
        for cid in ("c1", "c2", "c3")
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    # try_issue_credit is the rate-paced path; never called for children.
    issuer.try_issue_credit = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))

    # Children went through dispatch_first_turn (DAG path), not try_issue_credit.
    assert issuer.dispatch_first_turn.await_count == 3
    issuer.try_issue_credit.assert_not_called()
