# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial pathology tests targeting timing-strategy ↔ DAG-orchestrator"""

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


@pytest.mark.asyncio
async def test_fixed_schedule_out_of_order_timestamps_within_conversation() -> None:
    """Turn 5 has timestamp_ms < turn 4 within the same conversation. The"""
    timestamps = [0, 1000, 2000, 3000, 5000, 4000]
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

    credit = _mk_credit("c1", "x", turn_index=4, num_turns=6)
    await strategy.handle_credit_return(credit)

    scheduler.schedule_at_perf_sec.assert_called_once()
    target_perf, _ = scheduler.schedule_at_perf_sec.call_args.args
    assert target_perf == pytest.approx(5.0), (
        "out-of-order timestamps are passed through unvalidated"
    )


@pytest.mark.asyncio
async def test_fixed_schedule_negative_timestamp_no_validation() -> None:
    """Pydantic accepts negative timestamps (no min check). Document for"""
    tm = TurnMetadata(timestamp_ms=-1000)
    assert tm.timestamp_ms == -1000


@pytest.mark.asyncio
async def test_fixed_schedule_very_large_timestamp_no_overflow() -> None:
    """timestamp_ms = 2^53 (boundary of float-safe-integer)."""
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
    assert target_perf > 0


@pytest.mark.asyncio
async def test_fixed_schedule_setup_sorts_identical_timestamps_stably() -> None:
    """Three sibling conversations all with timestamp_ms=0 — the schedule"""
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


def test_request_rate_validates_zero_rate_at_interval_config() -> None:
    """Rate=0 must be rejected by ``IntervalGeneratorConfig`` (``gt=0``)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than 0"):
        IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONSTANT, request_rate=0.0
        )


def test_request_rate_validates_negative_rate() -> None:
    """Negative rate must be rejected by ``IntervalGeneratorConfig`` (``gt=0``)."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="greater than 0"):
        IntervalGeneratorConfig(
            arrival_pattern=ArrivalPattern.CONSTANT, request_rate=-1.0
        )


def test_request_rate_set_rate_rejects_zero() -> None:
    cfg = IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.CONSTANT, request_rate=10.0
    )
    from aiperf.timing.intervals import ConstantIntervalGenerator

    gen = ConstantIntervalGenerator(cfg)
    with pytest.raises(ValueError, match="must be > 0"):
        gen.set_rate(0.0)


def test_request_rate_infinity_passes_validation_but_yields_zero_period() -> None:
    """rate=inf passes the > 0 check; ConstantIntervalGenerator returns 1/inf=0."""
    cfg = IntervalGeneratorConfig(
        arrival_pattern=ArrivalPattern.CONSTANT, request_rate=float("inf")
    )
    from aiperf.timing.intervals import ConstantIntervalGenerator

    gen = ConstantIntervalGenerator(cfg)
    assert gen.next_interval() == 0.0


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Depends on a dispatch_child_turn API that CreditIssuer does not implement."
)
async def test_request_rate_dag_child_continuation_bypasses_continuation_queue() -> (
    None
):
    """RequestRate.handle_credit_return path for a credit with agent_depth>0"""
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
    """If the child's next-turn metadata has delay_ms, the rate strategy"""
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


@pytest.mark.asyncio
async def test_orchestrator_very_wide_fan_out_1000_children() -> None:
    """Single branch with 1000 children — orchestrator must dispatch each,"""
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

    for child_corr in list(orch._child_to_join.keys()):
        await orch.on_child_leaf_reached(child_corr)
    issuer.dispatch_join_turn.assert_awaited_once()


@pytest.mark.slow
@pytest.mark.stress
@pytest.mark.asyncio
async def test_orchestrator_high_k_10000_intermediate_turns_no_suspension() -> None:
    """K=10000: parent has 10000 turns between spawn (0) and gate. Children"""
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

    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=K + 1))
    assert len(orch._future_joins["p"]) == 1

    [child_corr] = list(orch._child_to_join.keys())
    await orch.on_child_leaf_reached(child_corr)
    assert "p" not in orch._future_joins or not orch._future_joins["p"]

    for t in range(1, K + 1):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=K + 1))
        assert s is False, f"turn {t} must not suspend"
    assert orch.stats.parents_suspended == 0


@pytest.mark.asyncio
async def test_orchestrator_zero_child_branch_via_direct_construction() -> None:
    """Pydantic does NOT reject ConversationBranchInfo with empty children"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=[],
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
    assert s is False
    assert orch.stats.children_spawned == 0
    assert "p" not in orch._active_joins
    assert "p" not in orch._future_joins or not orch._future_joins.get("p")


@pytest.mark.asyncio
async def test_orchestrator_zero_child_branch_with_gate_does_not_hang() -> None:
    """Branch with zero children but the parent's next turn declares a"""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=[],
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
    assert s is False, "zero-child branch must not deadlock parent at next turn"
    issuer.dispatch_join_turn.assert_not_awaited()
    assert "p" not in orch._active_joins
    assert not orch._future_joins.get("p")
    assert orch.stats.parents_resumed == 0
    assert orch.stats.parents_suspended == 0


@pytest.mark.asyncio
async def test_phase_replay_active_joins_do_not_leak() -> None:
    """Run a complete spawn → suspend → drain cycle on phase 1, cleanup, then"""
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

    warmup = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await warmup.intercept(_mk_credit("root", "p1", turn_index=0, num_turns=2))
    [child_corr] = list(warmup._child_to_join.keys())
    await warmup.on_child_leaf_reached(child_corr)
    warmup.cleanup()
    assert not warmup._active_joins
    assert not warmup._future_joins
    assert not warmup._child_to_join
    assert not warmup._descendant_counts

    measurement = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    assert not measurement._active_joins
    assert not measurement._future_joins
    assert not measurement._child_to_join
    assert not measurement._descendant_counts
    assert measurement.stats.children_spawned == 0


@pytest.mark.asyncio
async def test_phase_shutdown_with_stuck_child_fail_fast(monkeypatch) -> None:
    """One child errors -> fail-fast aborts the parent and any orphan siblings."""
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

    [c1, c2] = list(orch._child_to_join.keys())
    await orch.on_child_errored(c1)

    assert issuer.abort_session.await_count >= 1
    assert "p" not in orch._active_joins
    assert "p" not in orch._future_joins
    assert c2 not in orch._child_to_join


@pytest.mark.asyncio
async def test_phase_shutdown_cleanup_idempotent_under_late_returns() -> None:
    """After cleanup, a late ``intercept`` call must short-circuit (return"""
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
    orch.cleanup()

    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    assert s is False
    issuer.dispatch_first_turn.assert_not_called()


@pytest.mark.asyncio
async def test_pre_session_child_with_own_dag_does_not_recurse_pre_dispatch() -> None:
    """A pre-session child has its own DAG metadata with a 'pre' branch on"""
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
    assert orch.stats.children_spawned == 1
    assert ("root", "root:0") in orch._pre_dispatched_branches
    assert ("pre_child", "pre_child:0") not in orch._pre_dispatched_branches


@pytest.mark.asyncio
async def test_fixed_schedule_resumed_gated_turn_uses_authored_timestamp() -> None:
    """When a parent's gated turn dispatches via ``CreditIssuer.dispatch_join_turn``,"""
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

    issuer.dispatch_join_turn.assert_awaited_once()
    sent_pending = issuer.dispatch_join_turn.call_args.args[0]
    assert sent_pending.gated_turn_index == 1
    assert not hasattr(sent_pending, "delay_ms")
    assert not hasattr(sent_pending, "timestamp_ms")


@pytest.mark.asyncio
async def test_intercept_cancellation_surfaces_cleanly() -> None:
    """If ``dispatch_first_turn`` is cancelled mid-spawn, the CancelledError"""
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

    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))

    assert orch.stats.children_spawned == 1
    assert orch.stats.children_errored == 1
    assert len(orch._child_to_join) == 1


@pytest.mark.asyncio
async def test_dag_child_dispatch_path_decoupled_from_main_rate_loop() -> None:
    """Child dispatch goes through ``credit_issuer.dispatch_first_turn`` (the"""
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
    issuer.try_issue_credit = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))

    assert issuer.dispatch_first_turn.await_count == 3
    issuer.try_issue_credit.assert_not_called()
