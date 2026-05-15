# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial component-integration tests for the DAG ``BranchOrchestrator``
exercised against the three strategy-agnostic shapes (FIXED_SCHEDULE,
REQUEST_RATE, USER_CENTRIC_RATE).

The orchestrator integrates strategy-agnostically through
``CreditCallbackHandler`` (intercept(credit) returns True iff strategy
dispatch should be suppressed). Tests here mock the credit issuer and drive
``orchestrator.intercept`` / ``on_child_leaf_reached`` directly with credits
shaped per timing mode (timestamps for FIXED_SCHEDULE, delay_ms for
REQUEST_RATE / USER_CENTRIC_RATE) and assert orchestrator-level invariants
that must hold *identically* across the three modes.

Coverage is the 20 attack vectors in the prompt that follows the
2026-04-24-dag-delayed-multi-gate-fan-in plan: K=1/5/50, multi-gate, fan-in,
pre-session, FORK+SPAWN mixing, stop conditions during gap, cancellation
during pre-dispatch, phase replay, strategy-specific rate-limit and
slot-reuse interactions.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy, TimingMode
from aiperf.timing.branch_orchestrator import BranchOrchestrator

pytestmark = pytest.mark.component_integration


# =============================================================================
# Strategy parametrization shape
# =============================================================================
#
# The orchestrator intercept path runs identically across strategies. Each
# strategy reads a different field from TurnMetadata to schedule the *next*
# turn (see strategies/{fixed_schedule,request_rate,user_centric_rate}.py):
#
#   FIXED_SCHEDULE   -> turns[i].timestamp_ms drives schedule_at_perf_sec
#   REQUEST_RATE     -> turns[i].delay_ms threads through schedule_later
#   USER_CENTRIC_RATE-> per-user turn_gap; metadata.delay_ms is honoured iff
#                       set, otherwise turn_gap is the only spacing
#
# Parametrizing over the TimingMode label below keeps the tests' shape
# identical but exercises each strategy's preferred metadata channel and
# documents per-strategy variance in xfails when present.

STRATEGY_IDS = [
    TimingMode.FIXED_SCHEDULE,
    TimingMode.REQUEST_RATE,
    TimingMode.USER_CENTRIC_RATE,
]


def _ts_kwargs(strategy: TimingMode, idx: int, base_ms: int = 1000, step_ms: int = 500):
    """Per-strategy TurnMetadata kwargs.

    FIXED_SCHEDULE uses absolute timestamps; the rate-based strategies use
    delay_ms after the first turn. Both are valid orchestration inputs and
    both go through ``ConversationSource.get_next_turn_metadata``.
    """
    if strategy == TimingMode.FIXED_SCHEDULE:
        return {"timestamp_ms": base_ms + idx * step_ms}
    if idx == 0:
        return {}
    return {"delay_ms": float(step_ms)}


# =============================================================================
# Helpers
# =============================================================================


def _mk_credit(
    conv_id: str,
    x_corr: str,
    turn_index: int = 0,
    num_turns: int = 1,
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
) -> Credit:
    """Build a Credit-shaped MagicMock — the orchestrator only reads attrs."""
    c = MagicMock(spec=Credit)
    c.conversation_id = conv_id
    c.x_correlation_id = x_corr
    c.turn_index = turn_index
    c.num_turns = num_turns
    c.agent_depth = agent_depth
    c.parent_correlation_id = parent_correlation_id
    c.branch_mode = ConversationBranchMode.FORK
    c.is_final_turn = turn_index == num_turns - 1
    return c


def _mk_source(
    conversations: list[ConversationMetadata],
    *,
    pre_session_factory: Callable[[str], Any] | None = None,
):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    lookup = {c.conversation_id: c for c in conversations}
    cs.get_metadata.side_effect = lambda cid: lookup[cid]

    corr_counter = {"n": 0}

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **_kw
    ):
        corr_counter["n"] += 1
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}-{corr_counter['n']}"
        s.conversation_id = child_conversation_id
        s.agent_depth = agent_depth
        s.parent_correlation_id = parent_correlation_id
        s.branch_mode = branch_mode
        return s

    cs.start_branch_child.side_effect = _start

    def _start_pre(child_conversation_id, **_kw):
        if pre_session_factory is not None:
            return pre_session_factory(child_conversation_id)
        corr_counter["n"] += 1
        s = MagicMock()
        s.x_correlation_id = f"pre-{child_conversation_id}-{corr_counter['n']}"
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
# 1. K=1 baseline (regression). All three strategies.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_k1_baseline_dispatches_join_turn(strategy: TimingMode) -> None:
    """K=1 must produce bit-identical orchestrator behaviour across strategies."""
    branch = _make_branch("root:0", ["c1"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"], **_ts_kwargs(strategy, 0)),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ],
                **_ts_kwargs(strategy, 1),
            ),
        ],
        branches=[branch],
    )
    child = ConversationMetadata(
        conversation_id="c1", turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
    )

    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suppressed = await orch.intercept(
        _mk_credit("root", "p", turn_index=0, num_turns=2)
    )
    assert suppressed is True, f"{strategy}: parent should be suspended at K=1 gate"
    assert "p" in orch._active_joins

    # Drive the single child to completion via the leaf hook (callback handler
    # invokes this on a final-turn child credit).
    [child_corr] = list(orch._child_to_join.keys())
    await orch.on_child_leaf_reached(child_corr)

    issuer.dispatch_join_turn.assert_awaited_once()
    sent = issuer.dispatch_join_turn.call_args.args[0]
    assert sent.gated_turn_index == 1
    assert orch.stats.parents_resumed == 1


# =============================================================================
# 2. K=5 delayed join: parent-progresses semantics
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_k5_delayed_join_parent_progresses_then_suspends(
    strategy: TimingMode,
) -> None:
    """Parent dispatches turns 1..4 normally; suspends at 5; resumes after children.

    The DAG's invariant flip (Phase 1) is that ``intercept`` no longer returns
    True on the spawning turn — only on the turn whose NEXT turn is gated.
    Drive turn 0 (spawn), then turns 1..3 (no suspend), then turn 4 (suspend
    because next is gated turn 5).
    """
    branch = _make_branch("root:0", ["c1", "c2"])
    parent_turns = [TurnMetadata(branch_ids=["root:0"], **_ts_kwargs(strategy, 0))]
    for i in range(1, 5):
        parent_turns.append(TurnMetadata(**_ts_kwargs(strategy, i)))
    parent_turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
            ],
            **_ts_kwargs(strategy, 5),
        )
    )
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=[branch]
    )
    children = [
        ConversationMetadata(
            conversation_id=cid, turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
        )
        for cid in ("c1", "c2")
    ]

    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0: spawn — should NOT suspend (parent progresses).
    suppressed = await orch.intercept(
        _mk_credit("root", "p", turn_index=0, num_turns=6)
    )
    assert suppressed is False, f"{strategy}: spawn turn must not suspend (Phase 1)"
    assert "p" not in orch._active_joins
    assert "p" in orch._future_joins

    # Turns 1..3: parent in gap; intercept returns False every time.
    for t in range(1, 4):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=6))
        assert s is False, f"{strategy}: turn {t} in K=5 gap must not suspend"

    # Turn 4: next turn (5) is gated and prereqs unsatisfied -> suspend.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=4, num_turns=6))
    assert s is True, f"{strategy}: turn 4 must suspend (next turn is gated)"
    assert "p" in orch._active_joins

    # Drain children -> dispatch join turn.
    for child_corr in list(orch._child_to_join.keys()):
        await orch.on_child_leaf_reached(child_corr)
    issuer.dispatch_join_turn.assert_awaited_once()
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 5
    assert orch.stats.parents_resumed == 1


# =============================================================================
# 3. K=5 children-finish-before-parent-arrives — no spurious suspension.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_k5_children_finish_before_parent_arrives(strategy: TimingMode) -> None:
    branch = _make_branch("root:0", ["c1"])
    parent_turns = [TurnMetadata(branch_ids=["root:0"], **_ts_kwargs(strategy, 0))]
    for i in range(1, 5):
        parent_turns.append(TurnMetadata(**_ts_kwargs(strategy, i)))
    parent_turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
            ],
            **_ts_kwargs(strategy, 5),
        )
    )
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=[branch]
    )
    child = ConversationMetadata(
        conversation_id="c1", turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
    )

    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Spawn turn 0.
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=6))
    [child_corr] = list(orch._child_to_join.keys())

    # Child completes before parent reaches turn 4.
    await orch.on_child_leaf_reached(child_corr)

    # Future gate auto-popped; intermediate intercepts must not see a gate.
    for t in range(1, 5):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=6))
        assert s is False, f"{strategy}: turn {t} must not suspend after early child"

    # Critical: stats.parents_suspended must be 0 — children finished early.
    assert orch.stats.parents_suspended == 0, (
        f"{strategy}: spurious suspension when children finished before parent arrived"
    )
    assert orch.stats.children_completed == 1
    issuer.dispatch_join_turn.assert_not_called()


# =============================================================================
# 4. Multi-gate per spawning turn (Phase 2).
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_multi_gated_branches_per_spawning_turn(strategy: TimingMode) -> None:
    """Turn 0 spawns three branches gated at T+1, T+2, T+4. Parent must
    suspend each time it reaches a gated turn (3 separate suspensions)."""
    branches = [
        _make_branch("a", ["ca"]),
        _make_branch("b", ["cb"]),
        _make_branch("c", ["cc"]),
    ]
    parent_turns = [
        TurnMetadata(branch_ids=["a", "b", "c"], **_ts_kwargs(strategy, 0)),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="a")
            ],
            **_ts_kwargs(strategy, 1),
        ),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b")
            ],
            **_ts_kwargs(strategy, 2),
        ),
        TurnMetadata(**_ts_kwargs(strategy, 3)),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="c")
            ],
            **_ts_kwargs(strategy, 4),
        ),
    ]
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=branches
    )
    children = [
        ConversationMetadata(
            conversation_id=cid, turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
        )
        for cid in ("ca", "cb", "cc")
    ]

    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Turn 0 spawns three branches with three independent gates.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=5))
    # Next turn (1) is gated -> True.
    assert s is True
    # Three independent future gates registered.
    assert (
        len(orch._future_joins.get("p", {})) + (1 if "p" in orch._active_joins else 0)
        == 3
    ), f"{strategy}: expected 3 gates total"

    # Map each child to its prereq_key by inspecting registrations.
    child_to_branch: dict[str, str] = {}
    for child_corr, entries in orch._child_to_join.items():
        # one entry per child (single gate per child here)
        child_to_branch[child_corr] = entries[0].prereq_key

    # Complete child for branch a; gated turn 1 dispatches.
    ca = next(cc for cc, k in child_to_branch.items() if k == "SPAWN_JOIN:a")
    await orch.on_child_leaf_reached(ca)
    assert issuer.dispatch_join_turn.await_count == 1
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 1

    # Parent's turn 1 returns -> next turn (2) gated, b not yet done.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=1, num_turns=5))
    assert s is True

    cb = next(cc for cc, k in child_to_branch.items() if k == "SPAWN_JOIN:b")
    await orch.on_child_leaf_reached(cb)
    assert issuer.dispatch_join_turn.await_count == 2
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 2

    # Turn 2 returns -> next turn (3) is NOT gated.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=2, num_turns=5))
    assert s is False

    # Turn 3 returns -> next turn (4) is gated on c.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=3, num_turns=5))
    assert s is True

    cc = next(cc for cc, k in child_to_branch.items() if k == "SPAWN_JOIN:c")
    await orch.on_child_leaf_reached(cc)
    assert issuer.dispatch_join_turn.await_count == 3
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 4

    assert orch.stats.parents_suspended == 3
    assert orch.stats.parents_resumed == 3


# =============================================================================
# 5. Fan-in across spawning turns (Phase 3).
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_fan_in_across_spawning_turns(strategy: TimingMode) -> None:
    """Turn 0 spawns A, turn 2 spawns B. Turn 5 has prereqs [A, B]. Gate
    waits for both branches.

    This stresses the Phase-3 ``_gated_turn_prereq_keys`` seed: when the
    spawning turn for A fires before B, the gate must NOT be satisfied
    until B's spawning turn has registered its prereq AND completed.
    """
    branches = [_make_branch("a", ["ca"]), _make_branch("b", ["cb"])]
    parent_turns = [
        TurnMetadata(branch_ids=["a"], **_ts_kwargs(strategy, 0)),
        TurnMetadata(**_ts_kwargs(strategy, 1)),
        TurnMetadata(branch_ids=["b"], **_ts_kwargs(strategy, 2)),
        TurnMetadata(**_ts_kwargs(strategy, 3)),
        TurnMetadata(**_ts_kwargs(strategy, 4)),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="a"),
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b"),
            ],
            **_ts_kwargs(strategy, 5),
        ),
    ]
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=branches
    )
    children = [
        ConversationMetadata(
            conversation_id=cid, turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
        )
        for cid in ("ca", "cb")
    ]

    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Spawn A on turn 0.
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=6))
    [ca_corr] = list(orch._child_to_join.keys())

    # Complete A *before* B has even spawned. Gate must NOT fire.
    await orch.on_child_leaf_reached(ca_corr)
    issuer.dispatch_join_turn.assert_not_called()

    # Turn 1, 2 (spawn B), 3, 4.
    await orch.intercept(_mk_credit("root", "p", turn_index=1, num_turns=6))
    await orch.intercept(_mk_credit("root", "p", turn_index=2, num_turns=6))
    cb_corrs = [c for c in orch._child_to_join if c != ca_corr]
    assert len(cb_corrs) == 1
    await orch.intercept(_mk_credit("root", "p", turn_index=3, num_turns=6))
    s = await orch.intercept(_mk_credit("root", "p", turn_index=4, num_turns=6))
    assert s is True, f"{strategy}: turn 4 should suspend (next turn is gated)"

    # Now complete B -> gate fires.
    await orch.on_child_leaf_reached(cb_corrs[0])
    issuer.dispatch_join_turn.assert_awaited_once()
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 5


# =============================================================================
# 6. Pre-session background spawn (Phase 2b).
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_pre_session_background_dispatched_before_parent_turn0(
    strategy: TimingMode,
) -> None:
    """Children dispatched via ``dispatch_pre_session_branches`` appear in
    the dispatch log BEFORE any parent turn-0 credit issuance; subsequent
    parent turn-0 intercept does NOT re-dispatch them.
    """
    branch = _make_branch(
        "root:0",
        ["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"], **_ts_kwargs(strategy, 0)),
            TurnMetadata(**_ts_kwargs(strategy, 1)),
        ],
        branches=[branch],
    )
    child = ConversationMetadata(
        conversation_id="early", turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
    )

    dispatch_log: list[str] = []

    async def _dispatch_first_turn(session):
        dispatch_log.append(getattr(session, "conversation_id", "<unk>"))
        return True

    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    issuer.dispatch_first_turn = AsyncMock(side_effect=_dispatch_first_turn)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    assert dispatch_log == ["early"], f"{strategy}: pre-session must fire 'early' first"
    assert ("root", "root:0") in orch._pre_dispatched_branches
    assert orch.stats.children_spawned == 1

    # Parent turn-0 intercept must NOT re-dispatch (the pre-dispatched filter).
    pre_count = len(dispatch_log)
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    assert len(dispatch_log) == pre_count, (
        f"{strategy}: pre-dispatched branch must not re-dispatch on parent turn 0"
    )


# =============================================================================
# 7. FixedSchedule-specific: child has timestamp BEFORE parent's spawning turn.
# =============================================================================


@pytest.mark.asyncio
async def test_fixed_schedule_child_timestamp_before_parent_spawn() -> None:
    """Author a JSONL where child's first-turn timestamp is BEFORE the
    parent's spawning-turn timestamp. The orchestrator dispatches children
    only after the parent's spawning credit returns — so the child fires
    after, regardless of authored timestamp.

    Documented behaviour: post-dispatch wins over the authored timestamp.
    The strategy's ``_timestamp_to_perf_sec`` would re-anchor against the
    schedule zero, but ``BranchOrchestrator.dispatch_first_turn`` enters
    ``credit_issuer.try_issue_credit`` directly and ignores timestamp_ms.
    """
    branch = _make_branch("root:0", ["early"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(
                branch_ids=["root:0"], timestamp_ms=5000
            ),  # parent spawns later
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ],
                timestamp_ms=6000,
            ),
        ],
        branches=[branch],
    )
    # Child timestamp is BEFORE parent spawn.
    child = ConversationMetadata(
        conversation_id="early", turns=[TurnMetadata(timestamp_ms=1000)]
    )

    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    # Child was dispatched, *not* timestamp-reordered. Orchestrator stats reflect
    # this: child spawned via post-dispatch path.
    assert orch.stats.children_spawned == 1
    issuer.dispatch_first_turn.assert_awaited_once()


# =============================================================================
# 8. FixedSchedule-specific: child timestamps overlap parent's gated turn.
# =============================================================================


@pytest.mark.asyncio
async def test_fixed_schedule_child_late_timestamp_does_not_release_gate() -> None:
    """DAG semantics MUST override timestamps: parent's gated turn is suppressed
    until child completes, even if the child's last timestamp is later than
    the parent's gated-turn timestamp.
    """
    branch = _make_branch("root:0", ["c1"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"], timestamp_ms=1000),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ],
                timestamp_ms=2000,
            ),
        ],
        branches=[branch],
    )
    # Child's only turn timestamp is later than parent's gated turn.
    child = ConversationMetadata(
        conversation_id="c1", turns=[TurnMetadata(timestamp_ms=5000)]
    )

    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    assert s is True, (
        "gate must be active (DAG suspends parent regardless of timestamps)"
    )
    # Gated turn must NOT have been dispatched yet.
    issuer.dispatch_join_turn.assert_not_called()

    # After child completes, gated turn dispatches via orchestrator (NOT via
    # the strategy's timestamp scheduler).
    [child_corr] = list(orch._child_to_join.keys())
    await orch.on_child_leaf_reached(child_corr)
    issuer.dispatch_join_turn.assert_awaited_once()


# =============================================================================
# 9. RequestRate-specific: child dispatch contributes to rate (rate-limited).
# =============================================================================


@pytest.mark.asyncio
async def test_request_rate_child_dispatch_uses_credit_issuer() -> None:
    """Children dispatched via ``CreditIssuer.dispatch_first_turn`` route
    through ``try_issue_credit`` which honours rate / concurrency limits.
    Verify that under saturation (try_issue_credit returns None), the
    orchestrator rolls back per-child bookkeeping and the gate sees zero
    expected — auto-firing the join immediately to avoid hangs.
    """
    branch = _make_branch("root:0", ["c1", "c2", "c3", "c4", "c5"])
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
        for cid in ("c1", "c2", "c3", "c4", "c5")
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()

    # Simulate rate limit: third call onwards returns None (no slot).
    call_count = {"n": 0}

    async def _try(session):
        call_count["n"] += 1
        # dispatch_first_turn maps None|False to False (no-slot rollback).
        return call_count["n"] <= 2

    issuer.dispatch_first_turn = AsyncMock(side_effect=_try)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))

    # Two children landed, three rolled back.
    assert orch.stats.children_spawned == 2
    # Saturated ``dispatch_first_turn`` (returns False) is stop-condition
    # refusal, not an error — tally as truncated.
    assert orch.stats.children_truncated == 3
    assert orch.stats.children_errored == 0


# =============================================================================
# 10. RequestRate-specific: gated turn dispatch goes through try_issue_credit.
# =============================================================================


@pytest.mark.asyncio
async def test_request_rate_gated_turn_uses_try_issue_credit() -> None:
    """``dispatch_join_turn`` calls ``try_issue_credit`` which respects the
    rate/concurrency. When suppressed (False), ``joins_suppressed`` increments.
    """
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
    issuer = _mk_issuer(dispatch_join_returns=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    [child_corr] = list(orch._child_to_join.keys())
    await orch.on_child_leaf_reached(child_corr)

    # Gate fired but issuer suppressed -> stats reflect.
    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.joins_suppressed == 1
    assert orch.stats.parents_resumed == 0


# =============================================================================
# 11/12. UserCentric-specific: agent_depth>0 children bypass slot acquisition.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_children_use_agent_depth_for_slot_bypass(strategy: TimingMode) -> None:
    """The orchestrator dispatches children with ``agent_depth=parent_depth+1``.
    Verify the SampledSession built for each child carries agent_depth=1, which
    is what ``CreditIssuer.try_issue_credit`` and the callback handler use to
    bypass session-slot acquisition / release."""
    branch = _make_branch("root:0", ["c1", "c2"])
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"], **_ts_kwargs(strategy, 0)),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ],
                **_ts_kwargs(strategy, 1),
            ),
        ],
        branches=[branch],
    )
    children = [
        ConversationMetadata(
            conversation_id=cid, turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
        )
        for cid in ("c1", "c2")
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))

    # start_branch_child is invoked with kwargs by the orchestrator. Both
    # children must be created with agent_depth=1 so the slot-bypass path
    # in CreditIssuer.try_issue_credit / CreditCallbackHandler activates.
    assert cs.start_branch_child.call_count == 2
    for kall in cs.start_branch_child.call_args_list:
        assert kall.kwargs["agent_depth"] == 1
        assert kall.kwargs["parent_correlation_id"] == "p"


# =============================================================================
# 13. Stop condition during delayed gap (all strategies).
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_stop_condition_during_delayed_gap_suppresses_join(
    strategy: TimingMode,
) -> None:
    """When the issuer's join dispatch returns False (stop condition fired),
    ``joins_suppressed`` increments and ``parents_resumed`` does not."""
    branch = _make_branch("root:0", ["c1"])
    parent_turns = [TurnMetadata(branch_ids=["root:0"], **_ts_kwargs(strategy, 0))]
    for i in range(1, 5):
        parent_turns.append(TurnMetadata(**_ts_kwargs(strategy, i)))
    parent_turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
            ],
            **_ts_kwargs(strategy, 5),
        )
    )
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=[branch]
    )
    child = ConversationMetadata(
        conversation_id="c1", turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
    )
    cs = _mk_source([root, child])
    issuer = _mk_issuer(dispatch_join_returns=False)
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Spawn at turn 0; advance to turn 4 (suspend); complete child.
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=6))
    for t in range(1, 5):
        await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=6))
    [child_corr] = list(orch._child_to_join.keys())
    await orch.on_child_leaf_reached(child_corr)

    assert orch.stats.joins_suppressed == 1
    assert orch.stats.parents_resumed == 0


# =============================================================================
# 14. Cancellation during pre-session dispatch.
# =============================================================================


@pytest.mark.asyncio
async def test_cancellation_during_pre_session_dispatch_no_hang() -> None:
    """If the issuer raises mid pre-session dispatch (simulated Ctrl-C),
    the orchestrator should not hang. The current code does not
    catch exceptions during dispatch_first_turn — verify the failure is
    surfaced and stats reflect partial progress.
    """
    branch = _make_branch(
        "root:0",
        ["e1", "e2", "e3"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(branch_ids=["root:0"]), TurnMetadata()],
        branches=[branch],
    )
    children = [
        ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
        for cid in ("e1", "e2", "e3")
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()

    # Fire successfully twice, then raise — emulates worker cancellation.
    call_count = {"n": 0}

    async def _ds(session):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise asyncio.CancelledError("simulated ctrl-c")
        return True

    issuer.dispatch_first_turn = AsyncMock(side_effect=_ds)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    with pytest.raises(asyncio.CancelledError):
        await orch.dispatch_pre_session_branches()

    # First two children spawned successfully — graceful surface, no hang.
    assert orch.stats.children_spawned == 2


# =============================================================================
# 15. Phase replay: warmup + measurement use independent orchestrators.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_phase_replay_independent_orchestrator_state(
    strategy: TimingMode,
) -> None:
    """A second BranchOrchestrator (per-phase fresh) must not see leaked
    ``_pre_dispatched_branches`` from the first phase."""
    branch = _make_branch(
        "root:0",
        ["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"], **_ts_kwargs(strategy, 0)),
            TurnMetadata(**_ts_kwargs(strategy, 1)),
        ],
        branches=[branch],
    )
    child = ConversationMetadata(
        conversation_id="early", turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
    )
    cs = _mk_source([root, child])
    issuer = _mk_issuer()

    warmup = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await warmup.dispatch_pre_session_branches()
    assert ("root", "root:0") in warmup._pre_dispatched_branches
    warmup.cleanup()

    # Fresh orchestrator for the next phase.
    measurement = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    assert ("root", "root:0") not in measurement._pre_dispatched_branches
    assert measurement.stats.children_spawned == 0


# =============================================================================
# 16. Combined: pre-session + delayed join + fan-in in one conversation.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_combined_pre_session_delayed_fan_in(strategy: TimingMode) -> None:
    """One conversation exercising all three Phase 2b/1/3 features together.
    Phase 2b: pre-session background SPAWN.
    Phase 1: delayed join K=3 on a different branch.
    Phase 3: fan-in (two prereqs) on a later turn.
    """
    branches = [
        _make_branch(
            "early",
            ["bg1"],
            mode=ConversationBranchMode.SPAWN,
            dispatch_timing="pre",
        ),
        _make_branch("a", ["ca"]),  # delayed join
        _make_branch("b", ["cb"]),  # fan-in partner
    ]
    parent_turns = [
        TurnMetadata(branch_ids=["early", "a"], **_ts_kwargs(strategy, 0)),
        TurnMetadata(**_ts_kwargs(strategy, 1)),
        TurnMetadata(**_ts_kwargs(strategy, 2)),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="a"),
            ],
            **_ts_kwargs(strategy, 3),
        ),
        TurnMetadata(branch_ids=["b"], **_ts_kwargs(strategy, 4)),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b"),
                # Re-consumer of "a" -- Phase 3 multi-consumer.
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="a"),
            ],
            **_ts_kwargs(strategy, 5),
        ),
    ]
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=branches
    )
    children = [
        ConversationMetadata(
            conversation_id=cid, turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
        )
        for cid in ("bg1", "ca", "cb")
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Pre-session dispatch (bg1).
    await orch.dispatch_pre_session_branches()
    assert orch.stats.children_spawned == 1

    # Turn 0: spawn 'early' (filtered) + 'a'.
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=6))
    # Both 'a' child and possibly 'early' but early is pre-dispatched.
    # children_spawned now 2 (bg1 + ca). ca is the only child with a gate.
    assert orch.stats.children_spawned == 2

    # Find ca by prereq_key.
    ca_corr = next(
        cc
        for cc, ents in orch._child_to_join.items()
        if ents and ents[0].prereq_key == "SPAWN_JOIN:a"
    )
    # Complete ca early; gate at turn 3 future-popped, gate at turn 5 needs both.
    await orch.on_child_leaf_reached(ca_corr)

    # Turns 1, 2 — no suspend. Turn 3 also no suspend (a already complete, future-popped).
    for t in range(1, 4):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=6))
        assert s is False, f"{strategy}: turn {t} should not suspend"

    # Turn 4 spawns b.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=4, num_turns=6))
    # Next turn (5) is gated and b is not yet complete -> suspend.
    assert s is True

    cb_corr = next(
        cc
        for cc, ents in orch._child_to_join.items()
        if ents and ents[0].prereq_key == "SPAWN_JOIN:b"
    )
    await orch.on_child_leaf_reached(cb_corr)
    # Gate at 5 should fire (a already done, b just done).
    issuer.dispatch_join_turn.assert_awaited_once()
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 5


# =============================================================================
# 17. Mixed FORK + SPAWN at same parent turn.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_mixed_fork_and_spawn_at_same_turn(strategy: TimingMode) -> None:
    """Branch A is FORK with 2 children; Branch B is SPAWN with 2 children;
    both gated at T+1. Verify both gate correctly and FORK children acquire
    sticky-router refcounts while SPAWN children do not."""
    branches = [
        _make_branch("a", ["fa1", "fa2"], mode=ConversationBranchMode.FORK),
        _make_branch("b", ["sb1", "sb2"], mode=ConversationBranchMode.SPAWN),
    ]
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["a", "b"], **_ts_kwargs(strategy, 0)),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="a"),
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b"),
                ],
                **_ts_kwargs(strategy, 1),
            ),
        ],
        branches=branches,
    )
    children = [
        ConversationMetadata(
            conversation_id=cid, turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
        )
        for cid in ("fa1", "fa2", "sb1", "sb2")
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()

    sticky = MagicMock()
    sticky.register_child_routing = MagicMock()
    sticky.release_child_routing = MagicMock()

    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))

    # FORK children: 2 sticky registrations. SPAWN children: 0.
    assert sticky.register_child_routing.call_count == 2

    # Drain all four; gate fires.
    for child_corr in list(orch._child_to_join.keys()):
        await orch.on_child_leaf_reached(child_corr)

    issuer.dispatch_join_turn.assert_awaited_once()
    assert orch.stats.children_completed == 4
    # 2 FORK release_child_routing calls on completion.
    assert sticky.release_child_routing.call_count == 2


# =============================================================================
# 18. High K under FixedSchedule (K=50).
# =============================================================================


@pytest.mark.asyncio
async def test_high_k_50_intermediate_turns_dispatch_normally() -> None:
    """K=50: parent has 50 turns with timestamps spread across 30 seconds.
    Children fire concurrently. Verify timing fidelity: parent's intermediate
    turns are NOT blocked by the orchestrator (intercept returns False on
    every non-final-pre-gate turn)."""
    K = 50
    branch = _make_branch("root:0", ["c1"])
    parent_turns = [TurnMetadata(branch_ids=["root:0"], timestamp_ms=0)]
    for i in range(1, K):
        parent_turns.append(TurnMetadata(timestamp_ms=int(i * 600)))  # 600ms steps
    parent_turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0")
            ],
            timestamp_ms=int(K * 600),
        )
    )
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=[branch]
    )
    child = ConversationMetadata(
        conversation_id="c1", turns=[TurnMetadata(timestamp_ms=100)]
    )
    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    suspend_count = 0
    for t in range(K + 1):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=K + 1))
        if s:
            suspend_count += 1
        # Complete child once (during the gap) so the gate is satisfied at K=49.
        if t == 5 and orch._child_to_join:
            [child_corr] = list(orch._child_to_join.keys())
            await orch.on_child_leaf_reached(child_corr)

    # Child finished early -> 0 suspensions across the whole 50-turn parent.
    assert suspend_count == 0
    assert orch.stats.parents_suspended == 0


# =============================================================================
# 19. Background spawn at turn N with long-running child outliving parent.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_background_spawn_child_outlives_parent(strategy: TimingMode) -> None:
    """Background branch with no gate: parent completes turn 2 (final) while
    the child is still in flight. ``has_pending_branch_work()`` must remain
    True until the child completes; cleanup leak diagnostic must NOT fire
    after the child completes."""
    branch = _make_branch(
        "root:0",
        ["bg"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=["root:0"], **_ts_kwargs(strategy, 0)),
            TurnMetadata(**_ts_kwargs(strategy, 1)),
            TurnMetadata(**_ts_kwargs(strategy, 2)),
        ],
        branches=[branch],
    )
    child = ConversationMetadata(
        conversation_id="bg", turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
    )
    cs = _mk_source([root, child])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=3))
    [bg_corr] = list(orch._child_to_join.keys())

    # Parent dispatches turns 1, 2 — neither suspends (background).
    for t in (1, 2):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=3))
        assert s is False
    # Parent done; orchestrator still has pending background work.
    assert orch.has_pending_branch_work() is True

    # Child completes long after parent.
    await orch.on_child_leaf_reached(bg_corr)
    assert orch.has_pending_branch_work() is False


# =============================================================================
# 20. Phase shutdown timeout with stuck child + fail-fast.
# =============================================================================


@pytest.mark.asyncio
async def test_fail_fast_aborts_parent_on_child_error(monkeypatch) -> None:
    """With ``AIPERF_DAG_FAIL_FAST=true`` the parent's pending join is dropped
    and the parent is aborted on child error. Without the flag the error is
    treated as leaf-reached (gate decrements normally)."""
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
    child_corrs = list(orch._child_to_join.keys())
    assert len(child_corrs) == 2

    # First child errors -> abort parent; orphan sibling also aborted.
    await orch.on_child_errored(child_corrs[0])

    # Parent abort_session called.
    issuer.abort_session.assert_any_await("p")
    # Pending join purged.
    assert "p" not in orch._active_joins
    assert "p" not in orch._future_joins
    # Stat increment.
    assert orch.stats.parents_failed_due_to_child_error == 1
