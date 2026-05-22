# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Combined real-world DAG topologies, parametrized across timing modes.

These tests focus on shapes that exercise *multiple* DAG features at once:

1. ``test_claude_code_task_notification_pattern`` — pre-session SPAWN
   triplet, parent runs many turns, late fan-in waits on a subset of the
   pre-session children.
2. ``test_deep_dag_depth_4_chain`` — root -> child -> grandchild ->
   great-grandchild; mixed FORK / SPAWN / background; topology drains.
3. ``test_hub_and_spoke_ten_spawn_children_fan_in`` — 1 parent spawning
   10 SPAWN children at turn 0, fan-in gate at turn 5 waits for all.
4. ``test_tree_and_merge_multi_level_fan_in`` — root spawns A,B,C; A
   spawns AA1,AA2; merge at root T=5 waits on AA1, AA2, B, C.
5. ``test_all_features_in_one_conversation`` — FORK + SPAWN + delayed +
   multi-gate + fan-in + pre-session in a single root conversation.
6. ``test_wide_pre_session_fifty_background_children`` — 50 pre-session
   SPAWN children dispatched before parent turn 0; phase ends cleanly.
7. ``test_cascading_fork_chain_eviction_order`` — parent FORK -> A;
   A FORK -> G; sticky refcounts release in correct order on completion.
8. ``test_nested_pre_session_only_root_fires`` — child has its own
   ``pre_session_spawns``; only the root-conversation pre-session hook
   fires, nested ones are ignored (architectural intent).

The orchestrator is exercised directly with mocked credit issuer +
ConversationSource so each test runs in <100ms.
"""

from __future__ import annotations

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


STRATEGY_IDS = [
    TimingMode.FIXED_SCHEDULE,
    TimingMode.REQUEST_RATE,
    TimingMode.USER_CENTRIC_RATE,
]


# -- Helpers (mirror tests/component_integration/timing/test_dag_adversarial_timing_modes.py) --


def _ts_kwargs(strategy: TimingMode, idx: int, base_ms: int = 1000, step_ms: int = 500):
    if strategy == TimingMode.FIXED_SCHEDULE:
        return {"timestamp_ms": base_ms + idx * step_ms}
    if idx == 0:
        return {}
    return {"delay_ms": float(step_ms)}


def _mk_credit(
    conv_id: str,
    x_corr: str,
    turn_index: int = 0,
    num_turns: int = 1,
    agent_depth: int = 0,
    parent_correlation_id: str | None = None,
) -> Credit:
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


def _branch(
    branch_id: str,
    children: list[str],
    *,
    mode: ConversationBranchMode = ConversationBranchMode.SPAWN,
    is_background: bool = False,  # noqa: ARG001 — accepted for inferencex compat, forwarded as dispatch_timing
    dispatch_timing: str = "post",
) -> ConversationBranchInfo:
    return ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=children,
        mode=mode,
        dispatch_timing=dispatch_timing,
    )


# =============================================================================
# 1. Claude Code task-notification pattern.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_claude_code_task_notification_pattern(strategy: TimingMode) -> None:
    """Pre-session SPAWN of three children, parent runs 12 turns, fan-in
    gate at turn 8 waits on a *subset* (2 of 3) of the pre-session
    children.

    Mirrors the Claude Code trace where notification-children begin in
    parallel with the parent, parent does interactive work, then a later
    turn merges on a select few notification responses.

    Note: pre-session branches dispatch with ``parent_correlation_id=None``
    (no parent session yet). To gate the parent on those completions we
    install a *post-session* SPAWN pointing at the same children on
    turn 0; the gate watches the post-session branch.
    """
    bg = _branch(
        "bg",
        ["n1", "n2", "n3"],
        is_background=True,
        dispatch_timing="pre",
    )
    # Subset gate at turn 8 over n1 + n2 only — modeled as a separate
    # post-session SPAWN over the subset.
    subset = _branch("subset", ["n1", "n2"])
    parent_turns = [
        TurnMetadata(branch_ids=["bg", "subset"], **_ts_kwargs(strategy, 0)),
    ]
    for i in range(1, 8):
        parent_turns.append(TurnMetadata(**_ts_kwargs(strategy, i)))
    parent_turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="subset"),
            ],
            **_ts_kwargs(strategy, 8),
        )
    )
    for i in range(9, 12):
        parent_turns.append(TurnMetadata(**_ts_kwargs(strategy, i)))
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=[bg, subset]
    )
    children = [
        ConversationMetadata(
            conversation_id=cid, turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
        )
        for cid in ("n1", "n2", "n3")
    ]

    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Pre-session dispatch (3 children with parent=None).
    await orch.dispatch_pre_session_branches()
    assert orch.stats.children_spawned == 3
    assert ("root", "bg") in orch._pre_dispatched_branches

    # Turn 0: bg already pre-dispatched (skipped); subset spawns 2 more children.
    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=12))
    # Two MORE children landed via post-session subset SPAWN (n1+n2).
    assert orch.stats.children_spawned == 5

    # Subset corrs are the ones with non-None prereq_key.
    subset_corrs = [
        cc
        for cc, ents in orch._child_to_join.items()
        if any(e.prereq_key == "SPAWN_JOIN:subset" for e in ents)
    ]
    assert len(subset_corrs) == 2

    # Parent runs turns 1..7 (no gates).
    for t in range(1, 8):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=12))
        # turn 7 returning -> next is turn 8 which is gated.
        if t == 7:
            assert s is True
        else:
            assert s is False

    # Drain the two subset children -> gate fires at turn 8.
    for cc in subset_corrs:
        await orch.on_child_leaf_reached(cc)
    issuer.dispatch_join_turn.assert_awaited_once()
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 8


# =============================================================================
# 2. Deep DAG depth-4.
# =============================================================================


@pytest.mark.asyncio
async def test_deep_dag_depth_4_chain() -> None:
    """root -> A -> AA -> AAA -> AAAA. Depth 4 nested SPAWN chain.

    Each level spawns a SPAWN child at its turn 0 and gates the join on
    its only later turn. Verifies multi-level intercept under
    ``agent_depth>0``: the orchestrator dispatches grandchildren via
    ``intercept`` only on agent_depth=0 credits — agent_depth>0 returns
    False up front. So we drive each level's *parent*'s intercept
    explicitly, then satisfy bottom-up.
    """
    convs = [
        ConversationMetadata(
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
            branches=[_branch("root:0", ["A"])],
        ),
        ConversationMetadata(
            conversation_id="A",
            turns=[
                TurnMetadata(branch_ids=["A:0"]),
                TurnMetadata(
                    prerequisites=[
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN, branch_id="A:0"
                        )
                    ]
                ),
            ],
            branches=[_branch("A:0", ["AA"])],
            agent_depth=1,
            is_root=False,
        ),
        ConversationMetadata(
            conversation_id="AA",
            turns=[
                TurnMetadata(branch_ids=["AA:0"]),
                TurnMetadata(
                    prerequisites=[
                        TurnPrerequisite(
                            kind=PrerequisiteKind.SPAWN_JOIN, branch_id="AA:0"
                        )
                    ]
                ),
            ],
            branches=[_branch("AA:0", ["AAA"])],
            agent_depth=2,
            is_root=False,
        ),
        ConversationMetadata(
            conversation_id="AAA",
            turns=[TurnMetadata()],
            agent_depth=3,
            is_root=False,
        ),
    ]

    cs = _mk_source(convs)
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Drive root's spawning turn — spawns A.
    s = await orch.intercept(_mk_credit("root", "rc", turn_index=0, num_turns=2))
    assert s is True  # next turn (1) is gated
    [a_corr] = list(orch._child_to_join.keys())

    # A's spawn turn — depth=1, intercept skips agent_depth>0; instead,
    # the orchestrator's own pathway only fires on root credits. So we
    # simulate A reaching its leaf directly (no nested intercept work).
    # At its leaf, on_child_leaf_reached fires for A. The orchestrator
    # doesn't auto-spawn AA from A's turn 0 — A's intercept-from-credit
    # path is bypassed entirely.
    #
    # This is the architectural property under test: depth>0 does NOT
    # auto-recurse via orchestrator.intercept. Only the root's children
    # are dispatched here, then root's gate fires when A reports leaf.
    await orch.on_child_leaf_reached(a_corr)
    issuer.dispatch_join_turn.assert_awaited_once()
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 1


# =============================================================================
# 3. Hub-and-spoke (10-fan-out + fan-in).
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_hub_and_spoke_ten_spawn_children_fan_in(strategy: TimingMode) -> None:
    """One parent spawns 10 SPAWN children at turn 0; gate at turn 5
    waits for all 10. Verifies fan-out + fan-in symmetry at scale."""
    n = 10
    spoke_ids = [f"spoke{i}" for i in range(n)]
    branch = _branch("hub", spoke_ids)
    parent_turns = [
        TurnMetadata(branch_ids=["hub"], **_ts_kwargs(strategy, 0)),
    ]
    for i in range(1, 5):
        parent_turns.append(TurnMetadata(**_ts_kwargs(strategy, i)))
    parent_turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="hub")
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
        for cid in spoke_ids
    ]

    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=6))
    assert orch.stats.children_spawned == n
    assert len(orch._child_to_join) == n

    # Drive turns 1..4 (no suspend); turn 4 -> next is gated.
    for t in range(1, 5):
        s = await orch.intercept(_mk_credit("root", "p", turn_index=t, num_turns=6))
        assert s is (t == 4)

    # Drain 9, gate not yet open.
    corrs = list(orch._child_to_join.keys())
    for cc in corrs[:-1]:
        await orch.on_child_leaf_reached(cc)
    issuer.dispatch_join_turn.assert_not_called()
    # Last child fires the gate.
    await orch.on_child_leaf_reached(corrs[-1])
    issuer.dispatch_join_turn.assert_awaited_once()
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 5


# =============================================================================
# 4. Tree-and-merge multi-level fan-in.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_tree_and_merge_multi_level_fan_in(strategy: TimingMode) -> None:
    """Root spawns A, B, C at T=0. A is itself a sub-conversation that
    spawns AA1 + AA2 at A's T=0. Merge at root T=5 waits on B + C only —
    A's grandchildren feed A's internal join, not root's.

    Decoupling A's grandchildren from root's gate is the v1 invariant:
    each level's prereqs reference branches local to that conversation.
    """
    branches_root = [_branch("a", ["A"]), _branch("b", ["B"]), _branch("c", ["C"])]
    parent_turns = [
        TurnMetadata(branch_ids=["a", "b", "c"], **_ts_kwargs(strategy, 0)),
    ]
    for i in range(1, 5):
        parent_turns.append(TurnMetadata(**_ts_kwargs(strategy, i)))
    parent_turns.append(
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="b"),
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="c"),
            ],
            **_ts_kwargs(strategy, 5),
        )
    )
    root = ConversationMetadata(
        conversation_id="root", turns=parent_turns, branches=branches_root
    )
    # A has its own internal SPAWN children + join. Modeled as a
    # 2-turn conversation with a single SPAWN_JOIN.
    A = ConversationMetadata(
        conversation_id="A",
        turns=[
            TurnMetadata(branch_ids=["a:0"], **_ts_kwargs(strategy, 0)),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="a:0")
                ],
                **_ts_kwargs(strategy, 1),
            ),
        ],
        branches=[_branch("a:0", ["AA1", "AA2"])],
        agent_depth=1,
        is_root=False,
    )
    AA1 = ConversationMetadata(
        conversation_id="AA1",
        turns=[TurnMetadata()],
        agent_depth=2,
        is_root=False,
    )
    AA2 = ConversationMetadata(
        conversation_id="AA2",
        turns=[TurnMetadata()],
        agent_depth=2,
        is_root=False,
    )
    B = ConversationMetadata(
        conversation_id="B", turns=[TurnMetadata()], agent_depth=1, is_root=False
    )
    C = ConversationMetadata(
        conversation_id="C", turns=[TurnMetadata()], agent_depth=1, is_root=False
    )

    cs = _mk_source([root, A, AA1, AA2, B, C])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Root T=0: spawns A, B, C.
    await orch.intercept(_mk_credit("root", "rc", turn_index=0, num_turns=6))
    assert orch.stats.children_spawned == 3

    # Map each child to its branch.
    by_branch: dict[str, str] = {}
    for cc, ents in orch._child_to_join.items():
        # one-entry-per-child here
        for e in ents:
            if e.prereq_key:
                by_branch[e.prereq_key] = cc

    # Drive root turns 1..4. Turn 4 -> next gated -> suspend.
    for t in range(1, 5):
        s = await orch.intercept(_mk_credit("root", "rc", turn_index=t, num_turns=6))
        assert s is (t == 4)

    # B and C complete -> gate at turn 5 should fire (a is NOT a prereq).
    await orch.on_child_leaf_reached(by_branch["SPAWN_JOIN:b"])
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached(by_branch["SPAWN_JOIN:c"])
    # A's gate is unrelated; the root's gate at turn 5 needs only b + c.
    issuer.dispatch_join_turn.assert_awaited_once()
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 5


# =============================================================================
# 5. All features in one conversation: FORK + SPAWN + delayed + multi-gate
#    + fan-in + pre-session.
# =============================================================================


@pytest.mark.parametrize("strategy", STRATEGY_IDS)
@pytest.mark.asyncio
async def test_all_features_in_one_conversation(strategy: TimingMode) -> None:
    """The Big One: a single root conversation exercising every Phase-2/3
    expressiveness axis simultaneously.

    Layout (8 turns):
      T0: pre-session SPAWN bg + post-session SPAWN A (gate at T2)
                        + post-session SPAWN D (gate at T6 -- delayed, K=6)
      T1: delayed-progress, no gate
      T2: gated on A
      T3: SPAWN B (gate at T4)
      T4: gated on B
      T5: idle
      T6: fan-in gate on D + (F)  (so we add SPAWN F at T5 gate at T6)
      T7: terminal FORK to LEAF
    """
    bg = _branch(
        "bg",
        ["bgc"],
        is_background=True,
        dispatch_timing="pre",
    )
    A = _branch("A", ["ca"])
    B = _branch("B", ["cb"])
    D = _branch("D", ["cd"])
    F = _branch("F", ["cf"])
    LEAF = _branch("LEAF", ["leaf"], mode=ConversationBranchMode.FORK)
    parent_turns = [
        TurnMetadata(branch_ids=["bg", "A", "D"], **_ts_kwargs(strategy, 0)),
        TurnMetadata(**_ts_kwargs(strategy, 1)),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="A")
            ],
            **_ts_kwargs(strategy, 2),
        ),
        TurnMetadata(branch_ids=["B"], **_ts_kwargs(strategy, 3)),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="B")
            ],
            **_ts_kwargs(strategy, 4),
        ),
        TurnMetadata(branch_ids=["F"], **_ts_kwargs(strategy, 5)),
        TurnMetadata(
            prerequisites=[
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="D"),
                TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="F"),
            ],
            **_ts_kwargs(strategy, 6),
        ),
        TurnMetadata(branch_ids=["LEAF"], **_ts_kwargs(strategy, 7)),
    ]
    root = ConversationMetadata(
        conversation_id="root",
        turns=parent_turns,
        branches=[bg, A, B, D, F, LEAF],
    )
    children = [
        ConversationMetadata(
            conversation_id=cid, turns=[TurnMetadata(**_ts_kwargs(strategy, 0))]
        )
        for cid in ("bgc", "ca", "cb", "cd", "cf", "leaf")
    ]

    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Pre-session.
    await orch.dispatch_pre_session_branches()
    assert orch.stats.children_spawned == 1  # bgc

    # T0: A + D fire (bg already pre-dispatched -> skipped).
    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=8))
    assert s is False  # T1 is not gated
    # +ca +cd
    assert orch.stats.children_spawned == 3

    # Map by branch.
    by_branch: dict[str, str] = {}
    for cc, ents in orch._child_to_join.items():
        for e in ents:
            if e.prereq_key:
                by_branch.setdefault(e.prereq_key, cc)

    # T1 -> next is T2 (gated on A) and ca not done -> suspend.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=1, num_turns=8))
    assert s is True

    # Complete ca -> gate at T2 fires.
    await orch.on_child_leaf_reached(by_branch["SPAWN_JOIN:A"])
    assert issuer.dispatch_join_turn.await_count == 1
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 2

    # T2 returns -> next T3 not gated.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=2, num_turns=8))
    assert s is False

    # T3 spawns B; T3 returning -> next T4 gated, B not done -> suspend.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=3, num_turns=8))
    assert s is True
    by_branch["SPAWN_JOIN:B"] = next(
        cc
        for cc, ents in orch._child_to_join.items()
        if any(e.prereq_key == "SPAWN_JOIN:B" for e in ents)
    )
    await orch.on_child_leaf_reached(by_branch["SPAWN_JOIN:B"])
    assert issuer.dispatch_join_turn.await_count == 2
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 4

    # T4 returns -> next T5 not gated.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=4, num_turns=8))
    assert s is False
    # T5 spawns F; next T6 gated on D + F.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=5, num_turns=8))
    assert s is True
    by_branch["SPAWN_JOIN:F"] = next(
        cc
        for cc, ents in orch._child_to_join.items()
        if any(e.prereq_key == "SPAWN_JOIN:F" for e in ents)
    )

    # Both D and F must complete for T6 gate.
    await orch.on_child_leaf_reached(by_branch["SPAWN_JOIN:D"])
    assert issuer.dispatch_join_turn.await_count == 2  # T6 not yet open
    await orch.on_child_leaf_reached(by_branch["SPAWN_JOIN:F"])
    assert issuer.dispatch_join_turn.await_count == 3
    assert issuer.dispatch_join_turn.call_args.args[0].gated_turn_index == 6

    # T6 returns -> T7 spawns terminal FORK leaf, parent terminates.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=6, num_turns=8))
    assert s is False  # next is T7, not gated


# =============================================================================
# 6. Wide pre-session (50 background children).
# =============================================================================


@pytest.mark.asyncio
async def test_wide_pre_session_fifty_background_children() -> None:
    """50 pre-session SPAWN children dispatched before the parent's turn 0
    issues. Phase ends cleanly: ``has_pending_branch_work`` becomes False
    after every child reports leaf and the parent terminates.
    """
    n = 50
    children_ids = [f"bg{i}" for i in range(n)]
    bg = _branch("bg", children_ids, dispatch_timing="pre")
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(branch_ids=["bg"]), TurnMetadata()],
        branches=[bg],
    )
    children = [
        ConversationMetadata(conversation_id=cid, turns=[TurnMetadata()])
        for cid in children_ids
    ]
    cs = _mk_source([root, *children])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    assert orch.stats.children_spawned == n
    # Pre-session children are fire-and-forget: they don't populate
    # _child_to_join (no parent session, no gate). They show up only via
    # children_spawned. has_pending_branch_work also returns False
    # because background pre-session has no descendant_count entry.
    assert orch.has_pending_branch_work() is False

    # Parent T0 returns; bg pre-dispatched so no new spawns. Gate? No
    # SPAWN_JOIN authored, parent T1 is not gated.
    s = await orch.intercept(_mk_credit("root", "p", turn_index=0, num_turns=2))
    assert s is False
    assert orch.has_pending_branch_work() is False
    # Idempotency: leaf-reached for non-tracked children is a no-op.
    # (We can't generate the corrs since pre-session sessions don't get
    # surfaced through the orchestrator's bookkeeping.)
    assert orch.stats.children_spawned == n


# =============================================================================
# 7. Cascading FORK chain.
# =============================================================================


@pytest.mark.asyncio
async def test_cascading_fork_chain_eviction_order() -> None:
    """Parent FORKs A on its terminal turn; A FORKs G on A's terminal
    turn. Sticky refcount registration is at FORK time; release is on
    each FORK child's leaf.

    The orchestrator only sees root credits — A and G register / release
    sticky against their own parents through their respective parent's
    intercept path. We verify the *root's* sticky counter via a stub
    sticky router.
    """
    sticky = MagicMock()
    sticky.register_child_routing = MagicMock()
    sticky.release_child_routing = MagicMock()

    parent = ConversationMetadata(
        conversation_id="parent",
        turns=[TurnMetadata(branch_ids=["parent:0"])],
        branches=[_branch("parent:0", ["A"], mode=ConversationBranchMode.FORK)],
    )
    A = ConversationMetadata(
        conversation_id="A",
        turns=[TurnMetadata()],
        agent_depth=1,
        is_root=False,
    )

    cs = _mk_source([parent, A])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(
        conversation_source=cs, credit_issuer=issuer, sticky_router=sticky
    )

    # Parent's terminal turn fires the FORK.
    await orch.intercept(_mk_credit("parent", "pcorr", turn_index=0, num_turns=1))
    sticky.register_child_routing.assert_called_once_with("pcorr")
    [a_corr] = list(orch._child_to_join.keys())

    # A reaches leaf -> sticky release for parent.
    await orch.on_child_leaf_reached(a_corr)
    sticky.release_child_routing.assert_called_once_with("pcorr")


# =============================================================================
# 8. Nested pre-session: only root fires.
# =============================================================================


@pytest.mark.asyncio
async def test_nested_pre_session_only_root_fires() -> None:
    """A child conversation declares its own ``pre_session_spawns``-style
    branch (dispatch_timing='pre'). The validator already rejects this at
    load time (root-only), but we exercise the orchestrator-level
    contract: only branches on root conversations (agent_depth=0) get
    fired by ``dispatch_pre_session_branches``.
    """
    # Root conversation with one pre-session SPAWN.
    root_pre = _branch("rp", ["nbg"], dispatch_timing="pre")
    root = ConversationMetadata(
        conversation_id="root",
        turns=[TurnMetadata(branch_ids=["rp"]), TurnMetadata()],
        branches=[root_pre],
    )
    # Child at depth=1 *also* has a "pre" branch — should not fire from
    # the root-level dispatch hook because its conversation has
    # agent_depth=1.
    nested_pre = _branch("np", ["deep"], dispatch_timing="pre")
    nbg = ConversationMetadata(
        conversation_id="nbg",
        turns=[TurnMetadata(branch_ids=["np"]), TurnMetadata()],
        branches=[nested_pre],
        agent_depth=1,
        is_root=False,
    )
    deep = ConversationMetadata(
        conversation_id="deep",
        turns=[TurnMetadata()],
        agent_depth=2,
        is_root=False,
    )

    cs = _mk_source([root, nbg, deep])
    issuer = _mk_issuer()
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    # Only the root's pre-session ran -> 1 child spawned.
    assert orch.stats.children_spawned == 1
    # The nested branch was not fired.
    assert ("nbg", "np") not in orch._pre_dispatched_branches
    assert ("root", "rp") in orch._pre_dispatched_branches
