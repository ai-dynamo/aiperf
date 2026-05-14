# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 2b unit tests: pre-session background SPAWN dispatch.

Covers the Phase 2b semantics:

- A branch marked ``dispatch_timing="pre"`` fires via
  ``dispatch_pre_session_branches`` BEFORE the parent's turn 0 credit is
  issued. Children receive ``agent_depth=1`` and
  ``parent_correlation_id=None``.
- When the parent's turn 0 credit later returns, the per-turn spawn path
  skips pre-dispatched branches (records in ``_pre_dispatched_branches``)
  so children are never dispatched twice.
- Mixing a pre-session branch with a post branch on the same turn 0:
  pre-dispatch fires only the pre branch; intercept fires only the post
  branch.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator


def _mk_conv(
    cid: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo],
    agent_depth: int = 0,
    is_root: bool = True,
) -> ConversationMetadata:
    return ConversationMetadata(
        conversation_id=cid,
        turns=turns,
        branches=branches,
        agent_depth=agent_depth,
        is_root=is_root,
    )


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: next(
        c for c in conversations if c.conversation_id == cid
    )

    def _start_pre(child_cid, **kwargs):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_cid}"
        s.conversation_id = child_cid
        s.agent_depth = 1
        s.parent_correlation_id = None
        return s

    def _start_branch(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **kwargs
    ):
        s = MagicMock()
        s.x_correlation_id = f"corr-{child_conversation_id}"
        return s

    cs.start_pre_session_child = MagicMock(side_effect=_start_pre)
    cs.start_branch_child = MagicMock(side_effect=_start_branch)
    return cs


def _mk_credit(conv_id: str, corr_id: str, turn_index: int):
    return MagicMock(
        x_correlation_id=corr_id,
        conversation_id=conv_id,
        turn_index=turn_index,
        agent_depth=0,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )


def _pre_session_metadata() -> list[ConversationMetadata]:
    """Root conversation with a single pre-session SPAWN branch on turn 0."""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:pre"]), TurnMetadata()],
        [pre_branch],
    )
    early = _mk_conv("early", [TurnMetadata()], [])
    return [root, early]


@pytest.mark.asyncio
async def test_pre_session_background_spawn_dispatches_before_turn_0():
    """Pre-session dispatch fires the child BEFORE any parent credit is issued.

    Asserts:
    - ``start_pre_session_child`` is invoked once per child_conversation_id.
    - ``dispatch_first_turn`` is called with that session.
    - Stats record a spawn.
    - The (conv, branch) tuple is recorded in ``_pre_dispatched_branches``.
    """
    cs = _mk_source(_pre_session_metadata())
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()

    cs.start_pre_session_child.assert_called_once_with("early")
    issuer.dispatch_first_turn.assert_awaited_once()
    # Parent has NOT had any credit; no branch_child dispatch happened.
    cs.start_branch_child.assert_not_called()
    assert orch.stats.children_spawned == 1
    assert ("root", "root:pre") in orch._pre_dispatched_branches


@pytest.mark.asyncio
async def test_intercept_skips_pre_dispatched_on_turn_0_credit():
    """On parent turn-0 credit return, intercept must NOT re-dispatch the
    pre-dispatched branch's children."""
    cs = _mk_source(_pre_session_metadata())
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    await orch.dispatch_pre_session_branches()
    assert cs.start_pre_session_child.call_count == 1
    assert issuer.dispatch_first_turn.await_count == 1

    # Parent's turn 0 returns — branch_ids=["root:pre"], but it's already
    # in _pre_dispatched_branches so no new dispatch happens.
    result = await orch.intercept(_mk_credit("root", "corr-root", 0))
    # next turn (T=1) is not gated, so intercept returns False.
    assert result is False
    # No additional start_branch_child calls for the pre-dispatched branch.
    cs.start_branch_child.assert_not_called()
    # dispatch_first_turn count unchanged.
    assert issuer.dispatch_first_turn.await_count == 1


@pytest.mark.asyncio
async def test_mixed_pre_and_post_branches_on_turn_0_no_double_dispatch():
    """Turn 0 declares both a pre-session branch and a normal post-turn
    background SPAWN. Pre-dispatch fires only the pre branch; on turn-0
    credit return, intercept fires only the post branch."""
    pre_branch = ConversationBranchInfo(
        branch_id="root:pre",
        child_conversation_ids=["early"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    post_branch = ConversationBranchInfo(
        branch_id="root:0:spawn",
        child_conversation_ids=["post_child"],
        mode=ConversationBranchMode.SPAWN,
        # dispatch_timing defaults to "post"
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(branch_ids=["root:pre", "root:0:spawn"]),
            TurnMetadata(),
        ],
        [pre_branch, post_branch],
    )
    early = _mk_conv("early", [TurnMetadata()], [])
    post_child = _mk_conv("post_child", [TurnMetadata()], [])
    cs = _mk_source([root, early, post_child])

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Pre-session dispatch: only "early" should start.
    await orch.dispatch_pre_session_branches()
    assert cs.start_pre_session_child.call_count == 1
    cs.start_pre_session_child.assert_called_once_with("early")
    assert issuer.dispatch_first_turn.await_count == 1

    # Parent's turn 0 returns. intercept should fire post_child via
    # start_branch_child exactly once, and skip the pre branch.
    result = await orch.intercept(_mk_credit("root", "corr-root", 0))
    # No gate on T=1; not suspended.
    assert result is False
    cs.start_branch_child.assert_called_once()
    kwargs = cs.start_branch_child.call_args.kwargs
    assert kwargs["child_conversation_id"] == "post_child"
    assert issuer.dispatch_first_turn.await_count == 2


@pytest.mark.asyncio
async def test_pre_session_no_op_when_no_pre_branches():
    """Dispatch hook is safe to call when no branches are marked pre."""
    post_only = ConversationBranchInfo(
        branch_id="root:0:spawn",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
    )
    root = _mk_conv(
        "root",
        [TurnMetadata(branch_ids=["root:0:spawn"]), TurnMetadata()],
        [post_only],
    )
    child = _mk_conv("child", [TurnMetadata()], [])
    cs = _mk_source([root, child])
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.dispatch_pre_session_branches()
    cs.start_pre_session_child.assert_not_called()
    issuer.dispatch_first_turn.assert_not_called()
    assert not orch._pre_dispatched_branches


@pytest.mark.asyncio
async def test_cleanup_clears_pre_dispatched_set():
    """Cleanup must clear ``_pre_dispatched_branches`` alongside other state."""
    cs = _mk_source(_pre_session_metadata())
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.dispatch_pre_session_branches()
    assert orch._pre_dispatched_branches


@pytest.mark.asyncio
async def test_pre_session_skips_spawn_children_with_is_root_false():
    """SPAWN children intentionally keep ``agent_depth == 0`` while carrying
    ``is_root=False``. The dispatch path must filter on ``is_root`` so a
    SPAWN child's own pre-session branches are NOT fired at phase start as
    if the child were an independent root (which would add unauthored
    requests and break trace topology).
    """
    pre_branch = ConversationBranchInfo(
        branch_id="spawn_child:pre",
        child_conversation_ids=["grandchild"],
        mode=ConversationBranchMode.SPAWN,
        dispatch_timing="pre",
    )
    # SPAWN child: keeps agent_depth=0 but is_root=False.
    spawn_child = _mk_conv(
        "spawn_child",
        [TurnMetadata(branch_ids=["spawn_child:pre"]), TurnMetadata()],
        [pre_branch],
        agent_depth=0,
        is_root=False,
    )
    grandchild = _mk_conv("grandchild", [TurnMetadata()], [])
    cs = _mk_source([spawn_child, grandchild])
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.dispatch_pre_session_branches()

    cs.start_pre_session_child.assert_not_called()
    issuer.dispatch_first_turn.assert_not_called()
    assert not orch._pre_dispatched_branches

    orch.cleanup()
    assert not orch._pre_dispatched_branches
