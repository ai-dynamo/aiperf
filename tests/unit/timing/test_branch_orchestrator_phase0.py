# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 unit tests for :class:`BranchOrchestrator` and :class:`CreditIssuer`.

Covers Phase 0 adjacent bug fixes (still valid under Phase 1's revised
data model):

- ``dispatch_join_turn`` propagates ``parent_branch_mode`` and
  ``parent_has_forks_on_gated_turn`` from :class:`PendingBranchJoin` instead
  of hardcoding FORK.
- ``BranchOrchestrator.intercept`` pops the drained gate silently and returns
  False when every ``start_branch_child`` call fails (no children landed),
  so the strategy's normal continuation dispatches the gated turn exactly
  once - neither hanging the parent on a dead pending join nor
  double-dispatching it from the orchestrator.
"""

from __future__ import annotations

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
from aiperf.credit.structs import TurnToSend
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator, PendingBranchJoin


def _mk_conv(
    cid: str,
    turns: list[TurnMetadata],
    branches: list[ConversationBranchInfo],
) -> ConversationMetadata:
    return ConversationMetadata(conversation_id=cid, turns=turns, branches=branches)


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    cs.get_metadata.side_effect = lambda cid: next(
        c for c in conversations if c.conversation_id == cid
    )
    return cs


# ============================================================
# 0.1. dispatch_join_turn propagates SPAWN parent mode
# ============================================================


@pytest.mark.asyncio
async def test_dispatch_join_turn_preserves_spawn_parent_mode():
    from aiperf.credit.issuer import CreditIssuer

    captured: list[TurnToSend] = []

    async def _try_issue(turn: TurnToSend) -> bool:
        captured.append(turn)
        return True

    issuer = CreditIssuer.__new__(CreditIssuer)
    issuer.try_issue_credit = _try_issue  # type: ignore[assignment]

    pending = PendingBranchJoin(
        parent_x_correlation_id="parent-corr",
        parent_conversation_id="conv",
        parent_num_turns=5,
        parent_agent_depth=1,
        parent_parent_correlation_id="grand",
        gated_turn_index=3,
        parent_branch_mode=ConversationBranchMode.SPAWN,
        parent_has_forks_on_gated_turn=False,
    )
    result = await issuer.dispatch_join_turn(pending)
    assert result is True
    assert len(captured) == 1
    assert captured[0].branch_mode == ConversationBranchMode.SPAWN
    assert captured[0].has_forks is False


@pytest.mark.asyncio
async def test_dispatch_join_turn_preserves_has_forks_on_gated_turn():
    from aiperf.credit.issuer import CreditIssuer

    captured: list[TurnToSend] = []

    async def _try_issue(turn: TurnToSend) -> bool:
        captured.append(turn)
        return True

    issuer = CreditIssuer.__new__(CreditIssuer)
    issuer.try_issue_credit = _try_issue  # type: ignore[assignment]

    pending = PendingBranchJoin(
        parent_x_correlation_id="parent-corr",
        parent_conversation_id="conv",
        parent_num_turns=5,
        parent_agent_depth=0,
        parent_parent_correlation_id=None,
        gated_turn_index=2,
        parent_branch_mode=ConversationBranchMode.FORK,
        parent_has_forks_on_gated_turn=True,
    )
    result = await issuer.dispatch_join_turn(pending)
    assert result is True
    assert len(captured) == 1
    assert captured[0].has_forks is True


# ============================================================
# 0.3. intercept with all-children-failed + gate must not hang
# ============================================================


@pytest.mark.asyncio
async def test_intercept_all_children_failed_defers_gated_turn_to_strategy():
    """When every ``start_branch_child`` raises on a parent turn whose next
    turn is gated, the future join has zero outstanding children and would
    never fire via the child-leaf decrement path. The drained gate must be
    popped silently and intercept must return False: the strategy's normal
    continuation then dispatches turn k+1 exactly once. Dispatching the join
    here AND returning False sent the same turn twice (the callback handler
    falls through to handle_credit_return -> _dispatch_next_turn)."""
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["a", "b"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv = _mk_conv(
        "conv",
        [
            TurnMetadata(branch_ids=["root:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN, branch_id="root:0"
                    )
                ]
            ),
        ],
        [branch],
    )
    cs = _mk_source([conv])
    cs.start_branch_child = MagicMock(side_effect=RuntimeError("boom"))

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=False)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    credit = MagicMock(
        x_correlation_id="root-corr",
        conversation_id="conv",
        turn_index=0,
        agent_depth=0,
        parent_correlation_id=None,
        branch_mode=ConversationBranchMode.FORK,
    )

    # All children errored before any landed: the gate is "satisfied" with
    # zero outstanding, popped silently, and the parent is not suspended.
    result = await orch.intercept(credit)
    assert result is False

    # The single dispatch of turn 1 belongs to the strategy's continuation
    # path; the orchestrator must not issue the join turn itself.
    issuer.dispatch_join_turn.assert_not_awaited()

    # No leaked per-parent state to wedge phase-end draining.
    assert "root-corr" not in orch._active_joins
    assert "root-corr" not in orch._future_joins
    assert "root-corr" not in orch._descendant_counts
    assert orch.stats.parents_resumed == 0
    assert orch.stats.children_errored == 2
    assert orch.stats.children_spawned == 0
