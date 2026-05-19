# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: parent spawns two SPAWN children, parent suspends at spawn,
both children drain, parent's gated turn dispatches via dispatch_join_turn.
"""

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
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator

pytestmark = pytest.mark.component_integration


def _mk_credit(
    conv_id: str, x_corr: str, turn_index: int = 0, agent_depth: int = 0
) -> Credit:
    c = MagicMock(spec=Credit)
    c.conversation_id = conv_id
    c.x_correlation_id = x_corr
    c.turn_index = turn_index
    c.agent_depth = agent_depth
    c.parent_correlation_id = None
    return c


def _mk_source(conversations: list[ConversationMetadata]):
    cs = MagicMock()
    cs.dataset_metadata = DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    lookup = {c.conversation_id: c for c in conversations}
    cs.get_metadata.side_effect = lambda cid: lookup[cid]
    return cs


@pytest.mark.asyncio
async def test_parent_resumes_after_all_children_complete():
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1", "c2"],
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
    c1 = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()])
    c2 = ConversationMetadata(conversation_id="c2", turns=[TurnMetadata()])

    cs = _mk_source([root, c1, c2])

    # start_branch_child returns a fake SampledSession with a unique x_correlation_id.
    child_corrs = iter(["corr-c1", "corr-c2"])

    def _start(
        parent_correlation_id, child_conversation_id, agent_depth, branch_mode, **_kw
    ):
        s = MagicMock()
        s.x_correlation_id = next(child_corrs)
        return s

    cs.start_branch_child.side_effect = _start

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=True)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)

    # Parent completes turn 0.
    parent_credit = _mk_credit("root", "corr-root", turn_index=0)
    suppressed = await orch.intercept(parent_credit)
    assert suppressed is True
    # Parent is blocked at its next turn (gated on turn 1).
    assert "corr-root" in orch._active_joins
    assert orch._active_joins["corr-root"].gated_turn_index == 1

    # Children complete one at a time.
    await orch.on_child_leaf_reached("corr-c1")
    issuer.dispatch_join_turn.assert_not_called()
    await orch.on_child_leaf_reached("corr-c2")

    # Join dispatched exactly once with the correct PendingBranchJoin.
    issuer.dispatch_join_turn.assert_awaited_once()
    sent = issuer.dispatch_join_turn.call_args.args[0]
    assert sent.parent_x_correlation_id == "corr-root"
    assert sent.parent_conversation_id == "root"
    assert sent.gated_turn_index == 1
    assert orch.stats.parents_resumed == 1
    assert orch.stats.joins_suppressed == 0


@pytest.mark.asyncio
async def test_join_suppressed_when_issuer_returns_false():
    branch = ConversationBranchInfo(
        branch_id="root:0",
        child_conversation_ids=["c1"],
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
    c1 = ConversationMetadata(conversation_id="c1", turns=[TurnMetadata()])
    cs = _mk_source([root, c1])
    cs.start_branch_child.return_value = MagicMock(x_correlation_id="corr-c1")

    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.dispatch_join_turn = AsyncMock(return_value=False)

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    await orch.intercept(_mk_credit("root", "corr-root"))
    await orch.on_child_leaf_reached("corr-c1")

    assert orch.stats.parents_resumed == 0
    assert orch.stats.joins_suppressed == 1
