# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

from aiperf.common.enums import (
    ConversationBranchMode,
    PrerequisiteKind,
)
from aiperf.common.models import (
    ConversationBranchInfo,
    ConversationMetadata,
    DatasetMetadata,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator, PendingBranchJoin


def test_pending_branch_join_carries_parent_metadata():
    p = PendingBranchJoin(
        parent_x_correlation_id="corr-1",
        parent_conversation_id="conv-1",
        parent_num_turns=5,
        parent_agent_depth=0,
        parent_parent_correlation_id=None,
        gated_turn_index=1,
    )
    assert p.parent_conversation_id == "conv-1"
    assert p.parent_num_turns == 5
    assert p.gated_turn_index == 1
    assert p.outstanding == {}
    assert p.is_satisfied  # no prereqs -> satisfied trivially


def _mk_conv(
    cid: str, turns: list[TurnMetadata], branches: list[ConversationBranchInfo]
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


def test_orchestrator_builds_prereq_index():
    branch = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.SPAWN,
    )
    conv = _mk_conv(
        "r",
        [
            TurnMetadata(branch_ids=["r:0"]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(kind=PrerequisiteKind.SPAWN_JOIN, branch_id="r:0")
                ]
            ),
        ],
        [branch],
    )
    cs = _mk_source([conv])
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())
    # Spawn on turn 0 gates turn 1 for branch r:0.
    entries = orch._prereq_index.get(("r", 0), [])
    assert [(b, g) for b, g, _ in entries] == [("r:0", 1)]


def test_orchestrator_ignores_conversations_without_prereqs():
    branch = ConversationBranchInfo(
        branch_id="r:0",
        child_conversation_ids=["c"],
        mode=ConversationBranchMode.FORK,
    )
    conv = _mk_conv("r", [TurnMetadata(branch_ids=["r:0"])], [branch])
    cs = _mk_source([conv])
    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=MagicMock())
    assert orch._prereq_index == {}
