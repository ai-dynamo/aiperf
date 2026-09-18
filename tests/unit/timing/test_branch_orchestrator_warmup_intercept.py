# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WARMUP-phase short-circuit in ``BranchOrchestrator.intercept``."""

from __future__ import annotations

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
from aiperf.credit.dispatch import ChildDispatchResult
from aiperf.credit.structs import Credit
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.branch_orchestrator import BranchOrchestrator


def _spawn_declaring_source() -> MagicMock:
    """Conversation source whose turn 0 declares one FORK branch with 2 kids."""
    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id="root:0",
            child_conversation_ids=["a", "b"],
            dispatch_timing="post",
            mode=ConversationBranchMode.FORK,
            is_background=False,
        ),
    ]
    parent_meta.turns = [MagicMock(branch_ids=["root:0"])]
    cs.get_metadata = MagicMock(return_value=parent_meta)
    cs.start_branch_child = MagicMock(
        side_effect=lambda *, child_conversation_id, **kw: MagicMock(
            x_correlation_id=f"child-{child_conversation_id}"
        )
    )
    return cs


def _credit(phase: CreditPhase) -> MagicMock:
    return MagicMock(
        phase=phase,
        x_correlation_id="root",
        conversation_id="c",
        turn_index=0,
        agent_depth=0,
        effective_root_correlation_id="root",
    )


@pytest.mark.asyncio
async def test_warmup_credit_short_circuits_without_spawning():
    """A WARMUP credit returns False, spawns nothing, and leaks no state."""
    cs = _spawn_declaring_source()
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    sticky_router = MagicMock()

    orch = BranchOrchestrator(
        conversation_source=cs,
        credit_issuer=issuer,
        sticky_router=sticky_router,
    )

    result = await orch.intercept(_credit(CreditPhase.WARMUP))

    assert result is False
    # Short-circuit fires before any conversation-source / spawn work.
    cs.get_metadata.assert_not_called()
    cs.start_branch_child.assert_not_called()
    issuer.dispatch_first_turn.assert_not_awaited()
    sticky_router.register_child_routing.assert_not_called()
    # No descendant-count leak -> all_credits_returned_event cannot wedge.
    assert orch._descendant_counts == {}
    assert orch.stats.children_spawned == 0


@pytest.mark.asyncio
async def test_profiling_credit_with_same_source_does_process():
    """The identical fixture at PROFILING spawns the declared children, pinning the short-circuit to phase alone."""
    cs = _spawn_declaring_source()
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    sticky_router = MagicMock()

    orch = BranchOrchestrator(
        conversation_source=cs,
        credit_issuer=issuer,
        sticky_router=sticky_router,
    )

    result = await orch.intercept(_credit(CreditPhase.PROFILING))

    # No gate on the next turn -> intercept returns False, but it DID spawn.
    assert result is False
    cs.get_metadata.assert_called()
    assert cs.start_branch_child.call_count == 2
    assert issuer.dispatch_first_turn.await_count == 2
    assert orch.stats.children_spawned == 2


@pytest.mark.asyncio
async def test_quota_deferred_child_starts_preserve_parent_join_for_handoff():
    """Warmup quota deferral retains the exact parent-to-child join graph."""
    branch_id = "root:spawn"
    child_ids = ["child-a", "child-b", "child-c"]
    branch = ConversationBranchInfo(
        branch_id=branch_id,
        child_conversation_ids=child_ids,
        mode=ConversationBranchMode.SPAWN,
    )
    parent = ConversationMetadata(
        conversation_id="root",
        turns=[
            TurnMetadata(branch_ids=[branch_id]),
            TurnMetadata(
                prerequisites=[
                    TurnPrerequisite(
                        kind=PrerequisiteKind.SPAWN_JOIN,
                        branch_id=branch_id,
                    )
                ]
            ),
        ],
        branches=[branch],
    )
    children = [
        ConversationMetadata(
            conversation_id=child_id,
            turns=[TurnMetadata()],
            is_root=False,
            agent_depth=1,
            parent_conversation_id="root",
        )
        for child_id in child_ids
    ]
    source = MagicMock()
    source.dataset_metadata = DatasetMetadata(
        conversations=[parent, *children],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )
    source.get_metadata.return_value = parent

    def start_child(*, child_conversation_id: str, **_kwargs):
        child = MagicMock()
        child.conversation_id = child_conversation_id
        child.x_correlation_id = f"corr-{child_conversation_id}"
        return child

    source.start_branch_child.side_effect = start_child
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=ChildDispatchResult.DEFERRED)
    orchestrator = BranchOrchestrator(
        conversation_source=source,
        credit_issuer=issuer,
        allow_accelerated_warmup=True,
    )
    orchestrator.start_accelerated_warmup()
    parent_credit = Credit(
        id=0,
        phase=CreditPhase.WARMUP,
        conversation_id="root",
        x_correlation_id="root-corr",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.SPAWN,
    )

    assert await orchestrator.intercept(parent_credit) is True

    blocked, memberships = orchestrator.snapshot_annotations()
    assert blocked == {"root-corr": 1}
    assert memberships == {
        f"corr-{child_id}": [(branch_id, 1)] for child_id in child_ids
    }
    assert orchestrator.stats.children_spawned == 3
    assert orchestrator.stats.children_truncated == 0
