# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WARMUP-phase short-circuit in ``BranchOrchestrator.intercept``."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
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
    cs.get_metadata.assert_not_called()
    cs.start_branch_child.assert_not_called()
    issuer.dispatch_first_turn.assert_not_awaited()
    sticky_router.register_child_routing.assert_not_called()
    assert orch._descendant_counts == {}
    assert orch.stats.children_spawned == 0


@pytest.mark.asyncio
async def test_profiling_credit_with_same_source_does_process():
    """The identical fixture at PROFILING DOES spawn the declared children."""
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

    assert result is False
    cs.get_metadata.assert_called()
    assert cs.start_branch_child.call_count == 2
    assert issuer.dispatch_first_turn.await_count == 2
    assert orch.stats.children_spawned == 2
