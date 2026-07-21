# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WARMUP-phase short-circuit in ``BranchOrchestrator.intercept``.

Regression for the H100/B200 ``concurrency=16`` warmup hang: warmup is
one-shot per trajectory and the strategy refuses to advance child
continuation turns, so spawning branches during WARMUP leaks
``_descendant_counts`` (children never reach ``is_final_turn``) and wedges
``all_credits_returned_event``. ``intercept`` must therefore return early
for any ``credit.phase == CreditPhase.WARMUP`` BEFORE it touches the
conversation source or spawns children. DAG dispatch only runs in
PROFILING.

These tests share a single spawn-declaring conversation source so the
WARMUP case and the PROFILING case differ ONLY in ``credit.phase`` —
proving the short-circuit is the thing suppressing the spawn, not a
missing branch.
"""

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
    """The identical fixture at PROFILING DOES spawn the declared children.

    This is the contrast that pins the short-circuit to ``phase`` alone:
    same conversation source, same turn-0 branch, only the phase changed.
    """
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
