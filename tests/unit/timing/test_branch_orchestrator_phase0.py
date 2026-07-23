# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase 0 unit tests for :class:`BranchOrchestrator` and :class:`CreditIssuer`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind
from aiperf.common.models import (
    ConversationBranchInfo,
    TurnMetadata,
    TurnPrerequisite,
)
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from tests.unit.timing._shared_helpers import _mk_conv, _mk_source


@pytest.mark.asyncio
async def test_intercept_all_children_failed_with_gate_does_not_hang():
    """When every ``start_branch_child`` raises on a parent turn whose next"""
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

    result = await orch.intercept(credit)
    assert result is False

    assert issuer.dispatch_join_turn.await_count == 0

    assert "root-corr" not in orch._active_joins
    assert "root-corr" not in orch._future_joins
    assert "root-corr" not in orch._descendant_counts
    assert orch.stats.parents_resumed == 0
    assert orch.stats.children_errored == 2
    assert orch.stats.children_spawned == 0
