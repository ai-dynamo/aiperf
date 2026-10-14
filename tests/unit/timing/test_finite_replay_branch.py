# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cold-start finite child ownership survives dispatch and return races."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import ConversationBranchInfo, TurnMetadata
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from tests.unit.timing._shared_helpers import _mk_conv, _mk_source


@pytest.mark.asyncio
async def test_equal_timestamp_child_is_spawned_once_across_transport_and_return() -> (
    None
):
    branch = ConversationBranchInfo(
        branch_id="root:child",
        child_conversation_ids=["child"],
        mode=ConversationBranchMode.SPAWN,
        start_timestamp_ms=0.0,
    )
    root = _mk_conv(
        "root",
        [
            TurnMetadata(
                timestamp_ms=0.0, api_time_ms=100.0, branch_ids=[branch.branch_id]
            )
        ],
        [branch],
    )
    root.replay_scope_id = "root"
    child = _mk_conv("child", [TurnMetadata(timestamp_ms=0.0)], [])
    source = _mk_source([root, child])
    child_session = MagicMock(
        x_correlation_id="child-correlation",
        metadata=child,
        effective_root_correlation_id="root-correlation",
    )
    source.start_branch_child.return_value = child_session
    issuer = MagicMock()
    issuer.dispatch_first_turn = AsyncMock(return_value=True)
    issuer.replay_gate.fail_finite = MagicMock()
    orchestrator = BranchOrchestrator(conversation_source=source, credit_issuer=issuer)
    credit = MagicMock(
        finite_replay=True,
        no_request=False,
        phase=CreditPhase.PROFILING,
        conversation_id="root",
        x_correlation_id="root-correlation",
        effective_root_correlation_id="root-correlation",
        turn_index=0,
        agent_depth=0,
        num_turns=1,
    )

    await orchestrator.on_transport_dispatched(credit)
    await orchestrator.intercept(credit)
    await asyncio.gather(*tuple(orchestrator._delayed_dispatch_tasks))

    assert source.start_branch_child.call_count == 1
    assert issuer.dispatch_first_turn.await_count == 1
    assert orchestrator.stats.children_spawned == 1
    issuer.replay_gate.fail_finite.assert_not_called()
