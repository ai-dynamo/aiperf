# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend
from aiperf.timing.branch_orchestrator import PendingBranchJoin


def _make_issuer() -> CreditIssuer:
    """Bare CreditIssuer wired for the blocking ``issue_credit`` join path."""
    issuer = CreditIssuer.__new__(CreditIssuer)
    issuer._phase = CreditPhase.PROFILING
    issuer._phase_index = 0
    issuer._issuing_stopped = False
    issuer._concurrency_manager = MagicMock()
    issuer._stop_checker = MagicMock()
    issuer._stop_checker.can_send_any_turn.return_value = True
    issuer._stop_checker.can_start_new_session.return_value = True
    issuer._stop_checker.can_send_child_turn.return_value = True
    issuer._concurrency_manager.acquire_session_slot = AsyncMock(return_value=True)
    issuer._concurrency_manager.acquire_prefill_slot = AsyncMock(return_value=True)
    issuer._concurrency_manager.acquire_request_slot = AsyncMock(return_value=True)
    issuer._concurrency_manager.release_session_slot = MagicMock()
    issuer._issue_credit_internal = AsyncMock(return_value=True)
    return issuer


@pytest.mark.asyncio
async def test_dispatch_join_turn_reuses_session_slot():
    issuer = _make_issuer()

    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        parent_agent_depth=0,
        parent_parent_correlation_id=None,
        gated_turn_index=2,
    )
    result = await issuer.dispatch_join_turn(pending)
    assert result is True
    # Session slot NOT acquired (turn_index > 0 and agent_depth == 0 means
    # is_session_start is False -> needs_session_slot is False).
    issuer._concurrency_manager.acquire_session_slot.assert_not_called()
    issuer._concurrency_manager.acquire_prefill_slot.assert_called_once()
    sent: TurnToSend = issuer._issue_credit_internal.call_args.args[0]
    assert sent.conversation_id == "conv-parent"
    assert sent.x_correlation_id == "corr-parent"
    assert sent.turn_index == 2
    assert sent.num_turns == 3
    assert sent.agent_depth == 0


@pytest.mark.asyncio
async def test_dispatch_join_turn_suppresses_on_stop():
    issuer = _make_issuer()
    issuer._stop_checker.can_send_any_turn.return_value = False
    issuer._concurrency_manager.acquire_prefill_slot = AsyncMock(return_value=False)

    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    result = await issuer.dispatch_join_turn(pending)
    assert result is False
    issuer._issue_credit_internal.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_join_turn_blocks_on_prefill_saturation():
    """Prefill saturation makes a join wait on blocking acquire rather than dropping it."""
    issuer = _make_issuer()
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    result = await issuer.dispatch_join_turn(pending)
    assert result is True
    issuer._concurrency_manager.acquire_prefill_slot.assert_awaited_once()
    issuer._issue_credit_internal.assert_awaited_once()
