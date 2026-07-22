# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend
from aiperf.timing.branch_orchestrator import PendingBranchJoin


@pytest.mark.asyncio
async def test_dispatch_join_turn_reuses_session_slot():
    issuer = CreditIssuer.__new__(CreditIssuer)
    issuer._phase = CreditPhase.PROFILING
    issuer._phase_index = 0
    issuer._concurrency_manager = MagicMock()
    issuer._stop_checker = MagicMock()
    issuer._stop_checker.can_send_any_turn.return_value = True
    issuer._concurrency_manager.try_acquire_session_slot = MagicMock(return_value=True)
    issuer._concurrency_manager.try_acquire_prefill_slot = MagicMock(return_value=True)
    issuer._concurrency_manager.release_session_slot = MagicMock()
    issuer._issue_credit_internal = AsyncMock(return_value=True)

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
    # is_first_turn is False -> needs_session_slot is False).
    issuer._concurrency_manager.try_acquire_session_slot.assert_not_called()
    issuer._concurrency_manager.try_acquire_prefill_slot.assert_called_once()
    sent: TurnToSend = issuer._issue_credit_internal.call_args.args[0]
    assert sent.conversation_id == "conv-parent"
    assert sent.x_correlation_id == "corr-parent"
    assert sent.turn_index == 2
    assert sent.num_turns == 3
    assert sent.agent_depth == 0


@pytest.mark.asyncio
async def test_dispatch_join_turn_suppresses_on_stop():
    issuer = CreditIssuer.__new__(CreditIssuer)
    issuer._concurrency_manager = MagicMock()
    issuer._stop_checker = MagicMock()
    issuer._stop_checker.can_send_any_turn.return_value = False
    issuer._issue_credit_internal = AsyncMock()

    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    result = await issuer.dispatch_join_turn(pending)
    assert result is False
    issuer._issue_credit_internal.assert_not_called()
