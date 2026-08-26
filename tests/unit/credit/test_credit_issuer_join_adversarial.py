# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for ``CreditIssuer.dispatch_join_turn``."""

from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.credit import issuer as issuer_module
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend
from aiperf.timing.branch_orchestrator import PendingBranchJoin


def _make_issuer() -> CreditIssuer:
    """Build a bare CreditIssuer with mocks sufficient for dispatch_join_turn."""
    issuer = CreditIssuer.__new__(CreditIssuer)
    issuer._phase = CreditPhase.PROFILING
    issuer._phase_index = 0
    issuer._issuing_stopped = False
    issuer._concurrency_manager = MagicMock()
    issuer._concurrency_manager.acquire_session_slot = AsyncMock(return_value=True)
    issuer._concurrency_manager.acquire_prefill_slot = AsyncMock(return_value=True)
    issuer._concurrency_manager.acquire_request_slot = AsyncMock(return_value=True)
    issuer._concurrency_manager.release_session_slot = MagicMock()
    issuer._stop_checker = MagicMock()
    issuer._stop_checker.can_send_any_turn.return_value = True
    issuer._stop_checker.can_start_new_session.return_value = True
    issuer._stop_checker.can_send_child_turn.return_value = True
    issuer._issue_credit_internal = AsyncMock(return_value=True)
    return issuer


@pytest.mark.asyncio
async def test_dispatch_join_turn_asserts_gated_turn_index_not_none():
    """gated_turn_index=None must trip the precondition assertion."""
    issuer = _make_issuer()
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        gated_turn_index=None,
    )
    with pytest.raises(AssertionError, match="gated_turn_index"):
        await issuer.dispatch_join_turn(pending)


@pytest.mark.asyncio
async def test_dispatch_join_turn_reports_true_for_issued_but_final_credit():
    """Regression (#5): a join that IS issued but happens to be the phase's
    final credit must report True (resumed), not False. ``_issue_credit_internal``
    returning False signals "was the final credit" (still issued); dispatch must
    not surface that as suppression -- previously the orchestrator miscounted it
    as ``joins_suppressed``."""
    issuer = _make_issuer()
    issuer._issue_credit_internal = AsyncMock(return_value=False)
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    result = await issuer.dispatch_join_turn(pending)
    assert result is True
    issuer._issue_credit_internal.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_join_turn_returns_true_when_issue_credit_returns_true_and_builds_correct_turn():
    """Happy path: True propagates and TurnToSend carries all PendingBranchJoin fields."""
    issuer = _make_issuer()
    captured: dict[str, TurnToSend] = {}

    async def fake_issue(turn: TurnToSend):
        captured["turn"] = turn
        return True

    issuer._issue_credit_internal = fake_issue
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=4,
        parent_agent_depth=1,
        parent_parent_correlation_id="corr-grandparent",
        gated_turn_index=2,
    )

    result = await issuer.dispatch_join_turn(pending)
    assert result is True

    turn = captured["turn"]
    assert turn.turn_index == pending.gated_turn_index
    assert turn.agent_depth == pending.parent_agent_depth
    assert turn.parent_correlation_id == pending.parent_parent_correlation_id
    assert turn.conversation_id == pending.parent_conversation_id
    assert turn.x_correlation_id == pending.parent_x_correlation_id
    assert turn.num_turns == pending.parent_num_turns
    # Hardcoded "not first turn" semantics (driven by turn_index > 0).
    assert turn.turn_index != 0
    assert turn.has_forks is False
    assert turn.branch_mode == ConversationBranchMode.FORK


@pytest.mark.asyncio
async def test_dispatch_join_turn_hardcodes_branch_mode_fork_even_for_spawn_parent():
    """The issuer hardcodes branch_mode=FORK even when the parent was a SPAWN rejoin."""
    issuer = _make_issuer()
    captured: dict[str, TurnToSend] = {}

    async def fake_issue(turn: TurnToSend):
        captured["turn"] = turn
        return True

    issuer._issue_credit_internal = fake_issue
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-spawn-parent",
        parent_conversation_id="conv-spawn-parent",
        parent_num_turns=2,
        gated_turn_index=1,
    )

    await issuer.dispatch_join_turn(pending)
    assert captured["turn"].branch_mode == ConversationBranchMode.FORK


@pytest.mark.asyncio
async def test_dispatch_join_turn_with_gated_turn_index_zero_edge_behavior():
    """gated_turn_index=0 passes the assertion and builds a turn with turn_index=0."""
    issuer = _make_issuer()
    captured: dict[str, TurnToSend] = {}

    async def fake_issue(turn: TurnToSend):
        captured["turn"] = turn
        return True

    issuer._issue_credit_internal = fake_issue
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        gated_turn_index=0,
    )

    result = await issuer.dispatch_join_turn(pending)
    assert result is True
    assert captured["turn"].turn_index == 0


@pytest.mark.asyncio
async def test_multiple_parents_dispatch_join_turn_isolated_state():
    """Sequential dispatches for different parents don't leak fields between calls."""
    issuer = _make_issuer()
    captured: list[TurnToSend] = []

    async def fake_issue(turn: TurnToSend):
        captured.append(turn)
        return True

    issuer._issue_credit_internal = fake_issue

    pending_a = PendingBranchJoin(
        parent_x_correlation_id="corr-A",
        parent_conversation_id="conv-A",
        parent_num_turns=3,
        parent_agent_depth=0,
        parent_parent_correlation_id=None,
        gated_turn_index=2,
    )
    pending_b = PendingBranchJoin(
        parent_x_correlation_id="corr-B",
        parent_conversation_id="conv-B",
        parent_num_turns=5,
        parent_agent_depth=2,
        parent_parent_correlation_id="corr-B-grandparent",
        gated_turn_index=3,
    )

    assert await issuer.dispatch_join_turn(pending_a) is True
    assert await issuer.dispatch_join_turn(pending_b) is True

    assert len(captured) == 2
    turn_a, turn_b = captured
    assert turn_a.x_correlation_id == "corr-A"
    assert turn_a.conversation_id == "conv-A"
    assert turn_a.turn_index == 2
    assert turn_a.num_turns == 3
    assert turn_a.agent_depth == 0
    assert turn_a.parent_correlation_id is None

    assert turn_b.x_correlation_id == "corr-B"
    assert turn_b.conversation_id == "conv-B"
    assert turn_b.turn_index == 3
    assert turn_b.num_turns == 5
    assert turn_b.agent_depth == 2
    assert turn_b.parent_correlation_id == "corr-B-grandparent"


@pytest.mark.asyncio
async def test_dispatch_join_turn_graceful_when_issuer_stopped():
    """When stop_checker rejects, issue_credit returns False and dispatch_join_turn propagates False cleanly."""
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


def test_dispatch_join_turn_does_not_own_joins_suppressed_counter():
    """dispatch_join_turn code never touches the joins_suppressed counter owned by BranchOrchestrator."""

    def _strip_docstring_and_comments(src: str) -> str:
        tree = ast.parse(textwrap.dedent(src))
        fn = tree.body[0]
        # Drop the leading docstring Expr node if present.
        if (
            isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
            and fn.body
            and isinstance(fn.body[0], ast.Expr)
            and isinstance(fn.body[0].value, ast.Constant)
            and isinstance(fn.body[0].value.value, str)
        ):
            fn.body = fn.body[1:]
        return ast.unparse(fn)

    src = _strip_docstring_and_comments(
        inspect.getsource(CreditIssuer.dispatch_join_turn)
    )
    assert "joins_suppressed" not in src, (
        "dispatch_join_turn code must not touch joins_suppressed; "
        "that counter is owned by BranchOrchestrator."
    )
    # Sanity: no executable code in the issuer module touches the counter.
    module_tree = ast.parse(inspect.getsource(issuer_module))
    module_code = ast.unparse(module_tree)
    # Remove all docstrings at module/class/function level.
    for node in ast.walk(module_tree):
        if (
            isinstance(
                node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
    module_code = ast.unparse(module_tree)
    assert "joins_suppressed" not in module_code


@pytest.mark.asyncio
async def test_dispatch_join_turn_does_not_acquire_session_slot():
    """With turn_index > 0, the session-slot path in issue_credit is skipped."""
    issuer = _make_issuer()
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        parent_agent_depth=0,
        gated_turn_index=2,
    )
    result = await issuer.dispatch_join_turn(pending)
    assert result is True
    issuer._concurrency_manager.acquire_session_slot.assert_not_called()
    issuer._concurrency_manager.acquire_prefill_slot.assert_called_once()

    # Structural confirmation: the turn built by dispatch_join_turn uses
    # gated_turn_index directly, so turn_index > 0 drives is_session_start=False.
    sent: TurnToSend = issuer._issue_credit_internal.call_args.args[0]
    assert sent.turn_index == 2  # => is_session_start is False in issue_credit


@pytest.mark.asyncio
async def test_dispatch_join_turn_has_forks_false():
    """The constructed TurnToSend always carries has_forks=False."""
    issuer = _make_issuer()
    captured: dict[str, TurnToSend] = {}

    async def fake_issue(turn: TurnToSend):
        captured["turn"] = turn
        return True

    issuer._issue_credit_internal = fake_issue
    pending = PendingBranchJoin(
        parent_x_correlation_id="corr-parent",
        parent_conversation_id="conv-parent",
        parent_num_turns=3,
        gated_turn_index=2,
    )
    await issuer.dispatch_join_turn(pending)
    assert captured["turn"].has_forks is False


@pytest.mark.asyncio
async def test_dispatch_join_turn_uses_blocking_issuance_not_try():
    """Joins issue via the blocking prefill-slot path, never non-blocking
    try_issue_credit. An issued join returns True (resumed) -- it must NOT leak
    issue_credit's overloaded False (which also fires for an issued-but-final
    credit and would be miscounted as suppressed)."""
    issuer = _make_issuer()
    issuer.try_issue_credit = AsyncMock(return_value=None)
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
    issuer.try_issue_credit.assert_not_called()
