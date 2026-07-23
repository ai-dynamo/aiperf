# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``AgenticReplayStrategy._dispatch_next_turn`` child routing.

A DAG child continuation (``agent_depth > 0``) must go through the single
child-issuance chokepoint (``_issue_child_continuation_or_drain``), which calls
``dispatch_child_turn`` (clean True-iff-on-wire) and, on refusal (e.g. the
``--request-count`` wire cap), notifies ``BranchOrchestrator.on_child_stopped``
so the parent's join drains deterministically.

The regression this guards: routing children through the discarded
``issue_credit`` bool silently swallows a gate refusal, leaving the parent join
blocked forever on a child whose remaining turns will never be issued
(deadlock at the ``--request-count`` cutoff for DAG runs). The chokepoint must
be used on BOTH the immediate and the delayed (``delay_ms``) paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models.dataset_models import TurnMetadata
from aiperf.credit.structs import Credit
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy


def _make_strategy(
    *,
    branch_orchestrator: MagicMock | None = None,
    dispatch_result: bool = True,
    delay_ms: float | None = None,
) -> tuple[AgenticReplayStrategy, MagicMock, MagicMock]:
    """Build a strategy with only the attributes ``_dispatch_next_turn`` reads.

    Bypasses ``__init__`` (which requires a real ``TrajectorySource`` and
    ``CreditPhaseConfig``) because the routing logic under test touches only
    ``conversation_source``, ``credit_issuer``, ``scheduler``, and
    ``branch_orchestrator``.

    Returns ``(strategy, credit_issuer, scheduler)``.
    """
    strategy = AgenticReplayStrategy.__new__(AgenticReplayStrategy)

    conversation_source = MagicMock()
    conversation_source.get_next_turn_metadata.return_value = TurnMetadata(
        delay_ms=delay_ms, has_forks=False
    )

    credit_issuer = MagicMock()
    credit_issuer.dispatch_child_turn = AsyncMock(return_value=dispatch_result)
    credit_issuer.issue_credit = AsyncMock(return_value=True)

    scheduler = MagicMock()

    strategy.conversation_source = conversation_source
    strategy.credit_issuer = credit_issuer
    strategy.scheduler = scheduler
    strategy.branch_orchestrator = branch_orchestrator
    return strategy, credit_issuer, scheduler


def _child_credit() -> Credit:
    """A non-final DAG-child Credit (agent_depth > 0)."""
    return Credit(
        id=42,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-child",
        x_correlation_id="child-xcid",
        turn_index=0,
        num_turns=2,  # non-final: turn_index 0 of 2
        issued_at_ns=0,
        agent_depth=1,
        parent_correlation_id="parent-xcid",
        branch_mode=ConversationBranchMode.FORK,
    )


def _root_credit() -> Credit:
    """A non-final root Credit (agent_depth == 0)."""
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-root",
        x_correlation_id="root-xcid",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
        agent_depth=0,
    )


# Child continuation, immediate path (no delay_ms)


@pytest.mark.asyncio
async def test_child_below_cap_dispatches_via_chokepoint_no_orch_call() -> None:
    """Child on-wire: dispatch_child_turn used, issue_credit NOT, no drain."""
    orch = MagicMock()
    orch.on_child_stopped = AsyncMock()
    strategy, issuer, _ = _make_strategy(branch_orchestrator=orch, dispatch_result=True)

    await strategy._dispatch_next_turn(_child_credit())

    issuer.dispatch_child_turn.assert_awaited_once()
    issuer.issue_credit.assert_not_called()
    orch.on_child_stopped.assert_not_called()


@pytest.mark.asyncio
async def test_child_at_cap_routes_to_on_child_stopped() -> None:
    """Child refused at the gate -> parent join drained via on_child_stopped."""
    orch = MagicMock()
    orch.on_child_stopped = AsyncMock()
    strategy, issuer, _ = _make_strategy(
        branch_orchestrator=orch, dispatch_result=False
    )

    await strategy._dispatch_next_turn(_child_credit())

    issuer.dispatch_child_turn.assert_awaited_once()
    issuer.issue_credit.assert_not_called()
    orch.on_child_stopped.assert_awaited_once_with("child-xcid")


@pytest.mark.asyncio
async def test_child_at_cap_without_orchestrator_swallows_silently() -> None:
    """No orchestrator wired: refusal must not raise."""
    strategy, issuer, _ = _make_strategy(
        branch_orchestrator=None, dispatch_result=False
    )

    await strategy._dispatch_next_turn(_child_credit())

    issuer.dispatch_child_turn.assert_awaited_once()
    issuer.issue_credit.assert_not_called()


# Child continuation, delayed path (delay_ms > 0)


@pytest.mark.asyncio
async def test_child_delayed_schedules_chokepoint_coro_not_issue_credit() -> None:
    """The delayed (delay_ms) path must also route children through the chokepoint.

    The refusal would otherwise fire long after the callback handler decided
    the child could proceed, so the delayed branch must schedule the
    drain-aware coroutine -- never ``issue_credit``.
    """
    orch = MagicMock()
    orch.on_child_stopped = AsyncMock()
    strategy, issuer, scheduler = _make_strategy(
        branch_orchestrator=orch, dispatch_result=False, delay_ms=250.0
    )

    await strategy._dispatch_next_turn(_child_credit())

    # Immediate child dispatch did NOT happen; it was deferred to the scheduler.
    scheduler.schedule_later.assert_called_once()
    delay_s, coro = scheduler.schedule_later.call_args.args
    assert delay_s == pytest.approx(0.25)
    issuer.issue_credit.assert_not_called()

    # Driving the scheduled coroutine routes the refusal to on_child_stopped.
    await coro
    issuer.dispatch_child_turn.assert_awaited_once()
    orch.on_child_stopped.assert_awaited_once_with("child-xcid")


# Root continuation keeps issue_credit


@pytest.mark.asyncio
async def test_root_continuation_uses_issue_credit_not_chokepoint() -> None:
    """Root (agent_depth == 0) continuation must keep issue_credit."""
    orch = MagicMock()
    orch.on_child_stopped = AsyncMock()
    strategy, issuer, _ = _make_strategy(branch_orchestrator=orch)

    await strategy._dispatch_next_turn(_root_credit())

    issuer.issue_credit.assert_awaited_once()
    issuer.dispatch_child_turn.assert_not_called()
    orch.on_child_stopped.assert_not_called()


@pytest.mark.asyncio
async def test_root_continuation_delayed_schedules_issue_credit() -> None:
    """Root delayed continuation schedules issue_credit, not the chokepoint."""
    strategy, issuer, scheduler = _make_strategy(delay_ms=100.0)

    await strategy._dispatch_next_turn(_root_credit())

    scheduler.schedule_later.assert_called_once()
    _, coro = scheduler.schedule_later.call_args.args
    issuer.dispatch_child_turn.assert_not_called()
    await coro
    issuer.issue_credit.assert_awaited_once()
