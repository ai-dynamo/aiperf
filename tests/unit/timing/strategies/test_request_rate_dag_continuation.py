# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for RequestRateStrategy DAG child-continuation routing.

Covers ``_issue_child_continuation_or_release`` and the ``branch_orchestrator``
constructor parameter added in P2T18: when a DAG child credit returns and the
cap is reached, the strategy must route the gated continuation through
``BranchOrchestrator.on_child_stopped`` instead of leaving it stranded.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models.dataset_models import TurnMetadata
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.plugin.enums import ArrivalPattern, TimingMode
from aiperf.timing.config import CreditPhaseConfig
from aiperf.timing.strategies.request_rate import RequestRateStrategy


def _make_strategy(
    *,
    branch_orchestrator: MagicMock | None = None,
    dispatch_result: bool = True,
) -> tuple[RequestRateStrategy, MagicMock]:
    """Build a minimally-mocked RequestRateStrategy for unit testing.

    Returns (strategy, credit_issuer) so tests can assert on dispatch calls.
    """
    config = CreditPhaseConfig(
        phase=CreditPhase.PROFILING,
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=10.0,
        arrival_pattern=ArrivalPattern.CONSTANT,
        total_expected_requests=100,
    )

    conversation_source = MagicMock()
    # default: no delay, so handle_credit_return goes the immediate path
    conversation_source.get_next_turn_metadata.return_value = TurnMetadata(
        delay_ms=None, has_forks=False
    )

    scheduler = MagicMock()
    stop_checker = MagicMock()
    credit_issuer = MagicMock()
    credit_issuer.dispatch_child_turn = AsyncMock(return_value=dispatch_result)
    lifecycle = MagicMock()
    lifecycle.started_at_perf_ns = 0

    strategy = RequestRateStrategy(
        config=config,
        conversation_source=conversation_source,
        scheduler=scheduler,
        stop_checker=stop_checker,
        credit_issuer=credit_issuer,
        lifecycle=lifecycle,
        branch_orchestrator=branch_orchestrator,
    )
    return strategy, credit_issuer


def _child_credit(*, is_final: bool = False) -> Credit:
    """Construct a DAG-child Credit (agent_depth>0) with a non-final turn."""
    return Credit(
        id=42,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-child",
        x_correlation_id="child-xcid",
        turn_index=0,
        # is_final_turn := turn_index == num_turns - 1; pick num_turns=2 for non-final
        num_turns=1 if is_final else 2,
        issued_at_ns=0,
        agent_depth=1,
        parent_correlation_id="parent-xcid",
        branch_mode=ConversationBranchMode.FORK,
    )


# =============================================================================
# Constructor / API surface
# =============================================================================


def test_constructor_accepts_branch_orchestrator_kwarg() -> None:
    """The branch_orchestrator parameter must be optional and exposed."""
    orch = MagicMock()
    strategy, _ = _make_strategy(branch_orchestrator=orch)
    assert strategy._branch_orchestrator is orch


def test_constructor_defaults_branch_orchestrator_to_none() -> None:
    """Non-DAG runs construct without a branch_orchestrator."""
    strategy, _ = _make_strategy(branch_orchestrator=None)
    assert strategy._branch_orchestrator is None


def test_issue_child_continuation_or_release_method_exists() -> None:
    """Sanity: P2T18 adds this exact method name."""
    strategy, _ = _make_strategy()
    assert callable(strategy._issue_child_continuation_or_release)


# =============================================================================
# Below-cap dispatch (dispatch_child_turn returns True)
# =============================================================================


@pytest.mark.asyncio
async def test_below_cap_dispatches_normally_no_orchestrator_call() -> None:
    """When dispatch_child_turn succeeds, on_child_stopped is NOT called."""
    orch = MagicMock()
    orch.on_child_stopped = AsyncMock()
    strategy, credit_issuer = _make_strategy(
        branch_orchestrator=orch, dispatch_result=True
    )
    credit = _child_credit()
    turn = TurnToSend.from_previous_credit(credit)

    await strategy._issue_child_continuation_or_release(turn, credit)

    credit_issuer.dispatch_child_turn.assert_awaited_once_with(turn)
    orch.on_child_stopped.assert_not_called()


@pytest.mark.asyncio
async def test_handle_credit_return_child_below_cap_dispatches_directly() -> None:
    """Child credit return below cap: dispatched directly, NOT queued, no orch."""
    orch = MagicMock()
    orch.on_child_stopped = AsyncMock()
    strategy, credit_issuer = _make_strategy(
        branch_orchestrator=orch, dispatch_result=True
    )
    credit = _child_credit()

    await strategy.handle_credit_return(credit)

    credit_issuer.dispatch_child_turn.assert_awaited_once()
    orch.on_child_stopped.assert_not_called()
    assert strategy._continuation_turns.empty(), (
        "DAG child must not be enqueued onto the rate-loop continuation queue"
    )


# =============================================================================
# At-cap routing (dispatch_child_turn returns False)
# =============================================================================


@pytest.mark.asyncio
async def test_at_cap_routes_to_on_child_stopped() -> None:
    """When dispatch_child_turn refuses, the child x_correlation_id is released."""
    orch = MagicMock()
    orch.on_child_stopped = AsyncMock()
    strategy, credit_issuer = _make_strategy(
        branch_orchestrator=orch, dispatch_result=False
    )
    credit = _child_credit()
    turn = TurnToSend.from_previous_credit(credit)

    await strategy._issue_child_continuation_or_release(turn, credit)

    credit_issuer.dispatch_child_turn.assert_awaited_once_with(turn)
    orch.on_child_stopped.assert_awaited_once_with("child-xcid")


@pytest.mark.asyncio
async def test_at_cap_without_orchestrator_swallows_silently() -> None:
    """No orchestrator wired (non-DAG path): refusal must not raise."""
    strategy, credit_issuer = _make_strategy(
        branch_orchestrator=None, dispatch_result=False
    )
    credit = _child_credit()
    turn = TurnToSend.from_previous_credit(credit)

    # Should complete without exception even though dispatch returned False
    await strategy._issue_child_continuation_or_release(turn, credit)
    credit_issuer.dispatch_child_turn.assert_awaited_once_with(turn)


@pytest.mark.asyncio
async def test_on_child_stopped_exception_logged_not_raised() -> None:
    """If on_child_stopped raises, the strategy logs but does not propagate."""
    orch = MagicMock()
    orch.on_child_stopped = AsyncMock(side_effect=RuntimeError("boom"))
    strategy, _ = _make_strategy(branch_orchestrator=orch, dispatch_result=False)
    credit = _child_credit()
    turn = TurnToSend.from_previous_credit(credit)

    # Must not propagate
    await strategy._issue_child_continuation_or_release(turn, credit)
    orch.on_child_stopped.assert_awaited_once()


# =============================================================================
# Non-child credits (agent_depth == 0) keep the legacy queue path
# =============================================================================


@pytest.mark.asyncio
async def test_root_credit_return_uses_continuation_queue() -> None:
    """Root (agent_depth=0) returns must NOT go through dispatch_child_turn."""
    strategy, credit_issuer = _make_strategy(branch_orchestrator=MagicMock())
    root = Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conv-root",
        x_correlation_id="root-xcid",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
        agent_depth=0,
    )

    await strategy.handle_credit_return(root)

    credit_issuer.dispatch_child_turn.assert_not_called()
    assert not strategy._continuation_turns.empty()
