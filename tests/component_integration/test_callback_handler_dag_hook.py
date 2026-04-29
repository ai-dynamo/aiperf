# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task 14: CreditCallbackHandler DAG hook tests.

Verifies that ``BranchOrchestrator.intercept`` is offered the credit return
before the timing strategy's ``handle_credit_return`` runs, and that the
strategy is suppressed when intercept returns True.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.messages import CreditReturn
from aiperf.credit.structs import Credit


def _make_credit(
    *,
    turn_index: int = 0,
    num_turns: int = 1,
    parent_correlation_id: str | None = None,
    x_correlation_id: str = "corr-1",
    agent_depth: int = 0,
) -> Credit:
    return Credit(
        id=1,
        phase=CreditPhase.PROFILING,
        conversation_id="conv1",
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=time.time_ns(),
        parent_correlation_id=parent_correlation_id,
        agent_depth=agent_depth,
    )


def _make_child_credit(
    *,
    turn_index: int = 0,
    num_turns: int = 1,
    parent_correlation_id: str = "parent-1",
    x_correlation_id: str = "corr-1",
) -> Credit:
    """Shorthand for a DAG-child credit (agent_depth >= 1).

    Real children are produced by ``ConversationSource.start_branch_child``
    which sets ``agent_depth = parent_depth + 1``. The callback handler's
    child-hook guard is now keyed on ``credit.agent_depth > 0`` to mirror the
    ``is_child`` bypass in ``CreditIssuer``, so tests that simulate child
    returns must set agent_depth explicitly.
    """
    return _make_credit(
        turn_index=turn_index,
        num_turns=num_turns,
        parent_correlation_id=parent_correlation_id,
        x_correlation_id=x_correlation_id,
        agent_depth=1,
    )


def _make_handler_with_phase(
    orchestrator: object | None,
) -> tuple[CreditCallbackHandler, MagicMock]:
    concurrency = MagicMock()
    concurrency.release_session_slot = MagicMock()
    concurrency.release_prefill_slot = MagicMock()

    handler = CreditCallbackHandler(concurrency, branch_orchestrator=orchestrator)

    progress = MagicMock()
    progress.increment_returned = MagicMock(return_value=False)
    progress.increment_prefill_released = MagicMock()
    progress.all_credits_returned_event = asyncio.Event()
    progress.in_flight_sessions = 0

    lifecycle = MagicMock()
    lifecycle.is_complete = False

    stop_checker = MagicMock()
    stop_checker.can_send_any_turn = MagicMock(return_value=True)

    strategy = MagicMock()
    strategy.handle_credit_return = AsyncMock()

    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=progress,
        lifecycle=lifecycle,
        stop_checker=stop_checker,
        strategy=strategy,
    )
    return handler, strategy


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_orchestrator_intercept_short_circuits_strategy():
    orchestrator = MagicMock()
    orchestrator.intercept = AsyncMock(return_value=True)

    handler, strategy = _make_handler_with_phase(orchestrator)
    credit = _make_credit()
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(credit=credit, first_token_sent=True),
    )

    orchestrator.intercept.assert_awaited_once_with(credit)
    strategy.handle_credit_return.assert_not_awaited()


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_strategy_runs_when_orchestrator_intercept_returns_false():
    orchestrator = MagicMock()
    orchestrator.intercept = AsyncMock(return_value=False)

    handler, strategy = _make_handler_with_phase(orchestrator)
    credit = _make_credit()
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(credit=credit, first_token_sent=True),
    )

    orchestrator.intercept.assert_awaited_once_with(credit)
    strategy.handle_credit_return.assert_awaited_once_with(credit)


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_no_orchestrator_bypasses_intercept():
    handler, strategy = _make_handler_with_phase(None)
    credit = _make_credit()
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(credit=credit, first_token_sent=True),
    )
    strategy.handle_credit_return.assert_awaited_once_with(credit)


# =============================================================================
# Child-leaf completion hook tests
# =============================================================================


def _make_child_orchestrator() -> MagicMock:
    orchestrator = MagicMock()
    orchestrator.intercept = AsyncMock(return_value=False)
    orchestrator.on_child_leaf_reached = AsyncMock()
    orchestrator.on_child_errored = AsyncMock()
    return orchestrator


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_on_child_leaf_reached_called_on_child_final_turn():
    """When a child's final-turn credit is returned, the orchestrator's
    on_child_leaf_reached hook fires with the child's x_correlation_id."""
    orchestrator = _make_child_orchestrator()
    handler, _strategy = _make_handler_with_phase(orchestrator)

    child_credit = _make_child_credit(
        turn_index=0,
        num_turns=1,
        parent_correlation_id="parent-1",
        x_correlation_id="child-7",
    )
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(credit=child_credit, first_token_sent=True),
    )

    orchestrator.on_child_leaf_reached.assert_awaited_once_with("child-7")
    orchestrator.on_child_errored.assert_not_awaited()


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_on_child_leaf_reached_not_called_on_non_final_turn():
    """Intermediate turns of a child session must not trigger the
    leaf-reached hook."""
    orchestrator = _make_child_orchestrator()
    handler, _strategy = _make_handler_with_phase(orchestrator)

    mid_credit = _make_child_credit(
        turn_index=0,
        num_turns=3,  # not final
        parent_correlation_id="parent-1",
        x_correlation_id="child-7",
    )
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(credit=mid_credit, first_token_sent=True),
    )

    orchestrator.on_child_leaf_reached.assert_not_awaited()
    orchestrator.on_child_errored.assert_not_awaited()


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_on_child_leaf_reached_not_called_for_root_session():
    """Root sessions (parent_correlation_id is None) must never trigger
    child-completion hooks, even on the final turn."""
    orchestrator = _make_child_orchestrator()
    handler, _strategy = _make_handler_with_phase(orchestrator)

    root_credit = _make_credit(
        turn_index=0,
        num_turns=1,
        parent_correlation_id=None,
        x_correlation_id="root-1",
    )
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(credit=root_credit, first_token_sent=True),
    )

    orchestrator.on_child_leaf_reached.assert_not_awaited()
    orchestrator.on_child_errored.assert_not_awaited()


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_on_child_errored_called_when_credit_return_carries_error():
    """When a child's final-turn credit returns with an error string, the
    orchestrator's on_child_errored hook fires instead of on_child_leaf_reached."""
    orchestrator = _make_child_orchestrator()
    handler, _strategy = _make_handler_with_phase(orchestrator)

    child_credit = _make_child_credit(
        turn_index=0,
        num_turns=1,
        parent_correlation_id="parent-1",
        x_correlation_id="child-7",
    )
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(
            credit=child_credit,
            first_token_sent=False,
            error="connection reset",
        ),
    )

    orchestrator.on_child_errored.assert_awaited_once_with("child-7")
    orchestrator.on_child_leaf_reached.assert_not_awaited()


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_child_hook_does_not_require_can_send_any_turn():
    """Child-completion hook must fire even when the phase is draining
    (can_send_any_turn is False) — children may complete after the parent's
    own terminal turn has already sent.

    Strategy dispatch for the child's continuation is ALSO allowed to proceed
    while draining: DAG child subsequent-turns are bookkeeping outside the
    root-sampler plan that drives ``is_sending_complete``.
    """
    orchestrator = _make_child_orchestrator()
    handler, strategy = _make_handler_with_phase(orchestrator)

    # Flip can_send_any_turn off on the registered phase.
    handler._phase_handlers[
        CreditPhase.PROFILING
    ].stop_checker.can_send_any_turn = MagicMock(return_value=False)

    child_credit = _make_child_credit(
        turn_index=0,
        num_turns=1,
        parent_correlation_id="parent-1",
        x_correlation_id="child-drain",
    )
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(credit=child_credit, first_token_sent=True),
    )

    orchestrator.on_child_leaf_reached.assert_awaited_once_with("child-drain")
    # Strategy dispatch is allowed for DAG child continuations even while the
    # phase is draining. (The strategy itself is a no-op when the credit is
    # final — a separate concern from the callback-handler gating.)
    strategy.handle_credit_return.assert_awaited_once_with(child_credit)
