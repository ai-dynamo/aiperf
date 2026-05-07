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
    strategy.handle_credit_return.assert_awaited_once_with(credit, error=None)


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_no_orchestrator_bypasses_intercept():
    handler, strategy = _make_handler_with_phase(None)
    credit = _make_credit()
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(credit=credit, first_token_sent=True),
    )
    strategy.handle_credit_return.assert_awaited_once_with(credit, error=None)


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
    strategy.handle_credit_return.assert_awaited_once_with(child_credit, error=None)


# ============================================================================
# Drain-observer wiring tests
# ============================================================================
#
# Regression for the concurrency>=2 race where the orchestrator's last drain
# step (`_handle_child_done` decrement, `dispatch_join_turn` returning False
# under cap, all-children-rolled-back path) lands BETWEEN concurrent
# `on_credit_return` callbacks. Without the drain-observer hook,
# `all_credits_returned_event` is never set from the callback path; the
# phase runner relies on its pre-wait short-circuit (eager) or drain-timeout
# backstop (slow). This suite verifies the source-side fix in
# CreditCallbackHandler.set_branch_orchestrator wires the observer correctly
# and the closure honors the AND-of-predicates contract.


@pytest.mark.component_integration
def test_set_branch_orchestrator_registers_drain_observer() -> None:
    """Attaching an orchestrator must register the handler's drain
    callback. Detaching (set None) must clear it."""
    orchestrator = MagicMock()
    orchestrator.set_drain_observer = MagicMock()
    handler, _strategy = _make_handler_with_phase(None)

    handler.set_branch_orchestrator(orchestrator)
    orchestrator.set_drain_observer.assert_called_once()
    callback = orchestrator.set_drain_observer.call_args.args[0]
    assert callable(callback)

    handler.set_branch_orchestrator(None)
    # The previously-attached orchestrator gets a None observer to detach.
    orchestrator.set_drain_observer.assert_called_with(None)


@pytest.mark.component_integration
def test_drain_observer_sets_event_when_predicate_satisfied() -> None:
    """When the orchestrator fires its drain observer AND
    check_all_returned_or_cancelled() AND has_pending_branch_work()=False,
    the deferred all_credits_returned_event MUST fire. This is the
    race-closing path: the last drain step lands after every callback's
    deferred check has already run with `pending=True`, so without this
    hook the event is never set from the callback path."""
    orchestrator = MagicMock()
    orchestrator.set_drain_observer = MagicMock()
    orchestrator.has_pending_branch_work = MagicMock(return_value=False)
    handler, _strategy = _make_handler_with_phase(None)

    # Set the phase counters to "all returned" before attaching.
    progress = handler._phase_handlers[CreditPhase.PROFILING].progress
    progress.check_all_returned_or_cancelled = MagicMock(return_value=True)
    assert not progress.all_credits_returned_event.is_set()

    handler.set_branch_orchestrator(orchestrator)
    callback = orchestrator.set_drain_observer.call_args.args[0]
    callback()

    assert progress.all_credits_returned_event.is_set(), (
        "drain observer must set all_credits_returned_event when both "
        "counter check and orchestrator predicate are satisfied"
    )


@pytest.mark.component_integration
def test_drain_observer_no_op_when_pending_work_remains() -> None:
    """When has_pending_branch_work() is True the drain callback must NOT
    fire the event — there is still DAG work in flight; firing now would
    cause the phase to declare itself complete with children still
    running."""
    orchestrator = MagicMock()
    orchestrator.set_drain_observer = MagicMock()
    orchestrator.has_pending_branch_work = MagicMock(return_value=True)
    handler, _strategy = _make_handler_with_phase(None)
    progress = handler._phase_handlers[CreditPhase.PROFILING].progress
    progress.check_all_returned_or_cancelled = MagicMock(return_value=True)

    handler.set_branch_orchestrator(orchestrator)
    callback = orchestrator.set_drain_observer.call_args.args[0]
    callback()

    assert not progress.all_credits_returned_event.is_set(), (
        "drain observer must defer when orchestrator still has pending work"
    )


@pytest.mark.component_integration
def test_drain_observer_no_op_when_counters_disagree() -> None:
    """When check_all_returned_or_cancelled() is False the callback must
    not fire the event — sending isn't actually complete yet."""
    orchestrator = MagicMock()
    orchestrator.set_drain_observer = MagicMock()
    orchestrator.has_pending_branch_work = MagicMock(return_value=False)
    handler, _strategy = _make_handler_with_phase(None)
    progress = handler._phase_handlers[CreditPhase.PROFILING].progress
    progress.check_all_returned_or_cancelled = MagicMock(return_value=False)

    handler.set_branch_orchestrator(orchestrator)
    callback = orchestrator.set_drain_observer.call_args.args[0]
    callback()

    assert not progress.all_credits_returned_event.is_set(), (
        "drain observer must defer when counters say sending isn't complete"
    )


@pytest.mark.component_integration
def test_drain_observer_skips_completed_phase_handlers() -> None:
    """If a phase's lifecycle is already complete, the drain callback
    must skip it — that handler's event was already finalized through
    the normal phase-end path, and re-setting from here would be racy."""
    orchestrator = MagicMock()
    orchestrator.set_drain_observer = MagicMock()
    orchestrator.has_pending_branch_work = MagicMock(return_value=False)
    handler, _strategy = _make_handler_with_phase(None)
    ctx = handler._phase_handlers[CreditPhase.PROFILING]
    ctx.lifecycle.is_complete = True
    ctx.progress.check_all_returned_or_cancelled = MagicMock(return_value=True)

    handler.set_branch_orchestrator(orchestrator)
    callback = orchestrator.set_drain_observer.call_args.args[0]
    callback()

    assert not ctx.progress.all_credits_returned_event.is_set(), (
        "drain observer must skip phase handlers whose lifecycle is "
        "already complete (their event has already been handled by the "
        "normal phase-end path)"
    )


@pytest.mark.component_integration
def test_drain_observer_idempotent_on_already_set_event() -> None:
    """If the event is already set, calling the drain callback again
    must be a benign no-op. (The observer can fire multiple times in
    rapid succession — _handle_child_done plus dispatch_join_turn plus
    rollback paths all call _notify_drain.)"""
    orchestrator = MagicMock()
    orchestrator.set_drain_observer = MagicMock()
    orchestrator.has_pending_branch_work = MagicMock(return_value=False)
    handler, _strategy = _make_handler_with_phase(None)
    progress = handler._phase_handlers[CreditPhase.PROFILING].progress
    progress.check_all_returned_or_cancelled = MagicMock(return_value=True)
    progress.all_credits_returned_event.set()

    handler.set_branch_orchestrator(orchestrator)
    callback = orchestrator.set_drain_observer.call_args.args[0]
    callback()
    callback()
    callback()

    assert progress.all_credits_returned_event.is_set()


# ============================================================================
# Warmup spawn-skip tests
# ============================================================================
#
# Regression for the warmup-hang where AgenticReplayStrategy.handle_credit_return
# short-circuits warmup (warmup is one-shot per trajectory), so spawned children
# never advance past their first turn. Without is_final_turn returns,
# on_child_leaf_reached never fires, _descendant_counts leaks > 0, and
# has_pending_branch_work() stays True forever — wedging
# all_credits_returned_event and hanging PhaseRunner indefinitely.
#
# Fix: BranchOrchestrator.intercept must short-circuit when credit.phase is
# WARMUP, before any branch-spawn machinery runs. DAG dispatch is correctly
# active in PROFILING.


def _make_orchestrator_with_branches(
    branch_ids: list[str],
) -> tuple[object, MagicMock, AsyncMock]:
    """Build a BranchOrchestrator whose conversation source declares the given
    branch_ids on turn 0. Returns (orch, conversation_source, dispatch_first_turn)
    so callers can assert on spawn calls."""
    from aiperf.common.enums import ConversationBranchMode
    from aiperf.timing.branch_orchestrator import BranchOrchestrator

    cs = MagicMock()
    parent_meta = MagicMock()
    parent_meta.branches = [
        MagicMock(
            branch_id=bid,
            child_conversation_ids=[f"{bid}-child"],
            is_background=False,
            mode=ConversationBranchMode.FORK,
        )
        for bid in branch_ids
    ]
    parent_meta.turns = [MagicMock(branch_ids=branch_ids)]
    cs.get_metadata = MagicMock(return_value=parent_meta)
    cs.start_branch_child = MagicMock(
        side_effect=lambda **kwargs: MagicMock(
            x_correlation_id=f"child-{kwargs['child_conversation_id']}"
        )
    )

    issuer = MagicMock()
    dispatch = AsyncMock(return_value=True)
    issuer.dispatch_first_turn = dispatch

    orch = BranchOrchestrator(conversation_source=cs, credit_issuer=issuer)
    return orch, cs, dispatch


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_intercept_skips_spawn_during_warmup() -> None:
    """A WARMUP-phase credit return with declared branches MUST NOT spawn
    children. AgenticReplayStrategy refuses to advance child continuation
    turns during warmup, so spawned children would never reach
    is_final_turn — _descendant_counts would leak > 0 forever and
    has_pending_branch_work() would wedge all_credits_returned_event.
    Reproduced 100% on H100 + b200-nb at conc=16 with the
    inferencex-agentx-mvp scenario before the fix."""
    orch, cs, dispatch_first_turn = _make_orchestrator_with_branches(["root:0"])
    warmup_credit = Credit(
        id=1,
        phase=CreditPhase.WARMUP,
        conversation_id="conv1",
        x_correlation_id="root",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        parent_correlation_id=None,
        agent_depth=0,
    )

    result = await orch.intercept(warmup_credit)

    assert result is False, "warmup intercept must not gate the parent"
    cs.start_branch_child.assert_not_called()
    dispatch_first_turn.assert_not_awaited()
    assert orch.stats.children_spawned == 0


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_intercept_spawns_during_profiling() -> None:
    """Symmetric positive case: PROFILING-phase credits with declared
    branches MUST still spawn children. The warmup short-circuit must
    not regress the normal DAG dispatch path."""
    orch, cs, dispatch_first_turn = _make_orchestrator_with_branches(["root:0"])
    credit = _make_credit(turn_index=0)
    assert credit.phase == CreditPhase.PROFILING

    result = await orch.intercept(credit)

    assert result is False, "pure spawn with no gate returns False"
    assert cs.start_branch_child.call_count == 1
    assert dispatch_first_turn.await_count == 1
    assert orch.stats.children_spawned == 1


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_intercept_warmup_skip_runs_before_agent_depth_guard() -> None:
    """The warmup short-circuit must run before the agent_depth guard so
    that even a hypothetical depth-0 warmup credit with branches declared
    is rejected. Verifies guard ordering: cleaning_up -> warmup -> child."""
    orch, _cs, dispatch_first_turn = _make_orchestrator_with_branches(["root:0"])
    warmup_credit = Credit(
        id=1,
        phase=CreditPhase.WARMUP,
        conversation_id="conv1",
        x_correlation_id="root",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        parent_correlation_id=None,
        agent_depth=0,
    )

    assert await orch.intercept(warmup_credit) is False
    dispatch_first_turn.assert_not_awaited()


@pytest.mark.component_integration
@pytest.mark.asyncio
async def test_intercept_warmup_skip_does_not_leak_descendant_counts() -> None:
    """Direct assertion of the wedge-mechanism the fix prevents: after a
    warmup credit return, _descendant_counts MUST remain empty and
    has_pending_branch_work() MUST be False. Pre-fix this would leak: the
    parent would be registered with N descendants, no child would ever
    leaf-reach (strategy refuses warmup continuation), and the predicate
    would stay True forever."""
    orch, _cs, _dispatch = _make_orchestrator_with_branches(["root:0", "root:1"])
    warmup_credit = Credit(
        id=1,
        phase=CreditPhase.WARMUP,
        conversation_id="conv1",
        x_correlation_id="root",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        parent_correlation_id=None,
        agent_depth=0,
    )

    await orch.intercept(warmup_credit)

    assert orch._descendant_counts == {}, (
        "warmup must not leak descendant tracking — children would never "
        "leaf-reach and has_pending_branch_work would wedge forever"
    )
    assert orch.has_pending_branch_work() is False
