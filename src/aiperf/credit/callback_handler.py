# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Credit callback handler for credit lifecycle events.

Handles ALL credit lifecycle callbacks (returns + TTFT) directly from CreditRouter.

Key responsibilities:
- Track credit returns (increment_returned, release slots)
- Handle TTFT events (increment_prefill_released, release prefill slot)
- Dispatch next turn to timing strategy (handle_credit_return)
- Cleanup in-flight sessions on phase end
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase

if TYPE_CHECKING:
    from aiperf.credit.messages import CreditReturn, FirstToken
    from aiperf.credit.structs import Credit
    from aiperf.timing.branch_orchestrator import BranchOrchestrator
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.strategies.core import TimingStrategyProtocol

_logger = AIPerfLogger(__name__)


@dataclass(slots=True)
class PhaseCallbackContext:
    """Context for handling callbacks for a specific phase.

    Registered by PhaseRunner before phase execution starts.
    Contains all components needed to handle credit returns for this phase.
    """

    progress: PhaseProgressTracker
    lifecycle: PhaseLifecycle
    stop_checker: StopConditionChecker
    strategy: TimingStrategyProtocol
    concurrency_manager: ConcurrencyManager


# =============================================================================
# CreditCallbackHandler - Handle credit lifecycle callbacks
# =============================================================================


class CreditCallbackHandler:
    """Handles credit lifecycle callbacks from CreditRouter.

    Unified callback handler for all phases.

    Callback flow:
        Worker → CreditRouter → CreditCallbackHandler → [count, release slots, dispatch]

    Processing order for credit returns:
        1. Atomic counting (increment_returned)
        2. Track prefill release if TTFT never arrived
        3. Release concurrency slots
        4. Dispatch next turn via timing strategy (if applicable)

    Processing order for TTFT:
        1. Track prefill release (increment_prefill_released)
        2. Release prefill slot

    Phase Registration:
        PhaseRunner calls register_phase() BEFORE any credits are sent.
        This ensures callbacks work from the first credit.
    """

    def __init__(self, concurrency_manager: ConcurrencyManager) -> None:
        """Initialize callback handler.

        Args:
            concurrency_manager: Manages concurrency slots (shared across phases).
        """
        self._concurrency_manager = concurrency_manager
        self._phase_handlers: dict[CreditPhase, PhaseCallbackContext] = {}
        self._branch_orchestrator: BranchOrchestrator | None = None

    def set_branch_orchestrator(self, orchestrator: BranchOrchestrator | None) -> None:
        """Inject (or detach) the DAG branch orchestrator.

        Called by ``PhaseRunner`` before phase start when the dataset is
        DAG-shaped, and again with ``None`` after the phase finalizes so a
        subsequent non-DAG phase / cleanup doesn't dispatch into a torn-down
        orchestrator.

        Also registers a drain observer on the orchestrator so the deferred
        completion check fires when the orchestrator's last drain step
        lands AFTER the final ``on_credit_return`` callback (concurrency
        race: under N>1, ``has_pending_branch_work()`` can flip False
        between credit returns, with no further return arriving to
        re-trigger the check).
        """
        # Detach observer from any previously attached orchestrator.
        if (
            self._branch_orchestrator is not None
            and self._branch_orchestrator is not orchestrator
        ):
            self._branch_orchestrator.set_drain_observer(None)
            self._branch_orchestrator.set_abort_observer(None)
        self._branch_orchestrator = orchestrator
        if orchestrator is not None:
            orchestrator.set_drain_observer(self._on_orchestrator_drain)
            orchestrator.set_abort_observer(self._on_orchestrator_abort)

    def _on_orchestrator_drain(self) -> None:
        """Re-evaluate completion across every active phase handler.

        Fired by ``BranchOrchestrator`` after each state mutation that
        could drain ``has_pending_branch_work()`` to False. Idempotent: if
        the event is already set or the predicate disagrees, the per-handler
        check no-ops.
        """
        for handler in self._phase_handlers.values():
            if handler.lifecycle.is_complete:
                continue
            self._maybe_signal_dag_completion(handler)

    def _on_orchestrator_abort(self) -> None:
        """Cancel every active phase on FAIL_FAST.

        Fired by ``BranchOrchestrator._handle_child_errored_fail_fast``
        after parent + orphan-sibling tear-down. Cancels each phase's
        lifecycle so the strategy loop's next ``can_send_any_turn`` check
        returns False and no further wire credits are issued. In-flight
        credits drain naturally; the phase completes once they return.
        Without this hook, only the parent of the errored child was
        aborted while unrelated roots kept firing — the wire-request
        budget ran out as if FAIL_FAST were disabled.
        """
        for handler in self._phase_handlers.values():
            if handler.lifecycle.is_complete:
                continue
            handler.lifecycle.cancel()
            handler.progress.all_credits_returned_event.set()

    def register_phase(
        self,
        *,
        phase: CreditPhase,
        progress: PhaseProgressTracker,
        lifecycle: PhaseLifecycle,
        stop_checker: StopConditionChecker,
        strategy: TimingStrategyProtocol,
    ) -> None:
        """Register phase for callback handling.

        Called by PhaseRunner BEFORE phase execution starts.
        Must be called before any credits are sent for this phase.

        Args:
            phase: Phase enum (WARMUP or PROFILING).
            progress: Progress tracker for counting.
            lifecycle: Phase lifecycle for state checks.
            stop_checker: Evaluates stop conditions.
            strategy: Timing strategy for dispatching next turns.
        """
        self._phase_handlers[phase] = PhaseCallbackContext(
            progress=progress,
            lifecycle=lifecycle,
            stop_checker=stop_checker,
            strategy=strategy,
            concurrency_manager=self._concurrency_manager,
        )
        _logger.debug(lambda: f"Registered callback handler for phase {phase}")

    def unregister_phase(self, phase: CreditPhase) -> None:
        """Unregister phase when done.

        Called by PhaseRunner after phase completes.
        Late arrivals after unregister are logged but ignored.

        Args:
            phase: Phase to unregister.
        """
        if phase in self._phase_handlers:
            del self._phase_handlers[phase]
            _logger.debug(lambda: f"Unregistered callback handler for phase {phase}")

    async def on_credit_return(
        self, worker_id: str, credit_return: CreditReturn
    ) -> None:
        """Handle credit return from worker.

        Processing order:
        1. Atomic counting (increment_returned)
        2. Track prefill release if TTFT never arrived
        3. Release concurrency slots
        4. Dispatch next turn via strategy (if applicable)

        Args:
            worker_id: ID of the worker returning the credit.
            credit_return: Return details including credit and status.
        """
        credit = credit_return.credit
        handler = self._lookup_active_phase_handler(credit, worker_id)
        if handler is None:
            return

        self._count_and_release(credit, credit_return, handler)

        # 4b. DAG child completion hook.
        # When a child session's final turn returns, notify the orchestrator
        # so it can decrement join refcounts, release sticky-routing entries,
        # and dispatch the parent's gated turn (if any). Runs regardless of
        # whether the phase can still send, because children may finish
        # after the parent has already sent its terminal turn.
        await self._notify_orchestrator_of_child_completion(credit, credit_return)

        # 5. DAG intercept — root-credit returns may spawn child sessions.
        # Runs BEFORE strategy dispatch so the orchestrator can take over
        # the next-turn path when its own gated dispatch suppresses the
        # parent's continuation.
        intercepted = await self._intercept_for_dag(credit)

        # 6. Notify timing strategy for subsequent turns. Skipped when the
        # orchestrator intercepted (it owns the next turn now).
        # For child non-final returns we ALWAYS call the strategy so its
        # ``_issue_child_continuation_or_release`` can fire ``on_child_stopped``
        # when the cap blocks dispatch — otherwise the parent's pending join
        # would never drain. Root credits stay gated on ``can_send_any_turn``
        # so origin/main multi-turn semantics (truncate continuation past
        # the cap) are preserved.
        if not intercepted:
            is_child_non_final = credit.agent_depth > 0 and not credit.is_final_turn
            if is_child_non_final or handler.stop_checker.can_send_any_turn():
                await handler.strategy.handle_credit_return(credit)

        # 7. Deferred all-credits-returned check. Runs on every return —
        # the orchestrator can drain the DAG synchronously inside
        # ``intercept`` when every spawned child gets refused at the cap
        # gate (e.g. cap=1: root return triggers intercept, both children
        # refused, ``has_pending_branch_work`` flips False inside the same
        # call). If we skipped this, the event would never fire because
        # no future credit return is coming.
        self._maybe_signal_dag_completion(handler)

    def _lookup_active_phase_handler(
        self, credit: Credit, worker_id: str
    ) -> PhaseCallbackContext | None:
        """Return the active phase handler for this credit, or None.

        Returns None when the phase is unregistered (logged at debug) or
        already complete (logged at warning); callers must short-circuit on
        a None return.
        """
        phase = credit.phase
        handler = self._phase_handlers.get(phase)
        if not handler:
            _logger.debug(
                lambda: f"Credit return for unregistered phase {phase}, "
                f"credit_id={credit.id}, worker={worker_id}"
            )
            return None
        if handler.lifecycle.is_complete:
            _logger.warning(
                lambda: f"Credit return after phase {phase} complete, "
                f"credit_id={credit.id}, worker={worker_id}"
            )
            return None
        return handler

    def _count_and_release(
        self,
        credit: Credit,
        credit_return: CreditReturn,
        handler: PhaseCallbackContext,
    ) -> None:
        """Count this return, release any owed prefill slot, release session
        slot on root final turn, and signal phase-final-return event when
        appropriate. Atomic on the counter side (no awaits)."""
        # 1. ATOMIC COUNTING (no await before this!)
        is_final_returned = handler.progress.increment_returned(
            credit.is_final_turn,
            credit_return.cancelled,
            errored=credit_return.error is not None,
        )

        # 2. Track prefill release if TTFT never arrived
        if not credit_return.first_token_sent:
            handler.progress.increment_prefill_released()

        # 3. Release concurrency slots
        self._release_slots_for_return(
            credit.phase, credit, credit_return, is_final_returned, handler
        )

        # 4. Signal completion if this was the final return — but defer for
        # DAG runs where ``intercept`` is about to spawn children OR there
        # is already pending DAG work in flight. Without this defer, the
        # phase runner unblocks at sending-complete and tears the
        # orchestrator down before its children land. The deferred check
        # at the bottom of ``on_credit_return`` re-evaluates and sets the
        # event once intercept's synchronous work is done.
        if is_final_returned and not self._dag_work_pending(credit):
            handler.progress.all_credits_returned_event.set()

    def _dag_work_pending(self, credit: Credit) -> bool:
        """True iff the orchestrator has work in flight or will spawn on
        this credit return (so the all-credits-returned signal must defer
        until after ``intercept`` runs).

        ``intercept`` runs at every ``agent_depth`` (nested DAGs are
        supported), so the branch-id lookup must run at every depth too.
        Restricting it to root credits previously let nested grandchildren
        be truncated when their parent's return was the final outstanding
        credit at that moment.
        """
        if self._branch_orchestrator is None:
            return False
        if self._branch_orchestrator.has_pending_branch_work():
            return True
        try:
            if self._branch_orchestrator.get_branch_ids(credit):
                return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _maybe_signal_dag_completion(self, handler: PhaseCallbackContext) -> None:
        """Set the all-credits-returned event when the orchestrator drained
        the DAG synchronously inside ``intercept``.
        """
        if self._branch_orchestrator is None:
            return
        if (
            not handler.progress.all_credits_returned_event.is_set()
            and handler.progress.check_all_returned_or_cancelled()
            and not self._branch_orchestrator.has_pending_branch_work()
        ):
            handler.progress.all_credits_returned_event.set()

    async def _notify_orchestrator_of_child_completion(
        self, credit: Credit, credit_return: CreditReturn
    ) -> None:
        """Fire the orchestrator's child-completion hook on a child final return."""
        if (
            not credit.is_final_turn
            or credit.agent_depth == 0
            or self._branch_orchestrator is None
        ):
            return
        try:
            if credit_return.error is not None:
                await self._branch_orchestrator.on_child_errored(
                    credit.x_correlation_id
                )
            else:
                await self._branch_orchestrator.on_child_leaf_reached(
                    credit.x_correlation_id
                )
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                lambda exc=exc: f"BranchOrchestrator child-completion "
                f"hook failed for x_correlation_id="
                f"{credit.x_correlation_id}: {exc}"
            )

    async def _intercept_for_dag(self, credit: Credit) -> bool:
        """Offer a credit return to the orchestrator's intercept hook.

        Returns True iff the orchestrator suppressed the strategy's next-turn
        dispatch (parent suspended on a gated turn). Non-DAG runs return
        False unconditionally.
        """
        if self._branch_orchestrator is None:
            return False
        try:
            return await self._branch_orchestrator.intercept(credit)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                lambda exc=exc: f"BranchOrchestrator intercept failed for "
                f"credit {credit.id}: {exc}"
            )
            return False

    def _release_slots_for_return(
        self,
        phase: CreditPhase,
        credit: Credit,
        credit_return: CreditReturn,
        is_final_returned: bool,
        handler: PhaseCallbackContext,
    ) -> None:
        """Release slots based on credit state.

        Slot release rules:
        - Session slot: Released when conversation ends (final turn)
        - Prefill slot: Released if TTFT never arrived (error/cancellation path)
        - On final return: Cleanup in-flight sessions

        Args:
            phase: Credit phase.
            credit: The returned credit.
            credit_return: Return details.
            is_final_returned: True if this is the last credit of the phase.
            handler: Phase callback context.
        """
        concurrency = handler.concurrency_manager

        # Release session slot when a root conversation ends (final turn,
        # whether completed or cancelled). DAG children (agent_depth > 0)
        # inherit the root's session slot via the dispatch path that
        # bypasses ``acquire_session_slot``; releasing here would underflow.
        if credit.is_final_turn and credit.agent_depth == 0:
            concurrency.release_session_slot(phase)

        # On phase end, release slots for sessions still in flight.
        # These are sessions that started but whose final turn was never sent/returned.
        if is_final_returned:
            in_flight = handler.progress.in_flight_sessions
            if in_flight > 0:
                _logger.debug(
                    lambda: f"Releasing {in_flight} in-flight session slots for phase {phase}"
                )
                for _ in range(in_flight):
                    concurrency.release_session_slot(phase)

        # Prefill slot is normally released on TTFT. If the request failed or was
        # cancelled before first token, we release here to prevent slot leaks.
        if not credit_return.first_token_sent:
            concurrency.release_prefill_slot(phase)

    async def on_first_token(self, first_token: FirstToken) -> None:
        """Handle first token event (TTFT) from worker.

        Releases prefill concurrency slot, allowing another request
        to start prefilling.

        Args:
            first_token: TTFT event details including credit_id and phase.
        """
        phase = first_token.phase
        handler = self._phase_handlers.get(phase)

        if not handler:
            _logger.debug(
                lambda: f"TTFT for unregistered phase {phase}, "
                f"credit_id={first_token.credit_id}"
            )
            return

        # Track the release
        handler.progress.increment_prefill_released()

        # Release the prefill slot
        handler.concurrency_manager.release_prefill_slot(phase)
