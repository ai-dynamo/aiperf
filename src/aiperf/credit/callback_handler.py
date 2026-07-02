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

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase

if TYPE_CHECKING:
    from aiperf.credit.messages import CreditReturn, FirstToken
    from aiperf.credit.structs import Credit
    from aiperf.timing.branch_orchestrator import BranchOrchestrator
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.conversation_source import ConversationSource
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
    """Tracks credit send/return counts for this phase."""

    lifecycle: PhaseLifecycle
    """Phase lifecycle state (running, complete, etc.)."""

    stop_checker: StopConditionChecker
    """Evaluates whether the phase should stop sending credits."""

    strategy: TimingStrategyProtocol
    """Timing strategy that dispatches subsequent turns."""

    concurrency_manager: ConcurrencyManager
    """Manages session and prefill concurrency slots."""

    conversation_source: ConversationSource
    """Source for sampling conversations and resolving metadata."""

    handle_credit_result: Callable[[CreditReturn], Awaitable[None]] | None = None
    """Optional per-return callback the timing strategy exposes (adaptive
    scaling consumes each CreditReturn to drive its SLA controller)."""


@dataclass(slots=True)
class ReturnDisposition:
    """Callback outcome for a returned credit."""

    should_continue: bool
    """Whether the strategy should dispatch the next turn."""

    session_ended: bool
    """Whether this credit return ends the conversation session."""

    session_cancelled: bool | None = None
    """Whether the session was cancelled (None if not applicable)."""


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
        # Keyed by phase so a still-draining seamless phase keeps routing its
        # own child returns even after the next phase registers its own
        # orchestrator. A single shared slot would detach the previous phase's
        # drain/abort observers and misroute its in-flight child returns
        # (seamless warmup → profiling overlap).
        self._branch_orchestrators: dict[CreditPhase, BranchOrchestrator] = {}
        # Authoritative record of credits whose FirstToken already released the
        # prefill slot via ``on_first_token``. The worker stamps
        # ``CreditReturn.first_token_sent`` only AFTER awaiting the FirstToken
        # send (worker.py), so a credit-task cancellation at that await can put
        # the FirstToken on the wire while the subsequent CreditReturn still
        # carries ``first_token_sent=False``. Trusting that stale flag would
        # release the prefill slot a second time and over-grant prefill permits.
        # Keyed by (phase, credit_id) to mirror the router's own set.
        self._first_token_received: set[tuple[CreditPhase, int]] = set()

    def set_branch_orchestrator(
        self, phase: CreditPhase, orchestrator: BranchOrchestrator | None
    ) -> None:
        """Inject (or detach) the DAG branch orchestrator for ``phase``.

        Called by ``PhaseRunner`` before phase start when the dataset is
        DAG-shaped, and again with ``None`` after the phase finalizes so a
        subsequent non-DAG phase / cleanup doesn't dispatch into a torn-down
        orchestrator.

        Orchestrators are keyed by phase: a seamless non-final phase defers
        its detach to a background return-wait task, so the next phase's
        registration must NOT clobber the previous phase's orchestrator or
        its in-flight child returns would misroute. Each phase's orchestrator
        keeps its own drain/abort observers attached for the lifetime of that
        phase's handler.

        The drain observer fires the deferred completion check when the
        orchestrator's last drain step lands AFTER the final
        ``on_credit_return`` callback (concurrency race: under N>1,
        ``has_pending_branch_work()`` can flip False between credit returns,
        with no further return arriving to re-trigger the check).
        """
        previous = self._branch_orchestrators.get(phase)
        if previous is not None and previous is not orchestrator:
            previous.set_drain_observer(None)
            previous.set_abort_observer(None)
        if orchestrator is None:
            self._branch_orchestrators.pop(phase, None)
            return
        self._branch_orchestrators[phase] = orchestrator
        orchestrator.set_drain_observer(self._on_orchestrator_drain)
        orchestrator.set_abort_observer(self._on_orchestrator_abort)

    def _on_orchestrator_drain(self) -> None:
        """Re-evaluate completion across every active phase handler.

        Fired by ``BranchOrchestrator`` after each state mutation that
        could drain ``has_pending_branch_work()`` to False. Idempotent: if
        the event is already set or the predicate disagrees, the per-handler
        check no-ops.
        """
        for phase, handler in self._phase_handlers.items():
            if handler.lifecycle.is_complete:
                continue
            self._maybe_signal_dag_completion(phase, handler)

    def _on_orchestrator_abort(self) -> None:
        """Cancel every active phase on FAIL_FAST.

        Fired by ``BranchOrchestrator._handle_child_errored_fail_fast``
        after parent + orphan-sibling tear-down. Cancels each phase's
        lifecycle so the strategy loop's next ``can_send_any_turn`` check
        returns False and no further wire credits are issued. In-flight
        credits drain naturally; the phase completes once they return.
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
        conversation_source: ConversationSource,
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
        handle_credit_result = getattr(strategy, "handle_credit_result", None)
        self._phase_handlers[phase] = PhaseCallbackContext(
            progress=progress,
            lifecycle=lifecycle,
            stop_checker=stop_checker,
            strategy=strategy,
            concurrency_manager=self._concurrency_manager,
            conversation_source=conversation_source,
            handle_credit_result=handle_credit_result
            if inspect.iscoroutinefunction(handle_credit_result)
            else None,
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
            # Drop any first-token keys for this phase whose CreditReturn never
            # reached the reconcile path (e.g. returns dropped as late/reclaimed),
            # so the set can't grow unbounded across long-lived handlers.
            self._first_token_received = {
                key for key in self._first_token_received if key[0] != phase
            }
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
        phase = credit.phase

        # Get phase handler (returns None if phase already cleaned up)
        handler = self._phase_handlers.get(phase)
        if not handler:
            _logger.debug(
                lambda: f"Credit return for unregistered phase {phase}, "
                f"credit_id={credit.id}, worker={worker_id}"
            )
            return

        # Late arrivals after phase complete are logged but don't affect counts
        if handler.lifecycle.is_complete:
            _logger.warning(
                lambda: f"Credit return after phase {phase} complete, "
                f"credit_id={credit.id}, worker={worker_id}"
            )
            return

        disposition = self._get_return_disposition(credit_return, handler)

        # Reconcile against the authoritative first-token record. ``on_first_token``
        # already released the prefill slot for any credit in this set, so treat
        # first-token as sent even when the worker's CreditReturn flag is a stale
        # False (cancellation between FirstToken-send and the flag write). Discard
        # the key so a future late return can't re-consult it.
        first_token_key = (phase, credit.id)
        first_token_already_released = first_token_key in self._first_token_received
        self._first_token_received.discard(first_token_key)
        first_token_sent = (
            credit_return.first_token_sent or first_token_already_released
        )

        # 1. ATOMIC COUNTING (no await before this!)
        is_final_returned = handler.progress.increment_returned(
            credit.is_final_turn,
            credit_return.cancelled,
            session_ended=disposition.session_ended,
            session_cancelled=disposition.session_cancelled,
            errored=credit_return.error is not None,
        )

        # 2. Track prefill release if TTFT never arrived
        if not first_token_sent:
            handler.progress.increment_prefill_released()

        # 3. Release concurrency slots
        self._release_slots_for_return(
            phase,
            credit=credit,
            first_token_sent=first_token_sent,
            is_final_returned=is_final_returned,
            session_ended=disposition.session_ended,
            handler=handler,
        )

        # 4. Signal completion if this was the final return — but defer for
        # DAG runs where ``intercept`` is about to spawn children OR there
        # is already pending DAG work in flight. Without this defer, the
        # phase runner unblocks at sending-complete and tears the
        # orchestrator down before its children land. The deferred check at
        # the bottom re-evaluates and sets the event once intercept's
        # synchronous work is done.
        if is_final_returned and not self._dag_work_pending(credit):
            handler.progress.all_credits_returned_event.set()

        # 4a. Per-return strategy hook (adaptive scaling consumes each return
        # to drive its SLA controller). Runs before DAG hooks/dispatch.
        if handler.handle_credit_result is not None:
            await handler.handle_credit_result(credit_return)

        # 4b. DAG child completion hook. When a child session's final turn
        # returns, notify the orchestrator so it can decrement join refcounts,
        # release sticky-routing entries, and dispatch the parent's gated turn.
        await self._notify_orchestrator_of_child_completion(credit, credit_return)

        # 5. DAG intercept — parent/root credit returns may spawn child
        # sessions and may suspend the parent's next turn on a gated
        # SPAWN_JOIN. Runs BEFORE strategy dispatch so the orchestrator can
        # take over the next-turn path. Non-DAG runs return False.
        intercepted = await self._intercept_for_dag(credit)

        # 6. Cleanup ended sessions or notify strategy for subsequent turns.
        # Skipped when the orchestrator intercepted (it owns the next turn).
        if not intercepted:
            if disposition.session_ended:
                handle_session_ended = getattr(
                    handler.strategy, "handle_session_ended", None
                )
                if handle_session_ended is not None:
                    await handle_session_ended(credit)
            elif disposition.should_continue:
                # Child non-final returns ALWAYS notify the strategy so its
                # ``_issue_child_continuation_or_release`` can fire
                # ``on_child_stopped`` when the cap blocks dispatch — otherwise
                # the parent's pending join would never drain. Root credits
                # stay gated on ``can_send_any_turn``.
                is_child_non_final = credit.agent_depth > 0 and not credit.is_final_turn
                if is_child_non_final or handler.stop_checker.can_send_any_turn():
                    await handler.strategy.handle_credit_return(credit)

        # 7. Deferred all-credits-returned check. The orchestrator can drain
        # the DAG synchronously inside ``intercept`` (e.g. cap=1: every spawned
        # child refused at the gate). If we skipped this, the event would
        # never fire because no future credit return is coming.
        self._maybe_signal_dag_completion(phase, handler)

    def _dag_work_pending(self, credit: Credit) -> bool:
        """True iff the orchestrator has work in flight or will spawn on this
        credit return (so the all-credits-returned signal must defer until
        after ``intercept`` runs).

        ``intercept`` runs at every ``agent_depth`` (nested DAGs are
        supported), so the branch-id lookup must run at every depth too.
        """
        orchestrator = self._branch_orchestrators.get(credit.phase)
        if orchestrator is None:
            return False
        if orchestrator.has_pending_branch_work():
            return True
        try:
            if orchestrator.get_branch_ids(credit):
                return True
        except Exception:  # probe must never break the return path
            return False
        return False

    def _maybe_signal_dag_completion(
        self, phase: CreditPhase, handler: PhaseCallbackContext
    ) -> None:
        """Set the all-credits-returned event when the orchestrator drained the
        DAG synchronously inside ``intercept``.
        """
        orchestrator = self._branch_orchestrators.get(phase)
        if orchestrator is None:
            return
        if (
            not handler.progress.all_credits_returned_event.is_set()
            and handler.progress.check_all_returned_or_cancelled()
            and not orchestrator.has_pending_branch_work()
        ):
            handler.progress.all_credits_returned_event.set()

    async def _notify_orchestrator_of_child_completion(
        self, credit: Credit, credit_return: CreditReturn
    ) -> None:
        """Fire the orchestrator's child-completion hook on a child final return."""
        orchestrator = self._branch_orchestrators.get(credit.phase)
        if not credit.is_final_turn or credit.agent_depth == 0 or orchestrator is None:
            return
        try:
            if credit_return.error is not None:
                await orchestrator.on_child_errored(credit.x_correlation_id)
            else:
                await orchestrator.on_child_leaf_reached(credit.x_correlation_id)
        except Exception as exc:  # orchestrator hook boundary
            _logger.warning(
                lambda exc=exc: f"BranchOrchestrator child-completion hook "
                f"failed for x_correlation_id={credit.x_correlation_id}: {exc}"
            )

    async def _intercept_for_dag(self, credit: Credit) -> bool:
        """Offer a credit return to the orchestrator's intercept hook.

        Returns True iff the orchestrator suppressed the strategy's next-turn
        dispatch (parent suspended on a gated turn). Non-DAG runs return
        False unconditionally.
        """
        orchestrator = self._branch_orchestrators.get(credit.phase)
        if orchestrator is None:
            return False
        try:
            return await orchestrator.intercept(credit)
        except Exception as exc:  # orchestrator hook boundary
            _logger.warning(
                lambda exc=exc: f"BranchOrchestrator intercept failed for "
                f"credit {credit.id}: {exc}"
            )
            return False

    def _get_return_disposition(
        self,
        credit_return: CreditReturn,
        handler: PhaseCallbackContext,
    ) -> ReturnDisposition:
        """Determine whether a returned credit can advance the conversation.

        Lost-worker recovery synthesizes cancelled returns on the router. Those
        should only continue multi-turn sessions when the dataset already carries
        assistant responses, because another worker can then reconstruct context
        without the lost worker's live session state.
        """
        credit = credit_return.credit
        if credit.is_final_turn:
            # Only root credits (agent_depth == 0) end a session here. DAG
            # children inherit the parent's session slot and never bump
            # ``_sent_sessions`` in ``increment_sent``; counting their final
            # return as a session-end would over-count completed/cancelled
            # sessions and drive ``in_flight_sessions`` negative. A child's
            # final turn just stops continuation (no session counting, no
            # slot release); the orchestrator child-completion hook handles
            # the DAG-side bookkeeping independently.
            if credit.agent_depth == 0:
                return ReturnDisposition(
                    should_continue=False,
                    session_ended=True,
                    session_cancelled=credit_return.cancelled,
                )
            return ReturnDisposition(should_continue=False, session_ended=False)

        if not self._requires_worker_migration(credit_return):
            return ReturnDisposition(should_continue=True, session_ended=False)

        if credit.allow_worker_migration:
            return ReturnDisposition(should_continue=True, session_ended=False)

        # Migration is refused, so the session cannot continue. Only root
        # credits (agent_depth == 0) release the session slot and count as a
        # cancelled session here. DAG children inherit the parent's slot and
        # never acquired one, so forcing their cancellation must not bump
        # ``_cancelled_sessions`` or release a slot the child never held; the
        # orchestrator child-completion hook owns DAG-side bookkeeping.
        if credit.agent_depth == 0:
            return ReturnDisposition(
                should_continue=False,
                session_ended=True,
                session_cancelled=True,
            )
        return ReturnDisposition(should_continue=False, session_ended=False)

    @staticmethod
    def _requires_worker_migration(credit_return: CreditReturn) -> bool:
        """Whether continuation depends on routing the session to a different worker."""
        if credit_return.worker_detached:
            return True

        error = credit_return.error or ""
        return credit_return.cancelled and error.startswith("worker_unavailable:")

    def _release_slots_for_return(
        self,
        phase: CreditPhase,
        *,
        credit: Credit,
        first_token_sent: bool,
        is_final_returned: bool,
        session_ended: bool,
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
            first_token_sent: Authoritative first-token state (worker flag OR-ed
                with the handler's ``on_first_token`` record) so the prefill slot
                is never released twice.
            is_final_returned: True if this is the last credit of the phase.
            handler: Phase callback context.
        """
        concurrency = handler.concurrency_manager

        # Release session slot when a conversation ends (final turn, whether
        # completed or cancelled). DAG children (agent_depth > 0) inherit the
        # root's session slot via the dispatch path that bypasses
        # ``acquire_session_slot``; releasing here would underflow.
        if session_ended and credit.agent_depth == 0:
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
        if not first_token_sent:
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

        # Record that this credit's prefill slot was released here so a later
        # CreditReturn carrying a stale ``first_token_sent=False`` (cancellation
        # between FirstToken-send and the worker's flag write) cannot release it
        # a second time and over-grant prefill permits.
        self._first_token_received.add((phase, first_token.credit_id))

        # Track the release
        handler.progress.increment_prefill_released()

        # Release the prefill slot
        handler.concurrency_manager.release_prefill_slot(phase)
