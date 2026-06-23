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
    from collections.abc import Awaitable, Callable

    from aiperf.credit.messages import CreditReturn, FirstToken
    from aiperf.credit.structs import Credit
    from aiperf.timing.branch_orchestrator import BranchOrchestrator
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.phase.lifecycle import PhaseLifecycle
    from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
    from aiperf.timing.phase.stop_conditions import StopConditionChecker
    from aiperf.timing.session_tree import SessionTreeRegistry
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

    def __init__(
        self,
        concurrency_manager: ConcurrencyManager,
        branch_orchestrator: BranchOrchestrator | None = None,
        session_tree_registry: SessionTreeRegistry | None = None,
        on_warmup_abort: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Initialize callback handler.

        Args:
            concurrency_manager: Manages concurrency slots (shared across phases).
            branch_orchestrator: Optional DAG subagent orchestrator. When
                provided, credit returns are offered to ``orchestrator.intercept``
                before the strategy's ``handle_credit_return`` is called. If
                intercept returns True the strategy dispatch is suppressed (the
                orchestrator has taken over the next-turn path by spawning
                children / queuing a join turn).
            session_tree_registry: Optional per-tree session-slot ledger (agentic
                replay only). When engaged (PROFILING), the depth-0 root session
                slot is NOT released on the root's final turn -- it is held until
                the whole tree drains. The release is deferred to
                ``registry.on_root_terminal`` (after intercept, so final-turn
                spawns are counted first); the per-phase teardown releases any
                still-open trees via the runner's ``release_all``.
            on_warmup_abort: Optional async callback fired ONCE on the first
                terminal WARMUP failure. Used by agentic replay to abort the run
                early (broadcast ProfileCancelCommand) instead of letting warmup
                drain to teardown -- a degraded trajectory pool means PROFILING
                must not start, so there is no value in waiting for the rest of
                the warmup credits to return. ``None`` -> legacy teardown-time
                abort via the strategy's ``report_warmup_failures`` only.
        """
        self._concurrency_manager = concurrency_manager
        self._branch_orchestrator = branch_orchestrator
        self._session_tree_registry = session_tree_registry
        self._on_warmup_abort = on_warmup_abort
        self._warmup_abort_triggered = False
        self._phase_handlers: dict[CreditPhase, PhaseCallbackContext] = {}

    @property
    def on_warmup_abort(self) -> Callable[[], Awaitable[None]] | None:
        """The wired live warmup-abort callback, or None if not enabled.

        When non-None, the live path (first terminal warmup failure ->
        ProfileCancelCommand) is authoritative and PhaseRunner skips its
        teardown ``report_warmup_failures`` raise (which is only a backstop for
        the un-wired case).
        """
        return self._on_warmup_abort

    def _tree_registry_engaged(self, credit: Credit) -> bool:
        """True when per-tree session-slot accounting owns this credit's slot.

        Scoped to PROFILING (WARMUP keeps the legacy in-flight teardown release;
        warmup credits spawn no descendants). None elsewhere -> legacy path.
        """
        return (
            self._session_tree_registry is not None
            and credit.phase == CreditPhase.PROFILING
        )

    def set_branch_orchestrator(self, orchestrator: BranchOrchestrator | None) -> None:
        """Inject the subagent orchestrator post-construction.

        Also registers a drain observer on the orchestrator so the deferred
        completion check fires when the orchestrator's last drain step lands
        AFTER the final ``on_credit_return`` callback (concurrency race:
        under N>1, ``has_pending_branch_work()`` can flip False between
        credit returns, with no further return arriving to re-trigger the
        check). Without this hook the phase runner relies on the pre-wait
        short-circuit + drain-timeout backstop; the drain timeout cost is
        avoided here.
        """
        if (
            self._branch_orchestrator is not None
            and self._branch_orchestrator is not orchestrator
        ):
            self._branch_orchestrator.set_drain_observer(None)
        self._branch_orchestrator = orchestrator
        if orchestrator is not None:
            orchestrator.set_drain_observer(self._on_orchestrator_drain)

    def _on_orchestrator_drain(self) -> None:
        """Re-evaluate the deferred all-credits-returned check across every
        active phase handler. Idempotent: per-handler check no-ops if the
        event is already set or the predicate disagrees.
        """
        for handler in self._phase_handlers.values():
            if handler.lifecycle.is_complete:
                continue
            if (
                self._branch_orchestrator is not None
                and not handler.progress.all_credits_returned_event.is_set()
                and handler.progress.check_all_returned_or_cancelled()
                and not self._branch_orchestrator.has_pending_branch_work()
            ):
                handler.progress.all_credits_returned_event.set()

    def _credit_will_dispatch_children(self, credit: Credit) -> bool:
        """Return True if the completing credit's turn declares DAG spawns.

        Used to defer the phase-level ``all_credits_returned_event`` when a
        root's final return is about to trigger sub-agent dispatches via the
        orchestrator intercept.
        """
        orch = self._branch_orchestrator
        if orch is None:
            return False
        return bool(orch.get_branch_ids(credit))

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

    async def _handle_warmup_failure(
        self,
        credit: Credit,
        credit_return: CreditReturn,
        context: PhaseCallbackContext,
        phase: CreditPhase,
    ) -> None:
        """Accumulate, and live-abort on, a terminal WARMUP root failure.

        AgenticReplayStrategy exposes ``record_warmup_failure(trace_id)``;
        PhaseRunner calls ``report_warmup_failures()`` at WARMUP teardown to abort
        PROFILING if any trajectory burned its only warmup credit on a terminal
        error or cancellation. Duck-typed: only fires when the active strategy
        implements the hook, so non-replay strategies are unaffected.

        Do NOT gate on ``credit.is_final_turn``: a WARMUP credit primes the single
        turn k_i (the last request before t*), and PROFILING resumes the same
        trajectory at k_i+1, so for a session active at t* the warmed turn is never
        the trajectory's final turn (k_i < num_turns-1) and ``is_final_turn`` is
        False. WARMUP dispatches exactly one credit per session and its return is a
        strategy-level no-op, so every WARMUP root return IS the terminal warmup
        event for that trajectory -- gating on ``is_final_turn`` made this
        accumulation dead for the entire normal warmup population, silently letting
        a degraded pool proceed to PROFILING.

        When ``on_warmup_abort`` is wired, the FIRST terminal failure also aborts
        the run live (broadcast ProfileCancelCommand) instead of waiting the full
        warmup drain for the teardown-time ``report_warmup_failures`` raise: a
        single failure already breaks the contract, and the broadcast cancels
        in-flight warmup and drives a clean records-manager + system-controller
        shutdown. Fired at most once.
        """
        if not (
            phase == CreditPhase.WARMUP
            and credit.agent_depth == 0
            and (credit_return.error is not None or credit_return.cancelled)
        ):
            return
        record_warmup_failure = getattr(context.strategy, "record_warmup_failure", None)
        if record_warmup_failure is None:
            return
        record_warmup_failure(credit.conversation_id)
        if self._on_warmup_abort is None or self._warmup_abort_triggered:
            return
        self._warmup_abort_triggered = True
        _logger.warning(
            lambda: f"Terminal warmup failure for trace {credit.conversation_id}; "
            f"aborting run early (broadcasting ProfileCancelCommand)."
        )
        try:
            await self._on_warmup_abort()
        except Exception as exc:  # noqa: BLE001
            # Mirror the records-side threshold abort: if the broadcast fails,
            # reset the flag so a later warmup return retries, and the runner's
            # teardown backstop can still surface the failure.
            self._warmup_abort_triggered = False
            _logger.warning(lambda exc=exc: f"Failed to broadcast warmup abort: {exc!r}")

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

        # 1. ATOMIC COUNTING (no await before this!)
        # DAG children are off the phase's planning books — they inherit
        # the root's session slot and are tracked by the
        # ``BranchOrchestrator``. Their returns are signalled via the
        # ``on_child_*`` hooks below; passing ``is_child=True`` keeps
        # ``requests_completed`` / ``requests_cancelled`` root-only.
        is_final_returned = handler.progress.increment_returned(
            credit.is_final_turn,
            credit_return.cancelled,
            errored=credit_return.error is not None,
            is_child=credit.agent_depth > 0,
        )

        # 2. Track prefill release if TTFT never arrived
        if not credit_return.first_token_sent:
            handler.progress.increment_prefill_released()

        # 3. Release concurrency slots
        self._release_slots_for_return(
            phase, credit, credit_return, is_final_returned, handler
        )

        # 4. Signal completion if this was the final return. Deferred for
        # DAG runs: if the orchestrator already has pending descendants in
        # flight, or if this credit's intercept will spawn fresh children,
        # more credits will be sent/returned. We set the event only after
        # the orchestrator has confirmed no more work (see the post-intercept
        # guard below).
        defer_completion_signal_for_dag = False
        if (
            is_final_returned
            and self._branch_orchestrator is not None
            and (
                self._branch_orchestrator.has_pending_branch_work()
                or self._credit_will_dispatch_children(credit)
            )
        ):
            # Already-pending descendants (from prior spawns) or this credit's
            # own about-to-spawn children both require deferring the signal.
            defer_completion_signal_for_dag = True

        if is_final_returned and not defer_completion_signal_for_dag:
            handler.progress.all_credits_returned_event.set()

        # 4b. DAG child completion hook.
        # When a child session's final turn returns, notify the orchestrator so
        # it can decrement join refcounts, release sticky-routing entries, and
        # dispatch the parent's join turn (if any). Runs regardless of whether
        # the phase can still send, because children may finish after the
        # parent has already sent its terminal turn.
        # NOTE: credit_return.error is a free-form string produced by the
        # worker's transport/server error path. We treat any non-None value as
        # an error signal; cancellation is tracked separately via
        # credit_return.cancelled and is NOT treated as a child error.
        if (
            credit.is_final_turn
            and credit.agent_depth > 0
            and self._branch_orchestrator is not None
        ):
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

        # 5. Dispatch next turn / DAG spawn.
        #
        # The orchestrator intercept runs FIRST and unconditionally (not gated
        # behind ``can_send_any_turn``), because when a DAG root finishes its
        # own terminal turn the phase's "sending complete" lifecycle flag has
        # already flipped — but the children still need to dispatch. The
        # orchestrator owns its own dispatch path (``CreditIssuer.
        # dispatch_first_turn``) which bypasses the session-level stop checks
        # for DAG children (they inherit the root's session slot).
        #
        # Strategy dispatch (for regular multi-turn continuation) remains gated
        # behind ``can_send_any_turn`` as before.
        intercepted = False
        if self._branch_orchestrator is not None:
            intercepted = await self._branch_orchestrator.intercept(credit)
            if intercepted:
                return

        # Per-tree slot release: a root's final-turn return marks its tree's
        # root token complete. Run AFTER intercept so any children spawned on the
        # final turn are already registered with the registry (else the tree
        # could drain a beat too early). The registry releases the session slot
        # and recycles the freed lane only once every descendant has also
        # drained -- which may be now (no outstanding descendants) or later (when
        # the last background subagent finishes). A root that suspends on a gate
        # (intercepted) is never final, so it never reaches this point.
        if (
            credit.is_final_turn
            and credit.agent_depth == 0
            and self._tree_registry_engaged(credit)
        ):
            self._session_tree_registry.on_root_terminal(
                credit.effective_root_correlation_id
            )

        # Strategy dispatch (queue next turn of the same session). Normally
        # gated behind ``can_send_any_turn``; however, for DAG-spawned
        # descendants (``credit.agent_depth > 0``) the next turn is gated
        # behind ``can_send_child_turn`` instead — the phase-level
        # sending-complete flag is driven by root sampling exhaustion, not
        # by DAG work, but the global ``--request-count`` cap still
        # applies. When the cap blocks a non-final child continuation, we
        # notify the orchestrator (``on_child_stopped``) so the parent's
        # join still drains instead of deadlocking on a child whose
        # remaining turns will never be issued. Final-turn child returns
        # are always passed through (the strategy is a no-op for them, but
        # observer hooks still need to fire).
        is_child = credit.agent_depth > 0
        if not is_child:
            if handler.stop_checker.can_send_any_turn():
                await handler.strategy.handle_credit_return(
                    credit, error=credit_return.error
                )
        elif credit.is_final_turn or handler.stop_checker.can_send_child_turn():
            await handler.strategy.handle_credit_return(
                credit, error=credit_return.error
            )
        elif self._branch_orchestrator is not None:
            try:
                await self._branch_orchestrator.on_child_stopped(
                    credit.x_correlation_id
                )
            except Exception as exc:  # noqa: BLE001
                _logger.warning(
                    lambda exc=exc: f"BranchOrchestrator on_child_stopped "
                    f"hook failed for x_correlation_id="
                    f"{credit.x_correlation_id}: {exc}"
                )

        # WARMUP terminal-failure accumulation + live early-abort (agentic replay).
        await self._handle_warmup_failure(credit, credit_return, handler, phase)

        # Deferred all-credits-returned check. Runs on EVERY return — root
        # or child — because child returns don't bump the phase counters
        # (they're tracked by the BranchOrchestrator, not ``CreditCounter``)
        # and so can't flip ``is_final_returned`` themselves. The last
        # child's evict-and-drain cascade is what clears
        # ``has_pending_branch_work``, at which point this check on the
        # child's own return path fires the event.
        if (
            self._branch_orchestrator is not None
            and not handler.progress.all_credits_returned_event.is_set()
            and handler.progress.check_all_returned_or_cancelled()
            and not self._branch_orchestrator.has_pending_branch_work()
        ):
            handler.progress.all_credits_returned_event.set()

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
        tree_engaged = self._tree_registry_engaged(credit)

        # Release session slot when a root conversation ends (final turn,
        # whether completed or cancelled). DAG children (agent_depth > 0)
        # inherit the root's session slot via ``issue_credit``'s is_child
        # bypass and therefore never acquired one of their own; releasing
        # here would underflow the session semaphore.
        #
        # Under per-tree accounting the slot belongs to the whole TREE, not the
        # root credit: do NOT release on the root's final turn here. The release
        # is deferred to ``registry.on_root_terminal`` (called after intercept,
        # so children spawned on the final turn are counted first), which frees
        # the slot only once every descendant has also drained.
        if credit.is_final_turn and credit.agent_depth == 0:
            root_corr = credit.effective_root_correlation_id
            if not (tree_engaged and self._session_tree_registry.has_tree(root_corr)):
                concurrency.release_session_slot(phase)

        # On phase end, release slots for sessions still in flight.
        # These are sessions that started but whose final turn was never sent/returned.
        # Under per-tree accounting the registry owns every session slot, so the
        # runner releases any still-open trees at phase cleanup (release_all);
        # releasing here would race the registry and over-release.
        if is_final_returned and not tree_engaged:
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
