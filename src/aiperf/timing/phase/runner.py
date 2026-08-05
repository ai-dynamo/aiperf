# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase runner for credit phase lifecycle management.

Coordinates phase execution: create components → start → wait for sends → wait for returns → complete.
Owns the LoopScheduler and all per-phase components (lifecycle, progress, stop_checker, credit_issuer).
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from aiperf.common.enums import BaselineKind, CacheBustTarget, CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.mixins import TaskManagerMixin
from aiperf.common.phase import phase_runtime_key
from aiperf.credit.issuer import CreditIssuer
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TimingMode
from aiperf.timing.branch_orchestrator import BranchOrchestrator
from aiperf.timing.phase.lifecycle import PhaseLifecycle
from aiperf.timing.phase.progress_tracker import PhaseProgressTracker
from aiperf.timing.phase.stop_conditions import StopConditionChecker
from aiperf.timing.ramping import Ramper, RamperConfig, RampType
from aiperf.timing.rate_series import RateSeriesController
from aiperf.timing.replay_dependencies import ReplayBarrierCoordinator
from aiperf.timing.request_cancellation import RequestCancellationSimulator
from aiperf.timing.strategies.core import RateSettableProtocol
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.common.models import BranchStats, CreditPhaseStats, DatasetMetadata
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.credit.callback_handler import CreditCallbackHandler
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.conversation_source import ConversationSource
    from aiperf.timing.phase.publisher import PhasePublisher
    from aiperf.timing.request_cancellation import RequestCancellationSimulator
    from aiperf.timing.session_tree import SessionTreeRegistry
    from aiperf.timing.strategies.core import TimingStrategyProtocol


class RateControllerProtocol(Protocol):
    """Background controller that can update phase limits over time."""

    def start(self) -> asyncio.Task: ...

    def stop(self) -> None: ...


class PhaseRunner(TaskManagerMixin):
    """Executes credit phases with full lifecycle management.

    Creates all per-phase components lazily during run():
    - LoopScheduler (SINGLE owner - key architectural decision)
    - PhaseLifecycle (state machine)
    - PhaseProgressTracker (wraps counter + events)
    - StopConditionChecker (evaluates stop conditions)
    - CreditIssuer (issues credits with concurrency control)

    Lifecycle:
        1. Create components
        2. Register phase with callback handler
        3. Setup timing strategy with injected dependencies
        4. Start phase (mark started, publish)
        5. Execute timing strategy (with timeout)
        6. Wait for returns (with grace period)
        7. Complete phase (mark complete, publish)
        8. Cleanup (cancel scheduler, stop rampers)

    Component Ownership Diagram:
        PhaseRunner (owns)
            ├── LoopScheduler
            ├── PhaseLifecycle
            ├── PhaseProgressTracker
            │       └── CreditCounter (owned by tracker)
            ├── StopConditionChecker (reads lifecycle + counter)
            └── CreditIssuer (uses stop_checker, progress, concurrency, router)
    """

    def __init__(
        self,
        *,
        config: CreditPhaseConfig,
        conversation_source: ConversationSource,
        phase_publisher: PhasePublisher,
        credit_router: CreditRouterProtocol,
        concurrency_manager: ConcurrencyManager,
        cancellation_policy: RequestCancellationSimulator | None = None,
        callback_handler: CreditCallbackHandler,
        url_selection_strategy: URLSelectionStrategyProtocol | None = None,
        branch_orchestrator: BranchOrchestrator | None = None,
        run: BenchmarkRun | None = None,
        session_tree_registry: SessionTreeRegistry | None = None,
        **kwargs,
    ) -> None:
        """Initialize phase runner.

        Args:
            config: Phase configuration (phase enum, stop conditions, concurrency limits).
            conversation_source: Source for conversation data (shared across phases).
            phase_publisher: Publishes phase lifecycle events to message bus.
            credit_router: Routes credits to workers (for cancel_all_credits on timeout).
            concurrency_manager: Manages session and prefill concurrency slots.
            cancellation_policy: Determines credit cancellation delays.
            callback_handler: Handles credit returns and TTFT events.
            url_selection_strategy: Optional URL selection strategy for multi-URL
                load balancing. Passed to CreditIssuer.
            branch_orchestrator: Optional DAG branch orchestrator. When present,
                ``_is_phase_complete`` consults ``has_pending_branch_work`` so
                completion blocks while DAG children are still in flight, even
                after ``--request-count`` is reached.
            run: Optional ``BenchmarkRun`` threaded to strategies that need full
                config (AgenticReplayStrategy reads cache-bust target /
                benchmark_id) and into the ``BranchOrchestrator`` ctor.
            session_tree_registry: Optional per-tree session-slot ledger
                (AGENTIC_REPLAY only). Passed into the orchestrator + issuer +
                released at phase teardown via ``release_all``.
        """
        super().__init__(**kwargs)
        self._config = config
        self._conversation_source = conversation_source
        self._branch_orchestrator = branch_orchestrator
        self._run = run
        self._session_tree_registry = session_tree_registry
        self._cache_warmup_enabled = isinstance(
            getattr(config, "agentic_cache_warmup_duration_sec", None),
            int | float,
        )

        # For FIXED_SCHEDULE mode, use actual dataset size instead of config values.
        # Config values may reflect pre-filtered file size, but dataset_metadata
        # reflects the actual filtered dataset after start/end offset filtering.
        metadata = conversation_source.dataset_metadata
        if config.timing_mode == TimingMode.FIXED_SCHEDULE and metadata:
            self._config = config.model_copy(
                update={
                    "total_expected_requests": metadata.total_turn_count,
                    "expected_num_sessions": len(metadata.conversations),
                }
            )
        elif (
            config.timing_mode == TimingMode.AGENTIC_REPLAY
            and config.phase == CreditPhase.WARMUP
            and not self._cache_warmup_enabled
        ):
            # AGENTIC_REPLAY warmup dispatches one priming credit per warmable
            # stream (root + each mid-flight subagent at t*), which exceeds the
            # `concurrency` placeholder when lanes hold multiple streams. Without
            # this re-anchor the concurrency-sized barrier fires early and cancels
            # the closest-to-t* priming credits -- under-priming the server cache
            # and masking warmup failures for the cancelled streams. Re-anchor the
            # barrier to the actual dispatch count (``warmup_credit_count``
            # promises exactly this). Single-stream lanes already equal
            # concurrency, so this is a no-op for them.
            warmup_count = getattr(conversation_source, "warmup_credit_count", None)
            if warmup_count:
                self._config = config.model_copy(
                    update={"total_expected_requests": warmup_count}
                )
        self._phase_publisher = phase_publisher
        self._credit_router = credit_router
        self._concurrency_manager = concurrency_manager
        self._cancellation_policy = cancellation_policy or RequestCancellationSimulator(
            self._config.request_cancellation
        )
        self._callback_handler = callback_handler
        self._on_phase_complete: Callable[[], None] | None = None
        self._on_phase_error: Callable[[BaseException], None] | None = None

        # Per-phase components - order matters
        self._scheduler = LoopScheduler()
        self._lifecycle = PhaseLifecycle(self._config)
        self._progress = PhaseProgressTracker(self._config)
        self._stop_checker = StopConditionChecker(
            config=self._config,
            lifecycle=self._lifecycle,
            counter=self._progress.counter,
        )
        self._replay_barrier = (
            ReplayBarrierCoordinator(self._conversation_source.dataset_metadata)
            if (
                self._config.timing_mode == TimingMode.AGENTIC_REPLAY
                and self._conversation_source.dataset_metadata is not None
            )
            else None
        )
        self._credit_issuer = self._build_credit_issuer(url_selection_strategy)
        self._maybe_construct_branch_orchestrator(conversation_source)
        self._wire_replay_gate()

        self._execution_task: asyncio.Task | None = None
        self._progress_task: asyncio.Task | None = None
        self._return_wait_task: asyncio.Task | None = None
        self._was_cancelled = False
        self._rampers: list[RateControllerProtocol] = []
        self._baseline_start_ns: int | None = None
        self._baseline_end_ns: int | None = None

    def _build_credit_issuer(
        self, url_selection_strategy: URLSelectionStrategyProtocol | None
    ) -> CreditIssuer:
        """Construct the CreditIssuer with the per-phase components already
        wired by ``__init__``. Split out so ``__init__`` stays under the
        ergonomics file-size cap."""
        return CreditIssuer(
            phase=self._config.phase,
            phase_index=self._config.phase_index,
            profiling_index=self._config.profiling_index,
            phase_name=self._config.phase_name,
            phase_kind=self._config.phase_kind,
            stop_checker=self._stop_checker,
            progress=self._progress,
            concurrency_manager=self._concurrency_manager,
            credit_router=self._credit_router,
            cancellation_policy=self._cancellation_policy,
            lifecycle=self._lifecycle,
            url_selection_strategy=url_selection_strategy,
            session_tree_registry=self._session_tree_registry,
            session_tree_registry_enabled=(
                self._config.phase == CreditPhase.PROFILING
                or self._cache_warmup_enabled
            ),
            replay_barrier=self._replay_barrier,
            cache_bust_target=(
                self._run.cfg.get_cache_bust_target()
                if self._run is not None
                else CacheBustTarget.NONE
            ),
        )

    def _maybe_construct_branch_orchestrator(
        self, conversation_source: ConversationSource
    ) -> None:
        """Construct ``BranchOrchestrator`` for DAG-shaped or agentic datasets.

        Build policy:
        - AGENTIC_REPLAY: always build (the trajectory source spawns subagents
          reactively; its metadata may declare no static branches, so the
          ``_is_dag_dataset`` heuristic would miss it).
        - Other timing modes (REQUEST_RATE / USER_CENTRIC / FIXED_SCHEDULE):
          lazy-build only for DAG-shaped
          datasets (metadata declares branches OR has ``agent_depth > 0``
          conversations). Non-DAG runs leave ``self._branch_orchestrator``
          None and the callback / strategy paths skip orchestrator hooks.

        Additional constructor inputs (benchmark_id / cache_bust_target /
        session_tree_registry / cache_bust_ledger) are threaded in for all
        builds; they default to inert values for dag_jsonl
        (cache_bust_target NONE, registry None).
        """
        if self._branch_orchestrator is not None:
            return
        is_agentic_replay = self._config.timing_mode == TimingMode.AGENTIC_REPLAY
        if not is_agentic_replay and not self._is_dag_dataset(
            conversation_source.dataset_metadata
        ):
            return
        sticky_router = getattr(self._credit_router, "sticky_router", None)
        benchmark_id = self._run.benchmark_id if self._run is not None else "unknown"
        cache_bust_target = (
            self._run.cfg.get_cache_bust_target()
            if self._run is not None
            else CacheBustTarget.NONE
        )
        self._branch_orchestrator = BranchOrchestrator(
            conversation_source=conversation_source,
            credit_issuer=self._credit_issuer,
            sticky_router=sticky_router,
            benchmark_id=benchmark_id,
            cache_bust_target=cache_bust_target,
            session_tree_registry=self._session_tree_registry,
            cache_bust_ledger=getattr(conversation_source, "cache_bust_ledger", None),
            allow_accelerated_warmup=self._cache_warmup_enabled,
        )

    def _wire_replay_gate(self) -> None:
        """Connect the issuer's replay gate to the branch orchestrator.

        The gate calls back into the orchestrator to (a) drain a child's join
        when a deferred dispatch is refused at teardown and (b) fire overlap
        dispatch when a parent's spawning request reaches the wire. Both need
        the orchestrator, so this runs after it is constructed; it is a no-op
        on non-DAG runs where no orchestrator exists.
        """
        if self._branch_orchestrator is None:
            return
        self._credit_issuer.replay_gate.set_child_refused(
            self._branch_orchestrator.on_child_stopped
        )
        if self._replay_barrier is not None:
            self._credit_issuer.replay_gate.set_credit_issued(
                self._branch_orchestrator.on_credit_issued
            )

    @property
    def _phase_key(self) -> int | CreditPhase:
        return phase_runtime_key(self._config.phase, self._config.phase_index)

    @property
    def phase(self) -> CreditPhase:
        """Phase enum (WARMUP or PROFILING)."""
        return self._config.phase

    @staticmethod
    def _is_dag_dataset(dataset_metadata: DatasetMetadata | None) -> bool:
        """True iff the dataset declares any DAG fan-out.

        A DAG-shaped dataset has at least one conversation with branches
        attached, or at least one non-root conversation
        (``agent_depth > 0``). Non-DAG runs return False so the
        orchestrator is not constructed (saves the per-conv prereq-index
        build and keeps the callback path orchestrator-free).
        """
        if dataset_metadata is None:
            return False
        for conv in getattr(dataset_metadata, "conversations", None) or []:
            if getattr(conv, "branches", None):
                return True
            if getattr(conv, "agent_depth", 0) > 0:
                return True
        return False

    def set_phase_complete_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to invoke when phase fully completes.

        Used for seamless phases to notify the orchestrator when the background
        return wait task finishes, allowing cleanup of the runner from active list.
        """
        self._on_phase_complete = callback

    def set_phase_error_callback(
        self, callback: Callable[[BaseException], None]
    ) -> None:
        """Set callback to invoke when a seamless non-final phase's detached
        return-wait task ends with a fatal control-node failure.

        ``PhaseRunner.run()`` returns without awaiting the seamless return-wait
        task, so raising inside it would only reject the background task and
        never reach the orchestrator. This callback forwards the failure so the
        run fails instead of reporting success.
        """
        self._on_phase_error = callback

    @property
    def return_wait_task(self) -> asyncio.Task | None:
        """The detached seamless return-wait task (None unless this phase is
        seamless non-final). The orchestrator awaits it as a barrier so a late
        fatal control-node failure surfaces before the run reports success."""
        return self._return_wait_task

    @property
    def control_fatal_error(self) -> BaseException | None:
        """A fatal request-free control-node failure recorded on this phase's
        progress tracker, if any (see ``record_control_fatal_error``)."""
        return self._progress.fatal_error

    def record_control_fatal_error(self, error: BaseException) -> None:
        """Record a fatal control-node failure on THIS phase's progress tracker.

        Called by the orchestrator's fatal-error sink. A control-node failure is
        fatal to the whole run, so it is recorded on every active phase rather
        than only the callback handler's mutable ``progress`` slot -- which,
        under seamless mode's concurrent runners, may point at the wrong phase.
        """
        self._progress.record_fatal_error(error)

    def _is_phase_complete(self) -> bool:
        """Return True if the request-count cap has been reached AND no DAG
        children are still in flight.

        DAG-aware completion gate. ``--request-count`` is a wire-request cap
        that applies to roots and children alike (see
        ``RequestCountStopCondition.applies_to_dag_children``); however, even
        after the cap fires, ``BranchOrchestrator`` may still be holding
        children that have been dispatched but not yet returned. Closing the
        phase before those children land would freeze sent counts mid-DAG and
        drop the in-flight requests.

        Returns False when:
        - ``total_expected_requests`` is unset (this gate doesn't apply —
          completion is driven by other stop conditions like duration).
        - ``requests_sent`` has not yet reached the cap.
        - The branch orchestrator reports pending DAG work.
        """
        cap = self._config.total_expected_requests
        if cap is None:
            return False
        if self._progress.counter.requests_sent < cap:
            return False
        return not (
            self._branch_orchestrator is not None
            and self._branch_orchestrator.has_pending_branch_work()
        )

    def _snapshot_branch_stats(self) -> BranchStats | None:
        """Snapshot the BranchOrchestrator counters for publication.

        Returns None on non-DAG runs (no orchestrator wired). DAG runs
        return a copy of the counters so the published snapshot stays
        stable even if the orchestrator keeps mutating after we
        publish.
        """
        if self._branch_orchestrator is None:
            return None
        return self._branch_orchestrator.snapshot_branch_stats()

    def cancel(self) -> None:
        """Cancel the phase runner (external cancellation like Ctrl+C)."""
        self._was_cancelled = True
        self._lifecycle.cancel()
        if self._execution_task:
            self._execution_task.cancel()
        if self._progress_task:
            self._progress_task.cancel()
        if self._return_wait_task:
            self._return_wait_task.cancel()
        for ramper in self._rampers:
            ramper.stop()
        self._scheduler.cancel_all()

    def _raise_if_control_node_failed(self) -> None:
        """Re-raise a fatal request-free control-node failure recorded on the
        progress tracker (via the router's fatal-error sink), so the phase exits
        with a visible error instead of reporting the graph as complete."""
        if self._progress.fatal_error is not None:
            raise self._progress.fatal_error

    def _on_return_wait_complete(self, task: asyncio.Task) -> None:
        """Handle completion of background return wait task (seamless mode).

        Called when _return_wait_task finishes. Cancels progress reporting and
        notifies the orchestrator callback.
        """
        if self._progress_task:
            self._progress_task.cancel()

        # Retrieve the detached task's own exception so it is not left as an
        # unretrieved-task-exception, and treat it as a phase failure.
        task_exc = None if task.cancelled() else task.exception()

        # Seamless path: a fatal request-free control-node failure (recorded on
        # the progress tracker during the background wait, or raised by the task
        # itself) must not be reported as a clean phase completion. Forward it to
        # the orchestrator so the RUN fails -- the sync path raises
        # ``_raise_if_control_node_failed`` for this, but the seamless task is
        # never awaited, so we propagate via the error callback instead.
        fatal = self._progress.fatal_error or task_exc
        if fatal is not None:
            self.error(
                lambda: (
                    "fatal request-free control-node failure in seamless "
                    f"phase {self._config.phase}: {fatal!r}"
                )
            )
            if self._on_phase_error is not None:
                self._on_phase_error(fatal)
        if self._on_phase_complete:
            self._on_phase_complete()

    def _capture_baseline_boundary(self, phase_id: str, kind: BaselineKind) -> int:
        boundary_ns = time.time_ns()
        self.execute_async(self._publish_phase_baseline_request(phase_id, kind))
        return boundary_ns

    async def _capture_baseline_boundary_before_completion(
        self, phase_id: str, kind: BaselineKind
    ) -> int:
        boundary_ns = time.time_ns()
        await self._publish_phase_baseline_request(phase_id, kind)
        return boundary_ns

    async def _publish_phase_baseline_request(
        self, phase_id: str, kind: BaselineKind
    ) -> None:
        try:
            await self._phase_publisher.publish_phase_baseline_request(
                self._config, phase_id, kind
            )
        except Exception as exc:
            self.warning(
                f"Failed to publish {kind} phase baseline request for "
                f"phase {self._config.phase}: {exc}"
            )

    async def run(
        self,
        is_final_phase: bool,
    ) -> CreditPhaseStats:
        """Execute phase with full lifecycle management.

        Lifecycle: register callback handler → setup strategy → configure rampers →
        start phase → execute timing strategy → wait for sends → wait for returns →
        complete phase → cleanup (cancel scheduler, stop rampers).

        Args:
            is_final_phase: True if this is the last phase. Non-final seamless phases
                spawn background return-wait task; final phases wait synchronously.

        Returns:
            CreditPhaseStats snapshot of final phase state.
        """
        strategy = self._build_strategy()
        try:
            self._register_strategy_with_callback_handler(strategy)
            return await self._run_strategy(strategy, is_final_phase)
        except Exception as e:
            await self._publish_phase_failure_lifecycle()
            raise e
        finally:
            self._detach_orchestrator_and_cleanup()

    def _build_strategy(self) -> TimingStrategyProtocol:
        """Construct the timing strategy class for this phase."""
        StrategyClass = plugins.get_class(
            PluginType.TIMING_STRATEGY, self._config.timing_mode
        )
        return StrategyClass(
            config=self._config,
            conversation_source=self._conversation_source,
            scheduler=self._scheduler,
            stop_checker=self._stop_checker,
            credit_issuer=self._credit_issuer,
            lifecycle=self._lifecycle,
            branch_orchestrator=self._branch_orchestrator,
            # AgenticReplayStrategy reads these; other strategies absorb them
            # via **kwargs.
            run=self._run,
            session_tree_registry=self._session_tree_registry,
            concurrency_manager=self._concurrency_manager,
            progress=self._progress,
        )

    def _register_strategy_with_callback_handler(
        self, strategy: TimingStrategyProtocol
    ) -> None:
        """Register the phase's strategy + (optionally) the orchestrator
        with the shared CreditCallbackHandler before any credits are sent.
        """
        self._callback_handler.register_phase(
            phase=self._config.phase,
            phase_index=self._config.phase_index,
            progress=self._progress,
            lifecycle=self._lifecycle,
            stop_checker=self._stop_checker,
            strategy=strategy,
        )
        if self._branch_orchestrator is not None:
            self._callback_handler.set_branch_orchestrator(self._branch_orchestrator)

    def _detach_orchestrator_and_cleanup(self) -> None:
        """Final-pass orchestrator teardown for the phase.

        Detaches from the shared callback handler so a subsequent phase /
        non-DAG resumption doesn't dispatch into a torn-down orchestrator.
        Final stats are already snapshotted via ``_snapshot_branch_stats``
        before ``publish_phase_complete`` runs. Also sweeps any still-open
        session-tree slots (AGENTIC_REPLAY) so they don't leak into the next
        phase.
        """
        if self._branch_orchestrator is not None:
            self._callback_handler.set_branch_orchestrator(None)
            self._branch_orchestrator.cleanup()
        self._release_tree_slots()

    def _release_tree_slots(self) -> None:
        """Release any still-open session-tree slots at phase teardown.

        Under per-tree accounting (AGENTIC_REPLAY) the registry owns every
        session slot, so trees that never drained (stuck root, lost descendant)
        are swept here so their slots don't leak into the next phase. Idempotent
        (a second call finds no open trees) and a no-op when tree accounting is
        not engaged (registry None -> dag_jsonl / normal modes unaffected).
        """
        if self._session_tree_registry is None:
            return
        released = self._session_tree_registry.release_all(self._phase_key)
        self.info(
            lambda: (
                f"Session-tree slots for phase {self._config.phase}: "
                f"peak_open={self._session_tree_registry.peak_open} "
                f"(target concurrency {self._config.concurrency}); "
                f"released {released} still-open at teardown; "
                f"late_events={self._session_tree_registry.late_events}"
            )
        )

    async def _run_strategy(
        self, strategy: TimingStrategyProtocol, is_final_phase: bool
    ) -> CreditPhaseStats:
        """Drive the strategy through its execute → sending-complete →
        returning-complete pipeline. The exception path (publishing partial
        lifecycle state) lives in the caller's ``except``.
        """
        phase_id = uuid.uuid4().hex
        self._baseline_start_ns = None
        self._baseline_end_ns = None

        self._concurrency_manager.configure_for_phase(
            self._phase_key,
            self._config.concurrency,
            self._config.prefill_concurrency,
        )

        await strategy.setup_phase()

        # Gate credit issuance on worker readiness: on fast startup the first
        # credit can otherwise be issued before any worker registers, which
        # deadlocks the phase (see StickyCreditRouter.wait_for_workers).
        await self._credit_router.wait_for_workers(
            timeout=Environment.SERVICE.START_TIMEOUT
        )

        self._create_rampers(strategy)

        self._baseline_start_ns = self._capture_baseline_boundary(
            phase_id, BaselineKind.START
        )

        self._lifecycle.start()
        stats = self._progress.create_stats(self._lifecycle)
        self.notice(self._format_phase_started(stats))
        await self._phase_publisher.publish_phase_start(self._config, stats)

        self._progress_task = self.execute_async(self._progress_report_loop())

        # Start rampers BEFORE execution to ensure concurrency limits are
        # applied from the start. Otherwise, credits could be issued at full
        # concurrency before the ramper sets the initial (lower) limit.
        for ramper in self._rampers:
            ramper.start()

        # Pre-dispatch DAG SPAWN branches marked dispatch_timing='pre' before
        # the strategy begins issuing root turn-0 credits. No-op for non-DAG
        # runs (orchestrator is None). Suppressed during WARMUP: warmup is
        # one-shot per trajectory and its strategy refuses to advance the
        # children's continuation turns, so pre-dispatching here would leak
        # per-parent/tree descendant counts and can wedge the
        # all_credits_returned_event. Pre-branch dispatch belongs to PROFILING.
        if (
            self._branch_orchestrator is not None
            and self._config.phase != CreditPhase.WARMUP
        ):
            await self._branch_orchestrator.dispatch_pre_session_branches()

        self._execution_task = self.execute_async(strategy.execute_phase())

        await self._wait_for_sending_complete(strategy)

        if self._was_cancelled:
            if not self._lifecycle.is_complete:
                self._lifecycle.mark_complete(grace_period_triggered=False)
                self._progress.freeze_completed_counts()
            self._progress.all_credits_returned_event.set()
            self._baseline_end_ns = (
                await self._capture_baseline_boundary_before_completion(
                    phase_id, BaselineKind.END
                )
            )
            return self._create_final_stats()

        # Seamless mode: phase flows into next without waiting for returns.
        # Progress task continues in background until phase complete.
        if self._config.seamless and not is_final_phase:
            self._return_wait_task = self.execute_async(
                self._wait_for_returning_complete(strategy, phase_id=phase_id)
            )
            self._return_wait_task.add_done_callback(self._on_return_wait_complete)
        else:
            await self._wait_for_returning_complete(strategy, phase_id=phase_id)
            self._progress_task.cancel()
            # Surface a fatal control-node failure recorded during the wait.
            # Checked HERE -- after the wait returns via ANY of its exit paths,
            # including the "all credits already returned" fast path that the
            # fatal-error callback itself unblocks -- so it is never swallowed.
            self._raise_if_control_node_failed()

        for ramper in self._rampers:
            ramper.stop()
        self._scheduler.cancel_all()

        # Accelerated cache-pressure warmup persists its drained DAG state into
        # the shared TrajectorySource here -- after returns are complete and
        # before the orchestrator teardown that runs in the caller's ``finally``
        # (``_detach_orchestrator_and_cleanup``). No-op for other strategies.
        finalize_phase = getattr(strategy, "finalize_phase", None)
        if finalize_phase is not None:
            await finalize_phase()
        if self._preserve_replay_gate_until_finalize(strategy):
            await self._credit_issuer.replay_gate.cancel(notify_refused=False)

        # Strategy-specific phase teardown BACKSTOP. Skipped when the live
        # warmup early-abort already broadcast ProfileCancelCommand (see
        # _should_fire_warmup_backstop), to avoid a double-fire.
        if self._should_fire_warmup_backstop(strategy):
            self._report_warmup_failures(strategy)

        return self._create_final_stats()

    def _create_final_stats(self) -> CreditPhaseStats:
        return self._progress.create_stats_with_baseline_window(
            self._lifecycle,
            baseline_start_ns=self._baseline_start_ns,
            baseline_end_ns=self._baseline_end_ns,
        )

    def _should_fire_warmup_backstop(self, strategy: TimingStrategyProtocol) -> bool:
        """Whether the teardown warmup-failure raise (a BACKSTOP) should fire.

        Only AgenticReplayStrategy exposes ``report_warmup_failures`` (duck-typed);
        raising it aborts the benchmark via ``run()``'s except handler so PROFILING
        never starts with a degraded trajectory pool. No-op outside WARMUP.

        In production this is a backstop, not the primary path: when the live
        warmup early-abort is wired (``callback_handler.on_warmup_abort`` is not
        None), the FIRST terminal failure already broadcast ProfileCancelCommand
        and cancelled this runner, so raising here too is unnecessary and would
        double-fire. We therefore fire only when the live path is NOT wired (and
        the runner was not otherwise cancelled). Gating on ``on_warmup_abort is
        None`` -- a synchronous check -- also avoids the race where the async
        cancel round-trip has not yet set ``_was_cancelled`` at teardown.
        """
        return (
            self._config.phase == CreditPhase.WARMUP
            and getattr(strategy, "report_warmup_failures", None) is not None
            and self._callback_handler.on_warmup_abort is None
            and not self._was_cancelled
        )

    def _report_warmup_failures(self, strategy: TimingStrategyProtocol) -> None:
        """Surface accumulated terminal WARMUP failures before PROFILING starts.

        Duck-typed -- the strategy protocol does not require it; only
        AgenticReplayStrategy implements ``report_warmup_failures``. Raising
        here aborts the benchmark via ``run()``'s except handler so PROFILING
        never begins with a degraded trajectory pool. No-op outside WARMUP.
        """
        if self._config.phase != CreditPhase.WARMUP:
            return
        report_warmup_failures = getattr(strategy, "report_warmup_failures", None)
        if report_warmup_failures is not None:
            report_warmup_failures()

    async def _publish_phase_failure_lifecycle(self) -> None:
        """Flush phase-end lifecycle messages on a hard failure path so other
        services see the phase end and the benchmark doesn't hang forever.
        """
        # TODO: This can be improved a bit by having a better way to notify
        # other services and the system controller of a failure in the
        # benchmark. If there is an error while setting up or executing
        # the phase, we need to flush it through the lifecycle to ensure
        # the other services are notified.
        self.error(f"Error executing phase {self._config.phase.title}")
        if not self._was_cancelled:
            self.cancel()

        if not self._lifecycle.is_started:
            self._lifecycle.start()
            stats = self._progress.create_stats(self._lifecycle)
            await self._phase_publisher.publish_phase_start(self._config, stats)

        if not self._lifecycle.is_sending_complete:
            self._lifecycle.mark_sending_complete(timeout_triggered=False)
            self._progress.freeze_sent_counts()
            self._progress.all_credits_sent_event.set()
            stats = self._progress.create_stats(self._lifecycle)
            await self._phase_publisher.publish_phase_sending_complete(stats)

        if not self._lifecycle.is_complete:
            self._lifecycle.mark_complete(grace_period_triggered=False)
            self._progress.freeze_completed_counts()
            self._progress.all_credits_returned_event.set()
            stats = self._progress.create_stats(self._lifecycle)
            await self._phase_publisher.publish_phase_complete(
                stats, branch_stats=self._snapshot_branch_stats()
            )

    def _create_rampers(self, strategy: TimingStrategyProtocol) -> None:
        """Create rampers for concurrency and rate if ramp durations are configured.

        Concurrency rampers use stepped mode (discrete integer steps), starting at 1.
        Rate rampers use continuous mode (smooth float interpolation), starting at a
        rate proportional to target (to avoid issues when target < 1 QPS).
        """
        self._rampers = []
        config = self._config

        # Session concurrency ramper (stepped mode)
        if config.concurrency_ramp_duration_sec and config.concurrency:
            self.info(
                f"Starting session concurrency ramp: 1 → {config.concurrency} "
                f"over {config.concurrency_ramp_duration_sec}s"
            )
            ramp_config = RamperConfig(
                ramp_type=RampType.LINEAR,
                start=1,
                target=config.concurrency,
                duration_sec=config.concurrency_ramp_duration_sec,
            )

            def setter(limit: float) -> None:
                return self._concurrency_manager.set_session_limit(
                    self._phase_key, int(limit)
                )

            self._rampers.append(Ramper(setter=setter, config=ramp_config))

        # Prefill concurrency ramper (stepped mode)
        if config.prefill_concurrency_ramp_duration_sec and config.prefill_concurrency:
            self.info(
                f"Starting prefill concurrency ramp: 1 → {config.prefill_concurrency} "
                f"over {config.prefill_concurrency_ramp_duration_sec}s"
            )
            ramp_config = RamperConfig(
                ramp_type=RampType.LINEAR,
                start=1,
                target=config.prefill_concurrency,
                duration_sec=config.prefill_concurrency_ramp_duration_sec,
            )

            def setter(limit: float) -> None:
                return self._concurrency_manager.set_prefill_limit(
                    self._phase_key, int(limit)
                )

            self._rampers.append(Ramper(setter=setter, config=ramp_config))

        # Request rate ramper (continuous mode via update_interval)
        if config.request_rate_ramp_duration_sec and config.request_rate:
            # Start at one linear increment (proportional to target, not fixed 1 QPS).
            # This avoids awkward cases where target < 1 QPS would actually increase.
            update_interval = Environment.TIMING.RATE_RAMP_UPDATE_INTERVAL
            start_rate = config.request_rate * (
                update_interval / config.request_rate_ramp_duration_sec
            )
            self.info(
                f"Starting request rate ramp: {start_rate:.2f} → {config.request_rate} QPS "
                f"over {config.request_rate_ramp_duration_sec}s"
            )
            ramp_config = RamperConfig(
                ramp_type=RampType.LINEAR,
                start=start_rate,
                target=config.request_rate,
                duration_sec=config.request_rate_ramp_duration_sec,
                update_interval=update_interval,
            )
            if isinstance(strategy, RateSettableProtocol):
                self._rampers.append(
                    Ramper(setter=strategy.set_request_rate, config=ramp_config)
                )
            else:
                self.warning(
                    f"Strategy {strategy.__class__.__name__} does not implement RateSettableProtocol. "
                    "Request rate will be fixed at the target value."
                )

        self._create_rate_series_controller(strategy)

    def _create_rate_series_controller(self, strategy: TimingStrategyProtocol) -> None:
        """Create a request-rate series controller when configured."""
        config = self._config
        if not config.request_rate_series or not config.request_rate:
            return
        if not isinstance(strategy, RateSettableProtocol):
            self.warning(
                f"Strategy {strategy.__class__.__name__} does not implement RateSettableProtocol. "
                "Request rate series will be ignored."
            )
            return

        points = config.request_rate_series.points
        start_delay = config.request_rate_ramp_duration_sec or 0.0
        self.info(
            f"Starting request rate series: {len(points)} points, "
            f"initial={points[0].qps} QPS, final={points[-1].qps} QPS, "
            f"start_delay={start_delay}s"
        )
        self._rampers.append(
            RateSeriesController(
                setter=strategy.set_request_rate,
                config=config.request_rate_series,
                update_interval=Environment.TIMING.RATE_RAMP_UPDATE_INTERVAL,
                start_delay=start_delay,
            )
        )

    def _format_phase_started(self, stats: CreditPhaseStats) -> str:
        """Format a concise log message for phase start."""
        parts = [
            f"Phase {stats.phase_name or stats.phase} ({stats.phase_kind or stats.phase}) started"
        ]
        if stats.phase_index is not None:
            parts.append(f"phase_index={stats.phase_index}")
        if stats.profiling_index is not None:
            parts.append(f"profiling_index={stats.profiling_index}")
        targets = []
        if stats.total_expected_requests:
            targets.append(f"{stats.total_expected_requests:,} requests")
        if stats.expected_duration_sec:
            targets.append(f"{stats.expected_duration_sec:.1f}s duration")
        if stats.expected_num_sessions:
            targets.append(f"{stats.expected_num_sessions:,} sessions")
        if targets:
            parts.append(f"target: {', '.join(targets)}")
        return " | ".join(parts)

    def _format_phase_sending_complete(self, stats: CreditPhaseStats) -> str:
        """Format a concise log message for phase sending complete."""
        parts = [
            f"Phase {stats.phase_name or stats.phase} ({stats.phase_kind or stats.phase}) sending complete"
        ]
        parts.append(
            f"sent={stats.requests_sent:,}, "
            f"completed={stats.requests_completed:,}, "
            f"in_flight={stats.in_flight_requests:,}"
        )
        if stats.sent_sessions > 0:
            parts.append(
                f"sessions: sent={stats.sent_sessions:,}, "
                f"completed={stats.completed_sessions:,}"
            )
        if stats.timeout_triggered:
            parts.append("timeout_triggered=True")
        return " | ".join(parts)

    def _format_phase_complete(self, stats: CreditPhaseStats) -> str:
        """Format a concise log message for phase complete."""
        parts = [
            f"Phase {stats.phase_name or stats.phase} ({stats.phase_kind or stats.phase}) complete"
        ]
        parts.append(
            f"completed={stats.final_requests_completed:,}, "
            f"cancelled={stats.final_requests_cancelled:,}, "
            f"errors={stats.final_request_errors:,}"
        )
        if stats.final_sent_sessions and stats.final_sent_sessions > 0:
            parts.append(
                f"sessions: completed={stats.final_completed_sessions:,}, "
                f"cancelled={stats.final_cancelled_sessions:,}"
            )
        elapsed = stats.requests_elapsed_time
        parts.append(f"elapsed={elapsed:.2f}s")
        if stats.grace_period_timeout_triggered:
            parts.append("grace_period_timeout=True")
        if stats.was_cancelled:
            parts.append("was_cancelled=True")
        return " | ".join(parts)

    @staticmethod
    def _format_warmup_progress(stats: CreditPhaseStats) -> str:
        """Format a periodic warmup heartbeat for non-interactive logs."""
        returned = stats.requests_completed + stats.requests_cancelled
        target = stats.final_requests_sent or stats.total_expected_requests
        returned_desc = (
            f"returned={returned:,}/{target:,}" if target else f"returned={returned:,}"
        )
        parts = [
            f"Phase {stats.phase} progress",
            returned_desc,
            f"sent={stats.requests_sent:,}",
            f"in_flight={stats.in_flight_requests:,}",
            f"errors={stats.request_errors:,}",
            f"elapsed={stats.requests_elapsed_time:.1f}s",
        ]
        return " | ".join(parts)

    def _preserve_replay_gate_until_finalize(
        self, strategy: TimingStrategyProtocol
    ) -> bool:
        return self._config.phase == CreditPhase.WARMUP and getattr(
            strategy,
            "allows_pending_branch_handoff_after_sending_complete",
            False,
        )

    async def _wait_for_accelerated_warmup_wire_drain(self) -> None:
        while self._progress.in_flight > 0:
            await asyncio.sleep(0.1)

    async def _cancel_accelerated_warmup_drain(self, *, timeout: float | None) -> None:
        stats = self._progress.create_stats(self._lifecycle)
        self.warning(
            "Accelerated warmup drain timed out"
            + (f" after {timeout:.1f}s" if timeout is not None else "")
            + "; cancelling all in-flight warmup credits. "
            f"Stats: sent={stats.requests_sent}, "
            f"completed={stats.requests_completed}, "
            f"cancelled={stats.requests_cancelled}, "
            f"in_flight={stats.in_flight_requests}"
        )
        await self._credit_router.cancel_all_credits()
        drain_timeout = Environment.TIMING.CANCEL_DRAIN_TIMEOUT
        try:
            await asyncio.wait_for(
                self._wait_for_accelerated_warmup_wire_drain(),
                timeout=drain_timeout,
            )
            self.info("All cancelled accelerated-warmup credits returned")
        except TimeoutError:
            self.error(
                f"Timeout waiting {drain_timeout}s for cancelled accelerated-warmup "
                "credits to return. Forcing phase completion."
            )
            self._release_stuck_slots()
        self._progress.all_credits_returned_event.set()

    async def _wait_for_accelerated_warmup_handoff(self) -> None:
        timeout = self._config.grace_period_sec
        if timeout is None or math.isinf(timeout):
            await self._wait_for_accelerated_warmup_wire_drain()
        else:
            try:
                await asyncio.wait_for(
                    self._wait_for_accelerated_warmup_wire_drain(),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                await self._cancel_accelerated_warmup_drain(timeout=timeout)
                raise TimeoutError(
                    "Accelerated warmup drain timed out before all wire "
                    "requests returned"
                ) from exc
        self.info(
            "All accelerated-warmup wire requests returned; "
            "preserving paused DAG work for profiling handoff."
        )
        self._progress.all_credits_returned_event.set()

    async def _wait_for_sending_complete(
        self, strategy: TimingStrategyProtocol
    ) -> None:
        """Wait for phase to send all credits (with timeout).

        Uses lifecycle.time_left_in_seconds() for timeout duration.
        On timeout or completion, cancels pending scheduled requests,
        freezes sent counts, and marks sending complete.
        """
        timed_out = False
        try:
            timeout = self._lifecycle.time_left_in_seconds()
            timed_out = await self._wait_for_event_with_timeout(
                name=f"{self._config.phase} phase sending",
                event=self._progress.all_credits_sent_event,
                timeout=timeout,
                task_to_cancel=self._execution_task,
                set_event_on_timeout=True,
            )
        except Exception as e:
            self.error(
                f"Error waiting for phase {self._config.phase} to send all credits: {e!r}"
            )
        finally:
            if not self._lifecycle.is_sending_complete:
                self._lifecycle.mark_sending_complete(timeout_triggered=timed_out)
                self._progress.freeze_sent_counts()
                self._scheduler.cancel_all_pending()
                self._progress.all_credits_sent_event.set()

            if not self._preserve_replay_gate_until_finalize(strategy):
                await self._credit_issuer.replay_gate.cancel(
                    notify_refused=self._config.phase == CreditPhase.PROFILING
                )

            stats = self._progress.create_stats(self._lifecycle)
            self.notice(self._format_phase_sending_complete(stats))
            await self._phase_publisher.publish_progress(stats)
            await self._phase_publisher.publish_phase_sending_complete(stats)

    async def _wait_for_returning_complete(
        self,
        strategy: TimingStrategyProtocol | None = None,
        *,
        phase_id: str | None = None,
    ) -> None:
        """Wait for all credits to return (with grace period).

        Multi-stage process on timeout:
        1. Initial wait with grace period timeout
        2. If timed out: cancel_all_credits() via credit router
        3. Wait for cancelled credits to drain (CANCEL_DRAIN_TIMEOUT)
        4. If drain times out: release stuck concurrency slots and force completion

        Accelerated cache-pressure warmup hands its paused DAG branches off to
        profiling, so once issuance stops it completes on ``in_flight == 0``
        alone -- it must NOT wait for the orchestrator's pending branch work to
        drain (that paused work IS the handoff payload).
        """
        timed_out = False
        try:
            allows_pending_branch_handoff = (
                getattr(
                    strategy,
                    "allows_pending_branch_handoff_after_sending_complete",
                    False,
                )
                is True
                and self._lifecycle.is_sending_complete
            )
            all_wire_requests_returned = (
                self._progress.in_flight == 0
                if allows_pending_branch_handoff
                else self._progress.check_all_returned_or_cancelled()
            )
            if all_wire_requests_returned and (
                allows_pending_branch_handoff
                or self._branch_orchestrator is None
                or not self._branch_orchestrator.has_pending_branch_work()
            ):
                self.info(
                    "All credits already returned. Setting all_credits_returned_event."
                )
                self._progress.all_credits_returned_event.set()
                return

            if allows_pending_branch_handoff:
                await self._wait_for_accelerated_warmup_handoff()
                return

            timeout = self._lifecycle.time_left_in_seconds(include_grace_period=True)
            timed_out = await self._wait_for_event_with_timeout(
                name=f"{self._config.phase} phase credits returned",
                event=self._progress.all_credits_returned_event,
                timeout=timeout,
                task_to_cancel=None,
                set_event_on_timeout=False,
            )
            if timed_out:
                stats = self._progress.create_stats(self._lifecycle)
                self.warning(
                    f"Phase {self._config.phase} timed out, cancelling all "
                    f"credits. Stats: sent={stats.requests_sent}, "
                    f"completed={stats.requests_completed}, "
                    f"cancelled={stats.requests_cancelled}, "
                    f"in_flight={stats.in_flight_requests}"
                )
                await self._credit_router.cancel_all_credits()
                stats = self._progress.create_stats(self._lifecycle)
                need = (
                    stats.final_requests_sent
                    - stats.requests_completed
                    - stats.requests_cancelled
                )
                self.info(
                    f"Waiting for all cancelled credits to be returned for "
                    f"phase {self._config.phase}. Need {need} more credits."
                )
                if need <= 0:
                    # Forced-completion path: DAG children are being cancelled
                    # too, so don't defer on pending branch work here.
                    self.info(
                        f"All credits already returned after cancel for phase "
                        f"{self._config.phase}. Skipping drain wait."
                    )
                    self._progress.all_credits_returned_event.set()
                # Wait with timeout to avoid hanging indefinitely
                drain_timeout = Environment.TIMING.CANCEL_DRAIN_TIMEOUT
                try:
                    await asyncio.wait_for(
                        self._progress.all_credits_returned_event.wait(),
                        timeout=drain_timeout,
                    )
                    self.info(
                        f"All cancelled credits returned for phase {self._config.phase}"
                    )
                except TimeoutError:
                    self.error(
                        f"Timeout waiting {drain_timeout}s for cancelled credits to return. "
                        f"Some credits may be stuck. Forcing phase completion."
                    )
                    # Release slots for sessions/requests that will never return.
                    self._release_stuck_slots()

                    if not self._lifecycle.is_complete:
                        self._lifecycle.mark_complete(grace_period_triggered=True)
                        self._progress.freeze_completed_counts()
                    self._progress.all_credits_returned_event.set()
        finally:
            if not self._lifecycle.is_complete:
                self._lifecycle.mark_complete(grace_period_triggered=timed_out)
                self._progress.freeze_completed_counts()
            if phase_id is not None and self._baseline_end_ns is None:
                self._baseline_end_ns = (
                    await self._capture_baseline_boundary_before_completion(
                        phase_id, BaselineKind.END
                    )
                )
            stats = self._create_final_stats()
            self.notice(self._format_phase_complete(stats))
            await self._phase_publisher.publish_progress(stats)
            await self._phase_publisher.publish_phase_complete(
                stats, branch_stats=self._snapshot_branch_stats()
            )

    def _release_stuck_slots(self) -> None:
        """Release concurrency slots for credits that will never return."""
        session_released, prefill_released = (
            self._concurrency_manager.release_stuck_slots(self._phase_key)
        )
        if session_released or prefill_released:
            self.warning(
                f"Released stuck slots for phase {self._config.phase}: "
                f"session={session_released}, prefill={prefill_released}"
            )

    async def _wait_for_event_with_timeout(
        self,
        *,
        name: str,
        event: asyncio.Event,
        timeout: float | None,
        task_to_cancel: asyncio.Task | None,
        set_event_on_timeout: bool = False,
    ) -> bool:
        """Wait for event with optional timeout.

        Args:
            name: The name of the event to wait for.
            event: The event to wait for.
            timeout: The timeout in seconds.
                If None, the event will be waited for indefinitely.
                If timeout is <= 0, returns immediately with timeout.
            task_to_cancel: The optional task to cancel when the timeout occurs.
            set_event_on_timeout: If True, the event will also be set when the timeout occurs.

        Returns:
            True if the event timed out, False if the event was set before timeout.
        """
        if timeout is None:
            self.debug(lambda: f"Waiting for event '{name}' indefinitely")
            await event.wait()
            return False

        def _on_timeout() -> bool:
            self.info(f"Timeout of {timeout}s elapsed for event '{name}'")
            if set_event_on_timeout:
                event.set()
            if task_to_cancel:
                task_to_cancel.cancel()
            return True

        if timeout <= 0:
            self.debug(lambda: f"Timeout already elapsed for event '{name}'")
            return _on_timeout()

        try:
            self.info(f"Waiting for event '{name}' with timeout of {timeout}s")
            await asyncio.wait_for(event.wait(), timeout=timeout)
            self.debug(lambda: f"Event '{name}' set before timeout of {timeout}s")
            return False

        except TimeoutError:
            return _on_timeout()

        except Exception as e:
            self.error(f"Error waiting for event '{name}' with timeout: {e!r}")
            raise

    async def _progress_report_loop(self) -> None:
        """Publish phase progress stats at regular intervals.

        Runs as a background task until the phase is complete.
        Publishes progress at CREDIT_PROGRESS_REPORT_INTERVAL intervals.
        During warmup, also emits a throttled INFO heartbeat so headless runs
        remain observable when no interactive UI consumes progress messages.
        """
        self.debug(f"Starting progress reporting loop for phase {self._config.phase}")
        warmup_log_interval = Environment.SERVICE.WARMUP_PROGRESS_LOG_INTERVAL
        next_warmup_log_at = time.monotonic() + warmup_log_interval
        try:
            while True:
                try:
                    stats = self._progress.create_stats(self._lifecycle)
                    await self._phase_publisher.publish_progress(stats)
                    now = time.monotonic()
                    if (
                        self._config.phase == CreditPhase.WARMUP
                        and warmup_log_interval > 0
                        and now >= next_warmup_log_at
                    ):
                        self.info(lambda s=stats: self._format_warmup_progress(s))
                        next_warmup_log_at = now + warmup_log_interval
                except Exception as e:
                    self.error(
                        f"Error publishing progress for phase {self._config.phase}: {e!r}"
                    )
                await asyncio.sleep(Environment.SERVICE.CREDIT_PROGRESS_REPORT_INTERVAL)
        except asyncio.CancelledError:
            self.debug(
                f"Progress reporting loop cancelled for phase {self._config.phase}"
            )
            raise
