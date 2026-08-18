# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase orchestrator for credit phase execution.

The orchestrator handles all orchestration concerns:
- Lifecycle management (init, start, stop)
- Phase execution loop (creates PhaseRunner per phase)
- Cancellation

The actual timing logic is delegated to a pluggable TimingMode (created per-phase).
Credit callbacks are handled by CreditCallbackHandler (registered directly with router).
Progress reporting is delegated to PhaseRunner.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from aiperf.common.control_hooks import (
    PreparedEndpointControlHooks,
    start_server_profiler,
    stop_server_profiler,
)
from aiperf.common.control_plane_http import ControlPlaneHttpError
from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.hooks import on_init, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, TimingMode
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.request_cancellation import RequestCancellationSimulator
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.trajectory_source import TrajectorySource
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.common.models import DatasetMetadata
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.timing.config import TimingConfig
    from aiperf.timing.phase.publisher import PhasePublisher


async def _stop_server_profiler_warn_only(
    hooks: PreparedEndpointControlHooks,
    headers: dict[str, str],
    stop_fn: Callable[..., Awaitable[None]],
    warn_fn: Callable[[str], None],
) -> None:
    """Stop the server profiler without masking the primary phase outcome."""
    try:
        await stop_fn(hooks, headers)
    except ControlPlaneHttpError as error:
        warn_fn(f"server_profiler stop failed: {error}")


async def run_phase_with_server_profiler(
    *,
    phase: CreditPhase,
    hooks: PreparedEndpointControlHooks | None,
    headers: dict[str, str],
    run_phase: Callable[[], Awaitable[None]],
    start_fn: Callable[..., Awaitable[None]],
    stop_fn: Callable[..., Awaitable[None]],
    warn_fn: Callable[[str], None],
    defer_stop: bool = False,
) -> bool:
    """Start/stop server profiler around a profiling phase only.

    When ``defer_stop`` is True (seamless non-final profiling), start runs
    before ``run_phase`` but stop is left to the caller after drain. Returns
    True when a deferred stop is still owed; False otherwise.
    """
    enabled = (
        phase == CreditPhase.PROFILING
        and hooks is not None
        and bool(hooks.profiler_start_urls)
    )
    if not enabled:
        await run_phase()
        return False

    await start_fn(hooks, headers)
    if defer_stop:
        try:
            await run_phase()
        except asyncio.CancelledError:
            await _stop_server_profiler_warn_only(hooks, headers, stop_fn, warn_fn)
            raise
        except Exception:
            await _stop_server_profiler_warn_only(hooks, headers, stop_fn, warn_fn)
            raise
        return True

    try:
        await run_phase()
    finally:
        await _stop_server_profiler_warn_only(hooks, headers, stop_fn, warn_fn)
    return False


class PhaseOrchestrator(AIPerfLifecycleMixin):
    """Orchestrates credit phase execution (warmup → profiling).

    The orchestrator handles:
    - Component composition (ConversationSource, ConcurrencyManager, CancellationPolicy)
    - Lifecycle hooks (@on_init, @on_start)
    - Phase execution loop (creates PhaseRunner per phase)
    - Cancellation

    The orchestrator does NOT handle:
    - Credit callbacks (handled by CreditCallbackHandler, registered directly with router)
    - Per-phase lifecycle (handled by PhaseRunner)

    The TimingMode (created per-phase by PhaseRunner) handles:
    - Timing logic (execute_phase)
    - Dispatching subsequent turns on credit return (handle_credit_return)

    ```
    Architecture (Simplified)
    =========================

    TimingManager
        └── PhaseOrchestrator
                │
                │ owns (long-lived, shared across phases):
                ├── ConcurrencyManager
                ├── CancellationPolicy
                ├── ConversationSource
                └── CreditCallbackHandler ──► registered with CreditRouter
                │
                │ creates per phase:
                └── PhaseRunner ──► TimingMode
                        │
                        ├── LoopScheduler (SINGLE owner)
                        ├── PhaseLifecycle
                        ├── PhaseProgressTracker ──► CreditCounter
                        ├── StopConditionChecker
                        └── CreditIssuer

    Callback Flow (direct, no orchestrator in middle):
        Worker ──► CreditRouter ──► CreditCallbackHandler ──► [count, release slots, dispatch]
    ```
    """

    def __init__(
        self,
        *,
        config: TimingConfig,
        phase_publisher: PhasePublisher,
        credit_router: CreditRouterProtocol,
        dataset_metadata: DatasetMetadata,
        control_hooks: PreparedEndpointControlHooks | None = None,
        control_headers: dict[str, str] | None = None,
        run: BenchmarkRun | None = None,
        **kwargs,
    ) -> None:
        """Initialize timing strategy and orchestration components.

        Args:
            config: Timing configuration (phases, limits, etc.)
            phase_publisher: Publishes phase events to message bus
            credit_router: Routes credits to workers
            dataset_metadata: Dataset for conversation sampling
            control_hooks: Prepared endpoint control hooks (profiler URLs);
                owned by TimingManager, never by workers
            control_headers: Auth headers for control-plane POSTs
            run: Full ``BenchmarkRun`` for strategies that need it (e.g.
                AgenticReplayStrategy reads ``run.cfg.get_cache_bust_target()``
                and ``run.benchmark_id``). Optional; strategies that don't need
                it ignore the value.
        """
        super().__init__(**kwargs)
        self._config = config
        self._phase_publisher = phase_publisher
        self._credit_router = credit_router
        self._dataset_metadata = dataset_metadata
        self._control_hooks = control_hooks
        self._control_headers = control_headers or {}
        self._run = run

        # Create dataset sampler
        SamplerClass = plugins.get_class(
            PluginType.DATASET_SAMPLER,
            self._dataset_metadata.sampling_strategy,
        )
        # Only root conversations are sampled by the strategy. DAG
        # children belong to their root's session and are dispatched by
        # the BranchOrchestrator on credit return — sampling them as
        # roots would create duplicate root sessions. Filter on
        # ``is_root`` rather than ``agent_depth == 0`` so SPAWN-mode
        # children (which keep ``agent_depth == 0`` for fresh-context
        # semantics but carry ``is_root=False``) are also excluded.
        root_conv_ids = [
            c.conversation_id
            for c in self._dataset_metadata.conversations
            if getattr(c, "is_root", True)
        ]
        self._dataset_sampler = SamplerClass(
            conversation_ids=root_conv_ids
            or [c.conversation_id for c in self._dataset_metadata.conversations]
        )

        # Long-lived components (shared across phases).
        # AGENTIC_REPLAY needs trajectories built once at orchestrator-
        # construction time so trajectory state survives the WARMUP ->
        # PROFILING boundary; it also needs a per-session-TREE slot ledger
        # (a session = root + every subagent it spawns; the slot is held
        # until the whole tree drains, giving exactly-N concurrency). Other
        # timing modes (including dag_jsonl) keep the plain ConversationSource
        # + legacy per-root-credit slot release (registry stays None).
        is_agentic_replay = any(
            pc.timing_mode == TimingMode.AGENTIC_REPLAY for pc in config.phase_configs
        )
        benchmark_id = run.benchmark_id if run is not None else "unknown"
        cache_bust_target = (
            run.cfg.get_cache_bust_target() if run is not None else CacheBustTarget.NONE
        )
        if is_agentic_replay:
            if config.concurrency is None:
                raise ValueError(
                    "AGENTIC_REPLAY timing mode requires concurrency to be set on "
                    "TimingConfig (sourced from the profiling phase concurrency)."
                )
            profiling = next(
                (
                    pc
                    for pc in config.phase_configs
                    if pc.phase == CreditPhase.PROFILING
                ),
                None,
            )
            self._conversation_source = TrajectorySource(
                dataset_metadata=self._dataset_metadata,
                dataset_sampler=self._dataset_sampler,
                concurrency=config.concurrency,
                random_seed=config.random_seed if config.random_seed is not None else 0,
                start_min_ratio=config.trajectory_start_min_ratio,
                start_max_ratio=config.trajectory_start_max_ratio,
                allow_dataset_wrap=config.allow_dataset_wrap,
                cache_bust_enabled=config.cache_bust_enabled,
                expected_num_sessions=(
                    profiling.expected_num_sessions if profiling is not None else None
                ),
                total_expected_requests=(
                    profiling.total_expected_requests if profiling is not None else None
                ),
                expected_duration_sec=(
                    profiling.expected_duration_sec if profiling is not None else None
                ),
            )
        else:
            self._conversation_source = ConversationSource(
                self._dataset_metadata,
                self._dataset_sampler,
                benchmark_id=benchmark_id,
                cache_bust_target=cache_bust_target,
            )
        self._concurrency_manager = ConcurrencyManager()
        self._session_tree_registry = (
            SessionTreeRegistry(self._concurrency_manager)
            if is_agentic_replay
            else None
        )
        self._cancellation_policy = RequestCancellationSimulator(
            config.request_cancellation
        )

        # URL sampler for multi-URL load balancing (None if single URL)
        self._url_sampler: URLSelectionStrategyProtocol | None = None
        if len(config.urls) > 1:
            StrategyClass = plugins.get_class(
                PluginType.URL_SELECTION_STRATEGY, config.url_selection_strategy
            )
            self._url_sampler = StrategyClass(urls=config.urls)

        # Callback handler registered directly with router (no orchestrator in middle)
        self._callback_handler = CreditCallbackHandler(
            self._concurrency_manager,
            session_tree_registry=self._session_tree_registry,
            on_warmup_abort=self._phase_publisher.publish_profile_cancel,
        )
        self._credit_router.set_return_callback(self._callback_handler.on_credit_return)
        self._credit_router.set_first_token_callback(
            self._callback_handler.on_first_token
        )
        self._credit_router.set_fatal_error_callback(self._record_control_fatal_error)

        # Phase configuration
        self._ordered_phase_configs = config.phase_configs

        # Active phase runners (for cancellation) - multiple possible with seamless mode
        self._active_runners: list[PhaseRunner] = []
        self._server_profiler_owners: set[PhaseRunner] = set()
        # In-flight deferred profiler stops spawned from phase-complete
        # callbacks; awaited by _execute_phases before the run reports success.
        self._deferred_profiler_stops: set[asyncio.Task] = set()
        # First fatal failure surfaced by a seamless phase's detached return-wait
        # task; re-raised by _execute_phases so the run fails (not reports success).
        self._seamless_phase_error: BaseException | None = None

    def _record_control_fatal_error(self, error: BaseException) -> None:
        """Record a fatal request-free control-node failure.

        Seamless mode can keep several phase runners active at once, so pinning
        the error to whichever phase currently owns the callback handler's
        mutable ``progress`` slot could record it on the wrong phase -- or drop
        it between phases. A control-node failure is fatal to the whole run, so
        record it on every active phase's tracker (each runner surfaces it on
        its own exit path). Fall back to the callback handler's current progress
        only when no runner is active (a failure arriving between phases).
        """
        recorded = False
        for runner in list(self._active_runners):
            runner.record_control_fatal_error(error)
            recorded = True
        if not recorded:
            progress = getattr(self._callback_handler, "progress", None)
            if progress is not None:
                progress.record_fatal_error(error)

    def _on_seamless_phase_error(self, error: BaseException) -> None:
        """Capture a fatal failure surfaced by a seamless phase's detached
        return-wait task so ``_execute_phases`` re-raises it. First error wins."""
        if self._seamless_phase_error is None:
            self._seamless_phase_error = error

    @property
    def conversation_source(self) -> ConversationSource:
        """Conversation source for dataset access."""
        return self._conversation_source

    @on_init
    async def _init_orchestrator(self) -> None:
        """Log configured phases (actual initialization happens per-phase in _execute_phases)."""
        self.info(
            lambda: (
                f"Initialized {len(self._ordered_phase_configs)} phase(s): "
                f"{[p.phase.replace('_', ' ').title() for p in self._ordered_phase_configs]}"
            )
        )

    @on_start
    async def _start_orchestrator(self) -> None:
        """Execute all phases and publish completion when done."""
        self.debug(lambda: "Starting PhaseOrchestrator")

        try:
            # Execute all phases sequentially (each PhaseRunner handles its own progress reporting)
            await self._execute_phases()
        finally:
            # Cleanup
            self.notice("All credits completed")
            self._credit_router.mark_credits_complete()
            await self._phase_publisher.publish_credits_complete()

    async def _execute_phases(self) -> None:
        """Execute phases in order (typically: warmup → profiling).

        For each phase:
        1. Create PhaseRunner with conversation_source
        2. Execute phase via runner.run() (runner creates timing strategy internally)
        3. Runner handles setup, execution, and cleanup

        Seamless Mode:
            When the next phase has seamless=True, it can start before the current
            phase completes waiting for returns. This allows smooth phase
            transitions without gaps in request issuance. Multiple runners may be
            active simultaneously (old phase waiting for returns while new phase
            sends).
        """
        for i, phase_config in enumerate(self._ordered_phase_configs):
            is_final_phase = i == len(self._ordered_phase_configs) - 1
            seamless_to_next = (
                not is_final_phase and self._ordered_phase_configs[i + 1].seamless
            )

            runner = PhaseRunner(
                config=phase_config,
                conversation_source=self._conversation_source,
                phase_publisher=self._phase_publisher,
                credit_router=self._credit_router,
                concurrency_manager=self._concurrency_manager,
                cancellation_policy=RequestCancellationSimulator(
                    phase_config.request_cancellation
                ),
                callback_handler=self._callback_handler,
                url_selection_strategy=self._url_sampler,
                run=self._run,
                session_tree_registry=self._session_tree_registry,
            )

            # Seamless non-final profiling: stop after drain (phase-complete
            # callback), not when run() returns at send-complete.
            profiler_will_defer_stop = (
                seamless_to_next
                and phase_config.phase == CreditPhase.PROFILING
                and self._control_hooks is not None
                and bool(self._control_hooks.profiler_start_urls)
            )

            # For seamless non-final phases, set callbacks before run() so a
            # fast drain cannot fire the wrong (cleanup-only) callback. The
            # error callback is set for every seamless non-final runner: the
            # post-loop barrier only inspects runners still in _active_runners,
            # and the complete-callback removes the runner from that list, so a
            # drained phase can only surface a fatal error through this callback.
            if seamless_to_next:
                runner.set_phase_complete_callback(
                    self._phase_runner_cleanup_and_stop_profiler_callback(runner)
                    if profiler_will_defer_stop
                    else self._phase_runner_cleanup_callback(runner)
                )
                runner.set_phase_error_callback(self._on_seamless_phase_error)

            # Track active runner (multiple possible with seamless mode)
            self._active_runners.append(runner)

            async def _run(
                phase_runner: PhaseRunner = runner,
                final: bool = is_final_phase,
                seamless: bool = seamless_to_next,
            ) -> None:
                # Execute phase (runner.run() returns after sending complete for seamless,
                # or after all returns complete for non-seamless/final phases)
                await phase_runner.run(is_final_phase=final, seamless_to_next=seamless)

            async def _start_profiler(
                hooks: PreparedEndpointControlHooks,
                headers: dict[str, str],
                phase_runner: PhaseRunner = runner,
            ) -> None:
                await self._start_server_profiler_for_runner(
                    phase_runner, hooks, headers
                )

            async def _stop_profiler(
                hooks: PreparedEndpointControlHooks,
                headers: dict[str, str],
                phase_runner: PhaseRunner = runner,
            ) -> None:
                await self._stop_server_profiler_for_runner(
                    phase_runner, hooks, headers
                )

            try:
                await run_phase_with_server_profiler(
                    phase=phase_config.phase,
                    hooks=self._control_hooks,
                    headers=self._control_headers,
                    run_phase=_run,
                    start_fn=_start_profiler,
                    stop_fn=_stop_profiler,
                    warn_fn=self.warning,
                    defer_stop=profiler_will_defer_stop,
                )
            except Exception as e:
                self.error(f"Error executing phase {runner.phase}: {e!r}")
                await self.cancel()
                raise e

            # Remove from active runners when fully complete
            # For seamless phases, this happens after returns complete (background task)
            if not seamless_to_next:
                self._active_runners.remove(runner)

        # Barrier: a seamless phase's return-wait runs detached and
        # ``runner.run()`` never awaited it, so await any still outstanding here.
        # This guarantees a late fatal control-node failure has been surfaced
        # before the run is allowed to report success.
        await self._await_outstanding_seamless_waits()
        await self._drain_deferred_profiler_stops()
        if self._seamless_phase_error is not None:
            error = self._seamless_phase_error
            self.error(f"Fatal control-node failure in a seamless phase: {error!r}")
            # cancel() runs _stop_server_profiler_warn_only, so any profiler
            # ownership still outstanding here is released before we raise.
            await self.cancel()
            raise error

    async def _await_outstanding_seamless_waits(self) -> None:
        """Await any detached seamless return-wait tasks still running after the
        phase loop and capture a fatal control-node failure from each.

        Reads each runner's recorded fatal error directly rather than relying on
        the task's done-callback having fired, so the check is deterministic.
        """
        for runner in list(self._active_runners):
            task = runner.return_wait_task
            if task is not None and not task.done():
                try:
                    await task
                except Exception as exc:  # noqa: BLE001 - propagated below
                    if self._seamless_phase_error is None:
                        self._seamless_phase_error = exc
            fatal = runner.control_fatal_error
            if fatal is not None and self._seamless_phase_error is None:
                self._seamless_phase_error = fatal

    async def _drain_deferred_profiler_stops(self) -> None:
        """Ensure every deferred profiler stop has completed before returning.

        Two paths must be covered. A runner whose phase-complete callback has
        already fired left a task in ``_deferred_profiler_stops``; await it. A
        runner whose return-wait finished but whose done-callback has not run
        yet (done-callbacks are scheduled via ``call_soon``, so awaiting the
        return-wait task does not guarantee it fired) still holds ownership;
        stop it directly. ``_stop_server_profiler_for_runner`` is a no-op for a
        runner that no longer owns the profiler, so the two paths cannot issue
        a duplicate stop.
        """
        while True:
            pending = [t for t in self._deferred_profiler_stops if not t.done()]
            if not pending:
                break
            await asyncio.gather(*pending, return_exceptions=True)
        for runner in list(self._server_profiler_owners):
            await self._stop_server_profiler_for_runner_warn_only(runner)

    async def _start_server_profiler_for_runner(
        self,
        runner: PhaseRunner,
        hooks: PreparedEndpointControlHooks,
        headers: dict[str, str],
    ) -> None:
        """Start the profiler when the first profiling runner takes ownership."""
        if not self._server_profiler_owners:
            await start_server_profiler(hooks, headers)
        self._server_profiler_owners.add(runner)

    async def _stop_server_profiler_for_runner(
        self,
        runner: PhaseRunner,
        hooks: PreparedEndpointControlHooks,
        headers: dict[str, str],
    ) -> None:
        """Release one runner's profiler ownership and stop after the last one."""
        if runner not in self._server_profiler_owners:
            return
        self._server_profiler_owners.remove(runner)
        if self._server_profiler_owners:
            return
        await stop_server_profiler(hooks, headers)

    async def _stop_server_profiler_for_runner_warn_only(
        self, runner: PhaseRunner
    ) -> None:
        """Release deferred profiler ownership without failing phase cleanup."""
        if self._control_hooks is None or not self._control_hooks.profiler_stop_urls:
            return
        try:
            await self._stop_server_profiler_for_runner(
                runner, self._control_hooks, self._control_headers
            )
        except ControlPlaneHttpError as error:
            self.warning(f"server_profiler stop failed: {error}")

    def _phase_runner_cleanup_callback(self, runner: PhaseRunner) -> Callable[[], None]:
        """Create callback that removes runner from active list when phase completes."""

        def cleanup() -> None:
            if runner in self._active_runners:
                self._active_runners.remove(runner)
                self.debug(f"Removed completed runner for phase {runner.phase}")

        return cleanup

    def _phase_runner_cleanup_and_stop_profiler_callback(
        self, runner: PhaseRunner
    ) -> Callable[[], None]:
        """Seamless profiling: after drain, remove runner and stop the profiler."""

        cleanup = self._phase_runner_cleanup_callback(runner)

        def cleanup_and_stop() -> None:
            cleanup()
            # Fired from a done-callback, so the stop cannot be awaited here.
            # Track the task so _execute_phases can await it before returning
            # rather than letting the run finish with a stop still in flight.
            task = self.execute_async(
                self._stop_server_profiler_for_runner_warn_only(runner)
            )
            self._deferred_profiler_stops.add(task)
            task.add_done_callback(self._deferred_profiler_stops.discard)

        return cleanup_and_stop

    async def _stop_server_profiler_warn_only(self) -> None:
        """Best-effort stop for any outstanding profiler ownership."""
        if (
            not self._server_profiler_owners
            or self._control_hooks is None
            or not self._control_hooks.profiler_stop_urls
        ):
            return
        self._server_profiler_owners.clear()
        try:
            await stop_server_profiler(self._control_hooks, self._control_headers)
        except ControlPlaneHttpError as error:
            self.warning(f"server_profiler stop failed: {error}")

    async def cancel(self) -> None:
        """Cancel the orchestrator gracefully.

        Stops issuing new credits and cancels in-flight requests.
        Called when user requests cancellation (e.g., Ctrl+C).
        """
        self.warning("Cancelling phase orchestrator")

        # Cancel all in-flight credits first
        await self._credit_router.cancel_all_credits()

        self._cancel_active_runners()
        # If seamless profiling deferred stop, cancel may skip the drain callback.
        await self._stop_server_profiler_warn_only()

    @on_stop
    async def _stop_orchestrator(self) -> None:
        """Clean up orchestrator state on normal stop.

        Cancels any still-active phase runners. Without this hook, runners
        tracked in ``_active_runners`` are leaked on the non-cancellation
        shutdown path (only ``cancel()`` cleaned them up before, and it is
        only called for Ctrl+C).

        Callback registrations on the credit router are not explicitly
        unregistered: the router is a child lifecycle of ``TimingManager``
        and is torn down alongside the orchestrator, so its callback table
        does not outlive us.
        """
        if self._active_runners:
            self.debug(
                lambda: (
                    f"Stopping orchestrator with {len(self._active_runners)} active runner(s)"
                )
            )
            self._cancel_active_runners()
        await self._stop_server_profiler_warn_only()

    def _cancel_active_runners(self) -> None:
        """Cancel every tracked phase runner and clear the active list."""
        for runner in self._active_runners:
            runner.cancel()
            self.debug(f"Cancelled active phase runner for phase {runner.phase}")
        self._active_runners.clear()
