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

from collections.abc import Callable
from typing import TYPE_CHECKING

from aiperf.common.hooks import on_init, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType
from aiperf.timing.concurrency import ConcurrencyManager
from aiperf.timing.conversation_source import ConversationSource
from aiperf.timing.graph_channel import GraphPhaseChannel
from aiperf.timing.phase.runner import PhaseRunner
from aiperf.timing.request_cancellation import RequestCancellationSimulator
from aiperf.timing.url_samplers import URLSelectionStrategyProtocol

if TYPE_CHECKING:
    from aiperf.common.models import DatasetMetadata
    from aiperf.credit.sticky_router import CreditRouterProtocol
    from aiperf.dataset.graph.models import ParsedGraph
    from aiperf.dataset.protocols import DatasetSamplingStrategyProtocol
    from aiperf.timing.config import TimingConfig
    from aiperf.timing.phase.publisher import PhasePublisher


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
        parsed_graph: ParsedGraph | None = None,
        **kwargs,
    ) -> None:
        """Initialize timing strategy and orchestration components.

        Args:
            config: Timing configuration (phases, limits, etc.)
            phase_publisher: Publishes phase events to message bus
            credit_router: Routes credits to workers
            dataset_metadata: Dataset for conversation sampling
            parsed_graph: For graph IR workloads, the parsed conversation DAG.
                When set, the orchestrator owns a ``GraphPhaseChannel`` carrying
                it (and the WARMUP handoff) to every ``PhaseRunner``; the
                conversation-shaped sampler and source are NOT built because
                graph runs sample inside the graph strategy over
                ``parsed_graph.traces``. ``None`` for non-graph workloads.
        """
        super().__init__(**kwargs)
        self._config = config
        self._phase_publisher = phase_publisher
        self._credit_router = credit_router
        self._dataset_metadata = dataset_metadata

        # Long-lived components (shared across phases)
        if parsed_graph is None:
            self._dataset_sampler: DatasetSamplingStrategyProtocol | None = (
                self._build_dataset_sampler()
            )
            self._conversation_source: ConversationSource | None = (
                self._build_conversation_source()
            )
            self._graph_channel: GraphPhaseChannel | None = None
        else:
            # Graph runs sample inside the graph strategy
            # (GraphIRConversationSource over parsed_graph.traces); the
            # conversation-shaped sampler and source have nothing to sample
            # (conversations is empty) and are not built.
            self._dataset_sampler = None
            self._conversation_source = None
            self._graph_channel = GraphPhaseChannel(parsed_graph=parsed_graph)
        self._concurrency_manager = ConcurrencyManager()
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
        self._callback_handler = CreditCallbackHandler(self._concurrency_manager)
        self._credit_router.set_return_callback(self._callback_handler.on_credit_return)
        self._credit_router.set_first_token_callback(
            self._callback_handler.on_first_token
        )

        # Phase configuration
        self._ordered_phase_configs = config.phase_configs

        # Active phase runners (for cancellation) - multiple possible with seamless mode
        self._active_runners: list[PhaseRunner] = []

    def _build_dataset_sampler(self) -> DatasetSamplingStrategyProtocol:
        """Construct the root-conversation sampler for this run.

        Only root conversations are sampled by the strategy. DAG children belong
        to their root's session and are dispatched by the BranchOrchestrator on
        credit return -- sampling them as roots would create duplicate root
        sessions. Filter on ``is_root`` rather than ``agent_depth == 0`` so
        SPAWN-mode children (which keep ``agent_depth == 0`` for fresh-context
        semantics but carry ``is_root=False``) are also excluded.
        """
        SamplerClass = plugins.get_class(
            PluginType.DATASET_SAMPLER,
            self._dataset_metadata.sampling_strategy,
        )
        root_conv_ids = [
            c.conversation_id
            for c in self._dataset_metadata.conversations
            if getattr(c, "is_root", True)
        ]
        return SamplerClass(
            conversation_ids=root_conv_ids
            or [c.conversation_id for c in self._dataset_metadata.conversations]
        )

    def _build_conversation_source(self) -> ConversationSource:
        """Construct the long-lived conversation source (non-graph runs only)."""
        return ConversationSource(self._dataset_metadata, self._dataset_sampler)

    @property
    def conversation_source(self) -> ConversationSource | None:
        """Conversation source for dataset access (``None`` for graph runs)."""
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
            With seamless=True, a phase can start before the previous phase
            completes waiting for returns. This allows smooth phase transitions
            without gaps in request issuance. Multiple runners may be active
            simultaneously (old phase waiting for returns while new phase sends).
        """
        for i, phase_config in enumerate(self._ordered_phase_configs):
            is_final_phase = i == len(self._ordered_phase_configs) - 1
            is_seamless_non_final = phase_config.seamless and not is_final_phase

            runner = PhaseRunner(
                config=phase_config,
                conversation_source=self._conversation_source,
                phase_publisher=self._phase_publisher,
                credit_router=self._credit_router,
                concurrency_manager=self._concurrency_manager,
                cancellation_policy=self._cancellation_policy,
                callback_handler=self._callback_handler,
                url_selection_strategy=self._url_sampler,
                graph_channel=self._graph_channel,
            )

            # For seamless non-final phases, set callback to remove from active runners
            # when background return wait completes
            if is_seamless_non_final:
                runner.set_phase_complete_callback(
                    self._phase_runner_cleanup_callback(runner)
                )

            # Track active runner (multiple possible with seamless mode)
            self._active_runners.append(runner)

            try:
                # Execute phase (runner.run() returns after sending complete for seamless,
                # or after all returns complete for non-seamless/final phases)
                await runner.run(is_final_phase=is_final_phase)
            except Exception as e:
                self.error(f"Error executing phase {runner.phase}: {e!r}")
                await self.cancel()
                raise e

            # Remove from active runners when fully complete.
            # For seamless phases, this happens after returns complete (background
            # task). A concurrent graceful cancel (PROFILE_CANCEL / Ctrl+C ->
            # ``cancel`` -> ``_cancel_active_runners``) clears the whole list while
            # this coroutine is parked in ``runner.run`` (e.g. a graph phase parked
            # mid idle-gap replay), so the runner may already be gone here.
            # Discard rather than ``remove`` so the teardown does not raise
            # ``ValueError: list.remove(x): x not in list`` -- which would fail the
            # orchestrator and abort the records-manager's in-flight graceful export
            # (no metrics JSON, ``was_cancelled`` never set).
            if not is_seamless_non_final and runner in self._active_runners:
                self._active_runners.remove(runner)

    def _phase_runner_cleanup_callback(self, runner: PhaseRunner) -> Callable[[], None]:
        """Create callback that removes runner from active list when phase completes."""

        def cleanup() -> None:
            if runner in self._active_runners:
                self._active_runners.remove(runner)
                self.debug(f"Removed completed runner for phase {runner.phase}")

        return cleanup

    async def cancel(self) -> None:
        """Cancel the orchestrator gracefully.

        Stops issuing new credits and cancels in-flight requests.
        Called when user requests cancellation (e.g., Ctrl+C).
        """
        self.warning("Cancelling phase orchestrator")

        # Cancel all in-flight credits first
        await self._credit_router.cancel_all_credits()

        self._cancel_active_runners()

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

    def _cancel_active_runners(self) -> None:
        """Cancel every tracked phase runner and clear the active list."""
        for runner in self._active_runners:
            runner.cancel()
            self.debug(f"Cancelled active phase runner for phase {runner.phase}")
        self._active_runners.clear()
