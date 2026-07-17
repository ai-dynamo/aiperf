# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.enums import CommandType, MessageType
from aiperf.common.environment import Environment
from aiperf.common.event_loop_monitor import EventLoopMonitor
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.hooks import (
    on_command,
    on_message,
    on_stop,
)
from aiperf.common.messages import (
    CommandMessage,
    DatasetConfigurationFailedNotification,
    DatasetConfiguredNotification,
    ProfileCancelCommand,
    ProfileConfigureCommand,
)
from aiperf.common.models import DatasetMetadata
from aiperf.common.models.dataset_models import GraphSegmentClientMetadata
from aiperf.credit.sticky_router import StickyCreditRouter
from aiperf.timing.config import TimingConfig
from aiperf.timing.phase.publisher import PhasePublisher
from aiperf.timing.phase_orchestrator import PhaseOrchestrator

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.dataset.graph.models import ParsedGraph

# Module logger for run-once configure-time advisories (a stable logger name the
# tests capture via caplog; the service's per-instance ``self.warning`` logger
# name is derived from the runtime service id and is not test-stable).
_logger = AIPerfLogger(__name__)


class TimingManager(BaseComponentService):
    """Service orchestrating credit issuance and request timing.

    Central Service for the credit system. Creates a PhaseOrchestrator
    which internally instantiates the appropriate TimingMode based on mode
    (REQUEST_RATE, FIXED_SCHEDULE, USER_CENTRIC_RATE, ADAPTIVE_SCALE, or
    GRAPH_IR).

    Handles commands: PROFILE_CONFIGURE (create orchestrator),
                      PROFILE_START (begin credit issuance),
                      PROFILE_CANCEL (cancel gracefully).
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )
        self.debug("Timing manager __init__")
        self.config = TimingConfig.from_run(self.run)

        self.phase_publisher = PhasePublisher(
            pub_client=self.pub_client,
            service_id=self.service_id,
        )

        self._dataset_configured_event = asyncio.Event()
        self._dataset_failed_event = asyncio.Event()
        self._dataset_failure_reason: str | None = None
        self._dataset_metadata: DatasetMetadata | None = None
        self._graph_client_metadata: GraphSegmentClientMetadata | None = None

        # StickyCreditRouter handles everything: routing, sending, returns,
        # worker lifecycle. Created early to handle worker connections
        # immediately, as well as attaching to the lifecycle.
        self.sticky_router: StickyCreditRouter = StickyCreditRouter(
            run=run,
            service_id=self.service_id,
        )
        self.attach_child_lifecycle(self.sticky_router)
        self.event_loop_monitor = EventLoopMonitor(self.service_id)

        self._phase_orchestrator: PhaseOrchestrator | None = None

    @on_message(MessageType.DATASET_CONFIGURED_NOTIFICATION)
    async def _on_dataset_configured_notification(
        self, message: DatasetConfiguredNotification
    ) -> None:
        """Store dataset metadata and signal configuration ready."""
        self.debug(
            lambda: (
                f"Received dataset configured notification: "
                f"{len(message.metadata.conversations)} conversations, "
                f"{message.metadata.sampling_strategy.value} sampling strategy"
            )
        )

        self._dataset_metadata = message.metadata
        self._graph_client_metadata = (
            message.client_metadata
            if isinstance(message.client_metadata, GraphSegmentClientMetadata)
            else None
        )
        self._dataset_configured_event.set()

    @on_message(MessageType.DATASET_CONFIGURATION_FAILED)
    async def _on_dataset_configuration_failed(
        self, message: DatasetConfigurationFailedNotification
    ) -> None:
        """Abort the dataset-config wait when DatasetManager reports a failure.

        Without this, _profile_configure_command would block on
        _dataset_configured_event for the full DATASET.CONFIGURATION_TIMEOUT
        (300s default) even though the SystemController has already seen the
        CommandErrorResponse from DatasetManager and is trying to abort.
        """
        self.error(
            f"Received dataset configuration failed notification from "
            f"{message.service_id}: {message.error}"
        )
        self._dataset_failure_reason = message.error
        self._dataset_failed_event.set()

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Create and configure phase orchestrator."""
        self.info("Waiting for dataset to be configured before configuring timing")
        await self._wait_for_dataset_or_failure()

        if self._dataset_failed_event.is_set():
            raise InvalidStateError(
                f"Dataset configuration failed: {self._dataset_failure_reason}"
            )

        if not self._dataset_metadata:
            raise InvalidStateError("Dataset metadata is not available")

        self.debug(f"Configuring phase orchestrator for {self.service_id}")

        # Graph IR workloads need the structural conversation DAG handed to
        # the graph timing strategy. Load the mandatory graph_meta sidecar
        # from the path the graph-typed dataset broadcast advertised (hard
        # fail if unadvertised or unloadable); runs after
        # _wait_for_dataset_or_failure, so the broadcast has been consumed.
        parsed_graph = self._load_graph_sidecar()
        if parsed_graph is not None:
            self._advise_non_streaming_first_token_sources(parsed_graph)

        # Create orchestrator that executes phases
        self._phase_orchestrator = PhaseOrchestrator(
            config=self.config,
            phase_publisher=self.phase_publisher,
            credit_router=self.sticky_router,
            dataset_metadata=self._dataset_metadata,
            parsed_graph=parsed_graph,
        )
        await self._phase_orchestrator.initialize()

    def _load_graph_sidecar(self) -> ParsedGraph | None:
        """Load the mandatory structural sidecar iff this run is a graph workload.

        The DatasetManager writes ``graph_meta.msgpack`` on EVERY graph build
        route and advertises its exact path on the graph-typed
        ``DatasetConfiguredNotification.client_metadata``; the schedule plane
        ingests the sidecar from that path only. A graph run whose broadcast
        is not graph-typed, or whose advertised file is missing, undecodable,
        or store-divergent, is a hard configure-time failure. No re-parse
        fallback, no env-convention path re-derivation.
        """
        from aiperf.dataset.graph.codecs import decode_graph_meta_sidecar
        from aiperf.dataset.graph.workload_detect import resolve_graph_workload

        if resolve_graph_workload(self.run) is None:
            return None

        meta = self._graph_client_metadata
        if meta is None:
            raise InvalidStateError(
                "graph workload run, but the DatasetConfiguredNotification "
                "did not carry GraphSegmentClientMetadata: every graph build "
                "must broadcast the graph store and sidecar locations"
            )
        sidecar = meta.sidecar_path
        if not sidecar.exists():
            raise InvalidStateError(
                f"graph_meta sidecar missing at {sidecar} (the path the "
                "DatasetManager advertised): if the DatasetManager runs on a "
                "different host or filesystem, set "
                "AIPERF_DATASET_MMAP_BASE_PATH to a location shared with the "
                "TimingManager."
            )
        try:
            graph, _fp, _version = decode_graph_meta_sidecar(sidecar.read_bytes())
        except Exception as e:
            raise InvalidStateError(
                f"graph_meta sidecar at {sidecar} is unreadable: {e!r}; "
                "rebuild the run so the DatasetManager rewrites it"
            ) from e
        if not self._sidecar_passes_index_check(graph, sidecar):
            raise InvalidStateError(
                f"graph_meta sidecar at {sidecar} failed the unified-store "
                "index cross-check: the sidecar topology diverged from the "
                "stored envelopes the workers read"
            )
        self.info(f"Loaded structural graph sidecar for timing: {sidecar}")
        return graph

    def _advise_non_streaming_first_token_sources(
        self, parsed_graph: ParsedGraph
    ) -> None:
        """Warn once if a post-TTFT edge's SOURCE node does not stream.

        A first-token-anchored ``StaticEdge``
        (``delay_after_predecessor_first_token_us`` set, post-TTFT anchoring,
        validator rule 55) releases its successor at the SOURCE node's OBSERVED
        first token + the refined delay. That observation only exists when the
        source node itself streams. Each graph node dispatches per its own
        recorded ``streaming`` mode (a per-request override), so the global
        ``--streaming`` flag does not govern whether a first-token event is
        emitted -- a recorded-streaming source streams regardless of the global.
        The residual failure is a first-token edge whose source ``LlmNode``
        carries ``streaming=False``: it emits no SSE first-token event, so its
        post-TTFT-anchored children SILENTLY fall back to waiting on the
        source's COMPLETION latch (dispatch-time delay) instead of its observed
        first token -- a replay-fidelity degradation with no other signal. This
        is possible only in hand-authored/degenerate graphs; recorded corpora
        are consistent by construction (the same recorded ttft drives both the
        edge refinement and the source node's streaming mode). Runs once per run
        (``_profile_configure_command`` fires once) and is a no-op when every
        first-token source streams (or the corpus carries no first-token edges).
        """
        from aiperf.dataset.graph.models import LlmNode
        from aiperf.timing.strategies.graph_ir_replay import first_token_sources

        graphs = [
            parsed_graph.graph,
            *parsed_graph.graphs.values(),
        ]
        for graph in graphs:
            if graph is None:
                continue
            for source_id in first_token_sources(graph):
                source = graph.nodes.get(source_id)
                if isinstance(source, LlmNode) and source.streaming is False:
                    _logger.warning(
                        "graph corpus contains first-token-anchored edges "
                        "(post-TTFT anchoring, validator rule 55) whose SOURCE "
                        f"node ({source_id!r}) has streaming=False: a "
                        "non-streaming source emits no SSE first-token event, so "
                        "its post-TTFT-anchored children silently wait on the "
                        "source's COMPLETION latch (dispatch-time delay) instead "
                        "of its observed first token -- a replay-fidelity "
                        "degradation. This is only possible in "
                        "hand-authored/degenerate graphs; recorded corpora are "
                        "consistent by construction. Set streaming=True on the "
                        "source node to restore post-TTFT anchoring."
                    )
                    return

    def _sidecar_passes_index_check(self, graph: ParsedGraph, sidecar: Path) -> bool:
        """Return True when the index cross-check passes or is not reachable.

        Cross-checks the sidecar's per-trace ordinals against the unified
        store's manifest index (the STORE the worker will actually read). Any
        open/read failure -- including a missing store -- is treated as "not
        reachable" so the sidecar is accepted as-is (best-effort, additive).

        The store location comes from the broadcast
        ``GraphSegmentClientMetadata`` typed fields (``store_base_path`` /
        ``benchmark_id``), the same source the workers open, not from
        string-parsing the ``sidecar`` path.
        """
        from aiperf.dataset.graph.graph_meta_sidecar import sidecar_matches_index

        def _matches(offsets: dict) -> bool:
            return sidecar_matches_index(graph, offsets)

        meta = self._graph_client_metadata
        if meta is None:  # no graph broadcast -> store not reachable
            return True
        base_path = meta.store_base_path
        benchmark_id = meta.benchmark_id

        # The unified store is the sole store shape; absent means not reachable.
        try:
            from aiperf.dataset.graph_segment_unified_store import (
                GraphSegmentUnifiedClient,
                _unified_dir,
            )

            if not _unified_dir(base_path, benchmark_id).exists():
                return True
            client = GraphSegmentUnifiedClient(
                base_path=base_path, benchmark_id=benchmark_id
            ).open()
            try:
                # inner keys are "<ordinal>:<variant>" (see _encode_inner_key);
                # sidecar_matches_index wants {(ordinal, variant): _} per trace.
                offsets: dict[str, dict[tuple[int, str], object]] = {}
                for trace_id, inner in client._node_offsets.items():
                    decoded: dict[tuple[int, str], object] = {}
                    for key, value in inner.items():
                        ord_str, _, variant = key.partition(":")
                        decoded[(int(ord_str), variant)] = value
                    offsets[trace_id] = decoded
            finally:
                client.close()
            return _matches(offsets)
        except Exception:  # any failure -> treat as not reachable
            return True

    async def _wait_for_dataset_or_failure(self) -> None:
        """Wait for either the dataset-configured or dataset-failed event.

        Returns as soon as either event fires. Raises asyncio.TimeoutError
        on the existing 300s envelope (preserving prior behavior for the
        case where neither event ever arrives).
        """
        configured_task = asyncio.create_task(self._dataset_configured_event.wait())
        failed_task = asyncio.create_task(self._dataset_failed_event.wait())
        try:
            done, _ = await asyncio.wait(
                {configured_task, failed_task},
                timeout=Environment.DATASET.CONFIGURATION_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                raise TimeoutError(
                    f"timed out waiting for dataset configuration after "
                    f"{Environment.DATASET.CONFIGURATION_TIMEOUT}s; check "
                    f"dataset-manager logs and consider raising "
                    f"AIPERF_DATASET_CONFIGURATION_TIMEOUT"
                )
        finally:
            for task in (configured_task, failed_task):
                if not task.done():
                    task.cancel()

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, _message: CommandMessage) -> None:
        """Start credit issuance.

        GC is already disabled for this process by the bootstrap path
        (``service_metadata.disable_gc=True`` for TimingManager); see
        ``aiperf.bootstrap``.
        """
        if not self._phase_orchestrator:
            raise InvalidStateError("No phase orchestrator configured")

        # Start event loop health monitoring only during the benchmark
        self.event_loop_monitor.start()

        self.debug("Starting profiling")
        task = self.execute_async(self._phase_orchestrator.start())
        task.add_done_callback(self._on_phase_orchestrator_done)

    def _on_phase_orchestrator_done(self, task: asyncio.Task) -> None:
        """Surface phase-orchestrator failures to the SystemController.

        ``execute_async`` is fire-and-forget, so a phase setup error (e.g.
        FixedScheduleStrategy rejecting an orphaned conversation with no
        first-turn timestamp) is otherwise stored on the task and never
        observed by the parent service. Without this hook, the run finishes
        with zero records but a clean ``os._exit(0)``, masking real bugs.
        Publish a ``BaseServiceErrorMessage`` so the SystemController can
        record it in its exit-error list and exit non-zero.

        Note: the orchestrator's ``_fail`` path raises ``CancelledError``
        after recording the original exception in the orchestrator's
        ``_exit_errors``. We therefore consult the orchestrator state
        rather than ``task.exception()`` (which is ``None`` for cancelled
        tasks) to decide whether the run actually failed.
        """
        from aiperf.common.enums import LifecycleState

        orchestrator = self._phase_orchestrator
        # task.exception() raises if the task was cancelled — guard with
        # cancelled() first. A bare CancelledError that wasn't preceded by
        # a real failure (e.g. user Ctrl+C) leaves the orchestrator in
        # STOPPED, not FAILED, and we shouldn't escalate that.
        if not task.cancelled():
            exc = task.exception()
            if exc is not None and not isinstance(exc, asyncio.CancelledError):
                self._publish_phase_failure(exc)
                return

        if orchestrator is not None and orchestrator.state == LifecycleState.FAILED:
            inner = orchestrator._exit_errors[0] if orchestrator._exit_errors else None
            err_details = inner.error_details if inner is not None else None
            self._publish_phase_failure_from_details(err_details)

    def _publish_phase_failure(self, exc: BaseException) -> None:
        from aiperf.common.messages import BaseServiceErrorMessage
        from aiperf.common.models.error_models import ErrorDetails

        self.error(f"Phase orchestrator failed: {exc!r}")
        self._publish_service_error_safely(
            BaseServiceErrorMessage(
                service_id=self.service_id,
                error=ErrorDetails.from_exception(exc),
            )
        )

    def _publish_phase_failure_from_details(self, details) -> None:
        from aiperf.common.messages import BaseServiceErrorMessage
        from aiperf.common.models.error_models import ErrorDetails

        self.error(f"Phase orchestrator entered FAILED state: {details}")
        self._publish_service_error_safely(
            BaseServiceErrorMessage(
                service_id=self.service_id,
                error=details
                or ErrorDetails(message="Phase orchestrator entered FAILED state"),
            )
        )

    def _publish_service_error_safely(self, message) -> None:
        try:
            self.execute_async(self.publish(message))
        except Exception as publish_error:
            self.debug(
                lambda e=publish_error: (
                    f"Failed to publish BaseServiceErrorMessage from phase failure "
                    f"(comms may already be down): {e!r}"
                )
            )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Cancel credit issuance gracefully.

        Stops new credits and cancels in-flight requests.
        """
        self.warning(f"Received profile cancel command: {message}")
        if self._phase_orchestrator:
            await self._phase_orchestrator.cancel()
            self.info("Phase orchestrator cancelled")

    @on_stop
    async def _timing_manager_stop(self) -> None:
        """Stop the timing manager."""
        self.debug("Stopping timing manager")

        if self._phase_orchestrator:
            await self._phase_orchestrator.stop()

        self.event_loop_monitor.stop()


def main() -> None:
    """Main entry point for the timing manager."""
    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.TIMING_MANAGER)


if __name__ == "__main__":
    main()
