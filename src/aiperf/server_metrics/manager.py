# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.enums import (
    CommandType,
    CreditPhase,
    MessageType,
    make_result_producer_capability,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PluginDisabled, PostProcessorDisabled
from aiperf.common.hooks import on_command, on_message, on_stop
from aiperf.common.messages import (
    PhaseBaselineRequestMessage,
    ProcessServerMetricsResultMessage,
    ProfileCancelCommand,
    ProfileCompleteCommand,
    ProfileConfigureCommand,
    ProfileStartCommand,
    RealtimeServerMetricsMessage,
    ServerMetricsStatusMessage,
)
from aiperf.common.metric_utils import normalize_metrics_endpoint_url
from aiperf.common.mixins import BaselineCollectorMixin
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    ProcessServerMetricsResult,
    ServerMetricsRecord,
)
from aiperf.common.redact import redact_url
from aiperf.common.types import PhaseKind
from aiperf.credit.messages import (
    CreditPhaseCompleteMessage,
    CreditPhaseStartMessage,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import AccumulatorType, PluginType
from aiperf.server_metrics.data_collector import ServerMetricsDataCollector
from aiperf.server_metrics.protocols import ServerMetricsAccumulatorProtocol

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


_SERVER_METRICS_RECORD_TYPE = "server_metrics"


@dataclass(frozen=True, slots=True, eq=False)
class _ServerMetricsPhaseIdentity:
    """Concrete phase identity captured when a scrape starts."""

    phase: CreditPhase
    phase_index: int | None = None
    profiling_index: int | None = None
    phase_name: str | None = None
    phase_kind: PhaseKind | None = None

    def __eq__(self, other: object) -> bool:
        """Compare full identities, while tolerating legacy phase-only checks."""
        if isinstance(other, CreditPhase):
            return self.phase == other
        if not isinstance(other, _ServerMetricsPhaseIdentity):
            return NotImplemented
        return (
            self.phase,
            self.phase_index,
            self.profiling_index,
            self.phase_name,
            self.phase_kind,
        ) == (
            other.phase,
            other.phase_index,
            other.profiling_index,
            other.phase_name,
            other.phase_kind,
        )

    def __hash__(self) -> int:
        """Hash the immutable identity fields."""
        return hash(
            (
                self.phase,
                self.phase_index,
                self.profiling_index,
                self.phase_name,
                self.phase_kind,
            )
        )


@dataclass(slots=True)
class _ErrorState:
    """Local collection and processor errors keyed for final export."""

    error_counts: dict[ErrorDetails, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record(self, error: ErrorDetails) -> None:
        """Increment the count for one error shape."""
        self.error_counts[error] += 1


_SERVER_METRICS_SCRAPE_PHASE: ContextVar[_ServerMetricsPhaseIdentity | None] = (
    ContextVar("server_metrics_scrape_phase", default=None)
)


class ServerMetricsManager(BaselineCollectorMixin, BaseComponentService):
    """Coordinates multiple ServerMetricsDataCollector instances for server metrics collection.

    The ServerMetricsManager coordinates multiple ServerMetricsDataCollector instances
    and owns the raw server-metric pipeline locally. Only bounded realtime summaries
    and one final result cross the message bus.

    This service:
    - Manages lifecycle of ServerMetricsDataCollector instances
    - Collects metrics from multiple Prometheus endpoints
    - Accumulates and writes raw server-metric records without a bus hop
    - Handles errors gracefully with ErrorDetails
    - Follows centralized architecture patterns

    Args:
        run: BenchmarkRun carrying the BenchmarkConfig + per-run state.
        service_id: Optional unique identifier for this service instance
    """

    extra_capabilities: ClassVar[tuple[str, ...]] = (
        make_result_producer_capability("server_metrics"),
    )

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

        self._collectors: dict[str, ServerMetricsDataCollector] = {}
        self._server_metrics_disabled = not self.run.cfg.server_metrics.enabled
        self._processors: list[Any] = []
        self._stream_exporters: list[Any] = []
        self._stream_exporters_finalized = False
        self._accumulator: ServerMetricsAccumulatorProtocol | None = None
        self._error_state = _ErrorState()
        self._result_published = False
        self._profiling_started = False
        self._last_realtime_publish_ns = 0
        self._profile_complete_lock = asyncio.Lock()
        self._load_server_metrics_processors()

        # Collect metrics from all endpoint URLs (for multi-URL load balancing)
        self._server_metrics_endpoints: list[str] = []
        for url in self.run.cfg.endpoint.urls:
            normalized_url = normalize_metrics_endpoint_url(url)
            if normalized_url not in self._server_metrics_endpoints:
                self._server_metrics_endpoints.append(normalized_url)
        self.info(
            f"Server Metrics: Discovered {len(self._server_metrics_endpoints)} "
            f"endpoints: {[redact_url(u) for u in self._server_metrics_endpoints]}"
        )

        # Add user-specified URLs if provided
        user_urls = self.run.cfg.server_metrics.urls
        if user_urls:
            for url in user_urls:
                normalized_url = normalize_metrics_endpoint_url(url)
                if normalized_url not in self._server_metrics_endpoints:
                    self._server_metrics_endpoints.append(normalized_url)

        # Use server metrics collection interval
        self._collection_interval = Environment.SERVER_METRICS.COLLECTION_INTERVAL

        # Task for delayed shutdown, created when no endpoints are reachable
        self._shutdown_task: asyncio.Task[None] | None = None
        self._active_phase: _ServerMetricsPhaseIdentity | None = None
        self._last_profiling_phase: _ServerMetricsPhaseIdentity | None = None

    def _load_server_metrics_processors(self) -> None:
        """Load only accumulator/exporter plugins consuming server metrics."""
        for plugin_type in (PluginType.ACCUMULATOR, PluginType.STREAM_EXPORTER):
            for entry in plugins.iter_entries(plugin_type):
                record_types = (
                    entry.metadata.get("record_types", []) if entry.metadata else []
                )
                if _SERVER_METRICS_RECORD_TYPE not in record_types:
                    continue
                try:
                    processor_class = plugins.get_class(plugin_type, entry.name)
                    processor = processor_class(
                        service_id=self.service_id,
                        run=self.run,
                        pub_client=self.pub_client,
                    )
                    self.attach_child_lifecycle(processor)
                    self._processors.append(processor)
                    if plugin_type == PluginType.STREAM_EXPORTER:
                        self._stream_exporters.append(processor)
                    if (
                        plugin_type == PluginType.ACCUMULATOR
                        and entry.name == AccumulatorType.SERVER_METRICS
                    ):
                        self._accumulator = processor
                    self.debug(
                        f"Created server metrics processor: {entry.name}: "
                        f"{processor.__class__.__name__}"
                    )
                except (PluginDisabled, PostProcessorDisabled):
                    self.debug(
                        f"Server metrics processor {entry.name} is disabled and will not be used"
                    )
                except Exception as exc:  # noqa: BLE001 - plugin extension boundary
                    self.error(
                        f"Failed to create server metrics processor {entry.name}: {exc}"
                    )

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the server metrics collectors but don't start them yet.

        Creates ServerMetricsDataCollector instances for each configured endpoint,
        tests reachability, and sends status message to RecordsManager.
        If no endpoints are reachable, disables metrics collection and stops the service.

        Args:
            message: Profile configuration command from SystemController
        """
        # Check if server metrics are disabled via CLI flag
        if self._server_metrics_disabled:
            await self._send_server_metrics_status(
                enabled=False,
                reason="disabled via --no-server-metrics",
                endpoints_configured=[],
                endpoints_reachable=[],
            )
            return

        self._collectors.clear()

        for endpoint_url in self._server_metrics_endpoints:
            self.debug(
                lambda url=endpoint_url: (
                    f"Server Metrics: Testing reachability of {url}"
                )
            )
            collector = ServerMetricsDataCollector(
                endpoint_url=endpoint_url,
                collection_interval=self._collection_interval,
                record_callback=self._on_server_metrics_records,
                error_callback=self._on_server_metrics_error,
                collector_id=redact_url(endpoint_url),
            )
            self._attach_phase_scoped_collection(collector)

            try:
                is_reachable = await collector.is_url_reachable()
                if is_reachable:
                    self._collectors[endpoint_url] = collector
                    self.debug(
                        lambda url=endpoint_url: (
                            f"Server Metrics: Prometheus endpoint {url} is reachable"
                        )
                    )
                else:
                    self.debug(
                        lambda url=endpoint_url: (
                            f"Server Metrics: Prometheus endpoint {url} is not reachable"
                        )
                    )
            except Exception as e:
                self.error(f"Server Metrics: Exception testing {endpoint_url}: {e}")

        reachable_endpoints = [redact_url(u) for u in self._collectors]

        if not self._collectors:
            # Server metrics manager shutdown occurs in _on_start_profiling to prevent hang
            await self._send_server_metrics_status(
                enabled=False,
                reason="no Prometheus endpoints reachable",
                endpoints_configured=[
                    redact_url(u) for u in self._server_metrics_endpoints
                ],
                endpoints_reachable=[],
            )
            return

        # Capture baseline metrics before profiling starts
        self.info("Server Metrics: Capturing baseline metrics...")
        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.initialize()
                await collector.collect_and_process_metrics()
                self.debug(
                    lambda url=endpoint_url: (
                        f"Server Metrics: Captured baseline from {url}"
                    )
                )
            except Exception as e:
                self.warning(
                    f"Server Metrics: Failed to capture baseline from {endpoint_url}: {e}"
                )

        await self._send_server_metrics_status(
            enabled=True,
            reason=None,
            endpoints_configured=[
                redact_url(u) for u in self._server_metrics_endpoints
            ],
            endpoints_reachable=reachable_endpoints,
        )

    async def collect_baseline(self, message: PhaseBaselineRequestMessage) -> None:
        """Capture a one-shot server-metrics scrape for a phase boundary."""
        if self._server_metrics_disabled or not self._collectors:
            return
        boundary_phase = _ServerMetricsPhaseIdentity(
            phase=(
                CreditPhase.WARMUP
                if message.phase_kind == "warmup"
                else CreditPhase.PROFILING
            ),
            phase_index=message.phase_index,
            profiling_index=message.profiling_index,
            phase_name=message.phase_name,
            phase_kind=message.phase_kind,
        )
        errors: list[str] = []
        for endpoint_url, collector in list(self._collectors.items()):
            try:
                await self._collect_and_process_metrics_for_phase(
                    collector, boundary_phase
                )
            except Exception as exc:
                errors.append(f"{endpoint_url}: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: ProfileStartCommand) -> None:
        """Start all server metrics collectors for profiling phase.

        Initializes and starts background collection tasks for each configured
        collector. Handles partial failures gracefully - continues profiling if
        at least one collector starts successfully, only shuts down if all fail.

        If no collectors exist (all endpoints were unreachable during configuration),
        performs graceful shutdown.

        Args:
            message: Profile start command from SystemController signaling
                    that profiling phase is beginning
        """
        if not self._collectors:
            # Server metrics disabled status already sent in _profile_configure_command, only shutdown here
            self._shutdown_task = self.execute_async(self._delayed_shutdown())
            return

        started_count = 0
        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.start()
                started_count += 1
            except Exception as e:
                self.error(f"Failed to start collector for {endpoint_url}: {e}")

        total_collectors = len(self._collectors)
        if started_count == 0:
            self.warning("No server metrics collectors successfully started")
            await self._send_server_metrics_status(
                enabled=False,
                reason="all collectors failed to start",
                endpoints_configured=[
                    redact_url(u) for u in self._server_metrics_endpoints
                ],
                endpoints_reachable=[],
            )
            self._shutdown_task = self.execute_async(self._delayed_shutdown())
            return
        elif started_count < total_collectors:
            self.warning(
                f"Partial collector startup: {started_count}/{total_collectors} collectors started successfully"
            )
        else:
            self.info(
                f"Server Metrics: Started {started_count} collector(s) successfully"
            )

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(self, message: CreditPhaseStartMessage) -> None:
        """Track which benchmark phase subsequent server-metric scrapes belong to."""
        stats = message.stats
        identity = _ServerMetricsPhaseIdentity(
            phase=stats.phase,
            phase_index=stats.phase_index,
            profiling_index=stats.profiling_index,
            phase_name=stats.phase_name or str(stats.phase),
            phase_kind=stats.phase_kind,
        )
        self._active_phase = identity
        self.debug(f"Server Metrics: active phase is now {identity.phase_name}")
        if stats.phase_kind == "profiling" or stats.phase == CreditPhase.PROFILING:
            self._profiling_started = True
            self._last_profiling_phase = identity

    @on_message(MessageType.CREDIT_PHASE_COMPLETE)
    async def _on_credit_phase_complete(
        self, message: CreditPhaseCompleteMessage
    ) -> None:
        """Capture an end-of-warmup scrape and retire non-profiling phases.

        ``PROFILE_COMPLETE`` still owns the final profiling scrape. We do not
        clear profiling here because the profile-complete command is delivered
        after the profiling phase completes and should still tag the final
        scrape as profiling.
        """
        stats = message.stats
        identity = _ServerMetricsPhaseIdentity(
            phase=stats.phase,
            phase_index=stats.phase_index,
            profiling_index=stats.profiling_index,
            phase_name=stats.phase_name or str(stats.phase),
            phase_kind=stats.phase_kind,
        )
        is_warmup = stats.phase_kind == "warmup" or stats.phase == CreditPhase.WARMUP
        is_profiling = (
            stats.phase_kind == "profiling" or stats.phase == CreditPhase.PROFILING
        )
        if is_warmup and self._collectors:
            self.info(
                "Server Metrics: Warmup complete, capturing final warmup metrics..."
            )
            for endpoint_url, collector in list(self._collectors.items()):
                try:
                    await self._collect_and_process_metrics_for_phase(
                        collector, identity
                    )
                    self.debug(
                        lambda url=endpoint_url: (
                            f"Server Metrics: Captured warmup final state from {url}"
                        )
                    )
                except Exception as e:  # noqa: BLE001 - one endpoint's scrape failure must not skip the rest
                    self.warning(
                        f"Server Metrics: Failed to capture warmup final state from {endpoint_url}: {e}"
                    )

        if (
            not is_profiling
            and self._active_phase is not None
            and self._active_phase.phase == identity.phase
            and self._active_phase.phase_index == identity.phase_index
        ):
            self._active_phase = None

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_command(
        self, message: ProfileCompleteCommand
    ) -> None:
        """Trigger final scrape when profiling completes.

        When the last profiling phase is also the final configured phase,
        performs one final metrics collection from all endpoints to capture
        counter/histogram changes that settle after its END boundary. An
        earlier profiling phase relies on its boundary scrape so later phase
        traffic cannot be attributed to it.

        Critical for accurate delta calculations on counters and histograms,
        where missing the final state would undercount the actual activity.

        Idempotent: Can be called multiple times safely (e.g., if multiple
        RecordsManager instances send the command). Subsequent calls are no-ops.

        Args:
            message: Profile complete command from RecordsManager signaling that
                    all client request records have been processed
        """
        async with self._profile_complete_lock:
            if self._result_published:
                self.debug(
                    "Server Metrics: PROFILE_COMPLETE re-entry, result already published"
                )
                return

            if not self._collectors:
                self.debug("Server Metrics: Already stopped, skipping final scrape")
            elif self._should_capture_profile_complete_scrape():
                flush_end_ns = (message.end_ns or time.time_ns()) + int(
                    Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD * 1_000_000_000
                )
                remaining_seconds = (flush_end_ns - time.time_ns()) / 1_000_000_000
                if remaining_seconds > 0:
                    self.info(
                        f"Waiting {remaining_seconds:.1f}s for server metrics flush period..."
                    )
                    await asyncio.sleep(remaining_seconds)

                self.info(
                    "Server Metrics: Profiling complete, capturing final metrics..."
                )
                final_phase = self._last_profiling_phase or _ServerMetricsPhaseIdentity(
                    phase=CreditPhase.PROFILING,
                    phase_name=str(CreditPhase.PROFILING),
                    phase_kind="profiling",
                )
                for endpoint_url, collector in list(self._collectors.items()):
                    try:
                        await self._collect_and_process_metrics_for_phase(
                            collector, final_phase
                        )
                        self.debug(
                            lambda url=endpoint_url: (
                                f"Server Metrics: Captured final state from {url}"
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - one endpoint must not block others
                        self.warning(
                            f"Server Metrics: Failed to capture final state from {endpoint_url}: {exc}"
                        )
            else:
                self.info(
                    "Server Metrics: Skipping late profiling scrape because a later "
                    "configured phase owns the current server state"
                )

            await self._stop_all_collectors()

            await self._publish_server_metrics_result(
                start_ns=message.start_ns,
                end_ns=message.end_ns,
                warmup_start_ns=message.warmup_start_ns,
                warmup_end_ns=message.warmup_end_ns,
            )

    def _should_capture_profile_complete_scrape(self) -> bool:
        """Return whether a late scrape can still belong to the last profile.

        Every phase already requests an END boundary scrape. PROFILE_COMPLETE
        adds a delayed scrape for counters that settle after the final profiling
        phase. If another configured phase follows, server state may already
        include that phase, so forcibly tagging the scrape with the prior
        profiling identity would corrupt its phase-scoped snapshot.
        """
        identity = self._last_profiling_phase
        if identity is None or identity.phase_index is None:
            return True
        return identity.phase_index == len(self.run.cfg.phases) - 1

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Stop all server metrics collectors when profiling is cancelled.

        Called when user cancels profiling or an error occurs during profiling.
        Waits for flush period to allow metrics to finalize, then stops collectors.

        Args:
            message: Profile cancel command from SystemController
        """
        async with self._profile_complete_lock:
            await self._stop_all_collectors()
            if not self._result_published:
                await self._publish_server_metrics_result(
                    start_ns=None,
                    end_ns=None,
                    warmup_start_ns=None,
                    warmup_end_ns=None,
                )

    @on_stop
    async def _server_metrics_manager_stop(self) -> None:
        """Stop all server metrics collectors during service shutdown.

        Called automatically by BaseComponentService lifecycle management via @on_stop hook.
        Ensures all collectors are properly stopped and cleaned up even if shutdown
        command was not received.
        """
        await self._stop_all_collectors()

    async def _stop_all_collectors(self) -> None:
        """Stop all server metrics collectors.

        Attempts to stop each collector gracefully, logging errors but continuing with
        remaining collectors to ensure all resources are released. Does nothing if no
        collectors are configured.

        Errors during individual collector shutdown do not prevent other collectors
        from being stopped.
        """
        if not self._collectors:
            return

        # Copy the collectors to a list to avoid modifying the dictionary while iterating
        # Also enabled idempotent check to avoid stopping collectors multiple times
        collectors = list(self._collectors.items())
        self._collectors.clear()

        for endpoint_url, collector in collectors:
            try:
                await collector.stop()
            except Exception as e:
                self.error(f"Failed to stop collector for {endpoint_url}: {e}")
        self._active_phase = None
        self._last_profiling_phase = None

    async def _publish_server_metrics_result(
        self,
        *,
        start_ns: int | None,
        end_ns: int | None,
        warmup_start_ns: int | None,
        warmup_end_ns: int | None,
    ) -> None:
        """Publish one final result after all local raw records are ingested."""
        await self._finalize_stream_exporters()
        error_summary = [
            ErrorDetailsCount(error_details=error, count=count)
            for error, count in self._error_state.error_counts.items()
        ]
        result = None
        if self._accumulator is not None:
            resolved_start_ns = start_ns or 0
            resolved_end_ns = end_ns or time.time_ns()
            if resolved_end_ns < resolved_start_ns:
                self.warning(
                    f"Invalid server-metrics window {resolved_start_ns} > "
                    f"{resolved_end_ns}; exporting full history"
                )
                resolved_start_ns = 0
            try:
                result = await self._accumulator.export_results(
                    ExportContext(
                        start_ns=resolved_start_ns,
                        end_ns=resolved_end_ns,
                        error_summary=error_summary,
                        warmup_start_ns=warmup_start_ns,
                        warmup_end_ns=warmup_end_ns,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - publish terminal None result
                self.exception(f"Failed to export server metrics results: {exc!r}")
                self._error_state.record(ErrorDetails.from_exception(exc))

        await self.publish(
            ProcessServerMetricsResultMessage(
                service_id=self.service_id,
                server_metrics_result=ProcessServerMetricsResult(results=result),
            )
        )
        self._result_published = True

    async def _finalize_stream_exporters(self) -> None:
        """Flush manager-owned raw artifacts before publishing the final result."""
        if self._stream_exporters_finalized:
            return
        self._stream_exporters_finalized = True
        if not self._stream_exporters:
            return

        results = await asyncio.gather(
            *[exporter.finalize() for exporter in self._stream_exporters],
            return_exceptions=True,
        )
        for exporter, result in zip(self._stream_exporters, results, strict=True):
            if isinstance(result, BaseException):
                self.error(
                    f"Failed to finalize server metrics exporter "
                    f"{exporter.__class__.__name__}: {result!r}"
                )
                self._error_state.record(ErrorDetails.from_exception(result))

    async def _delayed_shutdown(self) -> None:
        """Shutdown service after a delay to allow command response to be sent.

        Waits before calling stop() to ensure the command response
        has time to be published and transmitted to the SystemController.
        """
        await asyncio.sleep(Environment.SERVER_METRICS.SHUTDOWN_DELAY)
        await asyncio.shield(self.stop())

    async def _publish_realtime_server_metrics(self) -> None:
        """Publish a bounded live summary, never raw Prometheus samples."""
        if self._accumulator is None or not self._profiling_started:
            return
        now_ns = time.time_ns()
        if now_ns - self._last_realtime_publish_ns < 1_000_000_000:
            return
        endpoint_summaries = self._accumulator.compute_endpoint_summaries(0, now_ns)
        snapshot = self._accumulator.realtime_snapshot()
        if not endpoint_summaries and not snapshot:
            return
        try:
            await self.publish(
                RealtimeServerMetricsMessage(
                    service_id=self.service_id,
                    endpoint_summaries=endpoint_summaries,
                    snapshot=snapshot,
                )
            )
        except Exception as exc:  # noqa: BLE001 - realtime is best-effort
            self.warning(f"Server Metrics: Failed to publish realtime update: {exc}")
            return
        self._last_realtime_publish_ns = now_ns

    def _attach_phase_scoped_collection(
        self, collector: ServerMetricsDataCollector
    ) -> None:
        original_collect = collector.collect_and_process_metrics

        async def collect_with_phase_snapshot() -> None:
            if _SERVER_METRICS_SCRAPE_PHASE.get() is not None:
                await original_collect()
                return

            token = _SERVER_METRICS_SCRAPE_PHASE.set(self._active_phase)
            try:
                await original_collect()
            finally:
                _SERVER_METRICS_SCRAPE_PHASE.reset(token)

        collector.collect_and_process_metrics = collect_with_phase_snapshot

    async def _collect_and_process_metrics_for_phase(
        self,
        collector: ServerMetricsDataCollector,
        phase: _ServerMetricsPhaseIdentity | None,
    ) -> None:
        token = _SERVER_METRICS_SCRAPE_PHASE.set(phase)
        try:
            await collector.collect_and_process_metrics()
        finally:
            _SERVER_METRICS_SCRAPE_PHASE.reset(token)

    async def _on_server_metrics_records(
        self, records: list[ServerMetricsRecord], collector_id: str
    ) -> None:
        """Async callback for receiving server metrics records from collectors.

        Called by ServerMetricsDataCollector instances when they successfully
        collect metrics. Records stay in this process and are fanned out to the
        local accumulator and JSONL exporter.

        Handles errors locally instead of raising them, ensuring collection can
        continue despite individual processor failures.

        Args:
            records: List of ServerMetricsRecord objects from a collection cycle.
                    Typically 1 record per successful scrape, may be empty if
                    endpoint returned no metrics.
            collector_id: Unique identifier of the collector (typically endpoint URL)
        """
        if not records:
            return

        scrape_phase = _SERVER_METRICS_SCRAPE_PHASE.get() or self._active_phase
        if scrape_phase is not None:
            records = [
                record.model_copy(
                    update={
                        "benchmark_phase": scrape_phase.phase,
                        "phase_index": scrape_phase.phase_index,
                        "profiling_index": scrape_phase.profiling_index,
                        "phase_name": scrape_phase.phase_name,
                        "phase_kind": scrape_phase.phase_kind,
                    }
                )
                for record in records
            ]

        results = await asyncio.gather(
            *[
                processor.process_record(record)
                for processor in self._processors
                for record in records
            ],
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                self.exception(
                    f"Failed to process server metrics record from {collector_id}: {result!r}"
                )
                self._error_state.record(ErrorDetails.from_exception(result))

        await self._publish_realtime_server_metrics()

    async def _on_server_metrics_error(
        self, error: ErrorDetails, collector_id: str
    ) -> None:
        """Async callback for receiving server metrics errors from collectors.

        Called by ServerMetricsDataCollector when collection fails (e.g., network
        timeout, HTTP error, parsing failure). Tracks it locally for final export.

        This callback-based error handling prevents exceptions from crashing
        the collector's background task, enabling recovery on subsequent scrapes.

        Args:
            error: ErrorDetails describing the collection error with exception info
            collector_id: Unique identifier of the collector (typically endpoint URL)
        """
        self.debug(lambda: f"Server Metrics: collector {collector_id} error: {error}")
        self._error_state.record(error)

    async def _send_server_metrics_status(
        self,
        enabled: bool,
        reason: str | None = None,
        endpoints_configured: list[str] | None = None,
        endpoints_reachable: list[str] | None = None,
    ) -> None:
        """Send server metrics status message to SystemController.

        Publishes ServerMetricsStatusMessage to inform SystemController about metrics
        availability and endpoint reachability. Used during configuration phase and
        when metrics are disabled due to errors.

        Args:
            enabled: Whether server metrics collection is enabled/available
            reason: Optional human-readable reason for status (e.g., "no Prometheus endpoints reachable")
            endpoints_configured: List of Prometheus endpoint URLs configured
            endpoints_reachable: List of Prometheus endpoint URLs that are accessible
        """
        try:
            status_message = ServerMetricsStatusMessage(
                service_id=self.service_id,
                enabled=enabled,
                reason=reason,
                endpoints_configured=endpoints_configured or [],
                endpoints_reachable=endpoints_reachable or [],
            )

            await self.publish(status_message)

        except Exception as e:
            self.error(f"Failed to send server metrics status message: {e}")
