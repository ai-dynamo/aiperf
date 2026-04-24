# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import orjson

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.control_structs import Command, ServerMetricsStatus
from aiperf.common.enums import CommandType, MessageType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import on_command, on_message, on_stop
from aiperf.common.models import (
    ErrorDetails,
    ErrorTrackingState,
    ServerMetricsRecord,
)
from aiperf.credit.messages import CreditPhaseStartMessage
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServerMetricsProcessorType
from aiperf.server_metrics.data_collector import ServerMetricsDataCollector
from aiperf.server_metrics.endpoint_resolver import EndpointResolver
from aiperf.server_metrics.protocols import (
    ServerMetricsAccumulatorProtocol,
    ServerMetricsProcessorProtocol,
)
from aiperf.server_metrics.result_publisher import publish_server_metrics_result

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class ServerMetricsManager(BaseComponentService):
    """Coordinates multiple ServerMetricsDataCollector instances for server metrics collection.

    The ServerMetricsManager coordinates multiple ServerMetricsDataCollector
    instances to collect server metrics from multiple Prometheus endpoints, fan
    out each collected record to locally-loaded server metrics processors, and
    publish the accumulated ProcessServerMetricsResultMessage when profiling
    completes.

    This service:
    - Manages lifecycle of ServerMetricsDataCollector instances
    - Collects metrics from multiple Prometheus endpoints
    - Runs all configured server metrics processors (plugins registered under
      SERVER_METRICS_PROCESSOR) locally on each collected record
    - Accumulates results and publishes ProcessServerMetricsResultMessage on
      PROFILE_COMPLETE
    - Handles errors gracefully with ErrorDetails

    Args:
        config: AIPerf configuration including server_metrics endpoints
        service_id: Optional unique identifier for this service instance
    """

    def __init__(
        self,
        run: BenchmarkRun,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(run=run, service_id=service_id, **kwargs)

        self._collectors: dict[str, ServerMetricsDataCollector] = {}

        server_metrics_config = run.cfg.server_metrics
        self._server_metrics_disabled = (
            not server_metrics_config.enabled if server_metrics_config else True
        )

        self._resolver = EndpointResolver(run)
        self._server_metrics_endpoints = self._resolver.build_endpoints(
            include_default_endpoints=True
        )
        self.info(
            "Server Metrics: Configured "
            f"{len(self._server_metrics_endpoints)} initial endpoint(s): "
            f"{self._server_metrics_endpoints}"
        )

        self._collection_interval = Environment.SERVER_METRICS.COLLECTION_INTERVAL
        # Task for delayed shutdown, created when no endpoints are reachable
        self._shutdown_task: asyncio.Task[None] | None = None

        self._processors: list[ServerMetricsProcessorProtocol] = []
        self._accumulator: ServerMetricsAccumulatorProtocol | None = None
        self._error_state = ErrorTrackingState()
        self._result_published: bool = False

        self._init_processors()

    def _init_processors(self) -> None:
        """Instantiate all SERVER_METRICS_PROCESSOR plugins; skip disabled/failed ones."""
        for entry in plugins.iter_entries(PluginType.SERVER_METRICS_PROCESSOR):
            try:
                ProcessorClass = plugins.get_class(
                    PluginType.SERVER_METRICS_PROCESSOR, entry.name
                )
                processor = ProcessorClass(
                    service_id=self.service_id,
                    run=self.run,
                    pub_client=self.pub_client,
                )
                self.attach_child_lifecycle(processor)
                self._processors.append(processor)
                if entry.name == ServerMetricsProcessorType.SERVER_METRICS_ACCUMULATOR:
                    self._accumulator = processor
                self.debug(
                    f"Created server metrics processor: {entry.name}: "
                    f"{processor.__class__.__name__}"
                )
            except PostProcessorDisabled:
                self.debug(
                    f"Server metrics processor {entry.name} is disabled and will not be used"
                )
            except Exception as e:  # noqa: BLE001 - per-plugin; skip bad processor and continue
                self.error(
                    f"Failed to create server metrics processor {entry.name}: {e}"
                )

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(self, message: Command) -> None:
        """Configure the server metrics collectors but don't start them yet.

        Creates ServerMetricsDataCollector instances for each configured endpoint,
        tests reachability, and sends status message to RecordsManager.
        If no endpoints are reachable, disables metrics collection and stops the service.

        Args:
            message: Profile configuration command from SystemController
        """
        if self._server_metrics_disabled:
            await self._send_server_metrics_status(
                enabled=False,
                reason="disabled via --no-server-metrics",
                endpoints_configured=[],
                endpoints_reachable=[],
            )
            return

        await self._resolve_endpoints()
        await self._probe_collectors()
        reachable_endpoints = list(self._collectors.keys())

        if not self._collectors:
            # Server metrics manager shutdown occurs in _on_start_profiling to prevent hang
            await self._send_server_metrics_status(
                enabled=False,
                reason="no Prometheus endpoints reachable",
                endpoints_configured=self._server_metrics_endpoints,
                endpoints_reachable=[],
            )
            return

        await self._capture_baseline_metrics()
        await self._send_server_metrics_status(
            enabled=True,
            reason=None,
            endpoints_configured=self._server_metrics_endpoints,
            endpoints_reachable=reachable_endpoints,
        )

    async def _resolve_endpoints(self) -> None:
        """Rebuild scrape targets and fold in auto-discovery results."""
        self._server_metrics_endpoints = self._resolver.build_endpoints(
            include_default_endpoints=self._resolver.should_include_default_endpoints(),
        )

        discovered_urls = await self._run_metrics_discovery()
        added = 0
        for url in discovered_urls:
            if url not in self._server_metrics_endpoints:
                self._server_metrics_endpoints.append(url)
                added += 1
        if added > 0:
            self.info(f"Server Metrics: Auto-discovery added {added} endpoint(s)")

    async def _probe_collectors(self) -> None:
        """Create a collector per endpoint and keep only reachable ones."""
        self._collectors.clear()
        for endpoint_url in self._server_metrics_endpoints:
            self.debug(
                lambda url=endpoint_url: f"Server Metrics: Testing reachability of {url}"
            )
            collector = ServerMetricsDataCollector(
                endpoint_url=endpoint_url,
                collection_interval=self._collection_interval,
                record_callback=self._on_server_metrics_records,
                error_callback=self._on_server_metrics_error,
                collector_id=endpoint_url,
            )
            try:
                is_reachable = await collector.is_url_reachable()
                if is_reachable:
                    self._collectors[endpoint_url] = collector
                    self.debug(
                        lambda url=endpoint_url: f"Server Metrics: Prometheus endpoint {url} is reachable"
                    )
                else:
                    self.debug(
                        lambda url=endpoint_url: f"Server Metrics: Prometheus endpoint {url} is not reachable"
                    )
            except Exception as e:  # noqa: BLE001 - per-endpoint; skip unreachable and continue
                self.error(f"Server Metrics: Exception testing {endpoint_url}: {e}")

    async def _capture_baseline_metrics(self) -> None:
        """Capture pre-profiling baseline scrape from every reachable collector."""
        self.info("Server Metrics: Capturing baseline metrics...")
        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.initialize()
                await collector.collect_and_process_metrics()
                self.debug(
                    lambda url=endpoint_url: f"Server Metrics: Captured baseline from {url}"
                )
            except Exception as e:  # noqa: BLE001 - per-endpoint; skip baseline failure and continue
                self.warning(
                    f"Server Metrics: Failed to capture baseline from {endpoint_url}: {e}"
                )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: Command) -> None:
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
            self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
            return

        started_count = 0
        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.start()
                started_count += 1
            except Exception as e:  # noqa: BLE001 - per-collector; skip start failure and continue
                self.error(f"Failed to start collector for {endpoint_url}: {e}")

        total_collectors = len(self._collectors)
        if started_count == 0:
            self.warning("No server metrics collectors successfully started")
            await self._send_server_metrics_status(
                enabled=False,
                reason="all collectors failed to start",
                endpoints_configured=self._server_metrics_endpoints,
                endpoints_reachable=[],
            )
            self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
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
        """Force a boundary scrape when profiling phase starts.

        Captures a clean post-warmup reference point for counter/histogram delta
        calculations. Without this, the reference may be the pre-warmup baseline
        from PROFILE_CONFIGURE, causing warmup activity to leak into profiling deltas.

        Args:
            message: Credit phase start message from TimingManager
        """
        if message.config.phase != "profiling":
            return
        if not self._collectors:
            return

        self.info("Server Metrics: Capturing boundary metrics at profiling start...")
        for endpoint_url, collector in list(self._collectors.items()):
            try:
                await collector.collect_and_process_metrics()
                self.debug(
                    lambda url=endpoint_url: f"Server Metrics: Captured boundary state from {url}"
                )
            except Exception as e:  # noqa: BLE001 - per-endpoint; skip boundary failure and continue
                self.warning(
                    f"Server Metrics: Failed to capture boundary state from {endpoint_url}: {e}"
                )

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_command(self, message: Command) -> None:
        """Trigger final scrape when profiling completes.

        Performs one final metrics collection from all endpoints to capture
        the end state immediately after profiling finishes. This ensures we
        have metrics that cover the entire profiling period, including any
        counter/histogram changes that occurred during the final seconds.

        Critical for accurate delta calculations on counters and histograms,
        where missing the final state would undercount the actual activity.

        Idempotent: Can be called multiple times safely (e.g., if multiple
        RecordsManager instances send the command). Subsequent calls are no-ops.

        Args:
            message: Profile complete command from RecordsManager signaling that
                    all client request records have been processed
        """
        # Idempotent check - skip if already stopped or no collectors
        if not self._collectors:
            self.debug("Server Metrics: Already stopped, skipping final scrape")
        else:
            self.info("Server Metrics: Profiling complete, capturing final metrics...")

            # Trigger final scrape from all collectors
            for endpoint_url, collector in list(self._collectors.items()):
                try:
                    await collector.collect_and_process_metrics()
                    self.debug(
                        lambda url=endpoint_url: f"Server Metrics: Captured final state from {url}"
                    )
                except Exception as e:  # noqa: BLE001 - per-endpoint; skip final scrape failure and continue
                    self.warning(
                        f"Server Metrics: Failed to capture final state from {endpoint_url}: {e}"
                    )

            await self._stop_all_collectors()

        if self._result_published:
            self.debug(
                "Server Metrics: PROFILE_COMPLETE re-entry, result already published"
            )
            return

        # RecordsManager sends the results time window in the PROFILE_COMPLETE
        # payload. Fall back to current time if unavailable.
        start_ns: int | None = None
        end_ns: int | None = None
        if message.payload:
            try:
                parsed = orjson.loads(message.payload)
                start_ns = parsed.get("start_ns")
                end_ns = parsed.get("end_ns")
            except Exception as e:  # noqa: BLE001 - best-effort payload parse; fall back to current time
                self.warning(
                    f"Failed to parse PROFILE_COMPLETE payload ({e!r}); "
                    "using current time for start_ns/end_ns"
                )

        await self._publish_server_metrics_result(start_ns=start_ns, end_ns=end_ns)

    async def _publish_server_metrics_result(
        self, start_ns: int | None, end_ns: int | None
    ) -> None:
        """Publish accumulated server metrics results to subscribers. Idempotent."""
        # Latch before awaits so a failed publish still prevents republish on re-entry.
        self._result_published = True
        await publish_server_metrics_result(
            publisher=self,
            accumulator=self._accumulator,
            error_state=self._error_state,
            start_ns=start_ns,
            end_ns=end_ns,
        )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(self, message: Command) -> None:
        """Stop all server metrics collectors when profiling is cancelled.

        Called when user cancels profiling or an error occurs during profiling.
        Waits for flush period to allow metrics to finalize, then stops collectors.

        Args:
            message: Profile cancel command from SystemController
        """
        await self._stop_all_collectors()

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
            except Exception as e:  # noqa: BLE001 - per-collector; continue shutting others down
                self.error(f"Failed to stop collector for {endpoint_url}: {e}")

    async def _delayed_shutdown(self) -> None:
        """Sleep briefly so the command response flushes, then stop the service."""
        await asyncio.sleep(Environment.SERVER_METRICS.SHUTDOWN_DELAY)
        await asyncio.shield(self.stop())

    async def _on_server_metrics_records(
        self, records: list[ServerMetricsRecord], collector_id: str
    ) -> None:
        """Fan out records to all loaded server metrics processors.

        A single flattened gather runs over every (processor, record) pair;
        storage tolerates out-of-order ingestion.
        """
        if not records:
            return

        errors = await asyncio.gather(
            *[
                processor.process_server_metrics_record(record)
                for processor in self._processors
                for record in records
            ],
            return_exceptions=True,
        )
        for error in errors:
            if isinstance(error, BaseException):
                self.exception(f"Failed to process server metrics record: {error!r}")
                self._error_state.error_counts[ErrorDetails.from_exception(error)] += 1

    async def _on_server_metrics_error(
        self, error: ErrorDetails, collector_id: str
    ) -> None:
        """Callback from collectors to record scrape errors into the local error state.

        Records flow into the ProcessServerMetricsResultMessage published on
        PROFILE_COMPLETE; returning instead of raising keeps the collector's
        background task alive across transient failures.
        """
        self._error_state.error_counts[error] += 1

    async def _send_server_metrics_status(
        self,
        enabled: bool,
        reason: str | None = None,
        endpoints_configured: list[str] | None = None,
        endpoints_reachable: list[str] | None = None,
    ) -> None:
        """Publish ServerMetricsStatus to SystemController (config phase / disable paths)."""
        try:
            await self.control_client.send(
                ServerMetricsStatus(
                    sid=self.service_id,
                    enabled=enabled,
                    reason=reason,
                    endpoints_configured=tuple(endpoints_configured or []),
                    endpoints_reachable=tuple(endpoints_reachable or []),
                )
            )
        except Exception as e:  # noqa: BLE001 - best-effort status publish
            self.error(f"Failed to send server metrics status message: {e}")

    async def _run_metrics_discovery(self) -> list[str]:
        """Run metrics endpoint auto-discovery; delegates to EndpointResolver."""
        return await self._resolver.run_discovery(self)
