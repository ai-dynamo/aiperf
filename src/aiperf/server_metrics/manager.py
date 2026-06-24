# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.enums import CommandType, MessageType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import on_command, on_message, on_stop
from aiperf.common.messages import (
    ProfileCancelCommand,
    ProfileCompleteCommand,
    ProfileConfigureCommand,
    ProfileStartCommand,
    RealtimeServerMetricsMessage,
    ServerMetricsStatusMessage,
)
from aiperf.common.metric_utils import normalize_metrics_endpoint_url
from aiperf.common.models import ErrorDetails, ServerMetricsRecord
from aiperf.common.redact import redact_url
from aiperf.credit.messages import CreditPhaseStartMessage
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServerMetricsProcessorType
from aiperf.server_metrics.data_collector import ServerMetricsDataCollector
from aiperf.server_metrics.protocols import (
    ServerMetricsAccumulatorProtocol,
    ServerMetricsProcessorProtocol,
)

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class ServerMetricsManager(BaseComponentService):
    """Coordinates multiple ServerMetricsDataCollector instances for server metrics collection.

    The ServerMetricsManager coordinates multiple ServerMetricsDataCollector instances,
    fans records out to local server-metrics processors, and publishes final and live
    server-metrics summaries.

    This service:
    - Manages lifecycle of ServerMetricsDataCollector instances
    - Collects metrics from multiple Prometheus endpoints
    - Processes records through local server-metrics processors
    - Handles errors gracefully with ErrorDetails

    Args:
        run: BenchmarkRun carrying the BenchmarkConfig + per-run state.
        service_id: Optional unique identifier for this service instance
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

        self._collectors: dict[str, ServerMetricsDataCollector] = {}
        self._server_metrics_disabled = not self.run.cfg.server_metrics.enabled

        from aiperf.common.models import ErrorTrackingState

        self._processors: list[ServerMetricsProcessorProtocol] = []
        self._accumulator: ServerMetricsAccumulatorProtocol | None = None
        self._result_published: bool = False
        self._profiling_started: bool = False
        self._last_realtime_publish_ns: int = 0
        self._profile_complete_lock = asyncio.Lock()
        self._error_state: ErrorTrackingState = ErrorTrackingState()

        for entry in plugins.iter_entries(PluginType.SERVER_METRICS_PROCESSOR):
            try:
                ProcessorClass = plugins.get_class(
                    PluginType.SERVER_METRICS_PROCESSOR,
                    entry.name,
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
            except Exception as e:  # noqa: BLE001 - plugin constructors are an extension boundary
                self.error(
                    f"Failed to create server metrics processor {entry.name}: {e}"
                )

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

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the server metrics collectors but don't start them yet.

        Creates ServerMetricsDataCollector instances for each configured endpoint,
        tests reachability, and publishes server-metrics status.
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
                lambda url=endpoint_url: f"Server Metrics: Testing reachability of {url}"
            )
            collector = ServerMetricsDataCollector(
                endpoint_url=endpoint_url,
                collection_interval=self._collection_interval,
                record_callback=self._on_server_metrics_records,
                error_callback=self._on_server_metrics_error,
                collector_id=redact_url(endpoint_url),
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
                    lambda url=endpoint_url: f"Server Metrics: Captured baseline from {url}"
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

    @on_message(MessageType.CREDIT_PHASE_START)
    async def _on_credit_phase_start(self, message: CreditPhaseStartMessage) -> None:
        if message.config.phase != "profiling":
            return
        self._profiling_started = True
        if not self._collectors:
            return

        self.info("Server Metrics: Capturing boundary metrics at profiling start...")
        for endpoint_url, collector in list(self._collectors.items()):
            try:
                await collector.collect_and_process_metrics()
                self.debug(
                    lambda url=endpoint_url: f"Server Metrics: Captured boundary state from {url}"
                )
            except Exception as e:  # noqa: BLE001 - one collector failure must not stop others
                self.warning(
                    f"Server Metrics: Failed to capture boundary state from {endpoint_url}: {e}"
                )

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

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_command(
        self, message: ProfileCompleteCommand
    ) -> None:
        """Trigger final scrape when profiling completes.

        Performs one final metrics collection from all endpoints to capture
        the end state immediately after profiling finishes. This ensures we
        have metrics that cover the entire profiling period, including any
        counter/histogram changes that occurred during the final seconds.

        Critical for accurate delta calculations on counters and histograms,
        where missing the final state would undercount the actual activity.

        Idempotent: Can be called multiple times safely. Subsequent calls are no-ops.

        Args:
            message: Profile complete command signaling that all client request
                    records have been processed
        """
        async with self._profile_complete_lock:
            if self._result_published:
                self.debug(
                    "Server Metrics: PROFILE_COMPLETE re-entry, result already published"
                )
                return

            if not self._collectors:
                self.debug("Server Metrics: Already stopped, skipping final scrape")
            else:
                self.info(
                    "Server Metrics: Profiling complete, capturing final metrics..."
                )

                for endpoint_url, collector in list(self._collectors.items()):
                    try:
                        await collector.collect_and_process_metrics()
                        self.debug(
                            lambda url=endpoint_url: f"Server Metrics: Captured final state from {url}"
                        )
                    except Exception as e:
                        self.warning(
                            f"Server Metrics: Failed to capture final state from {endpoint_url}: {e}"
                        )

                await self._stop_all_collectors()

            start_ns, end_ns = self._parse_profile_complete_window(message)
            await self._publish_server_metrics_result(start_ns=start_ns, end_ns=end_ns)

    def _parse_profile_complete_window(
        self,
        message: ProfileCompleteCommand,
    ) -> tuple[int | None, int | None]:
        """Parse PROFILE_COMPLETE timing window from the optional command payload."""
        import orjson

        payload = getattr(message, "payload", None)
        if not isinstance(payload, (str, bytes)) or not payload:
            return None, None
        try:
            decoded = orjson.loads(payload)
        except orjson.JSONDecodeError:
            self.warning(
                f"Server Metrics: Failed to parse PROFILE_COMPLETE payload: {payload!r}"
            )
            return None, None
        if not isinstance(decoded, dict):
            return None, None
        return (
            self._parse_profile_complete_timestamp(decoded, "start_ns"),
            self._parse_profile_complete_timestamp(decoded, "end_ns"),
        )

    def _parse_profile_complete_timestamp(
        self,
        decoded: dict[object, object],
        field_name: str,
    ) -> int | None:
        """Return a validated PROFILE_COMPLETE timestamp field."""
        value = decoded.get(field_name)
        if value is None:
            return None
        if type(value) is int:
            return value
        self.warning(
            f"Server Metrics: Ignoring invalid PROFILE_COMPLETE {field_name}: {value!r}"
        )
        return None

    async def _publish_server_metrics_result(
        self,
        start_ns: int | None,
        end_ns: int | None,
    ) -> None:
        """Publish a single final server-metrics result message."""
        from aiperf.server_metrics.result_publisher import publish_server_metrics_result

        await publish_server_metrics_result(
            publisher=self,
            accumulator=self._accumulator,
            error_state=self._error_state,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        self._result_published = True

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
            except Exception as e:
                self.error(f"Failed to stop collector for {endpoint_url}: {e}")

    async def _delayed_shutdown(self) -> None:
        """Shutdown service after a delay to allow command response to be sent.

        Waits before calling stop() to ensure the command response
        has time to be published and transmitted to the SystemController.
        """
        await asyncio.sleep(Environment.SERVER_METRICS.SHUTDOWN_DELAY)
        await asyncio.shield(self.stop())

    async def _publish_realtime_server_metrics(self) -> None:
        if self._accumulator is None or not self._profiling_started:
            return
        now_ns = time.time_ns()
        if now_ns - self._last_realtime_publish_ns < 1_000_000_000:
            return
        endpoint_summaries = self._accumulator.compute_endpoint_summaries(
            0, now_ns, None
        )
        if not endpoint_summaries:
            return
        try:
            await self.publish(
                RealtimeServerMetricsMessage(
                    service_id=self.service_id,
                    endpoint_summaries=endpoint_summaries,
                )
            )
        except Exception as e:  # noqa: BLE001 - realtime update failures are non-fatal
            self.warning(f"Server Metrics: Failed to publish realtime update: {e}")
            return
        self._last_realtime_publish_ns = now_ns

    async def _on_server_metrics_records(
        self, records: list[ServerMetricsRecord], collector_id: str
    ) -> None:
        """Async callback for receiving server metrics records from collectors.

        Called by ServerMetricsDataCollector instances when they successfully
        collect metrics. Fans out records to local server-metrics processors.

        Handles processor errors locally without raising exceptions, ensuring
        collector operation continues despite individual record processing failures.

        Args:
            records: List of ServerMetricsRecord objects from a collection cycle.
                    Typically 1 record per successful scrape, may be empty if
                    endpoint returned no metrics.
            collector_id: Unique identifier of the collector (typically endpoint URL)
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
                self._error_state.record_error(ErrorDetails.from_exception(error))

        await self._publish_realtime_server_metrics()

    async def _on_server_metrics_error(
        self, error: ErrorDetails, collector_id: str
    ) -> None:
        """Async callback for receiving server metrics errors from collectors.

        Called by ServerMetricsDataCollector when collection fails (e.g., network
        timeout, HTTP error, parsing failure). Records the error locally while
        allowing collection to continue on subsequent scrapes.

        Args:
            error: ErrorDetails describing the collection error with exception info
            collector_id: Unique identifier of the collector (typically endpoint URL)
        """
        self._error_state.record_error(error)

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
