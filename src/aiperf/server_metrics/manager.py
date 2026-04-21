# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.channel_codecs import RECORDS_CODEC
from aiperf.common.control_structs import Command, ServerMetricsStatus
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    MessageType,
    ServerMetricsDiscoveryMode,
)
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_message, on_stop
from aiperf.common.metric_utils import normalize_metrics_endpoint_url
from aiperf.common.models import ErrorDetails, ServerMetricsRecord
from aiperf.common.protocols import PushClientProtocol
from aiperf.common.server_metrics_records_wire import (
    build_server_metrics_record_wire_message,
)
from aiperf.credit.messages import CreditPhaseStartMessage
from aiperf.server_metrics.data_collector import ServerMetricsDataCollector
from aiperf.server_metrics.discovery.kubernetes import (
    discover_kubernetes_endpoints,
    is_running_in_kubernetes,
)

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
    from aiperf.config.artifacts import ServerMetricsDiscoveryConfig


class ServerMetricsManager(BaseComponentService):
    """Coordinates multiple ServerMetricsDataCollector instances for server metrics collection.

    The ServerMetricsManager coordinates multiple ServerMetricsDataCollector instances
    to collect server metrics from multiple Prometheus endpoints and send unified
    ServerMetricsRecordWireMessage to RecordsManager.

    This service:
    - Manages lifecycle of ServerMetricsDataCollector instances
    - Collects metrics from multiple Prometheus endpoints
    - Sends ServerMetricsRecordWireMessage to RecordsManager via message system
    - Handles errors gracefully with ErrorDetails
    - Follows centralized architecture patterns

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
        super().__init__(
            run=run,
            service_id=service_id,
            **kwargs,
        )

        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
            codec=RECORDS_CODEC,
        )

        self._collectors: dict[str, ServerMetricsDataCollector] = {}

        # Check if server metrics is enabled in config
        server_metrics_config = run.cfg.server_metrics
        self._server_metrics_disabled = (
            not server_metrics_config.enabled if server_metrics_config else True
        )

        # Store discovery configuration
        self._discovery: ServerMetricsDiscoveryConfig | None = (
            server_metrics_config.discovery if server_metrics_config else None
        )

        # Keep default inference-derived metrics URLs separate from explicit server
        # metrics URLs so Kubernetes discovery can replace the load-balanced default
        # targets without discarding user-provided scrape endpoints.
        self._default_server_metrics_endpoints: list[str] = []
        for url in run.cfg.endpoint.urls:
            normalized_url = normalize_metrics_endpoint_url(url)
            if normalized_url not in self._default_server_metrics_endpoints:
                self._default_server_metrics_endpoints.append(normalized_url)

        self._explicit_server_metrics_endpoints: list[str] = []
        if server_metrics_config and server_metrics_config.urls:
            for url in server_metrics_config.urls:
                normalized_url = normalize_metrics_endpoint_url(url)
                if normalized_url not in self._explicit_server_metrics_endpoints:
                    self._explicit_server_metrics_endpoints.append(normalized_url)

        self._server_metrics_endpoints = self._build_server_metrics_endpoints(
            include_default_endpoints=True
        )
        self.info(
            "Server Metrics: Configured "
            f"{len(self._server_metrics_endpoints)} initial endpoint(s): "
            f"{self._server_metrics_endpoints}"
        )

        # Use server metrics collection interval
        self._collection_interval = Environment.SERVER_METRICS.COLLECTION_INTERVAL

        # Task for delayed shutdown, created when no endpoints are reachable
        self._shutdown_task: asyncio.Task[None] | None = None

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(self, message: Command) -> None:
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

        self._server_metrics_endpoints = self._build_server_metrics_endpoints(
            include_default_endpoints=self._should_include_default_endpoints()
        )

        # Run auto-discovery if enabled
        discovered_urls = await self._run_metrics_discovery()
        added = 0
        for url in discovered_urls:
            if url not in self._server_metrics_endpoints:
                self._server_metrics_endpoints.append(url)
                added += 1
        if added > 0:
            self.info(f"Server Metrics: Auto-discovery added {added} endpoint(s)")

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
            except Exception as e:
                self.error(f"Server Metrics: Exception testing {endpoint_url}: {e}")

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
            endpoints_configured=self._server_metrics_endpoints,
            endpoints_reachable=reachable_endpoints,
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
            except Exception as e:
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
            except Exception as e:
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
            return

        self.info("Server Metrics: Profiling complete, capturing final metrics...")

        # Trigger final scrape from all collectors
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

        # Stop all collectors after final scrape
        await self._stop_all_collectors()

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
            except Exception as e:
                self.error(f"Failed to stop collector for {endpoint_url}: {e}")

    async def _delayed_shutdown(self) -> None:
        """Shutdown service after a delay to allow command response to be sent.

        Waits before calling stop() to ensure the command response
        has time to be published and transmitted to the SystemController.
        """
        await asyncio.sleep(Environment.SERVER_METRICS.SHUTDOWN_DELAY)
        await asyncio.shield(self.stop())

    async def _on_server_metrics_records(
        self, records: list[ServerMetricsRecord], collector_id: str
    ) -> None:
        """Async callback for receiving server metrics records from collectors.

        Called by ServerMetricsDataCollector instances when they successfully
        collect metrics. Forwards records to RecordsManager via ZMQ push socket,
        preserving all metadata for hierarchical storage and processing.

        Handles errors gracefully by sending error messages to RecordsManager
        instead of raising exceptions, ensuring collector continues operation
        despite individual record processing failures.

        Args:
            records: List of ServerMetricsRecord objects from a collection cycle.
                    Typically 1 record per successful scrape, may be empty if
                    endpoint returned no metrics.
            collector_id: Unique identifier of the collector (typically endpoint URL)
        """
        if not records:
            return

        for record in records:
            try:
                message = build_server_metrics_record_wire_message(
                    service_id=self.service_id,
                    collector_id=collector_id,
                    record=record,
                    error=None,
                )

                await self.records_push_client.push(message)

            except Exception as e:
                self.error(
                    f"Failed to send server metrics record from {collector_id}: {e}"
                )
                # Send error message to RecordsManager to track the failure
                try:
                    error_message = build_server_metrics_record_wire_message(
                        service_id=self.service_id,
                        collector_id=collector_id,
                        record=None,
                        error=ErrorDetails.from_exception(e),
                    )
                    await self.records_push_client.push(error_message)
                except Exception as nested_error:
                    self.error(
                        f"Failed to send error message after record send failure: {nested_error}"
                    )

    async def _on_server_metrics_error(
        self, error: ErrorDetails, collector_id: str
    ) -> None:
        """Async callback for receiving server metrics errors from collectors.

        Called by ServerMetricsDataCollector when collection fails (e.g., network
        timeout, HTTP error, parsing failure). Forwards error to RecordsManager
        for tracking and reporting, allowing the system to continue operation
        despite individual collector failures.

        This callback-based error handling prevents exceptions from crashing
        the collector's background task, enabling recovery on subsequent scrapes.

        Args:
            error: ErrorDetails describing the collection error with exception info
            collector_id: Unique identifier of the collector (typically endpoint URL)
        """
        try:
            error_message = build_server_metrics_record_wire_message(
                service_id=self.service_id,
                collector_id=collector_id,
                record=None,
                error=error,
            )

            await self.records_push_client.push(error_message)

        except Exception as e:
            self.error(f"Failed to send server metrics error message: {e}")

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
            await self.control_client.send(
                ServerMetricsStatus(
                    sid=self.service_id,
                    enabled=enabled,
                    reason=reason,
                    endpoints_configured=tuple(endpoints_configured or []),
                    endpoints_reachable=tuple(endpoints_reachable or []),
                )
            )
        except Exception as e:
            self.error(f"Failed to send server metrics status message: {e}")

    def _build_server_metrics_endpoints(
        self, *, include_default_endpoints: bool
    ) -> list[str]:
        """Build the current scrape target list with stable deduplication."""
        endpoints: list[str] = []

        if include_default_endpoints:
            for url in self._default_server_metrics_endpoints:
                if url not in endpoints:
                    endpoints.append(url)

        for url in self._explicit_server_metrics_endpoints:
            if url not in endpoints:
                endpoints.append(url)

        return endpoints

    def _should_include_default_endpoints(self) -> bool:
        """Whether inference-derived metrics URLs should remain scrape targets.

        Kubernetes discovery should replace the default inference endpoint scrape
        target because that endpoint often resolves to a load balancer rather than
        a single pod, which can corrupt cumulative metrics.
        """
        if self._discovery is None:
            return True

        mode = self._discovery.mode
        if mode == ServerMetricsDiscoveryMode.DISABLED:
            return True
        if mode == ServerMetricsDiscoveryMode.KUBERNETES:
            return False
        if mode == ServerMetricsDiscoveryMode.AUTO:
            return not is_running_in_kubernetes()

        return True

    async def _run_metrics_discovery(self) -> list[str]:
        """Run metrics endpoint auto-discovery based on configuration.

        Returns:
            List of discovered endpoint URLs.
        """
        if self._discovery is None:
            return []

        mode = self._discovery.mode
        if mode == ServerMetricsDiscoveryMode.DISABLED:
            return []

        if mode == ServerMetricsDiscoveryMode.KUBERNETES:
            if not is_running_in_kubernetes():
                self.warning(
                    "Server Metrics: Kubernetes discovery requested but not running in K8s cluster"
                )
                return []
            self.info("Server Metrics: Running Kubernetes discovery...")
            try:
                return await discover_kubernetes_endpoints(
                    namespace=self._discovery.namespace,
                    label_selector=self._discovery.label_selector,
                )
            except Exception as e:
                self.warning(f"Server Metrics: Kubernetes discovery failed: {e}")
                return []

        # AUTO mode: try K8s if in-cluster
        if is_running_in_kubernetes():
            # Derive namespace from endpoint URLs if not explicitly set.
            # Service DNS: "svc-name.namespace.svc.cluster.local" → namespace
            ns = self._discovery.namespace
            if ns is None:
                ns = self._extract_namespace_from_endpoints()

            self.info(
                f"Server Metrics: Running Kubernetes auto-discovery"
                f"{f' (namespace={ns})' if ns else ''}..."
            )
            try:
                return await discover_kubernetes_endpoints(
                    namespace=ns,
                    label_selector=self._discovery.label_selector,
                )
            except Exception as e:
                self.warning(f"Server Metrics: Kubernetes auto-discovery failed: {e}")
                return []

        self.debug(lambda: "Server Metrics: Not in K8s, skipping auto-discovery")
        return []

    def _extract_namespace_from_endpoints(self) -> str | None:
        """Extract K8s namespace from endpoint service DNS names.

        Parses ``svc-name.namespace.svc.cluster.local`` to extract namespace.
        Returns the first namespace found, or None.
        """
        from urllib.parse import urlparse

        for url in (
            self._explicit_server_metrics_endpoints
            + self._default_server_metrics_endpoints
        ):
            try:
                host = urlparse(url).hostname or ""
                parts = host.split(".")
                if len(parts) >= 3 and parts[2] == "svc":
                    return parts[1]
            except Exception:
                continue
        return None
