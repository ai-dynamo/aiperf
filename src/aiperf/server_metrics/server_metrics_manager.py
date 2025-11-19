# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from urllib.parse import urlparse

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import on_command, on_init, on_stop
from aiperf.common.messages import (
    ProfileCancelCommand,
    ProfileConfigureCommand,
    ProfileStartCommand,
)
from aiperf.common.messages.server_metrics_messages import (
    ServerMetricsRecordsMessage,
    ServerMetricsStatusMessage,
)
from aiperf.common.metric_utils import (
    build_hostname_aware_prometheus_endpoints,
    normalize_metrics_endpoint_url,
)
from aiperf.common.models import ErrorDetails
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.common.protocols import (
    PushClientProtocol,
    ServiceProtocol,
)
from aiperf.server_metrics.server_metrics_data_collector import (
    ServerMetricsDataCollector,
)

__all__ = ["ServerMetricsManager"]


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.SERVER_METRICS_MANAGER)
class ServerMetricsManager(BaseComponentService):
    """Coordinates multiple ServerMetricsDataCollector instances for server metrics collection.

    The ServerMetricsManager coordinates multiple ServerMetricsDataCollector instances
    to collect server metrics from multiple Prometheus endpoints and send unified
    ServerMetricsRecordsMessage to RecordsManager.

    This service:
    - Manages lifecycle of ServerMetricsDataCollector instances
    - Collects metrics from multiple Prometheus endpoints
    - Sends ServerMetricsRecordsMessage to RecordsManager via message system
    - Handles errors gracefully with ErrorDetails
    - Follows centralized architecture patterns

    Args:
        service_config: Service-level configuration (logging, communication, etc.)
        user_config: User-provided configuration including server_metrics endpoints
        service_id: Optional unique identifier for this service instance
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )

        self.records_push_client: PushClientProtocol = self.comms.create_push_client(
            CommAddress.RECORDS,
        )

        self._collectors: dict[str, ServerMetricsDataCollector] = {}

        inference_port = None
        if user_config.endpoint.url:
            parsed = urlparse(user_config.endpoint.url)
            inference_port = parsed.port

        ports_to_check = []
        if inference_port:
            ports_to_check.append(inference_port)
        ports_to_check.extend(Environment.SERVER_METRICS.DEFAULT_BACKEND_PORTS)

        self._server_metrics_endpoints = build_hostname_aware_prometheus_endpoints(
            inference_endpoint_url=user_config.endpoint.url,
            default_ports=ports_to_check,
        )

        # Add user-specified URLs if provided
        if user_config.server_metrics_urls:
            # Add user URLs, avoiding duplicates
            for url in user_config.server_metrics_urls:
                normalized_url = normalize_metrics_endpoint_url(url)
                if normalized_url not in self._server_metrics_endpoints:
                    self._server_metrics_endpoints.append(normalized_url)

        # Use server metrics collection interval
        self._collection_interval = Environment.SERVER_METRICS.COLLECTION_INTERVAL

    @on_init
    async def _initialize(self) -> None:
        """Initialize server metrics manager.

        Called automatically during service startup via @on_init hook.
        Actual collector initialization happens in _profile_configure_command
        after configuration is received from SystemController.
        """
        pass

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
        self._collectors.clear()

        for endpoint_url in self._server_metrics_endpoints:
            self.debug(f"Server Metrics: Testing reachability of {endpoint_url}")
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
                        f"Server Metrics: Prometheus endpoint {endpoint_url} is reachable"
                    )
                else:
                    self.debug(
                        f"Server Metrics: Prometheus endpoint {endpoint_url} is not reachable"
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

        await self._send_server_metrics_status(
            enabled=True,
            reason=None,
            endpoints_configured=self._server_metrics_endpoints,
            endpoints_reachable=reachable_endpoints,
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: ProfileStartCommand) -> None:
        """Start all server metrics collectors.

        Initializes and starts each configured collector.
        If no collectors start successfully, sends disabled status to SystemController.

        Args:
            message: Profile start command from SystemController
        """
        if not self._collectors:
            # Server metrics disabled status already sent in _profile_configure_command, only shutdown here
            await self.stop()
            return

        started_count = 0
        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.initialize()
                await collector.start()
                started_count += 1
            except Exception as e:
                self.error(f"Failed to start collector for {endpoint_url}: {e}")

        if started_count == 0:
            self.warning("No server metrics collectors successfully started")
            await self._send_server_metrics_status(
                enabled=False,
                reason="all collectors failed to start",
                endpoints_configured=self._server_metrics_endpoints,
                endpoints_reachable=[],
            )
            await self.stop()
            return

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
        flush_period = Environment.SERVER_METRICS.COLLECTION_FLUSH_PERIOD
        if flush_period > 0:
            self.debug(
                f"Server Metrics: Waiting {flush_period}s flush period for final metrics to finalize"
            )
            await asyncio.sleep(flush_period)

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

        for endpoint_url, collector in self._collectors.items():
            try:
                await collector.stop()
            except Exception as e:
                self.error(f"Failed to stop collector for {endpoint_url}: {e}")

    async def _on_server_metrics_records(
        self, records: list[ServerMetricsRecord], collector_id: str
    ) -> None:
        """Async callback for receiving server metrics records from collectors.

        Sends full records with all metadata to RecordsManager.
        Empty record lists are ignored.

        Args:
            records: List of ServerMetricsRecord objects from a collector
            collector_id: Unique identifier of the collector that sent the records
        """
        if not records:
            return

        try:
            message = ServerMetricsRecordsMessage(
                service_id=self.service_id,
                collector_id=collector_id,
                records=records,
                error=None,
            )

            await self.records_push_client.push(message)

        except Exception as e:
            self.error(f"Failed to send server metrics records: {e}")
            # Send error message to RecordsManager to track the failure
            try:
                error_message = ServerMetricsRecordsMessage(
                    service_id=self.service_id,
                    collector_id=collector_id,
                    records=[],
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

        Sends error ServerMetricsRecordsMessage to RecordsManager via message system.
        The message contains an empty records list and the error details.

        Args:
            error: ErrorDetails describing the collection error
            collector_id: Unique identifier of the collector that encountered the error
        """
        try:
            error_message = ServerMetricsRecordsMessage(
                service_id=self.service_id,
                collector_id=collector_id,
                records=[],
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
