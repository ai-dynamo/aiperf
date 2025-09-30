# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommandType,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import on_command, on_init, on_stop
from aiperf.common.messages import (
    ProfileCancelCommand,
    ProfileConfigureCommand,
    TelemetryRecordsMessage,
    TelemetryStatusMessage,
)
from aiperf.common.models import ErrorDetails, TelemetryRecord
from aiperf.common.protocols import (
    ServiceProtocol,
)
from aiperf.gpu_telemetry.constants import (
    DEFAULT_COLLECTION_INTERVAL,
    DEFAULT_DCGM_ENDPOINT,
)
from aiperf.gpu_telemetry.telemetry_data_collector import TelemetryDataCollector

__all__ = ["TelemetryManager"]


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.TELEMETRY_MANAGER)
class TelemetryManager(BaseComponentService):
    """
    The TelemetryManager coordinates multiple TelemetryDataCollector instances
    to collect GPU telemetry from multiple DCGM endpoints and send unified
    TelemetryRecordsMessage to RecordsManager.

    This service:
    - Manages lifecycle of TelemetryDataCollector instances
    - Collects telemetry from multiple DCGM endpoints
    - Sends TelemetryRecordsMessage to RecordsManager via message system
    - Handles errors gracefully with ErrorDetails
    - Follows centralized architecture patterns
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
    ) -> None:
        """
        Initialize the TelemetryManager, register service metadata, and prepare internal collector state.
        
        Parameters:
            service_config (ServiceConfig): Configuration for the service runtime and framework integration.
            user_config (UserConfig): User-provided configuration; `server_metrics_url` is used to derive DCGM endpoints.
            service_id (str | None): Optional explicit service identifier; if omitted the base service will assign one.
        
        Side effects:
            - Calls the base service initializer.
            - Creates an internal mapping for telemetry collectors.
            - Builds the DCGM endpoints list, ensuring the default endpoint is present at the front.
            - Sets the telemetry collection interval to the module default.
        """
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )

        self._collectors: dict[str, TelemetryDataCollector] = {}

        user_endpoints = user_config.server_metrics_url or []

        if DEFAULT_DCGM_ENDPOINT not in user_endpoints:
            self._dcgm_endpoints = [DEFAULT_DCGM_ENDPOINT] + user_endpoints
        else:
            self._dcgm_endpoints = user_endpoints

        self._collection_interval = DEFAULT_COLLECTION_INTERVAL

    @on_init
    async def _initialize(self) -> None:
        """
        Perform any required initialization for the telemetry manager.
        
        This coroutine is a placeholder that currently performs no actions but is reserved for asynchronous initialization tasks needed by the service.
        """
        pass

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """
        Configure telemetry collectors for the configured DCGM endpoints without starting them.
        
        Creates a TelemetryDataCollector for each configured endpoint, tests reachability, registers only reachable collectors, and publishes a TelemetryStatusMessage that lists tested and reachable endpoints. If no endpoints are reachable, schedules the service to stop. This method does not start any collectors.
        """

        reachable_count = 0
        for _i, dcgm_url in enumerate(self._dcgm_endpoints):
            collector_id = f"collector_{dcgm_url.replace(':', '_').replace('/', '_')}"
            collector = TelemetryDataCollector(
                dcgm_url=dcgm_url,
                collection_interval=self._collection_interval,
                record_callback=self._on_telemetry_records,
                error_callback=self._on_telemetry_error,
                collector_id=collector_id,
            )

            try:
                is_reachable = await collector.is_url_reachable()
                if is_reachable:
                    self._collectors[dcgm_url] = collector
                    reachable_count += 1
            except Exception as e:
                self.error(f"Exception testing {dcgm_url}: {e}")

        if reachable_count == 0:
            self.info("GPU telemetry disabled - no DCGM endpoints reachable")

            await self._send_telemetry_status(
                enabled=False,
                reason="No DCGM endpoints reachable",
                endpoints_tested=self._dcgm_endpoints,
                endpoints_reachable=[],
            )

            asyncio.create_task(self.stop())
            return

        reachable_endpoints = list(self._collectors)
        await self._send_telemetry_status(
            enabled=True,
            endpoints_tested=self._dcgm_endpoints,
            endpoints_reachable=reachable_endpoints,
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message) -> None:
        """
        Start configured telemetry collectors that are reachable.
        
        Attempts to start each configured collector after verifying its endpoint is reachable. Logs an error for any collector that fails to start and logs a warning if no collectors were started.
        
        Parameters:
            message: The profile start message that triggered this action (not used by this handler).
        """

        if not self._collectors:
            return

        started_count = 0
        for dcgm_url, collector in self._collectors.items():
            try:
                if await collector.is_url_reachable():
                    await collector.initialize()
                    await collector.start()
                    started_count += 1
            except Exception as e:
                self.error(f"Failed to start collector for {dcgm_url}: {e}")

        if started_count == 0:
            self.warning("No telemetry collectors successfully started")

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """
        Stop all active telemetry collectors in response to a profiling cancellation.
        
        Parameters:
            message (ProfileCancelCommand): The incoming cancel command (unused); triggers stopping all collectors.
        """
        await self._stop_all_collectors()

    @on_command(CommandType.SHUTDOWN)
    async def _handle_shutdown_command(self, message) -> None:
        """
        Stop all telemetry collectors in response to a shutdown command.
        
        Parameters:
            message: The shutdown command message received from the SystemController (unused).
        """
        await self._stop_all_collectors()

    @on_stop
    async def _telemetry_manager_stop(self) -> None:
        """
        Stop all managed telemetry collectors and wait for each to finish shutting down.
        
        This is invoked during service shutdown to ensure every TelemetryDataCollector is cleanly stopped.
        """
        await self._stop_all_collectors()

    async def _stop_all_collectors(self) -> None:
        """
        Stop all managed telemetry collectors and await their shutdown.
        
        If no collectors are configured this returns immediately. Failures to stop individual collectors are logged and suppressed; the method does not raise for per-collector stop errors.
        """

        if not self._collectors:
            return

        for dcgm_url, collector in self._collectors.items():
            try:
                await collector.stop()
            except Exception as e:
                self.error(f"Failed to stop collector for {dcgm_url}: {e}")

    async def _on_telemetry_records(
        self, records: list[TelemetryRecord], collector_id: str
    ) -> None:
        """
        Handle telemetry records emitted by a collector and publish them as a TelemetryRecordsMessage.
        
        If `records` is empty the call is ignored. Otherwise constructs a TelemetryRecordsMessage containing this service's `service_id`, the provided `collector_id`, the `records`, and `error=None`, then publishes it. Any exceptions raised while publishing are logged.
        Parameters:
            records (list[TelemetryRecord]): Telemetry records produced by a collector.
            collector_id (str): Identifier of the collector that produced the records.
        """

        if not records:
            return

        try:
            message = TelemetryRecordsMessage(
                service_id=self.service_id,
                collector_id=collector_id,
                records=records,
                error=None,
            )

            await self.publish(message)

        except Exception as e:
            self.error(f"Failed to send telemetry records: {e}")

    async def _on_telemetry_error(self, error: ErrorDetails, collector_id: str) -> None:
        """
        Handle a telemetry error reported by a collector and publish it as a TelemetryRecordsMessage.
        
        Constructs a TelemetryRecordsMessage containing an empty records list and the provided error, then publishes it to the messaging system.
        
        Parameters:
        	error (ErrorDetails): Details of the telemetry error to forward.
        	collector_id (str): Identifier of the collector that reported the error.
        """

        try:
            error_message = TelemetryRecordsMessage(
                service_id=self.service_id,
                collector_id=collector_id,
                records=[],
                error=error,
            )

            await self.publish(error_message)

        except Exception as e:
            self.error(f"Failed to send telemetry error message: {e}")

    async def _send_telemetry_status(
        self,
        enabled: bool,
        reason: str | None = None,
        endpoints_tested: list[str] | None = None,
        endpoints_reachable: list[str] | None = None,
    ) -> None:
        """
        Publish a TelemetryStatusMessage describing telemetry availability and tested endpoints to the SystemController.
        
        Parameters:
            enabled (bool): Whether telemetry collection is currently enabled.
            reason (str | None): Optional human-readable explanation when telemetry is disabled or its status is noteworthy.
            endpoints_tested (list[str] | None): List of DCGM endpoint URLs that were tested for reachability (defaults to an empty list).
            endpoints_reachable (list[str] | None): List of DCGM endpoint URLs that were determined reachable (defaults to an empty list).
        """
        try:
            status_message = TelemetryStatusMessage(
                service_id=self.service_id,
                enabled=enabled,
                reason=reason,
                endpoints_tested=endpoints_tested or [],
                endpoints_reachable=endpoints_reachable or [],
            )

            await self.publish(status_message)

        except Exception as e:
            self.error(f"Failed to send telemetry status message: {e}")
