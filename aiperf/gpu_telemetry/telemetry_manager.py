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
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )

        self._collectors: dict[str, TelemetryDataCollector] = {}

        # Normalize user_endpoints to always be a list
        user_endpoints = user_config.server_metrics_url
        if not user_endpoints:
            user_endpoints = []
        elif isinstance(user_endpoints, str):
            user_endpoints = [user_endpoints]

        if DEFAULT_DCGM_ENDPOINT not in user_endpoints:
            self._dcgm_endpoints = [DEFAULT_DCGM_ENDPOINT] + user_endpoints
        else:
            self._dcgm_endpoints = user_endpoints

        self._collection_interval = DEFAULT_COLLECTION_INTERVAL

    @on_init
    async def _initialize(self) -> None:
        """Initialize telemetry manager."""
        pass

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the telemetry collectors but don't start them yet."""

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
            await self._disable_telemetry_and_stop("no DCGM endpoints reachable")
            return

        reachable_endpoints = list(self._collectors)
        await self._send_telemetry_status(
            enabled=True,
            endpoints_tested=self._dcgm_endpoints,
            endpoints_reachable=reachable_endpoints,
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message) -> None:
        """Start all telemetry collectors."""

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
            await self._disable_telemetry_and_stop("all collectors failed to start")
            return

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Stop all telemetry collectors when profiling is cancelled."""
        await self._stop_all_collectors()

    @on_command(CommandType.SHUTDOWN)
    async def _handle_shutdown_command(self, message) -> None:
        """Handle shutdown command from SystemController."""
        await self._stop_all_collectors()

    @on_stop
    async def _telemetry_manager_stop(self) -> None:
        """Stop all telemetry collectors during service shutdown."""
        await self._stop_all_collectors()

    async def _disable_telemetry_and_stop(self, reason: str) -> None:
        """Disable telemetry by sending status update and stopping service.

        Args:
            reason: Human-readable reason for disabling telemetry
        """
        self.info(f"GPU telemetry disabled - {reason}")

        await self._send_telemetry_status(
            enabled=False,
            reason=reason,
            endpoints_tested=self._dcgm_endpoints,
            endpoints_reachable=[],
        )

        asyncio.create_task(self.stop())

    async def _stop_all_collectors(self) -> None:
        """Stop all telemetry collectors."""

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
        """Async callback for receiving telemetry records from collectors.

        Sends TelemetryRecordsMessage to RecordsManager via message system.
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
        """Async callback for receiving telemetry errors from collectors.

        Sends error TelemetryRecordsMessage to RecordsManager via message system.
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
        """Send telemetry status message directly to SystemController."""
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
