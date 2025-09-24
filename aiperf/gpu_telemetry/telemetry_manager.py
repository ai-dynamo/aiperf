# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import (
    CommAddress,
    CommandType,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import on_command, on_init, on_stop
from aiperf.common.messages import (
    ProfileCancelCommand,
    ProfileConfigureCommand,
    TelemetryRecordsMessage,
)
from aiperf.common.mixins import PushClientMixin
from aiperf.common.models import ErrorDetails, TelemetryRecord
from aiperf.common.protocols import (
    ServiceProtocol,
)
from aiperf.gpu_telemetry.constants import (
    DEFAULT_COLLECTION_INTERVAL,
)
from aiperf.gpu_telemetry.telemetry_data_collector import TelemetryDataCollector


@implements_protocol(ServiceProtocol)
@ServiceFactory.register(ServiceType.TELEMETRY_MANAGER)
class TelemetryManager(PushClientMixin, BaseComponentService):
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
            push_client_address=CommAddress.RECORDS,
            push_client_bind=False,
        )
        self.debug("Telemetry manager __init__")

        self._collectors: dict[str, TelemetryDataCollector] = {}
        self._dcgm_endpoints = user_config.server_metrics_url
        self._collection_interval = DEFAULT_COLLECTION_INTERVAL

    @on_init
    async def _initialize(self) -> None:
        """Initialize telemetry manager."""

        self.debug("Initializing telemetry manager")

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the telemetry collectors but don't start them yet."""

        self.info(f"Configuring telemetry manager for {self.service_id}")

        reachable_count = 0
        for dcgm_url in self._dcgm_endpoints:
            collector_id = f"collector_{dcgm_url.replace(':', '_').replace('/', '_')}"

            collector = TelemetryDataCollector(
                dcgm_url=dcgm_url,
                collection_interval=self._collection_interval,
                record_callback=self._on_telemetry_records,
                error_callback=self._on_telemetry_error,
                collector_id=collector_id,
            )

            if await collector.is_url_reachable():
                self._collectors[dcgm_url] = collector
                reachable_count += 1
                self.info(f"DCGM endpoint reachable: {dcgm_url}")
            else:
                self.warning(f"DCGM endpoint not reachable: {dcgm_url}")

        if reachable_count == 0:
            self.warning(
                "No DCGM endpoints are reachable. Telemetry collection will be disabled."
            )
            # Graceful shutdown - this is not an error condition
            return

        self.info(
            f"Telemetry manager configured with {reachable_count}/{len(self._dcgm_endpoints)} reachable endpoints"
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message) -> None:
        """Start all telemetry collectors."""

        if not self._collectors:
            self.debug("No telemetry collectors configured, skipping start")
            return

        self.debug("Starting telemetry collection")

        started_count = 0
        for dcgm_url, collector in self._collectors.items():
            try:
                if await collector.is_url_reachable():
                    await collector.start()
                    started_count += 1
                    self.info(f"Started telemetry collection for {dcgm_url}")
                else:
                    self.warning(f"Skipping unreachable endpoint: {dcgm_url}")
            except Exception as e:
                self.error(f"Failed to start collector for {dcgm_url}: {e}")

        self.info(
            f"Started {started_count}/{len(self._collectors)} telemetry collectors"
        )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Stop all telemetry collectors."""

        self.debug(lambda: f"Received profile cancel command: {message}")
        await self._stop_all_collectors()

    @on_stop
    async def _telemetry_manager_stop(self) -> None:
        """Stop all telemetry collectors."""

        self.debug("Stopping telemetry manager")
        await self._stop_all_collectors()

    async def _stop_all_collectors(self) -> None:
        """Stop all telemetry collectors."""

        for dcgm_url, collector in self._collectors.items():
            try:
                await collector.stop()
                self.info(f"Stopped telemetry collection for {dcgm_url}")
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

            await self.push_client.push(message)

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

            await self.push_client.push(error_message)

        except Exception as e:
            self.error(f"Failed to send telemetry error message: {e}")


def main() -> None:
    """Main entry point for the telemetry manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TelemetryManager)


if __name__ == "__main__":
    main()
