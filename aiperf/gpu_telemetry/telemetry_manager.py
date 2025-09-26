# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


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
        self.debug("Telemetry manager __init__")

        self._collectors: dict[str, TelemetryDataCollector] = {}

        # Always include default endpoint, plus any user-specified endpoints
        user_endpoints = user_config.server_metrics_url or []

        # Ensure default endpoint is always included (avoid duplicates)
        if DEFAULT_DCGM_ENDPOINT not in user_endpoints:
            self._dcgm_endpoints = [DEFAULT_DCGM_ENDPOINT] + user_endpoints
        else:
            self._dcgm_endpoints = user_endpoints

        self._collection_interval = DEFAULT_COLLECTION_INTERVAL

        # Important: Use warning level to ensure visibility during debugging
        self.info(
            f"GPU TELEMETRY MANAGER: Initialized with {len(self._dcgm_endpoints)} endpoints: {self._dcgm_endpoints}"
        )

    @on_init
    async def _initialize(self) -> None:
        """Initialize telemetry manager."""

        self.info("GPU TELEMETRY MANAGER: Initializing...")
        self.info(f"GPU TELEMETRY: Service ID: {self.service_id}")
        self.info(f"GPU TELEMETRY: Endpoints to monitor: {self._dcgm_endpoints}")

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the telemetry collectors but don't start them yet."""

        self.warning(
            f"GPU TELEMETRY MANAGER: PROFILE_CONFIGURE received for {self.service_id}"
        )
        self.warning(
            f"GPU TELEMETRY: Configuring {len(self._dcgm_endpoints)} DCGM endpoints: {self._dcgm_endpoints}"
        )

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
                self.info(f"Reachability result for {dcgm_url}: {is_reachable}")

                if is_reachable:
                    self._collectors[dcgm_url] = collector
                    reachable_count += 1
                    self.info(
                        f"GPU TELEMETRY MANAGER: ✅ DCGM endpoint REACHABLE: {dcgm_url}"
                    )
                else:
                    self.warning(f"❌ DCGM endpoint NOT reachable: {dcgm_url}")
            except Exception as e:
                self.error(
                    f"GPU TELEMETRY MANAGER: ❌ Exception testing {dcgm_url}: {e}"
                )
                import traceback

                self.error(f"Traceback: {traceback.format_exc()}")

        self.info("GPU TELEMETRY MANAGER: === CONFIGURATION SUMMARY ===")
        self.info(f"Total endpoints tested: {len(self._dcgm_endpoints)}")
        self.info(f"Reachable endpoints: {reachable_count}")
        self.info(f"Collectors created: {len(self._collectors)}")

        if reachable_count == 0:
            self.info("GPU telemetry disabled - no DCGM endpoints reachable")
            self.info(f"Tested endpoints: {self._dcgm_endpoints}")

            # Notify SystemController that telemetry is disabled
            await self._send_telemetry_status(
                enabled=False,
                reason="No DCGM endpoints reachable",
                endpoints_tested=self._dcgm_endpoints,
                endpoints_reachable=[],
            )

            # Clean shutdown to free resources
            await self.stop()
            return

        self.info(
            f"GPU TELEMETRY MANAGER: ✅ Telemetry manager configured with {reachable_count}/{len(self._dcgm_endpoints)} reachable endpoints"
        )
        for url in self._collectors:
            self.info(f"  - Active collector: {url}")

        # Notify SystemController that telemetry is enabled and will produce results
        reachable_endpoints = list(self._collectors)
        await self._send_telemetry_status(
            enabled=True,
            endpoints_tested=self._dcgm_endpoints,
            endpoints_reachable=reachable_endpoints,
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message) -> None:
        """Start all telemetry collectors."""

        self.info(
            f"=== PROFILE_START received for telemetry manager {self.service_id} ==="
        )

        if not self._collectors:
            self.warning("❌ No telemetry collectors configured, skipping start")
            self.warning(f"Original endpoints: {self._dcgm_endpoints}")
            return

        self.info(f"Starting {len(self._collectors)} telemetry collectors...")

        started_count = 0
        for dcgm_url, collector in self._collectors.items():
            self.info(f"Starting collector for: {dcgm_url}")
            try:
                # Verify reachability again before starting
                if await collector.is_url_reachable():
                    # Initialize collector before starting (required lifecycle)
                    self.info(f"Initializing collector for {dcgm_url}...")
                    await collector.initialize()

                    self.info(f"Starting collector for {dcgm_url}...")
                    await collector.start()
                    started_count += 1
                    self.info(f"✅ Started telemetry collection for {dcgm_url}")
                else:
                    self.warning(f"❌ Skipping unreachable endpoint: {dcgm_url}")
            except Exception as e:
                self.error(f"❌ Failed to start collector for {dcgm_url}: {e}")
                import traceback

                self.error(f"Traceback: {traceback.format_exc()}")

        self.info("=== STARTUP SUMMARY ===")
        self.info(
            f"✅ Started {started_count}/{len(self._collectors)} telemetry collectors"
        )

        if started_count == 0:
            self.warning("❌ No telemetry collectors successfully started!")

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Stop all telemetry collectors when profiling is cancelled."""

        self.warning(
            "GPU TELEMETRY MANAGER: Received PROFILE_CANCEL - stopping all collectors"
        )
        await self._stop_all_collectors()

    @on_command(CommandType.SHUTDOWN)
    async def _handle_shutdown_command(self, message) -> None:
        """Handle shutdown command from SystemController."""

        self.warning(
            "GPU TELEMETRY MANAGER: Received SHUTDOWN command - stopping service"
        )
        await self._stop_all_collectors()

    @on_stop
    async def _telemetry_manager_stop(self) -> None:
        """Stop all telemetry collectors during service shutdown."""

        self.warning(
            "GPU TELEMETRY MANAGER: Service stopping - cleaning up all collectors"
        )
        await self._stop_all_collectors()

    async def _stop_all_collectors(self) -> None:
        """Stop all telemetry collectors."""

        if not self._collectors:
            self.debug("No collectors to stop")
            return

        self.info(f"Stopping {len(self._collectors)} telemetry collectors...")

        for dcgm_url, collector in self._collectors.items():
            try:
                await collector.stop()
                self.info(f"✅ Stopped telemetry collection for {dcgm_url}")
            except Exception as e:
                self.error(f"❌ Failed to stop collector for {dcgm_url}: {e}")

        self.info("GPU TELEMETRY MANAGER: All collectors stopped")

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
            self.error(f"❌ Failed to send telemetry records: {e}")
            import traceback

            self.error(f"Traceback: {traceback.format_exc()}")

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
            self.debug(
                f"Sent telemetry status: enabled={enabled}, reason={reason}, tested={len(endpoints_tested or [])}, reachable={len(endpoints_reachable or [])}"
            )

        except Exception as e:
            self.error(f"Failed to send telemetry status message: {e}")


def main() -> None:
    """Main entry point for the telemetry manager."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(TelemetryManager)


if __name__ == "__main__":
    main()
