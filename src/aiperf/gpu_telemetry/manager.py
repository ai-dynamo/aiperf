# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from typing import Any

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import CommAddress, CommandType
from aiperf.common.environment import Environment
from aiperf.common.hooks import on_command, on_init, on_stop
from aiperf.common.messages import (
    ProfileCancelCommand,
    ProfileCompleteCommand,
    ProfileConfigureCommand,
    TelemetryRecordsMessage,
    TelemetryStatusMessage,
)
from aiperf.common.models import ErrorDetails, TelemetryRecord
from aiperf.common.protocols import PushClientProtocol
from aiperf.gpu_telemetry.protocols import GPUTelemetryCollectorProtocol
from aiperf.plugin import plugins
from aiperf.plugin.enums import GPUTelemetryCollectorType, PluginType

__all__ = ["GPUTelemetryManager"]


@dataclass(slots=True)
class _CollectorCandidate:
    collector_type: GPUTelemetryCollectorType
    collector_id: str
    kwargs: dict[str, Any]


class GPUTelemetryManager(BaseComponentService):
    """Coordinates multiple TelemetryDataCollector instances for GPU telemetry collection.

    The GPUTelemetryManager coordinates multiple TelemetryDataCollector instances
    to collect GPU telemetry from multiple DCGM endpoints and send unified
    TelemetryRecordsMessage to RecordsManager.

    This service:
    - Manages lifecycle of TelemetryDataCollector instances
    - Collects telemetry from multiple DCGM endpoints
    - Sends TelemetryRecordsMessage to RecordsManager via message system
    - Handles errors gracefully with ErrorDetails
    - Follows centralized architecture patterns

    Args:
        service_config: Service-level configuration (logging, communication, etc.)
        user_config: User-provided configuration including gpu_telemetry list
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

        self._collectors: dict[str, GPUTelemetryCollectorProtocol] = {}
        self._collector_id_to_url: dict[str, str] = {}

        self._telemetry_disabled = user_config.gpu_telemetry_disabled
        self._user_explicitly_configured_telemetry = (
            user_config.gpu_telemetry is not None and not self._telemetry_disabled
        )

        self._collector_type = user_config.gpu_telemetry_collector_type

        # DCGM-specific endpoint configuration
        user_endpoints = user_config.gpu_telemetry_urls or []
        if isinstance(user_endpoints, str):
            user_endpoints = [user_endpoints]

        valid_endpoints = [self._normalize_dcgm_url(url) for url in user_endpoints]

        # Store user-provided endpoints separately for display filtering (excluding auto-inserted defaults)
        self._user_provided_endpoints = [
            ep
            for ep in valid_endpoints
            if ep not in Environment.GPU.DEFAULT_DCGM_ENDPOINTS
        ]

        # Combine defaults + user endpoints, preserving order and removing duplicates
        self._dcgm_endpoints = list(
            dict.fromkeys(
                list(Environment.GPU.DEFAULT_DCGM_ENDPOINTS)
                + self._user_provided_endpoints
            )
        )

        self._collection_interval = Environment.GPU.COLLECTION_INTERVAL

    @staticmethod
    def _normalize_dcgm_url(url: str) -> str:
        """Ensure DCGM URL ends with /metrics endpoint.

        Args:
            url: Base URL or full metrics URL

        Returns:
            str: URL ending with /metrics
        """
        url = url.rstrip("/")
        if not url.endswith("/metrics"):
            url = f"{url}/metrics"
        return url

    def _compute_endpoints_for_display(
        self, reachable_defaults: list[str]
    ) -> list[str]:
        """Compute which DCGM endpoints should be displayed to the user.

        Filters endpoints for clean console output based on user configuration
        and reachability. This intentional filtering prevents cluttering the UI
        with unreachable default endpoints that the user didn't explicitly configure.

        Args:
            reachable_defaults: List of default DCGM endpoints that are reachable

        Returns:
            List of endpoint URLs to display in console/export output:
            - reachable_defaults if any defaults are reachable
            - user_provided_endpoints + reachable_defaults if custom endpoints and defaults reachable
            - user_provided_endpoints if user configured but no defaults reachable
            - Empty list if no reachable defaults and user did not configure telemetry
        """
        if reachable_defaults and self._user_provided_endpoints:
            return list(self._user_provided_endpoints) + reachable_defaults
        elif reachable_defaults:
            return reachable_defaults
        elif self._user_provided_endpoints:
            return self._user_provided_endpoints
        return []

    @on_init
    async def _initialize(self) -> None:
        """Initialize telemetry manager.

        Called automatically during service startup via @on_init hook.
        Actual collector initialization happens in _profile_configure_command
        after configuration is received from SystemController.
        """
        pass

    @on_command(CommandType.PROFILE_CONFIGURE)
    async def _profile_configure_command(
        self, message: ProfileConfigureCommand
    ) -> None:
        """Configure the telemetry collectors but don't start them yet.

        Creates collector instances based on the configured collector type,
        tests reachability, and sends status message to RecordsManager.
        If no collectors can be created, disables telemetry and stops the service.

        Args:
            message: Profile configuration command from SystemController
        """
        if self._telemetry_disabled:
            await self._send_telemetry_status(
                enabled=False,
                reason="disabled via --no-gpu-telemetry",
                endpoints_configured=[],
                endpoints_reachable=[],
            )
            return

        self._collectors.clear()
        self._collector_id_to_url.clear()

        candidates = self._collector_candidates()
        configured_sources, failure_reason = await self._configure_reachable_collectors(
            candidates
        )
        await self._send_configure_status(configured_sources, failure_reason)

    def _collector_candidates(self) -> list[_CollectorCandidate]:
        collector_name = str(self._collector_type)
        if plugins.get_gpu_telemetry_collector_metadata(self._collector_type).is_local:
            return [
                _CollectorCandidate(
                    collector_type=self._collector_type,
                    collector_id=f"{collector_name}_collector",
                    kwargs={},
                )
            ]
        return [
            _CollectorCandidate(
                collector_type=self._collector_type,
                collector_id=f"collector_{dcgm_url.replace(':', '_').replace('/', '_')}",
                kwargs={"dcgm_url": dcgm_url},
            )
            for dcgm_url in self._dcgm_endpoints
        ]

    async def _configure_reachable_collectors(
        self, candidates: list[_CollectorCandidate]
    ) -> tuple[list[str], str | None]:
        configured_sources: list[str] = []
        failure_reason: str | None = None
        for candidate in candidates:
            collector_name = str(candidate.collector_type)
            self.debug(f"GPU Telemetry: Configuring {collector_name} collector")
            try:
                CollectorClass = plugins.get_class(
                    PluginType.GPU_TELEMETRY_COLLECTOR,
                    candidate.collector_type,
                )
                collector = CollectorClass(
                    **candidate.kwargs,
                    collection_interval=self._collection_interval,
                    record_callback=self._on_telemetry_records,
                    error_callback=self._on_telemetry_error,
                    collector_id=candidate.collector_id,
                )
                source_identifier = collector.endpoint_url
                configured_sources.append(source_identifier)
                is_reachable = await collector.is_url_reachable()
                if not is_reachable:
                    self.warning(f"GPU Telemetry: {source_identifier} is not reachable")
                    continue

                self._collectors[source_identifier] = collector
                self._collector_id_to_url[candidate.collector_id] = source_identifier
                self.debug(f"GPU Telemetry: {source_identifier} is reachable")
                baseline_failure_reason = await self._capture_collector_baseline(
                    collector,
                    candidate.collector_id,
                    source_identifier,
                )
                if baseline_failure_reason is not None:
                    failure_reason = baseline_failure_reason
            except RuntimeError as e:
                failure_reason = str(e)
                self.error(f"GPU Telemetry: {e}")
            except Exception as e:  # noqa: BLE001 - fault-tolerant telemetry
                failure_reason = f"{collector_name} configuration failed: {e}"
                self.error(
                    f"GPU Telemetry: Failed to configure {collector_name} collector: {e}"
                )
        return configured_sources, failure_reason

    async def _capture_collector_baseline(
        self,
        collector: GPUTelemetryCollectorProtocol,
        collector_id: str,
        source_identifier: str,
    ) -> str | None:
        self.info(f"GPU Telemetry: Capturing baseline metrics from {source_identifier}")
        try:
            await collector.initialize()
        except (Exception, asyncio.CancelledError) as e:  # noqa: BLE001
            self.warning(
                f"GPU Telemetry: Failed to initialize {source_identifier} during "
                f"baseline capture, disabling collector: {e!r}"
            )
            self._collectors.pop(source_identifier, None)
            self._collector_id_to_url.pop(collector_id, None)
            return f"{source_identifier} initialization failed: {e}"

        try:
            await collector.collect_and_process_metrics()
            self.debug(f"GPU Telemetry: Captured baseline from {source_identifier}")
        except Exception as e:  # noqa: BLE001 - baseline scrape best-effort
            self.warning(
                f"GPU Telemetry: Failed to capture baseline from {source_identifier} "
                f"(collector remains enabled): {e}"
            )
        return None

    async def _send_configure_status(
        self, configured_sources: list[str], failure_reason: str | None
    ) -> None:
        reachable_endpoints = list(self._collectors.keys())
        reachable_defaults = [
            ep
            for ep in Environment.GPU.DEFAULT_DCGM_ENDPOINTS
            if ep in reachable_endpoints
        ]
        is_local = plugins.get_gpu_telemetry_collector_metadata(
            self._collector_type
        ).is_local
        endpoints_for_display = (
            configured_sources
            if is_local
            else self._compute_endpoints_for_display(reachable_defaults)
        )

        if not self._collectors:
            reason = failure_reason or (
                f"{self._collector_type} not available or no GPUs found"
                if is_local
                else "no DCGM endpoints reachable"
            )
            await self._send_telemetry_status(
                enabled=False,
                reason=reason,
                endpoints_configured=endpoints_for_display,
                endpoints_reachable=[],
            )
            return

        await self._send_telemetry_status(
            enabled=True,
            reason=None,
            endpoints_configured=endpoints_for_display,
            endpoints_reachable=reachable_endpoints,
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message) -> None:
        """Start all telemetry collectors.

        Initializes and starts each configured collector.
        If no collectors start successfully, sends disabled status to SystemController.

        Args:
            message: Profile start command from SystemController
        """
        if not self._collectors:
            # Telemetry disabled status already sent in _profile_configure_command, only shutdown here
            self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
            return

        started_count = 0
        for source_url, collector in self._collectors.items():
            try:
                await collector.initialize()
                await collector.start()
                started_count += 1
            except Exception as e:  # noqa: BLE001 - fault-tolerant telemetry
                self.error(f"Failed to start collector for {source_url}: {e}")

        if started_count == 0:
            self.warning("No GPU telemetry collectors successfully started")
            await self._send_telemetry_status(
                enabled=False,
                reason="all collectors failed to start",
                endpoints_configured=self._compute_endpoints_for_display([]),
                endpoints_reachable=[],
            )
            self._shutdown_task = asyncio.create_task(self._delayed_shutdown())
            return

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(
        self, message: ProfileCancelCommand
    ) -> None:
        """Stop all telemetry collectors when profiling is cancelled.

        Called when user cancels profiling or an error occurs during profiling.
        Stops all running collectors gracefully and cleans up resources.

        Args:
            message: Profile cancel command from SystemController
        """
        await self._stop_all_collectors()

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_command(
        self, message: ProfileCompleteCommand
    ) -> None:
        """Trigger final scrape when profiling completes.

        Ensures GPU telemetry captures final state for accurate counter deltas.
        This final scrape provides the end-point values needed for metrics like
        energy_consumption which are computed as (final - baseline).

        Args:
            message: Profile complete command from SystemController
        """
        if not self._collectors:
            self.debug("GPU Telemetry: Already stopped, skipping final scrape")
            return

        self.info("GPU Telemetry: Profiling complete, capturing final metrics...")

        for dcgm_url, collector in list(self._collectors.items()):
            try:
                await collector.collect_and_process_metrics()
                self.debug(f"GPU Telemetry: Captured final state from {dcgm_url}")
            except Exception as e:
                self.warning(
                    f"GPU Telemetry: Failed to capture final state from {dcgm_url}: {e}"
                )

        await self._stop_all_collectors()

    @on_stop
    async def _telemetry_manager_stop(self) -> None:
        """Stop all telemetry collectors during service shutdown.

        Called automatically by BaseComponentService lifecycle management via @on_stop hook.
        Ensures all collectors are properly stopped and cleaned up even if shutdown
        command was not received.
        """
        await self._stop_all_collectors()

    async def _delayed_shutdown(self) -> None:
        """Shutdown service after a delay to allow command response to be sent.

        Waits before calling stop() to ensure the command response
        has time to be published and transmitted to the SystemController.
        """
        await asyncio.sleep(Environment.GPU.SHUTDOWN_DELAY)
        await asyncio.shield(self.stop())

    async def _stop_all_collectors(self) -> None:
        """Stop all telemetry collectors.

        Attempts to stop each collector gracefully, logging errors but continuing with
        remaining collectors to ensure all resources are released. Does nothing if no
        collectors are configured.

        Errors during individual collector shutdown do not prevent other collectors
        from being stopped.
        """

        if not self._collectors:
            return

        for source_url, collector in self._collectors.items():
            try:
                await collector.stop()
            except Exception as e:  # noqa: BLE001 - fault-tolerant telemetry
                self.error(f"Failed to stop collector for {source_url}: {e}")

    async def _on_telemetry_records(
        self, records: list[TelemetryRecord], collector_id: str
    ) -> None:
        """Async callback for receiving telemetry records from collectors.

        Sends TelemetryRecordsMessage to RecordsManager via message system.
        Empty record lists are ignored.

        Args:
            records: List of TelemetryRecord objects from a collector
            collector_id: Unique identifier of the collector that sent the records
        """

        if not records:
            return

        try:
            dcgm_url = self._collector_id_to_url.get(collector_id, "")
            message = TelemetryRecordsMessage(
                service_id=self.service_id,
                collector_id=collector_id,
                dcgm_url=dcgm_url,
                records=records,
                error=None,
            )

            await self.records_push_client.push(message)

        except Exception as e:
            self.error(f"Failed to send telemetry records: {e}")

    async def _on_telemetry_error(self, error: ErrorDetails, collector_id: str) -> None:
        """Async callback for receiving telemetry errors from collectors.

        Sends error TelemetryRecordsMessage to RecordsManager via message system.
        The message contains an empty records list and the error details.

        Args:
            error: ErrorDetails describing the collection error
            collector_id: Unique identifier of the collector that encountered the error
        """

        try:
            dcgm_url = self._collector_id_to_url.get(collector_id, "")
            error_message = TelemetryRecordsMessage(
                service_id=self.service_id,
                collector_id=collector_id,
                dcgm_url=dcgm_url,
                records=[],
                error=error,
            )

            await self.records_push_client.push(error_message)

        except Exception as e:
            self.error(f"Failed to send telemetry error message: {e}")

    async def _send_telemetry_status(
        self,
        enabled: bool,
        reason: str | None = None,
        endpoints_configured: list[str] | None = None,
        endpoints_reachable: list[str] | None = None,
    ) -> None:
        """Send telemetry status message to SystemController.

        Publishes TelemetryStatusMessage to inform SystemController about telemetry
        availability and endpoint reachability. Used during configuration phase and
        when telemetry is disabled due to errors.

        Args:
            enabled: Whether telemetry collection is enabled/available
            reason: Optional human-readable reason for status (e.g., "no DCGM endpoints reachable")
            endpoints_configured: List of DCGM endpoint URLs in configured scope for display
            endpoints_reachable: List of DCGM endpoint URLs that are accessible
        """
        try:
            status_message = TelemetryStatusMessage(
                service_id=self.service_id,
                enabled=enabled,
                reason=reason,
                endpoints_configured=endpoints_configured or [],
                endpoints_reachable=endpoints_reachable or [],
            )

            await self.publish(status_message)

        except Exception as e:
            self.error(f"Failed to send telemetry status message: {e}")
