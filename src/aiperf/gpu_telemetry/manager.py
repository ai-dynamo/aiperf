# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import orjson

from aiperf.common.base_component_service import BaseComponentService
from aiperf.common.control_structs import Command, TelemetryStatus
from aiperf.common.enums import CommandType, MessageType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun
from aiperf.common.environment import Environment
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.common.hooks import on_command, on_init, on_message, on_stop
from aiperf.common.messages import ProcessTelemetryResultMessage
from aiperf.common.models import (
    ErrorDetails,
    ErrorDetailsCount,
    ErrorTrackingState,
    ProcessTelemetryResult,
    TelemetryRecord,
)
from aiperf.credit.messages import CreditPhaseStartMessage
from aiperf.gpu_telemetry.constants import PYNVML_SOURCE_IDENTIFIER
from aiperf.gpu_telemetry.dcgm_collector import DCGMTelemetryCollector
from aiperf.gpu_telemetry.protocols import (
    GPUTelemetryAccumulatorProtocol,
    GPUTelemetryCollectorProtocol,
    GPUTelemetryProcessorProtocol,
)
from aiperf.plugin import plugins
from aiperf.plugin.enums import (
    GPUTelemetryCollectorType,
    GPUTelemetryProcessorType,
    PluginType,
)

__all__ = ["GPUTelemetryManager"]


class GPUTelemetryManager(BaseComponentService):
    """Coordinates multiple TelemetryDataCollector instances for GPU telemetry collection.

    The GPUTelemetryManager coordinates multiple TelemetryDataCollector instances
    to collect GPU telemetry from multiple DCGM endpoints, fan out each record
    to locally-loaded GPU telemetry processors, and publish the accumulated
    ProcessTelemetryResultMessage when profiling completes.

    This service:
    - Manages lifecycle of TelemetryDataCollector instances
    - Collects telemetry from multiple DCGM endpoints
    - Runs all configured GPU telemetry processors (plugins registered under
      GPU_TELEMETRY_PROCESSOR) locally on each collected record
    - Accumulates results and publishes ProcessTelemetryResultMessage on
      PROFILE_COMPLETE
    - Handles errors gracefully with ErrorDetails

    Args:
        service_config: Service-level configuration (logging, communication, etc.)
        user_config: User-provided configuration including gpu_telemetry list
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

        self._collectors: dict[str, GPUTelemetryCollectorProtocol] = {}
        self._shutdown_task: asyncio.Task[None] | None = None

        self._telemetry_disabled = not run.cfg.gpu_telemetry.enabled
        self._user_explicitly_configured_telemetry = bool(run.cfg.gpu_telemetry.urls)

        # Store the collector type (DCGM or PYNVML)
        self._collector_type = GPUTelemetryCollectorType.DCGM

        user_endpoints = run.cfg.gpu_telemetry.urls or []
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

        self._processors: list[GPUTelemetryProcessorProtocol] = []
        self._accumulator: GPUTelemetryAccumulatorProtocol | None = None
        self._error_state = ErrorTrackingState()
        self._result_published: bool = False

        for entry in plugins.iter_entries(PluginType.GPU_TELEMETRY_PROCESSOR):
            try:
                ProcessorClass = plugins.get_class(
                    PluginType.GPU_TELEMETRY_PROCESSOR, entry.name
                )
                processor = ProcessorClass(
                    service_id=self.service_id,
                    run=self.run,
                    pub_client=self.pub_client,
                )
                self.attach_child_lifecycle(processor)
                self._processors.append(processor)
                if entry.name == GPUTelemetryProcessorType.GPU_TELEMETRY_ACCUMULATOR:
                    self._accumulator = processor
                self.debug(
                    f"Created GPU telemetry processor: {entry.name}: "
                    f"{processor.__class__.__name__}"
                )
            except PostProcessorDisabled:
                self.debug(
                    f"GPU telemetry processor {entry.name} is disabled and will not be used"
                )
            except Exception as e:
                self.error(
                    f"Failed to create GPU telemetry processor {entry.name}: {e}"
                )

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
    async def _profile_configure_command(self, message: Command) -> None:
        """Configure the telemetry collectors but don't start them yet.

        Creates collector instances based on configured type (DCGM or PYNVML),
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

        # Phase 1: Test reachability for all endpoints
        if self._collector_type == GPUTelemetryCollectorType.PYNVML:
            await self._configure_pynvml_collector()
        else:
            await self._configure_dcgm_collectors()

    async def _configure_pynvml_collector(self) -> None:
        """Configure a single PyNVML collector for local GPU monitoring."""
        self.debug("GPU Telemetry: Configuring pynvml collector")

        try:
            CollectorClass = plugins.get_class(
                PluginType.GPU_TELEMETRY_COLLECTOR,
                GPUTelemetryCollectorType.PYNVML,
            )

            collector_id = "pynvml_collector"
            collector = CollectorClass(
                collection_interval=self._collection_interval,
                record_callback=self._on_telemetry_records,
                error_callback=self._on_telemetry_error,
                collector_id=collector_id,
            )

            is_available = await collector.is_url_reachable()
            if is_available:
                self._collectors[PYNVML_SOURCE_IDENTIFIER] = collector
                self.debug("GPU Telemetry: pynvml collector configured successfully")
                await self._send_telemetry_status(
                    enabled=True,
                    reason=None,
                    endpoints_configured=[PYNVML_SOURCE_IDENTIFIER],
                    endpoints_reachable=[PYNVML_SOURCE_IDENTIFIER],
                )
            else:
                self.warning("GPU Telemetry: pynvml not available or no GPUs found")
                await self._send_telemetry_status(
                    enabled=False,
                    reason="pynvml not available or no GPUs found",
                    endpoints_configured=[PYNVML_SOURCE_IDENTIFIER],
                    endpoints_reachable=[],
                )
        except RuntimeError as e:
            # pynvml package not installed
            self.error(f"GPU Telemetry: {e}")
            await self._send_telemetry_status(
                enabled=False,
                reason=str(e),
                endpoints_configured=[],
                endpoints_reachable=[],
            )
        except Exception as e:  # noqa: BLE001 - fault-tolerant telemetry
            self.error(f"GPU Telemetry: Failed to configure pynvml collector: {e}")
            await self._send_telemetry_status(
                enabled=False,
                reason=f"pynvml configuration failed: {e}",
                endpoints_configured=[],
                endpoints_reachable=[],
            )

    async def _configure_dcgm_collectors(self) -> None:
        """Configure DCGM collectors for HTTP-based GPU telemetry."""
        for dcgm_url in self._dcgm_endpoints:
            self.debug(f"GPU Telemetry: Testing reachability of {dcgm_url}")
            collector_id = f"collector_{dcgm_url.replace(':', '_').replace('/', '_')}"
            collector = DCGMTelemetryCollector(
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
                    self.debug(f"GPU Telemetry: DCGM endpoint {dcgm_url} is reachable")
                else:
                    self.debug(
                        f"GPU Telemetry: DCGM endpoint {dcgm_url} is not reachable"
                    )
            except Exception as e:
                self.error(f"GPU Telemetry: Exception testing {dcgm_url}: {e}")

        # Determine which defaults are reachable for display filtering
        reachable_endpoints = list(self._collectors.keys())
        reachable_defaults = [
            ep
            for ep in Environment.GPU.DEFAULT_DCGM_ENDPOINTS
            if ep in reachable_endpoints
        ]
        endpoints_for_display = self._compute_endpoints_for_display(reachable_defaults)

        if not self._collectors:
            # Telemetry manager shutdown occurs in _on_start_profiling to prevent hang
            await self._send_telemetry_status(
                enabled=False,
                reason="no DCGM endpoints reachable",
                endpoints_configured=endpoints_for_display,
                endpoints_reachable=[],
            )
            return

        # Phase 2: Capture baseline metrics before profiling starts
        self.info("GPU Telemetry: Capturing baseline metrics...")
        for dcgm_url, collector in self._collectors.items():
            try:
                await collector.initialize()
                await collector.collect_and_process_metrics()
                self.debug(f"GPU Telemetry: Captured baseline from {dcgm_url}")
            except Exception as e:
                self.warning(
                    f"GPU Telemetry: Failed to capture baseline from {dcgm_url}: {e}"
                )

        await self._send_telemetry_status(
            enabled=True,
            reason=None,
            endpoints_configured=endpoints_for_display,
            endpoints_reachable=reachable_endpoints,
        )

    @on_command(CommandType.PROFILE_START)
    async def _on_start_profiling(self, message: Command) -> None:
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

        self.info("GPU Telemetry: Capturing boundary metrics at profiling start...")
        for source_url, collector in list(self._collectors.items()):
            try:
                await collector.collect_and_process_metrics()
                self.debug(
                    lambda url=source_url: f"GPU Telemetry: Captured boundary state from {url}"
                )
            except Exception as e:
                self.warning(
                    f"GPU Telemetry: Failed to capture boundary state from {source_url}: {e}"
                )

    @on_command(CommandType.PROFILE_CANCEL)
    async def _handle_profile_cancel_command(self, message: Command) -> None:
        """Stop all telemetry collectors when profiling is cancelled.

        Called when user cancels profiling or an error occurs during profiling.
        Stops all running collectors gracefully and cleans up resources.

        Args:
            message: Profile cancel command from SystemController
        """
        await self._stop_all_collectors()

    @on_command(CommandType.PROFILE_COMPLETE)
    async def _handle_profile_complete_command(self, message: Command) -> None:
        """Trigger final scrape when profiling completes.

        Ensures GPU telemetry captures final state for accurate counter deltas.
        This final scrape provides the end-point values needed for metrics like
        energy_consumption which are computed as (final - baseline).

        Args:
            message: Profile complete command from SystemController
        """
        if self._collectors:
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
        else:
            self.debug("GPU Telemetry: Already stopped, skipping final scrape")

        if self._result_published:
            self.debug(
                "GPU Telemetry: PROFILE_COMPLETE re-entry, result already published"
            )
            return

        start_ns: int | None = None
        if message.payload:
            try:
                parsed = orjson.loads(message.payload)
                start_ns = parsed.get("start_ns")
            except Exception as e:
                self.warning(
                    f"Failed to parse PROFILE_COMPLETE payload ({e!r}); "
                    "using start_ns=0 (full-history export)"
                )

        await self._publish_telemetry_result(start_ns=start_ns)

    async def _publish_telemetry_result(self, start_ns: int | None) -> None:
        """Publish accumulated telemetry results to subscribers."""
        self._result_published = True
        error_summary = [
            ErrorDetailsCount(error_details=err, count=count)
            for err, count in self._error_state.error_counts.items()
        ]
        if not self._accumulator:
            await self.publish(
                ProcessTelemetryResultMessage(
                    service_id=self.service_id,
                    telemetry_result=ProcessTelemetryResult(results=None),
                )
            )
            return
        export_data = self._accumulator.export_results(
            start_ns=start_ns or 0,
            error_summary=error_summary,
        )
        await self.publish(
            ProcessTelemetryResultMessage(
                service_id=self.service_id,
                telemetry_result=ProcessTelemetryResult(results=export_data),
            )
        )

    @on_command(CommandType.START_REALTIME_TELEMETRY)
    async def _on_start_realtime_telemetry_command(self, message: Command) -> None:
        """Start the realtime telemetry background task on the local accumulator."""
        if self._accumulator:
            self._accumulator.start_realtime_telemetry()
        else:
            self.error(
                "GPU telemetry accumulator not found, cannot start realtime telemetry"
            )

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

        Fans out each record to locally-loaded GPU telemetry processors.
        Empty record lists are ignored.

        Args:
            records: List of TelemetryRecord objects from a collector
            collector_id: Unique identifier of the collector that sent the records
        """

        if not records:
            return

        errors = await asyncio.gather(
            *[
                processor.process_telemetry_record(record)
                for processor in self._processors
                for record in records
            ],
            return_exceptions=True,
        )
        for error in errors:
            if isinstance(error, BaseException):
                self.exception(f"Failed to process telemetry record: {error!r}")
                self._error_state.error_counts[ErrorDetails.from_exception(error)] += 1

    async def _on_telemetry_error(self, error: ErrorDetails, collector_id: str) -> None:
        """Async callback for receiving telemetry errors from collectors.

        Records the error against the local error state so it is reflected in the
        published ProcessTelemetryResultMessage on PROFILE_COMPLETE.

        Args:
            error: ErrorDetails describing the collection error
            collector_id: Unique identifier of the collector that encountered the error
        """
        self._error_state.error_counts[error] += 1

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
            await self.control_client.send(
                TelemetryStatus(
                    sid=self.service_id,
                    enabled=enabled,
                    reason=reason,
                    endpoints_configured=tuple(endpoints_configured or []),
                    endpoints_reachable=tuple(endpoints_reachable or []),
                )
            )
        except Exception as e:
            self.error(f"Failed to send telemetry status message: {e}")
