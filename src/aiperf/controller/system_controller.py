# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import sys
import time
from typing import cast

from rich.console import Console
from rich.panel import Panel

from aiperf.cli_utils import (
    print_developer_mode_warning,
    warn_osl_without_ignore_eos,
)
from aiperf.common.base_service import BaseService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    MessageType,
    ServiceRegistrationStatus,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.hooks import on_command, on_init, on_message, on_start, on_stop
from aiperf.common.logging import cleanup_global_log_queue, get_global_log_queue
from aiperf.common.messages import (
    CommandErrorResponse,
    CommandResponse,
    CommandSuccessResponse,
    HeartbeatMessage,
    ProcessRecordsResultMessage,
    ProcessServerMetricsResultMessage,
    ProcessTelemetryResultMessage,
    ProfileCancelCommand,
    ProfileConfigureCommand,
    ProfileStartCommand,
    RealtimeMetricsCommand,
    RegisterServiceCommand,
    ServerMetricsStatusMessage,
    ShutdownCommand,
    ShutdownWorkersCommand,
    SpawnWorkersCommand,
    StatusMessage,
    TelemetryStatusMessage,
)
from aiperf.common.models import (
    ErrorDetails,
    ProcessRecordsResult,
    ServiceRunInfo,
)
from aiperf.common.models.error_models import ExitErrorInfo
from aiperf.common.models.export_models import TelemetryExportData
from aiperf.common.models.server_metrics_models import ServerMetricsResults
from aiperf.common.types import ServiceTypeT
from aiperf.controller.controller_utils import print_exit_errors
from aiperf.controller.protocols import ServiceManagerProtocol
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.controller.system_mixins import SignalHandlerMixin
from aiperf.credit.messages import CreditsCompleteMessage
from aiperf.exporters.exporter_manager import ExporterManager
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceType, UIType
from aiperf.ui.protocols import AIPerfUIProtocol


class SystemController(SignalHandlerMixin, BaseService):
    """System Controller service.

    This service is responsible for managing the lifecycle of all other services.
    It will start, stop, and configure all other services.
    """

    def __init__(
        self,
        user_config: UserConfig,
        service_config: ServiceConfig,
        service_id: str | None = None,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
        )
        self.debug("Creating System Controller")
        if Environment.DEV.MODE:
            # Print a warning message to the console if developer mode is enabled, once at load time
            print_developer_mode_warning()

        # EOS may cause server to stop early, producing misleading OSL results
        if self._should_warn_osl_without_ignore_eos():
            warn_osl_without_ignore_eos()

        self._was_cancelled = False
        # List of required service types, in no particular order
        # These are services that must be running before the system controller can start profiling
        self.required_services: dict[ServiceTypeT, int] = {
            ServiceType.DATASET_MANAGER: 1,
            ServiceType.TIMING_MANAGER: 1,
            ServiceType.WORKER_MANAGER: 1,
            ServiceType.RECORDS_MANAGER: 1,
        }
        if self.service_config.record_processor_service_count is not None:
            self.required_services[ServiceType.RECORD_PROCESSOR] = (
                self.service_config.record_processor_service_count
            )
            self.scale_record_processors_with_workers = False
        else:
            self.scale_record_processors_with_workers = True

        self.proxy_manager: ProxyManager = ProxyManager(
            service_config=self.service_config
        )
        ServiceManagerClass = plugins.get_class(
            PluginType.SERVICE_MANAGER, self.service_config.service_run_type
        )

        using_dashboard = self.service_config.ui_type == UIType.DASHBOARD
        log_queue = get_global_log_queue() if using_dashboard else None

        self.service_manager: ServiceManagerProtocol = ServiceManagerClass(
            required_services=self.required_services,
            user_config=self.user_config,
            service_config=self.service_config,
            log_queue=log_queue,
        )
        UIClass = plugins.get_class(PluginType.UI, self.service_config.ui_type)
        self.ui: AIPerfUIProtocol = UIClass(
            service_config=self.service_config,
            user_config=self.user_config,
            log_queue=log_queue,
            controller=self,
        )
        self.attach_child_lifecycle(self.ui)
        self._stop_tasks: set[asyncio.Task] = set()
        self._profile_results: ProcessRecordsResult | None = None
        self._exit_errors: list[ExitErrorInfo] = []
        self._telemetry_results: TelemetryExportData | None = None
        self._server_metrics_results: ServerMetricsResults | None = None
        self._profile_results_received = False
        self._should_wait_for_telemetry = False
        self._should_wait_for_server_metrics = False

        self._shutdown_triggered = False
        self._shutdown_lock = asyncio.Lock()
        self._telemetry_endpoints_configured: list[str] = []
        self._telemetry_endpoints_reachable: list[str] = []
        self._server_metrics_endpoints_configured: list[str] = []
        self._server_metrics_endpoints_reachable: list[str] = []
        self.debug("System Controller created")

    def _should_warn_osl_without_ignore_eos(self) -> bool:
        """Check if --osl is used without ignore_eos or min_tokens in extra inputs."""
        osl_mean = self.user_config.input.prompt.output_tokens.mean
        if osl_mean is None:
            return False

        extra_inputs = self.user_config.input.extra
        if not extra_inputs:
            return True

        # Check if ignore_eos or min_tokens is set with a truthy value
        extra_dict = dict(extra_inputs)
        return not (extra_dict.get("ignore_eos") or extra_dict.get("min_tokens"))

    async def request_realtime_metrics(self) -> None:
        """Request real-time metrics from the RecordsManager."""
        await self.send_command_and_wait_for_response(
            RealtimeMetricsCommand(
                service_id=self.service_id,
                target_service_type=ServiceType.RECORDS_MANAGER,
            )
        )

    async def initialize(self) -> None:
        """We need to override the initialize method to run the proxy manager before the base service initialize.
        This is because the proxies need to be running before we can subscribe to the message bus.
        """
        self.debug("Running ZMQ Proxy Manager Before Initialize")
        await self.proxy_manager.initialize_and_start()
        # Once the proxies are running, call the original initialize method
        await super().initialize()

    @on_init
    async def _initialize_system_controller(self) -> None:
        self.debug("Initializing System Controller")

        self.setup_signal_handlers(self._handle_signal)
        self.debug("Setup signal handlers")

        async with self.try_operation_or_stop("Initialize Service Manager"):
            await self.service_manager.initialize()

        self.debug("System Controller initialized successfully")

    @on_start
    async def _start_services(self) -> None:
        """Bootstrap the system services.

        This method will:
        - Initialize all required services
        - Wait for all required services to be registered
        - Start all required services
        """
        self.debug("System Controller is bootstrapping services")

        # Start all required services
        async with self.try_operation_or_stop("Start Service Manager"):
            await self.service_manager.start()

        # Start optional services before waiting for registration so they can participate in configuration
        if not self.user_config.gpu_telemetry_disabled:
            await self.service_manager.run_service(ServiceType.GPU_TELEMETRY_MANAGER)
        else:
            self.info("GPU telemetry disabled via --no-gpu-telemetry")
            self._should_wait_for_telemetry = False

        if not self.user_config.server_metrics_disabled:
            self.debug("Starting optional ServerMetricsManager service")
            await self.service_manager.run_service(ServiceType.SERVER_METRICS_MANAGER)
        else:
            self.info("Server metrics disabled via --no-server-metrics")
            self._should_wait_for_server_metrics = False

        async with self.try_operation_or_stop("Register Services"):
            await self.service_manager.wait_for_all_services_registration(
                stop_event=self._stop_requested_event,
            )

        self.info("AIPerf System is CONFIGURING")
        await self._profile_configure_all_services()
        self.info("AIPerf System is CONFIGURED")
        await self._start_profiling_all_services()
        self.info("AIPerf System is PROFILING")

    async def _profile_configure_all_services(self) -> None:
        """Configure all services to start profiling.

        This is a blocking call that will wait for all services to be configured
        before returning. Uses fail-fast behavior: if any service returns an error,
        we abort immediately without waiting for the remaining services.
        """
        self.info("Configuring all services to start profiling")
        begin = time.perf_counter()
        responses = await self.send_command_and_wait_until_first_error(
            ProfileConfigureCommand(
                service_id=self.service_id,
                config=self.user_config,
            ),
            list(self.service_manager.service_id_map.keys()),
            timeout=Environment.SERVICE.PROFILE_CONFIGURE_TIMEOUT,
        )
        duration = time.perf_counter() - begin
        self._parse_responses_for_errors(responses, "Configure Profiling")
        self.info(f"All services configured in {duration:.2f} seconds")

        if not Environment.HTTP.SSL_VERIFY:
            self.warning(
                "SSL certificate verification is DISABLED - this is insecure. This should only be used for testing in a trusted environment."
            )

    async def _start_profiling_all_services(self) -> None:
        """Tell all services to start profiling.

        Uses fail-fast behavior: if any service returns an error,
        we abort immediately without waiting for the remaining services.
        """
        self.debug("Sending PROFILE_START command to all services")
        responses = await self.send_command_and_wait_until_first_error(
            ProfileStartCommand(
                service_id=self.service_id,
            ),
            list(self.service_manager.service_id_map.keys()),
            timeout=Environment.SERVICE.PROFILE_START_TIMEOUT,
        )
        self._parse_responses_for_errors(responses, "Start Profiling")
        self.info("All services started profiling successfully")

    def _parse_responses_for_errors(
        self, responses: list[CommandResponse | ErrorDetails], operation: str
    ) -> None:
        """Parse the responses for errors."""
        for response in responses:
            if isinstance(response, ErrorDetails):
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=response, operation=operation, service_id=None
                    )
                )
            elif isinstance(response, CommandErrorResponse):
                self._exit_errors.append(
                    ExitErrorInfo(
                        error_details=response.error,
                        operation=operation,
                        service_id=response.service_id,
                    )
                )
        if self._exit_errors:
            raise LifecycleOperationError(
                operation=operation,
                original_exception=None,
                lifecycle_id=self.id,
            )

    @on_command(CommandType.REGISTER_SERVICE)
    async def _handle_register_service_command(
        self, message: RegisterServiceCommand
    ) -> None:
        """Process a registration message from a service.

        Adds the service to the service manager's tracking maps (service_id_map and
        service_map) so it can participate in lifecycle coordination.

        Args:
            message: The registration message to process
        """

        self.debug(
            lambda: (
                f"Processing registration from {message.service_type} with ID: {message.service_id}"
            )
        )

        service_info = ServiceRunInfo(
            registration_status=ServiceRegistrationStatus.REGISTERED,
            service_type=message.service_type,
            service_id=message.service_id,
            first_seen=time.time_ns(),
            state=message.state,
            last_seen=time.time_ns(),
        )

        self.service_manager.service_id_map[message.service_id] = service_info
        if message.service_type not in self.service_manager.service_map:
            self.service_manager.service_map[message.service_type] = []
        self.service_manager.service_map[message.service_type].append(service_info)

        try:
            type_name = ServiceType(message.service_type).name.title().replace("_", " ")
        except (TypeError, ValueError):
            type_name = message.service_type
        self.info(lambda: f"Registered {type_name} (id: '{message.service_id}')")

    @on_message(MessageType.HEARTBEAT)
    async def _process_heartbeat_message(self, message: HeartbeatMessage) -> None:
        """Process a heartbeat message from a service. It will
        update the last seen timestamp and state of the service.

        Args:
            message: The heartbeat message to process
        """
        service_id = message.service_id
        service_type = message.service_type
        timestamp = message.request_ns

        # Update the last heartbeat timestamp if the component exists
        try:
            service_info = self.service_manager.service_id_map[service_id]
            service_info.last_seen = timestamp
            service_info.state = message.state
            self.debug(lambda: f"Updated heartbeat for '{service_id}' to {timestamp}")
        except Exception:
            self.warning(
                f"Received heartbeat from unknown service: '{service_id}' ('{service_type}')"
            )

    @on_message(MessageType.CREDITS_COMPLETE)
    async def _process_credits_complete_message(
        self, message: CreditsCompleteMessage
    ) -> None:
        """Log receipt of credits complete message from a service.

        Args:
            message: The credits complete message to process
        """
        service_id = message.service_id
        self.info(f"Received credits complete from '{service_id}'")

    @on_message(MessageType.STATUS)
    async def _process_status_message(self, message: StatusMessage) -> None:
        """Process a generic service lifecycle status message.

        Updates the service registry with lifecycle state changes (initializing,
        running, stopping, etc.).

        Args:
            message: The status message to process
        """
        service_id = message.service_id
        service_type = message.service_type
        state = message.state

        self.debug(
            lambda: (
                f"Received status update from '{service_type}' (ID: '{service_id}'): {state}"
            )
        )

        # Update the component state if the component exists
        if service_id not in self.service_manager.service_id_map:
            self.debug(
                lambda: (
                    f"Received status update from un-registered service: {service_id} ({service_type})"
                )
            )
            return

        service_info = self.service_manager.service_id_map.get(service_id)
        if service_info is None:
            return

        service_info.state = message.state

        self.debug(f"Updated state for {service_id} to {message.state}")

    @on_message(MessageType.TELEMETRY_STATUS)
    async def _on_telemetry_status_message(
        self, message: TelemetryStatusMessage
    ) -> None:
        """Handle telemetry status from TelemetryManager.

        TelemetryStatusMessage informs SystemController if telemetry results will be available.
        """

        self._telemetry_endpoints_configured = message.endpoints_configured
        self._telemetry_endpoints_reachable = message.endpoints_reachable
        self._should_wait_for_telemetry = message.enabled

        if not message.enabled:
            reason_msg = f" - {message.reason}" if message.reason else ""
            self.info(f"GPU telemetry disabled{reason_msg}")
        else:
            self.info(
                f"GPU telemetry enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable"
            )

        # Re-check shutdown readiness in case results arrived before status message
        await self._check_and_trigger_shutdown()

    @on_message(MessageType.SERVER_METRICS_STATUS)
    async def _on_server_metrics_status_message(
        self, message: ServerMetricsStatusMessage
    ) -> None:
        """Handle server metrics status from ServerMetricsManager.

        ServerMetricsStatusMessage informs SystemController if server metrics results will be available.
        """

        self._server_metrics_endpoints_configured = message.endpoints_configured
        self._server_metrics_endpoints_reachable = message.endpoints_reachable
        self._should_wait_for_server_metrics = message.enabled

        if not message.enabled:
            reason_msg = f" - {message.reason}" if message.reason else ""
            self.info(f"Server metrics disabled{reason_msg}")
        else:
            self.info(
                f"Server metrics enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable."
            )
            unreachable_endpoints = set(message.endpoints_configured) - set(
                message.endpoints_reachable
            )
            if unreachable_endpoints:
                self.warning(
                    f"Unreachable endpoints: {', '.join(unreachable_endpoints)}"
                )

        # Re-check shutdown readiness in case results arrived before status message
        await self._check_and_trigger_shutdown()

    @on_message(MessageType.COMMAND_RESPONSE)
    async def _process_command_response_message(self, message: CommandResponse) -> None:
        """Process a command response message."""
        self.debug(lambda: f"Received command response message: {message}")
        if message.status == CommandResponseStatus.SUCCESS:
            self.debug(f"Command {message.command} succeeded from {message.service_id}")
        elif message.status == CommandResponseStatus.ACKNOWLEDGED:
            self.debug(
                f"Command {message.command} acknowledged from {message.service_id}"
            )
        elif message.status == CommandResponseStatus.UNHANDLED:
            self.debug(f"Command {message.command} unhandled from {message.service_id}")
        elif message.status == CommandResponseStatus.FAILURE:
            message = cast(CommandErrorResponse, message)
            self.error(
                f"Command {message.command} failed from {message.service_id}: {message.error}"
            )

    @on_command(CommandType.SPAWN_WORKERS)
    async def _handle_spawn_workers_command(self, message: SpawnWorkersCommand) -> None:
        """Handle a spawn workers command."""
        self.debug(lambda: f"Received spawn workers command: {message}")
        # Spawn the workers
        await self.service_manager.run_service(ServiceType.WORKER, message.num_workers)
        # If we are scaling the record processor service count with the number of workers, spawn the record processors
        if self.scale_record_processors_with_workers:
            await self.service_manager.run_service(
                ServiceType.RECORD_PROCESSOR,
                max(
                    1, message.num_workers // Environment.RECORD.PROCESSOR_SCALE_FACTOR
                ),
            )

    @on_command(CommandType.SHUTDOWN_WORKERS)
    async def _handle_shutdown_workers_command(
        self, message: ShutdownWorkersCommand
    ) -> None:
        """Handle a shutdown workers command."""
        self.debug(lambda: f"Received shutdown workers command: {message}")
        # TODO: Handle individual worker shutdowns via worker id
        await self.service_manager.stop_service(ServiceType.WORKER)
        if self.scale_record_processors_with_workers:
            await self.service_manager.stop_service(ServiceType.RECORD_PROCESSOR)

    @on_message(MessageType.PROCESS_RECORDS_RESULT)
    async def _on_process_records_result_message(
        self, message: ProcessRecordsResultMessage
    ) -> None:
        """Handle a profile results message."""
        self.trace_or_debug(
            lambda: f"Received profile results message: {message}",
            lambda: (
                f"Received profile results message: {len(message.results.results.records) if message.results.results else 0} records"
            ),
        )
        if message.results.errors:
            self.error(
                f"Received process records result message with errors: {message.results.errors}"
            )

        self.debug(
            lambda: (
                f"Error summary: {message.results.results.error_summary if message.results.results else 'N/A'}"
            )
        )

        self._profile_results = message.results

        if not message.results.results:
            self.error(
                f"Received process records result message with no records: {message.results.results}"
            )

        self._profile_results_received = True
        # Coordinate with telemetry results before shutdown
        await self._check_and_trigger_shutdown()

    @on_message(MessageType.PROCESS_TELEMETRY_RESULT)
    async def _on_process_telemetry_result_message(
        self, message: ProcessTelemetryResultMessage
    ) -> None:
        """Handle a telemetry results message."""
        try:
            self.trace_or_debug(
                lambda: f"Received telemetry results message: {message}",
                lambda: (
                    f"Received telemetry results message: {len(message.telemetry_result.results.endpoints) if message.telemetry_result.results else 0} endpoints"
                ),
            )

            telemetry_results = message.telemetry_result.results
            if not telemetry_results:
                self.error(
                    f"Received process telemetry result message with no records: {telemetry_results}"
                )
            else:
                # Update endpoint info in the summary (TelemetryExportData structure)
                telemetry_results.summary.endpoints_configured = (
                    self._telemetry_endpoints_configured
                )
                telemetry_results.summary.endpoints_successful = (
                    self._telemetry_endpoints_reachable
                )

            self._telemetry_results = telemetry_results
        except Exception as e:
            self.exception(f"Error processing telemetry results message: {e!r}")
        finally:
            self._should_wait_for_telemetry = False
            await self._check_and_trigger_shutdown()

    @on_message(MessageType.PROCESS_SERVER_METRICS_RESULT)
    async def _on_process_server_metrics_result_message(
        self, message: ProcessServerMetricsResultMessage
    ) -> None:
        """Handle a server metrics results message."""
        try:
            self.trace_or_debug(
                lambda: f"Received server metrics results message: {message}",
                lambda: (
                    f"Received server metrics results message: {len(message.server_metrics_result.results.endpoint_summaries or {}) if message.server_metrics_result.results else 0} endpoints"
                ),
            )

            self.debug(
                lambda: (
                    f"Server metrics error summary: {message.server_metrics_result.results.error_summary if message.server_metrics_result.results else 'N/A'}"
                )
            )

            server_metrics_results = message.server_metrics_result.results

            if not server_metrics_results:
                self.debug(
                    f"Received process server metrics result message with no results: {server_metrics_results}"
                )
            else:
                server_metrics_results.endpoints_configured = (
                    self._server_metrics_endpoints_configured
                )
                server_metrics_results.endpoints_successful = (
                    self._server_metrics_endpoints_reachable
                )

            self._server_metrics_results = server_metrics_results
        except Exception as e:
            self.exception(f"Error processing server metrics results message: {e!r}")
        finally:
            self._should_wait_for_server_metrics = False
            await self._check_and_trigger_shutdown()

    async def _check_and_trigger_shutdown(self) -> None:
        """Check if all required results are received and trigger unified export + shutdown.

        Coordination logic:
        1. Always wait for profile results (ProcessRecordsResultMessage)
        2. If telemetry disabled OR telemetry results received → proceed
        3. If server metrics disabled OR server metrics results received → proceed
        4. Otherwise → wait (results arrive nearly simultaneously and will call this method again)

        Thread safety:
        Uses self._shutdown_lock to prevent race conditions when ProcessRecordsResultMessage,
        ProcessTelemetryResultMessage, and ProcessServerMetricsResultMessage arrive concurrently.
        The lock ensures atomic check-and-set of _shutdown_triggered, preventing double-triggering of stop().
        """
        self.debug(
            f"_check_and_trigger_shutdown: profile_received={self._profile_results_received}, "
            f"wait_telemetry={self._should_wait_for_telemetry}, telemetry_results={self._telemetry_results is not None}, "
            f"wait_server_metrics={self._should_wait_for_server_metrics}, server_metrics_results={self._server_metrics_results is not None}, "
            f"shutdown_triggered={self._shutdown_triggered}"
        )
        # Check if we should trigger shutdown (with lock protection)
        should_shutdown = False
        async with self._shutdown_lock:
            if self._shutdown_triggered:
                self.debug(
                    "_check_and_trigger_shutdown: shutdown already triggered, returning"
                )
                return

            if not self._profile_results_received:
                self.debug(
                    "_check_and_trigger_shutdown: profile results not received yet"
                )
                return

            telemetry_ready_for_shutdown = (
                not self._should_wait_for_telemetry
                or self._telemetry_results is not None
            )

            server_metrics_ready_for_shutdown = (
                not self._should_wait_for_server_metrics
                or self._server_metrics_results is not None
            )

            if telemetry_ready_for_shutdown and server_metrics_ready_for_shutdown:
                self._shutdown_triggered = True
                should_shutdown = True
                self.info("All results received, initiating shutdown")
            else:
                if not telemetry_ready_for_shutdown:
                    self.info("Waiting for telemetry results...")
                if not server_metrics_ready_for_shutdown:
                    self.info("Waiting for server metrics results...")

        # Call stop() OUTSIDE the lock to prevent deadlock
        if should_shutdown:
            self.debug("Calling self.stop()...")
            await asyncio.shield(self.stop())
            self.debug("self.stop() completed")

    async def _handle_signal(self, sig: int) -> None:
        """Handle received signals with two-stage cancellation.

        First Ctrl+C: Graceful cancel - stops issuing new credits, cancels
        in-flight requests, and writes results to files.

        Second Ctrl+C: Force quit - immediately terminates all processes.
        Results may be incomplete or not written.

        Args:
            sig: The signal number received
        """
        if self._was_cancelled:
            # SECOND Ctrl+C - Force quit immediately
            self._print_force_quit_warning()
            self.warning(f"Force quit requested (signal {sig})")
            await self._kill()
            return

        # FIRST Ctrl+C - Graceful cancel with warning
        self._print_cancel_warning()
        self.warning(f"Graceful shutdown requested (signal {sig})")
        await self._cancel_profiling()

    def _print_cancel_warning(self) -> None:
        """Print prominent warning panel on first Ctrl+C.

        Informs user that the benchmark is being cancelled gracefully and
        results are being processed. Also instructs how to force quit.

        Uses stderr to ensure visibility even when stdout is redirected or
        captured by the UI.
        """
        console = Console(file=sys.stderr, force_terminal=True)
        console.print()
        console.print(
            Panel(
                "[bold yellow]⚠️  BENCHMARK CANCELLED[/bold yellow]\n\n"
                "Stopping credit issuance and cancelling in-flight requests...\n"
                "Results will be written to files.\n\n"
                "[dim]Press Ctrl+C again to force quit immediately[/dim]\n"
                "[dim](results may be incomplete or not written)[/dim]",
                border_style="yellow",
                padding=(1, 2),
                title="[bold yellow]Cancellation in Progress[/bold yellow]",
            )
        )
        console.print()
        console.file.flush()

    def _print_force_quit_warning(self) -> None:
        """Print warning panel on second Ctrl+C (force quit).

        Warns user that results may be incomplete due to immediate termination.

        Uses stderr to ensure visibility even when stdout is redirected or
        captured by the UI.
        """
        console = Console(file=sys.stderr, force_terminal=True)
        console.print()
        console.print(
            Panel(
                "[bold red]🛑 FORCE QUIT[/bold red]\n\n"
                "Terminating all processes immediately.\n"
                "Results may be incomplete or not written to files.",
                border_style="red",
                padding=(1, 2),
                title="[bold red]Force Quit[/bold red]",
            )
        )
        console.print()
        console.file.flush()

    async def _cancel_profiling(self) -> None:
        self.debug("Cancelling profiling of all services")
        self._was_cancelled = True

        # Mark shutdown as triggered FIRST to prevent _check_and_trigger_shutdown()
        # from also calling stop() when results arrive during cancellation.
        # This prevents the race condition that causes SIGKILL (exit code -9).
        # Also track if shutdown was already triggered to avoid double-stop.
        should_call_stop = False
        async with self._shutdown_lock:
            if not self._shutdown_triggered:
                self._shutdown_triggered = True
                should_call_stop = True
            else:
                self.debug("Shutdown already triggered, skipping stop() call")

        # Only wait for RecordsManager's response since it returns ProcessRecordsResult.
        # Other services receive the broadcast cancel command but we don't wait for them.
        # This avoids blocking if a service has exited early (e.g., TelemetryManager).
        records_manager_ids = [
            service_id
            for service_id, info in self.service_manager.service_id_map.items()
            if info.service_type == ServiceType.RECORDS_MANAGER
        ]
        self.debug(
            f"Sending cancel to all services, waiting for {len(records_manager_ids)} RecordsManager(s)"
        )

        try:
            responses = await self.send_command_and_wait_for_all_responses(
                ProfileCancelCommand(
                    service_id=self.service_id,
                ),
                records_manager_ids,
                timeout=Environment.SERVICE.PROFILE_CANCEL_TIMEOUT,
            )

            # Log any errors but do NOT raise exceptions during cancellation.
            # Cancellation is best-effort - we must always proceed to stop().
            for response in responses:
                if isinstance(response, ErrorDetails):
                    self.warning(
                        f"Cancel command error (timeout or service unavailable): {response}"
                    )
                elif isinstance(response, CommandErrorResponse):
                    self.warning(
                        f"Cancel command failed from {response.service_id}: {response.error}"
                    )

            # Extract ProcessRecordsResult from the RecordsManager's response.
            # We must set _profile_results here because we've blocked the normal
            # message-based shutdown flow by setting _shutdown_triggered = True.
            # The command response contains the same data as ProcessRecordsResultMessage.
            for response in responses:
                if (
                    isinstance(response, CommandSuccessResponse)
                    and response.command == CommandType.PROFILE_CANCEL
                    and isinstance(response.data, ProcessRecordsResult)
                ):
                    self.debug(
                        lambda r=response: (
                            f"Received ProcessRecordsResult from cancel command: {r.data}"
                        )
                    )
                    self._profile_results = response.data
                    self._profile_results_received = True
                    break
        except Exception as e:
            # Catch ANY exception during cancellation - we must always proceed to stop().
            self.warning(f"Exception during cancel command (proceeding to stop): {e!r}")

        # Only call stop() if we were the first to trigger shutdown
        if should_call_stop:
            self.debug("Stopping system controller after profiling cancelled")
            await asyncio.shield(self.stop())

    @on_stop
    async def _stop_system_controller(self) -> None:
        """Stop the system controller and all running services."""
        # Broadcast a shutdown command to all services
        await self.publish(ShutdownCommand(service_id=self.service_id))

        # TODO: HACK: Wait for 0.5 seconds to ensure the shutdown command is received
        await asyncio.sleep(0.5)

        await self.service_manager.shutdown_all_services()
        await self.comms.stop()
        await self.proxy_manager.stop()

        # Wait for the UI to stop before exporting any results to the console
        await self.ui.stop()
        await self.ui.wait_for_tasks()
        await asyncio.sleep(0.1)  # Give time for screen clear to finish

        if not self._exit_errors:
            await self._print_post_benchmark_info_and_metrics()
        else:
            self._print_exit_errors_and_log_file()

        if Environment.DEV.MODE:
            # Print a warning message to the console if developer mode is enabled, on exit after results
            print_developer_mode_warning()

        # Clean up the global log queue to prevent semaphore leaks
        await cleanup_global_log_queue()

        # Exit the process in a more explicit way, to ensure that it stops
        os._exit(1 if self._exit_errors else 0)

    def _print_exit_errors_and_log_file(self) -> None:
        """Print post exit errors and log file info to the console."""
        console = Console()
        print_exit_errors(self._exit_errors, console=console)
        self._print_log_file_info(console)
        console.print()
        console.file.flush()

    async def _print_post_benchmark_info_and_metrics(self) -> None:
        """Print post benchmark info and metrics to the console."""
        if not self._profile_results or not self._profile_results.results.records:
            self.error("No profile results to export")
            sys.exit(1)

        console = Console()
        if console.width < 100:
            console.width = 100

        exporter_manager = ExporterManager(
            results=self._profile_results.results,
            user_config=self.user_config,
            service_config=self.service_config,
            telemetry_results=self._telemetry_results,
            server_metrics_results=self._server_metrics_results,
        )

        # Export data files (CSV, JSON) with complete dataset including telemetry
        await exporter_manager.export_data()

        # Export console output with complete dataset including telemetry
        await exporter_manager.export_console(console=console)

        console.print()
        self._print_cli_command(console)
        self._print_benchmark_duration(console)
        self._print_exported_file_infos(exporter_manager, console)
        self._print_log_file_info(console)
        if self._was_cancelled:
            console.print(
                "[italic yellow]The profile run was cancelled early. Results shown may be incomplete or inaccurate.[/italic yellow]"
            )

        console.print()
        console.file.flush()

    def _print_log_file_info(self, console: Console) -> None:
        """Print the log file info."""
        log_file = (
            self.user_config.output.artifact_directory
            / OutputDefaults.LOG_FOLDER
            / OutputDefaults.LOG_FILE
        )
        console.print(
            f"[bold green]Log File:[/bold green] [cyan]{log_file.resolve()}[/cyan]"
        )

    def _print_exported_file_infos(
        self, exporter_manager: ExporterManager, console: Console
    ) -> None:
        """Print the exported file infos."""
        file_infos = exporter_manager.get_exported_file_infos()
        for file_info in file_infos:
            console.print(
                f"[bold green]{file_info.export_type}[/bold green]: [cyan]{file_info.file_path.resolve()}[/cyan]"
            )

    def _print_cli_command(self, console: Console) -> None:
        """Print the CLI command that was used to run the benchmark."""
        console.print(
            f"[bold green]CLI Command:[/bold green] [italic]{self.user_config.cli_command}[/italic]"
        )

    def _print_benchmark_duration(self, console: Console) -> None:
        """Print the duration of the benchmark."""
        from aiperf.metrics.types.benchmark_duration_metric import (
            BenchmarkDurationMetric,
        )

        # Metrics are already in display units from summarize()
        duration = self._profile_results.get(BenchmarkDurationMetric.tag)
        if duration:
            duration_str = f"[bold green]{BenchmarkDurationMetric.header}[/bold green]: {duration.avg:.2f} {duration.unit}"
            if self._was_cancelled:
                duration_str += " [italic yellow](cancelled early)[/italic yellow]"
            console.print(duration_str)

    async def _kill(self):
        """Kill the system controller."""
        try:
            await self.service_manager.kill_all_services()
        except Exception as e:
            raise self._service_error("Failed to stop all services") from e

        await super()._kill()


def main() -> None:
    """Main entry point for the system controller."""

    from aiperf.common.bootstrap import bootstrap_and_run_service
    from aiperf.plugin.enums import ServiceType

    bootstrap_and_run_service(ServiceType.SYSTEM_CONTROLLER)


if __name__ == "__main__":
    main()
