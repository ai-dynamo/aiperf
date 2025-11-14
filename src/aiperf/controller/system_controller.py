# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import time
from typing import cast

from rich.console import Console

from aiperf.cli_utils import print_developer_mode_warning
from aiperf.common.base_service import BaseService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    MessageType,
    ServiceRegistrationStatus,
    ServiceType,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import LifecycleOperationError
from aiperf.common.factories import (
    AIPerfUIFactory,
    ServiceFactory,
    ServiceManagerFactory,
)
from aiperf.common.hooks import on_command, on_init, on_message, on_start, on_stop
from aiperf.common.logging import cleanup_global_log_queue, get_global_log_queue
from aiperf.common.messages import (
    CommandErrorResponse,
    CommandResponse,
    CreditsCompleteMessage,
    HeartbeatMessage,
    ProcessRecordsResultMessage,
    ProcessTelemetryResultMessage,
    ProfileCancelCommand,
    ProfileConfigureCommand,
    ProfileStartCommand,
    RealtimeMetricsCommand,
    RegisterServiceCommand,
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
    TelemetryResults,
)
from aiperf.common.models.error_models import ExitErrorInfo
from aiperf.common.protocols import AIPerfUIProtocol, ServiceManagerProtocol
from aiperf.common.types import ServiceTypeT
from aiperf.controller.controller_utils import print_exit_errors
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.controller.system_mixins import SignalHandlerMixin
from aiperf.exporters.exporter_manager import ExporterManager


@ServiceFactory.register(ServiceType.SYSTEM_CONTROLLER)
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
        self.service_manager: ServiceManagerProtocol = (
            ServiceManagerFactory.create_instance(
                self.service_config.service_run_type.value,
                required_services=self.required_services,
                user_config=self.user_config,
                service_config=self.service_config,
                log_queue=get_global_log_queue(),
            )
        )
        self.ui: AIPerfUIProtocol = AIPerfUIFactory.create_instance(
            self.service_config.ui_type,
            service_config=self.service_config,
            user_config=self.user_config,
            log_queue=get_global_log_queue(),
            controller=self,
        )
        self.attach_child_lifecycle(self.ui)
        self._stop_tasks: set[asyncio.Task] = set()
        self._profile_results: ProcessRecordsResult | None = None
        self._exit_errors: list[ExitErrorInfo] = []
        self._telemetry_results: TelemetryResults | None = None
        self._profile_results_received = False
        self._should_wait_for_telemetry = False

        self._shutdown_triggered = False
        self._shutdown_lock = asyncio.Lock()
        self._endpoints_configured: list[str] = []
        self._endpoints_reachable: list[str] = []
        self.debug("System Controller created")

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
        self.debug("Starting optional TelemetryManager service")
        await self.service_manager.run_service(ServiceType.TELEMETRY_MANAGER, 1)

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

        This is a blocking call that will wait for all services to be configured before returning. This way we can ensure that all services are configured before we start profiling.
        """
        self.info("Configuring all services to start profiling")
        begin = time.perf_counter()
        responses = await self.send_command_and_wait_for_all_responses(
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

    async def _start_profiling_all_services(self) -> None:
        """Tell all services to start profiling."""
        self.debug("Sending PROFILE_START command to all services")
        responses = await self.send_command_and_wait_for_all_responses(
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
        """Process a registration message from a service. It will
        add the service to the service manager and send a configure command
        to the service.

        Args:
            message: The registration message to process
        """

        self.debug(
            lambda: f"Processing registration from {message.service_type} with ID: {message.service_id}"
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

        self.debug(lambda: f"Received heartbeat from {service_type} (ID: {service_id})")

        # Update the last heartbeat timestamp if the component exists
        try:
            service_info = self.service_manager.service_id_map[service_id]
            service_info.last_seen = timestamp
            service_info.state = message.state
            self.debug(f"Updated heartbeat for {service_id} to {timestamp}")
        except Exception:
            self.warning(
                f"Received heartbeat from unknown service: {service_id} ({service_type})"
            )

    @on_message(MessageType.CREDITS_COMPLETE)
    async def _process_credits_complete_message(
        self, message: CreditsCompleteMessage
    ) -> None:
        """Process a credits complete message from a service. It will
        update the state of the service with the service manager.

        Args:
            message: The credits complete message to process
        """
        service_id = message.service_id
        self.info(f"Received credits complete from {service_id}")

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
            lambda: f"Received status update from {service_type} (ID: {service_id}): {state}"
        )

        # Update the component state if the component exists
        if service_id not in self.service_manager.service_id_map:
            self.debug(
                lambda: f"Received status update from un-registered service: {service_id} ({service_type})"
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

        self._endpoints_configured = message.endpoints_configured
        self._endpoints_reachable = message.endpoints_reachable
        self._should_wait_for_telemetry = message.enabled

        if not message.enabled:
            reason_msg = f" - {message.reason}" if message.reason else ""
            self.info(f"GPU telemetry disabled{reason_msg}")
        else:
            self.info(
                f"GPU telemetry enabled - {len(message.endpoints_reachable)}/{len(message.endpoints_configured)} endpoint(s) reachable"
            )

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
        self.debug(lambda: f"Received profile results message: {message}")
        if message.results.errors:
            self.error(
                f"Received process records result message with errors: {message.results.errors}"
            )

        self.debug(lambda: f"Error summary: {message.results.results.error_summary}")

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
        self.debug(lambda: f"Received telemetry results message: {message}")

        if message.telemetry_result.errors:
            self.warning(
                f"Received process telemetry result message with errors: {message.telemetry_result.errors}"
            )

        self.debug(
            lambda: f"Error summary: {message.telemetry_result.results.error_summary}"
        )

        telemetry_results = message.telemetry_result.results

        if not message.telemetry_result.results:
            self.error(
                f"Received process telemetry result message with no records: {telemetry_results}"
            )

        if telemetry_results:
            telemetry_results.endpoints_configured = self._endpoints_configured
            telemetry_results.endpoints_successful = self._endpoints_reachable

        self._telemetry_results = telemetry_results

        await self._check_and_trigger_shutdown()

    async def _check_and_trigger_shutdown(self) -> None:
        """Check if all required results are received and trigger unified export + shutdown.

        Coordination logic:
        1. Always wait for profile results (ProcessRecordsResultMessage)
        2. If telemetry disabled OR telemetry results received → proceed with shutdown
        3. Otherwise → wait (telemetry results arrive nearly simultaneously and will call this method again)

        Thread safety:
        Uses self._shutdown_lock to prevent race conditions when ProcessRecordsResultMessage
        and ProcessTelemetryResultMessage arrive concurrently. The lock ensures atomic
        check-and-set of _shutdown_triggered, preventing double-triggering of stop().
        """
        async with self._shutdown_lock:
            if self._shutdown_triggered:
                return

            if not self._profile_results_received:
                return

            telemetry_ready_for_shutdown = (
                not self._should_wait_for_telemetry
                or self._telemetry_results is not None
            )

            if telemetry_ready_for_shutdown:
                self._shutdown_triggered = True
                self.debug("All results received, initiating shutdown")
                await asyncio.shield(self.stop())
            else:
                self.debug("Waiting for telemetry results...")

    async def _handle_signal(self, sig: int) -> None:
        """Handle received signals by triggering graceful shutdown.

        Args:
            sig: The signal number received
        """
        if self.stop_requested:
            # If we are already in a stopping state, we need to kill the process to be safe.
            self.warning(f"Received signal {sig}, killing")
            await self._kill()
            return

        self.debug(lambda: f"Received signal {sig}, initiating graceful shutdown")
        await self._cancel_profiling()

    async def _cancel_profiling(self) -> None:
        self.debug("Cancelling profiling of all services")
        self._was_cancelled = True
        await self.publish(ProfileCancelCommand(service_id=self.service_id))

        # TODO: HACK: Wait for 2 seconds to ensure the profiling is cancelled
        # Wait for the profiling to be cancelled
        await asyncio.sleep(2)
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
            self.warning("No profile results to export")
            return

        console = Console()
        if console.width < 100:
            console.width = 100

        exporter_manager = ExporterManager(
            results=self._profile_results.results,
            user_config=self.user_config,
            service_config=self.service_config,
            telemetry_results=self._telemetry_results,
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

        duration = self._profile_results.get(BenchmarkDurationMetric.tag)
        if duration:
            duration = duration.to_display_unit()
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

    bootstrap_and_run_service(SystemController)


if __name__ == "__main__":
    main()
