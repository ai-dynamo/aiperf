# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import sys
import time
from typing import cast

from rich.console import Console

from aiperf.cli_utils import warn_cancelled_early
from aiperf.common.base_service import BaseService
from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.constants import (
    DEFAULT_PROFILE_CONFIGURE_TIMEOUT,
    DEFAULT_PROFILE_START_TIMEOUT,
)
from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    MessageType,
    ServiceRegistrationStatus,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory, ServiceManagerFactory
from aiperf.common.hooks import on_command, on_init, on_message, on_start, on_stop
from aiperf.common.logging import get_global_log_queue
from aiperf.common.messages import (
    CommandErrorResponse,
    CommandResponse,
    CreditsCompleteMessage,
    HeartbeatMessage,
    ProcessRecordsResultMessage,
    ProfileCancelCommand,
    ProfileConfigureCommand,
    ProfileStartCommand,
    RegisterServiceCommand,
    ShutdownCommand,
    ShutdownWorkersCommand,
    SpawnWorkersCommand,
    StatusMessage,
)
from aiperf.common.models import ServiceRunInfo
from aiperf.common.models.record_models import ProcessRecordsResult
from aiperf.common.protocols import ServiceManagerProtocol
from aiperf.common.types import ServiceTypeT
from aiperf.controller.proxy_manager import ProxyManager
from aiperf.controller.system_mixins import (
    SignalHandlerMixin,
)
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
        self._stop_tasks: set[asyncio.Task] = set()
        self._profile_results: ProcessRecordsResult | None = None
        self.debug("System Controller created")

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
        await self.service_manager.initialize()

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
        await self.service_manager.start()
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

        This is a blocking call that will wait for all services to be configured before returning. This way
        we can ensure that all services are configured before we start profiling.
        """
        self.info("Configuring all services to start profiling")
        begin = time.perf_counter()
        await self.send_command_and_wait_for_all_responses(
            ProfileConfigureCommand(
                service_id=self.service_id,
                config=self.user_config,
            ),
            list(self.service_manager.service_id_map.keys()),
            timeout=DEFAULT_PROFILE_CONFIGURE_TIMEOUT,
        )
        duration = time.perf_counter() - begin
        self.info(f"All services configured in {duration:.2f} seconds")

    async def _start_profiling_all_services(self) -> None:
        """Tell all services to start profiling."""
        self.debug("Sending PROFILE_START command to all services")
        await self.send_command_and_wait_for_all_responses(
            ProfileStartCommand(
                service_id=self.service_id,
            ),
            list(self.service_manager.service_id_map.keys()),
            timeout=DEFAULT_PROFILE_START_TIMEOUT,
        )
        self.info("All services started profiling successfully")

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
        """Process a status message from a service. It will
        update the state of the service with the service manager.

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
                ServiceType.RECORD_PROCESSOR, message.num_workers
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

        # This data will also be displayed by the console error exporter
        self.debug(lambda: f"Error summary: {message.results.results.error_summary}")

        self._profile_results = message.results

        if message.results.results:
            await ExporterManager(
                results=message.results.results,
                input_config=self.user_config,
            ).export_data()
        else:
            self.error(
                f"Received process records result message with no records: {message.results.results}"
            )

        # TODO: HACK: Stop the system controller after exporting the records
        self.debug("Stopping system controller after exporting records")
        await asyncio.shield(self.stop())

    async def _handle_signal(self, sig: int) -> None:
        """Handle received signals by triggering graceful shutdown.

        Args:
            sig: The signal number received
        """
        if self.stop_requested:
            # If we are already in a stopping state, we need to kill the process to be safe.
            self.warning(lambda: f"Received signal {sig}, killing")
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

        if self._profile_results:
            await ExporterManager(
                results=self._profile_results.results,
                input_config=self.user_config,
            ).export_console(console=Console())

            if (
                self._was_cancelled
                and self._profile_results.results
                and self._profile_results.results.records
            ):
                warn_cancelled_early()
        else:
            self.warning("No profile results to export")

        # Exit the process in a more explicit way, to ensure that it stops
        os._exit(0)

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
    sys.exit(0)
