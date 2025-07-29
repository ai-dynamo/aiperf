# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import signal
import sys
import time
from typing import cast

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.enums import (
    CommandResponseStatus,
    CommandType,
    MessageType,
    ServiceRegistrationStatus,
    ServiceType,
)
from aiperf.common.factories import ServiceFactory, ServiceManagerFactory
from aiperf.common.hooks import on_command, on_init, on_message, on_start
from aiperf.common.logging import get_global_log_queue
from aiperf.common.messages import (
    CommandResponse,
    CreditsCompleteMessage,
    HeartbeatMessage,
    NotificationMessage,
    ProfileConfigureCommand,
    ProfileResultsMessage,
    RegistrationMessage,
    ShutdownWorkersCommand,
    SpawnWorkersCommand,
    StatusMessage,
)
from aiperf.common.messages.command_messages import (
    CommandErrorResponse,
    ProcessRecordsCommand,
    ProfileStartCommand,
    ShutdownCommand,
)
from aiperf.common.models import ServiceRunInfo
from aiperf.common.protocols import ServiceManagerProtocol
from aiperf.common.types import ServiceTypeT
from aiperf.data_exporter.exporter_manager import ExporterManager
from aiperf.services.base_service import BaseService
from aiperf.services.system_controller.proxy_manager import ProxyManager
from aiperf.services.system_controller.system_mixins import (
    SignalHandlerMixin,
)


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
        # List of required service types, in no particular order
        # These are services that must be running before the system controller can start profiling
        self.required_services: dict[ServiceTypeT, int] = {
            ServiceType.DATASET_MANAGER: 1,
            ServiceType.TIMING_MANAGER: 1,
            ServiceType.WORKER_MANAGER: 1,
            ServiceType.RECORDS_MANAGER: 1,
            ServiceType.INFERENCE_RESULT_PARSER: service_config.result_parser_service_count,
        }

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

        self.debug("System Controller created")

    async def initialize(self) -> None:
        """We need to override the initialize method to run the proxy manager before the base service initialize.
        This is because the proxies need to be running before we can subscribe to the message bus.
        """
        self.debug("Running ZMQ Proxy Manager Before Initialize")
        await self.proxy_manager.initialize_and_start()
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
        try:
            await self.service_manager.start()
        except Exception as e:
            raise self._service_error(
                "Failed to initialize all services",
            ) from e

        await self.service_manager.wait_for_all_services_registration(
            stop_event=self._stop_requested_event,
        )

        # TODO: HACK: Wait for 1 second to ensure registrations made. This needs to be
        # removed once we have the ability to track registrations of services and their state before
        # starting the profiling.
        await asyncio.sleep(1)

        self.info("AIPerf System is READY")

        await self._start_profiling_all_services()

        self.debug("All required services started successfully")
        self.info("AIPerf System is RUNNING")

    async def stop(self) -> None:
        """Stop the system controller and all running services.

        This method will:
        - Stop all running services
        """
        self.debug("Stopping System Controller")
        self.debug("AIPerf System is EXITING")
        await self.service_manager.kill_all_services()
        await self.comms.stop()
        await self.proxy_manager.stop()

    def _handle_signal(self, sig: int) -> None:
        """Handle received signals by triggering graceful shutdown.

        Args:
            sig: The signal number received
        """
        if sig == signal.SIGINT:
            if self.stop_requested:
                # If we are already in a stopping state, we need to kill the process to be safe.
                self.warning(lambda: f"Received signal {sig}, killing")
                asyncio.create_task(self._kill())
                return
            self.debug(lambda: f"Received signal {sig}, initiating graceful shutdown")
            asyncio.create_task(self._stop_system_controller())
            return

    async def _stop_system_controller(self, cancelled: bool = False) -> None:
        """Stop the system controller and all running services."""
        # TODO: This is a hack to force printing results again
        if cancelled:
            # Process records command
            await self.publish(
                ProcessRecordsCommand(
                    service_id=self.service_id,
                    target_service_type=ServiceType.RECORDS_MANAGER,
                    cancelled=True,
                ),
            )

        # Broadcast a stop command to all services
        await self.publish(
            ShutdownCommand(service_id=self.service_id),
        )

        try:
            await self.service_manager.shutdown_all_services()
        except Exception as e:
            raise self._service_error(
                "Failed to stop all services",
            ) from e

        await self.cancel_all_tasks()

        # TODO: HACK: This is a bit of a hack, but without the call to sys.exit, the process
        # will continue to linger.
        task = asyncio.create_task(self.stop())
        task.add_done_callback(lambda _: sys.exit(0))
        await task

    async def _start_profiling_all_services(self) -> None:
        """Tell all services to start profiling."""
        # TODO: HACK: Wait for 1 second to ensure services are ready
        await asyncio.sleep(1)

        self.debug("Sending START_PROFILING command to all services")
        await self.publish(
            ProfileStartCommand(service_id=self.service_id),
        )

    @on_message(MessageType.REGISTRATION)
    async def _process_registration_message(self, message: RegistrationMessage) -> None:
        """Process a registration message from a service. It will
        add the service to the service manager and send a configure command
        to the service.

        Args:
            message: The registration message to process
        """
        service_id = message.service_id
        service_type = message.service_type

        self.debug(
            lambda: f"Processing registration from {service_type} with ID: {service_id}"
        )

        service_info = ServiceRunInfo(
            registration_status=ServiceRegistrationStatus.REGISTERED,
            service_type=service_type,
            service_id=service_id,
            first_seen=time.time_ns(),
            state=message.state,
            last_seen=time.time_ns(),
        )

        self.service_manager.service_id_map[service_id] = service_info
        if service_type not in self.service_manager.service_map:
            self.service_manager.service_map[service_type] = []
        self.service_manager.service_map[service_type].append(service_info)

        self.info(lambda: f"Registered service: {service_type=} with ID: {service_id=}")

        # Send configure command to the newly registered service
        try:
            await self.publish(
                ProfileConfigureCommand(service_id=service_id, config=self.user_config)
            )
        except Exception as e:
            raise self._service_error(
                f"Failed to send configure command to {service_type} (ID: {service_id})",
            ) from e

        self.debug(
            lambda: f"Sent configure command to {service_type} (ID: {service_id})"
        )

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

    @on_message(MessageType.NOTIFICATION)
    async def _process_notification_message(self, message: NotificationMessage) -> None:
        """Process a notification message."""
        self.info(f"Received notification message: {message}")

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
        await self.service_manager.run_service(ServiceType.WORKER, message.num_workers)

    @on_command(CommandType.SHUTDOWN_WORKERS)
    async def _handle_shutdown_workers_command(
        self, message: ShutdownWorkersCommand
    ) -> None:
        """Handle a shutdown workers command."""
        self.debug(lambda: f"Received shutdown workers command: {message}")
        # TODO: Handle individual worker shutdowns via worker id
        await self.service_manager.stop_service(ServiceType.WORKER)

    @on_message(MessageType.PROFILE_RESULTS)
    async def _on_profile_results_message(
        self, profile_results: ProfileResultsMessage
    ) -> None:
        """Handle a profile results message."""
        self.debug(lambda: f"Received profile results message: {profile_results}")
        await ExporterManager(
            results=profile_results, input_config=self.user_config
        ).export_all()
        await self._stop_system_controller()

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
