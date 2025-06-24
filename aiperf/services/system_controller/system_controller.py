# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import signal
import sys
import time
from typing import Any

from aiperf.common.config import ServiceConfig
from aiperf.common.enums import (
    CommandType,
    ServiceRegistrationStatus,
    ServiceRunType,
    ServiceState,
    ServiceType,
    SystemState,
    Topic,
)
from aiperf.common.exceptions import (
    CommunicationError,
    CommunicationErrorReason,
    ConfigError,
)
from aiperf.common.factories import ServiceFactory
from aiperf.common.hooks import (
    on_cleanup,
    on_init,
    on_stop,
)
from aiperf.common.messages import (
    CreditsCompleteMessage,
    HeartbeatMessage,
    ProcessRecordsCommandData,
    ProfileResultsMessage,
    ProfileStatsMessage,
    RegistrationMessage,
    StatusMessage,
)
from aiperf.common.models import ServiceRunInfo
from aiperf.common.service.base_controller_service import BaseControllerService
from aiperf.services.service_manager import (
    BaseServiceManager,
    KubernetesServiceManager,
    MultiProcessServiceManager,
)
from aiperf.services.system_controller.system_mixins import SignalHandlerMixin


@ServiceFactory.register(ServiceType.SYSTEM_CONTROLLER)
class SystemController(SignalHandlerMixin, BaseControllerService):
    """System Controller service.

    This service is responsible for managing the lifecycle of all other services.
    It will start, stop, and configure all other services.
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Creating System Controller")

        self._system_state: SystemState = SystemState.INITIALIZING

        # List of required service types, in no particular order
        self.required_service_types: list[ServiceType] = [
            ServiceType.DATASET_MANAGER,
            ServiceType.TIMING_MANAGER,
            ServiceType.WORKER_MANAGER,
            ServiceType.RECORDS_MANAGER,
            ServiceType.POST_PROCESSOR_MANAGER,
        ]

        self.service_manager: BaseServiceManager = None  # type: ignore - is set in _initialize
        self.logger.debug("System Controller created")

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.SYSTEM_CONTROLLER

    async def _forever_loop(self) -> None:
        """Run the system controller in a loop until the stop event is set."""
        try:
            await super()._forever_loop()
        except KeyboardInterrupt:
            await self.send_command_to_service(
                target_service_type=ServiceType.RECORDS_MANAGER,
                target_service_id=None,
                command=CommandType.PROCESS_RECORDS,
                data=ProcessRecordsCommandData(cancelled=True),
            )

    @on_init
    async def _initialize(self) -> None:
        """Initialize system controller-specific components.

        This method will:
        - Initialize the service manager
        - Subscribe to relevant messages
        """
        self.logger.debug("Initializing System Controller")

        self.setup_signal_handlers(self._handle_signal)
        self.logger.debug("Setup signal handlers")

        if self.service_config.service_run_type == ServiceRunType.MULTIPROCESSING:
            self.service_manager = MultiProcessServiceManager(
                self.required_service_types, self.service_config
            )

        elif self.service_config.service_run_type == ServiceRunType.KUBERNETES:
            self.service_manager = KubernetesServiceManager(
                self.required_service_types, self.service_config
            )

        else:
            raise ConfigError(
                f"Unsupported service run type: {self.service_config.service_run_type}"
            )

        # Subscribe to relevant messages
        subscribe_callbacks = [
            (Topic.REGISTRATION, self._process_registration_message),
            (Topic.HEARTBEAT, self._process_heartbeat_message),
            (Topic.STATUS, self._process_status_message),
            (Topic.CREDITS_COMPLETE, self._process_credits_complete_message),
            (Topic.PROFILE_STATS, self._process_profile_stats_message),
            (Topic.PROFILE_RESULTS, self._process_profile_results_message),
        ]
        for topic, callback in subscribe_callbacks:
            try:
                await self.comms.subscribe(topic=topic, callback=callback)
            except Exception as e:
                self.logger.error("Failed to subscribe to topic %s: %s", topic, e)
                raise CommunicationError(
                    CommunicationErrorReason.SUBSCRIBE_ERROR,
                    f"Failed to subscribe to topic {topic}: {e}",
                ) from e

        # TODO: HACK:
        # wait 1 second to ensure that the communication is initialized
        await asyncio.sleep(1)

        self._system_state = SystemState.CONFIGURING
        await self._bootstrap_system()

    async def _handle_signal(self, sig: int) -> None:
        """Handle received signals by triggering graceful shutdown.

        Args:
            sig: The signal number received
        """
        self.logger.debug("Received signal %s, initiating graceful shutdown", sig)
        if sig == signal.SIGINT:
            await self.send_command_to_service(
                target_service_id=None,
                target_service_type=ServiceType.RECORDS_MANAGER,
                command=CommandType.PROCESS_RECORDS,
                data=ProcessRecordsCommandData(cancelled=True),
            )
        else:
            self.stop_event.set()

    async def _bootstrap_system(self) -> None:
        """Bootstrap the system services.

        This method will:
        - Initialize all required services
        - Wait for all required services to be registered
        - Start all required services
        """
        self.logger.debug("Starting System Controller")

        # Start all required services
        try:
            await self.service_manager.run_all_services()
        except Exception as e:
            raise self._service_error("Failed to initialize all services") from e

        try:
            # Wait for all required services to be registered
            await self.service_manager.wait_for_all_services_registration(
                self.stop_event
            )

            if self.stop_event.is_set():
                self.logger.debug(
                    "System Controller stopped before all services registered"
                )
                return  # Don't continue with the rest of the initialization

        except Exception as e:
            raise self._service_error(
                "Not all required services registered within the timeout period"
            ) from e

        self.logger.debug("All required services registered successfully")

        self.logger.info("AIPerf System is READY")
        self._system_state = SystemState.READY

        await self.start_profiling_all_services()

        if self.stop_event.is_set():
            self.logger.debug("System Controller stopped before all services started")
            return  # Don't continue with the rest of the initialization

        self.logger.debug("All required services started successfully")
        self.logger.info("AIPerf System is RUNNING")

    @on_stop
    async def _stop(self) -> None:
        """Stop the system controller and all running services.

        This method will:
        - Stop all running services
        """
        self.logger.debug("Stopping System Controller")
        self.logger.info("AIPerf System is SHUTTING DOWN")

        self._system_state = SystemState.STOPPING

        # Broadcast a stop command to all services
        await self.send_command_to_service(
            target_service_id=None,
            command=CommandType.SHUTDOWN,
        )

        try:
            await self.service_manager.shutdown_all_services()
        except Exception as e:
            raise self._service_error("Failed to stop all services") from e

        # TODO: This is a hack to give the services time to produce results
        # await asyncio.sleep(3)

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up system controller-specific components."""
        self.logger.debug("Cleaning up System Controller")

        self._system_state = SystemState.SHUTDOWN

    async def start_profiling_all_services(self) -> None:
        """Tell all services to start profiling."""
        self._system_state = SystemState.PROFILING

        self.logger.debug("Starting services")
        for service_info in self.service_manager.service_id_map.values():
            if service_info.state == ServiceState.READY:
                try:
                    await self.send_command_to_service(
                        target_service_id=service_info.service_id,
                        command=CommandType.PROFILE_START,
                    )

                except Exception as e:
                    self.logger.warning("Failed to start service: %s", e)
                    # Continue to the next service
                    # TODO: should we have some sort of retries?
                    continue

    async def _process_profile_stats_message(
        self, message: ProfileStatsMessage
    ) -> None:
        """Process a profile stats message."""
        self.logger.debug("Received profile stats: %s", message)

    async def _process_profile_results_message(
        self, message: ProfileResultsMessage
    ) -> None:
        """Process a profile results message."""
        self.logger.debug("Received profile results: %s", message)
        self.stop_event.set()

    async def _process_registration_message(self, message: RegistrationMessage) -> None:
        """Process a registration message from a service. It will
        add the service to the service manager and send a configure command
        to the service.

        Args:
            message: The registration message to process
        """
        service_id = message.service_id
        service_type = message.service_type

        self.logger.debug(
            f"Processing registration from {service_type} with ID: {service_id}"
        )

        service_info = ServiceRunInfo(
            registration_status=ServiceRegistrationStatus.REGISTERED,
            service_type=service_type,
            service_id=service_id,
            first_seen=time.time_ns(),
            state=ServiceState.READY,
            last_seen=time.time_ns(),
        )

        self.service_manager.service_id_map[service_id] = service_info
        if service_type not in self.service_manager.service_map:
            self.service_manager.service_map[service_type] = []
        self.service_manager.service_map[service_type].append(service_info)

        is_required = service_type in self.required_service_types
        self.logger.debug(
            f"Registered {'required' if is_required else 'non-required'} "
            f"service: {service_type} with ID: {service_id}"
        )

        # Send configure command to the newly registered service
        try:
            await self.send_command_to_service(
                target_service_id=service_id,
                command=CommandType.PROFILE_CONFIGURE,
                data=None,
            )
        except Exception as e:
            raise self._service_error(
                f"Failed to send configure command to {service_type} (ID: {service_id})"
            ) from e

        self.logger.debug(
            f"Sent configure command to {service_type} (ID: {service_id})"
        )

    async def _process_heartbeat_message(self, message: HeartbeatMessage) -> None:
        """Process a heartbeat message from a service. It will
        update the last seen timestamp and state of the service.

        Args:
            message: The heartbeat message to process
        """
        service_id = message.service_id
        service_type = message.service_type
        timestamp = message.request_ns

        self.logger.debug(f"Received heartbeat from {service_type} (ID: {service_id})")

        # Update the last heartbeat timestamp if the component exists
        try:
            service_info = self.service_manager.service_id_map[service_id]
            service_info.last_seen = timestamp
            service_info.state = message.state
            self.logger.debug(f"Updated heartbeat for {service_id} to {timestamp}")
        except Exception:
            self.logger.warning(
                f"Received heartbeat from unknown service: {service_id} ({service_type})"
            )

    async def _process_credits_complete_message(
        self, message: CreditsCompleteMessage
    ) -> None:
        """Process a credits complete message from a service. It will
        update the state of the service with the service manager.

        Args:
            message: The credits complete message to process
        """
        service_id = message.service_id
        self.logger.info("Received credits complete from %s", service_id)

    async def _process_status_message(self, message: StatusMessage) -> None:
        """Process a status message from a service. It will
        update the state of the service with the service manager.

        Args:
            message: The status message to process
        """
        service_id = message.service_id
        service_type = message.service_type
        state = message.state

        self.logger.debug(
            f"Received status update from {service_type} (ID: {service_id}): {state}"
        )

        # Update the component state if the component exists
        if service_id not in self.service_manager.service_id_map:
            self.logger.debug(
                f"Received status update from un-registered service: {service_id} ({service_type})"
            )
            return

        service_info = self.service_manager.service_id_map.get(service_id)
        if service_info is None:
            return

        service_info.state = message.state

        self.logger.debug(f"Updated state for {service_id} to {state}")

    async def send_command_to_service(
        self,
        target_service_id: str | None,
        command: CommandType,
        data: Any | None = None,
        target_service_type: ServiceType | None = None,
    ) -> None:
        """Send a command to a specific service.

        Args:
            target_service_id: ID of the target service, or None to send to all services
            target_service_type: Type of the target service, or None to send to all services
            command: The command to send (from CommandType enum).
            data: Optional data to send with the command.

        Raises:
            CommunicationError: If the communication is not initialized
                or the command was not sent successfully
        """
        if not self._comms:
            self.logger.error("Cannot send command: Communication is not initialized")
            raise CommunicationError(
                CommunicationErrorReason.INITIALIZATION_ERROR,
                "Communication channels are not initialized",
            )

        # Create command message using the helper method
        command_message = self.create_command_message(
            command=command,
            target_service_id=target_service_id,
            target_service_type=target_service_type,
            data=data,
        )

        # Publish command message
        try:
            await self.comms.publish(
                topic=Topic.COMMAND,
                message=command_message,
            )
        except Exception as e:
            self.logger.error("Exception publishing command: %s", e)
            raise CommunicationError(
                CommunicationErrorReason.PUBLISH_ERROR,
                f"Failed to publish command: {e}",
            ) from e


def main() -> None:
    """Main entry point for the system controller."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(SystemController)


if __name__ == "__main__":
    sys.exit(main())
