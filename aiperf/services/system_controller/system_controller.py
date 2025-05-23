#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import asyncio
import sys
import time
from typing import Any

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.decorators import (
    on_cleanup,
    on_init,
    on_start,
    on_stop,
)
from aiperf.common.enums import (
    CommandType,
    ServiceRegistrationStatus,
    ServiceRunType,
    ServiceState,
    ServiceType,
    Topic,
)
from aiperf.common.exceptions import (
    CommunicationNotInitializedError,
    CommunicationPublishError,
    CommunicationSubscribeError,
    ConfigError,
    ServiceConfigureError,
    ServiceInitializationError,
    ServiceStopError,
)
from aiperf.common.models import (
    HeartbeatMessage,
    RegistrationMessage,
    ServiceRunInfo,
    StatusMessage,
)
from aiperf.common.service.base_controller_service import BaseControllerService
from aiperf.services.service_manager.base import BaseServiceManager
from aiperf.services.service_manager.kubernetes import (
    KubernetesServiceManager,
)
from aiperf.services.service_manager.multiprocess import (
    MultiProcessServiceManager,
)


class SystemController(BaseControllerService):
    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)
        self.logger.debug("Creating System Controller")

        # List of required service types, in no particular order
        self.required_service_types: list[ServiceType] = [
            ServiceType.DATASET_MANAGER,
            ServiceType.TIMING_MANAGER,
            ServiceType.WORKER_MANAGER,
            ServiceType.RECORDS_MANAGER,
            ServiceType.POST_PROCESSOR_MANAGER,
        ]

        self.service_manager: BaseServiceManager | None = None
        self.logger.debug("System Controller created")

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.SYSTEM_CONTROLLER

    @on_init
    async def _initialize(self) -> None:
        """Initialize system controller-specific components.

        This method will:
        - Initialize the service manager
        - Subscribe to relevant messages
        """
        self.logger.debug("Initializing System Controller")

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
        try:
            await self.comms.subscribe(
                topic=Topic.REGISTRATION,
                callback=self._process_registration_message,
            )
        except Exception as e:
            self.logger.error("Failed to subscribe to registration topic: %s", e)
            raise CommunicationSubscribeError from e

        try:
            await self.comms.subscribe(
                topic=Topic.HEARTBEAT,
                callback=self._process_heartbeat_message,
            )
        except Exception as e:
            self.logger.error("Failed to subscribe to heartbeat topic: %s", e)
            raise CommunicationSubscribeError from e

        try:
            await self.comms.subscribe(
                topic=Topic.STATUS,
                callback=self._process_status_message,
            )
        except Exception as e:
            self.logger.error("Failed to subscribe to status topic: %s", e)
            raise CommunicationSubscribeError from e

        self.logger.debug(
            "System controller waiting for 1 second to ensure that the "
            "communication is initialized"
        )

        # wait 1 second to ensure that the communication is initialized
        await asyncio.sleep(1)

    @on_start
    async def _start(self) -> None:
        """Start the system controller and launch required services.

        This method will:
        - Initialize all required services
        - Wait for all required services to be registered
        - Start all required services
        """
        self.logger.debug("Starting System Controller")

        # Start all required services
        try:
            await self.service_manager.initialize_all_services()
        except Exception as e:
            self.logger.error("Failed to initialize all services: %s", e)
            raise ServiceInitializationError from e

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
            self.logger.error(
                "Not all required services registered within the timeout period"
            )
            raise ServiceInitializationError(
                "Not all required services registered within the timeout period"
            ) from e

        self.logger.debug("All required services registered successfully")

        self.logger.info("AIPerf System is READY")
        # Wait for all required services to be started
        await self.start_all_services()
        try:
            await self.service_manager.wait_for_all_services_start()
        except Exception as e:
            self.logger.error("Failed to wait for all services to start: %s", e)
            raise ServiceInitializationError from e

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

        try:
            await self.service_manager.stop_all_services()
        except Exception as e:
            self.logger.error("Failed to stop all services: %s", e)
            raise ServiceStopError from e

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up system controller-specific components."""
        self.logger.debug("Cleaning up System Controller")
        # TODO: Additional cleanup if needed

    async def start_all_services(self) -> None:
        """Start all required services."""
        self.logger.debug("Starting services")
        for service_info in self.service_manager.service_id_map.values():
            if service_info.state == ServiceState.READY:
                try:
                    await self.send_command_to_service(
                        target_service_id=service_info.service_id,
                        command=CommandType.START,
                    )

                except Exception as e:
                    self.logger.warning("Failed to start service: %s", e)
                    # Continue to the next service
                    # TODO: should we have some sort of retries?
                    continue

    async def _process_registration_message(self, message: RegistrationMessage) -> None:
        """Process a registration response from a service. It will
        add the service to the service manager and send a configure command
        to the service.

        Args:
            message: The registration response to process
        """
        service_id = message.service_id
        service_type = message.payload.service_type

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
        # TODO: Retrieve the configuration to send to the service
        try:
            await self.send_command_to_service(
                target_service_id=service_id,
                command=CommandType.CONFIGURE,
                data=None,
            )
        except Exception as e:
            self.logger.error(
                f"Failed to send configure command to {service_type} (ID: {service_id}): {e}"
            )
            raise ServiceConfigureError from e

        self.logger.debug(
            f"Sent configure command to {service_type} (ID: {service_id})"
        )

    async def _process_heartbeat_message(self, message: HeartbeatMessage) -> None:
        """Process a heartbeat response from a service. It will
        update the last seen timestamp and state of the service.

        Args:
            message: The heartbeat response to process
        """
        service_id = message.service_id
        service_type = message.payload.service_type
        timestamp = message.timestamp

        self.logger.debug(f"Received heartbeat from {service_type} (ID: {service_id})")

        # Update the last heartbeat timestamp if the component exists
        try:
            service_info = self.service_manager.service_id_map.get(service_id)
            service_info.last_seen = timestamp
            service_info.state = message.payload.state
            self.logger.debug(f"Updated heartbeat for {service_id} to {timestamp}")
        except Exception:
            self.logger.warning(
                f"Received heartbeat from unknown service: {service_id} ({service_type})"
            )

    async def _process_status_message(self, message: StatusMessage) -> None:
        """Process a status response from a service. It will
        update the state of the service with the service manager.

        Args:
            message: The status response to process
        """
        service_id = message.service_id
        service_type = message.payload.service_type
        state = message.payload.state

        self.logger.debug(
            f"Received status update from {service_type} (ID: {service_id}): {state}"
        )

        # Update the component state if the component exists
        try:
            service_info = self.service_manager.service_id_map.get(service_id)
            service_info.state = message.payload.state
            self.logger.debug(f"Updated state for {service_id} to {state}")
        except Exception:
            self.logger.warning(
                f"Received status update from un-registered service: {service_id} ({service_type})"
            )

    async def send_command_to_service(
        self,
        target_service_id: str,
        command: CommandType,
        data: Any | None = None,
    ) -> None:
        """Send a command to a specific service.

        Args:
            target_service_id: ID of the target service
            command: The command to send (from CommandType enum).
            data: Optional data to send with the command.

        Raises:
            CommunicationNotInitializedError if the communication is not initialized
            CommunicationPublishError if the command was not sent successfully
        """
        if not self._comms:
            self.logger.error("Cannot send command: Communication is not initialized")
            raise CommunicationNotInitializedError()

        # Create command response using the helper method
        command_message = self.create_command_message(
            command=command,
            target_service_id=target_service_id,
            data=data,
        )

        # Publish command response
        try:
            await self.comms.publish(
                topic=Topic.COMMAND,
                message=command_message,
            )
        except Exception as e:
            self.logger.error("Exception publishing command: %s", e)
            raise CommunicationPublishError from e


def main() -> None:
    """Main entry point for the system controller."""

    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(SystemController)


if __name__ == "__main__":
    sys.exit(main())
