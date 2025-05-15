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
from typing import List

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import (
    CommandType,
    ServiceRegistrationStatus,
    ServiceRunType,
    ServiceState,
    ServiceType,
    Topic,
)
from aiperf.common.exceptions.service import ServiceInitializationException
from aiperf.common.models.messages import BaseMessage
from aiperf.common.models.service import ServiceRunInfo
from aiperf.common.service.controller import ControllerServiceBase
from aiperf.services.system_controller.kubernetes_manager import (
    KubernetesServiceManager,
)
from aiperf.services.system_controller.multiprocess_manager import MultiProcessManager
from aiperf.services.system_controller.service_manager import (
    ServiceManagerBase,
)


class SystemController(ControllerServiceBase):
    def __init__(self, service_config: ServiceConfig, service_id: str = None) -> None:
        super().__init__(service_config=service_config, service_id=service_id)

        # List of required service types, in the order they should be started
        self.required_service_types: List[ServiceType] = [
            ServiceType.DATASET_MANAGER,
            ServiceType.TIMING_MANAGER,
            ServiceType.WORKER_MANAGER,
            ServiceType.RECORDS_MANAGER,
            ServiceType.POST_PROCESSOR_MANAGER,
        ]

        self.service_manager: ServiceManagerBase = None

    @property
    def service_type(self) -> ServiceType:
        """The type of service."""
        return ServiceType.SYSTEM_CONTROLLER

    async def _initialize(self) -> None:
        """Initialize system controller-specific components."""
        self.logger.debug("Initializing System Controller")

        if self.service_config.service_run_type == ServiceRunType.MULTIPROCESSING:
            self.service_manager = MultiProcessManager(
                self.required_service_types, self.service_config
            )
        elif self.service_config.service_run_type == ServiceRunType.KUBERNETES:
            self.service_manager = KubernetesServiceManager(
                self.required_service_types, self.service_config
            )
        else:
            raise ValueError(
                f"Unsupported service run type: {self.service_config.service_run_type}"
            )

        # Subscribe to relevant messages
        await self.comms.subscribe(
            topic=Topic.REGISTRATION,
            callback=self._process_registration_message,
        )
        await self.comms.subscribe(
            topic=Topic.HEARTBEAT,
            callback=self._process_heartbeat_message,
        )
        await self.comms.subscribe(
            topic=Topic.STATUS,
            callback=self._process_status_message,
        )

        self.logger.debug(
            "System controller waiting for 1 second to ensure that the communication is initialized"
        )

        # wait 1 second to ensure that the communication is initialized
        await asyncio.sleep(1)

    async def _on_start(self) -> None:
        """Start the system controller and launch required services."""
        self.logger.debug("Starting System Controller")

        # Start all required services
        await self.service_manager.initialize_all_services()

        # Wait for all required services to be registered
        registered = await self.service_manager.wait_for_all_services_registration(
            self.stop_event
        )
        if self.stop_event.is_set():
            self.logger.info("System Controller stopped before all services registered")
            return  # Don't continue with the rest of the initialization
        if not registered:
            self.logger.error(
                "Not all required services registered within the timeout period"
            )
            raise ServiceInitializationException(
                "Not all required services registered within the timeout period"
            )
        else:
            self.logger.info("All required services registered successfully")

        # Wait for all required services to be started
        await self.start_all_services()
        await self.service_manager.wait_for_all_services_start()

    async def start_all_services(self) -> None:
        """Start all required services."""
        self.logger.debug("Starting services")
        for service_info in self.service_manager.service_id_map.values():
            if service_info.state == ServiceState.READY:
                await self.send_command_to_service(
                    target_service_id=service_info.service_id,
                    command=CommandType.START,
                )

    async def _on_stop(self) -> None:
        """Stop the system controller and all running services."""
        self.logger.debug("Stopping System Controller")
        await self.service_manager.stop_all_services()

    async def _cleanup(self) -> None:
        """Clean up system controller-specific components."""
        self.logger.debug("Cleaning up System Controller")
        # TODO: Additional cleanup if needed

    async def _process_registration_message(self, message: BaseMessage) -> None:
        """Process a registration response from a service.

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
        self.logger.info(
            f"Registered {'required' if is_required else 'non-required'} service: {service_type} with ID: {service_id}"
        )

        # Send configure command to the newly registered service
        success = await self.send_command_to_service(
            target_service_id=service_id, command=CommandType.CONFIGURE
        )
        if success:
            self.logger.debug(
                f"Sent configure command to {service_type} (ID: {service_id})"
            )
        else:
            self.logger.warning(
                f"Failed to send configure command to {service_type} (ID: {service_id})"
            )

    async def _process_heartbeat_message(self, message: BaseMessage) -> None:
        """Process a heartbeat response from a service.

        Args:
            message: The heartbeat response to process
        """
        service_id = message.service_id
        service_type = message.payload.service_type
        timestamp = message.timestamp

        self.logger.debug(f"Received heartbeat from {service_type} (ID: {service_id})")

        # Update the last heartbeat timestamp if the component exists
        if service_info := self.service_manager.get(service_id):
            service_info.last_seen = timestamp
            service_info.state = message.payload.state
            self.logger.debug(f"Updated heartbeat for {service_id} to {timestamp}")
        else:
            self.logger.warning(
                f"Received heartbeat from unknown service: {service_id} ({service_type})"
            )

    async def _process_status_message(self, message: BaseMessage) -> None:
        """Process a status response from a service.

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
        if service_info := self.service_manager.get(service_id):
            service_info.state = message.payload.state
            self.logger.debug(f"Updated state for {service_id} to {state}")
        else:
            self.logger.warning(
                f"Received status update from unknown service: {service_id} ({service_type})"
            )

    async def send_command_to_service(
        self, target_service_id: str, command: CommandType
    ) -> bool:
        """Send a command to a specific service.

        Args:
            target_service_id: ID of the target service
            command: The command to send (from CommandType enum)

        Returns:
            True if the command was sent successfully
        """
        if not self.comms:
            self.logger.error("Cannot send command: Communication is not initialized")
            return False

        # Create command response using the helper method
        command_message = self.create_command_message(
            command=command,
            target_service_id=target_service_id,
        )

        # Publish command response
        return await self.comms.publish(
            topic=Topic.COMMAND,
            message=command_message,
        )


def main() -> None:
    from aiperf.common.bootstrap import bootstrap_and_run_service

    bootstrap_and_run_service(SystemController)


if __name__ == "__main__":
    sys.exit(main())
