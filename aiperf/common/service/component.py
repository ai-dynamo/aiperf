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
from abc import ABC, abstractmethod

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import (
    ClientType,
    CommandType,
    PubClientType,
    ServiceState,
    SubClientType,
    Topic,
)
from aiperf.common.models.messages import BaseMessage
from aiperf.common.models.payloads import PayloadType
from aiperf.common.service.base import BaseService


class BaseComponentService(BaseService, ABC):
    """Base class for all component services.

    This class provides a common interface for all component services in the AIPerf framework
    such as the Timing Manager, Dataset Manager, etc.

    It inherits from the BaseService class and implements the required methods for component
    services.
    """

    def __init__(self, service_config: ServiceConfig, service_id: str = None) -> None:
        super().__init__(service_config=service_config, service_id=service_id)

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service.

        The component services subscribe to component messages and publish to the controller.
        """
        return [PubClientType.COMPONENT, SubClientType.CONTROLLER]

    @abstractmethod
    async def configure(self, payload: PayloadType) -> None:
        """Configure the service with the given configuration payload.

        This method is called when a configure command is received from the controller.
        It should be implemented by the derived class to configure the service.

        The service should validate the payload and configure itself accordingly. If successful,
        the service should publish a success message to the controller. On failure, the service
        should publish an error message to the controller.

        Args:
            payload: The configuration payload. This is a union type of all the possible
            configuration payloads.

        """
        pass

    async def run(self) -> None:
        """This method will start the service and initialize its components. It will also subscribe
        to the command topic and process commands as they are received.
        """
        try:
            # Initialize the service
            await self.initialize()

            # Subscribe to the command topic
            await self.comms.subscribe(
                Topic.COMMAND,
                self.process_command_message,
            )

            # TODO: Find a way to wait for the communication to be fully initialized
            # Wait for 1 second to ensure the communication is fully initialized
            await asyncio.sleep(1)

            # Start the heartbeat task
            await self.start_heartbeat_task()

            # Register the service
            await self.register()

            # Wait forever for the stop event to be set
            await self.stop_event.wait()

        except asyncio.exceptions.CancelledError:
            self.logger.debug("Service %s execution cancelled", self.service_type)
        except BaseException:
            self.logger.exception("Service %s execution failed:", self.service_type)
            await self.set_state(ServiceState.ERROR)
        finally:
            # Shutdown the service
            await self.stop()

    async def process_command_message(self, message: BaseMessage) -> None:
        """Process a command message received from the controller.

        This method will process the command message and execute the appropriate action.
        """
        if message.payload.target_service_id not in [None, self.service_id]:
            return  # Ignore commands meant for other services

        cmd = message.payload.command
        if cmd == CommandType.START:
            await self.start()
        elif cmd == CommandType.STOP:
            await self.stop()
        elif cmd == CommandType.CONFIGURE:
            await self.configure(message.payload)
        else:
            self.logger.warning(f"{self.service_type} received unknown command: {cmd}")

    async def set_state(self, state: ServiceState) -> None:
        """Set the state of the service.

        This method will also publish the status message to the status topic if the
        communications are initialized.
        """
        self._state = state
        if self.comms and self.comms.is_initialized:
            await self.comms.publish(
                topic=Topic.STATUS,
                message=self.create_status_message(state),
            )
