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


class ComponentServiceBase(BaseService, ABC):
    """Base class for all component services.
    This class provides a common interface for all component services in the AIPerf framework.
    It inherits from the ServiceBase class and implements the required methods for component
    services.
    """

    def __init__(self, service_config: ServiceConfig, service_id: str = None) -> None:
        super().__init__(service_config=service_config, service_id=service_id)

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service."""
        # The component service subscribes to component messages and publishes to the controller
        return [PubClientType.COMPONENT, SubClientType.CONTROLLER]

    @abstractmethod
    async def _configure(self, payload: PayloadType) -> None:
        """Configure the service.

        This method is called when a configure command is received from the controller.
        It should be implemented by the derived class to configure the service.
        """
        pass

    async def run(self) -> None:
        """Start the service and initialize its components."""
        try:
            await self._base_init()
            # Set up communication subscriptions if communication is available
            # Subscribe to common topics
            await self.comms.subscribe(
                Topic.COMMAND,
                self._process_command_message,
            )

            await self._initialize()

            # TODO: Find a way to wait for the communication to be fully initialized
            # Wait for 1 second to ensure the communication is fully initialized
            await asyncio.sleep(1)

            # Additional service-specific subscriptions can be added in derived classes

            await self._register()
            # Start heartbeat task
            await self._start_heartbeat_task()
            await self.set_state(ServiceState.READY)

            # Wait forever for the stop event to be set
            await self.stop_event.wait()
        except asyncio.exceptions.CancelledError:
            self.logger.debug("Service execution cancelled")
        except BaseException:
            self.logger.exception("Service execution failed:")
            await self.set_state(ServiceState.ERROR)
        finally:
            # Make sure to clean up properly even if there was an error
            if self.state == ServiceState.RUNNING:
                await self.stop()

    async def _process_command_message(self, message: BaseMessage) -> None:
        """Process a command response."""
        if message.payload.target_service_id not in [None, self.service_id]:
            return  # Ignore commands for other services

        cmd = message.payload.command
        if cmd == CommandType.START:
            await self._on_start()
        elif cmd == CommandType.STOP:
            await self.stop()
        elif cmd == CommandType.CONFIGURE:
            await self._configure(message.payload)
        else:
            self.logger.warning(f"Received unknown command: {cmd}")

    async def set_state(self, status: ServiceState) -> None:
        """Send a service state response to the system controller."""
        self._state = status
        status_message = self.create_status_message(self.state)
        await self.comms.publish(
            topic=Topic.STATUS,
            message=status_message,
        )
