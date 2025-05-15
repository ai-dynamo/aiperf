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
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import (
    ClientType,
    PubClientType,
    ServiceState,
    SubClientType,
    CommandType,
)
from aiperf.common.models.messages import BaseMessage
from aiperf.common.models.payloads import CommandPayload
from aiperf.common.service.base import BaseService


class BaseControllerService(BaseService):
    """Base class for all controller services, such as the System Controller.

    This class provides a common interface for all controller services in the AIPerf framework.
    It inherits from the BaseService class and implements the required methods for controller
    services.
    """

    def __init__(self, service_config: ServiceConfig, service_id: str = None) -> None:
        super().__init__(service_config=service_config, service_id=service_id)

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service.

        The controller service subscribes to controller messages and publishes to components.
        """
        return [PubClientType.CONTROLLER, SubClientType.COMPONENT]

    def create_command_message(
        self, command: CommandType, target_service_id: str
    ) -> BaseMessage:
        """Create a command message to be sent to a specific service.

        Args:
            command: The command to send
            target_service_id: The ID of the service to send the command to

        Returns:
            A command message
        """
        return self.create_message(
            CommandPayload(command=command, target_service_id=target_service_id)
        )

    # TODO: Complete the implementation of the controller service methods
    async def run(self) -> None:
        """This method will be the primary entry point for the service
        and will be called by the bootstrap script. It does not return until the service
        is completely shutdown."""

        try:
            # Initialize the service
            await self.initialize()

            # Start the service
            await self.start()

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
