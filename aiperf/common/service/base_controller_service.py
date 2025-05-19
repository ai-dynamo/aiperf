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

from typing import Any

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.decorators import on_run
from aiperf.common.enums import (
    ClientType,
    CommandType,
    PubClientType,
    SubClientType,
)
from aiperf.common.models.message_models import Message
from aiperf.common.models.payload_models import CommandPayload
from aiperf.common.service.base_service import BaseService


class BaseControllerService(BaseService):
    """Base class for all controller services, such as the System Controller.

    This class provides a common interface for all controller services in the AIPerf
    framework. It inherits from the BaseService class and implements the required
    methods for controller services.

    It extends the BaseService by:
    - Starting the service automatically when the run hook is called
    - Helpers to create command messages to be sent to a specific service
    - Request the appropriate communication clients for a controller service
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service.

        The controller service subscribes to controller messages and publishes
        to components.
        """
        return [
            *(super().required_clients or []),
            PubClientType.CONTROLLER,
            SubClientType.COMPONENT,
        ]

    @on_run
    async def _on_run(self) -> None:
        """Automatically start the service when the run hook is called."""
        await self.start()

    def create_command_message(
        self,
        command: CommandType,
        target_service_id: str,
        payload: Any | None = None,
    ) -> Message:
        """Create a command message to be sent to a specific service.

        Args:
            command: The command to send
            target_service_id: The ID of the service to send the command to
            payload: Optional payload to send with the command.

        Returns:
            A command message
        """
        return self.create_message(
            CommandPayload(command=command, target_service_id=target_service_id)
        )
