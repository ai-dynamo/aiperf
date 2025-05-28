# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.comms.client_enums import (
    ClientType,
    PubClientType,
    SubClientType,
)
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import CommandType
from aiperf.common.hooks import on_run
from aiperf.common.models import CommandPayload, Message
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
        data: Any | None = None,
    ) -> Message:
        """Create a command message to be sent to a specific service.

        Args:
            command: The command to send
            target_service_id: The ID of the service to send the command to
            data: Optional data to send with the command.

        Returns:
            A command message
        """
        return self.create_message(
            CommandPayload(
                command=command,
                target_service_id=target_service_id,
                data=data,
            )
        )
