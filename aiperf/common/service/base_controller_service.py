# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from aiperf.common.config import ServiceConfig
from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums import CommandType, ServiceType
from aiperf.common.hooks import on_run
from aiperf.common.messages import CommandMessage
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
        self,
        service_config: ServiceConfig,
        user_config: UserConfig | None = None,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

    @on_run
    async def _on_run(self) -> None:
        """Automatically start the service when the run hook is called."""
        await self.start()

    def create_command_message(
        self,
        command: CommandType,
        target_service_id: str | None,
        target_service_type: ServiceType | None = None,
        data: BaseModel | None = None,
    ) -> CommandMessage:
        """Create a command message to be sent to a specific service.

        Args:
            command: The command to send
            target_service_id: The ID of the service to send the command to
            target_service_type: The type of the service to send the command to
            data: Optional data to send with the command.

        Returns:
            A command message
        """
        return CommandMessage(
            service_id=self.service_id,
            command=command,
            target_service_id=target_service_id,
            target_service_type=target_service_type,
            data=data,
        )
