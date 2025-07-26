# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from aiperf.common.config import ServiceConfig, UserConfig
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommandType, LifecycleState
from aiperf.common.hooks import (
    background_task,
    on_command,
    on_start,
    on_state_change,
)
from aiperf.common.messages import (
    CommandMessage,
    HeartbeatMessage,
    RegistrationMessage,
    StatusMessage,
)
from aiperf.common.protocols import ServiceProtocol
from aiperf.services.base_service import BaseService


@implements_protocol(ServiceProtocol)
class BaseComponentService(BaseService):
    """Base class for all Component services.

    This class provides a common interface for all Component services in the AIPerf
    framework such as the Timing Manager, Dataset Manager, etc.

    It extends the BaseService by adding heartbeat and registration functionality, as well as
    publishing the current state of the service to the system controller.
    """

    def __init__(
        self,
        service_config: ServiceConfig,
        user_config: UserConfig,
        service_id: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            service_config=service_config,
            user_config=user_config,
            service_id=service_id,
            **kwargs,
        )

    @background_task(
        interval=lambda self: self.service_config.heartbeat_interval_seconds,
        immediate=False,
    )
    async def _heartbeat_task(self) -> None:
        """Send a heartbeat notification to the system controller."""
        await self.publish(
            HeartbeatMessage(
                service_id=self.service_id,
                service_type=self.service_type,
                state=self.state,
            )
        )

    @on_start
    async def _register_service(self) -> None:
        """Publish a registration request to the system controller.

        This method should be called after the service has been initialized and is
        ready to start processing messages.
        """
        self.debug(
            lambda: f"Attempting to register service {self} ({self.service_id}) with system controller"
        )
        await self.publish(
            RegistrationMessage(
                service_id=self.service_id,
                service_type=self.service_type,
                state=self.state,
            )
        )

    @on_state_change
    async def _on_state_change(
        self, old_state: LifecycleState, new_state: LifecycleState
    ) -> None:
        """Action to take when the service state is set.

        This method will also publish the status message to the status message_type if the
        communications are initialized.
        """
        if self.stop_requested:
            return
        if not self.comms.was_initialized:
            return
        await self.publish(
            StatusMessage(
                service_id=self.service_id,
                service_type=self.service_type,
                state=new_state,
            )
        )

    @on_command(CommandType.SHUTDOWN)
    async def _on_shutdown_command(self, message: CommandMessage) -> None:
        self.debug(f"Received shutdown command: {message}, {self.service_id}")
        try:
            await self.stop()
        except Exception as e:
            self.warning(
                f"Failed to stop service {self} ({self.service_id}) after receiving shutdown command: {e}. Killing."
            )
            await self._kill()
        raise asyncio.CancelledError()
