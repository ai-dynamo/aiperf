#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import cast

from aiperf.common.comms.client_enums import (
    ClientType,
    PubClientType,
    SubClientType,
)
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import CommandType, ServiceState, Topic
from aiperf.common.exceptions import (
    CommunicationSubscribeError,
    ServiceHeartbeatError,
    ServiceRegistrationError,
)
from aiperf.common.hooks import AIPerfHook, aiperf_task, on_run, on_set_state
from aiperf.common.models import (
    CommandMessage,
    HeartbeatMessage,
    HeartbeatPayload,
    RegistrationMessage,
    RegistrationPayload,
    StatusMessage,
    StatusPayload,
)
from aiperf.common.service.base_service import BaseService


class BaseComponentService(BaseService):
    """Base class for all Component services.

    This class provides a common interface for all Component services in the AIPerf
    framework such as the Timing Manager, Dataset Manager, etc.

    It extends the BaseService by:
    - Subscribing to the command topic
    - Processing command messages
    - Sending registration requests to the system controller
    - Sending heartbeat notifications to the system controller
    - Sending status notifications to the system controller
    - Helpers to create heartbeat, registration, and status messages
    - Request the appropriate communication clients for a component service
    """

    def __init__(
        self, service_config: ServiceConfig, service_id: str | None = None
    ) -> None:
        super().__init__(service_config=service_config, service_id=service_id)

    @property
    def required_clients(self) -> list[ClientType]:
        """The communication clients required by the service.

        The component services subscribe to controller messages and publish
        component messages.
        """
        return [
            *(super().required_clients or []),
            PubClientType.COMPONENT,
            SubClientType.CONTROLLER,
        ]

    # TODO: The configure method is turning into a service hook
    # @abstractmethod
    # async def _configure(self, payload: Payload) -> None:
    #     """Configure the service with the given configuration payload.

    #     This method is called when a configure command is received from the controller.
    #     It should be implemented by the derived class to configure the service.

    #     The service should validate the payload and configure itself accordingly.
    #     If successful, the service should publish a success message to the controller.
    #     On failure, the service should publish an error message to the controller.

    #     Args:
    #         payload: The configuration payload. This is a union type of all the possible
    #         configuration payloads.

    #     """
    #     pass

    @on_run
    async def _on_run(self) -> None:
        """Automatically subscribe to the command topic and register the service
        with the system controller when the run hook is called.

        This method will:
        - Subscribe to the command topic
        - Wait for the communication to be fully initialized
        - Register the service with the system controller
        """
        # Subscribe to the command topic
        try:
            await self.comms.subscribe(
                Topic.COMMAND,
                self.process_command_message,
            )
        except Exception as e:
            self.logger.error("Exception subscribing to command topic: %s", e)
            raise CommunicationSubscribeError(
                "Failed to subscribe to command topic"
            ) from e

        # TODO: Find a way to wait for the communication to be fully initialized
        # FIXME: This is a hack to ensure the communication is fully initialized
        await asyncio.sleep(1)

        # Register the service
        try:
            await self.register()
            await asyncio.sleep(0.5)
        except Exception as e:
            raise ServiceRegistrationError() from e

    @aiperf_task
    async def _heartbeat_task(self) -> None:
        """Starts a background task to send heartbeats at regular intervals. It
        will continue to send heartbeats even if an error occurs until the stop
        event is set.
        """
        while not self.stop_event.is_set():
            # Sleep first to avoid sending a heartbeat before the registration
            # message has been published
            await asyncio.sleep(self._heartbeat_interval)

            try:
                await self.send_heartbeat()
            except Exception as e:
                self.logger.warning("Exception sending heartbeat: %s", e)
                # continue to keep sending heartbeats regardless of the error

        self.logger.debug("Heartbeat task stopped")

    async def send_heartbeat(self) -> None:
        """Send a heartbeat notification to the system controller."""
        heartbeat_message = self.create_heartbeat_message()
        self.logger.debug("Sending heartbeat: %s", heartbeat_message)
        try:
            await self.comms.publish(
                topic=Topic.HEARTBEAT,
                message=heartbeat_message,
            )
        except Exception as e:
            raise ServiceHeartbeatError from e

    async def register(self) -> None:
        """Publish a registration request to the system controller.

        This method should be called after the service has been initialized and is
        ready to start processing messages.
        """
        self.logger.debug(
            "Attempting to register service %s (%s) with system controller",
            self.service_type,
            self.service_id,
        )
        try:
            await self.comms.publish(
                topic=Topic.REGISTRATION,
                message=self.create_registration_message(),
            )
        except Exception as e:
            raise ServiceRegistrationError() from e

    async def process_command_message(self, message: CommandMessage) -> None:
        """Process a command message received from the controller.

        This method will process the command message and execute the appropriate action.
        """
        if message.payload.target_service_id not in [None, self.service_id]:
            return  # Ignore commands meant for other services

        cmd = message.payload.command
        if cmd == CommandType.START:
            await self.start()

        elif cmd == CommandType.STOP:
            self.stop_event.set()

        elif cmd == CommandType.CONFIGURE:
            await self._run_hooks(AIPerfHook.CONFIGURE, message)

        else:
            self.logger.warning(f"{self.service_type} received unknown command: {cmd}")

    @on_set_state
    async def _on_set_state(self, state: ServiceState) -> None:
        """Action to take when the service state is set.

        This method will also publish the status message to the status topic if the
        communications are initialized.
        """
        if self._comms and self._comms.is_initialized:
            await self.comms.publish(
                topic=Topic.STATUS,
                message=self.create_status_message(state),
            )

    def create_heartbeat_message(self) -> HeartbeatMessage:
        """Create a heartbeat notification message."""
        return cast(
            HeartbeatMessage,
            self.create_message(
                HeartbeatPayload(
                    service_type=self.service_type,
                )
            ),
        )

    def create_registration_message(self) -> RegistrationMessage:
        """Create a registration request message."""
        return cast(
            RegistrationMessage,
            self.create_message(
                RegistrationPayload(
                    service_type=self.service_type,
                )
            ),
        )

    def create_status_message(self, state: ServiceState) -> StatusMessage:
        """Create a status notification message."""
        return cast(
            StatusMessage,
            self.create_message(
                StatusPayload(
                    state=state,
                    service_type=self.service_type,
                )
            ),
        )
