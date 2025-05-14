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
import contextlib
import logging
import signal
import uuid
from typing import Optional

from aiperf.common.comms.communication import BaseCommunication
from aiperf.common.comms.communication_factory import CommunicationFactory
from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import (
    CommandType,
    ServiceState,
    Topic,
)
from aiperf.common.models.messages import BaseMessage
from aiperf.common.models.payloads import (
    HeartbeatPayload,
    PayloadType,
    RegistrationPayload,
    StatusPayload,
    CommandPayload,
)
from aiperf.common.service.abstract import AbstractBaseService


class BaseService(AbstractBaseService):
    """Base class for all AIPerf services, providing common functionality for communication,
    state management, and lifecycle operations.

    This class provides the foundation for implementing the various components of the AIPerf system,
    such as the System Controller, Dataset Manager, Timing Manager, Worker Manager, etc.
    """

    def __init__(self, service_config: ServiceConfig, service_id: str = None):
        self.service_id: str = service_id or uuid.uuid4().hex
        self.service_config = service_config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            f"Initializing service {self.service_id} {self.service_type} {self.__class__.__name__}"
        )
        self._state: ServiceState = ServiceState.UNKNOWN
        self._heartbeat_task = None
        self._heartbeat_interval = (
            self.service_config.heartbeat_interval
        )  # Default interval in seconds
        self.stop_event = asyncio.Event()
        self.comms: Optional[BaseCommunication] = None
        # Set to store signal handler tasks
        self._signal_tasks = set()

    @property
    def state(self) -> ServiceState:
        """The current state of the service."""
        return self._state

    async def set_state(self, state: ServiceState) -> None:
        """Set the state of the service."""
        self._state = state
        if self.comms.is_initialized:
            await self.comms.publish(
                topic=Topic.STATUS,
                message=self.create_status_message(state),
            )

    def create_message(
        self, payload: PayloadType, request_id: Optional[str] = None
    ) -> BaseMessage:
        """Create a response of the given type.

        Pre-fills the service_id and service_type.

        Args:
            payload: The payload of the response
            Optional[request_id]: The request id of the response this is a response to

        Returns:
            A response of the given type
        """
        message = BaseMessage(
            service_id=self.service_id,
            request_id=request_id,
            payload=payload,
        )
        return message

    def create_heartbeat_message(self) -> BaseMessage:
        """Create a heartbeat response."""
        return self.create_message(
            HeartbeatPayload(
                service_type=self.service_type,
            )
        )

    def create_registration_message(self) -> BaseMessage:
        """Create a registration response."""
        return self.create_message(
            RegistrationPayload(
                service_type=self.service_type,
            )
        )

    def create_status_message(self, state: ServiceState) -> BaseMessage:
        """Create a status response."""
        return self.create_message(
            StatusPayload(
                state=state,
                service_type=self.service_type,
            )
        )

    def create_command_message(
        self, command: CommandType, target_service_id: str
    ) -> BaseMessage:
        """Create a command response."""
        return self.create_message(
            CommandPayload(command=command, target_service_id=target_service_id)
        )

    async def _base_init(self) -> None:
        """Initialize the service communication and signal handlers.

        This method should be called by derived classes to initialize the service.
        """
        await asyncio.sleep(0.1)  # Allow time for the event loop to start

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Initialize the service
        self._state = ServiceState.INITIALIZING

        # Initialize communication
        self.comms = CommunicationFactory.create_communication(self.service_config)
        success = await self.comms.initialize()
        if not success:
            self.logger.error(
                f"Failed to initialize {self.service_config.comm_backend} communication"
            )
            self._state = ServiceState.ERROR
            return

        self.logger.debug(
            "Creating communication clients (%s), service: %s",
            self.required_clients,
            self.service_type,
        )
        # Create the communication clients ahead of time
        await self.comms.create_clients(*self.required_clients)

    async def _send_heartbeat(self) -> None:
        """Send a heartbeat response to the system controller."""
        heartbeat_message = self.create_heartbeat_message()
        self.logger.debug("Sending heartbeat: %s", heartbeat_message)
        await self.comms.publish(
            topic=Topic.HEARTBEAT,
            message=heartbeat_message,
        )

    async def _start_heartbeat_task(self) -> None:
        """Start a background task to send heartbeats at regular intervals."""

        async def heartbeat_loop() -> None:
            while True:
                await asyncio.sleep(self._heartbeat_interval)
                await self._send_heartbeat()

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        self.logger.debug(
            "Started heartbeat task with interval %ss", self._heartbeat_interval
        )

    async def _register(self) -> None:
        """Register the service with the system controller.

        This method should be called after the service has been initialized and is ready to
        start processing messages.
        """
        self.logger.debug(
            "Attempting to register service %s (%s) with system controller",
            self.service_type,
            self.service_id,
        )
        await self.comms.publish(
            topic=Topic.REGISTRATION,
            message=self.create_registration_message(),
        )

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()

        def signal_handler(sig: int) -> None:
            # Create a task and store it so it doesn't get garbage collected
            task = asyncio.create_task(self._handle_signal(sig))
            # Store the task somewhere to prevent it from being garbage collected
            # before it completes
            self._signal_tasks.add(task)
            task.add_done_callback(self._signal_tasks.discard)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

        # self.logger.debug("Signal handlers set up for graceful shutdown")

    async def _handle_signal(self, sig: int) -> None:
        """Handle received signals by triggering graceful shutdown.

        Args:
            sig: The signal number received
        """
        sig_name = signal.Signals(sig).name
        self.logger.debug(f"Received signal {sig_name}, initiating graceful shutdown")

        # Stop the service if it's running
        if self._state == ServiceState.RUNNING:
            await self.stop()
        else:
            # Just set the stop event to break out of the run loop
            self.stop_event.set()

    async def _start(self) -> None:
        """Start the service and its components.

        This method should be called to start the service after it has been initialized.
        """
        self.logger.debug("Starting %s service %s", self.service_type, self.service_id)
        await self.set_state(ServiceState.STARTING)
        try:
            await self._on_start()
            await self.set_state(ServiceState.RUNNING)
        except BaseException:
            self.logger.exception(
                "Failed to start service %s: %s", self.service_type, self.service_id
            )
            await self.set_state(ServiceState.ERROR)
            raise

    async def stop(self) -> None:
        """Stop the service and clean up its components."""
        if self.state != ServiceState.RUNNING:
            self.logger.warning(
                "Service %s is not running, cannot stop", self.service_type
            )
            return
        await self.set_state(ServiceState.STOPPING)
        # Signal the run method to exit if it hasn't already
        self.stop_event.set()

        # Cancel heartbeat task if running
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        await self._on_stop()

        # Shutdown communication component
        if self.comms:
            await self.comms.shutdown()

        await self._cleanup()

        self._state = ServiceState.STOPPED
