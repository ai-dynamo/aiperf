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
    ServiceState,
    Topic,
)
from aiperf.common.models.messages import BaseMessage
from aiperf.common.models.payloads import (
    HeartbeatPayload,
    PayloadType,
    RegistrationPayload,
    StatusPayload,
)
from aiperf.common.service.abstract import AbstractBaseService


class BaseService(AbstractBaseService):
    """Base class for all AIPerf services, providing common functionality for communication,
    state management, and lifecycle operations.

    This class provides the foundation for implementing the various services of the AIPerf system.
    Some of the abstract methods are implemented here, while others are still required to be
    implemented by derived classes.
    """

    def __init__(self, service_config: ServiceConfig, service_id: str = None):
        self.service_id: str = service_id or uuid.uuid4().hex
        self.service_config = service_config

        self.logger = logging.getLogger(self.service_type)
        self.logger.debug(
            f"Initializing {self.service_type} service (id: {self.service_id})"
        )

        self._state: ServiceState = ServiceState.UNKNOWN

        self._heartbeat_task = None
        self._heartbeat_interval = self.service_config.heartbeat_interval

        self.stop_event = asyncio.Event()
        self.comms: Optional[BaseCommunication] = None

        # Set to store signal handler tasks
        self._signal_tasks = set()

    @property
    def state(self) -> ServiceState:
        """The current state of the service."""
        return self._state

    # Note: Not using as a setter so it can be overridden by derived classes and still be async
    async def set_state(self, state: ServiceState) -> None:
        """Set the state of the service."""
        self._state = state

    async def initialize(self) -> None:
        """Initialize the service communication and signal handlers."""

        # Allow time for the event loop to start
        await asyncio.sleep(0.1)

        self._state = ServiceState.INITIALIZING

        # Set up signal handlers for graceful shutdown
        self.setup_signal_handlers()

        # Initialize communication
        self.comms = CommunicationFactory.create_communication(self.service_config)
        success = await self.comms.initialize()
        if not success:
            self.logger.error(
                f"{self.service_type}: Failed to initialize {self.service_config.comm_backend} communication"
            )
            self._state = ServiceState.ERROR
            return

        if len(self.required_clients) > 0:
            # Create the communication clients ahead of time
            self.logger.debug(
                "%s: Creating communication clients (%s)",
                self.service_type,
                self.required_clients,
            )
            await self.comms.create_clients(*self.required_clients)

        await self.set_state(ServiceState.READY)

    async def send_heartbeat(self) -> None:
        """Send a heartbeat notification to the system controller."""
        heartbeat_message = self.create_heartbeat_message()
        self.logger.debug("Sending heartbeat: %s", heartbeat_message)
        await self.comms.publish(
            topic=Topic.HEARTBEAT,
            message=heartbeat_message,
        )

    async def register(self) -> None:
        """Publish a registration request to the system controller.

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

    async def start(self) -> None:
        """Start the service and its components.

        This method should be called to start the service after it has been initialized and configured.
        """
        self.logger.debug(
            "Starting %s service (id: %s)", self.service_type, self.service_id
        )
        await self.set_state(ServiceState.STARTING)

        try:
            await self._on_start()
            await self.set_state(ServiceState.RUNNING)
        except Exception:
            self.logger.exception(
                "Failed to start service %s (id: %s)",
                self.service_type,
                self.service_id,
            )
            await self.set_state(ServiceState.ERROR)
            raise

    async def stop(self) -> None:
        """Stop the service and clean up its components. It will also cancel the
        heartbeat task if it is running.
        """
        if self.state == ServiceState.STOPPED:
            self.logger.warning(
                "Service %s state %s is already STOPPED, ignoring stop request",
                self.service_type,
                self.state,
            )
            return

        await self.set_state(ServiceState.STOPPING)

        # Signal the run method to exit if it hasn't already
        if not self.stop_event.is_set():
            self.stop_event.set()

        # Cancel heartbeat task if running
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task

        # Custom stop logic implemented by derived classes
        await self._on_stop()

        # Shutdown communication component
        if self.comms and not self.comms.is_shutdown:
            await self.comms.shutdown()

        # Custom cleanup logic implemented by derived classes
        await self._cleanup()

        # Set the state to STOPPED. Communications are shutdown, so we don't need to
        # publish a status message
        self._state = ServiceState.STOPPED
        self.logger.debug(
            "Service %s (id: %s) stopped", self.service_type, self.service_id
        )

    def create_message(
        self, payload: PayloadType, request_id: Optional[str] = None
    ) -> BaseMessage:
        """Create a message of the given type, and pre-fill the service_id.

        Args:
            payload: The payload of the message
            Optional[request_id]: The request id of the message this is a response to

        Returns:
            A message of the given type
        """
        message = BaseMessage(
            service_id=self.service_id,
            request_id=request_id,
            payload=payload,
        )
        return message

    def create_heartbeat_message(self) -> BaseMessage:
        """Create a heartbeat notification message."""
        return self.create_message(
            HeartbeatPayload(
                service_type=self.service_type,
            )
        )

    def create_registration_message(self) -> BaseMessage:
        """Create a registration request message."""
        return self.create_message(
            RegistrationPayload(
                service_type=self.service_type,
            )
        )

    def create_status_message(self, state: ServiceState) -> BaseMessage:
        """Create a status notification message."""
        return self.create_message(
            StatusPayload(
                state=state,
                service_type=self.service_type,
            )
        )

    async def start_heartbeat_task(self) -> None:
        """Start a background task to send heartbeats at regular intervals."""

        async def heartbeat_loop() -> None:
            while not self.stop_event.is_set():
                # Sleep first to avoid sending a heartbeat before the registration message
                # has been published
                await asyncio.sleep(self._heartbeat_interval)
                await self.send_heartbeat()

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())
        self.logger.debug(
            "%s: Started heartbeat task with interval %fs",
            self.service_type,
            self._heartbeat_interval,
        )

    def setup_signal_handlers(self) -> None:
        """This method will set up signal handlers for the SIGTERM and SIGINT signals in order
        to trigger a graceful shutdown of the service.
        """
        loop = asyncio.get_running_loop()

        def signal_handler(signal: int) -> None:
            # Create a task and store it so it doesn't get garbage collected
            task = asyncio.create_task(self.handle_signal(signal))
            # Store the task somewhere to prevent it from being garbage collected
            # before it completes
            self._signal_tasks.add(task)
            task.add_done_callback(self._signal_tasks.discard)

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))

    async def handle_signal(self, signal: int) -> None:
        """Handle received signals by triggering graceful shutdown.

        Args:
            signal: The signal number received
        """
        signal_name = signal.Signals(signal).name
        self.logger.debug(
            "%s: Received signal %s, initiating graceful shutdown",
            self.service_type,
            signal_name,
        )

        await self.stop()
