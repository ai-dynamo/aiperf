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
import uuid
from abc import ABC, abstractmethod

from aiperf.common.config.service_config import ServiceConfig
from aiperf.common.enums import ServiceState, Topic
from aiperf.common.models.messages import (
    BaseMessage,
    HeartbeatMessage,
    RegistrationMessage,
    StatusMessage,
)


class BaseService(ABC):
    """Base class for all AIPerf services, providing common functionality for communication,
    state management, and lifecycle operations.

    This class provides the foundation for implementing the various components of the AIPerf system,
    such as the System Controller, Dataset Manager, Timing Manager, Worker Manager, etc.
    """

    def __init__(
        self,
        service_type: str,
        config: ServiceConfig,
        autostart: bool = False,
    ):
        self.service_id: str = uuid.uuid4().hex
        self.service_type: str = service_type
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug(
            f"Initializing service {self.service_id} {self.service_type} {self.__class__.__name__}"
        )
        self.state: ServiceState = ServiceState.UNKNOWN
        self.heartbeat_task = None
        self.heartbeat_interval = 10  # Default interval in seconds
        self.stop_event = asyncio.Event()
        self.autostart = autostart

    async def _subscribe_to_topic(self, topic: Topic) -> None:
        """Subscribe to a topic for receiving messages.

        Args:
            topic: The topic to subscribe to

        """
        # TODO: Implement the subscription logic internally here
        self.logger.debug("Subscribing to topic %s", topic)
        self.logger.warning("Not implemented")

    async def _publish_message(self, topic: Topic, message: BaseMessage) -> None:
        """Publish a message to a topic."""
        # TODO: implement the internal publish against the comms library
        self.logger.debug("Publishing message to topic %s: %s", topic, message)
        self.logger.warning("Not implemented")

    async def _send_heartbeat(self) -> None:
        """Send a heartbeat message to the system controller."""
        heartbeat_message = HeartbeatMessage(
            service_id=self.service_id,
            service_type=self.service_type,
        )
        self.logger.debug("Sending heartbeat message: %s", heartbeat_message)
        await self._publish_message(Topic.HEARTBEAT, heartbeat_message)

    async def _set_service_status(self, status: ServiceState) -> None:
        """Send a service state message to the system controller."""
        self.state = status
        status_message = StatusMessage(
            service_id=self.service_id,
            service_type=self.service_type,
            state=self.state,
        )
        await self._publish_message(Topic.STATUS, status_message)

    async def _start_heartbeat_task(self) -> None:
        """Start a background task to send heartbeats at regular intervals."""

        async def heartbeat_loop() -> None:
            while True:
                await self._send_heartbeat()
                await asyncio.sleep(self.heartbeat_interval)

        self.heartbeat_task = asyncio.create_task(heartbeat_loop())
        self.logger.debug(
            "Started heartbeat task with interval %ss", self.heartbeat_interval
        )

    async def _register(self) -> None:
        """Register the service with the system controller.

        This method should be called after the service has been initialized and is ready to
        start processing messages.
        """
        self.logger.debug("Registering service with system controller")
        await self._publish_message(
            Topic.REGISTRATION,
            RegistrationMessage(
                service_id=self.service_id,
                service_type=self.service_type,
            ),
        )

    async def run(self) -> None:
        """Start the service and initialize its components."""
        try:
            # Initialize the service
            self.state = ServiceState.INITIALIZING
            await self._initialize()
            await self._register()
            # Start heartbeat task
            await self._start_heartbeat_task()
            await self._set_service_status(ServiceState.READY)

            # Start the service if it is set to auto start.
            # Otherwise, wait for the System Controller to start it
            if self.autostart:
                await self._start()

            # Wait forever for the stop event to be set
            await self.stop_event.wait()
        except asyncio.exceptions.CancelledError:
            self.logger.debug("Service execution cancelled")
        except BaseException:
            self.logger.exception("Service execution failed:")
            await self._set_service_status(ServiceState.ERROR)
        finally:
            # Make sure to clean up properly even if there was an error
            await self.stop()

    async def _start(self) -> None:
        """Start the service and its components.

        This method should be called to start the service after it has been initialized.
        """
        self.logger.debug("Starting service %s", self.service_id)
        await self._set_service_status(ServiceState.STARTING)
        try:
            await self._on_start()
            await self._set_service_status(ServiceState.RUNNING)
        except BaseException as e:
            self.logger.exception("Failed to start service %s: %s", self.service_id, e)
            await self._set_service_status(ServiceState.ERROR)
            raise

    async def stop(self) -> None:
        """Stop the service and clean up its components."""
        await self._set_service_status(ServiceState.STOPPING)
        # Signal the run method to exit if it hasn't already
        self.stop_event.set()

        # Cancel heartbeat task if running
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.heartbeat_task

        await self._on_stop()
        await self._cleanup()
        self.state = ServiceState.STOPPED

    ################################################################################
    ## Abstract methods to be implemented by derived classes
    ################################################################################

    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize service-specific components.

        This method should be implemented by derived classes to set up any resources
        specific to that service.
        """

    @abstractmethod
    async def _on_start(self) -> None:
        """Start the service.

        This method should be implemented by derived classes to run any processes
        or components specific to that service.
        """

    @abstractmethod
    async def _on_stop(self) -> None:
        """Stop the service.

        This method should be implemented by derived classes to stop any processes
        or components specific to that service.
        """

    @abstractmethod
    async def _cleanup(self) -> None:
        """Clean up service-specific components.

        This method should be implemented by derived classes to clean up any resources
        specific to that service.
        """

    @abstractmethod
    async def _process_message(self, topic: Topic, message: BaseMessage) -> None:
        """Process a message from another service.

        This method should be implemented by derived classes to handle messages
        received from other services in the system.
        """
