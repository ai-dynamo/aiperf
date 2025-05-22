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
from abc import ABC
from collections import defaultdict
from collections.abc import Callable

import zmq.asyncio
from zmq import SocketType

from aiperf.common.comms.zmq.clients.metaclass import (
    ZMQClientMetaclass,
)
from aiperf.common.decorators import AIPerfHooks
from aiperf.common.exceptions.comms import (
    CommunicationError,
    CommunicationInitializationError,
    CommunicationNotInitializedError,
    CommunicationShutdownError,
)
from aiperf.common.utils import call_all_functions_self


class BaseZMQClient(ABC, metaclass=ZMQClientMetaclass):
    """Base class for all ZMQ clients.

    This class provides a common interface for all ZMQ clients in the AIPerf
    framework. It inherits from the ZMQClientMetaclass, allowing derived
    classes to implement specific hooks.
    """

    _aiperf_hooks: dict[str, list[Callable]] = defaultdict(list)

    def __init__(
        self,
        context: zmq.asyncio.Context,
        socket_type: SocketType,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ Base class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_type (SocketType): The type of ZMQ socket (PUB or SUB).
            socket_ops (dict, optional): Additional socket options to set.
        """
        self.logger = logging.getLogger(__name__)
        self.stop_event: asyncio.Event = asyncio.Event()
        self.initialized_event: asyncio.Event = asyncio.Event()
        self.context: zmq.asyncio.Context = context
        self.address: str = address
        self.bind: bool = bind
        self.socket_type: SocketType = socket_type
        self._socket: zmq.asyncio.Socket | None = None
        self.socket_ops: dict = socket_ops or {}
        self.client_id: str = f"{self.socket_type.name}_client_{uuid.uuid4().hex[:8]}"
        self._task_registry: dict[str, asyncio.Task] = {}

    @property
    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self.initialized_event.is_set()

    @property
    def is_shutdown(self) -> bool:
        """Check if the client is shutdown."""
        return self.stop_event.is_set()

    @property
    def socket_type_name(self) -> str:
        """Get the name of the socket type."""
        return self.socket_type.name

    @property
    def socket(self) -> zmq.asyncio.Socket:
        """Get the zmq socket for the client.

        Raises:
            CommunicationNotInitializedError: If the client is not initialized
        """
        if not self._socket:
            raise CommunicationNotInitializedError()
        return self._socket

    def _get_hooks(self, hook_type: AIPerfHooks) -> list[Callable]:
        """Get the hooks for the given hook type."""
        self.logger.debug(
            f"Getting hooks for {self.client_id}.{hook_type}: {self._aiperf_hooks[hook_type]}"
        )
        return self._aiperf_hooks[hook_type]

    async def _run_hooks(self, hook_type: AIPerfHooks, *args, **kwargs) -> None:
        """Run the hooks for the given hook type.

        Args:
            hook_type: The type of hook to run
            *args: The arguments to pass to the hooks
            **kwargs: The keyword arguments to pass to the hooks

        Raises:
            AIPerfMultiError: If any of the hooks raise an exception
        """
        await call_all_functions_self(self, self._get_hooks(hook_type), *args, **kwargs)

    def _ensure_initialized(self) -> None:
        """Ensure the communication channels are initialized and not shutdown.

        Raises:
            CommunicationNotInitializedError: If the communication channels are not initialized.
            CommunicationShutdownError: If the communication channels are shutdown.
        """
        if not self.is_initialized:
            raise CommunicationNotInitializedError()
        if self.is_shutdown:
            raise CommunicationShutdownError()

    async def initialize(self) -> None:
        """Initialize the communication, and start the tasks.

        This method will:
        - Create the zmq socket
        - Bind or connect the socket to the address
        - Set the socket options
        - Run the AIPerfHooks.INIT hooks
        - Start the tasks registered with the AIPerfHooks.TASK hooks
        """
        try:
            self._socket = self.context.socket(self.socket_type)
            if self.bind:
                self.logger.debug(
                    "ZMQ %s socket initialized and bound to %s (%s)",
                    self.socket_type_name,
                    self.address,
                    self.client_id,
                )
                self.socket.bind(self.address)
            else:
                self.logger.debug(
                    "ZMQ %s socket initialized and connected to %s (%s)",
                    self.socket_type_name,
                    self.address,
                    self.client_id,
                )
                self.socket.connect(self.address)

            # Set safe timeouts for send and receive operations
            self._socket.setsockopt(zmq.RCVTIMEO, 30 * 1000)
            self._socket.setsockopt(zmq.SNDTIMEO, 30 * 1000)

            # Set additional socket options requested by the caller
            for key, val in self.socket_ops.items():
                self._socket.setsockopt(key, val)

            await self._run_hooks(AIPerfHooks.INIT)

            # Start all registered tasks
            for hook in self._get_hooks(AIPerfHooks.TASK):
                # TODO: support task intervals
                self._task_registry[hook.__name__] = asyncio.create_task(hook(self))

            self.initialized_event.set()
            self.logger.debug(
                "ZMQ %s socket initialized and connected to %s (%s)",
                self.socket_type_name,
                self.address,
                self.client_id,
            )

        except Exception as e:
            self.logger.error("Exception initializing ZMQ socket: %s", e)
            raise CommunicationInitializationError from e

    async def shutdown(self) -> None:
        """Shutdown the communication.

        This method will:
        - Close the zmq socket
        - Run the AIPerfHooks.CLEANUP hooks
        - Cancel all registered tasks
        """
        if self.is_shutdown:
            return

        if not self.stop_event.is_set():
            self.stop_event.set()

        try:
            if self._socket:
                self.socket.close()
                self.logger.debug(
                    "ZMQ %s socket closed (%s)", self.socket_type_name, self.client_id
                )

        except Exception as e:
            self.logger.error(
                "Exception shutting down ZMQ socket: %s (%s)", e, self.client_id
            )
            raise CommunicationShutdownError("Failed to shutdown ZMQ socket") from e

        finally:
            self._socket = None

        try:
            await self._run_hooks(AIPerfHooks.CLEANUP)

        except Exception as e:
            self.logger.error(
                "Exception cleaning up ZMQ socket: %s (%s)", e, self.client_id
            )
            raise CommunicationError("Failed to cleanup ZMQ socket") from e

        # Cancel all registered tasks
        for task in self._task_registry.values():
            task.cancel()

        # Wait for all tasks to complete
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*self._task_registry.values())

        self._task_registry.clear()
