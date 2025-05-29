# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import uuid

import zmq.asyncio
from zmq import SocketType

from aiperf.common.exceptions import (
    CommunicationError,
    CommunicationInitializationError,
    CommunicationNotInitializedError,
    CommunicationShutdownError,
)
from aiperf.common.hooks import AIPerfHook, AIPerfTaskMixin, supports_hooks


@supports_hooks(
    AIPerfHook.ON_INIT,
    AIPerfHook.ON_CLEANUP,
    AIPerfHook.ON_STOP,
    AIPerfHook.AIPERF_TASK,
)
class BaseZMQClient(AIPerfTaskMixin):
    """Base class for all ZMQ clients.

    This class provides a common interface for all ZMQ clients in the AIPerf
    framework. It inherits from the :class:`AIPerfTaskMixin`, allowing derived
    classes to implement specific hooks.
    """

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
        super().__init__()

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
        """Initialize the communication.

        This method will:
        - Create the zmq socket
        - Bind or connect the socket to the address
        - Set the socket options
        - Run the AIPerfHook.ON_INIT hooks
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

            await self.run_hooks(AIPerfHook.ON_INIT)

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
        - Run the AIPerfHook.ON_CLEANUP hooks
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
            await self.run_hooks(AIPerfHook.ON_STOP)
            await self.run_hooks(AIPerfHook.ON_CLEANUP)

        except Exception as e:
            self.logger.error(
                "Exception cleaning up ZMQ socket: %s (%s)", e, self.client_id
            )
            raise CommunicationError("Failed to cleanup ZMQ socket") from e
