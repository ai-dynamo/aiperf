# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import logging
import uuid

import zmq.asyncio

from aiperf.common.exceptions import (
    AIPerfError,
    CommunicationError,
    CommunicationErrorReason,
)
from aiperf.common.hooks import AIPerfHook, AIPerfTaskMixin, supports_hooks


@supports_hooks(
    AIPerfHook.ON_INIT,
    AIPerfHook.ON_STOP,
    AIPerfHook.ON_CLEANUP,
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
        socket_type: zmq.SocketType,
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
        self.socket_type: zmq.SocketType = socket_type
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
            CommunicationError: If the client is not initialized
        """
        if not self._socket:
            raise CommunicationError(
                CommunicationErrorReason.NOT_INITIALIZED_ERROR,
                "Communication channels are not initialized",
            )
        return self._socket

    def _ensure_initialized(self) -> None:
        """Ensure the communication channels are initialized and not shutdown.

        Raises:
            CommunicationError: If the communication channels are not initialized
                or shutdown
        """
        if not self.is_initialized:
            raise CommunicationError(
                CommunicationErrorReason.NOT_INITIALIZED_ERROR,
                "Communication channels are not initialized",
            )
        if self.is_shutdown:
            raise asyncio.CancelledError()

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
                self._socket.bind(self.address)
            else:
                self.logger.debug(
                    "ZMQ %s socket initialized and connected to %s (%s)",
                    self.socket_type_name,
                    self.address,
                    self.client_id,
                )
                self._socket.connect(self.address)

            # TODO: Make these easier to configure by an end user

            # Use reasonable timeouts
            self._socket.setsockopt(zmq.RCVTIMEO, 30000)  # 30 seconds
            self._socket.setsockopt(zmq.SNDTIMEO, 30000)  # 30 seconds

            # Add performance-oriented socket options
            self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 10)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)
            self._socket.setsockopt(zmq.IMMEDIATE, 1)  # Don't queue messages
            self._socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close

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

        except AIPerfError:
            raise  # re-raise it up the stack
        except Exception as e:
            raise CommunicationError(
                CommunicationErrorReason.INITIALIZATION_ERROR,
                f"Failed to initialize ZMQ socket: {e}",
            ) from e

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

        # Cancel all registered tasks
        for task in self.registered_tasks.values():
            task.cancel()

        try:
            if self._socket:
                self._socket.close()
                self.logger.debug(
                    "ZMQ %s socket closed (%s)", self.socket_type_name, self.client_id
                )

        except Exception as e:
            self.logger.error(
                "Exception shutting down ZMQ socket: %s (%s)", e, self.client_id
            )
            raise CommunicationError(
                CommunicationErrorReason.SHUTDOWN_ERROR,
                f"Failed to shutdown ZMQ socket: {e}",
            ) from e

        finally:
            self._socket = None

            try:
                await self.run_hooks(AIPerfHook.ON_STOP)
                await self.run_hooks(AIPerfHook.ON_CLEANUP)

            except asyncio.CancelledError:
                return

            except AIPerfError:
                raise  # re-raise it up the stack

            except Exception as e:
                self.logger.error(
                    "Exception cleaning up ZMQ socket: %s (%s)", e, self.client_id
                )
                raise CommunicationError(
                    CommunicationErrorReason.CLEANUP_ERROR,
                    f"Failed to cleanup ZMQ socket: {e}",
                ) from e

            # Wait for all tasks to complete
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.gather(*self.registered_tasks.values())

            self.registered_tasks.clear()
