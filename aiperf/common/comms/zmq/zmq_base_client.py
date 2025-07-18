# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
import logging
import uuid

import zmq.asyncio

from aiperf.common.comms.zmq.zmq_defaults import ZMQSocketDefaults
from aiperf.common.constants import TASK_CANCEL_TIMEOUT_SHORT
from aiperf.common.exceptions import (
    AIPerfError,
    CommunicationError,
    InitializationError,
)
from aiperf.common.hooks import (
    AIPerfHook,
    AIPerfTaskHook,
    supports_hooks,
)
from aiperf.common.mixins import AIPerfTaskMixin

################################################################################
# Base ZMQ Client Class
################################################################################


@supports_hooks(
    AIPerfHook.ON_INIT,
    AIPerfHook.ON_STOP,
    AIPerfHook.ON_CLEANUP,
    AIPerfTaskHook.AIPERF_TASK,
)
class BaseZMQClient(AIPerfTaskMixin):
    """Base class for all ZMQ clients. It can be used as-is to create a new ZMQ client,
    or it can be subclassed to create specific ZMQ client functionality.

    It inherits from the :class:`AIPerfTaskMixin`, allowing derived
    classes to implement specific hooks.
    """

    def __init__(
        self,
        context: zmq.asyncio.Context,
        socket_type: zmq.SocketType,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        client_id: str | None = None,
    ) -> None:
        """
        Initialize the ZMQ Base class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to BIND or CONNECT the socket.
            socket_type (SocketType): The type of ZMQ socket (eg. PUB, SUB, ROUTER, DEALER, etc.).
            socket_ops (dict, optional): Additional socket options to set.
        """
        self.stop_event: asyncio.Event = asyncio.Event()
        self.initialized_event: asyncio.Event = asyncio.Event()
        self.context: zmq.asyncio.Context = context
        self.address: str = address
        self.bind: bool = bind
        self.socket_type: zmq.SocketType = socket_type
        self._socket: zmq.asyncio.Socket | None = None
        self.socket_ops: dict = socket_ops or {}
        self.client_id: str = (
            client_id
            or f"{self.socket_type.name.lower()}_client_{uuid.uuid4().hex[:8]}"
        )
        super().__init__()
        # Set the logger after the super init to override the name
        self.logger = logging.getLogger(self.client_id)

    @property
    def is_initialized(self) -> bool:
        """Check if the client is initialized."""
        return self.initialized_event.is_set()

    @property
    def stop_requested(self) -> bool:
        """Check if the client has been requested to stop."""
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
                "Communication channels are not initialized",
            )
        return self._socket

    async def _ensure_initialized(self) -> None:
        """Ensure the communication channels are initialized and not shutdown.

        Raises:
            CommunicationError: If the communication channels are not initialized
                or shutdown
        """
        if not self.is_initialized:
            await self.initialize()
        if self.stop_requested:
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
                    "ZMQ %s socket initialized, try BIND to %s (%s)",
                    self.socket_type_name,
                    self.address,
                    self.client_id,
                )
                self._socket.bind(self.address)
            else:
                self.logger.debug(
                    "ZMQ %s socket initialized, try CONNECT to %s (%s)",
                    self.socket_type_name,
                    self.address,
                    self.client_id,
                )
                self._socket.connect(self.address)

            # Set default timeouts
            self._socket.setsockopt(zmq.RCVTIMEO, ZMQSocketDefaults.RCVTIMEO)
            self._socket.setsockopt(zmq.SNDTIMEO, ZMQSocketDefaults.SNDTIMEO)

            # Set performance-oriented socket options
            self._socket.setsockopt(zmq.TCP_KEEPALIVE, ZMQSocketDefaults.TCP_KEEPALIVE)
            self._socket.setsockopt(
                zmq.TCP_KEEPALIVE_IDLE, ZMQSocketDefaults.TCP_KEEPALIVE_IDLE
            )
            self._socket.setsockopt(
                zmq.TCP_KEEPALIVE_INTVL, ZMQSocketDefaults.TCP_KEEPALIVE_INTVL
            )
            self._socket.setsockopt(
                zmq.TCP_KEEPALIVE_CNT, ZMQSocketDefaults.TCP_KEEPALIVE_CNT
            )
            self._socket.setsockopt(zmq.IMMEDIATE, ZMQSocketDefaults.IMMEDIATE)
            self._socket.setsockopt(zmq.LINGER, ZMQSocketDefaults.LINGER)

            # Set additional socket options requested by the caller
            for key, val in self.socket_ops.items():
                self._socket.setsockopt(key, val)

            await self.run_hooks(AIPerfHook.ON_INIT)

            self.initialized_event.set()
            self.logger.debug(
                "ZMQ %s socket %s to %s (%s)",
                self.socket_type_name,
                "BOUND" if self.bind else "CONNECTED",
                self.address,
                self.client_id,
            )

        except AIPerfError:
            raise  # re-raise it up the stack
        except Exception as e:
            raise InitializationError(
                f"Failed to initialize ZMQ socket: {e}",
            ) from e

    async def shutdown(self) -> None:
        """Shutdown the communication.

        This method will:
        - Close the zmq socket
        - Run the AIPerfHook.ON_CLEANUP hooks
        """
        if self.stop_requested:
            return

        self.stop_event.set()

        try:
            await self.run_hooks(AIPerfHook.ON_STOP)
        except Exception as e:
            self.logger.error(
                "Exception running ON_STOP hooks: %s (%s)", e, self.client_id
            )

        # Cancel all registered tasks
        for task in self.registered_tasks:
            task.cancel()

        # Wait for all tasks to complete
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*self.registered_tasks),
                timeout=TASK_CANCEL_TIMEOUT_SHORT,
            )
        self.registered_tasks.clear()

        # Run the ON_STOP and ON_CLEANUP hooks
        try:
            await self.run_hooks(AIPerfHook.ON_STOP)
            await self.run_hooks(AIPerfHook.ON_CLEANUP)

        except AIPerfError:
            raise  # re-raise it up the stack

        except Exception as e:
            self.logger.error(
                "Exception cleaning up ZMQ socket: %s (%s)", e, self.client_id
            )

        finally:
            try:
                if self._socket:
                    self._socket.close()

            except Exception as e:
                self.logger.error(
                    "Exception shutting down ZMQ socket: %s (%s)", e, self.client_id
                )
            finally:
                self._socket = None
