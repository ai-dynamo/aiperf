# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import uuid
from pathlib import Path

import zmq.asyncio

from aiperf.common.exceptions import InitializationError, NotInitializedError
from aiperf.common.hooks import on_init, on_stop
from aiperf.common.loop_scheduler import LoopScheduler
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.zmq.zmq_defaults import ZMQSocketDefaults

################################################################################
# Base ZMQ Client Class
################################################################################


class BaseZMQClient(AIPerfLifecycleMixin):
    """Base class for all ZMQ clients. It can be used as-is to create a new ZMQ client,
    or it can be subclassed to create specific ZMQ client functionality.

    It inherits from the :class:`AIPerfLifecycleMixin`, allowing derived
    classes to implement specific hooks.
    """

    def __init__(
        self,
        socket_type: zmq.SocketType,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        client_id: str | None = None,
        additional_bind_address: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ZMQ Base class.

        Args:
            address (str): The address to bind or connect to.
            bind (bool): Whether to BIND or CONNECT the socket.
            socket_type (SocketType): The type of ZMQ socket (eg. PUB, SUB, ROUTER, DEALER, etc.).
            socket_ops (dict, optional): Additional socket options to set.
            additional_bind_address (str, optional): Optional second address to bind to for dual-bind
                mode (e.g., IPC + TCP in Kubernetes). Only used when bind=True.
        """
        self.context: zmq.asyncio.Context = zmq.asyncio.Context.instance()
        self.socket_type: zmq.SocketType = socket_type
        self.socket: zmq.asyncio.Socket = None
        self.address: str = address
        self.bind: bool = bind
        self.socket_ops: dict = socket_ops or {}
        self.client_id: str = (
            client_id
            or f"{self.socket_type.name.lower()}_client_{uuid.uuid4().hex[:8]}"
        )
        self.scheduler: LoopScheduler | None = None
        self.additional_bind_address: str | None = (
            additional_bind_address if bind else None
        )
        super().__init__(id=self.client_id, **kwargs)
        if not self.bind and additional_bind_address:
            self.warning(
                f"Additional bind address provided but bind is False, ignoring: {additional_bind_address} for client {self.client_id}"
            )
        self.trace(lambda: f"ZMQ client __init__: {self.client_id}")

    async def _check_initialized(self) -> None:
        """Raise an exception if the socket is not initialized or closed."""
        if self.stop_requested:
            raise asyncio.CancelledError("Socket was stopped")
        if not self.socket:
            raise NotInitializedError("Socket not initialized or closed")
        if not self.scheduler:
            raise NotInitializedError("Scheduler not initialized")

    @property
    def socket_type_name(self) -> str:
        """Get the name of the socket type."""
        return self.socket_type.name

    @on_init
    async def _initialize_socket(self) -> None:
        """Initialize the communication.

        This method will:
        - Create the zmq socket
        - Bind or connect the socket to the address
        - Set the socket options
        - Run the AIPerfHook.ON_INIT hooks
        """
        try:
            self.scheduler = LoopScheduler()
            self.socket = self.context.socket(self.socket_type)
            self.debug(
                lambda: f"ZMQ {self.socket_type_name} socket initialized, try {'BIND' if self.bind else 'CONNECT'} to {self.address} ({self.client_id})"
            )

            if zmq.IDENTITY in self.socket_ops:
                # IMPORTANT! Set IDENTITY socket option immediately after socket creation, BEFORE bind/connect
                # otherwise it will not be properly set when the socket is bound/connected
                self.socket.setsockopt(zmq.IDENTITY, self.socket_ops[zmq.IDENTITY])
                self.debug(
                    lambda: f"Set IDENTITY socket option: {self.socket_ops[zmq.IDENTITY]}"
                )
                del self.socket_ops[zmq.IDENTITY]

            if self.bind:
                self.socket.bind(self.address)
            else:
                self.socket.connect(self.address)

            # Set default timeouts
            self.socket.setsockopt(zmq.RCVTIMEO, ZMQSocketDefaults.RCVTIMEO)
            self.socket.setsockopt(zmq.SNDTIMEO, ZMQSocketDefaults.SNDTIMEO)

            # Set high water mark
            self.socket.setsockopt(zmq.SNDHWM, ZMQSocketDefaults.SNDHWM)
            self.socket.setsockopt(zmq.RCVHWM, ZMQSocketDefaults.RCVHWM)

            # Set performance-oriented socket options
            self.socket.setsockopt(zmq.TCP_KEEPALIVE, ZMQSocketDefaults.TCP_KEEPALIVE)
            self.socket.setsockopt(
                zmq.TCP_KEEPALIVE_IDLE, ZMQSocketDefaults.TCP_KEEPALIVE_IDLE
            )
            self.socket.setsockopt(
                zmq.TCP_KEEPALIVE_INTVL, ZMQSocketDefaults.TCP_KEEPALIVE_INTVL
            )
            self.socket.setsockopt(
                zmq.TCP_KEEPALIVE_CNT, ZMQSocketDefaults.TCP_KEEPALIVE_CNT
            )
            self.socket.setsockopt(zmq.IMMEDIATE, ZMQSocketDefaults.IMMEDIATE)
            self.socket.setsockopt(zmq.LINGER, ZMQSocketDefaults.LINGER)

            # Set additional socket options requested by the caller
            for key, val in self.socket_ops.items():
                self.socket.setsockopt(key, val)

            self.debug(
                lambda: f"ZMQ {self.socket_type_name} socket {'BOUND' if self.bind else 'CONNECTED'} to {self.address} ({self.client_id})"
            )

        except Exception as e:
            raise InitializationError(f"Failed to initialize ZMQ socket: {e}") from e

    @on_init
    async def _bind_additional_address(self) -> None:
        """Bind to additional address for dual-bind mode (e.g., IPC + TCP)."""
        if self.additional_bind_address and self.socket:
            self.socket.bind(self.additional_bind_address)
            self.debug(
                lambda: f"Dual-bind: also bound to {self.additional_bind_address}"
            )

    def _cleanup_ipc_file(self) -> None:
        """Remove the IPC socket file if this client bound to one."""
        if self.bind and self.address.startswith("ipc://"):
            Path(self.address.removeprefix("ipc://")).unlink(missing_ok=True)

    @on_stop
    async def _shutdown_socket(self) -> None:
        """Shutdown the socket."""
        # TODO: Should we await the cancellation of the tasks?
        if self.scheduler:
            self.scheduler.cancel_all()
        try:
            if self.socket:
                self.socket.close()
        except zmq.ContextTerminated:
            self.debug(
                lambda: f"ZMQ context already terminated, skipping socket close ({self.client_id})"
            )
            return
        except Exception as e:
            self.exception(
                f"Uncaught exception shutting down ZMQ socket: {e} ({self.client_id})"
            )
        finally:
            self._cleanup_ipc_file()
