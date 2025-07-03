# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from abc import ABC, abstractmethod
from contextlib import suppress

import zmq
import zmq.asyncio
from zmq import SocketType

from aiperf.common.comms.zmq.zmq_base_client import BaseZMQClient
from aiperf.common.config.zmq_config import BaseZMQProxyConfig
from aiperf.common.constants import TASK_CANCEL_TIMEOUT_SHORT
from aiperf.common.enums import ZMQProxyType
from aiperf.common.exceptions import ProxyError
from aiperf.common.factories import FactoryMixin


class BaseZMQProxy(ABC):
    """
    A Base ZMQ Proxy class.

    - Frontend and backend sockets forward messages bidirectionally
        - Frontend and Backend sockets both BIND
    - Multiple clients CONNECT to `frontend_address`
    - Multiple services CONNECT to `backend_address`
    - Control: Optional REP socket for proxy commands (start/stop/pause) - not implemented yet
    - Monitoring: Optional PUB socket that broadcasts copies of all forwarded messages - not implemented yet
    - Proxy runs in separate thread to avoid blocking main event loop
    """

    def __init__(
        self,
        frontend_socket_class: type[BaseZMQClient],
        backend_socket_class: type[BaseZMQClient],
        context: zmq.asyncio.Context,
        zmq_proxy_config: BaseZMQProxyConfig,
        socket_ops: dict | None = None,
    ) -> None:
        """Initialize the ZMQ Proxy. This is a base class for all ZMQ Proxies.

        Args:
            frontend_socket_class (type[BaseZMQClient]): The frontend socket class.
            backend_socket_class (type[BaseZMQClient]): The backend socket class.
            context (zmq.asyncio.Context): The ZMQ context.
            zmq_proxy_config (BaseZMQProxyConfig): The ZMQ proxy configuration.
            socket_ops (dict, optional): Additional socket options to set.
        """

        self.logger = logging.getLogger(self.__class__.__name__)
        self.context = context
        self.socket_ops = socket_ops

        self.monitor_task: asyncio.Task | None = None
        self.control_client: BaseZMQClient | None = None
        self.capture_client: BaseZMQClient | None = None
        self.proxy_task: asyncio.Task | None = None
        self.proxy: zmq.asyncio.Socket | None = None

        self.frontend_address = zmq_proxy_config.frontend_address
        self.backend_address = zmq_proxy_config.backend_address
        self.control_address = zmq_proxy_config.control_address
        self.capture_address = zmq_proxy_config.capture_address

        self.logger.debug(
            "Proxy Initializing - Frontend: %s, Backend: %s",
            self.frontend_address,
            self.backend_address,
        )

        self.backend_socket = backend_socket_class(
            context=self.context,
            address=self.backend_address,
            socket_ops=self.socket_ops,
        )

        self.frontend_socket = frontend_socket_class(
            context=self.context,
            address=self.frontend_address,
            socket_ops=self.socket_ops,
        )

        if self.control_address:
            self.logger.debug("Proxy Control - Address: %s", self.control_address)
            self.control_client = BaseZMQClient(
                context=self.context,
                socket_type=SocketType.REP,
                address=self.control_address,
                bind=True,
                socket_ops=self.socket_ops,
            )

        if self.capture_address:
            self.logger.debug("Proxy Capture - Address: %s", self.capture_address)
            self.capture_client = BaseZMQClient(
                context=self.context,
                socket_type=SocketType.PUB,
                address=self.capture_address,
                bind=True,
                socket_ops=self.socket_ops,
            )

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        config: BaseZMQProxyConfig | None,
        socket_ops: dict | None = None,
    ) -> "BaseZMQProxy | None":
        """Create a BaseZMQProxy from a BaseZMQProxyConfig, or None if not provided."""
        ...

    async def _initialize(self) -> None:
        """Initialize and start the BaseZMQProxy."""
        self.logger.debug("Proxy Initializing Sockets...")
        self.logger.debug(
            "Frontend %s socket binding to: %s (for %s clients)",
            self.frontend_socket.socket_type.name,
            self.frontend_address,
            self.backend_socket.socket_type.name,
        )
        self.logger.debug(
            "Backend %s socket binding to: %s (for %s services)",
            self.backend_socket.socket_type.name,
            self.backend_address,
            self.frontend_socket.socket_type.name,
        )
        if hasattr(self.backend_socket, "proxy_id"):
            self.logger.debug(
                "Backend socket identity: %s",
                self.backend_socket.proxy_id,
            )

        try:
            await asyncio.gather(
                self.backend_socket.initialize(),
                self.frontend_socket.initialize(),
                *[
                    client.initialize()
                    for client in [self.control_client, self.capture_client]
                    if client
                ],
            )

            self.logger.debug("Proxy Sockets Initialized Successfully")

            if self.control_client:
                self.logger.debug("Control socket bound to: %s", self.control_address)
            if self.capture_client:
                self.logger.debug("Capture socket bound to: %s", self.capture_address)

        except Exception as e:
            self.logger.error(f"Proxy Socket Initialization Failed {e}")
            raise

    async def stop(self) -> None:
        """Shutdown the BaseZMQProxy."""
        self.logger.debug("Proxy Stopping...")

        try:
            if self.monitor_task is not None:
                self.monitor_task.cancel()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self.monitor_task, timeout=TASK_CANCEL_TIMEOUT_SHORT
                    )

            if self.proxy_task is not None:
                self.proxy_task.cancel()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self.proxy_task, timeout=TASK_CANCEL_TIMEOUT_SHORT
                    )

            await asyncio.wait_for(
                asyncio.gather(
                    self.backend_socket.shutdown(),
                    self.frontend_socket.shutdown(),
                    *[
                        client.shutdown()
                        for client in [self.control_client, self.capture_client]
                        if client
                    ],
                ),
                timeout=TASK_CANCEL_TIMEOUT_SHORT,
            )

        except Exception as e:
            self.logger.error("Proxy Stop Error: %s", e)

    async def run(self) -> None:
        """Start the Base ZMQ Proxy.

        This method starts the proxy and waits for it to complete asynchronously.
        The proxy forwards messages between the frontend and backend sockets.

        Raises:
            ProxyError: If the proxy produces an error.
        """
        try:
            await self._initialize()

            self.logger.debug("Proxy Starting...")

            if self.capture_client:
                self.monitor_task = asyncio.create_task(self._monitor_messages())
                self.logger.debug("Proxy Message Monitoring Started")

            self.proxy_task = asyncio.create_task(
                asyncio.to_thread(
                    zmq.proxy_steerable,
                    self.frontend_socket.socket,
                    self.backend_socket.socket,
                    capture=self.capture_client.socket if self.capture_client else None,
                    control=self.control_client.socket if self.control_client else None,
                )
            )

        except zmq.ContextTerminated:
            self.logger.debug("Proxy Terminated by Context")
            raise

        except Exception as e:
            self.logger.error("Proxy Error: %s", e)
            raise ProxyError(f"Proxy failed: {e}") from e

    async def _monitor_messages(self) -> None:
        """Monitor messages flowing through the proxy via the capture socket."""
        if not self.capture_client or not self.capture_address:
            raise ProxyError("Proxy Monitor Not Enabled")

        self.logger.debug(
            "Proxy Monitor Starting - Capture Address: %s",
            self.capture_address,
        )

        capture_socket = self.context.socket(SocketType.SUB)
        capture_socket.connect(self.capture_address)
        capture_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages

        try:
            while True:
                message = capture_socket.recv()
                self.logger.debug("Proxy Monitor Received Message: %s", message)
        except Exception as e:
            self.logger.error("Proxy Monitor Error - %s", e)
            raise
        finally:
            capture_socket.close()


class ZMQProxyFactory(FactoryMixin[ZMQProxyType, BaseZMQProxy]):
    """A factory for creating ZMQ proxies. see :class:`FactoryMixin` for more details."""
