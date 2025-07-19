# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid
from abc import ABC, abstractmethod
from contextlib import suppress

import zmq
import zmq.asyncio
from zmq import SocketType

from aiperf.common.comms.zmq.zmq_base_client import BaseZMQClient
from aiperf.common.config.zmq_config import BaseZMQProxyConfig
from aiperf.common.constants import TASK_CANCEL_TIMEOUT_SHORT
from aiperf.common.enums import CaseInsensitiveStrEnum, ZMQProxyType
from aiperf.common.exceptions import ProxyError
from aiperf.common.factories import FactoryMixin
from aiperf.common.mixins import AIPerfLoggerMixin


class ProxyEndType(CaseInsensitiveStrEnum):
    Frontend = "frontend"
    Backend = "backend"
    Capture = "capture"
    Control = "control"


class ProxySocketClient(BaseZMQClient):
    """A ZMQ Proxy socket client class that extends BaseZMQClient.

    This class is used to create proxy sockets for the frontend, backend, capture, and control
    endpoint types of a ZMQ Proxy.
    """

    def __init__(
        self,
        context: zmq.asyncio.Context,
        socket_type: SocketType,
        address: str,
        end_type: ProxyEndType,
        socket_ops: dict | None = None,
        proxy_uuid: str | None = None,
    ) -> None:
        self.client_id = f"proxy_{end_type}_{socket_type.name.lower()}_{proxy_uuid or uuid.uuid4().hex[:8]}"
        super().__init__(
            context,
            socket_type,
            address,
            bind=True,
            socket_ops=socket_ops,
            client_id=self.client_id,
        )
        self.debug(
            lambda: f"ZMQ Proxy {end_type.name} {socket_type.name} - Address: {address}"
        )


class BaseZMQProxy(AIPerfLoggerMixin, ABC):
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
        proxy_uuid: str | None = None,
    ) -> None:
        """Initialize the ZMQ Proxy. This is a base class for all ZMQ Proxies.

        Args:
            frontend_socket_class (type[BaseZMQClient]): The frontend socket class.
            backend_socket_class (type[BaseZMQClient]): The backend socket class.
            context (zmq.asyncio.Context): The ZMQ context.
            zmq_proxy_config (BaseZMQProxyConfig): The ZMQ proxy configuration.
            socket_ops (dict, optional): Additional socket options to set.
            proxy_uuid (str, optional): An optional UUID for the proxy instance. If not provided,
                a new UUID will be generated. This is useful for tracing and debugging purposes.
        """

        self.proxy_uuid = proxy_uuid or uuid.uuid4().hex[:8]
        self.proxy_id = f"{self.__class__.__name__.lower()}_{self.proxy_uuid}"
        super().__init__()
        self.context = context
        self.socket_ops = socket_ops

        self.monitor_task: asyncio.Task | None = None
        self.proxy_task: asyncio.Task | None = None
        self.control_client: ProxySocketClient | None = None
        self.capture_client: ProxySocketClient | None = None

        self.frontend_address = zmq_proxy_config.frontend_address
        self.backend_address = zmq_proxy_config.backend_address
        self.control_address = zmq_proxy_config.control_address
        self.capture_address = zmq_proxy_config.capture_address

        self.debug(
            lambda: f"Proxy Initializing - Frontend: {self.frontend_address}, Backend: {self.backend_address}"
        )

        self.backend_socket = backend_socket_class(
            context=self.context,
            address=self.backend_address,
            socket_ops=self.socket_ops,
            proxy_uuid=self.proxy_uuid,  # Pass the proxy UUID for tracing
        )

        self.frontend_socket = frontend_socket_class(
            context=self.context,
            address=self.frontend_address,
            socket_ops=self.socket_ops,
            proxy_uuid=self.proxy_uuid,  # Pass the proxy UUID for tracing
        )

        if self.control_address:
            self.debug(lambda: f"Proxy Control - Address: {self.control_address}")
            self.control_client = ProxySocketClient(
                context=self.context,
                socket_type=SocketType.REP,
                address=self.control_address,
                socket_ops=self.socket_ops,
                end_type=ProxyEndType.Control,
                proxy_uuid=self.proxy_uuid,
            )

        if self.capture_address:
            self.debug(lambda: f"Proxy Capture - Address: {self.capture_address}")
            self.capture_client = ProxySocketClient(
                context=self.context,
                socket_type=SocketType.PUB,
                address=self.capture_address,
                socket_ops=self.socket_ops,
                end_type=ProxyEndType.Capture,
                proxy_uuid=self.proxy_uuid,
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
        self.debug("Proxy Initializing Sockets...")
        self.debug(
            lambda: f"Frontend {self.frontend_socket.socket_type.name} socket binding to: {self.frontend_address} (for {self.backend_socket.socket_type.name} clients)"
        )
        self.debug(
            lambda: f"Backend {self.backend_socket.socket_type.name} socket binding to: {self.backend_address} (for {self.frontend_socket.socket_type.name} services)"
        )
        if hasattr(self.backend_socket, "proxy_id"):
            self.debug(
                lambda: f"Backend socket identity: {self.backend_socket.proxy_id}"
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

            self.debug("Proxy Sockets Initialized Successfully")

            if self.control_client:
                self.debug(lambda: f"Control socket bound to: {self.control_address}")
            if self.capture_client:
                self.debug(lambda: f"Capture socket bound to: {self.capture_address}")

        except Exception as e:
            self.exception(f"Proxy Socket Initialization Failed: {e}")
            raise

    async def stop(self) -> None:
        """Shutdown the BaseZMQProxy."""
        self.debug("Proxy Stopping...")

        try:
            if self.monitor_task is not None:
                self.debug("Cancelling Monitor Task")
                self.monitor_task.cancel()
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self.monitor_task, timeout=TASK_CANCEL_TIMEOUT_SHORT
                    )

        except Exception as e:
            self.exception(f"Proxy Stop Error: {e}")

    async def run(self) -> None:
        """Start the Base ZMQ Proxy.

        This method starts the proxy and waits for it to complete asynchronously.
        The proxy forwards messages between the frontend and backend sockets.

        Raises:
            ProxyError: If the proxy produces an error.
        """
        try:
            await self._initialize()

            self.debug("Starting Proxy...")

            if self.capture_client:
                self.monitor_task = asyncio.create_task(self._monitor_messages())
                self.debug("Proxy Message Monitoring Started")

            await asyncio.to_thread(
                zmq.proxy_steerable,
                self.frontend_socket.socket,
                self.backend_socket.socket,
                capture=self.capture_client.socket if self.capture_client else None,
                control=self.control_client.socket if self.control_client else None,
            )

        except zmq.ContextTerminated:
            self.debug("Proxy Terminated by Context")
            return

        except Exception as e:
            self.exception(f"Proxy Error: {e}")
            raise ProxyError(f"Proxy failed: {e}") from e

    async def _monitor_messages(self) -> None:
        """Monitor messages flowing through the proxy via the capture socket."""
        if not self.capture_client or not self.capture_address:
            raise ProxyError("Proxy Monitor Not Enabled")

        self.debug(
            lambda: f"Proxy Monitor Starting - Capture Address: {self.capture_address}"
        )

        capture_socket = self.context.socket(SocketType.SUB)
        capture_socket.connect(self.capture_address)
        self.debug(
            lambda: f"Proxy Monitor Connected to Capture Address: {self.capture_address}"
        )
        capture_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.debug("Proxy Monitor Subscribed to all messages")

        try:
            while True:
                recv_msg = await capture_socket.recv_multipart()
                self.trace(lambda msg=recv_msg: f"Proxy Monitor Received: {msg}")
        except Exception as e:
            self.exception(f"Proxy Monitor Error - {e}")
            raise
        finally:
            capture_socket.close()


class ZMQProxyFactory(FactoryMixin[ZMQProxyType, BaseZMQProxy]):
    """A factory for creating ZMQ proxies. see :class:`FactoryMixin` for more details."""
