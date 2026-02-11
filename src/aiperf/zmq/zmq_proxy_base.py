# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid
from abc import ABC, abstractmethod

import zmq
import zmq.asyncio
from zmq import SocketType

from aiperf.common.config import BaseZMQProxyConfig
from aiperf.common.enums import CaseInsensitiveStrEnum
from aiperf.common.hooks import background_task, on_init, on_start, on_stop
from aiperf.common.mixins import AIPerfLifecycleMixin
from aiperf.zmq.zmq_base_client import BaseZMQClient


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
        socket_type: SocketType,
        address: str,
        end_type: ProxyEndType,
        socket_ops: dict | None = None,
        proxy_uuid: str | None = None,
        **kwargs,
    ) -> None:
        self.client_id = f"proxy_{end_type}_{socket_type.name.lower()}_{proxy_uuid or uuid.uuid4().hex[:8]}"
        super().__init__(
            socket_type,
            address,
            bind=True,
            socket_ops=socket_ops,
            client_id=self.client_id,
            **kwargs,
        )
        self.debug(
            lambda: f"ZMQ Proxy {end_type.name} {socket_type.name} - Address: {address}"
        )


class BaseZMQProxy(AIPerfLifecycleMixin, ABC):
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
        zmq_proxy_config: BaseZMQProxyConfig,
        socket_ops: dict | None = None,
        proxy_uuid: str | None = None,
    ) -> None:
        """Initialize the ZMQ Proxy. This is a base class for all ZMQ Proxies.

        Args:
            frontend_socket_class (type[BaseZMQClient]): The frontend socket class.
            backend_socket_class (type[BaseZMQClient]): The backend socket class.
            zmq_proxy_config (BaseZMQProxyConfig): The ZMQ proxy configuration.
            socket_ops (dict, optional): Additional socket options to set.
            proxy_uuid (str, optional): An optional UUID for the proxy instance. If not provided,
                a new UUID will be generated. This is useful for tracing and debugging purposes.
        """

        self.proxy_uuid = proxy_uuid or uuid.uuid4().hex[:8]
        self.proxy_id = f"{self.__class__.__name__.lower()}_{self.proxy_uuid}"
        super().__init__()
        self.context = zmq.asyncio.Context.instance()
        self.socket_ops = socket_ops

        self.monitor_task: asyncio.Task | None = None
        self.proxy_task: asyncio.Task | None = None
        self.control_client: ProxySocketClient | None = None
        self.capture_client: ProxySocketClient | None = None

        self.config = zmq_proxy_config

        self.debug(
            lambda: f"Proxy Initializing - Frontend: {self.config.frontend_address}, Backend: {self.config.backend_address}"
        )

        self.backend_socket = backend_socket_class(
            address=self.config.backend_address,
            socket_ops=self.socket_ops,
            proxy_uuid=self.proxy_uuid,
            additional_bind_address=self.config.additional_backend_bind_address,
        )  # type: ignore

        self.frontend_socket = frontend_socket_class(
            address=self.config.frontend_address,
            socket_ops=self.socket_ops,
            proxy_uuid=self.proxy_uuid,
            additional_bind_address=self.config.additional_frontend_bind_address,
        )  # type: ignore

        if self.config.control_address:
            self.debug(
                lambda: f"Proxy Control - Address: {self.config.control_address}"
            )
            self.control_client = ProxySocketClient(
                socket_type=SocketType.REP,
                address=self.config.control_address,
                socket_ops=self.socket_ops,
                end_type=ProxyEndType.Control,
                proxy_uuid=self.proxy_uuid,
            )

        if self.config.capture_address:
            self.debug(
                lambda: f"Proxy Capture - Address: {self.config.capture_address}"
            )
            self.capture_client = ProxySocketClient(
                socket_type=SocketType.PUB,
                address=self.config.capture_address,
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

    @on_init
    async def _initialize(self) -> None:
        """Initialize and start the BaseZMQProxy."""
        self.debug("Proxy Initializing Sockets...")
        self.debug(
            lambda: f"Frontend {self.frontend_socket.socket_type.name} socket binding to: {self.config.frontend_address} (for {self.backend_socket.socket_type.name} clients)"
        )
        self.debug(
            lambda: f"Backend {self.backend_socket.socket_type.name} socket binding to: {self.config.backend_address} (for {self.frontend_socket.socket_type.name} services)"
        )

        try:
            exceptions = await asyncio.gather(
                self.backend_socket.initialize(),
                self.frontend_socket.initialize(),
                *[
                    client.initialize()
                    for client in [self.control_client, self.capture_client]
                    if client
                ],
                return_exceptions=True,
            )
            if any(exceptions):
                self.exception(f"Proxy Socket Initialization Failed: {exceptions}")
                # Find and raise the first actual exception, wrapped in LifecycleOperationError
                from aiperf.common.exceptions import LifecycleOperationError

                for exc in exceptions:
                    if exc is not None:
                        raise LifecycleOperationError(
                            operation="initialize",
                            original_exception=exc,
                            lifecycle_id=self.proxy_id,
                        ) from exc

            self.debug("Proxy Sockets Initialized Successfully")

            if self.control_client:
                self.debug(
                    lambda: f"Control socket bound to: {self.config.control_address}"
                )
            if self.capture_client:
                self.debug(
                    lambda: f"Capture socket bound to: {self.config.capture_address}"
                )

        except Exception as e:
            self.exception(f"Proxy Socket Initialization Failed: {e}")
            raise

    @on_start
    async def _start_proxy(self) -> None:
        """Start the Base ZMQ Proxy.

        This method starts the proxy and waits for it to complete asynchronously.
        The proxy forwards messages between the frontend and backend sockets.

        Raises:
            ProxyError: If the proxy produces an error.
        """
        self.debug("Starting Proxy...")

        self.proxy_task = asyncio.create_task(
            asyncio.to_thread(
                zmq.proxy_steerable,
                self.frontend_socket.socket,
                self.backend_socket.socket,
                capture=self.capture_client.socket if self.capture_client else None,
                control=self.control_client.socket if self.control_client else None,
            )
        )

    @background_task(immediate=True, interval=None)
    async def _monitor_messages(self) -> None:
        """Monitor messages flowing through the proxy via the capture socket."""
        if not self.capture_client or not self.config.capture_address:
            self.debug("Proxy Monitor Not Enabled")
            return

        self.debug(
            lambda: f"Proxy Monitor Starting - Capture Address: {self.config.capture_address}"
        )

        capture_socket = self.context.socket(SocketType.SUB)
        capture_socket.connect(self.config.capture_address)
        self.debug(
            lambda: f"Proxy Monitor Connected to Capture Address: {self.config.capture_address}"
        )
        capture_socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.debug("Proxy Monitor Subscribed to all messages")

        try:
            while not self.stop_requested:
                recv_msg = await capture_socket.recv_multipart()
                self.debug(lambda msg=recv_msg: f"Proxy Monitor Received: {msg}")
        except asyncio.CancelledError:
            self.debug("Proxy Monitor Task Cancelled")
            return
        except Exception as e:
            self.exception(f"Proxy Monitor Error - {e}")
            raise
        finally:
            capture_socket.close()

    @on_stop
    async def _stop_proxy(self) -> None:
        """Shutdown the BaseZMQProxy."""
        self.debug("Proxy Stopping...")
        if self.proxy_task:
            self.proxy_task.cancel()
            self.proxy_task = None
        await self.frontend_socket.stop()
        await self.backend_socket.stop()
        if self.control_client:
            await self.control_client.stop()
        if self.capture_client:
            await self.capture_client.stop()
