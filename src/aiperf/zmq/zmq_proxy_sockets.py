# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import zmq
from zmq import SocketType

from aiperf.common.config import BaseZMQProxyConfig
from aiperf.common.hooks import on_init
from aiperf.plugin.enums import ZMQProxyType
from aiperf.zmq.zmq_base_client import BaseZMQClient
from aiperf.zmq.zmq_proxy_base import (
    BaseZMQProxy,
    ProxyEndType,
    ProxySocketClient,
)

################################################################################
# Proxy Sockets
################################################################################


def create_proxy_socket_class(
    socket_type: SocketType,
    end_type: ProxyEndType,
) -> type[BaseZMQClient]:
    """Create a proxy socket class using the specified socket type. This is used to
    reduce the boilerplate code required to create a ZMQ Proxy class.
    """

    class_name = f"ZMQProxy{end_type.name}Socket{socket_type.name}"

    class ProxySocket(ProxySocketClient):
        """A ZMQ Proxy socket class with a specific socket type."""

        def __init__(
            self,
            address: str,
            socket_ops: dict | None = None,
            proxy_uuid: str | None = None,
        ):
            """Initialize the ZMQ Proxy socket class."""

            super().__init__(
                socket_type,
                address,
                end_type=end_type,
                socket_ops=socket_ops,
                proxy_uuid=proxy_uuid,
            )

        @on_init
        async def _initialize_socket(self) -> None:
            """Initialize the socket with proper configuration for XPUB/XSUB proxy."""
            if self.socket_type == SocketType.XPUB:
                self.socket.setsockopt(zmq.XPUB_VERBOSE, 1)
                self.debug(
                    lambda: "XPUB socket configured with XPUB_VERBOSE=1 for subscription forwarding"
                )

    # Dynamically set the class name and qualname based on the socket and end type
    ProxySocket.__name__ = class_name
    ProxySocket.__qualname__ = class_name
    ProxySocket.__doc__ = f"A ZMQ Proxy {end_type.name} socket implementation."
    return ProxySocket


def define_proxy_class(
    proxy_type: ZMQProxyType,
    frontend_socket_class: type[BaseZMQClient],
    backend_socket_class: type[BaseZMQClient],
) -> type[BaseZMQProxy]:
    """This function reduces the boilerplate code required to create a ZMQ Proxy class.
    It will generate a ZMQ Proxy class with the specified socket types.

    Args:
        proxy_type: The type of proxy to generate.
        frontend_socket_class: The class of the frontend socket.
        backend_socket_class: The class of the backend socket.
    """

    class ZMQProxy(BaseZMQProxy):
        """
        A Generated ZMQ Proxy class.

        This class is responsible for creating the ZMQ proxy that forwards messages
        between frontend and backend sockets.
        """

        def __init__(
            self,
            zmq_proxy_config: BaseZMQProxyConfig,
            socket_ops: dict | None = None,
        ) -> None:
            super().__init__(
                frontend_socket_class=frontend_socket_class,
                backend_socket_class=backend_socket_class,
                zmq_proxy_config=zmq_proxy_config,
                socket_ops=socket_ops,
            )

        @classmethod
        def from_config(
            cls,
            config: BaseZMQProxyConfig | None,
            socket_ops: dict | None = None,
        ) -> "ZMQProxy | None":
            if config is None:
                return None
            return cls(
                zmq_proxy_config=config,
                socket_ops=socket_ops,
            )

    # Dynamically set the class name and qualname based on the proxy type
    ZMQProxy.__name__ = f"ZMQ_{proxy_type.name}_Proxy"
    ZMQProxy.__qualname__ = ZMQProxy.__name__
    ZMQProxy.__doc__ = f"A ZMQ Proxy for {proxy_type.name} communication."
    return ZMQProxy


################################################################################
# XPUB/XSUB Proxy
################################################################################

ZMQXPubXSubProxy = define_proxy_class(
    ZMQProxyType.XPUB_XSUB,
    create_proxy_socket_class(SocketType.XSUB, ProxyEndType.Frontend),
    create_proxy_socket_class(SocketType.XPUB, ProxyEndType.Backend),
)
"""
An XSUB socket for the proxy's frontend and an XPUB socket for the proxy's backend.

ASCII Diagram:
┌───────────┐    ┌─────────────────────────────────┐    ┌───────────┐
│    PUB    │───>│              PROXY              │───>│    SUB    │
│  Client 1 │    │ ┌──────────┐       ┌──────────┐ │    │ Service 1 │
└───────────┘    │ │   XSUB   │──────>│   XPUB   │ │    └───────────┘
┌───────────┐    │ │ Frontend │       │ Backend  │ │    ┌───────────┐
│    PUB    │───>│ └──────────┘       └──────────┘ │───>│    SUB    │
│  Client N │    └─────────────────────────────────┘    │ Service N │
└───────────┘                                           └───────────┘

The XSUB frontend socket receives messages from PUB clients and forwards them
through the proxy to XPUB services. The ZMQ proxy handles the message
routing automatically.

The XPUB backend socket forwards messages from the proxy to SUB services.
The ZMQ proxy handles the message routing automatically.
"""

################################################################################
# ROUTER/DEALER Proxy
################################################################################

ZMQDealerRouterProxy = define_proxy_class(
    ZMQProxyType.DEALER_ROUTER,
    create_proxy_socket_class(SocketType.ROUTER, ProxyEndType.Frontend),
    create_proxy_socket_class(SocketType.DEALER, ProxyEndType.Backend),
)
"""
A ROUTER socket for the proxy's frontend and a DEALER socket for the proxy's backend.

ASCII Diagram:
┌───────────┐     ┌──────────────────────────────────┐      ┌───────────┐
│  DEALER   │<───>│              PROXY               │<────>│  ROUTER   │
│  Client 1 │     │ ┌──────────┐        ┌──────────┐ │      │ Service 1 │
└───────────┘     │ │  ROUTER  │<─────> │  DEALER  │ │      └───────────┘
┌───────────┐     │ │ Frontend │        │ Backend  │ │      ┌───────────┐
│  DEALER   │<───>│ └──────────┘        └──────────┘ │<────>│  ROUTER   │
│  Client N │     └──────────────────────────────────┘      │ Service N │
└───────────┘                                               └───────────┘

The ROUTER frontend socket receives messages from DEALER clients and forwards them
through the proxy to ROUTER services. The ZMQ proxy handles the message
routing automatically.

The DEALER backend socket receives messages from ROUTER services and forwards them
through the proxy to DEALER clients. The ZMQ proxy handles the message
routing automatically.

CRITICAL: This socket must NOT have an identity when used in a proxy
configuration, as it needs to be transparent to preserve routing envelopes
for proper response forwarding back to original DEALER clients.
"""


################################################################################
# PUSH/PULL Proxy
################################################################################

ZMQPushPullProxy = define_proxy_class(
    ZMQProxyType.PUSH_PULL,
    create_proxy_socket_class(SocketType.PULL, ProxyEndType.Frontend),
    create_proxy_socket_class(SocketType.PUSH, ProxyEndType.Backend),
)
"""
A PULL socket for the proxy's frontend and a PUSH socket for the proxy's backend.

ASCII Diagram:
┌───────────┐      ┌─────────────────────────────────┐      ┌───────────┐
│   PUSH    │─────>│              PROXY              │─────>│   PULL    │
│  Client 1 │      │ ┌──────────┐       ┌──────────┐ │      │ Service 1 │
└───────────┘      │ │   PULL   │──────>│   PUSH   │ │      └───────────┘
┌───────────┐      │ │ Frontend │       │ Backend  │ │      ┌───────────┐
│   PUSH    │─────>│ └──────────┘       └──────────┘ │─────>│   PULL    │
│  Client N │      └─────────────────────────────────┘      │ Service N │
└───────────┘                                               └───────────┘

The PULL frontend socket receives messages from PUSH clients and forwards them
through the proxy to PUSH services. The ZMQ proxy handles the message
routing automatically.

The PUSH backend socket forwards messages from the proxy to PULL services.
The ZMQ proxy handles the message routing automatically.
"""
