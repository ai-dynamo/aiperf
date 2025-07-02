# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

import zmq.asyncio
from zmq import SocketType

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.comms.zmq.clients.zmq_proxy_base import BaseZMQProxy
from aiperf.common.config.zmq_config import BaseZMQProxyConfig
from aiperf.common.enums import ZMQProxyType
from aiperf.common.factories import ZMQProxyFactory

################################################################################
# Proxy Sockets
################################################################################


def create_proxy_socket_class(
    socket_type: SocketType, is_backend: bool = False
) -> type[BaseZMQClient]:
    """Create a proxy socket class using the specified socket type. This is used to
    reduce the boilerplate code required to create a ZMQ Proxy class.
    """

    class_name = (
        f"ZMQProxy{'Backend' if is_backend else 'Frontend'}Socket{socket_type.name}"
    )

    class ProxySocket(BaseZMQClient):
        """A ZMQ Proxy socket class with a specific socket type."""

        def __init__(
            self,
            context: zmq.asyncio.Context,
            address: str,
            socket_ops: dict | None = None,
        ):
            super().__init__(
                context, socket_type, address, bind=True, socket_ops=socket_ops
            )
            self.logger.debug(
                "ZMQ Proxy %s %s - Address: %s",
                "backend" if is_backend else "frontend",
                socket_type.name,
                address,
            )

    ProxySocket.__name__ = class_name
    ProxySocket.__qualname__ = class_name
    return ProxySocket


def define_proxy_class(
    proxy_type: ZMQProxyType,
    frontend_socket_class: type[BaseZMQClient],
    backend_socket_class: type[BaseZMQClient],
) -> None:
    """This function reduces the boilerplate code required to create a ZMQ Proxy class.
    It will generate a ZMQ Proxy class and register it with the ZMQProxyFactory.

    Args:
        proxy_type: The type of proxy to generate.
        frontend_socket_class: The class of the frontend socket.
        backend_socket_class: The class of the backend socket.
    """

    @ZMQProxyFactory.register(proxy_type)
    class ZMQProxy(BaseZMQProxy):
        """
        A Generated ZMQ Proxy class.

        This class is responsible for creating the ZMQ proxy that forwards messages
        between frontend and backend sockets.
        """

        def __init__(
            self,
            context: zmq.asyncio.Context,
            zmq_proxy_config: BaseZMQProxyConfig,
            socket_ops: dict | None = None,
        ) -> None:
            super().__init__(
                frontend_socket_class=frontend_socket_class,
                backend_socket_class=backend_socket_class,
                context=context,
                zmq_proxy_config=zmq_proxy_config,
                socket_ops=socket_ops,
            )

        @classmethod
        def from_config(
            cls,
            config: BaseZMQProxyConfig | None,
            socket_ops: dict | None = None,
        ) -> Self | None:
            if config is None:
                return None
            return cls(
                context=zmq.asyncio.Context.instance(),
                zmq_proxy_config=config,
                socket_ops=socket_ops,
            )


################################################################################
# XPUB/XSUB Proxy
################################################################################

define_proxy_class(
    ZMQProxyType.XPUB_XSUB,
    create_proxy_socket_class(SocketType.XSUB, is_backend=False),
    create_proxy_socket_class(SocketType.XPUB, is_backend=True),
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

define_proxy_class(
    ZMQProxyType.DEALER_ROUTER,
    create_proxy_socket_class(SocketType.ROUTER, is_backend=False),
    create_proxy_socket_class(SocketType.DEALER, is_backend=True),
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

define_proxy_class(
    ZMQProxyType.PUSH_PULL,
    create_proxy_socket_class(SocketType.PULL, is_backend=False),
    create_proxy_socket_class(SocketType.PUSH, is_backend=True),
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
