# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import zmq.asyncio

from aiperf.common.comms.base import CommunicationClientFactory
from aiperf.common.comms.zmq.zmq_base_client import BaseZMQClient
from aiperf.common.enums import CommunicationClientType
from aiperf.common.exceptions import CommunicationError
from aiperf.common.messages import Message


@CommunicationClientFactory.register(CommunicationClientType.PUB)
class ZMQPubClient(BaseZMQClient):
    """
    The PUB socket broadcasts messages to all connected SUB sockets that have
    subscribed to the message topic/type.

    ASCII Diagram:
    ┌──────────────┐    ┌──────────────┐
    │     PUB      │───>│              │
    │ (Publisher)  │    │              │
    └──────────────┘    │     SUB      │
    ┌──────────────┐    │ (Subscriber) │
    │     PUB      │───>│              │
    │ (Publisher)  │    │              │
    └──────────────┘    └──────────────┘
    OR
    ┌──────────────┐    ┌──────────────┐
    │              │───>│     SUB      │
    │              │    │ (Subscriber) │
    │     PUB      │    └──────────────┘
    │ (Publisher)  │    ┌──────────────┐
    │              │───>│     SUB      │
    │              │    │ (Subscriber) │
    └──────────────┘    └──────────────┘

    Usage Pattern:
    - Single PUB socket broadcasts messages to all subscribers (One-to-Many)
    OR
    - Multiple PUB sockets broadcast messages to a single SUB socket (Many-to-One)

    - SUB sockets filter messages by topic/message_type
    - Fire-and-forget messaging (no acknowledgments)

    PUB/SUB is a One-to-Many communication pattern. If you need Many-to-Many,
    use a ZMQ Proxy as well. see :class:`ZMQXPubXSubProxy` for more details.
    """

    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ Publisher client class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(context, zmq.SocketType.PUB, address, bind, socket_ops)

    async def publish(self, message: Message) -> None:
        """Publish a message. The topic will be set automatically based on the message type.

        Args:
            message: Message to publish (must be a Message object)
        """
        await self._ensure_initialized()

        try:
            message_json = message.model_dump_json()

            # Publish message
            await self.socket.send_multipart(
                [message.message_type.encode(), message_json.encode()]
            )

        except (asyncio.CancelledError, zmq.ContextTerminated):
            return

        except Exception as e:
            raise CommunicationError(
                f"Failed to publish message {message.message_type}: {e}",
            ) from e
