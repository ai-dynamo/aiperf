# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import zmq.asyncio

from aiperf.common.exceptions import CommunicationError
from aiperf.common.messages import Message
from aiperf.zmq.zmq_base_client import BaseZMQClient
from aiperf.zmq.zmq_defaults import TOPIC_END


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
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ZMQ Publisher client class.

        Args:
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(zmq.SocketType.PUB, address, bind, socket_ops, **kwargs)

    async def publish(self, message: Message) -> None:
        """Publish a message. The topic will be set automatically based on the message type.

        Args:
            message: Message to publish (must be a Message object)
        """
        await self._check_initialized()

        try:
            topic = f"{message.message_type}{TOPIC_END}"
            message_json_bytes = message.to_json_bytes()

            message_type = message.message_type.lower()
            if message_type.startswith("dataset_"):
                self.debug(
                    lambda: f"PUB client {self.client_id} publishing {message.message_type} to topic: {topic}"
                )
            elif "notification" in message_type:
                self.info(
                    f"PUB client {self.client_id} publishing {message.message_type} to topic: {topic}"
                )
            else:
                self.trace(lambda: f"Publishing message {topic=} {message_json_bytes=}")
            await self.socket.send_multipart([topic.encode(), message_json_bytes])

        except (asyncio.CancelledError, zmq.ContextTerminated):
            self.debug(
                lambda: f"Pub client {self.client_id} cancelled or context terminated"
            )
            return

        except Exception as e:
            raise CommunicationError(
                f"Failed to publish message {message.message_type}: {e}",
            ) from e
