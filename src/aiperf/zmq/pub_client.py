# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import zmq.asyncio

from aiperf.common.exceptions import CommunicationError
from aiperf.common.messages import Message, TargetedServiceMessage
from aiperf.zmq.zmq_base_client import BaseZMQClient
from aiperf.zmq.zmq_defaults import TOPIC_DELIMITER, TOPIC_END


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
            topic = self._determine_topic(message)
            message_json_bytes = message.to_json_bytes()

            # Publish message
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

    def _determine_topic(self, message: Message) -> str:
        """Determine the topic based on the message."""
        # For targeted messages such as commands, we can set the topic to a specific service by id or type
        # Note that target_service_id always takes precedence over target_service_type

        # NOTE: Keep in mind that subscriptions in ZMQ are prefix based wildcards, so the unique portion has to come first.
        if isinstance(message, TargetedServiceMessage):
            if message.target_service_id:
                return f"{message.message_type}{TOPIC_DELIMITER}{message.target_service_id}{TOPIC_END}"
            if message.target_service_type:
                return f"{message.message_type}{TOPIC_DELIMITER}{message.target_service_type}{TOPIC_END}"
        return f"{message.message_type}{TOPIC_END}"
