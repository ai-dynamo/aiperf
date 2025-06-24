# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import zmq.asyncio

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.exceptions import CommunicationError, CommunicationErrorReason
from aiperf.common.messages import Message

logger = logging.getLogger(__name__)


class ZMQPubClient(BaseZMQClient):
    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ Publisher class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(context, zmq.SocketType.PUB, address, bind, socket_ops)

    async def publish(self, topic: str, message: Message) -> None:
        """Publish a message to a topic. Fairly straightforward, just dumps the message
        and sends it over the socket.

        Args:
            topic: Topic to publish to
            message: Message to publish (must be a Message object)

        Raises:
            CommunicationError: If the client is not initialized
                or the message was not published successfully
        """
        self._ensure_initialized()

        try:
            message_json = message.model_dump_json()

            # Publish message
            await self.socket.send_multipart([topic.encode(), message_json.encode()])

        except Exception as e:
            raise CommunicationError(
                CommunicationErrorReason.PUBLISH_ERROR,
                f"Failed to publish message to topic {topic}: {e}",
            ) from e
