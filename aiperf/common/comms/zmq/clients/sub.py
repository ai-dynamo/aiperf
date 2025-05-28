#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
from collections.abc import Callable
from typing import Any

import zmq.asyncio
from zmq import SocketType

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.exceptions import (
    CommunicationSubscribeError,
)
from aiperf.common.hooks import aiperf_task
from aiperf.common.models import BaseMessage, Message
from aiperf.common.utils import call_all_functions

logger = logging.getLogger(__name__)


class ZMQSubClient(BaseZMQClient):
    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ Subscriber class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(context, SocketType.SUB, address, bind, socket_ops)
        self._subscribers: dict[str, list[Callable[[Message], Any]]] = {}

    async def subscribe(self, topic: str, callback: Callable[[Message], Any]) -> None:
        """Subscribe to a topic.

        Args:
            topic: Topic to subscribe to
            callback: Function to call when a response is received (receives Message object)

        Raises:
            Exception if subscription was not successful, None otherwise
        """
        self._ensure_initialized()

        try:
            # Subscribe to topic
            self.socket.subscribe(topic.encode())

            # Register callback
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            self._subscribers[topic].append(callback)

            logger.debug("Subscribed to topic: %s, %s", topic, self._subscribers[topic])

        except Exception as e:
            logger.error("Exception subscribing to topic %s: %s", topic, e)
            raise CommunicationSubscribeError from e

    @aiperf_task
    async def _sub_receiver(self) -> None:
        """Background task for receiving messages from subscribed topics.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        while not self.is_shutdown:
            try:
                if not self.is_initialized:
                    logger.debug(
                        "Sub client %s waiting for initialization", self.client_id
                    )
                    await self.initialized_event.wait()
                    logger.debug("Sub client %s initialized", self.client_id)

                # Receive response
                (
                    topic_bytes,
                    message_bytes,
                ) = await self.socket.recv_multipart()
                topic = topic_bytes.decode()
                message_json = message_bytes.decode()
                logger.debug(
                    "Client %s received message from topic: '%s', message: %s",
                    self.client_id,
                    topic,
                    message_json,
                )

                message = BaseMessage.model_validate_json(message_json)

                # Call callbacks with the parsed response object
                if topic in self._subscribers:
                    await call_all_functions(self._subscribers[topic], message)

            except asyncio.CancelledError:
                break
            except zmq.Again:
                # Handle ZMQ timeout or interruption
                logger.debug(
                    "ZMQ recv timeout due to no messages. trying again @ %s",
                    self.address,
                )
                await asyncio.sleep(0.001)
            except Exception as e:
                logger.error(
                    "Exception receiving response from subscription: %s, %s",
                    e,
                    type(e),
                )
                await asyncio.sleep(0.1)
