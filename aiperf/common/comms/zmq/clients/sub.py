# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable
from typing import Any

import zmq.asyncio

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.exceptions import CommunicationError, CommunicationErrorReason
from aiperf.common.hooks import aiperf_task
from aiperf.common.messages import Message
from aiperf.common.utils import call_all_functions


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
        super().__init__(context, zmq.SocketType.SUB, address, bind, socket_ops)
        self._subscribers: dict[str, list[Callable[[Message], Any]]] = {}

    async def subscribe(self, topic: str, callback: Callable[[Message], Any]) -> None:
        """Subscribe to a topic.

        Args:
            topic: Topic to subscribe to
            callback: Function to call when a message is received (receives Message object)

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

            self.logger.debug(
                "Subscribed to topic: %s, %s", topic, self._subscribers[topic]
            )

        except Exception as e:
            self.logger.error("Exception subscribing to topic %s: %s", topic, e)
            raise CommunicationError(
                CommunicationErrorReason.SUBSCRIBE_ERROR,
                f"Failed to subscribe to topic {topic}: {e}",
            ) from e

    async def _handle_message(self, topic_bytes: bytes, message_bytes: bytes) -> None:
        """Handle a message from a subscribed topic."""
        topic = topic_bytes.decode()
        message_json = message_bytes.decode()
        self.logger.debug(
            "Received message from topic: '%s', message: %s",
            topic,
            message_json,
        )

        message = Message.from_json(message_json)

        # Call callbacks with the parsed message object
        if topic in self._subscribers:
            await call_all_functions(self._subscribers[topic], message)

    @aiperf_task
    async def _sub_receiver(self) -> None:
        """Background task for receiving messages from subscribed topics.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        if not self.is_initialized:
            self.logger.debug(
                "Sub client %s waiting for initialization", self.client_id
            )
            await self.initialized_event.wait()
            self.logger.debug("Sub client %s initialized", self.client_id)

        while not self.stop_event.is_set():
            try:
                # Receive message
                (
                    topic_bytes,
                    message_bytes,
                ) = await self.socket.recv_multipart()
                asyncio.create_task(self._handle_message(topic_bytes, message_bytes))

            except asyncio.CancelledError:
                break
            except zmq.Again:
                # Handle ZMQ timeout or interruption
                self.logger.debug(
                    "ZMQ recv timeout due to no messages. trying again @ %s",
                    self.address,
                )
                await asyncio.sleep(0.001)
            except Exception as e:
                self.logger.error(
                    "Exception receiving message from subscription: %s, %s",
                    e,
                    type(e),
                )
                await asyncio.sleep(0.1)
