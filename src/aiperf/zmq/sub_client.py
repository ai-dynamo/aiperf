# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import zmq.asyncio

from aiperf.common.environment import Environment
from aiperf.common.exceptions import CommunicationError
from aiperf.common.hooks import background_task
from aiperf.common.messages import Message
from aiperf.common.types import MessageTypeT
from aiperf.common.utils import call_all_functions, yield_to_event_loop
from aiperf.zmq.zmq_base_client import BaseZMQClient
from aiperf.zmq.zmq_defaults import (
    TOPIC_END_ENCODED,
    TOPIC_END_LENGTH,
)


class ZMQSubClient(BaseZMQClient):
    """
    ZMQ SUB socket client for subscribing to messages from PUB sockets.
    One-to-Many or Many-to-One communication pattern.

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
    - Single SUB socket subscribes to multiple PUB publishers (One-to-Many)
    OR
    - Multiple SUB sockets subscribe to a single PUB publisher (Many-to-One)

    - Subscribes to specific message topics/types
    - Receives all messages matching subscriptions

    SUB/PUB is a One-to-Many communication pattern. If you need Many-to-Many,
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
        Initialize the ZMQ Subscriber class.

        Args:
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(zmq.SocketType.SUB, address, bind, socket_ops, **kwargs)

        self._subscribers: dict[MessageTypeT, list[Callable[[Message], Any]]] = {}
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.SUB_YIELD_INTERVAL

    async def subscribe_all(
        self,
        message_callback_map: dict[
            MessageTypeT,
            Callable[[Message], Any] | list[Callable[[Message], Any]],
        ],
    ) -> None:
        """Subscribe to all message_types in the map. For each MessageType, a single
        callback or a list of callbacks can be provided."""
        await self._check_initialized()
        for message_type, callbacks in message_callback_map.items():
            if isinstance(callbacks, list):
                for callback in callbacks:
                    await self._subscribe_internal(message_type, callback)
            else:
                await self._subscribe_internal(message_type, callbacks)

    async def subscribe(
        self, message_type: MessageTypeT, callback: Callable[[Message], Any]
    ) -> None:
        """Subscribe to a message_type.

        Args:
            message_type: MessageTypeT to subscribe to
            callback: Function to call when a message is received (receives Message object)

        Raises:
            Exception if subscription was not successful, None otherwise
        """
        await self._check_initialized()
        await self._subscribe_internal(message_type, callback)

    async def _subscribe_internal(
        self, topic: str, callback: Callable[[Message], Any]
    ) -> None:
        """Subscribe to a topic.

        Args:
            topic: MessageTypeT to subscribe to
            callback: Function to call when a message is received (receives Message object)
        """
        try:
            # Only subscribe to topic if this is the first callback for this type
            if topic not in self._subscribers:
                self.debug(lambda: f"Subscribed to topic: {topic}")
                self.socket.setsockopt(
                    zmq.SUBSCRIBE, topic.encode() + TOPIC_END_ENCODED
                )
            else:
                self.debug(
                    lambda: f"Adding callback to existing subscription for topic: {topic}"
                )

            self._subscribers.setdefault(topic, []).append(callback)

        except Exception as e:
            self.exception(f"Exception subscribing to topic {topic}: {e}")
            raise CommunicationError(
                f"Failed to subscribe to topic {topic}: {e}",
            ) from e

    async def _handle_message(self, topic_bytes: bytes, message_bytes: bytes) -> None:
        """Handle a message from a subscribed message_type."""

        # strip the final TOPIC_END chars from the topic
        topic = topic_bytes.decode()[:-TOPIC_END_LENGTH]
        self.trace(
            lambda: f"Received message from topic: '{topic}', message: {message_bytes}"
        )

        # Use AUTO-LOOKUP for all messages - single parse with multi-level routing
        # This is optimal for our workload (84% large messages in push/pull, 45% in pub/sub)
        message = Message.from_json(message_bytes)

        self.trace(
            lambda: f"Calling callbacks for message: {message}, {self._subscribers.get(topic)}"
        )

        # Call callbacks with the parsed message object
        if topic in self._subscribers:
            with contextlib.suppress(Exception):  # Ignore errors, they will get logged
                await call_all_functions(self._subscribers[topic], message)

    @background_task(immediate=True, interval=None)
    async def _sub_receiver(self) -> None:
        """Background task for receiving messages from subscribed topics.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        while not self.stop_requested:
            try:
                topic_bytes, message_bytes = await self.socket.recv_multipart()
                if self.is_trace_enabled:
                    self.trace(
                        f"Socket received message: {topic_bytes} {message_bytes}"
                    )
                # NOTE: This must be async otherwise it may deadlock the event loop.
                self.execute_async(self._handle_message(topic_bytes, message_bytes))
                self._msg_count += 1
                # Yield periodically to allow scheduled handlers to run
                # and prevent event loop starvation during message bursts.
                if (
                    self._yield_interval > 0
                    and self._msg_count % self._yield_interval == 0
                ):
                    await yield_to_event_loop()

            except zmq.Again:
                self.debug(f"Sub client {self.client_id} receiver task timed out")
                await yield_to_event_loop()
            except (asyncio.CancelledError, zmq.ContextTerminated):
                self.debug(f"Sub client {self.client_id} receiver task cancelled")
                break
            except Exception as e:
                self.exception(
                    f"Exception receiving message from subscription: {e}, {type(e)}"
                )
                await yield_to_event_loop()
