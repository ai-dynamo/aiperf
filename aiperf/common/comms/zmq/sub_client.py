# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

import zmq.asyncio

from aiperf.common.comms.base import CommunicationClientFactory
from aiperf.common.comms.zmq.zmq_base_client import BaseZMQClient
from aiperf.common.enums import CommunicationClientType, MessageType
from aiperf.common.exceptions import CommunicationError
from aiperf.common.hooks import aiperf_task, on_stop
from aiperf.common.messages import Message
from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.common.utils import call_all_functions


@CommunicationClientFactory.register(CommunicationClientType.SUB)
class ZMQSubClient(BaseZMQClient, AsyncTaskManagerMixin):
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

        self._subscribers: dict[MessageType | str, list[Callable[[Message], Any]]] = {}

    @on_stop
    async def _on_stop(self) -> None:
        await self.cancel_all_tasks()

    async def subscribe_all(
        self, message_callback_map: dict[MessageType, Callable[[Message], Any]]
    ) -> None:
        """Subscribe to all message_types in the map."""
        await self._ensure_initialized()
        for message_type, callback in message_callback_map.items():
            await self._subscribe_internal(message_type, callback)
        await asyncio.sleep(0.1)

    async def subscribe(
        self, message_type: MessageType, callback: Callable[[Message], Any]
    ) -> None:
        """Subscribe to a message_type.

        Args:
            message_type: MessageType to subscribe to
            callback: Function to call when a message is received (receives Message object)

        Raises:
            Exception if subscription was not successful, None otherwise
        """
        await self._ensure_initialized()
        await self._subscribe_internal(message_type, callback)
        await asyncio.sleep(0.1)

    async def _subscribe_internal(
        self, message_type: MessageType, callback: Callable[[Message], Any]
    ) -> None:
        """Subscribe to a message_type.

        Args:
            message_type: MessageType to subscribe to
            callback: Function to call when a message is received (receives Message object)
        """
        try:
            # Only subscribe to message_type if this is the first callback for this type
            if message_type not in self._subscribers:
                self.socket.subscribe(message_type.encode())
                self._subscribers[message_type] = []

            # Register callback
            self._subscribers[message_type].append(callback)

            self.trace(
                lambda: f"Subscribed to message_type: {message_type}, {self._subscribers[message_type]}"
            )

        except Exception as e:
            self.exception(f"Exception subscribing to message_type {message_type}: {e}")
            raise CommunicationError(
                f"Failed to subscribe to message_type {message_type}: {e}",
            ) from e

    async def _handle_message(self, topic_bytes: bytes, message_bytes: bytes) -> None:
        """Handle a message from a subscribed message_type."""
        message_type = topic_bytes.decode()
        message_json = message_bytes.decode()
        self.trace(
            lambda: f"Received message from message_type: '{message_type}', message: {message_json}"
        )

        message = Message.from_json(message_json)

        # Call callbacks with the parsed message object
        if message_type in self._subscribers:
            with contextlib.suppress(Exception):  # Ignore errors, they will get logged
                await call_all_functions(self._subscribers[message_type], message)

    @aiperf_task
    async def _sub_receiver(self) -> None:
        """Background task for receiving messages from subscribed topics.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        if not self.is_initialized:
            self.trace("Sub client %s waiting for initialization", self.client_id)
            await self.initialized_event.wait()
            self.trace(lambda: f"Sub client {self.client_id} initialized")

        while not self.stop_event.is_set():
            try:
                (
                    topic_bytes,
                    message_bytes,
                ) = await self.socket.recv_multipart()

                self.execute_async(self._handle_message(topic_bytes, message_bytes))

            except (asyncio.CancelledError, zmq.ContextTerminated):
                self.trace(
                    lambda: f"Sub client {self.client_id} receiver task cancelled"
                )
                break

            except zmq.Again:
                await asyncio.sleep(0)  # yield to the event loop
                continue

            except Exception as e:
                self.exception(
                    f"Exception receiving message from subscription: {e}, {type(e)}"
                )
                await asyncio.sleep(0.1)
