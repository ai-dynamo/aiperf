# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable
from typing import Any

import zmq.asyncio

from aiperf.common.comms.base import CommunicationClientFactory
from aiperf.common.comms.zmq.zmq_base_client import BaseZMQClient
from aiperf.common.enums import CommunicationClientType, MessageType
from aiperf.common.exceptions import CommunicationError, CommunicationErrorReason
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

        try:
            # Subscribe to message_type
            self.socket.subscribe(message_type.encode())

            # Register callback
            if message_type not in self._subscribers:
                self._subscribers[message_type] = []
            self._subscribers[message_type].append(callback)

            self.logger.debug(
                "Subscribed to message_type: %s, %s",
                message_type,
                self._subscribers[message_type],
            )

        except Exception as e:
            self.logger.error(
                "Exception subscribing to message_type %s: %s", message_type, e
            )
            raise CommunicationError(
                CommunicationErrorReason.SUBSCRIBE_ERROR,
                f"Failed to subscribe to message_type {message_type}: {e}",
            ) from e

    async def _handle_message(self, topic_bytes: bytes, message_bytes: bytes) -> None:
        """Handle a message from a subscribed message_type."""
        message_type = topic_bytes.decode()
        message_json = message_bytes.decode()
        self.logger.debug(
            "Received message from message_type: '%s', message: %s",
            message_type,
            message_json,
        )

        message = Message.from_json(message_json)

        # Call callbacks with the parsed message object
        if message_type in self._subscribers:
            await call_all_functions(self._subscribers[message_type], message)

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
                (
                    topic_bytes,
                    message_bytes,
                ) = await self.socket.recv_multipart()

                self.execute_async(self._handle_message(topic_bytes, message_bytes))

            except (asyncio.CancelledError, zmq.ContextTerminated):
                break

            except zmq.Again:
                pass

            except Exception as e:
                self.logger.error(
                    "Exception receiving message from subscription: %s, %s",
                    e,
                    type(e),
                )
                await asyncio.sleep(0.1)
