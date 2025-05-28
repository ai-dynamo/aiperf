# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import asyncio
import logging
from collections.abc import Callable

import zmq.asyncio
from zmq import SocketType

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.decorators import aiperf_task
from aiperf.common.models import BaseMessage, Message
from aiperf.common.utils import call_all_functions

logger = logging.getLogger(__name__)


class ZMQPullClient(BaseZMQClient):
    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ Puller class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(context, SocketType.PULL, address, bind, socket_ops)
        self._pull_callbacks: dict[str, list[Callable[[Message], None]]] = {}

    @aiperf_task
    async def _pull_receiver(self) -> None:
        """Background task for receiving data from the pull socket.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        while not self.is_shutdown:
            try:
                if not self.is_initialized:
                    await self.initialized_event.wait()

                # Receive data
                message_bytes = await self.socket.recv()
                message_json = message_bytes.decode()

                # Parse JSON into a BaseMessage object
                message = BaseMessage.model_validate_json(message_json)
                topic = message.payload.message_type

                # Call callbacks with BaseMessage object
                if topic in self._pull_callbacks:
                    await call_all_functions(self._pull_callbacks[topic], message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Exception receiving data from pull socket: {e}")
                await asyncio.sleep(0.1)

    async def pull(
        self,
        topic: str,
        callback: Callable[[Message], None],
    ) -> None:
        """Register a ZMQ Pull data callback from a source (topic).

        Args:
            topic: Topic (source) to pull data from
            callback: function to call when data is received.

        Raises:
            CommunicationNotInitializedError: If the client is not initialized
            CommunicationPullError: If an exception occurred registering the pull callback
        """
        self._ensure_initialized()

        # Register callback
        if topic not in self._pull_callbacks:
            self._pull_callbacks[topic] = []
        self._pull_callbacks[topic].append(callback)
