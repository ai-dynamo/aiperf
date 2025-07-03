# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
from collections.abc import Callable, Coroutine
from typing import Any

import zmq.asyncio

from aiperf.common.comms.base import CommunicationClientFactory
from aiperf.common.comms.zmq.zmq_base_client import BaseZMQClient
from aiperf.common.enums import CommunicationClientType, MessageType
from aiperf.common.hooks import aiperf_task, on_stop
from aiperf.common.messages import Message
from aiperf.common.mixins import AsyncTaskManagerMixin


@CommunicationClientFactory.register(CommunicationClientType.PULL)
class ZMQPullClient(BaseZMQClient, AsyncTaskManagerMixin):
    """
    ZMQ PULL socket client for receiving work from PUSH sockets.

    The PULL socket receives messages from PUSH sockets in a pipeline pattern,
    distributing work fairly among multiple PULL workers.

    ASCII Diagram:
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │    PUSH     │      │    PULL     │      │    PULL     │
    │ (Producer)  │      │ (Worker 1)  │      │ (Worker 2)  │
    │             │      └─────────────┘      └─────────────┘
    │   Tasks:    │             ▲                     ▲
    │   - Task A  │─────────────┘                     │
    │   - Task B  │───────────────────────────────────┘
    │   - Task C  │─────────────┐
    │   - Task D  │             ▼
    └─────────────┘      ┌─────────────┐
                         │    PULL     │
                         │ (Worker N)  │
                         └─────────────┘

    Usage Pattern:
    - PULL receives work from multiple PUSH producers
    - Work is fairly distributed among PULL workers
    - Pipeline pattern for distributed processing
    - Each message is delivered to exactly one PULL socket
    """

    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        """
        Initialize the ZMQ Puller class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
            max_concurrency (int, optional): The maximum number of concurrent requests to allow.
        """
        super().__init__(context, zmq.SocketType.PULL, address, bind, socket_ops)
        self._pull_callbacks: dict[
            MessageType, Callable[[Message], Coroutine[Any, Any, None]]
        ] = {}

        if max_concurrency is not None:
            self.semaphore = asyncio.Semaphore(value=max_concurrency)
        else:
            self.semaphore = asyncio.Semaphore(
                value=int(os.getenv("AIPERF_WORKER_CONCURRENT_REQUESTS", 500))
            )

    @aiperf_task
    async def _pull_receiver(self) -> None:
        """Background task for receiving data from the pull socket.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        if not self.is_initialized:
            await self.initialized_event.wait()

        while not self.stop_event.is_set():
            try:
                # acquire the semaphore to limit the number of concurrent requests
                # NOTE: This MUST be done BEFORE calling recv_string() to allow the zmq push/pull
                # logic to properly load balance the requests.
                await self.semaphore.acquire()

                message_json = await self.socket.recv_string()
                self.logger.debug("Received message from pull socket: %s", message_json)
                self.execute_async(self._process_message(message_json))

            except zmq.Again:
                self.semaphore.release()  # release the semaphore as it was not used
                await asyncio.sleep(0)  # allow other tasks to run
                continue

            except (asyncio.CancelledError, zmq.ContextTerminated):
                self.semaphore.release()  # release the semaphore as it was not used
                break

            except Exception as e:
                self.logger.error(
                    "Exception receiving data from pull socket: %s %s",
                    type(e).__name__,
                    e,
                )
                await asyncio.sleep(0.1)

    @on_stop
    async def _stop(self) -> None:
        """Wait for all tasks to complete."""
        await self.cancel_all_tasks()

    async def _process_message(self, message_json: str) -> None:
        """Process a message from the pull socket.

        This method is called by the background task when a message is received from
        the pull socket. It will deserialize the message and call the appropriate
        callback function.
        """
        try:
            message = Message.from_json(message_json)

            # Call callbacks with Message object
            if message.message_type in self._pull_callbacks:
                await self._pull_callbacks[message.message_type](message)
            else:
                self.logger.warning(
                    "Pull message received for message type %s without callback",
                    message.message_type,
                )
        finally:
            # always release the semaphore to allow receiving more messages
            self.semaphore.release()

    async def register_pull_callback(
        self,
        message_type: MessageType,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
        max_concurrency: int | None = None,
    ) -> None:
        """Register a ZMQ Pull data callback for a given message type.

        Note that only one callback can be registered for a given message type.

        Args:
            message_type: The message type to register the callback for.
            callback: The function to call when data is received.
            max_concurrency: The maximum number of concurrent requests to allow.
        Raises:
            CommunicationError: If the client is not initialized
        """
        await self._ensure_initialized()

        # Register callback
        if message_type not in self._pull_callbacks:
            self._pull_callbacks[message_type] = callback
        else:
            raise ValueError(
                f"Callback already registered for message type {message_type}"
            )

        if max_concurrency is not None:
            self.semaphore = asyncio.Semaphore(value=max_concurrency)
