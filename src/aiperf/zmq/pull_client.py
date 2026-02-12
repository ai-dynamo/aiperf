# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import zmq.asyncio

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.messages import Message
from aiperf.common.types import MessageTypeT
from aiperf.common.utils import yield_to_event_loop
from aiperf.timing.concurrency import DynamicConcurrencyLimit
from aiperf.zmq.zmq_base_client import BaseZMQClient


class ZMQPullClient(BaseZMQClient):
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

    PULL/PUSH is a One-to-Many communication pattern. If you need Many-to-Many,
    use a ZMQ Proxy as well. see :class:`ZMQPushPullProxy` for more details.
    """

    def __init__(
        self,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
        additional_bind_address: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ZMQ Puller class.

        Args:
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
            max_pull_concurrency (int, optional): The maximum number of concurrent requests to allow.
            additional_bind_address (str, optional): Optional second address to bind to for dual-bind
                mode (e.g., IPC + TCP in Kubernetes). Only used when bind=True.
        """
        super().__init__(
            zmq.SocketType.PULL,
            address,
            bind,
            socket_ops,
            additional_bind_address=additional_bind_address,
            **kwargs,
        )
        self._pull_callbacks: dict[
            MessageTypeT, Callable[[Message], Coroutine[Any, Any, None]]
        ] = {}

        self.semaphore = DynamicConcurrencyLimit(
            max_pull_concurrency or Environment.ZMQ.PULL_MAX_CONCURRENCY
        )
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.PULL_YIELD_INTERVAL

    @background_task(immediate=True, interval=None)
    async def _pull_receiver(self) -> None:
        """Background task for receiving data from the pull socket.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        while not self.stop_requested:
            try:
                # acquire the semaphore to limit the number of concurrent requests
                # NOTE: This MUST be done BEFORE calling recv() to allow the zmq push/pull
                # logic to properly load balance the requests.
                await self.semaphore.acquire()

                message_json_bytes = await self.socket.recv()
                if self.is_trace_enabled:
                    self.trace(
                        f"Received message from pull socket: {message_json_bytes}"
                    )
                # Use AUTO-LOOKUP approach (no prefix) - optimal for large messages
                self.execute_async(self._process_message(message_json_bytes))
                self._msg_count += 1
                # Yield periodically to allow scheduled handlers to run
                # and prevent event loop starvation during message bursts.
                if (
                    self._yield_interval > 0
                    and self._msg_count % self._yield_interval == 0
                ):
                    await yield_to_event_loop()

            except zmq.Again:
                self.debug("Pull client receiver task timed out")
                self.semaphore.release()  # release the semaphore as it was not used
                await yield_to_event_loop()
            except (asyncio.CancelledError, zmq.ContextTerminated):
                self.debug("Pull client receiver task cancelled")
                self.semaphore.release()  # release the semaphore as it was not used
                break
            except Exception as e:
                self.exception(f"Exception receiving data from pull socket: {e}")
                self.semaphore.release()  # release the semaphore as it was not used
                await yield_to_event_loop()

    @on_stop
    async def _stop(self) -> None:
        """Wait for all tasks to complete."""
        await self.cancel_all_tasks()

    async def _process_message(self, message_json_bytes: bytes) -> None:
        """Process a message from the pull socket.

        This method is called by the background task when a message is received from
        the pull socket. It will deserialize the message and call the appropriate
        callback function.
        """
        try:
            # Use AUTO-LOOKUP: parse JSON to dict, extract type, validate
            # This is 40-60% faster for large messages (>2KB)
            message = Message.from_json(message_json_bytes)

            # Call callbacks with Message object
            if message.message_type in self._pull_callbacks:
                await self._pull_callbacks[message.message_type](message)
            else:
                self.warning(
                    f"Pull message received for message type {message.message_type} without callback"
                )
        finally:
            # always release the semaphore to allow receiving more messages
            self.semaphore.release()

    def register_pull_callback(
        self,
        message_type: MessageTypeT,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a ZMQ Pull data callback for a given message type.

        Note that only one callback can be registered for a given message type.

        Args:
            message_type: The message type to register the callback for.
            callback: The function to call when data is received.
        Raises:
            CommunicationError: If the client is not initialized
        """
        # Register callback
        if message_type not in self._pull_callbacks:
            self._pull_callbacks[message_type] = callback
        else:
            raise ValueError(
                f"Callback already registered for message type {message_type}"
            )
