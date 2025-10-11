# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

import zmq.asyncio

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommClientType
from aiperf.common.environment import Environment
from aiperf.common.exceptions import CommunicationError
from aiperf.common.factories import CommunicationClientFactory
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.messages import Message
from aiperf.common.mixins import TaskManagerMixin
from aiperf.common.protocols import RequestClientProtocol
from aiperf.common.utils import yield_to_event_loop
from aiperf.zmq.zmq_base_client import BaseZMQClient


@implements_protocol(RequestClientProtocol)
@CommunicationClientFactory.register(CommClientType.REQUEST)
class ZMQDealerRequestClient(BaseZMQClient, TaskManagerMixin):
    """
    ZMQ DEALER socket client for asynchronous request-response communication.

    The DEALER socket connects to ROUTER sockets and can send requests asynchronously,
    receiving responses through callbacks or awaitable futures.

    ASCII Diagram:
    ┌──────────────┐                    ┌──────────────┐
    │    DEALER    │───── Request ─────>│    ROUTER    │
    │   (Client)   │                    │  (Service)   │
    │              │<─── Response ──────│              │
    └──────────────┘                    └──────────────┘

    Usage Pattern:
    - DEALER Clients send requests to ROUTER Services
    - Responses are routed back to the originating DEALER

    DEALER/ROUTER is a Many-to-One communication pattern. If you need Many-to-Many,
    use a ZMQ Proxy as well. see :class:`ZMQDealerRouterProxy` for more details.
    """

    def __init__(
        self,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ZMQ Dealer (Req) client class.

        Args:
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(zmq.SocketType.DEALER, address, bind, socket_ops, **kwargs)

        self.request_callbacks: dict[
            str, Callable[[Message], Coroutine[Any, Any, None]]
        ] = {}

    @background_task(immediate=True, interval=None)
    async def _request_async_task(self) -> None:
        """Task to handle incoming requests."""
        while not self.stop_requested:
            try:
                message = await self.socket.recv_string()
                self.trace(lambda msg=message: f"Received response: {msg}")
                response_message = Message.from_json(message)

                # Call the callback if it exists
                if response_message.request_id in self.request_callbacks:
                    callback = self.request_callbacks.pop(response_message.request_id)
                    self.execute_async(callback(response_message))

            except zmq.Again:
                self.debug("No data on dealer socket received, yielding to event loop")
                await yield_to_event_loop()
            except Exception as e:
                self.exception(f"Exception receiving responses: {e}")
                await yield_to_event_loop()
            except asyncio.CancelledError:
                self.debug("Dealer request client receiver task cancelled")
                raise  # re-raise the cancelled error

    @on_stop
    async def _stop_remaining_tasks(self) -> None:
        """Wait for all tasks to complete."""
        await self.cancel_all_tasks()

    async def request_async(
        self,
        message: Message,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Send a request and be notified when the response is received."""
        await self._check_initialized()

        if not isinstance(message, Message):
            raise TypeError(
                f"message must be an instance of Message, got {type(message).__name__}"
            )

        # Generate request ID if not provided so that responses can be matched
        if not message.request_id:
            message.request_id = str(uuid.uuid4())

        self.request_callbacks[message.request_id] = callback

        request_json = message.model_dump_json()
        self.trace(lambda msg=request_json: f"Sending request: {msg}")

        try:
            await self.socket.send_string(request_json)

        except Exception as e:
            raise CommunicationError(
                f"Exception sending request: {e.__class__.__qualname__} {e}",
            ) from e

    async def request(
        self,
        message: Message,
        timeout: float = Environment.COMMS_REQUEST_TIMEOUT,
    ) -> Message:
        """Send a request and wait for a response up to timeout seconds.

        Args:
            message (Message): The request message to send.
            timeout (float): Maximum time to wait for a response in seconds.

        Returns:
            Message: The response message received.

        Raises:
            CommunicationError: if the request fails, or
            asyncio.TimeoutError: if the response is not received in time.
        """
        future = asyncio.Future[Message]()

        async def callback(response_message: Message) -> None:
            future.set_result(response_message)

        await self.request_async(message, callback)
        return await asyncio.wait_for(future, timeout=timeout)
