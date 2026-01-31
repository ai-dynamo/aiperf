# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import zmq.asyncio

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.messages import ErrorMessage, Message
from aiperf.common.models import ErrorDetails
from aiperf.common.types import MessageTypeT
from aiperf.common.utils import yield_to_event_loop
from aiperf.zmq.zmq_base_client import BaseZMQClient


class ZMQRouterReplyClient(BaseZMQClient):
    """
    ZMQ ROUTER socket client for handling requests from DEALER clients.

    The ROUTER socket receives requests from DEALER clients and sends responses
    back to the originating DEALER client using routing envelopes.

    ASCII Diagram:
    ┌──────────────┐                    ┌──────────────┐
    │    DEALER    │───── Request ─────>│              │
    │   (Client)   │<──── Response ─────│              │
    └──────────────┘                    │              │
    ┌──────────────┐                    │    ROUTER    │
    │    DEALER    │───── Request ─────>│  (Service)   │
    │   (Client)   │<──── Response ─────│              │
    └──────────────┘                    │              │
    ┌──────────────┐                    │              │
    │    DEALER    │───── Request ─────>│              │
    │   (Client)   │<──── Response ─────│              │
    └──────────────┘                    └──────────────┘

    Usage Pattern:
    - ROUTER handles requests from multiple DEALER clients
    - Maintains routing envelopes to send responses back
    - Many-to-one request handling pattern
    - Supports concurrent request processing

    ROUTER/DEALER is a Many-to-One communication pattern. If you need Many-to-Many,
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
        Initialize the ZMQ Router (Rep) client class.

        Args:
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(zmq.SocketType.ROUTER, address, bind, socket_ops, **kwargs)

        self._request_handlers: dict[
            MessageTypeT,
            tuple[str, Callable[[Message], Coroutine[Any, Any, Message | None]]],
        ] = {}
        self._response_futures: dict[str, asyncio.Future[Message | None]] = {}
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.REPLY_YIELD_INTERVAL

    @on_stop
    async def _clear_request_handlers(self) -> None:
        self._request_handlers.clear()

    def register_request_handler(
        self,
        service_id: str,
        message_type: MessageTypeT,
        handler: Callable[[Message], Coroutine[Any, Any, Message | None]],
    ) -> None:
        """Register a request handler. Anytime a request is received that matches the
        message type, the handler will be called. The handler should return a response
        message. If the handler returns None, the request will be ignored.

        Note that there is a limit of 1 to 1 mapping between message type and handler.

        Args:
            service_id: The service ID to register the handler for
            message_type: The message type to register the handler for
            handler: The handler to register
        """
        if message_type in self._request_handlers:
            raise ValueError(
                f"Handler already registered for message type {message_type}"
            )

        self.debug(
            lambda service_id=service_id,
            type=message_type: f"Registering request handler for {service_id} with message type {type}"
        )
        self._request_handlers[message_type] = (service_id, handler)

    async def _handle_request(self, request_id: str, request: Message) -> None:
        """Handle a request.

        This method will:
        - Parse the request JSON to create a Message object
        - Call the handler for the message type
        - Set the response future
        """
        message_type = request.message_type

        try:
            _, handler = self._request_handlers[message_type]
            response = await handler(request)

        except Exception as e:
            self.exception(f"Exception calling handler for {message_type}: {e}")
            response = ErrorMessage(
                request_id=request_id,
                error=ErrorDetails.from_exception(e),
            )

        try:
            self._response_futures[request_id].set_result(response)
        except Exception as e:
            self.exception(
                f"Exception setting response future for request {request_id}: {e}"
            )

    async def _wait_for_response(
        self, request_id: str, routing_envelope: tuple[bytes, ...]
    ) -> None:
        """Wait for a response to a request.

        This method will wait for the response future to be set and then send the response
        back to the client.
        """
        try:
            # Wait for the response asynchronously.
            response = await self._response_futures[request_id]

            if response is None:
                self.warning(
                    lambda req_id=request_id: f"Got None as response for request {req_id}"
                )
                response = ErrorMessage(
                    request_id=request_id,
                    error=ErrorDetails(
                        type="NO_RESPONSE",
                        message="No response was generated for the request.",
                    ),
                )

            self._response_futures.pop(request_id, None)

            # Send the response back to the client.
            await self.socket.send_multipart(
                [*routing_envelope, response.to_json_bytes()]
            )
        except Exception as e:
            self.exception(
                f"Exception waiting for response for request {request_id}: {e}"
            )

    @background_task(immediate=True, interval=None)
    async def _rep_router_receiver(self) -> None:
        """Background task for receiving requests and sending responses.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for requests from the socket and send responses in
        an asynchronous manner.
        """
        self.debug("Router reply client background task initialized")

        while not self.stop_requested:
            try:
                # Receive request
                try:
                    data = await self.socket.recv_multipart()
                    self.trace(lambda msg=data: f"Received request: {msg}")

                    request = Message.from_json(data[-1])
                    if not request.request_id:
                        self.exception(f"Request ID is missing from request: {data}")
                        continue

                    routing_envelope: tuple[bytes, ...] = (
                        tuple(data[:-1])
                        if len(data) > 1
                        else (request.request_id.encode(),)
                    )
                except zmq.Again:
                    # This means we timed out waiting for a request.
                    # We can continue to the next iteration of the loop.
                    self.debug("Router reply client receiver task timed out")
                    await yield_to_event_loop()
                    continue

                # Create a new response future for this request that will be resolved
                # when the handler returns a response.
                self._response_futures[request.request_id] = asyncio.Future()
                # Handle the request in a new task.
                self.execute_async(self._handle_request(request.request_id, request))
                self.execute_async(
                    self._wait_for_response(request.request_id, routing_envelope)
                )
                self._msg_count += 1
                # Yield periodically to allow scheduled handlers to run
                # and prevent event loop starvation during message bursts.
                if self._yield_interval > 0 and self._msg_count >= self._yield_interval:
                    await yield_to_event_loop()

            except Exception as e:
                self.exception(f"Exception receiving request: {e}")
                await yield_to_event_loop()
            except asyncio.CancelledError:
                self.debug("Router reply client receiver task cancelled")
                break
