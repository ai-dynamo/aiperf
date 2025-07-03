# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

import zmq.asyncio

from aiperf.common.comms.base import CommunicationClientFactory
from aiperf.common.comms.zmq.zmq_base_client import BaseZMQClient
from aiperf.common.enums import CommunicationClientType, MessageType
from aiperf.common.hooks import aiperf_task, on_cleanup, on_stop
from aiperf.common.messages import ErrorMessage, Message
from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.common.record_models import ErrorDetails


@CommunicationClientFactory.register(CommunicationClientType.REPLY)
class ZMQRouterReplyClient(BaseZMQClient, AsyncTaskManagerMixin):
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
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ Router (Rep) client class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(context, zmq.SocketType.ROUTER, address, bind, socket_ops)

        self._request_handlers: dict[
            MessageType,
            tuple[str, Callable[[Message], Coroutine[Any, Any, Message | None]]],
        ] = {}
        self._response_futures: dict[tuple[bytes], asyncio.Future[Message | None]] = {}

    @on_stop
    async def _on_stop(self) -> None:
        await self.cancel_all_tasks()

    @on_cleanup
    async def _cleanup(self) -> None:
        self._request_handlers.clear()

    def register_request_handler(
        self,
        service_id: str,
        message_type: MessageType,
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

        self._request_handlers[message_type] = (service_id, handler)

    async def _handle_request(
        self, routing_envelope: tuple[bytes], request: Message
    ) -> None:
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
            self.logger.error("Exception calling handler for %s: %s", message_type, e)
            response = ErrorMessage(
                request_id=request.request_id,
                error=ErrorDetails.from_exception(e),
            )

        self._response_futures[routing_envelope].set_result(response)

    async def _wait_for_response(
        self, request_id: str, routing_envelope: tuple[bytes]
    ) -> None:
        """Wait for a response to a request.

        This method will wait for the response future to be set and then send the response
        back to the client.
        """
        # Wait for the response asynchronously.
        response = await self._response_futures[routing_envelope]

        if response is None:
            self.logger.warning("Got None as response for request %s", request_id)
            response = ErrorMessage(
                request_id=request_id,
                error=ErrorDetails(
                    type="NO_RESPONSE",
                    message="No response was generated for the request.",
                ),
            )

        # Send the response back to the client.
        await self.socket.send_multipart(
            [*routing_envelope, response.model_dump_json().encode()]
        )

    @aiperf_task
    async def _rep_router_receiver(self) -> None:
        """Background task for receiving requests and sending responses.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for requests from the socket and send responses in
        an asynchronous manner.
        """
        if not self.is_initialized:
            await self.initialized_event.wait()

        while not self.stop_event.is_set():
            try:
                # Receive request
                try:
                    data = await self.socket.recv_multipart()
                    self.logger.debug("Received request: %s", data)
                    if len(data) < 2:
                        self.logger.error("Invalid request data: %s", data)
                        continue
                    request = Message.from_json(data[-1])
                    routing_envelope: tuple[bytes] = tuple(data[:-1])
                except zmq.Again:
                    # This means we timed out waiting for a request.
                    # We can continue to the next iteration of the loop.
                    continue

                # Create a new response future for this request that will be resolved
                # when the handler returns a response.
                self._response_futures[routing_envelope] = asyncio.Future()

                # Handle the request in a new task.
                self.execute_async(self._handle_request(routing_envelope, request))

                # TODO: Can we handle the waiting for responses on a separate task, or will this fail?
                if request.request_id:
                    self.execute_async(
                        self._wait_for_response(request.request_id, routing_envelope)
                    )

            except asyncio.CancelledError:
                break
            except Exception:
                self.logger.exception("Exception receiving request")
                await asyncio.sleep(0.1)
