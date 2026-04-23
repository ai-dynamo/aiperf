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
            tuple[str, Callable[[Message], Coroutine[Any, Any, Message | None]], bool],
        ] = {}
        self._response_futures: dict[str, asyncio.Future[Message | None]] = {}
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.REPLY_YIELD_INTERVAL

    @on_stop
    async def _clear_request_handlers(self) -> None:
        self._request_handlers.clear()
        for future in self._response_futures.values():
            if not future.done():
                future.cancel()
        self._response_futures.clear()

    def register_request_handler(
        self,
        service_id: str,
        message_type: MessageTypeT,
        handler: Callable[[Message], Coroutine[Any, Any, Message | None]],
        *,
        fire_and_forget: bool = False,
    ) -> None:
        """Register a handler for incoming messages of the given type.

        For request-reply handlers, the handler must return a Message. If it returns
        None, an ErrorMessage is sent back to the caller.

        For fire-and-forget handlers (fire_and_forget=True), no response is sent.
        The handler's return value is ignored.

        Args:
            service_id: The service ID to register the handler for.
            message_type: The message type to register the handler for.
            handler: The async handler function.
            fire_and_forget: If True, no response is sent back to the caller.
        """
        if message_type in self._request_handlers:
            raise ValueError(
                f"Handler already registered for message type {message_type}"
            )

        self.debug(
            lambda service_id=service_id,
            type=message_type: f"Registering {'fire-and-forget' if fire_and_forget else 'request-reply'} handler for {service_id} with message type {type}"
        )
        self._request_handlers[message_type] = (service_id, handler, fire_and_forget)

    async def _handle_request(self, request_id: str, request: Message) -> None:
        """Call the registered handler for a request and set the response future."""
        message_type = request.message_type

        try:
            _, handler, _ = self._request_handlers[message_type]
            response = await handler(request)

        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - request handler boundary, errors surfaced via ErrorMessage
            self.exception(f"Exception calling handler for {message_type}: {e}")
            response = ErrorMessage(
                request_id=request_id,
                error=ErrorDetails.from_exception(e),
            )

        try:
            self._response_futures[request_id].set_result(response)
        except (KeyError, asyncio.InvalidStateError) as e:
            self.exception(
                f"Exception setting response future for request {request_id}: {e}"
            )

    async def _handle_fire_and_forget(self, request: Message) -> None:
        """Call the registered handler for a fire-and-forget message. No response is sent."""
        message_type = request.message_type

        try:
            _, handler, _ = self._request_handlers[message_type]
            await handler(request)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - fire-and-forget handler boundary, no response channel
            self.exception(
                f"Exception calling fire-and-forget handler for {message_type}: {e}"
            )

    async def _send_duplicate_request_error(
        self, request_id: str, routing_envelope: tuple[bytes, ...]
    ) -> None:
        """Send an ErrorMessage back for a request_id that is already in flight."""
        error = ErrorMessage(
            request_id=request_id,
            error=ErrorDetails(
                type="DUPLICATE_REQUEST_ID",
                message=f"request_id {request_id} is already in flight",
            ),
        )
        try:
            await self.socket.send_multipart([*routing_envelope, error.to_json_bytes()])
        except asyncio.CancelledError:
            raise
        except (zmq.ZMQError, asyncio.TimeoutError) as e:
            self.exception(f"Failed to send duplicate-request error: {e}")

    async def _wait_for_response(
        self, request_id: str, routing_envelope: tuple[bytes, ...]
    ) -> None:
        """Wait for a response to a request and send it back to the caller.

        If the handler returned None, an ErrorMessage is sent back.
        """
        try:
            response = await self._response_futures[request_id]
            self._response_futures.pop(request_id, None)

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

            if not response.request_id:
                response.request_id = request_id
            await self.socket.send_multipart(
                [*routing_envelope, response.to_json_bytes()]
            )
        except asyncio.CancelledError:
            raise
        except (zmq.ZMQError, asyncio.TimeoutError, KeyError) as e:
            self.exception(
                f"Exception waiting for response for request {request_id}: {e}"
            )

    @background_task(immediate=True, interval=None)
    async def _rep_router_receiver(self) -> None:
        """Background task for receiving requests and sending responses."""
        self.debug("Router reply client background task initialized")

        while not self.stop_requested:
            try:
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
                    self.trace("Router reply client receiver task timed out")
                    await yield_to_event_loop()
                    continue

                _, _, fire_and_forget = self._request_handlers.get(
                    request.message_type, (None, None, False)
                )

                if fire_and_forget:
                    self.execute_async(self._handle_fire_and_forget(request))
                else:
                    if request.request_id in self._response_futures:
                        # Reject duplicate request_id to avoid overwriting the
                        # in-flight Future, which would cause responses to be
                        # routed to the wrong envelope.
                        self.warning(
                            lambda req_id=request.request_id: f"Duplicate request_id {req_id}, rejecting"
                        )
                        self.execute_async(
                            self._send_duplicate_request_error(
                                request.request_id, routing_envelope
                            )
                        )
                    else:
                        self._response_futures[request.request_id] = asyncio.Future()
                        self.execute_async(
                            self._handle_request(request.request_id, request)
                        )
                        self.execute_async(
                            self._wait_for_response(
                                request.request_id, routing_envelope
                            )
                        )

                self._msg_count += 1
                if (
                    self._yield_interval > 0
                    and self._msg_count % self._yield_interval == 0
                ):
                    await yield_to_event_loop()

            except asyncio.CancelledError:
                self.debug("Router reply client receiver task cancelled")
                raise
            except zmq.ContextTerminated:
                self.debug(
                    "Router reply client receiver task stopped (ZMQ context terminated)"
                )
                break
            except (zmq.ZMQError, asyncio.TimeoutError) as e:
                self.exception(f"Exception receiving request: {e}")
                await yield_to_event_loop()
