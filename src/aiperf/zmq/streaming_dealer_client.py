# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming DEALER client for bidirectional communication with ROUTER."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.utils import yield_to_event_loop
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Shared encoder (stateless, safe to reuse across instances)
_encoder = msgspec.msgpack.Encoder()


class ZMQStreamingDealerClient(BaseZMQClient):
    """
    ZMQ DEALER socket client for bidirectional streaming with ROUTER.

    Supports both pure streaming (fire-and-forget) and request-reply patterns.
    The message type is configurable via ``decode_type`` (defaults to
    ``RouterToWorkerMessage`` for backwards compatibility).

    The DEALER socket sets an identity which allows the ROUTER to send messages back
    to this specific DEALER instance.

    ASCII Diagram:
    ┌──────────────┐                    ┌──────────────┐
    │    DEALER    │◄──── Stream ──────►│    ROUTER    │
    │   (Client)   │                    │  (Service)   │
    │              │                    │              │
    └──────────────┘                    └──────────────┘

    Usage Pattern:
    - DEALER connects to ROUTER with a unique identity
    - DEALER sends messages to ROUTER
    - DEALER receives messages from ROUTER (routed by identity)
    - Supports both streaming and request-reply (via ``request()``)
    - Supports concurrent message processing
    """

    def __init__(
        self,
        address: str,
        identity: str,
        bind: bool = False,
        socket_ops: dict | None = None,
        *,
        decode_type: Any = None,
        **kwargs,
    ) -> None:
        """
        Initialize the streaming DEALER client.

        Args:
            address: The address to connect to (e.g., "tcp://localhost:5555")
            identity: Unique identity for this DEALER (used by ROUTER for routing)
            bind: Whether to bind (True) or connect (False) the socket.
                Usually False for DEALER.
            socket_ops: Additional socket options to set
            decode_type: The msgspec type (or union) to decode incoming messages.
                If None, defaults to RouterToWorkerMessage for backwards compatibility.
            **kwargs: Additional arguments passed to BaseZMQClient
        """
        super().__init__(
            zmq.SocketType.DEALER,
            address,
            bind,
            socket_ops={**(socket_ops or {}), zmq.IDENTITY: identity.encode()},
            client_id=identity,
            **kwargs,
        )
        if decode_type is None:
            from aiperf.credit.messages import RouterToWorkerMessage

            decode_type = RouterToWorkerMessage
        self._decoder = msgspec.msgpack.Decoder(decode_type)
        self.identity = identity
        self._receiver_handler: Callable[[Any], Awaitable[None]] | None = None
        self._pending_requests: dict[str, asyncio.Future[Any]] = {}
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.STREAMING_DEALER_YIELD_INTERVAL

    def register_receiver(self, handler: Callable[[Any], Awaitable[None]]) -> None:
        """
        Register handler for incoming messages from ROUTER.

        Args:
            handler: Async function that takes the decoded message.
        """
        if self._receiver_handler is not None:
            raise ValueError("Receiver handler already registered")
        self._receiver_handler = handler
        self.debug(
            lambda: f"Registered streaming DEALER receiver handler for {self.identity}"
        )

    @on_stop
    async def _clear_receiver(self) -> None:
        """Clear receiver handler and pending requests on stop."""
        self._receiver_handler = None
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    async def send(self, struct: Struct) -> None:
        """Send struct to ROUTER (fire-and-forget)."""
        await self._check_initialized()

        try:
            await self.socket.send(_encoder.encode(struct))
            if self.is_trace_enabled:
                self.trace(f"Sent struct: {struct}")
        except Exception as e:
            self.exception(f"Failed to send message: {e}")
            raise

    async def request(self, struct: Struct, timeout: float) -> Any:
        """Send a request and wait for a response matched by ``rid`` or ``cid``.

        The struct must have a ``rid`` or ``cid`` attribute. The response from
        the ROUTER is matched by the same attribute and returned. Other messages
        are dispatched normally to the receiver handler.

        Args:
            struct: The request struct to send (must have ``rid`` or ``cid``).
            timeout: Maximum time to wait for a response in seconds.

        Returns:
            The decoded response struct.

        Raises:
            asyncio.TimeoutError: If no response is received within timeout.
        """
        key = getattr(struct, "rid", None) or getattr(struct, "cid", None)
        if key is None:
            raise ValueError("request() requires a struct with 'rid' or 'cid'")

        future: asyncio.Future[Any] = asyncio.Future()
        self._pending_requests[key] = future

        try:
            await self.send(struct)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise
        finally:
            self._pending_requests.pop(key, None)

    async def _process_received_message(self, message: Any) -> None:
        """Dispatch a decoded message to a pending request future or the handler."""
        # Check if this is a response to a pending request (by rid or cid)
        key = getattr(message, "rid", None) or getattr(message, "cid", None)
        if key and key in self._pending_requests:
            future = self._pending_requests.pop(key)
            if not future.done():
                future.set_result(message)
            return

        if self._receiver_handler:
            self.execute_async(self._receiver_handler(message))
            self._msg_count += 1
            if self._yield_interval > 0 and self._msg_count % self._yield_interval == 0:
                await yield_to_event_loop()
        else:
            self.warning(f"Received {type(message).__name__} but no handler registered")

    @background_task(immediate=True, interval=None)
    async def _streaming_dealer_receiver(self) -> None:
        """Background task for receiving messages from ROUTER."""
        self.debug(
            lambda: f"Streaming DEALER receiver task started for {self.identity}"
        )

        while not self.stop_requested:
            try:
                message_bytes = await self.socket.recv()
                if self.is_trace_enabled:
                    self.trace(f"Received message: {message_bytes}")
                message = self._decoder.decode(message_bytes)
                await self._process_received_message(message)
            except zmq.Again:
                self.trace("No data on dealer socket received, yielding to event loop")
                await yield_to_event_loop()
            except asyncio.CancelledError:
                self.debug("Streaming DEALER receiver task cancelled")
                raise
            except zmq.ContextTerminated:
                self.debug(
                    "Streaming DEALER receiver task stopped (ZMQ context terminated)"
                )
                break
            except (zmq.ZMQError, asyncio.TimeoutError, msgspec.DecodeError) as e:
                self.exception(
                    f"Exception receiving messages for client {self.client_id}: {e!r}"
                )
                await yield_to_event_loop()

        self.debug(
            lambda: f"Streaming DEALER receiver task stopped for {self.identity}"
        )
