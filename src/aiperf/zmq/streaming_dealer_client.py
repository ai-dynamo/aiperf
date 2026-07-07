# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming DEALER client for bidirectional communication with ROUTER."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.zmq.fd_reader import FdEdgeReader
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Shared encoder (stateless, safe to reuse across instances). The decoder is
# per-instance because ``decode_type`` is configurable (see __init__).
_encoder = msgspec.msgpack.Encoder()

RouterToWorkerHandler: TypeAlias = Callable[[Any], Awaitable[None]]


class ZMQStreamingDealerClient(BaseZMQClient):
    """
    ZMQ DEALER socket client for bidirectional streaming with ROUTER.

    Unlike ZMQDealerRequestClient (request-response pattern), this client is
    designed for streaming scenarios where messages flow bidirectionally without
    request-response pairing.

    The DEALER socket sets an identity which allows the ROUTER to send messages back
    to this specific DEALER instance.

    ASCII Diagram:
    ┌──────────────┐                    ┌──────────────┐
    │    DEALER    │◄──── Stream ──────►│    ROUTER    │
    │   (Worker)   │                    │  (Manager)   │
    │              │                    │              │
    └──────────────┘                    └──────────────┘

    Usage Pattern:
    - DEALER connects to ROUTER with a unique identity
    - DEALER sends messages to ROUTER
    - DEALER receives messages from ROUTER (routed by identity)
    - No request-response pairing - pure streaming
    - Supports concurrent message processing

    Example:
    ```python
        from aiperf.credit.structs import Credit
        from aiperf.credit.messages import (
            CancelCredits, WorkerDispatchable, WorkerShutdown, CreditReturn
        )

        # Create via comms (recommended - handles lifecycle management)
        dealer = comms.create_streaming_dealer_client(
            address=CommAddress.CREDIT_ROUTER,
            identity="worker-1",
        )

        async def handle_message(message: Credit | CancelCredits) -> None:
            match message:
                case Credit() as credit:
                    do_some_work(credit)
                    await dealer.send(CreditReturn(credit_id=credit.id))
                case CancelCredits(credit_ids=ids):
                    cancel_credits(ids)

        dealer.register_receiver(handle_message)

        # Lifecycle managed by comms
        await comms.initialize()
        await comms.start()
        await dealer.send(WorkerDispatchable(worker_id="worker-1"))
        ...
        await dealer.send(WorkerShutdown(worker_id="worker-1"))
        await comms.stop()
    ```
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
        self._fd_reader: FdEdgeReader | None = None

    def register_receiver(self, handler: RouterToWorkerHandler) -> None:
        """
        Register handler for incoming messages from ROUTER.

        The handler will be called for each message received (Credit or CancelCredits).

        Args:
            handler: Async function that takes a RouterToWorkerMessage (Credit | CancelCredits)
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
        if self._fd_reader is not None:
            self._fd_reader.stop()
            self._fd_reader = None
        self._receiver_handler = None
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    def _recv_one_dealer(self) -> Any:
        """Synchronous NOBLOCK recv + decode for the FD-reader drain."""
        return self._decoder.decode(zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK))

    def _dispatch_dealer(self, message: Any) -> None:
        # Request-reply: a reply carrying the rid/cid of a pending request
        # resolves that future instead of going to the streaming handler. The
        # match is a synchronous dict lookup so it stays inside the FD drain.
        key = getattr(message, "rid", None) or getattr(message, "cid", None)
        if key and key in self._pending_requests:
            future = self._pending_requests.pop(key)
            if not future.done():
                future.set_result(message)
            return
        if self._receiver_handler is not None:
            self.execute_async(self._receiver_handler(message))
        else:
            self.warning(f"Received {type(message).__name__} but no handler registered")

    def _send_one_dealer(self, data: bytes) -> None:
        """Synchronous NOBLOCK single-frame send for the FD-driver."""
        zmq.Socket.send(self.socket, data, flags=zmq.NOBLOCK, copy=False)

    async def send(self, struct: Struct) -> None:
        """Send struct to ROUTER."""
        await self._check_initialized()

        # copy=False avoids memcpy'ing the encoded frame into libzmq on the event
        # loop thread; the encoded buffer is freshly produced here and never reused.
        data = _encoder.encode(struct)
        # FD-driver owns both directions of the socket; never touch zmq.asyncio
        # send here or it corrupts the shared FD edge-trigger. Before the
        # receiver task has created the driver (early WorkerDispatchable), send
        # directly — SNDHWM=0 means the NOBLOCK send will not block.
        if self._fd_reader is not None:
            self._fd_reader.send(data)
        else:
            self._send_one_dealer(data)
        if self.is_trace_enabled:
            self.trace(f"Sent struct: {struct}")

    async def request(self, struct: Struct, timeout: float) -> Any:
        """Send a request and await the response matched by ``rid``/``cid``.

        The struct must carry an ``rid`` or ``cid`` attribute; the ROUTER's reply
        with the same key resolves the returned future (see ``_dispatch_dealer``).
        Other messages dispatch normally to the receiver handler.

        Args:
            struct: The request struct to send (must have ``rid`` or ``cid``).
            timeout: Maximum time to wait for a response, in seconds.

        Returns:
            The decoded response struct.

        Raises:
            TimeoutError: If no response is received within ``timeout``.
        """
        key = getattr(struct, "rid", None) or getattr(struct, "cid", None)
        if key is None:
            raise ValueError("request() requires a struct with 'rid' or 'cid'")

        future: asyncio.Future[Any] = asyncio.Future()
        self._pending_requests[key] = future
        try:
            await self.send(struct)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_requests.pop(key, None)

    @background_task(immediate=True, interval=None)
    async def _streaming_dealer_receiver(self) -> None:
        """
        Background task for receiving messages from ROUTER.

        Runs continuously until stop is requested. Decodes messages as
        RouterToWorkerMessage (Credit | CancelCredits) using msgpack.
        """
        self.debug(
            lambda: f"Streaming DEALER receiver task started for {self.identity}"
        )

        # Always drive the DEALER off its raw FD: edge-triggered NOBLOCK drain on
        # recv, sync NOBLOCK on send (the driver owns both directions).
        self._fd_reader = FdEdgeReader(
            socket=self.socket,
            recv_one=self._recv_one_dealer,
            dispatch=self._dispatch_dealer,
            batch_limit=self._yield_interval,
            send_one=self._send_one_dealer,
            on_error=lambda e: self.exception(
                f"Exception draining dealer socket for {self.client_id}: {e!r}"
            ),
        )
        self._fd_reader.start()
