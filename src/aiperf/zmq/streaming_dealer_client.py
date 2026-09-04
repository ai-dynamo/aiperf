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
from aiperf.common.models.base_models import msgspec_enc_hook
from aiperf.credit.messages import RouterToWorkerMessage
from aiperf.zmq.fd_reader import FdEdgeReader
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Pre-created encoder/decoder for performance (caches schema).
# See streaming_router_client for why enc_hook is wired in.
_encoder = msgspec.msgpack.Encoder(enc_hook=msgspec_enc_hook)
_decoder = msgspec.msgpack.Decoder(RouterToWorkerMessage)

RouterToWorkerHandler: TypeAlias = Callable[[RouterToWorkerMessage], Awaitable[None]]


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
    - Streaming by default; `request()` opts into `rid`/`cid`-correlated request-reply
    - Supports concurrent message processing

    Example:
    ```python
        from aiperf.credit.messages import (
            Credit, CancelCredits, WorkerDispatchable, WorkerShutdown, CreditReturn
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
                    await dealer.send(CreditReturn(credit=credit))
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
            decode_type: msgspec type (or tagged union) used to decode incoming
                messages. Defaults to ``RouterToWorkerMessage`` -- the credit
                channel. Mirrors the same parameter on the ROUTER client, whose
                peers (e.g. the pod-lifecycle channel) speak a different union.
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
        self.identity = identity
        self._receiver_handler: RouterToWorkerHandler | None = None
        self._decoder = (
            _decoder if decode_type is None else msgspec.msgpack.Decoder(decode_type)
        )

        # Futures for in-flight request() calls, keyed by the request's rid/cid.
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
        """Clear receiver handler and pending requests on stop.

        Cancelling the pending futures is load-bearing: the socket is going
        away, so a reply can never arrive: an awaiter left pending here blocks
        until the process is killed. The ROUTER counterpart cancels the same
        way.
        """
        if self._fd_reader is not None:
            await self._fd_reader.stop()
            self._fd_reader = None
        self._receiver_handler = None
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    def _recv_one_dealer(self) -> RouterToWorkerMessage:
        """Synchronous NOBLOCK multipart recv + decode for the FD-reader drain.

        The payload is frame 1: a DEALER socket strips the routing identity
        before delivery, so unlike the ROUTER counterpart there is no envelope
        frame to skip. Any further frames are drained and discarded rather than
        left queued -- a ZMQ message is atomic, so leaving a tail behind would
        put every subsequent recv permanently off by one with no error
        anywhere. The ROUTER sends exactly ``(identity, payload)`` today, so
        the drain loop never runs; it exists so adding a frame degrades to
        "ignored" instead of "decodes the wrong frame", and warns rather than
        discarding a protocol mismatch in silence.
        """
        payload = zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK)
        while self.socket.getsockopt(zmq.RCVMORE):
            zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK)
            self.warning("Discarded unexpected trailing frame from ROUTER")
        return self._decoder.decode(payload)

    def _dispatch_dealer(self, message: RouterToWorkerMessage) -> None:
        # Request-reply: a reply carrying the rid/cid of a pending request
        # resolves that future instead of going to the streaming handler. The
        # match is a synchronous dict lookup so it stays inside the FD drain.
        # Both candidate keys are checked independently (not `rid or cid`) so
        # a reply with a non-matching rid but a matching cid still resolves.
        if self._pending_requests:
            rid = getattr(message, "rid", None)
            cid = getattr(message, "cid", None)
            for key in (rid, cid):
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

    async def _send_direct_with_retry(self, data: bytes) -> None:
        """NOBLOCK send with backoff, for the window before the FD driver exists.

        The backoff sleep can straddle the receiver task creating the FD
        driver, so the handoff is re-checked on every attempt: once the driver
        owns the socket, a direct send here would race it on the shared FD
        edge-trigger.
        """
        max_retries = Environment.ZMQ.PUSH_MAX_RETRIES
        for attempt in range(max_retries + 1):
            if self._fd_reader is not None:
                self._fd_reader.send(data)
                return
            try:
                self._send_one_dealer(data)
                return
            except zmq.Again:
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(Environment.ZMQ.PUSH_RETRY_DELAY)

    async def send(self, struct: Struct) -> None:
        """Send struct to ROUTER."""
        await self._check_initialized()

        # copy=False avoids memcpy'ing the encoded frame into libzmq on the event
        # loop thread; the encoded buffer is freshly produced here and never reused.
        data = _encoder.encode(struct)
        # FD-driver owns both directions of the socket; never touch zmq.asyncio
        # send here or it corrupts the shared FD edge-trigger. Before the
        # receiver task has created the driver (early WorkerDispatchable), send
        # directly. SNDHWM=0 means the NOBLOCK send never blocks on the queue and
        # libzmq queues rather than rejects a send before the ROUTER connection
        # has handshaked, but retry the way the PUSH clients do anyway to guard
        # against a genuine zmq.Again rather than dropping the first struct.
        if self._fd_reader is not None:
            self._fd_reader.send(data)
        else:
            await self._send_direct_with_retry(data)
        if self.is_trace_enabled:
            self.trace(f"Sent struct: {struct}")

    async def request(self, struct: Struct, timeout: float) -> Any:
        """Send a request and await the response matched by ``rid``/``cid``.

        The struct must carry an ``rid`` or ``cid`` attribute; the ROUTER's
        reply with the same key resolves the returned future (see
        ``_dispatch_dealer``). Other messages dispatch normally to the receiver
        handler.

        Args:
            struct: The request struct to send (must have ``rid`` or ``cid``).
            timeout: Maximum time to wait for a response, in seconds.

        Returns:
            The decoded response struct.

        Raises:
            ValueError: If the struct carries neither a truthy ``rid`` nor ``cid``.
            TimeoutError: If no response is received within ``timeout``.
        """
        key = getattr(struct, "rid", None) or getattr(struct, "cid", None)
        # ``not key`` (not ``is None``) so this agrees with the resolution guard
        # in _dispatch_dealer: an empty-string rid/cid would otherwise register
        # a future that side can never match, hanging until timeout.
        if not key:
            raise ValueError(
                "request() requires a struct with a 'rid' or 'cid' to "
                f"correlate the reply on; {type(struct).__name__} has neither."
            )

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
