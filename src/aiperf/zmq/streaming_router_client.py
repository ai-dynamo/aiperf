# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming ROUTER client for bidirectional communication with DEALER clients."""

from collections.abc import Awaitable, Callable
from typing import TypeAlias

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.credit.messages import WorkerToRouterMessage
from aiperf.zmq.fd_reader import FdEdgeReader
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Pre-created encoder/decoder for performance (caches schema)
_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(WorkerToRouterMessage)

WorkerToRouterHandler: TypeAlias = Callable[
    [str, WorkerToRouterMessage], Awaitable[None]
]


class ZMQStreamingRouterClient(BaseZMQClient):
    """
    ZMQ ROUTER socket client for bidirectional streaming with DEALER clients.

    Unlike ZMQRouterReplyClient (request-response pattern), this client is
    designed for streaming scenarios where messages flow bidirectionally without
    request-response pairing.

    Features:
    - Bidirectional streaming with automatic routing by peer identity
    - Message-based peer lifecycle tracking (ready/shutdown messages)
    - Works with both TCP and IPC transports

    ASCII Diagram:
    ┌──────────────┐                    ┌──────────────┐
    │    DEALER    │◄──── Stream ──────►│              │
    │   (Worker)   │                    │              │
    └──────────────┘                    │              │
    ┌──────────────┐                    │    ROUTER    │
    │    DEALER    │◄──── Stream ──────►│  (Manager)   │
    │   (Worker)   │                    │              │
    └──────────────┘                    │              │
    ┌──────────────┐                    │              │
    │    DEALER    │◄──── Stream ──────►│              │
    │   (Worker)   │                    │              │
    └──────────────┘                    └──────────────┘

    Usage Pattern:
    - ROUTER sends messages to specific DEALER clients by identity
    - ROUTER receives messages from DEALER clients (identity included in envelope)
    - No request-response pairing - pure streaming
    - Supports concurrent message processing
    - Automatic peer tracking via worker ready and shutdown messages

    Example:
    ```python
        from aiperf.common.structs import (
            Credit, WorkerReady, WorkerShutdown, CreditReturn
        )

        # Create via comms (recommended - handles lifecycle management)
        router = comms.create_streaming_router_client(
            address=CommAddress.CREDIT_ROUTER,
            bind=True,
        )

        async def handle_message(identity: str, message: WorkerToRouterMessage) -> None:
            match message:
                case WorkerReady():
                    await register_worker(identity)
                case WorkerShutdown():
                    await unregister_worker(identity)
                case CreditReturn(credit_id=id, cancelled=c, error=e):
                    await handle_credit_return(identity, id, c, e)

        router.register_receiver(handle_message)

        # Lifecycle managed by comms
        await comms.initialize()
        await comms.start()

        # Send Credit directly to specific worker
        await router.send_to("worker-1", credit)
        ...
        await comms.stop()
    ```
    """

    def __init__(
        self,
        address: str,
        bind: bool = True,
        socket_ops: dict | None = None,
        additional_bind_address: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the streaming ROUTER client.

        Args:
            address: The address to bind or connect to (e.g., "tcp://*:5555" or "ipc:///tmp/socket")
            bind: Whether to bind (True) or connect (False) the socket
            socket_ops: Additional socket options to set
            additional_bind_address: Optional second address to bind to for dual-bind mode
                (e.g., IPC + TCP in Kubernetes). Only used when bind=True.
            **kwargs: Additional arguments passed to BaseZMQClient
        """
        super().__init__(
            zmq.SocketType.ROUTER,
            address,
            bind,
            socket_ops,
            additional_bind_address=additional_bind_address,
            **kwargs,
        )
        self._receiver_handler: WorkerToRouterHandler | None = None
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.STREAMING_ROUTER_YIELD_INTERVAL
        self._fd_reader: FdEdgeReader | None = None

    def register_receiver(self, handler: WorkerToRouterHandler) -> None:
        """
        Register handler for incoming messages from DEALER clients.

        The handler will be called for each message received, with the DEALER's
        identity and the decoded message (WorkerReady | WorkerShutdown | CreditReturn).

        Args:
            handler: Async function that takes (identity: str, message: WorkerToRouterMessage)
        """
        if self._receiver_handler is not None:
            raise ValueError("Receiver handler already registered")
        self._receiver_handler = handler
        self.debug("Registered streaming ROUTER receiver handler")

    @on_stop
    async def _clear_receiver(self) -> None:
        """Clear receiver handler and callbacks on stop."""
        if self._fd_reader is not None:
            self._fd_reader.stop()
            self._fd_reader = None
        self._receiver_handler = None

    def _recv_one_router(self) -> tuple[str, WorkerToRouterMessage]:
        """Synchronous NOBLOCK multipart recv + decode for the FD-reader drain.

        ROUTER envelope: [identity, ..., message_bytes]. Assembled manually via the
        direct base-class ``recv`` because ``recv_multipart`` delegates to
        ``self.recv`` — which on a ``zmq.asyncio`` socket is the async override that
        returns a Future. The first frame raises ``zmq.Again`` when drained;
        subsequent frames (RCVMORE) are atomic and always immediately available.
        """
        identity = zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK)
        payload = identity
        while self.socket.getsockopt(zmq.RCVMORE):
            payload = zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK)
        return identity.decode("utf-8"), _decoder.decode(payload)

    def _dispatch_router(self, item: tuple[str, WorkerToRouterMessage]) -> None:
        identity, message = item
        if self._receiver_handler is not None:
            self.execute_async(self._receiver_handler(identity, message))
        else:
            self.warning(f"Received {type(message).__name__} but no handler registered")

    def _send_one_router(self, frames: tuple[bytes, bytes]) -> None:
        """Synchronous NOBLOCK multipart send for the FD-driver.

        Framed manually (identity SNDMORE + payload) because ``send_multipart``
        delegates to ``self.send`` -> the async override. With SNDHWM=0 neither
        frame blocks, so the two-frame message stays atomic.

        GUARDRAIL: this socket must keep ``SNDHWM=0``. ``FdEdgeReader.send`` buffers
        and retries the whole ``(identity, payload)`` tuple as one unit, so if a
        finite SNDHWM ever split the send (frame 1 sent, frame 2 -> ``zmq.Again``)
        the retry would re-emit the identity frame and desync the ROUTER framing.
        A finite SNDHWM here would first require making the send buffer per-frame
        (track partial-multipart state). The single-frame DEALER/PUSH paths have no
        such constraint.
        """
        identity, payload = frames
        zmq.Socket.send(
            self.socket, identity, flags=zmq.NOBLOCK | zmq.SNDMORE, copy=False
        )
        zmq.Socket.send(self.socket, payload, flags=zmq.NOBLOCK, copy=False)

    async def send_to(self, identity: str, struct: Struct) -> None:
        """
        Send struct to specific DEALER client by identity.

        Args:
            identity: The DEALER client's identity (routing key)
            struct: The msgspec Struct to send (Credit or CancelCredits)

        Raises:
            NotInitializedError: If socket not initialized
            CommunicationError: If send fails
        """
        await self._check_initialized()

        # copy=False avoids memcpy'ing the frames into libzmq on the event loop
        # thread; both frames are freshly produced here and never reused.
        frames = (identity.encode(), _encoder.encode(struct))
        # FD-driver owns both directions; never touch zmq.asyncio send here.
        if self._fd_reader is not None:
            self._fd_reader.send(frames)
        else:
            self._send_one_router(frames)
        if self.is_trace_enabled:
            self.trace(f"Sent {type(struct).__name__} to {identity}: {struct}")

    @background_task(immediate=True, interval=None)
    async def _streaming_router_receiver(self) -> None:
        """
        Background task for receiving messages from DEALER clients.

        Runs continuously until stop is requested. Decodes messages as
        WorkerToRouterMessage (WorkerReady | WorkerShutdown | CreditReturn) using msgpack.
        """
        self.debug("Streaming ROUTER receiver task started")

        # Always drive the ROUTER off its raw FD: edge-triggered NOBLOCK multipart
        # drain on recv, sync NOBLOCK on send (the driver owns both directions).
        self._fd_reader = FdEdgeReader(
            socket=self.socket,
            recv_one=self._recv_one_router,
            dispatch=self._dispatch_router,
            batch_limit=self._yield_interval,
            send_one=self._send_one_router,
            on_error=lambda e: self.exception(
                f"Exception draining router socket for {self.client_id}: {e!r}"
            ),
        )
        self._fd_reader.start()
