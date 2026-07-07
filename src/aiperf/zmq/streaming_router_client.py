# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming ROUTER client for bidirectional communication with DEALER clients."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.hooks import background_task, on_stop
from aiperf.zmq.fd_reader import FdEdgeReader
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Shared encoder (stateless, safe to reuse across instances)
_encoder = msgspec.msgpack.Encoder()


class ZMQStreamingRouterClient(BaseZMQClient):
    """
    ZMQ ROUTER socket client for bidirectional streaming with DEALER clients.

    Supports both pure streaming (fire-and-forget) and request-reply patterns.
    The message type is configurable via ``decode_type`` (defaults to
    ``WorkerToRouterMessage`` for backwards compatibility).

    Features:
    - Bidirectional streaming with automatic routing by peer identity
    - Configurable message deserialization via msgspec tagged unions
    - Optional request-reply: if the handler returns a Struct, it is sent back
    - Works with both TCP and IPC transports

    ASCII Diagram:
    ┌──────────────┐                    ┌──────────────┐
    │    DEALER    │◄──── Stream ──────►│              │
    │   (Client)   │                    │              │
    └──────────────┘                    │              │
    ┌──────────────┐                    │    ROUTER    │
    │    DEALER    │◄──── Stream ──────►│  (Service)   │
    │   (Client)   │                    │              │
    └──────────────┘                    │              │
    ┌──────────────┐                    │              │
    │    DEALER    │◄──── Stream ──────►│              │
    │   (Client)   │                    │              │
    └──────────────┘                    └──────────────┘

    Usage Pattern:
    - ROUTER sends messages to specific DEALER clients by identity
    - ROUTER receives messages from DEALER clients (identity included in envelope)
    - Supports both fire-and-forget and request-reply via handler return value
    - Supports concurrent message processing
    """

    # Peer-gone errors: the target DEALER disconnected but the ROUTER socket
    # is still valid for all other peers.  No recreation needed.
    _PEER_GONE_ERRNOS = frozenset({zmq.EHOSTUNREACH, zmq.ENOTCONN})

    # Socket-broken errors: a partial multipart send left the socket FSM in a
    # bad state; the only recovery is to recreate the socket.
    _SOCKET_BROKEN_ERRNOS = frozenset({zmq.EFSM})

    def __init__(
        self,
        address: str,
        bind: bool = True,
        socket_ops: dict | None = None,
        *,
        additional_bind_address: str | None = None,
        decode_type: Any = None,
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
            decode_type: The msgspec type (or union) to decode incoming messages.
                If None, defaults to WorkerToRouterMessage for backwards compatibility.
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
        if decode_type is None:
            from aiperf.credit.messages import WorkerToRouterMessage

            decode_type = WorkerToRouterMessage
        self._decoder = msgspec.msgpack.Decoder(decode_type)
        self._receiver_handler: (
            Callable[[str, Any], Awaitable[Struct | None]] | None
        ) = None
        self._pending_requests: dict[str, asyncio.Future[Any]] = {}
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.STREAMING_ROUTER_YIELD_INTERVAL
        self._fd_reader: FdEdgeReader | None = None

    def register_receiver(
        self, handler: Callable[[str, Any], Awaitable[Struct | None]]
    ) -> None:
        """
        Register handler for incoming messages from DEALER clients.

        The handler receives (identity, message) and may optionally return a Struct.
        If a Struct is returned, it is encoded and sent back to the originating DEALER
        (request-reply pattern). If None is returned, no response is sent (streaming).

        Args:
            handler: Async function ``(identity: str, message) -> Struct | None``
        """
        if self._receiver_handler is not None:
            raise ValueError("Receiver handler already registered")
        self._receiver_handler = handler
        self.debug("Registered streaming ROUTER receiver handler")

    @on_stop
    async def _clear_receiver(self) -> None:
        """Clear receiver handler, pending requests, and callbacks on stop."""
        if self._fd_reader is not None:
            self._fd_reader.stop()
            self._fd_reader = None
        self._receiver_handler = None
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

    def _recv_one_router(self) -> tuple[str, Any]:
        """Synchronous NOBLOCK multipart recv + decode for the FD-reader drain.

        ROUTER envelope: [identity, ..., message_bytes]. Assembled manually via the
        direct base-class ``recv`` because ``recv_multipart`` delegates to
        ``self.recv`` -- the async override on a ``zmq.asyncio`` socket. The first
        frame raises ``zmq.Again`` when drained; subsequent RCVMORE frames are
        atomic and always immediately available.
        """
        identity = zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK)
        payload = identity
        while self.socket.getsockopt(zmq.RCVMORE):
            payload = zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK)
        return identity.decode("utf-8", "surrogateescape"), self._decoder.decode(
            payload
        )

    def _dispatch_router(self, item: tuple[str, Any]) -> None:
        """Route one drained (identity, message): resolve a pending request by
        ``cid`` synchronously, else hand off to the handler (request-reply aware)."""
        identity, message = item
        if self._try_resolve_pending_request(message):
            return
        if self._receiver_handler is None:
            self.warning(f"Received {type(message).__name__} but no handler registered")
            return
        self.execute_async(self._dispatch_message(identity, message))

    def _send_one_router(self, frames: tuple[bytes, bytes]) -> None:
        """Synchronous NOBLOCK multipart send for the FD-driver.

        Framed manually (identity SNDMORE + payload) because ``send_multipart``
        delegates to the async ``self.send``. With SNDHWM=0 neither frame blocks,
        so the two-frame message stays atomic.

        GUARDRAIL: this socket must keep ``SNDHWM=0``. ``FdEdgeReader.send`` buffers
        and retries the whole ``(identity, payload)`` tuple as one unit, so a finite
        SNDHWM that split the send (frame 1 sent, frame 2 -> ``zmq.Again``) would
        re-emit the identity frame on retry and desync the ROUTER framing.
        """
        identity, payload = frames
        zmq.Socket.send(
            self.socket, identity, flags=zmq.NOBLOCK | zmq.SNDMORE, copy=False
        )
        zmq.Socket.send(self.socket, payload, flags=zmq.NOBLOCK, copy=False)

    async def send_to(self, identity: str, struct: Struct) -> None:
        """
        Send struct to specific DEALER client by identity.

        Skips the async _check_initialized guard, checking socket state inline
        instead to avoid an unnecessary coroutine switch.

        Args:
            identity: The DEALER client's identity (routing key)
            struct: The msgspec Struct to send

        Raises:
            NotInitializedError: If socket not initialized
        """
        if not self.socket:
            raise NotInitializedError("Socket not initialized or closed")
        if self.stop_requested:
            raise asyncio.CancelledError("Socket was stopped")

        # copy=False avoids memcpy'ing the frames into libzmq on the event loop
        # thread; both frames are freshly produced here and never reused.
        frames = (identity.encode("utf-8", "surrogateescape"), _encoder.encode(struct))
        # FD-driver owns both directions; never touch zmq.asyncio send here. Before
        # the receiver task creates the driver, send directly (SNDHWM=0, no block).
        if self._fd_reader is not None:
            self._fd_reader.send(frames)
        else:
            self._send_one_router(frames)
        if self.is_trace_enabled:
            self.trace(f"Sent {type(struct).__name__} to {identity}: {struct}")

    async def request_to(self, identity: str, struct: Struct, timeout: float) -> Any:
        """Send a request to a specific DEALER and wait for a response matched by ``cid``.

        Args:
            identity: The DEALER client's identity (routing key)
            struct: The request struct (must have ``cid`` attribute)
            timeout: Maximum seconds to wait for a response

        Returns:
            The decoded response struct.

        Raises:
            TimeoutError: If no response within timeout.
        """
        cid = getattr(struct, "cid", None)
        if cid is None:
            raise ValueError("request_to() requires a struct with 'cid'")

        future: asyncio.Future[Any] = asyncio.Future()
        self._pending_requests[cid] = future

        try:
            await self.send_to(identity, struct)
            return await asyncio.wait_for(future, timeout=timeout)
        except TimeoutError:
            raise
        finally:
            self._pending_requests.pop(cid, None)

    async def _dispatch_message(self, identity: str, message: Any) -> None:
        """Dispatch a received message to the handler.

        If the handler returns a Struct, encode and send it back to the
        originating DEALER (request-reply). Otherwise treat as fire-and-forget.
        """
        try:
            response = await self._receiver_handler(identity, message)  # type: ignore[misc]
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - receiver handler boundary, must not crash ROUTER loop
            self.exception(
                f"Exception in handler for {type(message).__name__} from {identity}: {e!r}"
            )
            return

        if response is not None:
            try:
                await self.send_to(identity, response)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - send boundary, must not crash ROUTER dispatcher
                self.exception(f"Failed to send response to {identity}: {e!r}")
                await self._recover_from_send_failure(identity, e)

    async def _recover_from_send_failure(self, identity: str, error: Exception) -> None:
        """Handle a ROUTER send failure.

        Peer-gone errors (EHOSTUNREACH, ENOTCONN) are expected when a DEALER
        disconnects between receive and reply -- the socket is still valid for
        other peers, so we just log and continue.

        Socket-broken errors (EFSM) mean a partial multipart send corrupted
        the socket state machine; the only fix is to recreate the socket.
        """
        if self.stop_requested or not isinstance(error, zmq.ZMQError):
            return

        if error.errno in self._PEER_GONE_ERRNOS:
            self.warning(
                f"Peer {identity} unreachable (errno={error.errno}), "
                "dropping response; ROUTER socket remains valid"
            )
            return

        if error.errno not in self._SOCKET_BROKEN_ERRNOS:
            return

        self.warning(
            "Recreating streaming ROUTER socket after broken state from send "
            f"failure to {identity}: errno={error.errno}"
        )
        try:
            await self._recreate_socket()
        except asyncio.CancelledError:
            raise
        except (TimeoutError, zmq.ZMQError) as recreate_error:
            if not self.stop_requested:
                self.exception(
                    "Failed to recreate streaming ROUTER socket after send "
                    f"failure to {identity}: {recreate_error!r}"
                )
            return

        # _recreate_socket() swapped self.socket for a fresh object with a new FD;
        # the FD reader is still registered against the old (now closed) FD and
        # pumping the old socket. Rebuild it against the new socket so the ROUTER
        # keeps receiving and we don't leak an add_reader on the recycled FD.
        if self._fd_reader is not None and not self.stop_requested:
            self._fd_reader.stop()
            self._start_fd_reader()

    def _try_resolve_pending_request(self, message: Any) -> bool:
        """Resolve the pending-request future if ``message.cid`` matches a pending one.

        Returns True if the message was consumed as a response, False otherwise.
        """
        cid = getattr(message, "cid", None)
        if not cid or cid not in self._pending_requests:
            return False
        future = self._pending_requests.pop(cid)
        if not future.done():
            future.set_result(message)
        return True

    @background_task(immediate=True, interval=None)
    async def _streaming_router_receiver(self) -> None:
        """Background task for receiving messages from DEALER clients.

        Drives the ROUTER off its raw FD: edge-triggered NOBLOCK multipart drain
        on recv, sync NOBLOCK on send (the driver owns both directions).
        """
        self.debug("Streaming ROUTER receiver task started")
        self._start_fd_reader()

    def _start_fd_reader(self) -> None:
        """Build and start the FD reader against the current ``self.socket``.

        Called once by the receiver task at startup, and again after
        ``_recreate_socket`` swaps in a new socket so the reader binds to the new
        FD. The recv/send callables read ``self.socket`` dynamically, so only the
        reader's captured socket and registered FD need re-pointing.
        """
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
