# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming ROUTER client for bidirectional communication with DEALER clients."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.credit.messages import WorkerToRouterMessage
from aiperf.zmq.fd_reader import FdEdgeReader
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Shared encoder (stateless, safe to reuse across instances)
_encoder = msgspec.msgpack.Encoder()

WorkerToRouterHandler: TypeAlias = Callable[[str, Any], Awaitable[Struct | None]]
"""Receiver signature: ``(identity, message) -> Struct | None``.

Returning ``None`` is fire-and-forget streaming; returning a ``Struct`` makes the
exchange request-reply -- the ROUTER sends it straight back to the originating
DEALER. The message type is whatever ``decode_type`` selected, hence ``Any``."""


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
            Credit, WorkerDispatchable, WorkerShutdown, CreditReturn
        )

        # Create via comms (recommended - handles lifecycle management)
        router = comms.create_streaming_router_client(
            address=CommAddress.CREDIT_ROUTER,
            bind=True,
        )

        async def handle_message(identity: str, message: WorkerToRouterMessage) -> None:
            match message:
                case WorkerDispatchable():
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
            decode_type: msgspec type (or tagged union) used to decode incoming
                messages. Defaults to ``WorkerToRouterMessage`` -- the credit
                plane's union -- so existing call sites are unaffected. The
                worker-pod lifecycle channel passes ``PeerToGroupManagerMessage``.
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
        self._decoder = msgspec.msgpack.Decoder(
            WorkerToRouterMessage if decode_type is None else decode_type
        )
        self._receiver_handler: WorkerToRouterHandler | None = None
        self._pending_requests: dict[str, asyncio.Future[Any]] = {}
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.STREAMING_ROUTER_YIELD_INTERVAL
        self._fd_reader: FdEdgeReader | None = None

    def register_receiver(self, handler: WorkerToRouterHandler) -> None:
        """
        Register handler for incoming messages from DEALER clients.

        The handler will be called for each message received, with the DEALER's
        identity and the decoded message (WorkerDispatchable | WorkerShutdown | CreditReturn).

        Args:
            handler: Async function that takes (identity: str, message: WorkerToRouterMessage)
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
        # Cancel rather than leave awaiters hanging on a socket that is going away.
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

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
        return identity.decode("utf-8"), self._decoder.decode(payload)

    def _dispatch_router(self, item: tuple[str, Any]) -> None:
        identity, message = item
        # Responses to an in-flight ``request_to`` are resolved here, before the
        # handler sees them: a reply belongs to its awaiting caller, not to the
        # general receiver.
        if self._try_resolve_pending_request(message):
            return
        if self._receiver_handler is None:
            self.warning(f"Received {type(message).__name__} but no handler registered")
            return
        self.execute_async(self._dispatch_message(identity, message))

    async def _dispatch_message(self, identity: str, message: Any) -> None:
        """Run the receiver handler and reply when it returns a Struct.

        BEHAVIOR NOTE: both the handler call and the reply send contain their
        exceptions rather than letting them propagate. This dispatch runs inside
        the FD-reader drain loop that serves *every* peer on this ROUTER, so one
        misbehaving handler must not take the socket down for the others. The
        exception is logged with the message type and originating identity.
        Callers that need failures to surface must handle them in the handler.
        """
        try:
            response = await self._receiver_handler(identity, message)  # type: ignore[misc]
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - receiver handler boundary, must not crash ROUTER loop
            # CONTAINED ON PURPOSE: this coroutine is fired from the FD-reader
            # drain loop that multiplexes every peer on this ROUTER. Letting a
            # single handler's exception escape would surface as an unhandled
            # task error and leave the other peers' traffic unserved, so it is
            # logged with enough context to identify the offending peer and
            # message instead. Handlers that need failures to be fatal must act
            # on them themselves.
            self.exception(
                f"Exception in handler for {type(message).__name__} "
                f"from {identity}: {e!r}"
            )
            return

        if response is None:
            return
        try:
            await self.send_to(identity, response)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - send boundary, must not crash ROUTER dispatcher
            # Same containment rationale as above. A DEALER that disconnected
            # between receive and reply is the expected cause (EHOSTUNREACH /
            # ENOTCONN); the ROUTER socket stays valid for every other peer.
            self.exception(f"Failed to send response to {identity}: {e!r}")
            await self._recover_from_send_failure(identity, e)

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

    # A departed DEALER: the ROUTER stays valid for every other peer.
    _PEER_GONE_ERRNOS = frozenset({zmq.EHOSTUNREACH, zmq.ENOTCONN})

    # A partial multipart send left the socket FSM broken; only a fresh socket
    # recovers it, and until then every send to every peer fails.
    _SOCKET_BROKEN_ERRNOS = frozenset({zmq.EFSM})

    async def _recover_from_send_failure(self, identity: str, error: Exception) -> None:
        """Decide whether a send failure needs the socket rebuilt."""
        if self.stop_requested or not isinstance(error, zmq.ZMQError):
            return

        if error.errno in self._PEER_GONE_ERRNOS:
            self.warning(
                f"Peer {identity} unreachable (errno={error.errno}), dropping "
                "response; ROUTER socket remains valid"
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

        # The new socket has a new FD. The reader is still registered against
        # the old, now-closed one: rebuild it or the ROUTER goes deaf and we
        # leak an add_reader on a recycled FD.
        if self._fd_reader is not None and not self.stop_requested:
            self._fd_reader.stop()
            self._start_fd_reader()

    def _start_fd_reader(self) -> None:
        """Build and start the edge-triggered FD reader for the current socket."""
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

    async def request_to(self, identity: str, struct: Struct, timeout: float) -> Any:
        """Send a request to one DEALER and await the reply matched by ``cid``.

        The peer must echo the request's ``cid`` on its response struct; that is
        what pairs the two. Used by the worker-pod lifecycle channel, e.g.
        ``request_to("worker_0", GroupPeerCommand(cid="a1b2", ...), timeout=60.0)``
        returning the peer's ``GroupPeerCommandAck``.

        Args:
            identity: The DEALER client's identity (routing key).
            struct: The request struct; must carry a non-empty ``cid``.
            timeout: Maximum seconds to wait for the reply.

        Returns:
            The decoded response struct whose ``cid`` matched the request.

        Raises:
            ValueError: If ``struct`` has no ``cid`` to correlate on.
            TimeoutError: If no matching reply arrives within ``timeout``.
        """
        cid = getattr(struct, "cid", None)
        # ``not cid`` (not ``is None``) so this agrees with the resolution guard
        # in _try_resolve_pending_request: an empty-string cid would otherwise
        # register a future that side can never match, hanging until timeout.
        if not cid:
            raise ValueError(
                f"request_to() requires a struct with a 'cid' to correlate the "
                f"reply on; {type(struct).__name__} has none. Use send_to() for "
                f"fire-and-forget sends."
            )

        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending_requests[cid] = future
        try:
            await self.send_to(identity, struct)
            return await asyncio.wait_for(future, timeout=timeout)
        finally:
            self._pending_requests.pop(cid, None)

    def _try_resolve_pending_request(self, message: Any) -> bool:
        """Resolve a pending ``request_to`` future when ``message.cid`` matches.

        The ``cid`` is the SOLE correlation key -- the sending identity is
        deliberately not checked. Each cid is a uuid4 minted per request and sent
        to exactly one peer, so a cross-peer collision requires a uuid4
        collision. Matching on identity as well would be strictly worse here: a
        DEALER that reconnects between request and reply can present a different
        routing key, and rejecting its reply converts a working exchange into a
        guaranteed timeout. Callers that need peer-authenticated replies must
        carry the peer identity inside the message and check it themselves.

        Returns True when the message was consumed as a response, so the caller
        skips normal handler dispatch.
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
        """
        Background task for receiving messages from DEALER clients.

        Runs continuously until stop is requested. Decodes messages as
        WorkerToRouterMessage (WorkerDispatchable | WorkerShutdown | CreditReturn) using msgpack.
        """
        self.debug("Streaming ROUTER receiver task started")

        # Always drive the ROUTER off its raw FD: edge-triggered NOBLOCK multipart
        # drain on recv, sync NOBLOCK on send (the driver owns both directions).
        self._start_fd_reader()
