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
from aiperf.common.utils import yield_to_event_loop
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
        self._receiver_handler = None
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

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
            CommunicationError: If send fails
        """
        if not self.socket:
            raise NotInitializedError("Socket not initialized or closed")
        if self.stop_requested:
            raise asyncio.CancelledError("Socket was stopped")

        try:
            await self.socket.send_multipart(
                [identity.encode("utf-8", "surrogateescape"), _encoder.encode(struct)]
            )
            if self.is_trace_enabled:
                self.trace(f"Sent {type(struct).__name__} to {identity}: {struct}")
        except Exception as e:
            self.exception(
                f"Failed to send to {identity} for client {self.client_id}: {e!r}"
            )
            raise

    async def request_to(self, identity: str, struct: Struct, timeout: float) -> Any:
        """Send a request to a specific DEALER and wait for a response matched by ``cid``.

        Args:
            identity: The DEALER client's identity (routing key)
            struct: The request struct (must have ``cid`` attribute)
            timeout: Maximum seconds to wait for a response

        Returns:
            The decoded response struct.

        Raises:
            asyncio.TimeoutError: If no response within timeout.
        """
        cid = getattr(struct, "cid", None)
        if cid is None:
            raise ValueError("request_to() requires a struct with 'cid'")

        future: asyncio.Future[Any] = asyncio.Future()
        self._pending_requests[cid] = future

        try:
            await self.send_to(identity, struct)
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise
        finally:
            self._pending_requests.pop(cid, None)

    async def _dispatch_message(
        self, identity: str, routing_envelope: tuple[bytes, ...], message: Any
    ) -> None:
        """Dispatch a received message to the handler.

        If the handler returns a Struct, encode and send it back via the
        routing envelope (request-reply). Otherwise treat as fire-and-forget.
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
                await self.socket.send_multipart(
                    [*routing_envelope, _encoder.encode(response)]
                )
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
        except (zmq.ZMQError, asyncio.TimeoutError) as recreate_error:
            if not self.stop_requested:
                self.exception(
                    "Failed to recreate streaming ROUTER socket after send "
                    f"failure to {identity}: {recreate_error!r}"
                )

    @background_task(immediate=True, interval=None)
    async def _streaming_router_receiver(self) -> None:
        """Background task for receiving messages from DEALER clients."""
        self.debug("Streaming ROUTER receiver task started")

        while not self.stop_requested:
            try:
                data = await self.socket.recv_multipart()
                if self.is_trace_enabled:
                    self.trace(f"Received message: {data}")

                # ROUTER envelope: [identity, message_bytes]
                identity = data[0].decode("utf-8", "surrogateescape")
                message = self._decoder.decode(data[-1])

                routing_envelope: tuple[bytes, ...] = tuple(data[:-1])

                if self.is_trace_enabled:
                    self.trace(
                        f"Received {type(message).__name__} from {identity}: {message}"
                    )

                # Check if this is a response to a pending request (by cid)
                cid = getattr(message, "cid", None)
                if cid and cid in self._pending_requests:
                    future = self._pending_requests.pop(cid)
                    if not future.done():
                        future.set_result(message)
                    continue

                if self._receiver_handler:
                    self.execute_async(
                        self._dispatch_message(identity, routing_envelope, message)
                    )
                    self._msg_count += 1
                    if (
                        self._yield_interval > 0
                        and self._msg_count % self._yield_interval == 0
                    ):
                        await yield_to_event_loop()
                else:
                    self.warning(
                        f"Received {type(message).__name__} but no handler registered"
                    )

            except zmq.Again:
                self.trace("Router receiver task timed out")
                await yield_to_event_loop()
                continue
            except asyncio.CancelledError:
                self.debug("Streaming ROUTER receiver task cancelled")
                raise
            except zmq.ContextTerminated:
                self.debug(
                    "Streaming ROUTER receiver task stopped (ZMQ context terminated)"
                )
                break
            except (zmq.ZMQError, asyncio.TimeoutError, msgspec.DecodeError) as e:
                if not self.stop_requested:
                    self.exception(
                        f"Error in streaming ROUTER receiver for client {self.client_id}: {e!r}"
                    )
                await yield_to_event_loop()

        self.debug("Streaming ROUTER receiver task stopped")
