# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming ROUTER client for bidirectional communication with DEALER clients."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeAlias

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import WorkerToRouterMessage
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
        **kwargs,
    ) -> None:
        """
        Initialize the streaming ROUTER client.

        Args:
            address: The address to bind or connect to (e.g., "tcp://*:5555" or "ipc:///tmp/socket")
            bind: Whether to bind (True) or connect (False) the socket
            socket_ops: Additional socket options to set
            **kwargs: Additional arguments passed to BaseZMQClient
        """
        super().__init__(zmq.SocketType.ROUTER, address, bind, socket_ops, **kwargs)
        self._receiver_handler: WorkerToRouterHandler | None = None
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.STREAMING_ROUTER_YIELD_INTERVAL

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
        self._receiver_handler = None

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

        try:
            # Send using routing envelope pattern (identity string → bytes)
            await self.socket.send_multipart(
                [identity.encode(), _encoder.encode(struct)]
            )
            if self.is_trace_enabled:
                self.trace(f"Sent {type(struct).__name__} to {identity}: {struct}")
        except Exception as e:
            self.exception(f"Failed to send to {identity}: {e}")
            raise

    @background_task(immediate=True, interval=None)
    async def _streaming_router_receiver(self) -> None:
        """
        Background task for receiving messages from DEALER clients.

        Runs continuously until stop is requested. Decodes messages as
        WorkerToRouterMessage (WorkerReady | WorkerShutdown | CreditReturn) using msgpack.
        """
        self.debug("Streaming ROUTER receiver task started")

        while not self.stop_requested:
            try:
                data = await self.socket.recv_multipart()
                if self.is_trace_enabled:
                    self.trace(f"Received message: {data}")

                # ROUTER envelope: [identity, message_bytes]
                identity = data[0].decode("utf-8")
                message = _decoder.decode(data[-1])

                if self.is_trace_enabled:
                    self.trace(
                        f"Received {type(message).__name__} from {identity}: {message}"
                    )

                if self._receiver_handler:
                    self.execute_async(self._receiver_handler(identity, message))
                    self._msg_count += 1
                    # Yield periodically to allow scheduled handlers to run
                    # and prevent event loop starvation during message bursts.
                    if (
                        self._yield_interval > 0
                        and self._msg_count >= self._yield_interval
                    ):
                        await yield_to_event_loop()
                else:
                    self.warning(
                        f"Received {type(message).__name__} but no handler registered"
                    )

            except zmq.Again:
                self.debug("Router receiver task timed out")
                await yield_to_event_loop()
                continue
            except Exception as e:
                if not self.stop_requested:
                    self.exception(f"Error in streaming ROUTER receiver: {e}")
                await yield_to_event_loop()
            except asyncio.CancelledError:
                self.debug("Streaming ROUTER receiver task cancelled")
                break

        self.debug("Streaming ROUTER receiver task stopped")
