# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming ROUTER client for bidirectional communication with DEALER clients."""

import asyncio
from collections.abc import Awaitable, Callable

import zmq

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommClientType
from aiperf.common.factories import CommunicationClientFactory
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.messages import Message
from aiperf.common.protocols import StreamingRouterClientProtocol
from aiperf.common.utils import yield_to_event_loop
from aiperf.zmq.zmq_base_client import BaseZMQClient


@implements_protocol(StreamingRouterClientProtocol)
@CommunicationClientFactory.register(CommClientType.STREAMING_ROUTER)
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
        # Create via comms (recommended - handles lifecycle management)
        router = comms.create_streaming_router_client(
            address=CommAddress.CREDIT_ROUTER,  # or "tcp://*:5555"
            bind=True,
        )

        async def handle_message(identity: str, message: Message) -> None:
            if message.message_type == MessageType.WORKER_READY:
                await register_worker(identity)
            elif message.message_type == MessageType.WORKER_SHUTDOWN:
                await unregister_worker(identity)

        router.register_receiver(handle_message)

        # Lifecycle managed by comms - initialize/start/stop comms instead
        await comms.initialize()
        await comms.start()

        # Send message to specific DEALER
        await router.send_to("worker-1", CreditDropMessage(...))
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
        self._receiver_handler: Callable[[str, Message], Awaitable[None]] | None = None

    def register_receiver(
        self, handler: Callable[[str, Message], Awaitable[None]]
    ) -> None:
        """
        Register handler for incoming messages from DEALER clients.

        The handler will be called for each message received, with the DEALER's
        identity (routing key) and the message.

        Args:
            handler: Async function that takes (identity: str, message: Message)
        """
        if self._receiver_handler is not None:
            raise ValueError("Receiver handler already registered")
        self._receiver_handler = handler
        self.debug("Registered streaming ROUTER receiver handler")

    @on_stop
    async def _clear_receiver(self) -> None:
        """Clear receiver handler and callbacks on stop."""
        self._receiver_handler = None

    async def send_to(self, identity: str, message: Message) -> None:
        """
        Send message to specific DEALER client by identity.

        Args:
            identity: The DEALER client's identity (routing key)
            message: The message to send

        Raises:
            NotInitializedError: If socket not initialized
            CommunicationError: If send fails
        """
        await self._check_initialized()

        if not isinstance(message, Message):
            raise TypeError(
                f"message must be an instance of Message, got {type(message).__name__}"
            )

        try:
            # Send using routing envelope pattern (identity string → bytes)
            routing_envelope = (identity.encode(),)
            await self.socket.send_multipart(
                [*routing_envelope, message.to_json_bytes()]
            )
            if self.is_trace_enabled:
                self.trace(f"Sent message to {identity}: {message}")
        except Exception as e:
            self.exception(f"Failed to send message to {identity}: {e}")
            raise

    @background_task(immediate=True, interval=None)
    async def _streaming_router_receiver(self) -> None:
        """
        Background task for receiving messages from DEALER clients.

        Runs continuously until stop is requested. Receives messages with ROUTER
        envelope format: [identity, empty_delimiter, message_bytes]
        """
        self.debug("Streaming ROUTER receiver task started")

        while not self.stop_requested:
            try:
                data = await self.socket.recv_multipart()
                if self.is_trace_enabled:
                    self.trace(f"Received message: {data}")

                message = Message.from_json(data[-1])

                routing_envelope: tuple[bytes, ...] = (
                    tuple(data[:-1]) if len(data) > 1 else (b"",)
                )

                # Decode identity for tracking (first frame of routing envelope)
                identity_bytes = routing_envelope[0] if routing_envelope else b""
                identity = identity_bytes.decode("utf-8")

                if self.is_trace_enabled:
                    self.trace(
                        f"Received {message.message_type} message from {identity}: {message}"
                    )

                if self._receiver_handler:
                    self.execute_async(self._receiver_handler(identity, message))
                else:
                    self.warning(
                        f"Received {message.message_type} message but no handler registered"
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
