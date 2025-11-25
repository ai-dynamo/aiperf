# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming DEALER client for bidirectional communication with ROUTER."""

import asyncio
from collections.abc import Awaitable, Callable

import zmq

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommClientType
from aiperf.common.factories import CommunicationClientFactory
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.messages import Message
from aiperf.common.protocols import StreamingDealerClientProtocol
from aiperf.common.utils import yield_to_event_loop
from aiperf.zmq.zmq_base_client import BaseZMQClient


@implements_protocol(StreamingDealerClientProtocol)
@CommunicationClientFactory.register(CommClientType.STREAMING_DEALER)
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
        # Create via comms (recommended - handles lifecycle management)
        dealer = comms.create_streaming_dealer_client(
            address=CommAddress.CREDIT_ROUTER,  # or "tcp://localhost:5555"
            identity="worker-1",
        )

        async def handle_message(message: Message) -> None:
            if message.message_type == MessageType.CREDIT_DROP:
                do_some_work(message.credit)
                await dealer.send(CreditReturnMessage(...))

        dealer.register_receiver(handle_message)

        # Lifecycle managed by comms - initialize/start/stop comms instead
        await comms.initialize()
        await comms.start()
        await dealer.send(WorkerReadyMessage(...))
        ...
        await dealer.send(WorkerShutdownMessage(...))
        await comms.stop()
    ```
    """

    def __init__(
        self,
        address: str,
        identity: str,
        bind: bool = False,
        socket_ops: dict | None = None,
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
        self._receiver_handler: Callable[[Message], Awaitable[None]] | None = None

    def register_receiver(self, handler: Callable[[Message], Awaitable[None]]) -> None:
        """
        Register handler for incoming messages from ROUTER.

        The handler will be called for each message received.

        Args:
            handler: Async function that takes (message: Message)
        """
        if self._receiver_handler is not None:
            raise ValueError("Receiver handler already registered")
        self._receiver_handler = handler
        self.debug(
            lambda: f"Registered streaming DEALER receiver handler for {self.identity}"
        )

    @on_stop
    async def _clear_receiver(self) -> None:
        """Clear receiver handler on stop."""
        self._receiver_handler = None

    async def send(self, message: Message) -> None:
        """
        Send message to ROUTER.

        Args:
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
            # DEALER automatically handles framing - use single-frame send
            await self.socket.send(message.to_json_bytes())
            if self.is_trace_enabled:
                self.trace(f"Sent message: {message}")
        except Exception as e:
            self.exception(f"Failed to send message: {e}")
            raise

    @background_task(immediate=True, interval=None)
    async def _streaming_dealer_receiver(self) -> None:
        """
        Background task for receiving messages from ROUTER.

        Runs continuously until stop is requested. Receives messages with DEALER
        envelope format: [empty_delimiter, message_bytes] or just [message_bytes]
        """
        self.debug(
            lambda: f"Streaming DEALER receiver task started for {self.identity}"
        )

        while not self.stop_requested:
            try:
                message_bytes = await self.socket.recv()
                if self.is_trace_enabled:
                    self.trace(f"Received message: {message_bytes}")
                message = Message.from_json(message_bytes)

                if self._receiver_handler:
                    self.execute_async(self._receiver_handler(message))
                else:
                    self.warning(
                        f"Received {message.message_type} message but no handler registered"
                    )

            except zmq.Again:
                self.debug("No data on dealer socket received, yielding to event loop")
                await yield_to_event_loop()
            except Exception as e:
                self.exception(f"Exception receiving messages: {e}")
                await yield_to_event_loop()
            except asyncio.CancelledError:
                self.debug("Streaming DEALER receiver task cancelled")
                raise  # re-raise the cancelled error

        self.debug(
            lambda: f"Streaming DEALER receiver task stopped for {self.identity}"
        )
