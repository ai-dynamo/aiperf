# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming DEALER client for bidirectional communication with ROUTER."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeAlias

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.utils import yield_to_event_loop
from aiperf.credit.messages import RouterToWorkerMessage
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Pre-created encoder/decoder for performance (caches schema)
_encoder = msgspec.msgpack.Encoder()
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
    - No request-response pairing - pure streaming
    - Supports concurrent message processing

    Example:
    ```python
        from aiperf.common.structs import (
            Credit, CancelCredits, WorkerReady, WorkerShutdown, CreditReturn
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
        await dealer.send(WorkerReady(worker_id="worker-1"))
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
        self._receiver_handler: RouterToWorkerHandler | None = None
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.STREAMING_DEALER_YIELD_INTERVAL

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
        """Clear receiver handler on stop."""
        self._receiver_handler = None

    async def send(self, struct: Struct) -> None:
        """Send struct to ROUTER."""
        await self._check_initialized()

        try:
            # DEALER automatically handles framing - use single-frame send
            await self.socket.send(_encoder.encode(struct))
            if self.is_trace_enabled:
                self.trace(f"Sent struct: {struct}")
        except Exception as e:
            self.exception(f"Failed to send message: {e}")
            raise

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

        while not self.stop_requested:
            try:
                message_bytes = await self.socket.recv()
                if self.is_trace_enabled:
                    self.trace(f"Received message: {message_bytes}")
                message = _decoder.decode(message_bytes)

                if self._receiver_handler:
                    self.execute_async(self._receiver_handler(message))
                    self._msg_count += 1
                    # Yield periodically to allow scheduled handlers to run
                    # and prevent event loop starvation during message bursts.
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
                self.debug("No data on dealer socket received, yielding to event loop")
                await yield_to_event_loop()
            except asyncio.CancelledError:
                self.debug("Streaming DEALER receiver task cancelled")
                raise  # re-raise the cancelled error
            except Exception as e:
                self.exception(
                    f"Exception receiving messages for client {self.client_id}: {e!r}"
                )
                await yield_to_event_loop()

        self.debug(
            lambda: f"Streaming DEALER receiver task stopped for {self.identity}"
        )
