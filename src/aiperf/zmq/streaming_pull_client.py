# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming PULL client for the typed credit-return fan-in channel.

Receive-only counterpart of :class:`ZMQStreamingPushClient`. Decodes the typed
``WorkerToRouterMessage`` msgpack structs (the same wire as the streaming ROUTER
credit client) off a PULL socket, so the sticky router collects
``CreditReturn``/``FirstToken`` over the dedicated credit-return fan-in channel.

Unlike the streaming ROUTER, PULL carries no peer identity, so the receiver
handler takes only the decoded message (the worker id, when needed, travels
inside the message). PULL is recv-only, so its FD has no POLLOUT to mask the
recv edge — the FdEdgeReader is used recv-only (no send path).
"""

from collections.abc import Awaitable, Callable
from typing import TypeAlias

import msgspec
import zmq

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.credit.messages import WorkerToRouterMessage
from aiperf.zmq.fd_reader import FdEdgeReader
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Pre-created decoder (caches schema); matches the streaming ROUTER wire.
_decoder = msgspec.msgpack.Decoder(WorkerToRouterMessage)

StreamingPullHandler: TypeAlias = Callable[[WorkerToRouterMessage], Awaitable[None]]


class ZMQStreamingPullClient(BaseZMQClient):
    """ZMQ PULL client that decodes typed ``WorkerToRouterMessage`` structs.

    Mirrors the receive path of :class:`ZMQStreamingDealerClient` (single-frame
    typed decode, edge-triggered FD drain) on a recv-only PULL socket that fans
    in credit returns from every worker's PUSH socket.

    ASCII Diagram (credit-return fan-in, manager side):
    ┌──────────────┐                    ┌──────────────┐
    │     PUSH     │───── returns ─────►│              │
    │  (Worker 1)  │                    │              │
    └──────────────┘                    │              │
    ┌──────────────┐                    │     PULL     │
    │     PUSH     │───── returns ─────►│  (Manager)   │
    │  (Worker 2)  │                    │              │
    └──────────────┘                    │              │
    ┌──────────────┐                    │              │
    │     PUSH     │───── returns ─────►│              │
    │  (Worker N)  │                    │              │
    └──────────────┘                    └──────────────┘

    Usage Pattern:
    - PULL binds and fans in returns from every worker's PUSH socket
    - PULL decodes typed CreditReturn/FirstToken structs (recv-only, no send)
    - No peer identity in the envelope - the worker id travels in the message
    - Edge-triggered FD drain (recv-only, so no POLLOUT to mask the recv edge)
    """

    def __init__(
        self,
        *,
        address: str,
        bind: bool = False,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
        additional_bind_address: str | None = None,
        **kwargs,
    ) -> None:
        # max_pull_concurrency is accepted for factory-call uniformity; this
        # client dispatches via execute_async and does not gate concurrency.
        del max_pull_concurrency
        super().__init__(
            zmq.SocketType.PULL,
            address,
            bind,
            socket_ops,
            additional_bind_address=additional_bind_address,
            **kwargs,
        )
        self._receiver_handler: StreamingPullHandler | None = None
        self._yield_interval: int = Environment.ZMQ.STREAMING_ROUTER_YIELD_INTERVAL
        self._msg_count: int = 0
        self._fd_reader: FdEdgeReader | None = None

    def register_receiver(self, handler: StreamingPullHandler) -> None:
        """Register the handler invoked for each decoded WorkerToRouterMessage."""
        if self._receiver_handler is not None:
            raise ValueError("Receiver handler already registered")
        self._receiver_handler = handler

    @on_stop
    async def _clear_receiver(self) -> None:
        if self._fd_reader is not None:
            self._fd_reader.stop()
            self._fd_reader = None
        self._receiver_handler = None

    def _recv_one(self) -> WorkerToRouterMessage:
        """Synchronous NOBLOCK recv + typed decode for the FD-reader drain."""
        return _decoder.decode(zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK))

    def _dispatch(self, message: WorkerToRouterMessage) -> None:
        if self._receiver_handler is not None:
            self.execute_async(self._receiver_handler(message))
        else:
            self.warning(f"Received {type(message).__name__} but no handler registered")

    @background_task(immediate=True, interval=None)
    async def _streaming_pull_receiver(self) -> None:
        """Receive typed structs from the PUSH peers until stop is requested."""
        # Recv-only: no send_one, so no POLLOUT to coordinate on the FD edge.
        self._fd_reader = FdEdgeReader(
            socket=self.socket,
            recv_one=self._recv_one,
            dispatch=self._dispatch,
            batch_limit=self._yield_interval,
            on_error=lambda e: self.exception(
                f"Exception draining pull socket for {self.client_id}: {e!r}"
            ),
        )
        self._fd_reader.start()
