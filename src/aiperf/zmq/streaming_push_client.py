# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming PUSH client for the typed credit-return fan-in channel.

Send-only counterpart of :class:`ZMQStreamingPullClient`. Encodes the same typed
msgpack structs as the streaming DEALER/ROUTER credit clients (no Message-bus
JSON envelope), so workers PUSH ``CreditReturn``/``FirstToken`` to the
timing-manager's PULL fan-in on the dedicated credit-return channel.

PUSH is send-only, so there is no receive path and no FD edge-trigger to share;
with the default ``SNDHWM=0`` a synchronous NOBLOCK send never blocks, matching
the PUSH sync-send fast path used elsewhere.
"""

import asyncio

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.exceptions import CommunicationError
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Pre-created encoder (caches schema); matches the streaming DEALER/ROUTER wire.
_encoder = msgspec.msgpack.Encoder()


class ZMQStreamingPushClient(BaseZMQClient):
    """ZMQ PUSH client that sends typed msgpack structs (no Message-bus envelope).

    Mirrors the encode/send fast path of :class:`ZMQStreamingDealerClient` but on
    a send-only PUSH socket (no identity, no receiver). One of many worker PUSH
    sockets fanning credit returns in to the manager's single PULL socket.

    ASCII Diagram (credit-return fan-in, worker side):
    ┌──────────────┐                    ┌──────────────┐
    │     PUSH     │───── returns ─────►│     PULL     │
    │   (Worker)   │                    │  (Manager)   │
    └──────────────┘                    └──────────────┘

    Usage Pattern:
    - PUSH connects to the manager's PULL on the dedicated credit-return channel
    - PUSH sends typed CreditReturn/FirstToken structs (send-only, no receiver)
    - Worker identity travels inside the message, not a ZMQ envelope
    - SNDHWM=0 so the NOBLOCK send never blocks the event loop
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
        # max_pull_concurrency is accepted for factory-call uniformity and ignored
        # (PUSH has no receive path).
        del max_pull_concurrency
        super().__init__(
            zmq.SocketType.PUSH,
            address,
            bind,
            socket_ops,
            additional_bind_address=additional_bind_address,
            **kwargs,
        )

    async def send(
        self,
        struct: Struct,
        retry_count: int = 0,
        max_retries: int | None = None,
    ) -> None:
        """Encode and send a typed struct to the PULL peer.

        The fast path is a sync NOBLOCK send straight to libzmq, skipping
        zmq.asyncio's Future/polling machinery. With ``SNDHWM=0`` the send never
        blocks on the high-water mark, but ``IMMEDIATE=1`` makes a send to a
        not-yet-connected peer raise ``zmq.Again`` (e.g. a startup race before the
        manager's PULL has connected). Retry with backoff so a credit return is
        never silently dropped, matching :class:`ZMQPushClient`.
        """
        await self._check_initialized()
        if max_retries is None:
            max_retries = Environment.ZMQ.PUSH_MAX_RETRIES
        data = _encoder.encode(struct)
        try:
            zmq.Socket.send(self.socket, data, flags=zmq.NOBLOCK, copy=False)
        except (asyncio.CancelledError, zmq.ContextTerminated):
            return
        except zmq.Again as e:
            if retry_count >= max_retries:
                raise CommunicationError(
                    f"Failed to send {type(struct).__name__} after {retry_count} retries: {e}"
                ) from e
            await asyncio.sleep(Environment.ZMQ.PUSH_RETRY_DELAY)
            return await self.send(struct, retry_count + 1, max_retries)
        if self.is_trace_enabled:
            self.trace(f"Sent struct: {struct}")
