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
from collections import deque
from collections.abc import Callable

import msgspec
import zmq
from msgspec import Struct

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.models.base_models import msgspec_enc_hook
from aiperf.zmq.zmq_base_client import BaseZMQClient

# Pre-created encoder (caches schema); matches the streaming DEALER/ROUTER wire.
# See streaming_router_client for why enc_hook is wired in.
_encoder = msgspec.msgpack.Encoder(enc_hook=msgspec_enc_hook)


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
        # Backlog of encoded frames the socket refused (zmq.Again). Unbounded, to
        # match the DEALER's FdEdgeReader._send_buf: this channel carries credit
        # returns, and a dropped return stalls its phase until the run-level
        # timeout, so retaining the frame always beats discarding it.
        self._send_buf: deque[bytes] = deque()
        self._drain_wakeup = asyncio.Event()
        # Set by the owning service (e.g. Worker) to learn when a non-empty
        # backlog is discarded on stop. The client has no access to the
        # exit-error mechanism itself, so a dropped backlog would otherwise be
        # visible only as a log line -- never reaching the CLI exit code or
        # run summary.
        self.on_backlog_dropped: Callable[[int], None] | None = None

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
        not-yet-connected peer raise ``zmq.Again`` -- during a startup race, and
        again for the whole of every reconnect backoff (100ms..5s), which far
        outlasts the bounded inline retry. Frames that outlive the retries are
        parked in an unbounded backlog and re-sent by the drain task, so a
        credit return completing inside a reconnect window is never lost.
        """
        await self._check_initialized()
        if max_retries is None:
            max_retries = Environment.ZMQ.PUSH_MAX_RETRIES
        data = _encoder.encode(struct)
        # FIFO: once anything is parked, everything queues behind it, or a
        # return could overtake the FirstToken of the same request.
        if self._send_buf:
            self._buffer(data)
            return
        while True:
            try:
                zmq.Socket.send(self.socket, data, flags=zmq.NOBLOCK, copy=False)
                break
            except (asyncio.CancelledError, zmq.ContextTerminated):
                return
            except zmq.Again:
                if retry_count >= max_retries or self._send_buf:
                    self._buffer(data)
                    return
                retry_count += 1
                await asyncio.sleep(Environment.ZMQ.PUSH_RETRY_DELAY)
        if self.is_trace_enabled:
            self.trace(f"Sent struct: {struct}")

    def _buffer(self, data: bytes) -> None:
        """Park an unsendable frame and wake the drain task."""
        self._send_buf.append(data)
        self._drain_wakeup.set()
        if len(self._send_buf) % 1000 == 0:
            self.warning(
                f"Credit-return PUSH backlog at {len(self._send_buf)} frames; "
                "the PULL peer has been unreachable for a while."
            )

    @background_task(immediate=True, interval=None)
    async def _drain_send_buffer(self) -> None:
        """Re-send parked frames once the peer accepts writes again.

        Polls rather than watching the FD: PUSH has no receive path, so nothing
        else on this socket needs the edge-trigger, and the backlog is empty in
        every healthy run.
        """
        while not self.stop_requested:
            await self._drain_wakeup.wait()
            self._drain_wakeup.clear()
            while self._send_buf and not self.stop_requested:
                try:
                    zmq.Socket.send(
                        self.socket, self._send_buf[0], flags=zmq.NOBLOCK, copy=False
                    )
                except (asyncio.CancelledError, zmq.ContextTerminated):
                    return
                except zmq.Again:
                    await asyncio.sleep(Environment.ZMQ.PUSH_RETRY_DELAY)
                    continue
                except zmq.ZMQError as e:
                    self.warning(
                        lambda e=e: f"ZMQError draining credit-return buffer; retrying: {e}"
                    )
                    await asyncio.sleep(Environment.ZMQ.PUSH_RETRY_DELAY)
                    continue
                self._send_buf.popleft()

    @on_stop
    async def _warn_on_undrained_backlog(self) -> None:
        """LINGER=0 discards anything still parked; say so rather than hide it.

        A log line alone never reaches the CLI exit code or run summary, so
        the drop is also reported through ``on_backlog_dropped`` when set,
        letting the owning service surface it via its own exit-error list.
        """
        self._drain_wakeup.set()
        if self._send_buf:
            dropped = len(self._send_buf)
            self.warning(f"Dropping {dropped} undelivered credit-return frames on stop")
            self._send_buf.clear()
            if self.on_backlog_dropped is not None:
                self.on_backlog_dropped(dropped)
