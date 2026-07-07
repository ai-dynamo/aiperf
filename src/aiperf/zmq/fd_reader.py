# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge-triggered ZMQ FD driver for streaming PULL/DEALER/ROUTER sockets.

Drives a ZMQ socket entirely off its raw FD via ``loop.add_reader``, bypassing
``zmq.asyncio``'s await-recv/await-send wrappers so message bursts clear in fewer
event-loop round-trips. Generalizes the single-socket PUSH/PULL FD-drain
technique to the bidirectional DEALER/ROUTER credit channel.

A ZMQ socket exposes ONE fd (``ZMQ_FD``) that becomes OS-readable when *either*
recv-ready (``POLLIN``) or send-ready (``POLLOUT``) and is edge-triggered: it only
re-signals on a 0->nonzero ``ZMQ_EVENTS`` transition, and ``ZMQ_EVENTS`` must be
re-read after every send/recv to re-arm it. Therefore a single ``add_reader``
owns BOTH directions — recv drains on ``POLLIN``, the (rarely-used) send backlog
drains on ``POLLOUT`` — and the same socket must never also be driven by
``zmq.asyncio`` (its async send/recv read ``ZMQ_EVENTS`` too and corrupt the
shared edge-trigger). Callers route sends through :meth:`send`.

With the default ``SNDHWM=0`` (unbounded send queue) the NOBLOCK send never
blocks, so the send buffer / ``POLLOUT`` path stays empty; it exists only to stay
correct under a finite send HWM.

Per-readable drain cycle (``_pump``):

    ZMQ_FD readable   (libzmq queue went 0 -> nonzero: a ZMQ_EVENTS edge)
        |
        v
    +- _pump() --------------------------------------------------
    |  recv NOBLOCK in a batch loop (up to batch_limit):
    |      dispatch each item; re-read ZMQ_EVENTS after each recv
    |      stop on Again, on POLLIN clearing, or at the batch cap
    |  then flush any buffered sends while POLLOUT is set
    +------------------------------------------------------------
        |
        v
    POLLIN still set after the pass?
        |- yes -> call_soon(_pump)   (re-arm: the edge will NOT
        |                             re-fire for mid-drain arrivals)
        '- no  -> idle until the next 0 -> nonzero ZMQ_EVENTS edge
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable
from typing import Any

import zmq


class FdEdgeReader:
    """Drive a ZMQ socket's recv (and optionally send) via an asyncio FD reader.

    Args:
        socket: The ZMQ socket to drive (``zmq.asyncio`` socket is fine; this uses
            the synchronous NOBLOCK ops supplied by ``recv_one``/``send_one``).
        recv_one: One synchronous NOBLOCK recv returning the decoded item, or
            raising ``zmq.Again`` when drained.
        dispatch: Called with each received item (must not block).
        batch_limit: Max recvs drained per pass so the drain cannot monopolize the
            loop; <=0 falls back to 256.
        send_one: One synchronous NOBLOCK send of a buffered item, or raising
            ``zmq.Again`` if the send HWM is hit. Required to use :meth:`send`.
        on_error: Optional callback for unexpected exceptions during a pass.
    """

    def __init__(
        self,
        *,
        socket: Any,
        recv_one: Callable[[], Any],
        dispatch: Callable[[Any], None],
        batch_limit: int,
        send_one: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._socket = socket
        self._recv_one = recv_one
        self._dispatch = dispatch
        self._batch_limit = batch_limit if batch_limit and batch_limit > 0 else 256
        self._send_one = send_one
        self._on_error = on_error
        self._loop: asyncio.AbstractEventLoop | None = None
        self._fd: int | None = None
        self._send_buf: deque[Any] = deque()
        self._rearm_pending = False
        self._stopped = False

    def start(self) -> None:
        """Register the FD reader and drain anything already queued."""
        self._loop = asyncio.get_running_loop()
        self._fd = self._socket.getsockopt(zmq.FD)
        self._loop.add_reader(self._fd, self._pump)
        self._pump()

    def stop(self) -> None:
        """Unregister the FD reader."""
        self._stopped = True
        if self._fd is not None and self._loop is not None:
            with contextlib.suppress(ValueError, OSError):
                self._loop.remove_reader(self._fd)
            self._fd = None

    def send(self, item: Any) -> None:
        """Send an item synchronously (NOBLOCK), buffering only if the HWM blocks.

        Reading ``ZMQ_EVENTS`` after the send (via the trailing re-arm) keeps the
        recv edge-trigger consistent on this shared FD.
        """
        if self._send_one is None:
            raise RuntimeError("FdEdgeReader.send called without a send_one callable")
        if self._send_buf:
            # Preserve ordering: something is already queued behind a full HWM.
            self._send_buf.append(item)
        else:
            try:
                self._send_one(item)
            except zmq.Again:
                self._send_buf.append(item)
        # Keep the FD armed for recv and schedule a drain if work remains.
        self._rearm()

    def _pump(self) -> None:
        if self._stopped:
            return
        try:
            events = self._socket.getsockopt(zmq.EVENTS)
            drained = 0
            while (events & zmq.POLLIN) and drained < self._batch_limit:
                try:
                    item = self._recv_one()
                except zmq.Again:
                    break
                self._dispatch(item)
                drained += 1
                events = self._socket.getsockopt(zmq.EVENTS)
            while self._send_buf and (events & zmq.POLLOUT):
                try:
                    self._send_one(self._send_buf[0])  # type: ignore[misc]
                except zmq.Again:
                    break
                self._send_buf.popleft()
                events = self._socket.getsockopt(zmq.EVENTS)
        except (zmq.ContextTerminated, zmq.ZMQError):
            return
        except Exception as e:  # drain boundary; report this pass
            # A non-ZMQ failure (e.g. a malformed frame that fails to decode) has
            # already consumed its frame. Re-arm before returning: the FD is
            # edge-triggered and POLLIN may still be set, so without re-checking
            # ZMQ_EVENTS here the remaining queued messages would be stranded
            # until the next 0->nonzero edge (which a backlog will never produce).
            if self._on_error is not None:
                self._on_error(e)
            self._rearm()
            return
        self._rearm(events)

    def _rearm(self, events: int | None = None) -> None:
        """Reschedule a pass if recv is still pending (the edge-trigger won't
        re-fire for messages that arrived mid-drain). A send backlog is left to
        the FD's own POLLOUT signal to avoid busy-looping against a full HWM."""
        if self._stopped or self._rearm_pending or self._loop is None:
            return
        try:
            if events is None:
                events = self._socket.getsockopt(zmq.EVENTS)
        except zmq.ZMQError:
            return
        if events & zmq.POLLIN:
            self._rearm_pending = True
            self._loop.call_soon(self._run)

    def _run(self) -> None:
        self._rearm_pending = False
        self._pump()
