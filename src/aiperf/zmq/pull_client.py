# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
from collections.abc import Callable, Coroutine
from typing import Any

import zmq.asyncio

from aiperf.common.environment import Environment
from aiperf.common.hooks import background_task, on_stop
from aiperf.common.messages import Message
from aiperf.common.types import MessageTypeT
from aiperf.zmq.zmq_base_client import BaseZMQClient


class ZMQPullClient(BaseZMQClient):
    """
    ZMQ PULL socket client for receiving work from PUSH sockets.

    The PULL socket receives messages from PUSH sockets in a pipeline pattern,
    distributing work fairly among multiple PULL workers.

    ASCII Diagram:
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │    PUSH     │      │    PULL     │      │    PULL     │
    │ (Producer)  │      │ (Worker 1)  │      │ (Worker 2)  │
    │             │      └─────────────┘      └─────────────┘
    │   Tasks:    │             ▲                     ▲
    │   - Task A  │─────────────┘                     │
    │   - Task B  │───────────────────────────────────┘
    │   - Task C  │─────────────┐
    │   - Task D  │             ▼
    └─────────────┘      ┌─────────────┐
                         │    PULL     │
                         │ (Worker N)  │
                         └─────────────┘

    Usage Pattern:
    - PULL receives work from multiple PUSH producers
    - Work is fairly distributed among PULL workers
    - Pipeline pattern for distributed processing
    - Each message is delivered to exactly one PULL socket

    PULL/PUSH is a One-to-Many communication pattern. If you need Many-to-Many,
    use a ZMQ Proxy as well. see :class:`ZMQPushPullProxy` for more details.
    """

    def __init__(
        self,
        *,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
        additional_bind_address: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ZMQ Puller class.

        Args:
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
            max_pull_concurrency (int, optional): The maximum number of concurrent requests to allow.
            additional_bind_address (str, optional): Optional second address to bind to for dual-bind
                mode (e.g., IPC + TCP in Kubernetes). Only used when bind=True.
        """
        super().__init__(
            zmq.SocketType.PULL,
            address,
            bind,
            socket_ops,
            additional_bind_address=additional_bind_address,
            **kwargs,
        )
        self._pull_callbacks: dict[
            MessageTypeT, Callable[[Message], Coroutine[Any, Any, None]]
        ] = {}

        # `or` (not `is not None`): 0 is not a valid bound here — it would halt
        # the FD drain entirely (`while self._inflight < self._max_inflight`), so
        # coalesce a falsy value to the configured default.
        self._max_inflight: int = (
            max_pull_concurrency or Environment.ZMQ.PULL_MAX_CONCURRENCY
        )
        self._msg_count: int = 0
        self._yield_interval: int = Environment.ZMQ.PULL_YIELD_INTERVAL

        # FD-reader state. Edge-triggered drive over the raw socket FD; flow
        # control via _inflight (the reader stops draining at _max_inflight,
        # leaving messages in the ZMQ buffer so the HWM backpressures the pusher).
        self._loop: asyncio.AbstractEventLoop | None = None
        self._fd: int | None = None
        self._inflight: int = 0
        self._rearm_pending: bool = False

    @background_task(immediate=True, interval=None)
    async def _pull_receiver(self) -> None:
        """Background task for receiving data from the pull socket.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        # Always drive the PULL socket off its raw FD with an edge-triggered
        # NOBLOCK drain.
        self._start_fd_reader()

    def _start_fd_reader(self) -> None:
        """Register an edge-triggered drain on the raw socket FD.

        Bypasses zmq.asyncio's await-recv polling: asyncio watches the ZMQ FD and
        calls ``_on_pull_readable``, which drains all immediately-available messages
        with NOBLOCK recv.
        """
        self._loop = asyncio.get_running_loop()
        self._fd = self.socket.getsockopt(zmq.FD)
        self._loop.add_reader(self._fd, self._on_pull_readable)
        # The FD only fires on a 0->nonzero events transition, so drain anything
        # already queued before we registered.
        self._on_pull_readable()

    def _on_pull_readable(self) -> None:
        """Drain ready messages with NOBLOCK recv (bounded per call so the drain
        itself cannot monopolize the event loop)."""
        if self.stop_requested:
            return
        limit = self._yield_interval if self._yield_interval > 0 else 256
        drained = 0
        try:
            while self._inflight < self._max_inflight and drained < limit:
                # Re-check EVENTS each iteration: the FD is edge-triggered, so the
                # only reliable readiness signal is ZMQ_EVENTS, not the FD itself.
                if not (self.socket.getsockopt(zmq.EVENTS) & zmq.POLLIN):
                    break
                try:
                    data = zmq.Socket.recv(self.socket, flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if self.is_trace_enabled:
                    self.trace(lambda d=data: f"Received message from pull socket: {d}")
                self._inflight += 1
                drained += 1
                self.execute_async(self._process_message_fd(data))
        except (zmq.ContextTerminated, zmq.ZMQError):
            return
        # More may be pending (we hit the batch/inflight cap, or a message raced in
        # during draining); reschedule rather than rely on the FD re-firing.
        self._maybe_rearm()

    def _maybe_rearm(self) -> None:
        """Reschedule a drain if work is pending and we have inflight capacity."""
        if self.stop_requested or self._rearm_pending or self._loop is None:
            return
        try:
            pending = bool(self.socket.getsockopt(zmq.EVENTS) & zmq.POLLIN)
        except zmq.ZMQError:
            return
        if pending and self._inflight < self._max_inflight:
            self._rearm_pending = True
            self._loop.call_soon(self._run_reader)

    def _run_reader(self) -> None:
        self._rearm_pending = False
        self._on_pull_readable()

    async def _process_message_fd(self, message_json_bytes: bytes) -> None:
        """FD-reader variant of ``_process_message`` using the _inflight counter."""
        try:
            message = Message.from_json(message_json_bytes)
            del message_json_bytes
            if message.message_type in self._pull_callbacks:
                await self._pull_callbacks[message.message_type](message)
            else:
                self.warning(
                    f"Pull message received for message type {message.message_type} without callback"
                )
        finally:
            self._inflight -= 1
            # A slot freed; resume draining if the reader paused at the cap.
            self._maybe_rearm()

    @on_stop
    async def _stop(self) -> None:
        """Wait for all tasks to complete."""
        if self._fd is not None and self._loop is not None:
            with contextlib.suppress(ValueError, OSError):
                self._loop.remove_reader(self._fd)
            self._fd = None
        await self.cancel_all_tasks()

    def register_pull_callback(
        self,
        message_type: MessageTypeT,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a ZMQ Pull data callback for a given message type.

        Note that only one callback can be registered for a given message type.

        Args:
            message_type: The message type to register the callback for.
            callback: The function to call when data is received.
        Raises:
            CommunicationError: If the client is not initialized
        """
        # Register callback
        if message_type not in self._pull_callbacks:
            self._pull_callbacks[message_type] = callback
        else:
            raise ValueError(
                f"Callback already registered for message type {message_type}"
            )
