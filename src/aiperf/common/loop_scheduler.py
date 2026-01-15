# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
High-performance coroutine scheduler for asyncio/uvloop with automatic cleanup.

Wraps asyncio's low-level scheduling (call_later, call_at, call_soon) and solves:

1. **Coroutine lifecycle**: Closes unawaited coroutines on cancellation to prevent
   "coroutine was never awaited" warnings and resource leaks.

2. **Time domain bridging**: Converts perf_counter_ns and perf_counter timestamps to loop.time()
   for precise absolute-time scheduling (e.g., trace replay).

3. **Centralized cleanup**: Cancel all pending/running work atomically on shutdown.

Example::

    scheduler = LoopScheduler()
    scheduler.schedule_later(1.5, send_request())
    scheduler.schedule_at_perf_ns(target_ns, send_request())

    # Shutdown
    cancelled_tasks = scheduler.cancel_all()
    await asyncio.gather(*cancelled_tasks, return_exceptions=True)
"""

import asyncio
import time
from collections.abc import Callable, Coroutine
from typing import TypeAlias

from aiperf.common.constants import NANOS_PER_SECOND

HandleId: TypeAlias = int
"""Unique identifier for a scheduled coroutine that can be used to cancel it."""


class LoopScheduler:
    """
    Schedule coroutines with automatic tracking and cleanup.

    Two-state model:
        - **Pending**: Timer scheduled, not yet fired (in _handles dict)
        - **Running**: Timer fired, coroutine executing (in _tasks set)

    Thread Safety: NOT thread-safe. Call from event loop thread only.
    Performance: O(1) tracking via dict/set, ~1μs overhead per schedule.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop | None = None,
        exception_handler: Callable[[asyncio.Task], None] | None = None,
    ) -> None:
        """
        Args:
            loop: Event loop to use. If None, uses asyncio.get_running_loop().
                Must be called from within an async context if loop is not provided.
            exception_handler: Called when a task raises an unhandled exception.
        """
        self._loop: asyncio.AbstractEventLoop = loop or asyncio.get_running_loop()
        self._tasks: set[asyncio.Task] = set()
        # Maps handle ID → (handle, coroutine) so we can close() coroutines when handles are cancelled.
        # Use id(handle) as key to avoid recursion when handles contain circular refs (handle_container).
        self._handles: dict[
            HandleId, tuple[asyncio.TimerHandle | asyncio.Handle, Coroutine]
        ] = {}
        self._exception_handler: Callable[[asyncio.Task], None] | None = (
            exception_handler
        )

    def set_exception_handler(self, handler: Callable[[asyncio.Task], None]) -> None:
        """Set callback for unhandled task exceptions."""
        self._exception_handler = handler

    def _done_callback(self, task: asyncio.Task) -> None:
        """Remove completed task from tracking; invoke exception handler if failed."""
        self._tasks.discard(task)  # discard() is safe if already removed by cancel_all
        # Must check cancelled() first - calling exception() on a cancelled task raises CancelledError
        if (
            not task.cancelled()
            and task.exception() is not None
            and self._exception_handler is not None
        ):
            self._exception_handler(task)

    def _safe_callback(
        self,
        handle_container: list[asyncio.TimerHandle | asyncio.Handle],
        coro: Coroutine,
    ) -> None:
        """
        Timer callback: transition coroutine from pending to running.

        The handle_container is a single-element list [handle] used to solve a
        chicken-and-egg problem: we need to pass the handle to this callback,
        but the handle doesn't exist until call_later() returns. We pass a
        mutable list, then populate it after call_later() returns.
        """
        if handle_container[0] is not None:
            self._handles.pop(id(handle_container[0]), None)
        task = self._loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._done_callback)

    def _track_handle_and_return_id(
        self, handle: asyncio.TimerHandle | asyncio.Handle, coro: Coroutine
    ) -> HandleId:
        """Track a handle and coroutine and return the handle ID.

        We use the handle ID as the key to avoid recursion when handles contain circular refs (handle_container).
        Also, we pass the handle id back to the caller instead of the original handle to ensure they
        do not keep a reference to the handle, and do not attempt to cancel it without stopping the coroutine.
        Returning the handle ID forces the caller to use the cancel_handle_id() method to cancel the coroutine.
        """
        handle_id = id(handle)
        self._handles[handle_id] = (handle, coro)
        return handle_id

    def schedule_later(self, delay_sec: float, coro: Coroutine) -> HandleId:
        """
        Schedule coroutine after a relative delay (seconds).

        Internally, it calls loop.call_later() and returns the handle ID.

        Args:
            delay_sec: Seconds to wait. If <= 0, delegates to schedule_soon().
            coro: Coroutine object to execute.

        Returns:
            Handle for cancellation. Cancelled coroutines are properly closed.
        """
        if delay_sec <= 0:
            return self.schedule_soon(coro)

        handle_container = [None]
        handle = self._loop.call_later(
            delay_sec, self._safe_callback, handle_container, coro
        )
        handle_container[0] = handle
        return self._track_handle_and_return_id(handle, coro)

    def schedule_soon(self, coro: Coroutine) -> HandleId:
        """
        Schedule coroutine on the next event loop iteration (immediate).

        Internally, it calls loop.call_soon() and returns the handle ID.

        Returns:
            Handle for cancellation. Cancelled coroutines are properly closed.
        """
        handle_container = [None]
        handle = self._loop.call_soon(self._safe_callback, handle_container, coro)
        handle_container[0] = handle
        return self._track_handle_and_return_id(handle, coro)

    def schedule_at(self, loop_time: float, coro: Coroutine) -> HandleId:
        """
        Schedule coroutine at an absolute loop.time() timestamp.

        Internally, it calls loop.call_at() and returns the handle ID.

        Args:
            loop_time: Target time from loop.time().
            coro: Coroutine object to execute.

        Returns:
            Handle for cancellation. Cancelled coroutines are properly closed.
        """
        handle_container = [None]
        handle = self._loop.call_at(
            loop_time, self._safe_callback, handle_container, coro
        )
        handle_container[0] = handle
        return self._track_handle_and_return_id(handle, coro)

    def schedule_at_perf_sec(self, perf_sec: float, coro: Coroutine) -> HandleId:
        """
        Schedule coroutine at an absolute perf_counter seconds timestamp.

        Bridges between time.perf_counter() (monotonic seconds) and
        loop.time() (event loop's monotonic seconds). Useful for replaying
        traces or scheduling at precise absolute times.

        Args:
            perf_sec: Target time from time.perf_counter().
                      If in the past, delegates to schedule_soon().
            coro: Coroutine object to execute.

        Returns:
            Handle for cancellation. Cancelled coroutines are properly closed.

        Precision: Input is float seconds, uvloop timer resolution is ~1ms.
        """
        # Always sample loop.time() FIRST to avoid early firing bias.
        cur_loop_time, cur_perf_sec = self._loop.time(), time.perf_counter()
        offset_sec = perf_sec - cur_perf_sec

        if offset_sec <= 0:
            return self.schedule_soon(coro)
        return self.schedule_at(cur_loop_time + offset_sec, coro)

    def schedule_at_perf_ns(self, perf_time_ns: int, coro: Coroutine) -> HandleId:
        """
        Schedule coroutine at an absolute perf_counter_ns timestamp.

        Bridges between time.perf_counter_ns() (monotonic nanoseconds) and
        loop.time() (event loop's monotonic seconds). Useful for replaying
        traces or scheduling at precise absolute times.

        Args:
            perf_time_ns: Target time from time.perf_counter_ns().
                          If in the past, delegates to schedule_soon().
            coro: Coroutine object to execute.

        Returns:
            Handle for cancellation. Cancelled coroutines are properly closed.

        Precision: Input is nanoseconds, but uvloop timer resolution is ~1ms.
        """
        # Always sample loop.time() FIRST to avoid early firing bias.
        cur_loop_time, cur_perf_ns = self._loop.time(), time.perf_counter_ns()
        offset_sec = (perf_time_ns - cur_perf_ns) / NANOS_PER_SECOND

        if offset_sec <= 0:
            return self.schedule_soon(coro)
        return self.schedule_at(cur_loop_time + offset_sec, coro)

    def cancel_handle_id(self, handle_id: HandleId) -> bool:
        """Cancel a scheduled coroutine by its handle ID.

        Args:
            handle_id: The ID of the handle to cancel. This is the ID returned by the schedule_* methods.

        Returns:
            True if the coroutine was cancelled, False if it was not found.
        """
        handle_data = self._handles.pop(handle_id, None)
        if handle_data is None:
            return False
        handle, coro = handle_data
        handle.cancel()
        coro.close()
        return True

    def cancel_all_pending(self) -> None:
        """
        Cancel all pending timers and close their coroutines.

        Order: snapshot → clear dict → cancel handles → close coroutines.
        This prevents races with callbacks firing during iteration.
        """
        items = list(self._handles.values())
        self._handles.clear()

        # Cancel handles first, then close coroutines
        for handle, coro in items:
            handle.cancel()
            coro.close()

    def cancel_all_running(self) -> list[asyncio.Task]:
        """
        Cancel all running tasks.

        Returns cancelled tasks so caller can await cleanup::

            tasks = scheduler.cancel_all_running()
            await asyncio.gather(*tasks, return_exceptions=True)
        """
        tasks = list(self._tasks)
        self._tasks.clear()
        for task in tasks:
            task.cancel()
        return tasks

    def cancel_all(self) -> list[asyncio.Task]:
        """Cancel all pending timers and running tasks. Returns cancelled tasks."""
        self.cancel_all_pending()
        return self.cancel_all_running()

    @property
    def pending_count(self) -> int:
        """Scheduled timers not yet fired."""
        return len(self._handles)

    @property
    def running_count(self) -> int:
        """Tasks currently executing (timer fired, not yet complete)."""
        return len(self._tasks)

    def is_idle(self) -> bool:
        """True if no pending timers and no running tasks."""
        return not self._handles and not self._tasks

    def __repr__(self) -> str:
        return (
            f"LoopScheduler(pending={self.pending_count}, running={self.running_count})"
        )
