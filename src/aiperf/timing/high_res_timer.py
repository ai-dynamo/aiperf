# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-resolution absolute-deadline sleeps via Linux timerfd.

Event-loop timers (libuv under uvloop, epoll_wait under stock asyncio) have
~1ms granularity, so the sub-millisecond sleeps needed by high request rates
(e.g. 200us intervals at 5,000 req/s) reliably oversleep to the next timer
tick. ``timerfd_create(CLOCK_MONOTONIC)`` timers are backed by kernel hrtimers
instead: expirations are delivered as fd readability with ~50us wakeup
precision, and the event loop observes them through ``loop.add_reader`` — the
same fd path the ZMQ clients use — bypassing the loop's timer wheel entirely.

``time.perf_counter()`` is ``clock_gettime(CLOCK_MONOTONIC)`` on Linux
(asserted at construction), so perf_counter deadlines are passed directly to
``timerfd_settime(TFD_TIMER_ABSTIME)`` without clock conversion.

Example:
    pacer = TimerFdPacer()
    deadline = time.perf_counter() + 0.0002
    await pacer.sleep_until(deadline)  # wakes ~50us after deadline, not ~1ms
    pacer.close()

Linux-only: callers must gate on ``IS_LINUX`` and fall back to
``asyncio.sleep`` elsewhere.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import os
import threading
import time

_CLOCK_MONOTONIC = 1
_TFD_NONBLOCK = 0o4000
_TFD_CLOEXEC = 0o2000000
_TFD_TIMER_ABSTIME = 1
_NANOS_PER_SECOND = 1_000_000_000


class _Timespec(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]


class _Itimerspec(ctypes.Structure):
    _fields_ = [("it_interval", _Timespec), ("it_value", _Timespec)]


class TimerFdPacer:
    """Await absolute perf_counter deadlines with hrtimer (~50us) precision.

    Must be constructed inside a running event loop. Not thread-safe and
    supports one waiter at a time (the rate loop is a single coroutine).
    """

    def __init__(self) -> None:
        clock_impl = time.get_clock_info("perf_counter").implementation
        if "CLOCK_MONOTONIC" not in clock_impl:
            raise OSError(
                f"TimerFdPacer requires perf_counter to be CLOCK_MONOTONIC-based, "
                f"got {clock_impl!r}: timerfd deadlines would be on a different clock"
            )
        self._libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self._libc.timerfd_create.argtypes = [ctypes.c_int, ctypes.c_int]
        self._libc.timerfd_create.restype = ctypes.c_int
        self._libc.timerfd_settime.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_Itimerspec),
            ctypes.POINTER(_Itimerspec),
        ]
        self._libc.timerfd_settime.restype = ctypes.c_int
        fd = self._libc.timerfd_create(_CLOCK_MONOTONIC, _TFD_NONBLOCK | _TFD_CLOEXEC)
        if fd < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"timerfd_create failed: {os.strerror(errno)}")
        self._fd = fd
        self._loop = asyncio.get_running_loop()
        self._tick = asyncio.Event()
        self._loop.add_reader(self._fd, self._on_readable)
        self._closed = False

    def _on_readable(self) -> None:
        with contextlib.suppress(BlockingIOError):
            os.read(self._fd, 8)  # drain the expiration count
        self._tick.set()

    async def sleep_until(self, deadline_perf_s: float) -> None:
        """Sleep until an absolute ``time.perf_counter()`` deadline.

        Returns immediately if the deadline is already in the past (the kernel
        fires expired absolute timers right away).
        """
        deadline_ns = int(deadline_perf_s * _NANOS_PER_SECOND)
        spec = _Itimerspec()
        spec.it_value.tv_sec = deadline_ns // _NANOS_PER_SECOND
        spec.it_value.tv_nsec = deadline_ns % _NANOS_PER_SECOND or 1
        self._tick.clear()
        rc = self._libc.timerfd_settime(
            self._fd, _TFD_TIMER_ABSTIME, ctypes.byref(spec), None
        )
        if rc != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"timerfd_settime failed: {os.strerror(errno)}")
        await self._tick.wait()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.remove_reader(self._fd)
        os.close(self._fd)


class ThreadPacer:
    """Cross-platform fallback: absolute-deadline sleeps in a dedicated thread.

    ``time.sleep`` in a plain thread bypasses the event loop's ~1ms timer
    wheel entirely: CPython backs it with ``clock_nanosleep`` on POSIX
    (~50-100us wakeup precision, ``mach``-timer based on macOS) and
    high-resolution waitable timers on Windows (~0.5ms, Python 3.11+). The
    thread wakes the event loop via ``call_soon_threadsafe``, which every
    loop implementation supports — including the Windows proactor loop,
    where fd-reader integration (the timerfd approach) is unavailable.

    Same contract as :class:`TimerFdPacer`: construct inside a running event
    loop, one waiter at a time, ``close()`` when done.
    """

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._tick = asyncio.Event()
        self._deadline: float | None = None
        self._wakeup = threading.Event()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="aiperf-rate-pacer", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while True:
            self._wakeup.wait()
            self._wakeup.clear()
            if self._closed:
                return
            deadline = self._deadline
            if deadline is None:
                continue
            # Loop compensates for undersleep; oversleep is bounded by the
            # platform's thread-sleep precision, not the event-loop timer wheel.
            while not self._closed:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                time.sleep(remaining)
            # The loop may already be closing during teardown.
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._tick.set)

    async def sleep_until(self, deadline_perf_s: float) -> None:
        """Sleep until an absolute ``time.perf_counter()`` deadline.

        Returns immediately if the deadline is already in the past.
        """
        self._tick.clear()
        self._deadline = deadline_perf_s
        self._wakeup.set()
        await self._tick.wait()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wakeup.set()
