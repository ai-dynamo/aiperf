# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time abstraction for AIPerf control-flow components.

AIPerf historically reads time directly via ``time.perf_counter_ns()`` /
``time.monotonic_ns()`` / ``asyncio.sleep()`` everywhere a load loop, arrival
pacer, ramp, timeout, or trace-replay firing gate consults the clock. That
ties the control flow to wall-clock time, which is correct for live HTTP
transports but undermines two things:

* **Virtual-time validation.** The graph ``TraceExecutor`` replays a recorded
  trace by sleeping each node's incoming firing-edge delay. Driven by wall
  time, validating that the executor reproduces the recorded timeline of a
  multi-hour agentic trace would itself take multiple hours. Driven by a
  virtual clock advanced by an external pump, the same replay completes in
  milliseconds and is deterministic.
* **Deterministic wake order.** Sleepers parked on a virtual clock must wake
  in a fixed, reproducible order when sim time crosses their deadlines, so a
  control flow that samples ``now_ns()`` right after a wake is reproducible.

The fix is a clock abstraction that every time-sensitive component consults
instead of reading ``time.*`` directly. In wall mode it defers to the standard
library (behavior unchanged); in virtual mode it is advanced by an external
driver.

Usage:

.. code-block:: python

    from aiperf.common.clock import AIPerfClock, WallClock, VirtualClock

    async def pace(clock: AIPerfClock, rate_hz: float) -> None:
        interval_ns = int(1e9 / rate_hz)
        while True:
            await clock.sleep_ns(interval_ns)
            do_one()

The default clock is a :class:`WallClock`; behavior is identical to reading
``time.*`` directly. A :class:`VirtualClock` is installed by validation
harnesses that pump ``advance_to`` to fast-forward sim time to the next parked
waiter whenever the event loop goes idle.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["AIPerfClock", "WallClock", "VirtualClock"]


@runtime_checkable
class AIPerfClock(Protocol):
    """Time source for AIPerf control flow.

    Three unit conventions on every clock -- pick the one that matches the
    call site so there is no per-call boilerplate conversion:

    * **Seconds** (default, no suffix): ``now()``, ``sleep(s)``,
      ``sleep_until(s)``. Float seconds.
    * **Nanoseconds** (``_ns`` suffix): ``now_ns()``, ``sleep_ns(ns)``,
      ``sleep_until_ns(ns)``. Int. Matches ``time.perf_counter_ns()`` /
      ``time.monotonic_ns()``; use when integrating with code that already
      passes ns.
    * **Milliseconds** (``_ms`` suffix): ``now_ms()``, ``sleep_ms(ms)``,
      ``sleep_until_ms(ms)``. Float. Useful when interfacing with timeout
      configs / SLA thresholds expressed in ms.

    Two concrete implementations:

    * :class:`WallClock` -- defers to ``time.perf_counter_ns()`` +
      ``asyncio.sleep()``. The default for live replay/transports.
    * :class:`VirtualClock` -- virtual clock; ``advance_to`` is called by an
      external driver pump to fast-forward sim time. Sleepers park on a
      per-waiter ``asyncio.Event`` and wake when sim time crosses their
      deadline.

    Implementations are NOT required to be thread-safe across the read
    methods because AIPerf's control-flow path runs on a single asyncio event
    loop.
    """

    # ----- Seconds (default) -----

    def now(self) -> float:
        """Return the current time in float seconds."""
        ...

    async def sleep(self, duration_s: float) -> None:
        """Sleep for ``duration_s`` seconds.

        ``duration_s <= 0`` returns immediately (matches ``asyncio.sleep``
        semantics).
        """
        ...

    async def sleep_until(self, deadline_s: float) -> None:
        """Sleep until the clock reaches ``deadline_s`` seconds (absolute)."""
        ...

    # ----- Nanoseconds -----

    def now_ns(self) -> int:
        """Return the current time in nanoseconds."""
        ...

    async def sleep_ns(self, duration_ns: int) -> None:
        """Sleep for ``duration_ns`` nanoseconds."""
        ...

    async def sleep_until_ns(self, deadline_ns: int) -> None:
        """Sleep until the clock reaches ``deadline_ns`` (absolute)."""
        ...

    # ----- Milliseconds -----

    def now_ms(self) -> float:
        """Return the current time in float milliseconds."""
        ...

    async def sleep_ms(self, duration_ms: float) -> None:
        """Sleep for ``duration_ms`` milliseconds."""
        ...

    async def sleep_until_ms(self, deadline_ms: float) -> None:
        """Sleep until the clock reaches ``deadline_ms`` (absolute)."""
        ...


class _UnitConversionsMixin:
    """Provide the seconds + ms layers in terms of the ns ones.

    Concrete clocks inherit from this so they only implement the ns layer.
    """

    __slots__ = ()

    # Seconds layer.
    def now(self) -> float:
        return self.now_ns() / 1_000_000_000  # type: ignore[attr-defined]

    async def sleep(self, duration_s: float) -> None:
        await self.sleep_ns(int(duration_s * 1_000_000_000))  # type: ignore[attr-defined]

    async def sleep_until(self, deadline_s: float) -> None:
        await self.sleep_until_ns(int(deadline_s * 1_000_000_000))  # type: ignore[attr-defined]

    # Milliseconds layer.
    def now_ms(self) -> float:
        return self.now_ns() / 1_000_000  # type: ignore[attr-defined]

    async def sleep_ms(self, duration_ms: float) -> None:
        await self.sleep_ns(int(duration_ms * 1_000_000))  # type: ignore[attr-defined]

    async def sleep_until_ms(self, deadline_ms: float) -> None:
        await self.sleep_until_ns(int(deadline_ms * 1_000_000))  # type: ignore[attr-defined]


class WallClock(_UnitConversionsMixin):
    """Default real-time clock -- defers to the standard library.

    Stateless; one instance can be shared across all consumers.
    """

    __slots__ = ()

    def now_ns(self) -> int:
        return time.perf_counter_ns()

    async def sleep_ns(self, duration_ns: int) -> None:
        if duration_ns <= 0:
            # Yield once even for non-positive durations (asyncio.sleep
            # semantics) so behind-schedule pacing loops cannot starve the
            # event loop.
            await asyncio.sleep(0)
            return
        await asyncio.sleep(duration_ns / 1e9)

    async def sleep_until_ns(self, deadline_ns: int) -> None:
        await self.sleep_ns(deadline_ns - self.now_ns())


@dataclass(slots=True, order=True)
class _ClockSleeper:
    """Heap entry for a sleeper parked on a :class:`VirtualClock`."""

    deadline_ns: int
    """Absolute sim-time deadline the sleeper waits for (primary heap key)."""

    insertion_id: int
    """Monotonic tie-break so equal deadlines wake in registration order."""

    event: asyncio.Event = field(compare=False)
    """Per-waiter wake event set by ``advance_to`` when the deadline crosses."""

    alive: bool = field(default=True, compare=False)
    """False once the sleeper exited (woke or was cancelled). Dead entries are
    reaped lazily so a cancelled sleeper's phantom deadline cannot fast-forward
    sim time via ``peek_min_waiter_ns``."""


class VirtualClock(_UnitConversionsMixin):
    """Virtual clock advanced by an external driver pump.

    ``advance_to(ns)`` is monotonic -- calls with ``ns <= now_ns()`` are
    silently ignored, matching the semantics of monotonically-progressing sim
    time. Each parked sleeper owns its own ``asyncio.Event`` and is keyed in a
    ``(deadline, insertion_id)`` min-heap, so ``advance_to`` wakes sleepers in
    strict ``(deadline, registration-order)`` priority rather than via
    ``Condition.notify_all`` -- which would let asyncio dispatch ready
    callbacks in arbitrary order and bake nondeterminism into any control flow
    that samples ``now_ns()`` right after a wake.

    Waiter tracking: every parked sleeper registers a
    :class:`_ClockSleeper` ``(deadline_ns, insertion_id, event, alive)`` in the
    heap. ``advance_to`` pops every entry with ``deadline <= ns`` in heap
    order, setting each live entry's event in turn -- the asyncio loop then
    schedules the awaiting coroutines via ``call_soon`` in the same order.
    Deterministic wake order downstream. A cancelled sleeper marks its entry
    dead (``alive=False``); dead entries are skipped by ``advance_to`` and
    reaped lazily by ``has_waiters`` / ``peek_min_waiter_ns`` so phantom
    deadlines never fast-forward sim time.

    A driver pump fast-forwards sim time when the event loop goes idle: it
    reads :meth:`peek_min_waiter_ns` and calls :meth:`advance_to`. Use
    :meth:`set_on_waiter_parked` to make idle handling level-triggered (the
    callback fires the instant a fresh waiter parks, so the pump never drops a
    kick).
    """

    __slots__ = (
        "_now_ns",
        "_lock",
        "_waiters",
        "_insertion_counter",
        "_on_waiter_parked",
    )

    def __init__(self) -> None:
        self._now_ns: int = 0
        # Protects the waiter heap. ``asyncio.Lock`` (not Condition) -- we use
        # per-waiter ``asyncio.Event`` for wake instead of a shared condvar so
        # wake order is heap-priority deterministic.
        self._lock: asyncio.Lock = asyncio.Lock()
        # Min-heap of _ClockSleeper(deadline_ns, insertion_id, ...). The insertion
        # id is the secondary key so simultaneous-deadline parks wake in
        # registration order -- also deterministic.
        self._waiters: list[_ClockSleeper] = []
        self._insertion_counter: int = 0
        # Optional sync callback invoked right after a new waiter's entry is
        # pushed onto the heap. Lets a driver react when a fresh waiter parks
        # (e.g., to fast-forward sim time if the pump is currently idle and the
        # kick would otherwise be dropped).
        self._on_waiter_parked: Callable[[], None] | None = None

    def set_on_waiter_parked(self, cb: Callable[[], None] | None) -> None:
        """Register a callback fired right after a new waiter is parked.

        Called synchronously from ``sleep_until_ns`` while the clock's
        ``_lock`` is held. The callback MUST NOT await, MUST NOT acquire
        ``self._lock``, and should complete in microseconds -- its purpose is
        to flip a flag or set an ``asyncio.Event`` the driver pump observes.
        Pass ``None`` to clear.
        """
        self._on_waiter_parked = cb

    def now_ns(self) -> int:
        return self._now_ns

    def _reap_dead_head(self) -> None:
        """Pop cancelled sleepers' entries off the heap root.

        Safe without the lock: every heap mutation happens in a synchronous
        (no-await) section on the single event loop, so this sync reap cannot
        interleave with a mutation in flight.
        """
        while self._waiters and not self._waiters[0].alive:
            heapq.heappop(self._waiters)

    def has_waiters(self) -> bool:
        """Return True if at least one live sleeper is parked."""
        self._reap_dead_head()
        return bool(self._waiters)

    def peek_min_waiter_ns(self) -> int | None:
        """Return the earliest live parked waiter's deadline, or None if empty.

        Lock-free read of the heap root (dead-head entries from cancelled
        sleepers are reaped first so a phantom deadline never fast-forwards
        sim time). ``advance_to`` reaps crossed entries under the lock; in the
        rare race where this peek sees a stale root, the worst case is a no-op
        fast-forward -- the next ``advance_to`` will set the now-crossed
        entry's event either way.
        """
        self._reap_dead_head()
        if not self._waiters:
            return None
        head_deadline = self._waiters[0].deadline_ns
        return head_deadline if head_deadline > self._now_ns else None

    async def advance_to(self, ns: int) -> None:
        """Advance sim time to ``ns`` (no-op if ``ns <= now``).

        Pops every waiter with ``deadline <= ns`` in (deadline, insertion_id)
        order and sets each waiter's individual ``Event``. asyncio schedules
        the awaiting coroutines via ``call_soon`` in the same order they were
        set, giving deterministic wake-up ordering for sleepers that crossed
        their deadline on the same advance.
        """
        async with self._lock:
            if ns <= self._now_ns:
                return
            self._now_ns = ns
            while self._waiters and self._waiters[0].deadline_ns <= ns:
                waiter = heapq.heappop(self._waiters)
                if waiter.alive:
                    waiter.event.set()

    async def sleep_ns(self, duration_ns: int) -> None:
        if duration_ns <= 0:
            # Yield once even for non-positive durations (asyncio.sleep
            # semantics) so zero-delay replay loops cannot starve the loop.
            await asyncio.sleep(0)
            return
        await self.sleep_until_ns(self._now_ns + duration_ns)

    async def sleep_until_ns(self, deadline_ns: int) -> None:
        # Fast path: already crossed. Still yield once (asyncio.sleep
        # semantics) so crossed-deadline loops cannot starve the event loop.
        if self._now_ns >= deadline_ns:
            await asyncio.sleep(0)
            return
        waiter: _ClockSleeper | None = None
        async with self._lock:
            # Re-check under the lock; advance_to could have run between the
            # fast path and acquiring the lock.
            if self._now_ns < deadline_ns:
                self._insertion_counter += 1
                waiter = _ClockSleeper(
                    deadline_ns, self._insertion_counter, asyncio.Event()
                )
                heapq.heappush(self._waiters, waiter)
                # Synchronous notification -- driver pumps observe a fresh
                # waiter without an extra round-trip. Must not await or
                # acquire self._lock.
                if self._on_waiter_parked is not None:
                    self._on_waiter_parked()
        if waiter is None:
            await asyncio.sleep(0)
            return
        try:
            # Wait outside the lock. ``advance_to`` will set this event in
            # (deadline, insertion_id) priority order.
            await waiter.event.wait()
        finally:
            # Mark dead on ANY exit (wake or cancellation). A cancelled
            # sleeper's entry stays in the heap until lazily reaped, but a
            # dead entry never wakes and never counts as a pending deadline.
            waiter.alive = False
