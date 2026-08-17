# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal time seam for the agent-graph replay plane.

The graph ``TraceExecutor`` replays a recorded trace by sleeping each node's
incoming firing-edge delay, and ``AgentGraphReplayStrategy`` paces whole traces
onto their recorded start instants. Driven by wall time, validating that either
reproduces the recorded timeline of a multi-hour capture would itself take
multiple hours. Driven by a virtual clock advanced by an external pump, the same
replay completes in milliseconds and is deterministic.

DELIBERATELY SMALL, AND PROVISIONAL
-----------------------------------
This is a local seam for the graph plane, not a general clock framework. It is
TWO methods (:meth:`Clock.perf_ns` / :meth:`Clock.sleep_ns`) plus
:class:`SimClock`'s ``next_event_time`` / ``advance_to`` / ``has_sleepers``
driving surface, and it stops there ON PURPOSE:

* **No driver.** There is no virtual-time event loop here, so ``asyncio.sleep``
  and ``LoopScheduler`` do NOT ride virtual time -- only code that explicitly
  awaits ``clock.sleep_ns`` does. Callers drive :class:`SimClock` themselves via
  ``next_event_time`` + ``advance_to``.
* **No unit conversions.** Integer nanoseconds only. Earlier revisions carried a
  seconds/milliseconds layer whose ``sleep_ms`` / ``sleep_until`` /
  ``sleep_until_ms`` members had no callers anywhere.
* **Timing plane only.** Transports, record timestamping, and readiness probes
  keep reading ``time.*`` directly, where real time is the correct answer.
* **Nothing speculative.** Every member here has a caller. A monotonic-origin
  anchor, an ``is_virtual`` predicate, a ``schedule``/waker seam and a
  ``sleep_until_ns`` overload were all dropped once measurement showed zero
  production callers -- keeping them "because the fuller port has them" would
  be copying a shape we have no need for yet.

The names and semantics of what IS here match the fuller clock port (``Clock`` /
``RealClock`` / ``SimClock``, ``perf_ns``, sync ``advance_to``,
``next_event_time`` clamped up to ``now``), so adopting that port later is an
import change rather than a redesign -- and it restores the dropped members for
free.

Usage:

.. code-block:: python

    from aiperf.common.clock import Clock, RealClock, SimClock

    async def pace(clock: Clock, interval_ns: int) -> None:
        while True:
            await clock.sleep_ns(interval_ns)
            do_one()

The default is a :class:`RealClock`; behavior is identical to reading ``time.*``
directly.
"""

from __future__ import annotations

import asyncio
import heapq
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "RealClock", "SimClock"]


@runtime_checkable
class Clock(Protocol):
    """Time source for the graph replay control flow.

    Integer nanoseconds throughout -- no seconds/millisecond layer. Callers that
    need other units convert at the boundary.

    Two implementations:

    * :class:`RealClock` -- ``time.perf_counter_ns()`` + ``asyncio.sleep()``.
      The default for live replay.
    * :class:`SimClock` -- virtual time advanced by an external driver. Sleepers
      park on a per-waiter ``asyncio.Event`` and wake when sim time crosses
      their deadline.

    Implementations are NOT required to be thread-safe: the control-flow path
    runs on a single asyncio event loop.
    """

    def perf_ns(self) -> int:
        """Return the current time in monotonic nanoseconds."""
        ...

    async def sleep_ns(self, duration_ns: int) -> None:
        """Sleep for ``duration_ns`` nanoseconds.

        ``duration_ns <= 0`` yields to the event loop exactly once (matching
        ``asyncio.sleep`` semantics) so a behind-schedule pacing loop cannot
        starve the loop.
        """
        ...


class RealClock:
    """Default real-time clock -- a thin pass-through to the standard library.

    Stateless; one instance can be shared across all consumers.

    The non-positive contract is inherited rather than reimplemented:
    ``asyncio.sleep`` itself yields exactly once for ``delay <= 0``, so a
    behind-schedule pacing loop cannot starve the event loop without this class
    branching on it.
    """

    __slots__ = ()

    def perf_ns(self) -> int:
        return time.perf_counter_ns()

    async def sleep_ns(self, duration_ns: int) -> None:
        await asyncio.sleep(duration_ns / 1e9)


@dataclass(slots=True, order=True)
class _Sleeper:
    """Heap entry for a sleeper parked on a :class:`SimClock`."""

    deadline_ns: int
    """Absolute sim-time deadline the sleeper waits for (primary heap key)."""

    seq_no: int
    """Monotonic tie-break so equal deadlines wake in registration order."""

    event: asyncio.Event = field(compare=False)
    """Per-waiter wake event set by ``advance_to`` when the deadline crosses."""

    alive: bool = field(default=True, compare=False)
    """False once the sleeper exited (woke or was cancelled). Dead entries are
    reaped lazily so a cancelled sleeper's phantom deadline cannot fast-forward
    sim time via ``next_event_time``."""


class SimClock:
    """Discrete-event virtual clock advanced by an external driver.

    :meth:`advance_to` is monotonic -- a call with ``ns <= perf_ns()`` never
    rewinds, though it still drains already-due sleepers so a zero-advance tick
    makes progress. Each parked sleeper owns its own ``asyncio.Event`` and is
    keyed in a ``(deadline, seq_no)`` min-heap, so wakes fire in strict
    ``(deadline, registration-order)`` priority rather than via
    ``notify_all`` -- which would let asyncio dispatch ready callbacks in
    arbitrary order and bake nondeterminism into any control flow that samples
    ``perf_ns()`` right after a wake.

    Cancellation: a cancelled sleeper marks its entry dead (``alive=False``);
    dead entries are skipped by :meth:`advance_to` and reaped lazily by
    :meth:`has_sleepers` / :meth:`next_event_time`, so a phantom deadline never
    fast-forwards sim time.

    Driving it: read :meth:`next_event_time` and call :meth:`advance_to` once
    the loop goes idle. Both are synchronous, so a driver can pump from a
    non-async context.
    """

    __slots__ = ("_now_ns", "_sleepers", "_seq")

    def __init__(self) -> None:
        self._now_ns: int = 0
        # Min-heap of _Sleeper(deadline_ns, seq_no, ...). The sequence number is
        # the secondary key so simultaneous-deadline parks wake in registration
        # order -- also deterministic.
        self._sleepers: list[_Sleeper] = []
        self._seq: int = 0

    def perf_ns(self) -> int:
        return self._now_ns

    def _reap_dead_head(self) -> None:
        """Pop cancelled sleepers' entries off the heap root.

        Safe without a lock: every heap mutation happens in a synchronous
        (no-await) section on the single event loop, so this reap cannot
        interleave with a mutation in flight.
        """
        while self._sleepers and not self._sleepers[0].alive:
            heapq.heappop(self._sleepers)

    def has_sleepers(self) -> bool:
        """Return True if at least one live sleeper is parked."""
        self._reap_dead_head()
        return bool(self._sleepers)

    def next_event_time(self) -> int | None:
        """The instant to fast-forward to, or ``None`` when idle.

        ``max(earliest live deadline, now)`` -- an already-due top sleeper
        yields ``now`` rather than a past time, so a driver never tries to
        advance backwards.
        """
        self._reap_dead_head()
        if not self._sleepers:
            return None
        head = self._sleepers[0].deadline_ns
        return head if head > self._now_ns else self._now_ns

    def advance_to(self, ns: int) -> None:
        """Advance sim time to ``ns`` and wake every sleeper now due.

        Monotonic: ``ns <= now`` does not rewind the cursor, but still drains
        already-due sleepers. Pops in ``(deadline, seq_no)`` order and sets each
        live waiter's ``Event``; asyncio then schedules the awaiting coroutines
        via ``call_soon`` in the same order, giving deterministic wake ordering.
        """
        if ns > self._now_ns:
            self._now_ns = ns
        while self._sleepers and self._sleepers[0].deadline_ns <= self._now_ns:
            sleeper = heapq.heappop(self._sleepers)
            if sleeper.alive:
                sleeper.event.set()

    async def sleep_ns(self, duration_ns: int) -> None:
        if duration_ns <= 0:
            # Yield once even for non-positive durations (asyncio.sleep
            # semantics) so zero-delay replay loops cannot starve the loop.
            await asyncio.sleep(0)
            return
        deadline_ns = self._now_ns + duration_ns
        self._seq += 1
        sleeper = _Sleeper(deadline_ns, self._seq, asyncio.Event())
        heapq.heappush(self._sleepers, sleeper)
        try:
            await sleeper.event.wait()
        finally:
            # Mark dead on ANY exit (wake or cancellation). A cancelled
            # sleeper's entry stays in the heap until lazily reaped, but a dead
            # entry never wakes and never counts as a pending deadline.
            sleeper.alive = False
