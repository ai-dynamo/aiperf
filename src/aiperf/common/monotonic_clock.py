# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monotonic wall-clock timestamp source.

Captures ``time.time_ns()`` once as an anchor, then derives all subsequent
wall-clock timestamps from ``time.perf_counter_ns()`` deltas. This produces
timestamps that are:

- In the wall-clock domain (comparable across machines via shared Unix epoch)
- Monotonic (immune to NTP step corrections during a benchmark)
- High resolution (nanosecond, from perf_counter)

Used by the worker-side ClockOffsetTracker to ensure consistent, monotonic
timestamps for cross-machine offset measurement, and (via ``process_clock``,
anchored with ``MonotonicClock.calibrated``) for worker-side
``RequestRecord.timestamp_ns``. The controller (CreditIssuer) anchors its own
perf_counter baseline
inline rather than through this class.
"""

import threading
import time

from aiperf.common.constants import NANOS_PER_SECOND


class MonotonicClock:
    """Wall-clock source anchored to ``perf_counter`` for monotonicity.

    At construction, captures both ``time.time_ns()`` and ``time.perf_counter_ns()``.
    ``now_ns()`` then computes wall-clock time as::

        wall_anchor + (perf_counter_now - perf_anchor)

    This matches the dual-clock bootstrap pattern used throughout AIPerf's
    timing subsystem.

    Example:
        ```python
        clock = MonotonicClock()
        request_start_ns = clock.now_ns()
        ...
        elapsed = clock.elapsed_sec()
        ```
    """

    __slots__ = ("perf_anchor_ns", "wall_anchor_ns")
    perf_anchor_ns: int
    wall_anchor_ns: int

    def __init__(
        self, perf_anchor_ns: int | None = None, wall_anchor_ns: int | None = None
    ) -> None:
        if perf_anchor_ns is None and wall_anchor_ns is None:
            perf_anchor_ns, wall_anchor_ns = time.perf_counter_ns(), time.time_ns()
        elif perf_anchor_ns is None or wall_anchor_ns is None:
            raise ValueError("perf_anchor_ns and wall_anchor_ns must be given together")
        self.perf_anchor_ns, self.wall_anchor_ns = perf_anchor_ns, wall_anchor_ns

    def now_ns(self) -> int:
        """Current wall-clock time derived from perf_counter delta."""
        return self.wall_anchor_ns + (time.perf_counter_ns() - self.perf_anchor_ns)

    def wall_ns_at(self, perf_ns: int) -> int:
        """Wall-clock time of an instant already read as ``perf_counter_ns``."""
        return self.wall_anchor_ns + (perf_ns - self.perf_anchor_ns)

    def elapsed_ns(self) -> int:
        """Nanoseconds elapsed since the performance-counter anchor."""
        return time.perf_counter_ns() - self.perf_anchor_ns

    def elapsed_sec(self) -> float:
        """Seconds elapsed since the performance-counter anchor."""
        return self.elapsed_ns() / NANOS_PER_SECOND

    @classmethod
    def calibrated(cls, samples: int = 32) -> "MonotonicClock":
        """Anchor on the tightest of ``samples`` (perf, wall, perf) brackets.

        A single anchor pair is off by however long the process is preempted
        between its two reads, and that error then rides on every timestamp
        derived from the clock. Bracketing each wall read between two
        ``perf_counter`` reads bounds its error by half the bracket, and a
        preemption only widens the sample it lands in, so the tightest of a
        few dozen samples is accurate to tens of nanoseconds.
        """
        if samples < 1:
            raise ValueError(f"samples must be at least 1, got {samples}")
        brackets = []
        for _ in range(samples):
            before_ns = time.perf_counter_ns()
            wall_ns = time.time_ns()
            brackets.append((before_ns, wall_ns, time.perf_counter_ns()))
        before_ns, wall_ns, after_ns = min(brackets, key=lambda b: b[2] - b[0])
        return cls((before_ns + after_ns) // 2, wall_ns)


_process_clock: MonotonicClock | None = None
_process_clock_lock = threading.Lock()


def process_clock() -> MonotonicClock:
    """The one anchored clock for wall-clock timestamps taken in this process.

    Every request record a worker exports is compared against records from the
    same worker (SPAWN_JOIN ordering, per-session turn order), so all of them
    must come from one anchor. A fresh ``time.time_ns()`` per request shifts
    that request's exported start and end by however long the process was
    preempted between it and the paired ``perf_counter`` read, which on a
    loaded host reaches milliseconds. The lock keeps two threads racing the
    first call from creating two different anchors.
    """
    global _process_clock
    if _process_clock is None:
        with _process_clock_lock:
            if _process_clock is None:
                _process_clock = MonotonicClock.calibrated()
    return _process_clock
