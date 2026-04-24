# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monotonic wall-clock timestamp source.

Captures ``time.time_ns()`` once as an anchor, then derives all subsequent
wall-clock timestamps from ``time.perf_counter_ns()`` deltas. This produces
timestamps that are:

- In the wall-clock domain (comparable across machines via shared Unix epoch)
- Monotonic (immune to NTP step corrections during a benchmark)
- High resolution (nanosecond, from perf_counter)

Used by both the controller (CreditIssuer) and workers (ClockOffsetTracker)
to ensure consistent, monotonic timestamps for cross-machine offset measurement.
"""

import time

from aiperf.common.constants import NANOS_PER_SECOND


class MonotonicClock:
    """Wall-clock source anchored to ``perf_counter`` for monotonicity.

    At construction, captures both ``time.time_ns()`` and ``time.perf_counter_ns()``.
    ``now_ns()`` then computes wall-clock time as::

        wall_anchor + (perf_counter_now - perf_anchor)

    This matches the dual-clock bootstrap pattern used throughout AIPerf's
    timing subsystem.
    """

    __slots__ = ("perf_anchor_ns", "wall_anchor_ns")

    def __init__(self) -> None:
        self.perf_anchor_ns, self.wall_anchor_ns = (
            time.perf_counter_ns(),
            time.time_ns(),
        )

    def now_ns(self) -> int:
        """Current wall-clock time derived from perf_counter delta."""
        return self.wall_anchor_ns + (time.perf_counter_ns() - self.perf_anchor_ns)

    def elapsed_ns(self) -> int:
        """Nanoseconds elapsed since this clock was created."""
        return time.perf_counter_ns() - self.perf_anchor_ns

    def elapsed_sec(self) -> float:
        """Seconds elapsed since this clock was created."""
        return self.elapsed_ns() / NANOS_PER_SECOND
