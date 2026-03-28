# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clock offset tracking for cross-machine time synchronization.

In Kubernetes deployments, TimingManager (controller pod) and Workers (worker pods)
run on different machines with potentially different clocks. This module tracks the
clock offset between them using credit timestamps as a synchronization signal.

Each credit carries ``issued_at_ns`` (controller wall clock). When the worker receives
the credit, it computes ``sample = T2 - T1`` where T2 is the worker's wall clock.
Because this is a one-way measurement, every sample includes network transit time
as positive bias: ``sample = clock_skew + network_transit``.

Minimum offset filtering (inspired by NTP's clock filter algorithm, RFC 5905) takes
the smallest sample in a sliding window. The minimum has the least network delay,
making it the closest approximation to the true clock skew.

Both the controller (CreditIssuer) and this tracker use a dual-clock bootstrap pattern:
capture ``time.time_ns()`` once at startup as a wall-clock anchor, then derive all
subsequent timestamps from ``time.perf_counter_ns()`` deltas. This makes both sides
immune to NTP step corrections during the benchmark while keeping timestamps in the
wall-clock domain for cross-machine comparison.

An optional pre-flight RTT measurement (ping/pong probes) establishes baseline
network latency at startup, allowing the offset to be decomposed into estimated
clock skew and network transit for diagnostics.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.monotonic_clock import MonotonicClock
from aiperf.credit.messages import TimePing, TimePong

# Type alias for the send callback provided by Worker
SendPingCallback = Callable[[TimePing], Awaitable[None]]


class ClockOffsetTracker:
    """Tracks clock offset between controller and worker using minimum offset filtering.

    Uses a sliding window of recent offset measurements and selects the minimum
    as the best estimate of clock skew. This rejects network jitter (which only
    adds positive bias) rather than averaging it in.

    Timestamps are derived from a wall-clock anchor captured once at initialization
    plus ``perf_counter_ns`` deltas, matching the pattern used by the controller's
    ``CreditIssuer``. This avoids sensitivity to NTP step corrections mid-benchmark.

    Optionally measures baseline RTT at startup via ping/pong probes to decompose
    the offset into estimated clock skew and network transit.

    To convert a worker timestamp to controller time::

        controller_time = worker_time - tracker.offset_ns

    Attributes:
        offset_ns: Current best-estimate offset in nanoseconds (None before first measurement).
        sample_count: Total number of offset measurements recorded.
        baseline_rtt_ns: Minimum RTT from pre-flight probes (None if not measured).
        estimated_one_way_ns: Half of baseline RTT (None if not measured).
    """

    __slots__ = (
        "_clock",
        "_logger",
        "_min_samples",
        "_pending_pong_future",
        "_window",
        "baseline_rtt_ns",
        "estimated_one_way_ns",
        "offset_ns",
        "sample_count",
    )

    def __init__(
        self,
        logger_name: str,
        window_size: int = 20,
        min_samples: int = 5,
    ) -> None:
        """Initialize the tracker.

        Captures a wall-clock anchor and perf_counter anchor at construction time.
        All subsequent ``now()`` calls derive wall-clock values from perf_counter
        deltas, making them monotonic and immune to NTP step corrections.

        Args:
            logger_name: Name for the AIPerfLogger (typically the worker's service_id).
            window_size: Number of recent samples to retain in the sliding window.
            min_samples: Minimum samples required before ``is_calibrated`` returns True.
        """
        self._logger = AIPerfLogger(f"{logger_name}.clock_offset")
        self._clock = MonotonicClock()
        self._window: deque[int] = deque(maxlen=window_size)
        self._min_samples = min_samples
        self.offset_ns: int | None = None
        self.sample_count: int = 0
        self.baseline_rtt_ns: int | None = None
        self.estimated_one_way_ns: int | None = None
        self._pending_pong_future: asyncio.Future[TimePong] | None = None

    # =========================================================================
    # Credit-based offset tracking
    # =========================================================================

    def update(self, issued_at_ns: int) -> int:
        """Record a new offset measurement from a credit timestamp.

        Args:
            issued_at_ns: Wall clock timestamp from the credit (controller time).

        Returns:
            The updated best-estimate offset in nanoseconds.
        """
        sample = self._clock.now_ns() - issued_at_ns
        self._window.append(sample)
        self.sample_count += 1
        self.offset_ns = min(self._window)
        return self.offset_ns

    @property
    def is_calibrated(self) -> bool:
        """True when enough samples have been collected for a reliable estimate."""
        return self.sample_count >= self._min_samples

    @property
    def offset_range_ns(self) -> int | None:
        """Spread between max and min samples in the window (jitter indicator).

        Returns None before any measurements.
        """
        if not self._window:
            return None
        return max(self._window) - min(self._window)

    @property
    def estimated_clock_skew_ns(self) -> int | None:
        """Estimated clock skew with network transit removed.

        Computed as ``offset_ns - estimated_one_way_ns``. Only available after
        both offset measurement and baseline RTT have been established.

        Returns None if either component is missing.
        """
        if self.offset_ns is None or self.estimated_one_way_ns is None:
            return None
        return self.offset_ns - self.estimated_one_way_ns

    def now_with_offset(self) -> tuple[int, int | None]:
        """Return the current monotonic wall-clock time and the offset used.

        Both values share the same clock read, so the offset is exactly the one
        that would be needed to correct this timestamp to controller time.

        Returns:
            (now_ns, offset_ns) where now_ns is from MonotonicClock and
            offset_ns is the current best-estimate (None before first measurement).
        """
        return self._clock.now_ns(), self.offset_ns

    def correct_timestamp(self, worker_timestamp_ns: int) -> int:
        """Convert a worker wall-clock timestamp to the controller's time frame.

        Args:
            worker_timestamp_ns: A wall-clock-domain timestamp from this worker.

        Returns:
            The timestamp adjusted to controller time. Returns the input unchanged
            if no offset has been measured yet.
        """
        if self.offset_ns is None:
            return worker_timestamp_ns
        return worker_timestamp_ns - self.offset_ns

    # =========================================================================
    # Pre-flight RTT measurement
    # =========================================================================

    def handle_pong(self, pong: TimePong) -> None:
        """Resolve a pending pong future from an incoming TimePong message.

        Called by the Worker's message handler when a TimePong arrives on the
        credit DEALER socket.

        Args:
            pong: The TimePong message received from the router.
        """
        if self._pending_pong_future and not self._pending_pong_future.done():
            self._pending_pong_future.set_result(pong)

    async def measure_baseline_rtt(
        self,
        send_ping: SendPingCallback,
        probe_count: int = 5,
        timeout: float = 5.0,
    ) -> None:
        """Measure baseline RTT on the credit channel via ping/pong probes.

        Sends ``probe_count`` TimePing messages through the provided callback
        and waits for TimePong responses (delivered via ``handle_pong``).
        The minimum RTT is stored as ``baseline_rtt_ns``.

        This should be called once during startup before WorkerDispatchable is sent.

        Args:
            send_ping: Async callable that sends a TimePing on the credit channel.
            probe_count: Number of ping/pong round trips to perform.
            timeout: Seconds to wait for each pong response.
        """
        rtts: list[int] = []
        loop = asyncio.get_running_loop()

        for seq in range(probe_count):
            self._pending_pong_future = loop.create_future()
            sent_at_perf_ns = time.perf_counter_ns()
            await send_ping(TimePing(sequence=seq, sent_at_ns=sent_at_perf_ns))
            try:
                await asyncio.wait_for(self._pending_pong_future, timeout=timeout)
                rtt = time.perf_counter_ns() - sent_at_perf_ns
                rtts.append(rtt)
            except TimeoutError:
                self._logger.warning(f"TimePing {seq} timed out")

        self._pending_pong_future = None

        if rtts:
            min_rtt = min(rtts)
            self.baseline_rtt_ns = min_rtt
            self.estimated_one_way_ns = min_rtt // 2
            self._logger.info(
                f"Baseline RTT: {min_rtt / 1e6:.2f}ms "
                f"(from {len(rtts)}/{probe_count} probes, "
                f"estimated one-way: {min_rtt / 2 / 1e6:.2f}ms)"
            )
        else:
            self._logger.warning(
                f"All {probe_count} RTT probes timed out, baseline RTT not established"
            )
