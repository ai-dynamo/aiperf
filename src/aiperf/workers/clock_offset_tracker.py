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

SendPingCallback = Callable[[TimePing], Awaitable[None]]
"""Async callback supplied by the Worker that puts a TimePing on the credit channel."""


class ClockOffsetTracker:
    """Tracks clock offset between controller and worker using minimum offset filtering.

    Uses a sliding window of recent offset measurements and selects the minimum
    as the best estimate of clock skew. This rejects network jitter (which only
    adds positive bias) rather than averaging it in.

    Min filtering is asymmetric under drift: a *falling* true offset is picked up on
    the very next sample, but a *rising* one is only tracked once the window fully
    evicts the stale low sample -- up to ``window_size`` credits. A worker receiving
    credits slowly therefore holds a stale-low offset for that period, which biases
    corrected latencies high. Sub-second at normal credit rates; shrink
    ``window_size`` if a deployment sees sustained one-directional drift.

    Timestamps are derived from a wall-clock anchor captured once at initialization
    plus ``perf_counter_ns`` deltas, matching the pattern used by the controller's
    ``CreditIssuer``. This avoids sensitivity to NTP step corrections mid-benchmark.

    To convert a worker timestamp to controller time::

        tracker = ClockOffsetTracker(logger_name="worker-7f2a")
        tracker.update(issued_at_ns=credit.issued_at_ns)
        controller_time_ns = tracker.correct_timestamp(worker_time_ns)

    Attributes:
        offset_ns: Current best-estimate offset in nanoseconds (None before first sample).
        sample_count: Total number of offset measurements recorded.
        baseline_rtt_ns: Minimum RTT from pre-flight probes (None if not measured).
        estimated_one_way_ns: Half of baseline RTT (None if not measured).
    """

    __slots__ = (
        "_clock",
        "_logger",
        "_min_samples",
        "_pending_pong_future",
        "_pending_pong_sequence",
        "_window",
        "baseline_rtt_ns",
        "estimated_one_way_ns",
        "offset_ns",
        "sample_count",
    )

    def __init__(
        self,
        logger_name: str = "aiperf.worker",
        window_size: int = 20,
        min_samples: int = 5,
    ) -> None:
        """Initialize the tracker.

        Captures a wall-clock anchor and perf_counter anchor at construction time.
        All subsequent clock reads derive wall-clock values from perf_counter deltas,
        making them monotonic and immune to NTP step corrections.

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
        self._pending_pong_sequence: int | None = None

    def _now_ns(self) -> int:
        """Current wall-clock-domain time, advanced monotonically from the anchors."""
        return self._clock.now_ns()

    # =========================================================================
    # Credit-based offset tracking
    # =========================================================================

    def observe(self, issued_at_ns: int, received_at_ns: int) -> int:
        """Record an offset measurement from an explicit pair of timestamps.

        Args:
            issued_at_ns: Wall-clock timestamp from the controller (credit issue time).
            received_at_ns: Wall-clock timestamp from this worker (credit receipt time).

        Returns:
            The updated best-estimate offset in nanoseconds.
        """
        self._window.append(received_at_ns - issued_at_ns)
        self.sample_count += 1
        self.offset_ns = min(self._window)
        return self.offset_ns

    def update(self, issued_at_ns: int) -> int:
        """Record a new offset measurement from a credit timestamp.

        Reads this worker's clock as the receive-side timestamp.

        Args:
            issued_at_ns: Wall clock timestamp from the credit (controller time).

        Returns:
            The updated best-estimate offset in nanoseconds.
        """
        return self.observe(issued_at_ns=issued_at_ns, received_at_ns=self._now_ns())

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
            (now_ns, offset_ns) where offset_ns is None before the first sample.
        """
        return self._now_ns(), self.offset_ns

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

        A pong whose sequence does not match the probe currently in flight is
        dropped: after a probe times out its reply may still arrive, and
        crediting it to the next probe would report an RTT far shorter than the
        real round trip, which then wins ``min(rtts)`` in the baseline.

        Args:
            pong: The TimePong message received from the router.
        """
        if self._pending_pong_sequence is None or pong.sequence != (
            self._pending_pong_sequence
        ):
            self._logger.debug(
                lambda: f"Ignoring TimePong {pong.sequence}, "
                f"awaiting {self._pending_pong_sequence}"
            )
            return
        if self._pending_pong_future and not self._pending_pong_future.done():
            self._pending_pong_future.set_result(pong)

    async def measure_baseline_rtt(
        self,
        send_ping: SendPingCallback,
        probe_count: int = 5,
        timeout: float = 5.0,
        max_attempts: int | None = None,
    ) -> None:
        """Measure baseline RTT on the credit channel via ping/pong probes.

        Sends TimePing messages through the provided callback and waits for
        TimePong responses (delivered via ``handle_pong``) until
        ``probe_count`` of them round-trip or ``max_attempts`` pings have been
        sent. The minimum RTT is stored as ``baseline_rtt_ns``.

        This should be called once during startup before the worker declares itself
        dispatchable, so that probes are not queued behind real credits.

        Timed-out probes are retried rather than consumed from the quota: on a
        real cluster the credit ROUTER is frequently not echoing yet when the
        worker container starts, so a fixed ``probe_count`` attempts with a long
        per-probe timeout burns the caller's whole budget before the router is
        reachable and the baseline is never measured. Each successful RTT is
        applied immediately, so a caller that cancels this coroutine on a budget
        expiry still keeps whatever was measured.

        Args:
            send_ping: Async callable that sends a TimePing on the credit channel.
            probe_count: Number of *successful* ping/pong round trips to collect.
            timeout: Seconds to wait for each pong response.
            max_attempts: Cap on pings sent. Defaults to ``probe_count``, i.e. no
                retries; callers that bound the whole sequence with their own
                deadline should pass a larger value to enable retries.
        """
        rtts: list[int] = []
        loop = asyncio.get_running_loop()
        attempts = probe_count if max_attempts is None else max_attempts

        try:
            for seq in range(attempts):
                if len(rtts) >= probe_count:
                    break
                self._pending_pong_future = loop.create_future()
                self._pending_pong_sequence = seq
                sent_at_perf_ns = time.perf_counter_ns()
                await send_ping(TimePing(sequence=seq, sent_at_ns=sent_at_perf_ns))
                try:
                    await asyncio.wait_for(self._pending_pong_future, timeout=timeout)
                except TimeoutError:
                    self._logger.warning(f"TimePing {seq} timed out")
                    continue
                rtts.append(time.perf_counter_ns() - sent_at_perf_ns)
                self._apply_baseline_rtt(rtts, probe_count)
        finally:
            self._pending_pong_future = None
            self._pending_pong_sequence = None

        if not rtts:
            self._logger.warning(
                f"All {attempts} RTT probes timed out, baseline RTT not established"
            )

    def _apply_baseline_rtt(self, rtts: list[int], probe_count: int) -> None:
        """Store the minimum observed RTT as the baseline and log it."""
        min_rtt = min(rtts)
        self.baseline_rtt_ns = min_rtt
        self.estimated_one_way_ns = min_rtt // 2
        self._logger.info(
            lambda: f"Baseline RTT: {min_rtt / 1e6:.2f}ms "
            f"(from {len(rtts)}/{probe_count} probes, "
            f"estimated one-way: {min_rtt / 2 / 1e6:.2f}ms)"
        )
