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
but it is still ``clock_skew + min_transit``, not the skew: min-filtering makes the
transit term SMALL, never zero.

That residual transit is removed separately. A pre-flight RTT measurement (ping/pong
probes echoed verbatim by the credit ROUTER) yields ``estimated_one_way_ns =
min_rtt // 2``, and ``correction_ns`` -- the value actually applied to timestamps --
is ``offset_ns - estimated_one_way_ns``. Without that second term the correction
shifts every worker timestamp EARLIER than true controller time by one one-way hop,
which directly understates ``credit_to_start_latency`` (``request_start_ns -
credit_issued_ns``), the very quantity that hop belongs to. On TCP loopback the
uncorrected bias measures ~17us against a ~20us one-way estimate; cross-pod in a
cluster both scale by two to three orders of magnitude, to the same order as the
metric itself.

The estimate assumes path symmetry and is measured on an idle channel at startup,
so it is not exact -- it slightly overshoots when router turnaround dominates and
undershoots under load. It is applied anyway because the uncorrected value carries a
guaranteed one-directional bias, whereas the residual is smaller and unsigned. When
no baseline RTT was established (all probes timed out), ``correction_ns`` falls back
to the raw ``offset_ns`` -- the transit term is then unknown, not zero.

Both the controller (CreditIssuer) and this tracker use a dual-clock bootstrap pattern:
capture ``time.time_ns()`` once at startup as a wall-clock anchor, then derive all
subsequent timestamps from ``time.perf_counter_ns()`` deltas. This makes both sides
immune to NTP step corrections during the benchmark while keeping timestamps in the
wall-clock domain for cross-machine comparison.
"""

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable, Sequence

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.environment import Environment
from aiperf.common.monotonic_clock import MonotonicClock
from aiperf.credit.messages import TimePing, TimePong

SendPingCallback = Callable[[TimePing], Awaitable[None]]
"""Async callback supplied by the Worker that puts a TimePing on the credit channel."""


def _median(values: Sequence[int]) -> float:
    """Median of a non-empty sequence, without pulling in statistics/numpy."""
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


class ClockOffsetTracker:
    """Tracks clock offset between controller and worker using minimum offset filtering.

    Uses a sliding window of recent offset measurements and selects the minimum
    as the best estimate of ``clock_skew + one_way_transit``. This rejects network
    jitter (which only adds positive bias) rather than averaging it in. The
    surviving transit term is then removed using the pre-flight baseline RTT; see
    ``correction_ns``, which is the value callers should apply.

    Min filtering is asymmetric under drift: a *falling* true offset is picked up on
    the very next sample, but a *rising* one is only tracked once the window fully
    evicts the stale low sample -- up to ``window_size`` credits. A worker receiving
    credits slowly therefore holds a stale-low offset for that period, which biases
    corrected latencies high. Sub-second at normal credit rates; shrink
    ``window_size`` if a deployment sees sustained one-directional drift.

    That same asymmetry is why samples are screened before they enter the window:
    a minimum estimator gives a single spuriously low sample authority over the
    correction until the window evicts it. Two gates run in ``observe``: an
    absolute plausibility bound, and rejection of samples sitting more than
    ``outlier_factor`` median absolute deviations below the window's median. A
    run of ``reset_after_rejects`` consecutive rejections is read as a real
    downward step (controller restart, NTP step) and re-seeds the window.

    Timestamps are derived from a wall-clock anchor captured once at initialization
    plus ``perf_counter_ns`` deltas, matching the pattern used by the controller's
    ``CreditIssuer``. This avoids sensitivity to NTP step corrections mid-benchmark.

    ``measure_baseline_rtt`` is re-entrant: the Worker re-runs it periodically so
    the one-way transit estimate follows a path that changed mid-run instead of
    being frozen at the startup value for the rest of the benchmark.

    To convert a worker timestamp to controller time::

        tracker = ClockOffsetTracker(logger_name="worker-7f2a")
        tracker.update(issued_at_ns=credit.issued_at_ns)
        controller_time_ns = tracker.correct_timestamp(worker_time_ns)

    Attributes:
        offset_ns: Raw min-filtered sample, ``clock_skew + min_transit`` (None
            before first sample). Use ``correction_ns`` to correct a timestamp.
        sample_count: Number of offset measurements admitted to the window.
        rejected_sample_count: Number of samples discarded by the sanity gates.
        baseline_rtt_ns: Minimum RTT from the most recent probe round (None if
            never measured).
        baseline_measurement_count: Completed probe rounds that produced an RTT.
        estimated_one_way_ns: Half of baseline RTT (None if not measured).
    """

    __slots__ = (
        "_clock",
        "_consecutive_rejects",
        "_logger",
        "_max_abs_ns",
        "_max_rtt_ns",
        "_min_samples",
        "_outlier_factor",
        "_outlier_floor_ns",
        "_next_ping_sequence",
        "_pending_pong_future",
        "_pending_pong_sequence",
        "_reset_after_rejects",
        "_window",
        "baseline_measurement_count",
        "baseline_rtt_ns",
        "estimated_one_way_ns",
        "offset_ns",
        "rejected_sample_count",
        "sample_count",
    )

    def __init__(
        self,
        logger_name: str = "aiperf.worker",
        window_size: int = Environment.WORKER.CLOCK_OFFSET_WINDOW_SIZE,
        min_samples: int = Environment.WORKER.CLOCK_OFFSET_MIN_SAMPLES,
        max_abs_sec: float = Environment.WORKER.CLOCK_OFFSET_MAX_ABS_SEC,
        outlier_factor: float = Environment.WORKER.CLOCK_OFFSET_OUTLIER_FACTOR,
        outlier_floor_sec: float = Environment.WORKER.CLOCK_OFFSET_OUTLIER_FLOOR_SEC,
        reset_after_rejects: int = Environment.WORKER.CLOCK_OFFSET_RESET_AFTER_REJECTS,
        max_rtt_sec: float = Environment.WORKER.CLOCK_PROBE_MAX_RTT_SEC,
    ) -> None:
        """Initialize the tracker.

        Captures a wall-clock anchor and perf_counter anchor at construction time.
        All subsequent clock reads derive wall-clock values from perf_counter deltas,
        making them monotonic and immune to NTP step corrections.

        Args:
            logger_name: Name for the AIPerfLogger (typically the worker's service_id).
            window_size: Number of recent samples to retain in the sliding window.
            min_samples: Minimum samples required before ``is_calibrated`` returns True,
                and before low-outlier rejection has a population to judge against.
            max_abs_sec: Absolute plausibility bound on a single sample, in seconds.
                0 disables the bound.
            outlier_factor: Multiplier on the window's median absolute deviation that
                sets the low-outlier rejection band. 0 disables rejection.
            outlier_floor_sec: Floor on that band, in seconds, so a window of
                near-identical samples does not reject everything that follows.
            reset_after_rejects: Consecutive low-outlier rejections after which the
                window is cleared and re-seeded from the rejected sample.
            max_rtt_sec: Plausibility bound on a single ping/pong round trip, in
                seconds. 0 disables the bound.
        """
        self._logger = AIPerfLogger(f"{logger_name}.clock_offset")
        self._clock = MonotonicClock()
        self._window: deque[int] = deque(maxlen=window_size)
        self._min_samples = min_samples
        self._max_abs_ns = int(max_abs_sec * NANOS_PER_SECOND)
        self._outlier_factor = outlier_factor
        self._outlier_floor_ns = int(outlier_floor_sec * NANOS_PER_SECOND)
        self._reset_after_rejects = reset_after_rejects
        self._max_rtt_ns = int(max_rtt_sec * NANOS_PER_SECOND)
        self._consecutive_rejects = 0
        self.offset_ns: int | None = None
        self.sample_count: int = 0
        self.rejected_sample_count: int = 0
        self.baseline_rtt_ns: int | None = None
        self.baseline_measurement_count: int = 0
        self.estimated_one_way_ns: int | None = None
        self._pending_pong_future: asyncio.Future[TimePong] | None = None
        self._pending_pong_sequence: int | None = None
        self._next_ping_sequence: int = 0

    def _now_ns(self) -> int:
        """Current wall-clock-domain time, advanced monotonically from the anchors."""
        return self._clock.now_ns()

    # =========================================================================
    # Credit-based offset tracking
    # =========================================================================

    def observe(self, issued_at_ns: int, received_at_ns: int) -> int | None:
        """Record an offset measurement from an explicit pair of timestamps.

        The sample is admitted only if it survives two sanity gates: an absolute
        plausibility bound, and rejection as a low outlier against the current
        window. Low outliers matter and high ones do not, because the estimator
        is a minimum: one spuriously low sample would otherwise own the
        correction until the window evicted it, whereas a high one is ignored
        for free. See ``_is_low_outlier`` for how a genuine downward step is
        distinguished from a run of bad samples.

        Args:
            issued_at_ns: Wall-clock timestamp from the controller (credit issue time).
            received_at_ns: Wall-clock timestamp from this worker (credit receipt time).

        Returns:
            The updated raw min-filtered offset in nanoseconds, or the previous
            one when the sample was rejected (None before any sample was
            admitted). This still includes one-way transit; see ``correction_ns``
            for the value to apply to a timestamp.
        """
        sample = received_at_ns - issued_at_ns
        if not self._accept_sample(sample):
            self.rejected_sample_count += 1
            return self.offset_ns
        self._window.append(sample)
        self.sample_count += 1
        self.offset_ns = min(self._window)
        return self.offset_ns

    def _accept_sample(self, sample: int) -> bool:
        """Decide whether a raw offset sample may enter the window."""
        if self._max_abs_ns and abs(sample) > self._max_abs_ns:
            self._logger.warning(
                lambda: f"Discarding implausible clock-offset sample "
                f"{sample / 1e6:.3f}ms (bound: "
                f"{self._max_abs_ns / 1e6:.3f}ms)"
            )
            return False
        if not self._is_low_outlier(sample):
            self._consecutive_rejects = 0
            return True
        self._consecutive_rejects += 1
        if self._consecutive_rejects < self._reset_after_rejects:
            return False
        # A sustained run of low outliers is not noise, it is the true offset
        # having stepped down (controller restart, NTP step). Drop the stale
        # window so the filter re-seeds from the new level immediately instead
        # of rejecting reality for the rest of the run.
        self._logger.warning(
            lambda: f"Clock offset stepped down past the outlier band for "
            f"{self._consecutive_rejects} consecutive samples; re-seeding "
            f"window at {sample / 1e6:.3f}ms"
        )
        self._window.clear()
        self._consecutive_rejects = 0
        return True

    def _is_low_outlier(self, sample: int) -> bool:
        """True when ``sample`` sits implausibly far below the window's median.

        Uses median absolute deviation rather than standard deviation because
        the sample distribution is right-skewed (transit only ever adds) and a
        single outlier would inflate a standard deviation enough to hide itself.
        """
        if not self._outlier_factor or len(self._window) < self._min_samples:
            return False
        median = _median(self._window)
        mad = _median([int(abs(value - median)) for value in self._window])
        band = max(self._outlier_factor * mad, self._outlier_floor_ns)
        return sample < median - band

    def update(self, issued_at_ns: int) -> int | None:
        """Record a new offset measurement from a credit timestamp.

        Reads this worker's clock as the receive-side timestamp.

        Args:
            issued_at_ns: Wall clock timestamp from the credit (controller time).

        Returns:
            The updated raw min-filtered offset in nanoseconds (see ``observe``).
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

    @property
    def correction_ns(self) -> int | None:
        """The offset to SUBTRACT from a worker timestamp to reach controller time.

        This is the only value that should be applied to a timestamp or stamped
        onto a record. It is ``estimated_clock_skew_ns`` when a baseline RTT was
        measured, and the raw ``offset_ns`` otherwise.

        The distinction matters: ``offset_ns`` is ``clock_skew + min_transit``, so
        applying it directly biases corrected timestamps one one-way hop EARLIER
        than true controller time. Subtracting ``estimated_one_way_ns`` cancels
        that term to within the path-asymmetry error.

        Returns None before the first sample.
        """
        skew = self.estimated_clock_skew_ns
        return self.offset_ns if skew is None else skew

    def now_with_offset(self) -> tuple[int, int | None]:
        """Return the current monotonic wall-clock time and the correction used.

        Both values share the same clock read, so the correction is exactly the
        one that would be needed to convert this timestamp to controller time.

        Returns:
            (now_ns, correction_ns) where correction_ns is None before the first
            sample.
        """
        return self._now_ns(), self.correction_ns

    def correct_timestamp(self, worker_timestamp_ns: int) -> int:
        """Convert a worker wall-clock timestamp to the controller's time frame.

        Args:
            worker_timestamp_ns: A wall-clock-domain timestamp from this worker.

        Returns:
            The timestamp adjusted to controller time by subtracting
            ``correction_ns``. Returns the input unchanged if no offset has been
            measured yet.
        """
        correction = self.correction_ns
        if correction is None:
            return worker_timestamp_ns
        return worker_timestamp_ns - correction

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
        real round trip, which then wins ``min(rtts)`` in the baseline. Ping
        sequence numbers are monotonic across the tracker's whole lifetime
        (never reset per round), so a stale reply from a prior
        ``measure_baseline_rtt`` round can never numerically collide with the
        sequence the current round is awaiting.

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
        probe_count: int = Environment.WORKER.CLOCK_PROBE_COUNT,
        timeout: float = 5.0,
        max_attempts: int | None = None,
    ) -> None:
        """Measure baseline RTT on the credit channel via ping/pong probes.

        Sends TimePing messages through the provided callback and waits for
        TimePong responses (delivered via ``handle_pong``) until
        ``probe_count`` of them round-trip or ``max_attempts`` pings have been
        sent. The minimum RTT is stored as ``baseline_rtt_ns``.

        This should first be called during startup before the worker declares
        itself dispatchable, so that probes are not queued behind real credits.
        It is safe to call again during the run -- the Worker does so on the
        ``CLOCK_REMEASURE_INTERVAL`` cadence -- and each round replaces the
        baseline with its own minimum, so a path that changed is tracked rather
        than frozen at the startup value.

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
        previous_baseline = self.baseline_rtt_ns

        try:
            for _ in range(attempts):
                if len(rtts) >= probe_count:
                    break
                seq = self._next_ping_sequence
                self._next_ping_sequence += 1
                self._pending_pong_future = loop.create_future()
                self._pending_pong_sequence = seq
                sent_at_perf_ns = time.perf_counter_ns()
                await send_ping(TimePing(sequence=seq, sent_at_ns=sent_at_perf_ns))
                try:
                    await asyncio.wait_for(self._pending_pong_future, timeout=timeout)
                except TimeoutError:
                    self._logger.warning(f"TimePing {seq} timed out")
                    continue
                rtt = time.perf_counter_ns() - sent_at_perf_ns
                if self._max_rtt_ns and rtt > self._max_rtt_ns:
                    # Halving an inflated round trip produces an inflated one-way
                    # estimate, which over-corrects every exported timestamp. A
                    # probe that slow says more about queueing behind real credits
                    # or a GC pause than about the path, so it is not a sample.
                    self._logger.warning(
                        lambda rtt=rtt: f"Discarding implausible RTT probe "
                        f"{rtt / 1e6:.2f}ms (bound: {self._max_rtt_ns / 1e6:.2f}ms)"
                    )
                    continue
                rtts.append(rtt)
        except asyncio.CancelledError:
            # A caller that cancels on budget expiry still keeps whatever
            # round-tripped: unwinding past the apply below would throw away
            # probes that already succeeded and leave the baseline unmeasured.
            if rtts:
                self._apply_baseline_rtt(rtts, probe_count)
                self.baseline_measurement_count += 1
            raise
        finally:
            self._pending_pong_future = None
            self._pending_pong_sequence = None

        if not rtts:
            self._logger.warning(
                f"All {attempts} RTT probes timed out, baseline RTT not established"
            )
            return

        # Apply the baseline once per round, not after each probe.  If applied
        # mid-round, a re-measurement under load (first probe slow: 50ms) replaces
        # the startup baseline (400µs) immediately, stepping every subsequent
        # exported timestamp by ~24.8ms for the rest of the round.  Deferring to
        # here means the round's full min() is used, not the running partial min.
        self._apply_baseline_rtt(rtts, probe_count)
        self.baseline_measurement_count += 1
        min_rtt = min(rtts)
        if previous_baseline is not None and (
            min_rtt > 2 * previous_baseline or 2 * min_rtt < previous_baseline
        ):
            self._logger.warning(
                lambda: f"Baseline RTT changed materially: "
                f"{previous_baseline / 1e6:.2f}ms -> {min_rtt / 1e6:.2f}ms"
            )

    def _apply_baseline_rtt(self, rtts: list[int], probe_count: int) -> None:
        """Store the minimum observed RTT of this round as the baseline and log it.

        Each measurement round replaces the baseline outright rather than
        folding into the previous one, so a re-measurement tracks a path that
        genuinely changed. The cost is that a re-measurement racing real credit
        traffic can read higher than the idle startup probe; taking the minimum
        over the round's probes is what keeps that bounded.
        """
        min_rtt = min(rtts)
        self.baseline_rtt_ns = min_rtt
        self.estimated_one_way_ns = min_rtt // 2
        self._logger.info(
            lambda: f"Baseline RTT: {min_rtt / 1e6:.2f}ms "
            f"(from {len(rtts)}/{probe_count} probes, "
            f"estimated one-way: {min_rtt / 2 / 1e6:.2f}ms)"
        )
