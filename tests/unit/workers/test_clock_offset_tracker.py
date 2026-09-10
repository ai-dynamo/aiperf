# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cross-machine clock-offset tracking.

The estimator is a minimum-sample filter over a sliding window (NTP clock-filter
style, RFC 5905): every one-way sample carries network transit as *positive*
bias, so the smallest sample in the window is the smallest ``skew + transit``
seen. These tests pin that behavior, including its asymmetry -- positive
outliers are rejected outright, negative ones are adopted.

They also pin the second stage: min filtering shrinks the transit term but never
removes it, so ``correction_ns`` -- the only value that should ever be applied to
a timestamp -- subtracts the pre-flight one-way RTT estimate on top. ``offset_ns``
stays the raw sample.
"""

import asyncio

import pytest

from aiperf.common.environment import Environment
from aiperf.credit.messages import TimePing, TimePong
from aiperf.workers import clock_offset_tracker
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker


def test_default_filter_dimensions_use_environment_settings():
    tracker = ClockOffsetTracker()

    assert tracker._window.maxlen == Environment.WORKER.CLOCK_OFFSET_WINDOW_SIZE
    assert tracker._min_samples == Environment.WORKER.CLOCK_OFFSET_MIN_SAMPLES


def test_offset_is_none_before_any_sample():
    tracker = ClockOffsetTracker()
    assert tracker.offset_ns is None
    assert tracker.sample_count == 0
    assert tracker.offset_range_ns is None
    assert not tracker.is_calibrated


def test_offset_tracks_a_constant_skew():
    tracker = ClockOffsetTracker()
    for i in range(10):
        tracker.observe(issued_at_ns=1_000 + i, received_at_ns=6_000 + i)
    assert tracker.offset_ns == 5_000
    assert tracker.sample_count == 10
    assert tracker.is_calibrated


def test_positive_outlier_sample_does_not_dominate():
    tracker = ClockOffsetTracker()
    for i in range(20):
        tracker.observe(issued_at_ns=1_000 + i, received_at_ns=6_000 + i)
    tracker.observe(issued_at_ns=2_000, received_at_ns=900_000)
    # Minimum filtering rejects the transit-inflated sample entirely.
    assert tracker.offset_ns == 5_000


def test_negative_outlier_is_adopted_because_estimator_is_a_minimum():
    """A sample *below* the running minimum is taken as the new estimate.

    This is the defining asymmetry of min filtering and the reason it is only
    valid when network transit can bias samples in one direction.
    """
    tracker = ClockOffsetTracker()
    for _ in range(5):
        tracker.observe(issued_at_ns=1_000, received_at_ns=6_000)
    tracker.observe(issued_at_ns=1_000, received_at_ns=4_500)
    assert tracker.offset_ns == 3_500


def test_window_evicts_old_minimum():
    tracker = ClockOffsetTracker(window_size=3)
    tracker.observe(issued_at_ns=0, received_at_ns=1_000)
    for _ in range(3):
        tracker.observe(issued_at_ns=0, received_at_ns=5_000)
    assert tracker.offset_ns == 5_000


def test_offset_range_reports_window_jitter():
    tracker = ClockOffsetTracker()
    tracker.observe(issued_at_ns=0, received_at_ns=5_000)
    tracker.observe(issued_at_ns=0, received_at_ns=9_000)
    assert tracker.offset_range_ns == 4_000


def test_is_calibrated_requires_min_samples():
    tracker = ClockOffsetTracker(min_samples=3)
    tracker.observe(issued_at_ns=0, received_at_ns=1)
    assert not tracker.is_calibrated
    tracker.observe(issued_at_ns=0, received_at_ns=1)
    tracker.observe(issued_at_ns=0, received_at_ns=1)
    assert tracker.is_calibrated


def test_correct_timestamp_subtracts_offset():
    tracker = ClockOffsetTracker()
    assert tracker.correct_timestamp(1_234) == 1_234
    tracker.observe(issued_at_ns=1_000, received_at_ns=6_000)
    assert tracker.correct_timestamp(10_000) == 5_000


def test_correction_ns_is_none_before_any_sample():
    tracker = ClockOffsetTracker()
    assert tracker.correction_ns is None
    # A baseline RTT alone is not a correction; there is nothing to correct yet.
    tracker.baseline_rtt_ns = 2_000
    tracker.estimated_one_way_ns = 1_000
    assert tracker.correction_ns is None


def test_correction_ns_falls_back_to_raw_offset_without_a_baseline_rtt():
    """No RTT baseline means the transit term is unknown, not zero.

    All probes timing out is a supported outcome (the worker announces readiness
    anyway), so the correction must degrade to the raw sample rather than to
    None -- a skew of seconds uncorrected is far worse than one left carrying a
    sub-millisecond transit term.
    """
    tracker = ClockOffsetTracker()
    tracker.observe(issued_at_ns=1_000, received_at_ns=6_000)
    assert tracker.estimated_one_way_ns is None
    assert tracker.correction_ns == tracker.offset_ns == 5_000


def test_correction_ns_removes_the_estimated_one_way_transit():
    """The applied correction is skew, not skew-plus-transit.

    Regression: the worker used to stamp the raw ``offset_ns`` onto every
    record. Because a one-way sample is ``skew + transit``, that shifted every
    corrected timestamp one network hop EARLIER than true controller time, which
    subtracts a whole delivery hop from ``credit_to_start_latency`` -- the
    metric whose entire purpose is to measure that hop plus queueing.
    """
    tracker = ClockOffsetTracker()
    tracker.observe(issued_at_ns=1_000, received_at_ns=6_000)
    tracker.baseline_rtt_ns = 2_000
    tracker.estimated_one_way_ns = 1_000

    assert tracker.offset_ns == 5_000
    assert tracker.correction_ns == 4_000
    assert tracker.correction_ns == tracker.estimated_clock_skew_ns


def test_correct_timestamp_recovers_true_controller_time_under_symmetric_transit():
    """With a symmetric path the residual bias is zero, not one hop.

    Ground truth here is exact: the worker clock leads by ``skew``, every credit
    takes at least ``transit``, and the RTT probe sees ``2 * transit``. A worker
    instant of ``W`` is controller instant ``W - skew``; the estimator must land
    on that, not on ``W - skew - transit``.
    """
    tracker = ClockOffsetTracker()
    skew_ns = 5_000_000
    transit_ns = 400_000

    issued = 1_000_000_000
    for extra_delay_ns in (0, 250_000, 3_000_000):
        tracker.observe(
            issued_at_ns=issued,
            received_at_ns=issued + skew_ns + transit_ns + extra_delay_ns,
        )
    tracker.baseline_rtt_ns = 2 * transit_ns
    tracker.estimated_one_way_ns = transit_ns

    worker_instant_ns = 2_000_000_000 + skew_ns
    assert tracker.correct_timestamp(worker_instant_ns) == 2_000_000_000

    # And the size of the bug the correction removes, in the same units.
    tracker.estimated_one_way_ns = None
    assert tracker.correct_timestamp(worker_instant_ns) == 2_000_000_000 - transit_ns


def test_now_with_offset_reports_the_correction_that_would_be_applied():
    """The second element must be the applied correction, not the raw sample.

    They differ once a baseline RTT exists, and a caller that pairs the returned
    timestamp with the raw offset would reintroduce the transit bias.
    """
    tracker = ClockOffsetTracker()
    tracker.observe(issued_at_ns=1_000, received_at_ns=6_000)
    tracker.baseline_rtt_ns = 2_000
    tracker.estimated_one_way_ns = 1_000

    now_ns, correction = tracker.now_with_offset()
    assert correction == 4_000
    assert now_ns - correction == tracker.correct_timestamp(now_ns)


def test_update_uses_the_trackers_own_clock():
    tracker = ClockOffsetTracker()
    now_ns, _ = tracker.now_with_offset()
    offset = tracker.update(issued_at_ns=now_ns - 1_000_000)
    # ~1ms skew plus whatever elapsed between the two clock reads.
    assert 1_000_000 <= offset < 1_000_000_000


def test_estimated_clock_skew_requires_baseline_rtt():
    tracker = ClockOffsetTracker()
    tracker.observe(issued_at_ns=0, received_at_ns=5_000)
    assert tracker.estimated_clock_skew_ns is None
    tracker.baseline_rtt_ns = 2_000
    tracker.estimated_one_way_ns = 1_000
    assert tracker.estimated_clock_skew_ns == 4_000


@pytest.mark.asyncio
async def test_measure_baseline_rtt_uses_minimum_round_trip():
    tracker = ClockOffsetTracker()
    sent: list[TimePing] = []

    async def send_ping(ping: TimePing) -> None:
        sent.append(ping)
        tracker.handle_pong(
            TimePong(sequence=ping.sequence, sent_at_ns=ping.sent_at_ns)
        )

    await tracker.measure_baseline_rtt(send_ping, probe_count=3)

    assert [p.sequence for p in sent] == [0, 1, 2]
    assert tracker.baseline_rtt_ns is not None
    assert tracker.estimated_one_way_ns == tracker.baseline_rtt_ns // 2


@pytest.mark.asyncio
async def test_measure_baseline_rtt_ignores_stale_pong_from_a_prior_round():
    """A late TimePong from a timed-out round-1 probe must not resolve round 2.

    Regression: sequence numbers used to reset to 0 at the start of every
    round, so a round-1 probe (seq=0) that timed out could still deliver its
    TimePong after round 2 also assigned seq=0 to its own probe. Under the bug
    that stale reply resolves round 2's pending future with round 1's
    (unrelated) arrival time, producing a bogus RTT.
    """
    tracker = ClockOffsetTracker()
    stale_pong: TimePong | None = None

    async def send_ping_round_one(ping: TimePing) -> None:
        nonlocal stale_pong
        # Round 1's probe never replies in time -- capture its pong so it can
        # be delivered late, during round 2.
        stale_pong = TimePong(sequence=ping.sequence, sent_at_ns=ping.sent_at_ns)

    await tracker.measure_baseline_rtt(send_ping_round_one, probe_count=1, timeout=0.01)
    assert tracker.baseline_rtt_ns is None
    assert stale_pong is not None

    async def send_ping_round_two(ping: TimePing) -> None:
        # Deliver round 1's stale pong before round 2's own pong ever arrives.
        tracker.handle_pong(stale_pong)

    await tracker.measure_baseline_rtt(send_ping_round_two, probe_count=1, timeout=0.01)

    assert tracker.baseline_rtt_ns is None


@pytest.mark.asyncio
async def test_measure_baseline_rtt_leaves_baseline_unset_when_probes_time_out():
    tracker = ClockOffsetTracker()

    async def send_ping(ping: TimePing) -> None:
        return None

    await tracker.measure_baseline_rtt(send_ping, probe_count=2, timeout=0.01)

    assert tracker.baseline_rtt_ns is None
    assert tracker.estimated_one_way_ns is None


@pytest.mark.asyncio
async def test_measure_baseline_rtt_retries_until_router_starts_echoing():
    """A router that is silent at worker start must not exhaust the probe quota.

    Regression: on a real cluster the credit ROUTER is not echoing when the
    worker container starts, so a fixed probe_count of long probes expired the
    caller's budget and no baseline RTT was ever measured.
    """
    tracker = ClockOffsetTracker()
    sent: list[TimePing] = []
    echo_after = 3

    async def send_ping(ping: TimePing) -> None:
        sent.append(ping)
        if len(sent) > echo_after:
            tracker.handle_pong(
                TimePong(sequence=ping.sequence, sent_at_ns=ping.sent_at_ns)
            )

    await tracker.measure_baseline_rtt(
        send_ping, probe_count=2, timeout=0.01, max_attempts=10
    )

    assert len(sent) == echo_after + 2
    assert tracker.baseline_rtt_ns is not None


@pytest.mark.asyncio
async def test_measure_baseline_rtt_keeps_partial_result_when_cancelled():
    """Cancelling on budget expiry must keep RTTs that already round-tripped."""
    tracker = ClockOffsetTracker()
    sent: list[TimePing] = []

    async def send_ping(ping: TimePing) -> None:
        sent.append(ping)
        if len(sent) == 1:
            tracker.handle_pong(
                TimePong(sequence=ping.sequence, sent_at_ns=ping.sent_at_ns)
            )

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(
            tracker.measure_baseline_rtt(
                send_ping, probe_count=5, timeout=10.0, max_attempts=5
            ),
            timeout=0.05,
        )

    assert tracker.baseline_rtt_ns is not None


# =============================================================================
# Sample sanity gates
# =============================================================================


def test_absolute_bound_rejects_an_implausible_sample():
    """A sample far outside any real skew is corrupt data, not a measurement.

    The estimator is a minimum, so admitting one would hand the correction to
    that sample for the whole window.
    """
    tracker = ClockOffsetTracker(max_abs_sec=1.0)
    tracker.observe(issued_at_ns=0, received_at_ns=5_000_000_000)

    assert tracker.offset_ns is None
    assert tracker.sample_count == 0
    assert tracker.rejected_sample_count == 1


def test_absolute_bound_rejects_a_negative_implausible_sample():
    tracker = ClockOffsetTracker(max_abs_sec=1.0)
    tracker.observe(issued_at_ns=5_000_000_000, received_at_ns=0)

    assert tracker.offset_ns is None
    assert tracker.rejected_sample_count == 1


def _seed(tracker: ClockOffsetTracker, base_ns: int = 50_000_000) -> None:
    """Fill the window with a realistically jittered 50ms offset."""
    for jitter_ns in (0, 200_000, 100_000, 300_000, 150_000, 50_000):
        tracker.observe(issued_at_ns=0, received_at_ns=base_ns + jitter_ns)


def test_single_low_outlier_does_not_capture_the_minimum():
    """One bad sample must not own the correction until the window evicts it."""
    tracker = ClockOffsetTracker()
    _seed(tracker)
    assert tracker.offset_ns == 50_000_000

    tracker.observe(issued_at_ns=0, received_at_ns=10_000_000)

    assert tracker.offset_ns == 50_000_000
    assert tracker.rejected_sample_count == 1


def test_high_outlier_is_still_admitted_because_the_minimum_ignores_it():
    """Only the low side is dangerous; rejecting high samples buys nothing."""
    tracker = ClockOffsetTracker()
    _seed(tracker)

    tracker.observe(issued_at_ns=0, received_at_ns=900_000_000)

    assert tracker.offset_ns == 50_000_000
    assert tracker.sample_count == 7
    assert tracker.rejected_sample_count == 0


def test_sustained_low_run_is_read_as_a_real_step_and_re_seeds_the_window():
    """A controller restart or NTP step moves the true offset; follow it.

    Rejection has to be self-limiting: a filter that keeps rejecting reality
    would pin the correction at a stale level for the rest of the run.
    """
    tracker = ClockOffsetTracker(reset_after_rejects=3)
    _seed(tracker)

    for _ in range(3):
        tracker.observe(issued_at_ns=0, received_at_ns=10_000_000)

    assert tracker.offset_ns == 10_000_000
    assert tracker.rejected_sample_count == 2


def test_outlier_rejection_is_disabled_by_a_zero_factor():
    tracker = ClockOffsetTracker(outlier_factor=0.0)
    _seed(tracker)

    tracker.observe(issued_at_ns=0, received_at_ns=10_000_000)

    assert tracker.offset_ns == 10_000_000
    assert tracker.rejected_sample_count == 0


def test_outlier_rejection_waits_for_a_population_to_judge_against():
    """With fewer than min_samples there is no median worth trusting."""
    tracker = ClockOffsetTracker(min_samples=5)
    for _ in range(3):
        tracker.observe(issued_at_ns=0, received_at_ns=50_000_000)

    tracker.observe(issued_at_ns=0, received_at_ns=10_000_000)

    assert tracker.offset_ns == 10_000_000
    assert tracker.rejected_sample_count == 0


def test_rejected_sample_leaves_the_previous_estimate_in_place():
    tracker = ClockOffsetTracker()
    _seed(tracker)

    assert tracker.observe(issued_at_ns=0, received_at_ns=10_000_000) == 50_000_000


# =============================================================================
# Baseline RTT sanity and re-measurement
# =============================================================================


def _echo(tracker: ClockOffsetTracker, sent: list[TimePing]):
    async def send_ping(ping: TimePing) -> None:
        sent.append(ping)
        tracker.handle_pong(
            TimePong(sequence=ping.sequence, sent_at_ns=ping.sent_at_ns)
        )

    return send_ping


@pytest.mark.asyncio
async def test_implausibly_slow_probe_is_not_used_as_the_baseline():
    """Halving a queued-behind-credits round trip overstates one-way transit.

    ``correction_ns`` subtracts that estimate, so one bad probe would shift every
    exported timestamp by half of it.
    """
    tracker = ClockOffsetTracker(max_rtt_sec=1e-9)
    sent: list[TimePing] = []

    await tracker.measure_baseline_rtt(_echo(tracker, sent), probe_count=2)

    assert sent, "probes must still be sent"
    assert tracker.baseline_rtt_ns is None
    assert tracker.estimated_one_way_ns is None


@pytest.mark.asyncio
async def test_baseline_rtt_re_measurement_replaces_a_stale_estimate(
    monkeypatch: pytest.MonkeyPatch,
):
    """The startup probe goes stale when the path or the controller changes."""
    tracker = ClockOffsetTracker()
    sent: list[TimePing] = []
    perf_values = iter([0, 1_000_000, 0, 5_000_000])
    monkeypatch.setattr(
        clock_offset_tracker.time, "perf_counter_ns", lambda: next(perf_values)
    )

    await tracker.measure_baseline_rtt(_echo(tracker, sent), probe_count=1)
    assert tracker.baseline_rtt_ns == 1_000_000
    assert tracker.baseline_measurement_count == 1

    await tracker.measure_baseline_rtt(_echo(tracker, sent), probe_count=1)

    assert tracker.baseline_rtt_ns == 5_000_000
    assert tracker.estimated_one_way_ns == 2_500_000
    assert tracker.baseline_measurement_count == 2


@pytest.mark.asyncio
async def test_failed_re_measurement_keeps_the_previous_baseline():
    tracker = ClockOffsetTracker()
    sent: list[TimePing] = []
    await tracker.measure_baseline_rtt(_echo(tracker, sent), probe_count=1)
    established = tracker.baseline_rtt_ns

    async def never_echoes(ping: TimePing) -> None:
        return None

    await tracker.measure_baseline_rtt(never_echoes, probe_count=1, timeout=0.01)

    assert tracker.baseline_rtt_ns == established
    assert tracker.baseline_measurement_count == 1
