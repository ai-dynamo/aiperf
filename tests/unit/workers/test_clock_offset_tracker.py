# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cross-machine clock-offset tracking.

The estimator is a minimum-sample filter over a sliding window (NTP clock-filter
style, RFC 5905): every one-way sample carries network transit as *positive*
bias, so the smallest sample in the window is the closest approximation of the
true clock skew. These tests pin that behavior, including its asymmetry --
positive outliers are rejected outright, negative ones are adopted.
"""

import asyncio

import pytest

from aiperf.credit.messages import TimePing, TimePong
from aiperf.workers.clock_offset_tracker import ClockOffsetTracker


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
