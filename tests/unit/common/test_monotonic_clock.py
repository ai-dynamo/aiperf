# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perf-counter-anchored wall clock."""

import time

from aiperf.common.monotonic_clock import MonotonicClock, process_clock


class TestMonotonicClock:
    def test_now_ns_is_in_the_wall_clock_domain(self) -> None:
        clock = MonotonicClock()
        # Within a second of the real wall clock: same Unix epoch domain.
        assert abs(clock.now_ns() - time.time_ns()) < 1_000_000_000

    def test_now_ns_is_non_decreasing(self) -> None:
        clock = MonotonicClock()
        samples = [clock.now_ns() for _ in range(100)]
        assert samples == sorted(samples)

    def test_now_ns_ignores_wall_clock_steps(self, monkeypatch) -> None:
        clock = MonotonicClock()
        before = clock.now_ns()
        # An NTP step backwards must not move derived timestamps.
        monkeypatch.setattr(time, "time_ns", lambda: 0)
        assert clock.now_ns() >= before

    def test_elapsed_ns_and_sec_agree(self) -> None:
        clock = MonotonicClock()
        elapsed_ns = clock.elapsed_ns()
        assert elapsed_ns >= 0
        assert clock.elapsed_sec() >= elapsed_ns / 1e9

    def test_wall_ns_at_maps_a_perf_reading_through_the_anchor(self) -> None:
        clock = MonotonicClock()
        perf_ns = clock.perf_anchor_ns + 1_234_567
        assert clock.wall_ns_at(perf_ns) == clock.wall_anchor_ns + 1_234_567


class TestProcessClock:
    def test_process_clock_returns_one_instance_per_process(self) -> None:
        assert process_clock() is process_clock()

    def test_process_clock_ignores_wall_clock_steps(self, monkeypatch) -> None:
        clock = process_clock()
        perf_ns = time.perf_counter_ns()
        before = clock.wall_ns_at(perf_ns)
        monkeypatch.setattr(time, "time_ns", lambda: 0)
        assert process_clock().wall_ns_at(perf_ns) == before
