# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perf-counter-anchored wall clock."""

import time

from aiperf.common.monotonic_clock import MonotonicClock


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
