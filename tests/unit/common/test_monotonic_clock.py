# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the perf-counter-anchored wall clock."""

import contextlib
import threading
import time

import pytest
from pytest import param

from aiperf.common import monotonic_clock
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

    def test_explicit_anchors_are_used(self) -> None:
        clock = MonotonicClock(100, 5_000)
        assert (clock.perf_anchor_ns, clock.wall_anchor_ns) == (100, 5_000)
        assert clock.wall_ns_at(150) == 5_050

    @pytest.mark.parametrize(
        ("perf_anchor_ns", "wall_anchor_ns"),
        [
            param(100, None, id="perf-only"),
            param(None, 5_000, id="wall-only"),
        ],
    )  # fmt: skip
    def test_partial_anchor_raises_value_error(
        self, perf_anchor_ns: int | None, wall_anchor_ns: int | None
    ) -> None:
        with pytest.raises(ValueError, match="must be given together"):
            MonotonicClock(perf_anchor_ns, wall_anchor_ns)


class TestCalibrated:
    """``calibrated`` must anchor on the tightest (perf, wall, perf) bracket."""

    @pytest.mark.parametrize(
        ("perf_reads", "wall_reads", "expected_perf", "expected_wall"),
        [
            param(
                [0, 2_000_000, 10_000_000, 10_000_100],
                [5_000, 6_000],
                10_000_050,
                6_000,
                id="preempted-first-sample-is-skipped",
            ),
            param(
                [0, 100, 10_000_000, 12_000_000],
                [5_000, 6_000],
                50,
                5_000,
                id="later-wider-sample-does-not-replace-tighter",
            ),
        ],
    )  # fmt: skip
    def test_calibrated_anchors_on_tightest_bracket(
        self,
        monkeypatch: pytest.MonkeyPatch,
        perf_reads: list[int],
        wall_reads: list[int],
        expected_perf: int,
        expected_wall: int,
    ) -> None:
        perf_iter, wall_iter = iter(perf_reads), iter(wall_reads)
        monkeypatch.setattr(time, "perf_counter_ns", lambda: next(perf_iter))
        monkeypatch.setattr(time, "time_ns", lambda: next(wall_iter))

        clock = MonotonicClock.calibrated(samples=2)

        assert clock.perf_anchor_ns == expected_perf
        assert clock.wall_anchor_ns == expected_wall

    def test_calibrated_is_in_the_wall_clock_domain(self) -> None:
        clock = MonotonicClock.calibrated()
        assert abs(clock.now_ns() - time.time_ns()) < 1_000_000_000

    @pytest.mark.parametrize(
        "samples",
        [
            param(0, id="zero"),
            param(-1, id="negative"),
        ],
    )  # fmt: skip
    def test_calibrated_non_positive_samples_raises_value_error(
        self, samples: int
    ) -> None:
        with pytest.raises(ValueError, match="samples must be at least 1"):
            MonotonicClock.calibrated(samples=samples)


class TestProcessClock:
    def test_process_clock_returns_one_instance_per_process(self) -> None:
        assert process_clock() is process_clock()

    def test_process_clock_ignores_wall_clock_steps(self, monkeypatch) -> None:
        clock = process_clock()
        perf_ns = time.perf_counter_ns()
        before = clock.wall_ns_at(perf_ns)
        monkeypatch.setattr(time, "time_ns", lambda: 0)
        assert process_clock().wall_ns_at(perf_ns) == before

    def test_process_clock_is_calibrated(self, monkeypatch) -> None:
        sentinel = MonotonicClock()
        monkeypatch.setattr(monotonic_clock, "_process_clock", None)
        monkeypatch.setattr(
            MonotonicClock, "calibrated", classmethod(lambda cls: sentinel)
        )
        assert process_clock() is sentinel

    def test_process_clock_concurrent_first_calls_share_one_anchor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(monotonic_clock, "_process_clock", None)
        real_calibrated = MonotonicClock.calibrated.__func__
        entered = threading.Barrier(2, timeout=5)

        def slow_calibrated(cls: type[MonotonicClock]) -> MonotonicClock:
            # Hold the first caller inside creation so the second races it.
            with contextlib.suppress(threading.BrokenBarrierError):
                entered.wait(timeout=0.2)
            return real_calibrated(cls)

        monkeypatch.setattr(MonotonicClock, "calibrated", classmethod(slow_calibrated))
        results: list[MonotonicClock] = []
        threads = [
            threading.Thread(target=lambda: results.append(process_clock()))
            for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert results[0] is results[1]
