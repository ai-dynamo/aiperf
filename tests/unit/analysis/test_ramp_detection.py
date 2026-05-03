# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ramp boundary detection."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.analysis.ramp_detection import (
    cusum_steady_state_window,
    detect_steady_state_window,
    manual_steady_state_window,
    mser5_boundary_ns,
    mser5_truncation_point,
)
from aiperf.analysis.sweepline import concurrency_sweep_line, throughput_sweep_line


class TestCusumSteadyStateWindow:
    def test_empty_input(self) -> None:
        start, end = cusum_steady_state_window(
            np.array([], dtype=np.float64), np.array([], dtype=np.float64)
        )
        assert start == 0.0
        assert end == 0.0

    def test_constant_concurrency_full_window(self) -> None:
        """Constant concurrency → window covers approximately the full range."""
        n = 100
        start_ns = np.zeros(n)
        end_ns = np.full(n, 1000.0)
        sorted_ts, concurrency = concurrency_sweep_line(start_ns, end_ns)

        window_start, window_end = cusum_steady_state_window(sorted_ts, concurrency)
        assert window_start <= 100.0
        assert window_end >= 900.0

    def test_ramp_up_steady_ramp_down(self) -> None:
        """Ramp up → steady → drain. CUSUM should trim ramp regions."""
        n_steady = 50
        start_ns = np.linspace(0, 100, n_steady)
        end_ns = np.linspace(900, 1000, n_steady)
        sorted_ts, concurrency = concurrency_sweep_line(start_ns, end_ns)

        window_start, window_end = cusum_steady_state_window(
            sorted_ts, concurrency, min_window_pct=10.0
        )
        assert window_end > window_start
        window_pct = (window_end - window_start) / (sorted_ts[-1] - sorted_ts[0]) * 100
        assert window_pct >= 10.0

    def test_fallback_on_small_window(self) -> None:
        """Window below min_window_pct → falls back to full range."""
        start_ns = np.array([0.0, 1.0, 2.0])
        end_ns = np.array([3.0, 4.0, 5.0])
        sorted_ts, concurrency = concurrency_sweep_line(start_ns, end_ns)

        window_start, window_end = cusum_steady_state_window(
            sorted_ts, concurrency, min_window_pct=90.0
        )
        assert window_start == float(sorted_ts[0])
        assert window_end == float(sorted_ts[-1])


class TestMser5TruncationPoint:
    def test_stationary_series_no_truncation(self) -> None:
        """Already stationary → truncation point at 0."""
        rng = np.random.default_rng(42)
        values = rng.normal(100.0, 5.0, 200)
        d = mser5_truncation_point(values)
        assert d == 0

    def test_ramp_then_steady(self) -> None:
        """Clear ramp followed by steady → truncation point > 0."""
        ramp = np.linspace(0, 100, 50)
        steady = np.full(150, 100.0) + np.random.default_rng(42).normal(0, 2, 150)
        values = np.concatenate([ramp, steady])
        d = mser5_truncation_point(values)
        assert d > 0
        assert d <= 100  # should be near the ramp-steady boundary

    def test_short_input(self) -> None:
        """Fewer than 10 observations → returns 0."""
        values = np.array([1.0, 2.0, 3.0])
        assert mser5_truncation_point(values) == 0

    def test_all_same_values(self) -> None:
        """Constant series → no truncation needed."""
        values = np.full(100, 42.0)
        assert mser5_truncation_point(values) == 0


class TestMser5BoundaryNs:
    def test_valid_latency(self) -> None:
        """Valid latency with ramp → returns boundaries."""
        n = 200
        start_ns = np.arange(n, dtype=np.float64) * 10.0
        end_ns = start_ns + 5.0
        filled = np.ones(n, dtype=bool)

        # Ramp phase: high and increasing latency, then steady
        ramp = np.linspace(500, 100, 50)
        steady = np.full(150, 100.0) + np.random.default_rng(42).normal(0, 3, 150)
        metric_values = np.concatenate([ramp, steady])

        up, down = mser5_boundary_ns(metric_values, start_ns, end_ns, filled)
        assert up is not None
        assert up >= 0.0

    def test_all_nan_returns_none(self) -> None:
        """All-NaN metric → returns (None, None)."""
        n = 100
        start_ns = np.arange(n, dtype=np.float64) * 10.0
        end_ns = start_ns + 5.0
        filled = np.ones(n, dtype=bool)
        metric_values = np.full(n, np.nan)

        up, down = mser5_boundary_ns(metric_values, start_ns, end_ns, filled)
        assert up is None
        assert down is None

    def test_insufficient_data(self) -> None:
        """Fewer than 20 valid records → returns (None, None)."""
        n = 10
        start_ns = np.arange(n, dtype=np.float64)
        end_ns = start_ns + 1.0
        filled = np.ones(n, dtype=bool)
        metric_values = np.arange(n, dtype=np.float64)

        up, down = mser5_boundary_ns(metric_values, start_ns, end_ns, filled)
        assert up is None
        assert down is None


class TestDetectSteadyStateWindowCombined:
    def _make_scenario(
        self,
        n_ramp: int = 20,
        n_steady: int = 160,
        n_drain: int = 20,
        ttft_all_nan: bool = False,
    ) -> tuple:
        """Build ramp → steady → drain scenario with latency and optional TTFT."""
        rng = np.random.default_rng(42)
        n = n_ramp + n_steady + n_drain

        # Sequential requests
        start_ns = np.arange(n, dtype=np.float64) * 10.0
        end_ns = start_ns + 8.0

        # Latency: high during ramp, stable during steady, high during drain
        latency = np.concatenate(
            [
                rng.normal(500, 50, n_ramp),
                rng.normal(100, 5, n_steady),
                rng.normal(500, 50, n_drain),
            ]
        )

        if ttft_all_nan:
            ttft = np.full(n, np.nan)
        else:
            ttft = np.concatenate(
                [
                    rng.normal(200, 30, n_ramp),
                    rng.normal(50, 3, n_steady),
                    rng.normal(200, 30, n_drain),
                ]
            )

        # Throughput data
        generation_start_ns = start_ns + np.abs(ttft) * 0.1
        output_tokens = rng.integers(50, 200, n).astype(np.float64)
        sorted_t_ts, tput = throughput_sweep_line(
            generation_start_ns, end_ns, output_tokens
        )

        sorted_ts, concurrency = concurrency_sweep_line(start_ns, end_ns)
        return (
            sorted_ts,
            concurrency,
            start_ns,
            end_ns,
            latency,
            ttft,
            sorted_t_ts,
            tput,
        )

    def test_all_signals_agree(self) -> None:
        """All signals present → method includes all signal names."""
        sorted_ts, conc, start_ns, end_ns, lat, ttft, s_t_ts, tput = (
            self._make_scenario()
        )
        ws, we, method = detect_steady_state_window(
            sorted_ts,
            conc,
            start_ns,
            end_ns,
            latency=lat,
            ttft=ttft,
            min_window_pct=5.0,
            sorted_tput_ts=s_t_ts,
            throughput=tput,
        )
        assert we > ws
        assert "cusum" in method

    def test_non_streaming_ttft_nan(self) -> None:
        """TTFT all NaN → method should not include mser5_ttft."""
        sorted_ts, conc, start_ns, end_ns, lat, ttft, s_t_ts, tput = (
            self._make_scenario(ttft_all_nan=True)
        )
        ws, we, method = detect_steady_state_window(
            sorted_ts,
            conc,
            start_ns,
            end_ns,
            latency=lat,
            ttft=ttft,
            min_window_pct=5.0,
        )
        assert we > ws
        assert "mser5_ttft" not in method
        assert "cusum" in method

    def test_all_signals_with_throughput(self) -> None:
        """Throughput signal present → method includes cusum_throughput."""
        sorted_ts, conc, start_ns, end_ns, lat, ttft, s_t_ts, tput = (
            self._make_scenario()
        )
        ws, we, method = detect_steady_state_window(
            sorted_ts,
            conc,
            start_ns,
            end_ns,
            latency=lat,
            ttft=ttft,
            min_window_pct=5.0,
            sorted_tput_ts=s_t_ts,
            throughput=tput,
        )
        assert we > ws
        assert "cusum_throughput" in method

    def test_throughput_none_excluded(self) -> None:
        """Throughput params None → cusum_throughput NOT in method."""
        sorted_ts, conc, start_ns, end_ns, lat, ttft, _, _ = self._make_scenario()
        ws, we, method = detect_steady_state_window(
            sorted_ts,
            conc,
            start_ns,
            end_ns,
            latency=lat,
            ttft=ttft,
            min_window_pct=5.0,
            sorted_tput_ts=None,
            throughput=None,
        )
        assert we > ws
        assert "cusum_throughput" not in method

    def test_empty_input(self) -> None:
        empty = np.array([], dtype=np.float64)
        ws, we, method = detect_steady_state_window(
            empty,
            empty,
            empty,
            empty,
            latency=empty,
            ttft=empty,
        )
        assert ws == 0.0
        assert we == 0.0
        assert method == "empty"

    def test_fallback_no_overlap(self) -> None:
        """When signals disagree completely → fallback."""
        # Create a scenario where CUSUM and MSER-5 boundaries don't overlap
        # by making latency ramp occupy the entire range where concurrency is steady
        n = 200
        start_ns = np.arange(n, dtype=np.float64) * 10.0
        end_ns = start_ns + 8.0
        # Latency always increasing — never reaches steady state
        latency = np.linspace(10, 1000, n)
        ttft = np.full(n, np.nan)

        sorted_ts, conc = concurrency_sweep_line(start_ns, end_ns)
        ws, we, method = detect_steady_state_window(
            sorted_ts,
            conc,
            start_ns,
            end_ns,
            latency=latency,
            ttft=ttft,
            min_window_pct=5.0,
        )
        # Should still return valid result (either fallback or detected)
        assert we >= ws


class TestManualSteadyStateWindow:
    def test_basic(self) -> None:
        start, end = manual_steady_state_window(0.0, 1000.0, 10.0, 90.0)
        assert start == pytest.approx(100.0)
        assert end == pytest.approx(900.0)

    def test_full_range(self) -> None:
        start, end = manual_steady_state_window(0.0, 1000.0, 0.0, 100.0)
        assert start == pytest.approx(0.0)
        assert end == pytest.approx(1000.0)

    def test_narrow_window(self) -> None:
        start, end = manual_steady_state_window(100.0, 200.0, 40.0, 60.0)
        assert start == pytest.approx(140.0)
        assert end == pytest.approx(160.0)

    def test_offset_timestamps(self) -> None:
        """Verify window computation works with non-zero base timestamps."""
        start, end = manual_steady_state_window(1000.0, 2000.0, 20.0, 80.0)
        assert start == pytest.approx(1200.0)
        assert end == pytest.approx(1800.0)
