# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy tests for steady-state detection across synthetic profiles."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.analysis.ramp_detection import detect_steady_state_window
from aiperf.analysis.stationarity import batch_means_trend_test
from aiperf.analysis.sweepline import concurrency_sweep_line, throughput_sweep_line
from tests.unit.analysis.profiles import (
    SyntheticBenchmark,
    no_steady_state,
    short_benchmark,
)


def _detect(bench: SyntheticBenchmark) -> tuple[float, float, str]:
    """Run detect_steady_state_window on a benchmark profile."""
    sorted_ts, concurrency = concurrency_sweep_line(bench.start_ns, bench.end_ns)
    s_tput_ts = None
    tput = None
    if bench.generation_start_ns is not None and bench.output_tokens is not None:
        s_tput_ts, tput = throughput_sweep_line(
            bench.generation_start_ns, bench.end_ns, bench.output_tokens
        )
        if len(s_tput_ts) == 0:
            s_tput_ts = None
            tput = None
    return detect_steady_state_window(
        sorted_ts,
        concurrency,
        bench.start_ns,
        bench.end_ns,
        latency=bench.latency,
        ttft=bench.ttft,
        min_window_pct=5.0,
        sorted_tput_ts=s_tput_ts,
        throughput=tput,
    )


class TestBoundaryAccuracy:
    # CUSUM+MSER-5 is conservative by design: it takes the most restrictive
    # boundary across multiple signals, so detected windows often extend
    # slightly beyond ground-truth boundaries. 25% tolerance accounts for
    # gradual ramp transitions in realistic profiles.
    _THRESHOLD = 0.25

    def test_ramp_up_within_tolerance(self, ramp_benchmark: SyntheticBenchmark) -> None:
        """Detected ramp-up end is within 25% of total duration from ground truth."""
        bench = ramp_benchmark
        window_start, _window_end, _method = _detect(bench)
        total = bench.total_duration
        error_pct = abs(window_start - bench.true_ramp_up_end_ns) / total
        assert error_pct < self._THRESHOLD, (
            f"{bench.profile_name}: ramp-up boundary error {error_pct:.1%} "
            f"exceeds {self._THRESHOLD:.0%}"
        )

    def test_ramp_down_within_tolerance(
        self, ramp_benchmark: SyntheticBenchmark
    ) -> None:
        """Detected ramp-down start is within 25% of total duration from ground truth."""
        bench = ramp_benchmark
        _window_start, window_end, _method = _detect(bench)
        total = bench.total_duration
        error_pct = abs(window_end - bench.true_ramp_down_start_ns) / total
        assert error_pct < self._THRESHOLD, (
            f"{bench.profile_name}: ramp-down boundary error {error_pct:.1%} "
            f"exceeds {self._THRESHOLD:.0%}"
        )


class TestWindowCoverage:
    def test_window_coverage_ratio(self, ramp_benchmark: SyntheticBenchmark) -> None:
        """Detected window covers 50-150% of true window.

        Upper bound is 1.50 because CUSUM+MSER-5 is conservative — it prefers
        a slightly wider window over accidentally truncating valid data.
        """
        bench = ramp_benchmark
        window_start, window_end, _method = _detect(bench)
        detected_dur = window_end - window_start
        true_dur = bench.true_window_duration
        if true_dur <= 0:
            pytest.skip("No true window for flat profile")
        ratio = detected_dur / true_dur
        assert 0.50 <= ratio <= 1.50, (
            f"{bench.profile_name}: coverage ratio {ratio:.2f} outside [0.50, 1.50]"
        )


class TestMetricBias:
    def test_mean_latency_bias(self, ramp_benchmark: SyntheticBenchmark) -> None:
        """Mean latency in detected window close to ground truth."""
        bench = ramp_benchmark
        window_start, window_end, _method = _detect(bench)
        mask = (bench.start_ns >= window_start) & (bench.end_ns <= window_end)
        if mask.sum() == 0:
            pytest.skip("No requests in detected window")
        detected_mean = float(np.mean(bench.latency[mask]))
        true_mean = bench.true_steady_state_mean_latency
        if true_mean == 0:
            pytest.skip("True mean is 0")
        bias = abs(detected_mean - true_mean) / true_mean
        assert bias < 0.15, (
            f"{bench.profile_name}: mean latency bias {bias:.2%} exceeds 15%"
        )


class TestRequestInclusionF1:
    def test_f1_score(self, ramp_benchmark: SyntheticBenchmark) -> None:
        """F1 score of request inclusion relative to ground truth."""
        bench = ramp_benchmark
        window_start, window_end, _method = _detect(bench)
        detected = (bench.start_ns >= window_start) & (bench.end_ns <= window_end)
        true = bench.true_mask()

        tp = float((detected & true).sum())
        fp = float((detected & ~true).sum())
        fn = float((~detected & true).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        threshold = (
            0.80
            if bench.profile_name in ("clean_ramp", "flat_profile", "high_concurrency")
            else 0.60
        )
        assert f1 > threshold, (
            f"{bench.profile_name}: F1 {f1:.2f} below {threshold:.2f} "
            f"(precision={precision:.2f}, recall={recall:.2f})"
        )


class TestSampleSizeWarning:
    def test_short_benchmark_triggers_warning(self) -> None:
        """Short benchmark (200 reqs) triggers sample_size_warning."""
        bench = short_benchmark.generate(rng=np.random.default_rng(42))
        window_start, window_end, _method = _detect(bench)
        filled = ~np.isnan(bench.start_ns) & ~np.isnan(bench.end_ns)
        ss_mask = (
            filled & (bench.start_ns >= window_start) & (bench.end_ns <= window_end)
        )
        n_ss = int(ss_mask.sum())
        effective_p99 = int(n_ss * 0.01)
        assert effective_p99 < 10, f"expected p99 sample < 10, got {effective_p99}"


class TestStationarityWarning:
    def test_no_steady_state_triggers_warning(self) -> None:
        """Continuously drifting latency triggers stationarity_warning."""
        bench = no_steady_state.generate(rng=np.random.default_rng(42))
        window_start, window_end, _method = _detect(bench)

        mask = (bench.start_ns >= window_start) & (bench.end_ns <= window_end)
        windowed_latency = bench.latency[mask]
        valid = windowed_latency[~np.isnan(windowed_latency)]
        assert len(valid) >= 10, "Need at least 10 observations for trend test"

        rho, p = batch_means_trend_test(valid)
        warning = abs(rho) > 0.65 and p < 0.05
        assert warning, (
            f"Expected stationarity warning for drifting profile (rho={rho:.2f}, p={p:.3f})"
        )
