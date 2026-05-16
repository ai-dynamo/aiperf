# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for bootstrap confidence intervals on steady-state boundaries."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aiperf.analysis.bootstrap import BootstrapResult, bootstrap_detection
from aiperf.analysis.ramp_detection import detect_steady_state_window
from aiperf.analysis.sweepline import concurrency_sweep_line
from tests.unit.analysis.profiles import clean_ramp


@pytest.fixture
def clean_profile():
    """Generate a clean ramp profile for bootstrap tests."""
    return clean_ramp.generate(rng=np.random.default_rng(42))


class TestBootstrapDeterministic:
    def test_same_seed_same_result(self, clean_profile) -> None:
        """Fixed seed produces reproducible CIs."""
        r1 = bootstrap_detection(
            clean_profile.start_ns,
            clean_profile.end_ns,
            clean_profile.latency,
            clean_profile.ttft,
            n_iterations=20,
            rng=np.random.default_rng(123),
        )
        r2 = bootstrap_detection(
            clean_profile.start_ns,
            clean_profile.end_ns,
            clean_profile.latency,
            clean_profile.ttft,
            n_iterations=20,
            rng=np.random.default_rng(123),
        )
        assert r1.ci_ramp_up_ns == pytest.approx(r2.ci_ramp_up_ns)
        assert r1.ci_ramp_down_ns == pytest.approx(r2.ci_ramp_down_ns)
        assert r1.ci_mean_latency == pytest.approx(r2.ci_mean_latency)
        assert r1.ci_p99_latency == pytest.approx(r2.ci_p99_latency)


class TestBootstrapCleanRamp:
    def test_ci_is_valid_interval(self, clean_profile) -> None:
        """CI lower bound <= upper bound."""
        result = bootstrap_detection(
            clean_profile.start_ns,
            clean_profile.end_ns,
            clean_profile.latency,
            clean_profile.ttft,
            n_iterations=50,
            rng=np.random.default_rng(42),
        )
        assert result.ci_ramp_up_ns[0] <= result.ci_ramp_up_ns[1]
        assert result.ci_ramp_down_ns[0] <= result.ci_ramp_down_ns[1]
        assert result.ci_mean_latency[0] <= result.ci_mean_latency[1]
        assert result.ci_p99_latency[0] <= result.ci_p99_latency[1]

    def test_n_iterations_recorded(self, clean_profile) -> None:
        """n_iterations is correctly recorded."""
        result = bootstrap_detection(
            clean_profile.start_ns,
            clean_profile.end_ns,
            clean_profile.latency,
            clean_profile.ttft,
            n_iterations=30,
            rng=np.random.default_rng(42),
        )
        assert result.n_iterations == 30


class TestBootstrapContainsPointEstimate:
    def test_point_estimate_within_ci(self, clean_profile) -> None:
        """Point estimate from single detection falls within bootstrap CI."""
        sorted_ts, conc = concurrency_sweep_line(
            clean_profile.start_ns, clean_profile.end_ns
        )
        w_start, w_end, _ = detect_steady_state_window(
            sorted_ts,
            conc,
            clean_profile.start_ns,
            clean_profile.end_ns,
            latency=clean_profile.latency,
            ttft=clean_profile.ttft,
            min_window_pct=5.0,
        )

        result = bootstrap_detection(
            clean_profile.start_ns,
            clean_profile.end_ns,
            clean_profile.latency,
            clean_profile.ttft,
            n_iterations=100,
            rng=np.random.default_rng(42),
        )

        # Point estimate should be within (or very close to) the CI
        # Use a small margin for edge cases
        assert result.ci_ramp_up_ns[0] <= w_start <= result.ci_ramp_up_ns[1] * 1.01
        assert result.ci_ramp_down_ns[0] * 0.99 <= w_end <= result.ci_ramp_down_ns[1]


class TestBootstrapShortInput:
    def test_short_input_returns_valid_result(self) -> None:
        """Small dataset (30 records) doesn't crash and produces valid CIs."""
        rng = np.random.default_rng(42)
        n = 30
        start = np.arange(n, dtype=np.float64) * 100e6
        end = start + rng.uniform(20, 80, n) * 1e6
        latency = rng.normal(50, 5, n) * 1e6
        ttft = latency * 0.1

        result = bootstrap_detection(
            start,
            end,
            latency,
            ttft,
            n_iterations=20,
            rng=np.random.default_rng(42),
        )
        assert isinstance(result, BootstrapResult)
        assert result.n_iterations == 20
        # CIs should be finite (not NaN for small-but-valid input)
        assert not math.isnan(result.ci_ramp_up_ns[0])


class TestBootstrapThroughput:
    def test_none_throughput_backward_compat(self, clean_profile) -> None:
        """None throughput params do not break bootstrap."""
        result = bootstrap_detection(
            clean_profile.start_ns,
            clean_profile.end_ns,
            clean_profile.latency,
            clean_profile.ttft,
            n_iterations=10,
            rng=np.random.default_rng(42),
            generation_start_ns=None,
            output_tokens=None,
        )
        assert isinstance(result, BootstrapResult)
        assert result.n_iterations == 10

    def test_with_throughput_data(self, clean_profile) -> None:
        """Passing throughput data produces valid CIs."""
        result = bootstrap_detection(
            clean_profile.start_ns,
            clean_profile.end_ns,
            clean_profile.latency,
            clean_profile.ttft,
            n_iterations=10,
            rng=np.random.default_rng(42),
            generation_start_ns=clean_profile.generation_start_ns,
            output_tokens=clean_profile.output_tokens,
        )
        assert isinstance(result, BootstrapResult)
        assert result.ci_ramp_up_ns[0] <= result.ci_ramp_up_ns[1]
