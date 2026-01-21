# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Poisson rate timing mode.

Poisson rate mode issues credits with exponentially distributed inter-arrival
times, simulating realistic traffic patterns with natural variability.

Key characteristics:
- Mean inter-arrival time = 1/rate
- Coefficient of variation (CV) ~ 1.0 for exponential distribution
- Natural variability around the target rate

Tests cover:
- Basic functionality at various QPS levels
- Statistical distribution verification (comprehensive test with 4 checks)
- Multi-turn conversations
- Concurrency interactions
- Stress tests including bursty pattern verification
"""

import pytest

from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    BaseConcurrencyTests,
    BaseCreditFlowTests,
    TimingTestConfig,
    build_timing_command,
)
from tests.harness.analyzers import (
    CreditFlowAnalyzer,
    StatisticalAnalyzer,
    TimingAnalyzer,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestPoissonRateBasic:
    """Basic functionality tests for Poisson rate timing."""

    @pytest.mark.parametrize(
        "num_sessions,qps",
        [
            (15, 50.0),
            (25, 100.0),
            (35, 150.0),
            (50, 200.0),
        ],
    )
    def test_poisson_rate_completes(
        self, cli: AIPerfCLI, num_sessions: int, qps: float
    ):
        """Test Poisson rate mode completes at various QPS levels."""
        config = TimingTestConfig(num_sessions=num_sessions, qps=qps)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions
        assert result.has_streaming_metrics

    def test_poisson_rate_multi_turn(self, cli: AIPerfCLI):
        """Test Poisson rate with multi-turn conversations."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=75.0,
            turns_per_session=4,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
        assert result.has_streaming_metrics


@pytest.mark.component_integration
class TestPoissonRateCreditFlow(BaseCreditFlowTests):
    """Credit flow verification for Poisson rate timing.

    Inherits common credit flow tests from BaseCreditFlowTests.
    Tests: credits_balanced, credits_per_session, turn_indices_sequential
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Poisson rate timing command."""
        return build_timing_command(config, arrival_pattern="poisson")


@pytest.mark.component_integration
class TestPoissonRateStatistics:
    """Statistical distribution tests for Poisson rate mode.

    Tests verify that the timing system produces inter-arrival times following
    an exponential distribution (Poisson process). The comprehensive test validates:

    1. Mean ≈ 1/rate (correct average spacing)
    2. Std ≈ Mean (exponential property: sigma = mu)
    3. CV ≈ 1.0 (coefficient of variation for exponential)
    4. CDF property: ~63.2% of values below mean
    5. Independence: consecutive intervals uncorrelated
    6. Index of dispersion ≈ 1.0 (variance/mean of event counts)
    """

    def test_poisson_distribution_comprehensive(self, cli: AIPerfCLI):
        """Comprehensive Poisson validation using multiple statistical tests.

        Runs 4 independent statistical tests and passes if at least 3 pass.
        This is more robust than single-test validation.

        Tests include:
        - Mean/Std/CV verification (exponential property)
        - CDF property (63.2% below mean)
        - Independence (memoryless property)
        - Index of dispersion
        """
        config = TimingTestConfig(num_sessions=60, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 30, (
            f"Insufficient data for comprehensive Poisson analysis: got {len(gaps)} gaps, "
            f"need >= 30. 60 sessions should produce enough data."
        )

        passed, summary, details = StatisticalAnalyzer.comprehensive_poisson_check(
            gaps, expected_rate=config.qps, tolerance_pct=30.0
        )

        assert passed, f"Comprehensive Poisson check failed: {summary}"


@pytest.mark.component_integration
class TestPoissonRateWithConcurrency(BaseConcurrencyTests):
    """Tests for Poisson rate with concurrency limits.

    Inherits common concurrency tests from BaseConcurrencyTests.
    Tests: test_with_concurrency_limit, test_with_prefill_concurrency,
           test_multi_turn_with_concurrency
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Poisson rate timing command."""
        return build_timing_command(config, arrival_pattern="poisson")


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestPoissonRateStress:
    """Stress tests for Poisson rate timing."""

    def test_high_volume(self, cli: AIPerfCLI):
        """Test high volume with Poisson rate."""
        config = TimingTestConfig(num_sessions=100, qps=300.0, timeout=90.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_sustained_multi_turn(self, cli: AIPerfCLI):
        """Test sustained multi-turn Poisson workload."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=100.0,
            turns_per_session=5,
            timeout=90.0,
        )
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()

    def test_bursty_pattern(self, cli: AIPerfCLI):
        """Test that Poisson produces bursty patterns (some clustering).

        Exponential distribution has high variability, so we should see some
        very short gaps (bursts) where requests cluster together.
        """
        config = TimingTestConfig(num_sessions=50, qps=150.0)
        cmd = build_timing_command(config, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 10, (
            f"Insufficient data for bursty pattern analysis: got {len(gaps)} gaps, need >= 10."
        )
        # Poisson should have some very short intervals (bursts)
        min_gap = min(gaps)
        mean_gap = timing.calculate_mean(gaps)
        # Some gaps should be much shorter than mean
        assert min_gap < mean_gap * 0.5, (
            f"No bursty behavior detected: min_gap={min_gap:.4f}s, mean_gap={mean_gap:.4f}s. "
            f"Expected min_gap < {mean_gap * 0.5:.4f}s for exponential distribution."
        )
