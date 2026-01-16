# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for constant rate timing mode.

Constant rate mode issues credits at fixed intervals (period = 1/rate).
This provides deterministic, evenly-spaced request timing.

Tests cover:
- Basic functionality at various QPS levels
- Credit flow verification
- Timing accuracy (intervals should be constant)
- Multi-turn conversations
- Concurrency interactions
- Stress tests (high volume, sustained workloads)
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
class TestConstantRateBasic:
    """Basic functionality tests for constant rate timing."""

    @pytest.mark.parametrize(
        "num_sessions,qps",
        [
            (10, 50.0),
            (20, 100.0),
            (30, 150.0),
            (50, 200.0),
        ],
    )
    def test_constant_rate_completes(
        self, cli: AIPerfCLI, num_sessions: int, qps: float
    ):
        """Test constant rate mode completes at various QPS levels."""
        config = TimingTestConfig(num_sessions=num_sessions, qps=qps)
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions
        assert result.has_streaming_metrics

    def test_constant_rate_multi_turn(self, cli: AIPerfCLI):
        """Test constant rate with multi-turn conversations."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=75.0,
            turns_per_session=4,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
        assert result.has_streaming_metrics


@pytest.mark.component_integration
class TestConstantRateCreditFlow(BaseCreditFlowTests):
    """Credit flow verification for constant rate timing.

    Inherits common credit flow tests from BaseCreditFlowTests.
    Tests: credits_balanced, credits_per_session, turn_indices_sequential
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build constant rate timing command."""
        return build_timing_command(config, arrival_pattern="constant")


@pytest.mark.component_integration
class TestConstantRateTiming:
    """Timing accuracy tests for constant rate mode.

    Note: These tests verify that constant rate timing produces intervals with
    correct mean values. CV thresholds are relaxed because the test harness
    (FakeCommunication, async scheduling) introduces timing variability that
    makes precise interval verification impractical.

    Tests marked xfail(strict=False) due to inherent timing variability
    in the test harness environment.
    """

    @pytest.mark.parametrize(
        "num_sessions,qps",
        [
            (15, 50.0),  # 20ms intervals
            (20, 100.0),  # 10ms intervals
            (25, 150.0),  # ~6.7ms intervals
        ],
    )
    def test_constant_intervals(self, cli: AIPerfCLI, num_sessions: int, qps: float):
        """Verify intervals have correct mean (rate is correct)."""
        config = TimingTestConfig(num_sessions=num_sessions, qps=qps)
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 3, (
            f"Insufficient data for timing analysis: got {len(gaps)} gaps, need >= 3. "
            f"This indicates a test harness issue - {num_sessions} sessions should produce enough data."
        )
        # Use relaxed CV threshold (0.8) - test harness introduces timing jitter
        passed, reason = StatisticalAnalyzer.is_approximately_constant(
            gaps, expected=config.expected_gap_sec, tolerance_pct=50.0, max_cv=0.8
        )
        assert passed, f"Intervals not constant: {reason}"


@pytest.mark.component_integration
class TestConstantRateWithConcurrency(BaseConcurrencyTests):
    """Tests for constant rate with concurrency limits.

    Inherits common concurrency tests from BaseConcurrencyTests.
    Tests: test_with_concurrency_limit, test_with_prefill_concurrency,
           test_multi_turn_with_concurrency
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build constant rate timing command."""
        return build_timing_command(config, arrival_pattern="constant")


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestConstantRateStress:
    """Stress tests for constant rate timing."""

    def test_high_volume(self, cli: AIPerfCLI):
        """Test high volume of requests."""
        config = TimingTestConfig(num_sessions=100, qps=300.0, timeout=90.0)
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

    def test_sustained_multi_turn(self, cli: AIPerfCLI):
        """Test sustained multi-turn workload."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=100.0,
            turns_per_session=5,
            timeout=90.0,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        analyzer = CreditFlowAnalyzer(runner)
        assert analyzer.credits_balanced()
