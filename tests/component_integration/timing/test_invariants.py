# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariant tests for timing strategies.

These tests verify fundamental correctness properties that MUST hold regardless
of timing mode, rate, concurrency, or other configuration. Failures here indicate
serious bugs in the timing framework.

Invariants tested via InvariantChecker (TestCreditLifecycleInvariants):
- Credit ID uniqueness within each phase
- Credit return matching (each credit returned exactly once)
- No double returns
- Timestamp monotonicity (credit issue times strictly increasing)
- Turn index correctness (0-indexed, sequential per session)
- Session metadata consistency
- Return-after-issue ordering

Additional invariants tested separately:
- Concurrency bounds (TestConcurrencyInvariants)
- Rate limiting bounds (TestRateLimitingInvariants)

These invariants are checked across multiple timing modes (rate-limited, burst,
user-centric) to ensure consistent behavior.
"""

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    TimingTestConfig,
    build_burst_command,
    build_timing_command,
)
from tests.harness.analyzers import (
    ConcurrencyAnalyzer,
    InvariantChecker,
    TimingAnalyzer,
)
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestCreditLifecycleInvariants:
    """Tests for credit lifecycle invariants across all timing modes."""

    @pytest.mark.parametrize(
        "arrival_pattern",
        ["constant", "poisson"],
    )
    def test_credit_lifecycle_with_rate(self, cli: AIPerfCLI, arrival_pattern: str):
        """Verify credit lifecycle invariants with rate limiting."""
        config = TimingTestConfig(
            num_sessions=30,
            qps=400.0,
            turns_per_session=3,
        )
        cmd = build_timing_command(config, arrival_pattern=arrival_pattern)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        checker = InvariantChecker(runner)

        for name, passed, reason in checker.run_all_checks():
            assert passed, f"Invariant '{name}' failed: {reason}"

    def test_credit_lifecycle_burst_mode(self, cli: AIPerfCLI):
        """Verify credit lifecycle invariants in burst mode."""
        config = TimingTestConfig(
            num_sessions=40,
            qps=0,
            turns_per_session=3,
            concurrency=10,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        checker = InvariantChecker(runner)

        for name, passed, reason in checker.run_all_checks():
            assert passed, f"Invariant '{name}' failed: {reason}"

    def test_credit_lifecycle_user_centric(self, cli: AIPerfCLI):
        """Verify credit lifecycle invariants in user-centric mode."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=400.0,
            turns_per_session=4,
        )
        cmd = build_timing_command(config, user_centric_rate=config.qps)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        checker = InvariantChecker(runner)

        for name, passed, reason in checker.run_all_checks():
            assert passed, f"Invariant '{name}' failed: {reason}"


@pytest.mark.component_integration
class TestConcurrencyInvariants:
    """Tests for concurrency limit enforcement invariants."""

    @pytest.mark.parametrize(
        "concurrency,qps",
        [
            (2, 500.0),   # Low concurrency, high rate (extreme backpressure)
            (5, 300.0),   # Moderate concurrency, high rate
            (10, 200.0),  # Higher concurrency, moderate rate
        ],
    )  # fmt: skip
    def test_concurrency_limit_never_exceeded(
        self, cli: AIPerfCLI, concurrency: int, qps: float
    ):
        """Verify concurrency limit is NEVER exceeded, even under backpressure.

        When rate >> concurrency capacity, the system must queue requests
        rather than exceeding the concurrency limit.
        """
        config = TimingTestConfig(
            num_sessions=50,
            qps=qps,
            concurrency=concurrency,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        analyzer = ConcurrencyAnalyzer(result)
        max_observed = analyzer.get_max_concurrent()

        assert max_observed <= concurrency, (
            f"Concurrency limit VIOLATED: observed {max_observed}, limit was {concurrency}. "
            f"This indicates a race condition in the concurrency semaphore."
        )

    @pytest.mark.parametrize(
        "prefill_concurrency,qps",
        [
            (1, 500.0),   # Single prefill slot with high rate
            (2, 400.0),   # Two prefill slots with high rate
            (3, 300.0),   # Three prefill slots
        ],
    )  # fmt: skip
    def test_prefill_concurrency_limit_never_exceeded(
        self, cli: AIPerfCLI, prefill_concurrency: int, qps: float
    ):
        """Verify prefill concurrency limit is NEVER exceeded.

        Prefill phase is typically much shorter than decode, so this limit
        is harder to saturate but still must be enforced.
        """
        config = TimingTestConfig(
            num_sessions=40,
            qps=qps,
            concurrency=20,  # High overall concurrency
            prefill_concurrency=prefill_concurrency,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        analyzer = ConcurrencyAnalyzer(result)
        max_prefill = analyzer.get_max_prefill_concurrent()

        assert max_prefill <= prefill_concurrency, (
            f"Prefill concurrency limit VIOLATED: observed {max_prefill}, limit was {prefill_concurrency}. "
            f"This indicates a race condition in the prefill semaphore."
        )


@pytest.mark.component_integration
class TestRateLimitingInvariants:
    """Tests for rate limiting enforcement invariants."""

    def test_actual_rate_approximately_matches_configured(self, cli: AIPerfCLI):
        """Verify actual throughput approximately matches configured rate.

        The actual rate should be within a reasonable tolerance of the configured
        rate. Too fast = rate limiting not working. Too slow = performance bug.
        """
        target_qps = 400.0
        config = TimingTestConfig(
            num_sessions=50,
            qps=target_qps,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()

        assert len(issue_times) >= 10, "Not enough data for rate analysis"

        # Calculate actual rate
        duration_sec = (issue_times[-1] - issue_times[0]) / NANOS_PER_SECOND
        actual_rate = (len(issue_times) - 1) / duration_sec if duration_sec > 0 else 0

        # Allow 40% tolerance (rate limiting has inherent variability)
        tolerance = target_qps * 0.4
        assert abs(actual_rate - target_qps) < tolerance, (
            f"Actual rate {actual_rate:.1f} QPS differs from target {target_qps:.1f} QPS "
            f"by more than {tolerance:.1f} (40%). "
            f"Rate limiting may not be working correctly."
        )

    def test_rate_not_exceeded_burst_stress(self, cli: AIPerfCLI):
        """Verify rate is not exceeded even when concurrency allows it.

        With high concurrency, the rate limiter must still enforce the QPS limit.
        """
        target_qps = 50.0
        config = TimingTestConfig(
            num_sessions=40,
            qps=target_qps,
            concurrency=40,  # High concurrency - could issue fast if rate limiting fails
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 5, "Not enough data for gap analysis"

        # Minimum expected gap = 1/QPS
        min_expected_gap = 1.0 / target_qps

        # Check mean gap is at least the expected (allowing 30% tolerance for jitter)
        mean_gap = timing.calculate_mean(gaps)
        assert mean_gap >= min_expected_gap * 0.7, (
            f"Mean gap {mean_gap:.4f}s is less than 70% of expected {min_expected_gap:.4f}s. "
            f"Rate limiting may be allowing requests faster than configured."
        )
