# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariant tests for timing strategies.

These tests verify fundamental correctness properties that MUST hold regardless
of timing mode, rate, concurrency, or other configuration. Failures here indicate
serious bugs in the timing framework.

Invariants tested:
1. **Credit Lifecycle**: Each credit is issued exactly once and returned exactly once
2. **Credit ID Uniqueness**: No duplicate credit IDs within a run
3. **Timestamp Monotonicity**: Credit issue times are strictly increasing
4. **Turn Index Correctness**: Turn indices are 0-indexed and sequential per session
5. **Session Isolation**: Each session's credits have consistent metadata
6. **Concurrency Bounds**: Actual concurrency never exceeds configured limits
7. **Rate Limiting Bounds**: Actual rate approximately matches configured rate
8. **No Credit Leaks**: Every issued credit is eventually returned

These invariants are checked across ALL timing modes to ensure consistent behavior.

References:
- Race conditions in rate limiting: https://www.thegreenreport.blog/articles/using-stress-tests-to-catch-race-conditions-in-api-rate-limiting-logic/
- Statistical validation: https://www.itl.nist.gov/div898/handbook/eda/section3/eda35f.htm
- Reliable benchmarking: https://dl.acm.org/doi/10.1007/s10009-017-0469-y
"""

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.timing.conftest import (
    TimingTestConfig,
    build_timing_command,
    defaults,
)
from tests.harness.analyzers import (
    ConcurrencyAnalyzer,
    InvariantChecker,
    TimingAnalyzer,
)
from tests.harness.utils import AIPerfCLI


def build_burst_command(config: TimingTestConfig) -> str:
    """Build burst mode command."""
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {config.num_sessions} \
            --concurrency {config.concurrency} \
            --osl {config.osl} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """
    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )
    return cmd


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
class TestTimestampInvariants:
    """Tests for timestamp ordering and consistency."""

    def test_credit_issue_ordering_under_high_concurrency(self, cli: AIPerfCLI):
        """Verify timestamp monotonicity under high concurrency stress.

        High concurrency is most likely to expose race conditions in timestamp
        generation or credit issuance ordering.
        """
        config = TimingTestConfig(
            num_sessions=100,
            qps=0,
            concurrency=50,  # High concurrency stress
            timeout=90.0,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        checker = InvariantChecker(runner)

        passed, reason = checker.check_timestamp_monotonicity()
        assert passed, reason

    def test_credit_issue_ordering_high_rate(self, cli: AIPerfCLI):
        """Verify timestamp monotonicity under high rate stress.

        High QPS is most likely to expose race conditions in rapid-fire
        credit issuance.
        """
        config = TimingTestConfig(
            num_sessions=80,
            qps=500.0,  # High rate stress
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        checker = InvariantChecker(runner)

        passed, reason = checker.check_timestamp_monotonicity()
        assert passed, reason


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

        # Check mean gap is at least the expected (allowing 20% tolerance for jitter)
        mean_gap = timing.calculate_mean(gaps)
        assert mean_gap >= min_expected_gap * 0.8, (
            f"Mean gap {mean_gap:.4f}s is less than 80% of expected {min_expected_gap:.4f}s. "
            f"Rate limiting may be allowing requests faster than configured."
        )


@pytest.mark.component_integration
class TestTurnIndexInvariants:
    """Tests for turn index correctness across all modes."""

    @pytest.mark.parametrize(
        "turns_per_session",
        [2, 5, 10],
    )
    def test_turn_indices_0_indexed_and_sequential(
        self, cli: AIPerfCLI, turns_per_session: int
    ):
        """Verify turn indices start at 0 and are sequential.

        This is a fundamental correctness requirement for multi-turn conversations.
        """
        config = TimingTestConfig(
            num_sessions=15,
            qps=400.0,
            turns_per_session=turns_per_session,
        )
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        runner: AIPerfRunnerResultWithSharedBus = result.runner_result
        checker = InvariantChecker(runner)

        passed, reason = checker.check_turn_index_correctness()
        assert passed, reason
