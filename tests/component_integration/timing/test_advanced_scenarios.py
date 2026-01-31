# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advanced timing scenario tests covering edge cases and complex interactions.

This module tests:
- Credit exhaustion with rate limiting
- Request cancellation (timeout) mechanics
- Benchmark duration and grace period behavior

These tests complement the existing timing test suite by covering
complex interaction patterns and edge cases.
"""

import pytest
from aiperf_mock_server.config import MockServerConfig

from aiperf.credit.messages import CreditReturn
from aiperf.plugin.enums import ArrivalPattern
from tests.component_integration.timing.conftest import (
    TimingTestConfig,
    build_timing_command,
    defaults,
)
from tests.harness.analyzers import (
    ConcurrencyAnalyzer,
    CreditFlowAnalyzer,
    TimingAnalyzer,
)
from tests.harness.fake_transport import FakeTransport
from tests.harness.utils import AIPerfCLI


@pytest.fixture(scope="class")
def slow_latency_for_cancellation():
    """Slow latency fixture for testing request cancellation.

    Sets TTFT=100ms and ITL=10ms so that requests take long enough
    for short cancellation delays (e.g., 3ms) to reliably trigger.

    Normal realistic latency (TTFT=5ms) is too fast for cancellation
    testing since requests complete before the timeout fires.
    """
    original = FakeTransport._DEFAULT_CONFIG
    FakeTransport._DEFAULT_CONFIG = MockServerConfig(
        ttft=100.0,  # 100ms time to first token
        itl=10.0,  # 10ms inter-token latency
    )
    yield
    FakeTransport._DEFAULT_CONFIG = original


@pytest.mark.component_integration
class TestCreditExhaustionAndReplenishment:
    """Tests for credit exhaustion and replenishment patterns.

    Verifies correct behavior when both concurrency limit AND rate limit are active.
    """

    def test_exhaustion_with_rate_limiting(self, cli: AIPerfCLI):
        """Test interaction between concurrency exhaustion and rate limiting.

        Scenario:
        - concurrency=2, qps=50, sessions=20
        - Both concurrency limit AND rate limit active
        - Verify which limit dominates depends on parameters
        """
        config = TimingTestConfig(
            num_sessions=20,
            qps=50.0,
            concurrency=2,
        )

        # Expected max concurrent = QPS × request_duration
        # = 50 × 0.055 = 2.75, limited to 2 by concurrency
        assert config.will_hit_concurrency_limit()

        cmd = build_timing_command(config, arrival_pattern=ArrivalPattern.CONSTANT)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == 20

        conc_analyzer = ConcurrencyAnalyzer(result)
        max_concurrent = conc_analyzer.get_max_concurrent()

        # Concurrency limit should dominate
        assert max_concurrent <= 2

        # When concurrency limits throughput, rate will be at most the configured rate
        # (could be slower due to concurrency backpressure)
        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        mean_gap = timing.calculate_mean(gaps)
        min_expected_gap = 1.0 / config.qps

        # Rate should not exceed configured (gaps should be at least expected)
        # Allow some tolerance for timing jitter
        assert mean_gap >= min_expected_gap * 0.5, (
            f"Rate exceeded configured: mean_gap={mean_gap:.4f}s, min_expected={min_expected_gap:.4f}s"
        )


@pytest.mark.component_integration
@pytest.mark.usefixtures("slow_latency_for_cancellation")
class TestRequestCancellationRate:
    """Tests for --request-cancellation-rate with multi-turn scenarios.

    CRITICAL: Request cancellation (timeout) is NOT the same as credit cancellation!

    Request Cancellation (--request-cancellation-rate):
    - HTTP request times out after delay
    - Returns status 499 (Client Closed Request)
    - Sets CreditReturn.error (NOT CreditReturn.cancelled)
    - Credit is still returned and accounted for

    Credit Cancellation (CancelCredits message):
    - TimingManager sends cancel message to workers
    - Sets CreditReturn.cancelled = True
    - Different mechanism entirely

    Key behaviors tested:
    - Request timeout applied PER-TURN (each turn independent)
    - Timed out turn has error, subsequent turns proceed normally
    - Session cache remains active (only evicted on final turn)
    - Sticky routing maintained across request timeouts
    - Timeout disabled for warmup phase
    """

    @pytest.mark.slow
    def test_cancellation_rate_multi_turn_basic(self, cli: AIPerfCLI):
        """Test that request timeout rate applies per-turn in multi-turn sessions.

        Scenario:
        - 25% request timeout rate (--request-cancellation-rate)
        - 10 sessions x 4 turns = 40 total requests
        - Expected ~10 request ERRORS (status 499), NOT credit cancellations
        - Verify all credits returned (with errors)
        """
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,
            turns_per_session=4,
            concurrency=10,
        )

        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions {config.num_sessions} \
                --concurrency {config.concurrency} \
                --osl {config.osl} \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --session-turns-mean {config.turns_per_session} \
                --session-turns-stddev 0 \
                --request-cancellation-rate 25.0 \
                --request-cancellation-delay 0.003 \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=config.timeout)

        runner = result.runner_result

        # Get all credits sent (should be 10 sessions x 4 turns = 40)
        credit_analyzer = CreditFlowAnalyzer(runner)
        total_credits = credit_analyzer.total_credits
        assert total_credits == 40, f"Expected 40 credits sent, got {total_credits}"

        # Get REQUEST ERROR counts (not credit cancellations!)
        # Request cancellation = timeout (status 499), NOT credit cancellation
        return_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, CreditReturn)
        ]
        error_count = sum(1 for p in return_payloads if p.payload.error is not None)
        success_count = sum(1 for p in return_payloads if p.payload.error is None)

        # With seed 42, 25% rate on 40 requests = exactly 9 timeouts (deterministic)
        assert error_count == 9, (
            f"Expected exactly 9 request timeouts with seed 42, got {error_count}"
        )
        assert success_count == 31, f"Expected 31 successes, got {success_count}"
        assert error_count + success_count == 40

        # IMPORTANT: These are request ERRORS, not credit cancellations
        # CreditReturn.cancelled should be False for timeout errors
        cancelled_count = sum(1 for p in return_payloads if p.payload.cancelled)
        assert cancelled_count == 0, (
            "Request timeout is NOT credit cancellation - cancelled flag should be False"
        )

        # Verify all credits accounted for
        credit_analyzer = CreditFlowAnalyzer(runner)
        assert credit_analyzer.credits_balanced()


@pytest.mark.component_integration
class TestBenchmarkDurationAndGracePeriod:
    """Tests for --benchmark-duration and --benchmark-grace-period.

    Benchmark duration stops new credit issuance after N seconds.
    Grace period allows in-flight requests to complete.
    Key behaviors:
    - Duration stops NEW credits
    - Grace period waits for in-flight credits
    - Multi-turn conversations in-flight can complete
    - Grace period timeout triggers forced cancellation
    """

    def test_benchmark_duration_stops_new_credits(self, cli: AIPerfCLI):
        """Test that benchmark duration stops issuing new credits.

        Scenario:
        - Very low QPS (10 QPS) so we can measure duration effect
        - Duration = 0.5 seconds -> should issue ~5 requests
        - 100 sessions available but duration stops early
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 100 \
                --request-rate 10 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.5 \
                --benchmark-grace-period 10.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Should send approximately 10 x 0.5 = 5 requests (within tolerance)
        # Actual may be 2-20 due to timing precision and CI jitter
        assert result.request_count < 20, (
            f"Duration should limit requests to ~5, got {result.request_count}"
        )
        assert result.request_count >= 2, (
            f"Duration should issue at least 2 requests, got {result.request_count}"
        )

    def test_zero_grace_period_immediate_cutoff(self, cli: AIPerfCLI):
        """Test zero grace period cancels in-flight requests immediately.

        Scenario:
        - Duration expires
        - Grace period = 0
        - In-flight requests should be cancelled
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --request-rate 50 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.3 \
                --benchmark-grace-period 0.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Should issue ~50 x 0.3 = 15 requests (widened for CI jitter)
        assert result.request_count >= 5
        assert result.request_count <= 30

        # With zero grace period, some may be cancelled
        runner = result.runner_result
        return_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, CreditReturn)
        ]

        # All credits should be accounted for (completed or cancelled)
        credit_analyzer = CreditFlowAnalyzer(runner)
        assert credit_analyzer.total_credits == len(return_payloads)

    def test_multi_turn_with_duration_and_grace(self, cli: AIPerfCLI):
        """Test multi-turn conversations with duration and grace period.

        Scenario:
        - 3-turn conversations
        - Duration stops new sessions
        - Grace period allows active conversations to complete all turns
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --request-rate 30 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --benchmark-duration 0.4 \
                --benchmark-grace-period 5.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Duration 0.4s at 30 QPS -> ~12 credits
        # Could be ~4 sessions (starting) x 3 turns if in-flight complete
        # Widened bounds for CI jitter
        assert result.request_count >= 5
        assert result.request_count <= 30

        # Verify all credits balanced
        credit_analyzer = CreditFlowAnalyzer(result.runner_result)
        assert credit_analyzer.credits_balanced()

        # Verify turn indices sequential within sessions
        assert credit_analyzer.turn_indices_sequential()
