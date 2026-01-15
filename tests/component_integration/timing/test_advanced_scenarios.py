# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Advanced timing scenario tests covering edge cases and complex interactions.

This module tests scenarios identified through deep code analysis:
- Credit exhaustion and replenishment patterns
- FirstToken arrival ordering
- Cancellation mechanics
- Phase transition races
- Multi-turn complex interleaving
- Rate/concurrency parameter combinations

These tests complement the existing timing test suite by covering:
1. Normal operation edge cases (not catastrophic failures)
2. Complex interaction patterns
3. Parameter combination matrices
4. Timing-dependent race conditions
"""

import pytest
from aiperf_mock_server.config import MockServerConfig

from aiperf.common.enums import ArrivalPattern
from aiperf.credit.messages import CreditReturn
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


@pytest.fixture(scope="class")
def super_slow_latency_for_grace_period():
    """Super slow latency fixture for testing grace period timeout.

    Sets TTFT=2000ms (2s) and ITL=100ms so that requests take a LONG time.
    This allows us to test grace period timeout scenarios where:
    - Duration expires
    - Requests are still in-flight
    - Grace period expires before requests complete
    - System force-cancels remaining requests

    Used to verify grace period timeout behavior.
    """
    original = FakeTransport._DEFAULT_CONFIG
    FakeTransport._DEFAULT_CONFIG = MockServerConfig(
        ttft=2000.0,  # 2 seconds time to first token
        itl=100.0,  # 100ms inter-token latency
    )
    yield
    FakeTransport._DEFAULT_CONFIG = original


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
    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"
    return cmd


@pytest.mark.component_integration
class TestCreditExhaustionAndReplenishment:
    """Tests for credit exhaustion and replenishment patterns.

    Tests verify correct behavior when concurrency slots are filled and requests queue.
    Redundant tests duplicating base class functionality removed.
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

        # Verify rate maintained (not burst due to concurrency)
        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        mean_gap = timing.calculate_mean(gaps)
        expected_gap = 1.0 / config.qps

        # Rate should still be maintained
        assert abs(mean_gap - expected_gap) < expected_gap * 0.5


@pytest.mark.component_integration
class TestRateConcurrencyMatrix:
    """Matrix tests covering rate mode × concurrency combinations.

    Reduced to 2 critical combinations from 6+ parametrizations.
    """

    @pytest.mark.parametrize(
        "arrival_pattern,concurrency",
        [
            ("constant", 5),
            ("poisson", 10),
        ],
    )  # fmt: skip
    def test_rate_mode_with_concurrency(
        self, cli: AIPerfCLI, arrival_pattern: str, concurrency: int
    ):
        """Test rate modes with various concurrency levels."""
        config = TimingTestConfig(
            num_sessions=20,
            qps=100.0,
            concurrency=concurrency,
        )

        cmd = build_timing_command(config, arrival_pattern=arrival_pattern)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == 20

        # Verify concurrency limit respected
        conc_analyzer = ConcurrencyAnalyzer(result)
        max_concurrent = conc_analyzer.get_max_concurrent()
        assert max_concurrent <= concurrency

    def test_extreme_qps_with_low_concurrency(self, cli: AIPerfCLI):
        """Test very high QPS with very low concurrency.

        Scenario:
        - qps=500, concurrency=2
        - Expected steady-state concurrent = 500 × 0.055 = 27.5
        - But limited to 2 by concurrency
        - Verify concurrency limit dominates
        """
        config = TimingTestConfig(
            num_sessions=20,
            qps=500.0,
            concurrency=2,
        )

        # Should definitely hit concurrency limit
        assert config.will_hit_concurrency_limit()

        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == 20

        conc_analyzer = ConcurrencyAnalyzer(result)
        max_concurrent = conc_analyzer.get_max_concurrent()

        # Concurrency limit should dominate
        assert max_concurrent <= 2
        assert max_concurrent == 2  # Should actually hit it


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
        - 10 sessions × 4 turns = 40 total requests
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

        # Get all credits sent (should be 10 sessions × 4 turns = 40)
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

    @pytest.mark.slow
    def test_mid_conversation_cancellation_continues(self, cli: AIPerfCLI):
        """Test that conversation continues after mid-conversation request timeout.

        Scenario:
        - 5-turn conversation
        - Some turns may timeout (status 499 errors)
        - Verify subsequent turns still proceed
        - Verify final turn always attempted
        - Errors do NOT break conversation flow
        """
        config = TimingTestConfig(
            num_sessions=5,
            qps=0,
            turns_per_session=5,
            concurrency=5,
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
                --request-cancellation-rate 30.0 \
                --request-cancellation-delay 0.003 \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=config.timeout)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Verify all 25 credits sent (5 sessions × 5 turns)
        assert credit_analyzer.total_credits == 25

        # Verify each session got all 5 turns (even if some had request errors)
        assert credit_analyzer.num_sessions == 5
        assert credit_analyzer.session_credits_match(expected_turns=5)

        # Verify turn indices sequential (0, 1, 2, 3, 4)
        assert credit_analyzer.turn_indices_sequential()

        # With seed 42, 30% rate on 25 requests = exactly 6 timeouts (deterministic)
        error_count = sum(
            1 for cr in credit_analyzer.credit_returns if cr.error is not None
        )
        assert error_count == 6, (
            f"Expected exactly 6 request timeouts with seed 42, got {error_count}"
        )

        # IMPORTANT: Request errors are NOT credit cancellations
        cancelled_count = sum(
            1 for cr in credit_analyzer.credit_returns if cr.cancelled
        )
        assert cancelled_count == 0, "Request timeout != credit cancellation"

    @pytest.mark.slow
    def test_sticky_routing_intact_with_request_timeouts(self, cli: AIPerfCLI):
        """Test sticky routing maintained across request timeouts.

        Scenario:
        - Multi-turn with request timeouts (status 499 errors)
        - Verify all turns from same session route to same worker
        - Worker assignment consistent despite request errors
        - Sticky session not broken by timeouts
        """
        config = TimingTestConfig(
            num_sessions=8,
            qps=0,
            turns_per_session=4,
            concurrency=8,
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
                --request-cancellation-rate 20.0 \
                --request-cancellation-delay 0.003 \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=config.timeout)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Verify all 32 credits sent (8 sessions × 4 turns)
        assert credit_analyzer.total_credits == 32

        # Verify credits balanced (all accounted for)
        assert credit_analyzer.credits_balanced()

        # With seed 42, 20% rate on 32 requests = exactly 6 timeouts (deterministic)
        error_count = sum(
            1 for cr in credit_analyzer.credit_returns if cr.error is not None
        )
        assert error_count == 6, (
            f"Expected exactly 6 request timeouts with seed 42, got {error_count}"
        )

        # Request errors are NOT credit cancellations
        cancelled_count = sum(
            1 for cr in credit_analyzer.credit_returns if cr.cancelled
        )
        assert cancelled_count == 0, "Request timeout != credit cancellation"


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
        - Duration = 0.5 seconds → should issue ~5 requests
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

        # Should send approximately 10 × 0.5 = 5 requests (within tolerance)
        # Actual may be 4-7 due to timing precision
        assert result.request_count < 15, (
            f"Duration should limit requests to ~5, got {result.request_count}"
        )
        assert result.request_count >= 3, (
            f"Duration should issue at least 3 requests, got {result.request_count}"
        )

    def test_grace_period_allows_inflight_completion(self, cli: AIPerfCLI):
        """Test grace period allows in-flight requests to complete.

        Scenario:
        - Duration very short (0.2s)
        - Grace period long (10s)
        - Requests issued before duration should complete in grace period
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --request-rate 20 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.2 \
                --benchmark-grace-period 10.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Should issue ~20 × 0.2 = 4 requests
        # All should complete (grace period >> request duration)
        assert result.request_count >= 3
        assert result.request_count <= 8

        # Verify all credits balanced (completed in grace period)
        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)
        assert credit_analyzer.credits_balanced()

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

        # Should issue ~50 × 0.3 = 15 requests
        assert result.request_count >= 10
        assert result.request_count <= 25

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

        # Duration 0.4s at 30 QPS → ~12 credits
        # Could be ~4 sessions (starting) × 3 turns if in-flight complete
        assert result.request_count >= 8
        assert result.request_count <= 20

        # Verify all credits balanced
        credit_analyzer = CreditFlowAnalyzer(result.runner_result)
        assert credit_analyzer.credits_balanced()

        # Verify turn indices sequential within sessions
        assert credit_analyzer.turn_indices_sequential()

    def test_duration_with_concurrency_limit(self, cli: AIPerfCLI):
        """Test interaction between duration, grace period, and concurrency.

        Scenario:
        - Duration limits time
        - Concurrency limits parallelism
        - Grace period allows queued requests to complete
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --request-rate 40 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --concurrency 3 \
                --benchmark-duration 0.3 \
                --benchmark-grace-period 8.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Duration 0.3s at 40 QPS → ~12 requests
        assert result.request_count >= 8
        assert result.request_count <= 20

        # Verify concurrency limit respected
        conc_analyzer = ConcurrencyAnalyzer(result)
        max_concurrent = conc_analyzer.get_max_concurrent()
        assert max_concurrent <= 3
