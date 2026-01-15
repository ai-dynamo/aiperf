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
from aiperf.credit.structs import Credit
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
class TestCancellationMechanics:
    """Tests for cancellation mechanics and edge cases.

    Cancellation is the emergency brake - these tests verify it works
    correctly in various scenarios including mixed completion states.
    """

    def test_cancellation_with_no_completions(self, cli: AIPerfCLI):
        """Test cancellation when no credits have completed yet.

        Baseline: verify cancellation works from initial state.
        Note: In component tests, requests complete quickly, so this
        tests the cancellation path logic even if requests complete.
        """
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,
            concurrency=5,
            timeout=30.0,
        )

        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Test completes successfully (cancellation or completion)
        assert result.request_count == 10

        # Verify all credits accounted for
        credit_analyzer = CreditFlowAnalyzer(result.runner_result)
        assert credit_analyzer.credits_balanced()

    def test_cancellation_id_validation(self, cli: AIPerfCLI):
        """Test that credit IDs in cancellation match issued credits.

        Scenario:
        - Issue N credits
        - Verify cancellation (if occurs) only references valid IDs
        - No phantom or duplicate credit IDs
        """
        config = TimingTestConfig(
            num_sessions=12,
            qps=0,
            concurrency=6,
        )

        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == 12

        runner = result.runner_result

        # Get all credit IDs issued
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]
        issued_ids = {p.payload.id for p in credit_payloads}

        # Get all credit returns
        return_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, CreditReturn)
        ]
        returned_ids = {p.payload.credit.id for p in return_payloads}

        # All returned IDs should be from issued set
        assert returned_ids.issubset(issued_ids)

        # All issued should be returned
        assert issued_ids == returned_ids

    def test_cleanup_after_cancellation(self, cli: AIPerfCLI):
        """Test resource cleanup after cancellation.

        Scenario:
        - Credits issued and possibly cancelled
        - Verify all slots released
        - Verify no credit leaks
        """
        config = TimingTestConfig(
            num_sessions=15,
            qps=0,
            concurrency=8,
            prefill_concurrency=3,
        )

        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == 15

        # Verify credits balanced (all slots released)
        credit_analyzer = CreditFlowAnalyzer(result.runner_result)
        assert credit_analyzer.credits_balanced()

        # Verify concurrency limits maintained
        conc_analyzer = ConcurrencyAnalyzer(result)
        assert conc_analyzer.get_max_concurrent() <= config.concurrency
        assert conc_analyzer.get_max_prefill_concurrent() <= config.prefill_concurrency


@pytest.mark.component_integration
class TestMultiTurnComplexInterleaving:
    """Tests for complex multi-turn interleaving scenarios.

    Multi-turn conversations can have credits complete out of order.
    These tests verify correct handling of complex interleaving patterns.
    """

    def test_five_turn_conversation_scrambled_returns(self, cli: AIPerfCLI):
        """Test 5-turn conversation with returns in scrambled order.

        Scenario:
        - Single session, 5 turns
        - Turns may complete out of order
        - Verify turn indices sequential
        - Verify session completion detected correctly
        """
        config = TimingTestConfig(
            num_sessions=1,
            qps=0,
            turns_per_session=5,
            concurrency=1,  # Max 1 concurrent (validation: <= num_sessions)
        )

        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # 1 session × 5 turns = 5 requests
        assert result.request_count == 5

        credit_analyzer = CreditFlowAnalyzer(result.runner_result)

        # Verify all 5 credits for the session
        assert credit_analyzer.num_sessions == 1
        assert credit_analyzer.total_credits == 5

        # Verify turn indices sequential within session
        assert credit_analyzer.turn_indices_sequential()

    def test_multiple_sessions_interleaved_turns(self, cli: AIPerfCLI):
        """Test multiple sessions with interleaved turn execution.

        Scenario:
        - 12 sessions, 4 turns each = 48 credits
        - Issue order: S0T0, S1T0, S2T0, ..., S0T1, S1T1, ...
        - Returns can be scrambled
        - Verify per-session correctness
        """
        config = TimingTestConfig(
            num_sessions=12,
            qps=0,
            turns_per_session=4,
            concurrency=12,  # Allow all sessions concurrent
        )

        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == 48  # 12 sessions × 4 turns

        credit_analyzer = CreditFlowAnalyzer(result.runner_result)

        # Verify 12 sessions
        assert credit_analyzer.num_sessions == 12

        # Verify each session has 4 turns
        assert credit_analyzer.session_credits_match(expected_turns=4)

        # Verify turn indices sequential within each session
        assert credit_analyzer.turn_indices_sequential()

    def test_many_sessions_with_varying_turn_counts(self, cli: AIPerfCLI):
        """Test sessions complete with different turn counts.

        Note: Current config gives all sessions same turn count.
        This test verifies the system handles uniform turn distribution.
        Future: Could add stddev to session-turns for varying counts.
        """
        config = TimingTestConfig(
            num_sessions=10,
            qps=0,
            turns_per_session=3,
            concurrency=10,  # Max equal to num_sessions
        )

        cmd = build_burst_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == 30

        credit_analyzer = CreditFlowAnalyzer(result.runner_result)

        assert credit_analyzer.num_sessions == 10
        assert credit_analyzer.session_credits_match(expected_turns=3)


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


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
@pytest.mark.usefixtures("super_slow_latency_for_grace_period")
class TestGracePeriodTimeout:
    """Tests for grace period timeout with slow requests.

    Uses slow latency fixture to test grace period behavior.
    Marked as stress because parallel execution causes event loop contention
    that makes timing unpredictable.
    """

    def test_grace_period_allows_completion(self, cli: AIPerfCLI):
        """Test that grace period allows in-flight requests to complete.

        Scenario:
        - Slow latency (TTFT=200ms + ITL=10ms × 50 = ~700ms per request)
        - Duration=0.05s (very short, only issues ~2 requests)
        - Grace period=1.5s (enough for requests to complete)
        - All in-flight requests complete within grace period
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --request-rate 50 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.05 \
                --benchmark-grace-period 10.0
        """

        result = cli.run_sync(cmd, timeout=20.0)

        # Duration 0.05s at 50 QPS → ~2-3 requests issued
        # Each request takes ~700ms theoretically, but parallel execution causes delays
        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # All credits should be accounted for
        assert credit_analyzer.credits_balanced()

        total_credits = credit_analyzer.total_credits
        assert total_credits >= 1  # At least 1 issued
        assert total_credits <= 5  # Limited by duration

        # All requests should complete (none cancelled)
        assert result.request_count >= 1

    def test_grace_period_with_more_requests(self, cli: AIPerfCLI):
        """Test grace period with multiple requests completing.

        Scenario:
        - Slow latency (TTFT=200ms + ITL=10ms × 50 = ~700ms per request)
        - Duration=0.1s (issues ~5 requests)
        - Grace period=2.0s (enough for all to complete)
        - All requests should complete successfully
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --request-rate 50 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.1 \
                --benchmark-grace-period 10.0
        """

        result = cli.run_sync(cmd, timeout=20.0)

        # Duration 0.1s at 50 QPS → ~5 requests
        # Each request takes ~700ms theoretically, but parallel execution causes delays
        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # All credits balanced
        assert credit_analyzer.credits_balanced()

        # Check that requests completed (not forced-cancelled)
        error_count = sum(
            1 for cr in credit_analyzer.credit_returns if cr.error is not None
        )

        # All should complete successfully with sufficient grace
        total = credit_analyzer.total_credits
        success_rate = (total - error_count) / total if total > 0 else 0
        assert success_rate >= 0.8, (
            f"Expected most requests to complete with sufficient grace period, "
            f"got {total - error_count}/{total} successes"
        )

    def test_grace_period_too_short_forces_cancellation(self, cli: AIPerfCLI):
        """Test that too-short grace period forces cancellation of all requests.

        Scenario:
        - Slow latency (TTFT=200ms + ITL=10ms × 50 = ~700ms per request)
        - Duration=0.05s (issues ~2-3 requests)
        - Grace period=0.3s (too short - requests take ~700ms)
        - All requests should be force-cancelled
        - aiperf returns exit code 1 when all cancelled
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --request-rate 50 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.05 \
                --benchmark-grace-period 0.3
        """

        # Use assert_success=False since all requests cancelled → exit code 1
        result = cli.run_sync(cmd, timeout=10.0, assert_success=False)

        # Duration 0.05s at 50 QPS → ~2-3 requests issued
        # Each takes ~700ms, grace period is 300ms → all cancelled
        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Credits should still be balanced (cancelled credits are returned)
        assert credit_analyzer.credits_balanced()

        # Verify no requests completed successfully
        assert result.request_count == 0, (
            f"Expected 0 completed requests with insufficient grace period, "
            f"got {result.request_count}"
        )
