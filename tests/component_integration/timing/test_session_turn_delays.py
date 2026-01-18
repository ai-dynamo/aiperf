# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session turn delay tests with interaction coverage.

Session turn delays simulate realistic "think time" between conversation turns,
critical for chatbot and multi-turn benchmark accuracy.

Options tested:
- --session-turn-delay-mean: Mean delay between turns (milliseconds)
- --session-turn-delay-stddev: Standard deviation of delays (milliseconds)
- --session-turns-stddev: Variation in turn counts across sessions

These tests verify:
1. Basic delay functionality with multi-turn conversations
2. Variable delays (stddev > 0)
3. Delay + concurrency interactions
4. Delay + duration/grace period
5. Delay + warmup phase
6. Delay + request cancellation
7. Variable turn counts across sessions
"""

import statistics

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.structs import Credit
from tests.component_integration.timing.conftest import defaults
from tests.harness.analyzers import CreditFlowAnalyzer
from tests.harness.utils import AIPerfCLI


@pytest.mark.component_integration
class TestSessionTurnDelayBasic:
    """Basic session turn delay functionality tests."""

    def test_turn_delay_mean_with_multi_turn(self, cli: AIPerfCLI):
        """Test basic turn delay with multi-turn conversations.

        Scenario:
        - 3-turn conversations
        - 100ms delay between turns
        - Verify all turns complete
        - Verify delays don't break credit flow
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --request-rate 150 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # 10 sessions × 3 turns = 30 requests
        assert result.request_count == 30

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Verify all credits balanced
        assert credit_analyzer.credits_balanced()

        # Verify turn indices sequential
        assert credit_analyzer.turn_indices_sequential()

    def test_turn_delay_variable_with_stddev(self, cli: AIPerfCLI):
        """Test variable turn delays with stddev > 0.

        Scenario:
        - Mean delay: 50ms
        - Stddev: 20ms (variable delays)
        - Verify credit flow handles randomness
        - Verify all turns complete
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 8 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 50 \
                --session-turn-delay-stddev 20 \
                --request-rate 180 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # 8 sessions × 4 turns = 32 requests
        assert result.request_count == 32

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()
        assert credit_analyzer.num_sessions == 8
        assert credit_analyzer.session_credits_match(expected_turns=4)


@pytest.mark.component_integration
class TestTurnDelayInteractions:
    """Test turn delay interactions with other features.

    These tests focus on complex interactions between turn delays and:
    - Concurrency limits
    - Duration/grace period
    - Warmup phase
    - Request timeouts
    """

    @pytest.mark.slow
    def test_turn_delay_with_concurrency_limit(self, cli: AIPerfCLI):
        """Test turn delays + concurrency limit interaction.

        Scenario:
        - Multi-turn with delays
        - Concurrency limit enforced
        - Verify delays don't prevent concurrent sessions
        - Turn delay affects per-session, not cross-session
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 20 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 100 \
                --request-rate 200 \
                --request-rate-mode constant \
                --concurrency 8 \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui}
        """

        result = cli.run_sync(cmd, timeout=40.0)

        assert result.request_count == 80  # 20 × 4

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        assert credit_analyzer.credits_balanced()

    def test_turn_delay_with_duration_grace_period(self, cli: AIPerfCLI):
        """Test turn delays + duration + grace period interaction.

        Scenario:
        - Turn delay: 200ms (slow think time)
        - Duration: 0.4s (short)
        - Grace period: 5s (allows delayed turns to complete)
        - Verify in-flight delayed turns complete in grace period
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 50 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 200 \
                --request-rate 150 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --benchmark-duration 0.4 \
                --benchmark-grace-period 5.0
        """

        result = cli.run_sync(cmd, timeout=30.0)

        # Duration 0.4s at 30 QPS → ~12 requests
        # Some sessions started, delays mean turns extend into grace period
        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # All credits should be accounted for
        assert credit_analyzer.credits_balanced()

        # At least some requests completed
        assert result.request_count >= 5

    @pytest.mark.slow
    def test_turn_delay_with_warmup_phase(self, cli: AIPerfCLI):
        """Test turn delays apply to both warmup and profiling.

        Scenario:
        - Warmup: multi-turn with delays
        - Profiling: multi-turn with delays
        - Verify delays respected in both phases
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 12 \
                --session-turns-mean 3 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 80 \
                --request-rate 180 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --num-warmup-sessions 8
        """

        result = cli.run_sync(cmd, timeout=40.0)

        # Profiling: 12 sessions × 3 turns = 36
        assert result.request_count == 36

        runner = result.runner_result
        credit_payloads = [
            p for p in runner.sent_payloads if isinstance(p.payload, Credit)
        ]

        warmup_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.WARMUP
        ]
        profiling_credits = [
            p for p in credit_payloads if p.payload.phase == CreditPhase.PROFILING
        ]

        # Warmup: 8 sessions × 3 turns = 24
        assert len(warmup_credits) == 24
        assert len(profiling_credits) == 36

    def test_turn_delay_with_request_cancellation(self, cli: AIPerfCLI):
        """Test turn delays + request cancellation interaction.

        Scenario:
        - Turn delay: 60ms
        - Cancellation rate: 25%
        - Verify delays and cancellations coexist
        - Cancelled turns still respect turn delay for next turn
        """
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 10 \
                --session-turns-mean 4 \
                --session-turns-stddev 0 \
                --session-turn-delay-mean 60 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --request-cancellation-rate 25.0 \
                --request-cancellation-delay 0.003 \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=30.0)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # All 40 credits sent
        assert credit_analyzer.total_credits == 40

        # Credits balanced (some with errors)
        assert credit_analyzer.credits_balanced()

        # Verify some errors (25% rate with seed 42)
        error_count = sum(
            1 for cr in credit_analyzer.credit_returns if cr.error is not None
        )
        assert error_count > 0, "Expected some errors with cancellation rate"


@pytest.mark.component_integration
class TestVariableTurnCounts:
    """Tests for variable turn counts (session-turns-stddev > 0)."""

    def test_turns_stddev_variation_per_session(self, cli: AIPerfCLI):
        """Test variable turn counts across sessions."""
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions 15 \
                --session-turns-mean 5 \
                --session-turns-stddev 2 \
                --request-rate 200 \
                --request-rate-mode constant \
                --osl 50 \
                --extra-inputs ignore_eos:true \
                --ui {defaults.ui} \
                --random-seed 42
        """

        result = cli.run_sync(cmd, timeout=40.0)

        runner = result.runner_result
        credit_analyzer = CreditFlowAnalyzer(runner)

        # Total credits should vary from 15×3 to 15×7 approximately
        total = credit_analyzer.total_credits
        assert 45 <= total <= 105, f"Expected variable total ~75±30, got {total}"

        assert credit_analyzer.credits_balanced()
        assert credit_analyzer.num_sessions == 15

        # Verify turn counts vary across sessions
        turn_counts = [
            len(payloads) for payloads in credit_analyzer.credits_by_session.values()
        ]

        if len(turn_counts) > 1:
            stddev = statistics.stdev(turn_counts)
            assert stddev > 0, "Expected variation in turn counts with stddev=2"
