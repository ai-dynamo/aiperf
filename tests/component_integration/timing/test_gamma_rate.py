# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for Gamma rate timing mode.

Gamma rate mode generalizes Poisson arrivals with a tunable smoothness parameter:
- smoothness = 1.0: Equivalent to Poisson (exponential inter-arrivals, CV = 1.0)
- smoothness < 1.0: More bursty/clustered arrivals (higher CV)
- smoothness > 1.0: More regular/smooth arrivals (lower CV)

Key characteristics:
- Mean inter-arrival time = 1/rate (same as Poisson)
- CV = 1/sqrt(smoothness) for Gamma distribution
- Matches vLLM's burstiness parameter for realistic traffic modeling

Tests cover:
- Basic functionality at various QPS and smoothness levels
- Statistical distribution verification (CV matches theory)
- Smoothness comparison (higher smoothness = lower variance)
- Multi-turn conversations
- Concurrency interactions
- Equivalence to Poisson at smoothness=1.0
"""

import pytest

from tests.component_integration.timing.conftest import (
    BaseConcurrencyTests,
    BaseCreditFlowTests,
    TimingTestConfig,
    defaults,
)
from tests.harness.analyzers import (
    StatisticalAnalyzer,
    TimingAnalyzer,
)
from tests.harness.utils import AIPerfCLI

DEFAULT_RANDOM_SEED = 42


def build_gamma_command(
    config: TimingTestConfig,
    smoothness: float,
    *,
    random_seed: int = DEFAULT_RANDOM_SEED,
    extra_args: str = "",
) -> str:
    """Build a CLI command for gamma rate tests.

    Args:
        config: Test configuration
        smoothness: Gamma smoothness parameter (1.0 = Poisson)
        random_seed: Random seed for deterministic tests (default: 42)
        extra_args: Additional CLI arguments
    """
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {config.num_sessions} \
            --request-rate {config.qps} \
            --arrival-pattern gamma \
            --arrival-smoothness {smoothness} \
            --osl {config.osl} \
            --extra-inputs ignore_eos:true \
            --random-seed {random_seed} \
            --ui {defaults.ui}
    """

    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )

    if config.concurrency is not None:
        cmd += f" --concurrency {config.concurrency}"

    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"

    if extra_args:
        cmd += f" {extra_args}"

    return cmd


@pytest.mark.component_integration
class TestGammaRateBasic:
    """Basic functionality tests for Gamma rate timing."""

    @pytest.mark.parametrize(
        "num_sessions,qps,smoothness",
        [
            (20, 100.0, 1.0),   # smoothness=1.0 (Poisson equivalent)
            (25, 100.0, 2.0),   # smoothness=2.0 (smoother)
            (30, 150.0, 4.0),   # smoothness=4.0 (much smoother)
            (20, 100.0, 0.5),   # smoothness=0.5 (burstier)
        ],
    )  # fmt: skip
    def test_gamma_rate_completes(
        self, cli: AIPerfCLI, num_sessions: int, qps: float, smoothness: float
    ):
        """Test Gamma rate mode completes at various configurations."""
        config = TimingTestConfig(num_sessions=num_sessions, qps=qps)
        cmd = build_gamma_command(config, smoothness=smoothness)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions
        assert result.has_streaming_metrics

    def test_gamma_rate_multi_turn(self, cli: AIPerfCLI):
        """Test Gamma rate with multi-turn conversations."""
        config = TimingTestConfig(
            num_sessions=15,
            qps=75.0,
            turns_per_session=4,
        )
        cmd = build_gamma_command(config, smoothness=2.0)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.expected_requests
        assert result.has_streaming_metrics


@pytest.mark.component_integration
class TestGammaRateCreditFlow(BaseCreditFlowTests):
    """Credit flow verification for Gamma rate timing.

    Inherits common credit flow tests from BaseCreditFlowTests.
    Tests: credits_balanced, credits_per_session, turn_indices_sequential
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Gamma rate timing command with default smoothness=2.0."""
        return build_gamma_command(config, smoothness=2.0)


@pytest.mark.component_integration
class TestGammaRateStatistics:
    """Statistical distribution tests for Gamma rate mode."""

    def test_gamma_distribution_characteristics(self, cli: AIPerfCLI):
        """Verify Gamma distribution statistical properties.

        Uses the StatisticalAnalyzer.is_approximately_gamma method
        for rigorous distribution verification.
        """
        smoothness = 2.0
        config = TimingTestConfig(num_sessions=50, qps=100.0)
        cmd = build_gamma_command(config, smoothness=smoothness)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == config.num_sessions

        timing = TimingAnalyzer(result)
        issue_times = timing.get_credit_issue_times_ns()
        gaps = timing.calculate_gaps_sec(issue_times)

        assert len(gaps) >= 20, (
            f"Insufficient data for Gamma analysis: got {len(gaps)} gaps, need >= 20"
        )

        passed, reason = StatisticalAnalyzer.is_approximately_gamma(
            gaps, expected_rate=config.qps, smoothness=smoothness, tolerance_pct=40.0
        )
        assert passed, f"Distribution not Gamma-like: {reason}"


@pytest.mark.component_integration
class TestGammaRateWithConcurrency(BaseConcurrencyTests):
    """Tests for Gamma rate with concurrency limits.

    Inherits common concurrency tests from BaseConcurrencyTests.
    Tests: test_with_concurrency_limit, test_with_prefill_concurrency,
           test_multi_turn_with_concurrency
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build Gamma rate timing command with default smoothness=2.0."""
        return build_gamma_command(config, smoothness=2.0)


@pytest.mark.component_integration
class TestVLLMBurstinessAlias:
    """Tests verifying --vllm-burstiness alias works."""

    def test_vllm_burstiness_alias(self, cli: AIPerfCLI):
        """Test that --vllm-burstiness is an alias for --arrival-smoothness.

        vLLM uses "burstiness" for the same parameter, so we support it.
        The key validation is that the command executes successfully.
        """
        config = TimingTestConfig(num_sessions=30, qps=100.0)

        # Use --vllm-burstiness instead of --arrival-smoothness
        cmd = f"""
            aiperf profile \
                --model {defaults.model} \
                --streaming \
                --num-sessions {config.num_sessions} \
                --request-rate {config.qps} \
                --arrival-pattern gamma \
                --vllm-burstiness 4.0 \
                --osl {config.osl} \
                --extra-inputs ignore_eos:true \
                --random-seed 42 \
                --ui {defaults.ui}
        """
        result = cli.run_sync(cmd, timeout=config.timeout)

        # Key validation: command executed successfully with --vllm-burstiness
        assert result.request_count == config.num_sessions
