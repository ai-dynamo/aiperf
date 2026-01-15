# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Component integration tests for sticky credit router load balancing.

These tests verify that the StickyCreditRouter:
1. Distributes credits fairly across workers (least-loaded selection)
2. Maintains sticky routing (all turns of a session go to same worker)
3. Handles various worker counts and workload patterns

The router uses random selection among tied workers (same load level)
to achieve fair distribution without round-robin bias.
"""

import pytest

from tests.component_integration.timing.conftest import (
    TimingTestConfig,
    defaults,
)
from tests.harness.analyzers import LoadBalancingAnalyzer
from tests.harness.utils import AIPerfCLI


def build_multi_worker_command(
    config: TimingTestConfig,
    workers_max: int,
    *,
    arrival_pattern: str = "constant",
) -> str:
    """Build command with specified number of workers."""
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {config.num_sessions} \
            --osl {config.osl} \
            --extra-inputs ignore_eos:true \
            --workers-max {workers_max} \
            --random-seed 42 \
            --ui {defaults.ui}
    """

    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )

    if config.concurrency is not None:
        cmd += f" --concurrency {config.concurrency}"

    if config.qps > 0:
        cmd += f" --request-rate {config.qps} --request-rate-mode {arrival_pattern}"

    return cmd


def build_burst_multi_worker_command(
    config: TimingTestConfig,
    workers_max: int,
) -> str:
    """Build burst mode command with specified number of workers."""
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --num-sessions {config.num_sessions} \
            --concurrency {config.concurrency} \
            --osl {config.osl} \
            --extra-inputs ignore_eos:true \
            --workers-max {workers_max} \
            --random-seed 42 \
            --ui {defaults.ui}
    """

    if config.turns_per_session > 1:
        cmd += (
            f" --session-turns-mean {config.turns_per_session} --session-turns-stddev 0"
        )

    return cmd


@pytest.mark.component_integration
class TestFairDistributionSingleTurn:
    """Test fair credit distribution with single-turn sessions.

    Single-turn sessions test pure load balancing without sticky routing effects.
    Each session is a single credit, so distribution should be near-perfect.
    """

    @pytest.mark.parametrize(
        "num_sessions,workers_max,tolerance_pct",
        [
            (100, 2, 20.0),   # 50 credits/worker expected
            (100, 3, 25.0),   # ~33 credits/worker expected
            (100, 5, 30.0),   # 20 credits/worker expected
            pytest.param(200, 3, 20.0, marks=pytest.mark.slow),  # Higher volume
            pytest.param(150, 5, 25.0, marks=pytest.mark.slow),  # 30 credits/worker
        ],
    )  # fmt: skip
    def test_fair_distribution_constant_rate(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        workers_max: int,
        tolerance_pct: float,
    ):
        """Test fair credit distribution with constant rate."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=400.0,
            turns_per_session=1,
        )
        cmd = build_multi_worker_command(
            config, workers_max, arrival_pattern="constant"
        )
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)
        assert analyzer.num_workers == workers_max, (
            f"Expected {workers_max} workers, got {analyzer.num_workers}"
        )

        passed, reason = analyzer.verify_fair_distribution(tolerance_pct=tolerance_pct)
        assert passed, reason

    @pytest.mark.parametrize(
        "num_sessions,workers_max,tolerance_pct",
        [
            (100, 2, 25.0),
            (100, 3, 30.0),
            (100, 5, 35.0),
        ],
    )  # fmt: skip
    def test_fair_distribution_poisson_rate(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        workers_max: int,
        tolerance_pct: float,
    ):
        """Test fair distribution with Poisson (random) arrival rate."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=400.0,
            turns_per_session=1,
        )
        cmd = build_multi_worker_command(config, workers_max, arrival_pattern="poisson")
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)
        assert analyzer.num_workers == workers_max

        passed, reason = analyzer.verify_fair_distribution(tolerance_pct=tolerance_pct)
        assert passed, reason

    @pytest.mark.parametrize(
        "num_sessions,workers_max,concurrency,tolerance_pct",
        [
            (100, 2, 10, 20.0),
            (100, 3, 15, 25.0),
            (100, 5, 20, 30.0),
        ],
    )  # fmt: skip
    def test_fair_distribution_burst_mode(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        workers_max: int,
        concurrency: int,
        tolerance_pct: float,
    ):
        """Test fair distribution in burst mode (no rate limiting)."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,
            turns_per_session=1,
            concurrency=concurrency,
        )
        cmd = build_burst_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)
        assert analyzer.num_workers == workers_max

        passed, reason = analyzer.verify_fair_distribution(tolerance_pct=tolerance_pct)
        assert passed, reason


@pytest.mark.component_integration
class TestStickyRoutingMultiTurn:
    """Test sticky routing for multi-turn sessions.

    Multi-turn sessions must route all turns to the same worker for
    conversation state caching. Verify sticky routing is maintained.
    """

    @pytest.mark.parametrize(
        "num_sessions,turns_per_session,workers_max",
        [
            (20, 3, 2),   # 20 sessions × 3 turns = 60 credits
            (20, 3, 3),
            (20, 5, 3),   # More turns per session
            pytest.param(30, 4, 5, marks=pytest.mark.slow),  # More workers
            (50, 2, 3),   # Many short sessions
        ],
    )  # fmt: skip
    def test_sticky_routing_maintained(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        turns_per_session: int,
        workers_max: int,
    ):
        """Verify all turns of each session go to the same worker."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=400.0,
            turns_per_session=turns_per_session,
        )
        cmd = build_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=config.timeout)

        expected_requests = num_sessions * turns_per_session
        assert result.request_count == expected_requests

        analyzer = LoadBalancingAnalyzer(result)

        # Verify sticky routing
        passed, reason = analyzer.verify_sticky_routing()
        assert passed, reason

        # Verify correct number of sessions per worker
        assert analyzer.total_sessions == num_sessions

    @pytest.mark.parametrize(
        "num_sessions,turns_per_session,workers_max,concurrency",
        [
            (30, 3, 3, 15),
            (40, 4, 5, 20),
        ],
    )  # fmt: skip
    def test_sticky_routing_burst_mode(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        turns_per_session: int,
        workers_max: int,
        concurrency: int,
    ):
        """Verify sticky routing under burst mode (high concurrency)."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,
            turns_per_session=turns_per_session,
            concurrency=concurrency,
        )
        cmd = build_burst_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=config.timeout)

        expected_requests = num_sessions * turns_per_session
        assert result.request_count == expected_requests

        analyzer = LoadBalancingAnalyzer(result)
        passed, reason = analyzer.verify_sticky_routing()
        assert passed, reason


@pytest.mark.component_integration
class TestMultiWorkerScaling:
    """Test load balancing scales correctly with worker count."""

    @pytest.mark.parametrize(
        "workers_max",
        [2, 3, 5, 7],
    )
    def test_all_workers_receive_credits(
        self,
        cli: AIPerfCLI,
        workers_max: int,
    ):
        """Verify all workers receive at least some credits."""
        # Use enough sessions to ensure all workers get work
        num_sessions = workers_max * 20
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=500.0,
            turns_per_session=1,
        )
        cmd = build_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)
        assert analyzer.num_workers == workers_max, (
            f"Expected all {workers_max} workers to receive credits, "
            f"but only {analyzer.num_workers} did"
        )

        # Every worker should have received credits
        credits_per_worker = analyzer.credits_per_worker()
        for worker_id, count in credits_per_worker.items():
            assert count > 0, f"Worker {worker_id} received no credits"

    def test_single_worker_gets_all_credits(self, cli: AIPerfCLI):
        """Verify single worker receives all credits."""
        num_sessions = 50
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=400.0,
            turns_per_session=1,
        )
        cmd = build_multi_worker_command(config, workers_max=1)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)
        assert analyzer.num_workers == 1
        assert analyzer.total_credits == num_sessions


@pytest.mark.component_integration
@pytest.mark.stress
@pytest.mark.slow
class TestHighLoadBalancing:
    """Test load balancing under high load conditions."""

    def test_high_session_count_fair_distribution(self, cli: AIPerfCLI):
        """Test fair distribution with many sessions."""
        workers_max = 5
        num_sessions = 500
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=500.0,  # High throughput
            turns_per_session=1,
            timeout=120.0,
        )
        cmd = build_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        # With high volume, distribution should be very fair
        passed, reason = analyzer.verify_fair_distribution(tolerance_pct=15.0)
        assert passed, reason

        # Check distribution stats
        stats = analyzer.get_distribution_stats()
        # Coefficient of variation should be low for fair distribution
        assert stats["cv"] < 0.15, (
            f"Distribution CV {stats['cv']:.3f} too high (expected < 0.15)"
        )

    def test_high_concurrency_fair_distribution(self, cli: AIPerfCLI):
        """Test fair distribution under high concurrency burst."""
        workers_max = 5
        num_sessions = 200
        concurrency = 50
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=0,
            turns_per_session=1,
            concurrency=concurrency,
            timeout=90.0,
        )
        cmd = build_burst_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        passed, reason = analyzer.verify_fair_distribution(tolerance_pct=25.0)
        assert passed, reason


@pytest.mark.component_integration
class TestResearchFairnessMetrics:
    """Tests using research-grade fairness metrics."""

    @pytest.mark.parametrize(
        "num_sessions,workers_max,min_jfi",
        [
            (100, 2, 0.95),   # 2 workers should be very fair
            (100, 5, 0.90),   # More workers, slightly more variance
            (200, 5, 0.93),
            pytest.param(500, 5, 0.97, marks=[pytest.mark.stress, pytest.mark.slow])
        ],
    )  # fmt: skip
    def test_jains_fairness_index(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        workers_max: int,
        min_jfi: float,
    ):
        """Test Jain's Fairness Index meets research-grade thresholds.

        JFI of 0.9+ indicates highly fair distribution.
        JFI of 0.95+ indicates near-perfect fairness.
        """
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=500.0,
            turns_per_session=1,
        )
        cmd = build_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)
        jfi = analyzer.jains_fairness_index()

        assert jfi >= min_jfi, (
            f"Jain's Fairness Index {jfi:.4f} below threshold {min_jfi}. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )

    @pytest.mark.stress
    @pytest.mark.slow
    def test_multi_turn_jains_fairness(self, cli: AIPerfCLI):
        """Test JFI with multi-turn sessions (sticky routing adds complexity)."""
        workers_max = 4
        num_sessions = 80
        turns_per_session = 4
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=400.0,
            turns_per_session=turns_per_session,
        )
        cmd = build_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=config.timeout)

        expected_requests = num_sessions * turns_per_session
        assert result.request_count == expected_requests

        analyzer = LoadBalancingAnalyzer(result)

        # Sticky routing maintains fairness because first turns are fair
        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.85, (
            f"JFI {jfi:.4f} too low for multi-turn. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )

        # Sticky routing must still work
        sticky_passed, sticky_reason = analyzer.verify_sticky_routing()
        assert sticky_passed, sticky_reason


# =============================================================================
# Adversarial Tests (Merged from test_load_balancing_adversarial.py)
# =============================================================================


@pytest.mark.component_integration
class TestPrimeNumberAdversarial:
    """Adversarial tests with prime numbers that don't divide evenly."""

    @pytest.mark.parametrize(
        "num_sessions,workers_max",
        [
            (97, 7),    # Prime sessions, prime workers
            (101, 11),  # Both prime
        ],
    )  # fmt: skip
    def test_prime_combinations(
        self,
        cli: AIPerfCLI,
        num_sessions: int,
        workers_max: int,
    ):
        """Prime numbers can't divide evenly but should still be fair."""
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=500.0,
            turns_per_session=1,
        )
        cmd = build_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=120.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)

        # Even with primes, JFI should be high
        jfi = analyzer.jains_fairness_index()
        assert jfi >= 0.90, (
            f"JFI {jfi:.4f} too low for prime combination {num_sessions}/{workers_max}. "
            f"Distribution: {analyzer.credits_per_worker()}"
        )


@pytest.mark.component_integration
@pytest.mark.stress
class TestTimingPatternAdversarial:
    """Adversarial tests with specific timing patterns."""

    @pytest.mark.slow
    def test_original_problematic_scenario(self, cli: AIPerfCLI):
        """Original problematic scenario: 10 sessions, 5 workers, slow QPS.

        Before the fix, this produced distribution like {3, 4, 1, 1, 1} with JFI=0.714.
        With the new algorithm using total_sent_credits as tie-breaker,
        this should produce near-perfect distribution with high JFI.

        Note: We use >= 0.95 instead of == 1.0 to account for timing jitter on CI
        that can occasionally cause minor imbalances.
        """
        num_sessions = 10
        workers_max = 5
        config = TimingTestConfig(
            num_sessions=num_sessions,
            qps=500.0,
            turns_per_session=1,
        )
        cmd = build_multi_worker_command(config, workers_max)
        result = cli.run_sync(cmd, timeout=30.0)

        assert result.request_count == num_sessions

        analyzer = LoadBalancingAnalyzer(result)
        jfi = analyzer.jains_fairness_index()
        distribution = sorted(analyzer.credits_per_worker().values())

        # With the new algorithm, should achieve near-perfect fairness
        # Allow minor variance due to CI timing jitter (0.95+ is still excellent)
        assert jfi >= 0.95, (
            f"JFI {jfi:.4f} below threshold 0.95. Distribution: {distribution}. "
            f"Expected near-perfect fairness with new tie-breaking algorithm."
        )
