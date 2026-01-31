# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures and helpers for timing strategy tests.

This module provides:
- Test configuration dataclasses (TimingTestConfig, RealisticLatencyConfig)
- Command building utilities (build_timing_command, build_burst_command)
- Assertion helpers for common test patterns
- Base test classes for shared test logic (BaseCreditFlowTests, BaseConcurrencyTests)
- Session-scoped realistic_latency fixture for FakeTransport timing simulation

Analyzer classes (CreditFlowAnalyzer, TimingAnalyzer, etc.) are imported from
tests.harness.analyzers for reuse across all test modules.
"""

from dataclasses import dataclass

import pytest
from aiperf_mock_server.config import MockServerConfig

from aiperf.plugin.enums import ArrivalPattern
from tests.component_integration.conftest import (
    AIPerfRunnerResultWithSharedBus,
)
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.analyzers import (
    ConcurrencyAnalyzer,
    CreditFlowAnalyzer,
    LoadBalancingAnalyzer,
)
from tests.harness.fake_transport import FakeTransport
from tests.harness.utils import AIPerfCLI, AIPerfResults


@pytest.fixture(autouse=True, scope="session")
def realistic_latency():
    """Enable realistic latency simulation for FakeTransport.

    Sets TTFT=5ms and ITL=1ms to simulate realistic prefill and decode phases.
    This is essential for testing prefill concurrency limits accurately.
    """
    original = FakeTransport._DEFAULT_CONFIG
    FakeTransport._DEFAULT_CONFIG = MockServerConfig(
        ttft=5.0,  # 5ms time to first token (prefill)
        itl=1.0,  # 1ms inter-token latency (decode)
    )
    yield FakeTransport._DEFAULT_CONFIG
    FakeTransport._DEFAULT_CONFIG = original


@dataclass
class RealisticLatencyConfig:
    """Configuration for realistic latency simulation.

    Used to calculate expected request durations and concurrency behavior.
    Must match the values in the realistic_latency fixture.
    """

    ttft_ms: float = 5.0  # Time to first token (prefill phase)
    itl_ms: float = 1.0  # Inter-token latency (decode phase)

    @property
    def ttft_sec(self) -> float:
        """TTFT in seconds."""
        return self.ttft_ms / 1000.0

    @property
    def itl_sec(self) -> float:
        """ITL in seconds."""
        return self.itl_ms / 1000.0

    def request_duration_sec(self, osl: int) -> float:
        """Calculate expected request duration.

        Request duration ≈ TTFT + (OSL × ITL)
        """
        return self.ttft_sec + (osl * self.itl_sec)

    def expected_max_concurrent(self, qps: float, osl: int) -> float:
        """Calculate expected maximum concurrency for rate-limited modes.

        For steady-state rate-limited traffic:
        max_concurrent ≈ QPS × request_duration
        """
        if qps <= 0:
            return float("inf")  # Burst mode - no rate limiting
        return qps * self.request_duration_sec(osl)

    def expected_max_prefill_concurrent(self, qps: float) -> float:
        """Calculate expected maximum prefill concurrency.

        max_prefill_concurrent ≈ QPS × TTFT
        """
        if qps <= 0:
            return float("inf")  # Burst mode - no rate limiting
        return qps * self.ttft_sec


# Global instance matching the realistic_latency fixture
REALISTIC_LATENCY = RealisticLatencyConfig()


@dataclass
class TimingTestConfig:
    """Configuration for a timing test scenario."""

    num_sessions: int
    qps: float
    turns_per_session: int = 1
    concurrency: int | None = None
    prefill_concurrency: int | None = None
    osl: int = 50
    timeout: float = 60.0

    @property
    def expected_requests(self) -> int:
        """Calculate expected total requests."""
        return self.num_sessions * self.turns_per_session

    @property
    def expected_gap_sec(self) -> float:
        """Calculate expected gap between requests at this QPS."""
        return 1.0 / self.qps

    @property
    def expected_user_gap_sec(self) -> float:
        """Calculate expected per-user gap for user-centric mode."""
        return self.num_sessions / self.qps

    @property
    def expected_request_duration_sec(self) -> float:
        """Calculate expected request duration based on realistic latency."""
        return REALISTIC_LATENCY.request_duration_sec(self.osl)

    @property
    def expected_max_concurrent(self) -> float:
        """Calculate expected maximum concurrency for this config.

        For burst mode (qps=0), returns inf (limited only by concurrency setting).
        For rate-limited modes, returns QPS × request_duration.
        """
        return REALISTIC_LATENCY.expected_max_concurrent(self.qps, self.osl)

    @property
    def expected_max_prefill_concurrent(self) -> float:
        """Calculate expected maximum prefill concurrency.

        For burst mode (qps=0), returns inf (limited only by prefill_concurrency setting).
        For rate-limited modes, returns QPS × TTFT.
        """
        return REALISTIC_LATENCY.expected_max_prefill_concurrent(self.qps)

    def will_hit_concurrency_limit(self) -> bool:
        """Check if this config will hit the concurrency limit.

        For burst mode: always True (if num_sessions > concurrency)
        For rate-limited: True if expected_max_concurrent >= concurrency
        """
        if self.concurrency is None:
            return False
        if self.qps <= 0:  # Burst mode
            return self.num_sessions > self.concurrency
        return self.expected_max_concurrent >= self.concurrency

    def will_hit_prefill_limit(self) -> bool:
        """Check if this config will hit the prefill concurrency limit.

        For burst mode: always True (if num_sessions > prefill_concurrency)
        For rate-limited: True if expected_max_prefill_concurrent >= prefill_concurrency
        """
        if self.prefill_concurrency is None:
            return False
        if self.qps <= 0:  # Burst mode
            return self.num_sessions > self.prefill_concurrency
        return self.expected_max_prefill_concurrent >= self.prefill_concurrency


# Default random seed for deterministic Poisson tests
DEFAULT_RANDOM_SEED = 42


# Convenience function for building CLI commands
def build_timing_command(
    config: TimingTestConfig,
    *,
    arrival_pattern: ArrivalPattern | None = None,
    user_centric_rate: float | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    extra_args: str = "",
) -> str:
    """Build a CLI command for timing tests.

    Args:
        config: Test configuration
        arrival_pattern: Arrival pattern (constant, poisson)
        user_centric_rate: User-centric rate QPS
        random_seed: Random seed for deterministic Poisson timing (default: 42)
        extra_args: Additional CLI arguments

    Returns:
        CLI command string
    """
    cmd = f"""
        aiperf profile \
            --model {defaults.model} \
            --streaming \
            --osl {config.osl} \
            --extra-inputs ignore_eos:true \
            --ui {defaults.ui}
    """

    # User-centric mode requires multi-turn conversations (session_turns_mean >= 2).
    # For single-turn workloads, it degenerates to request-rate mode with extra overhead.
    turns = config.turns_per_session
    if user_centric_rate is not None:
        turns = max(turns, 2)  # Minimum 2 turns for user-centric mode

    if turns > 1:
        cmd += f" --session-turns-mean {turns} --session-turns-stddev 0"

    if config.concurrency is not None:
        cmd += f" --concurrency {config.concurrency}"

    if config.prefill_concurrency is not None:
        cmd += f" --prefill-concurrency {config.prefill_concurrency}"

    if user_centric_rate is not None:
        # User-centric rate: use --benchmark-duration as stop condition, --num-users for user count
        cmd += f" --num-users {config.num_sessions}"
        cmd += f" --user-centric-rate {user_centric_rate}"
        cmd += " --benchmark-duration 1.0 --benchmark-grace-period 0.0"
    else:
        # Non user-centric modes use --num-sessions as stop condition
        cmd += f" --num-sessions {config.num_sessions}"
        if config.qps > 0:
            cmd += f" --request-rate {config.qps}"
            if arrival_pattern:
                cmd += f" --arrival-pattern {arrival_pattern}"

    # Add random seed for deterministic Poisson timing
    if random_seed is not None:
        cmd += f" --random-seed {random_seed}"

    if extra_args:
        cmd += f" {extra_args}"

    return cmd


def build_burst_command(config: TimingTestConfig) -> str:
    """Build burst mode command (no rate limiting, concurrency-limited only)."""
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


# ============================================================================
# Assertion Helper Functions
# ============================================================================
# These helpers reduce boilerplate in timing tests by encapsulating common
# assertion patterns with descriptive error messages.


def assert_request_count(
    result: AIPerfResults, expected: int, message: str = ""
) -> None:
    """Assert request count matches expected with detailed error message.

    Args:
        result: Test result containing request count
        expected: Expected number of completed requests
        message: Optional context message for assertion failure
    """
    actual = result.request_count
    context = f"{message}: " if message else ""
    assert actual == expected, (
        f"{context}Expected {expected} requests, got {actual}. "
        f"Total records: {len(result.jsonl)}"
    )


def assert_credits_balanced(result: AIPerfResults) -> None:
    """Assert all issued credits were returned (no credit leaks).

    Args:
        result: Test result with runner_result containing credit flow data
    """
    runner: AIPerfRunnerResultWithSharedBus = result.runner_result
    analyzer = CreditFlowAnalyzer(runner)
    assert analyzer.credits_balanced(), (
        f"Credits not balanced: {analyzer.total_credits} issued, "
        f"{analyzer.total_returns} returned. "
        f"Leaked: {analyzer.total_credits - analyzer.total_returns}"
    )


def assert_concurrency_limit_respected(
    result: AIPerfResults,
    limit: int,
    prefill: bool = False,
) -> None:
    """Assert concurrency never exceeded the specified limit.

    Args:
        result: Test result
        limit: Maximum allowed concurrency
        prefill: If True, check prefill concurrency; else check total concurrency
    """
    analyzer = ConcurrencyAnalyzer(result)
    max_concurrent = (
        analyzer.get_max_prefill_concurrent()
        if prefill
        else analyzer.get_max_concurrent()
    )
    limit_type = "prefill" if prefill else "total"
    assert max_concurrent <= limit, (
        f"Max {limit_type} concurrency {max_concurrent} exceeded limit {limit}"
    )


def assert_concurrency_limit_hit(
    result: AIPerfResults,
    limit: int,
    prefill: bool = False,
) -> None:
    """Assert concurrency limit was actually reached (not artificially low).

    This validates that the test configuration was correct and the limit
    was exercised, not just respected.

    Args:
        result: Test result
        limit: Expected concurrency limit that should be reached
        prefill: If True, check prefill concurrency; else check total concurrency
    """
    analyzer = ConcurrencyAnalyzer(result)
    max_concurrent = (
        analyzer.get_max_prefill_concurrent()
        if prefill
        else analyzer.get_max_concurrent()
    )
    limit_type = "prefill" if prefill else "total"
    assert max_concurrent == limit, (
        f"Max {limit_type} concurrency {max_concurrent} did not reach limit {limit}. "
        f"Test configuration may be incorrect (QPS too low, not enough sessions, etc.)"
    )


def assert_fair_load_distribution(
    result: AIPerfResults,
    num_workers: int,
    tolerance_pct: float = 30.0,
) -> None:
    """Assert requests were fairly distributed across workers.

    Args:
        result: Test result
        num_workers: Expected number of workers
        tolerance_pct: Allowed deviation from perfect balance (default 30%)
    """
    analyzer = LoadBalancingAnalyzer(result)
    distribution = analyzer.credits_per_worker()

    assert len(distribution) == num_workers, (
        f"Expected {num_workers} workers, got {len(distribution)}: {list(distribution.keys())}"
    )

    passed, reason = analyzer.verify_fair_distribution(tolerance_pct=tolerance_pct)
    assert passed, f"Load not fairly distributed: {reason}"


def assert_session_credits_match(
    result: AIPerfResults,
    expected_turns: int,
) -> None:
    """Assert each session received exactly the expected number of credits.

    Args:
        result: Test result with credit flow data
        expected_turns: Expected number of turns (credits) per session
    """
    runner: AIPerfRunnerResultWithSharedBus = result.runner_result
    analyzer = CreditFlowAnalyzer(runner)
    assert analyzer.session_credits_match(expected_turns), (
        f"Not all sessions have {expected_turns} credits. "
        f"Session credit counts: {dict(list(analyzer.session_credit_counts.items())[:5])}"
    )


def assert_turn_indices_sequential(result: AIPerfResults) -> None:
    """Assert turn indices are sequential (0, 1, 2, ...) within each session.

    Args:
        result: Test result with credit flow data
    """
    runner: AIPerfRunnerResultWithSharedBus = result.runner_result
    analyzer = CreditFlowAnalyzer(runner)
    assert analyzer.turn_indices_sequential(), (
        "Turn indices are not sequential within sessions"
    )


# ============================================================================
# Base Test Classes for DRY Test Implementation
# ============================================================================


class BaseCreditFlowTests:
    """Base class for credit flow tests across all timing modes.

    This abstract base class provides common credit flow test implementations
    that work across all timing strategies (constant, poisson, gamma, user-centric, etc.).

    Subclasses must:
    1. Inherit from this class AND pytest.mark.component_integration marker class
    2. Implement build_command() to create timing-specific commands

    Example usage:
        @pytest.mark.component_integration
        class TestConstantRateCreditFlow(BaseCreditFlowTests):
            def build_command(self, config: TimingTestConfig) -> str:
                return build_timing_command(config, arrival_pattern="constant")

    This eliminates ~40 lines of duplicate test code per timing mode.
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build timing-specific command for credit flow tests.

        This abstract method must be implemented by subclasses to provide
        the appropriate command for their timing mode.

        Args:
            config: Test configuration with num_sessions, qps, turns, etc.

        Returns:
            Command string to pass to cli.run_sync()
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_command()"
        )

    def test_credits_balanced(self, cli: AIPerfCLI):
        """Verify all credits sent are returned (no credit leaks)."""
        config = TimingTestConfig(num_sessions=20, qps=100.0)
        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_credits_balanced(result)

    def test_credits_per_session(self, cli: AIPerfCLI):
        """Verify each session gets expected number of credits."""
        config = TimingTestConfig(
            num_sessions=12,
            qps=60.0,
            turns_per_session=3,
        )
        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        # For multi-turn tests, total requests = num_sessions * turns_per_session
        assert_request_count(result, config.expected_requests, "Total requests")
        assert_session_credits_match(result, config.turns_per_session)

    def test_turn_indices_sequential(self, cli: AIPerfCLI):
        """Verify turn indices are sequential per session."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=50.0,
            turns_per_session=5,
        )
        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_turn_indices_sequential(result)


class BaseConcurrencyTests:
    """Base class for concurrency limit tests across all timing modes.

    This abstract base class provides common concurrency test implementations
    that work across all timing strategies (constant, poisson, gamma, burst).

    Mathematical basis for parameter selection:
    - Request duration = TTFT + OSL×ITL = 5ms + 50×1ms = 55ms
    - For limit to be hit: QPS × request_duration >= concurrency
    - Required QPS >= concurrency / 0.055 ≈ concurrency × 18.2
    - For prefill limit: QPS × TTFT >= prefill_concurrency

    Subclasses must:
    1. Inherit from this class AND pytest.mark.component_integration marker class
    2. Implement build_command() to create timing-specific commands

    Example usage:
        @pytest.mark.component_integration
        class TestConstantRateWithConcurrency(BaseConcurrencyTests):
            def build_command(self, config: TimingTestConfig) -> str:
                return build_timing_command(config, arrival_pattern="constant")

    This eliminates ~100 lines of duplicate test code per timing mode.
    """

    def build_command(self, config: TimingTestConfig) -> str:
        """Build timing-specific command for concurrency tests.

        This abstract method must be implemented by subclasses to provide
        the appropriate command for their timing mode.

        Args:
            config: Test configuration with num_sessions, qps, concurrency, etc.

        Returns:
            Command string to pass to cli.run_sync()
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement build_command()"
        )

    @pytest.mark.parametrize(
        "concurrency,qps",
        [
            (3, 100.0),   # 100 × 0.055 = 5.5 > 3 ✓
            (5, 100.0),   # 100 × 0.055 = 5.5 >= 5 ✓
            (8, 200.0),   # 200 × 0.055 = 11 > 8 ✓
            (12, 300.0),  # 300 × 0.055 = 16.5 > 12 ✓
        ],
    )  # fmt: skip
    def test_with_concurrency_limit(self, cli: AIPerfCLI, concurrency: int, qps: float):
        """Test timing mode respects and reaches concurrency limit."""
        config = TimingTestConfig(
            num_sessions=50,
            qps=qps,
            concurrency=concurrency,
            osl=50,  # Need longer OSL to hit concurrency limits
        )

        # Validate test parameters will hit the limit
        assert config.will_hit_concurrency_limit(), (
            f"Test config won't hit concurrency limit: "
            f"expected_max={config.expected_max_concurrent:.1f}, concurrency={concurrency}"
        )

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)
        assert_concurrency_limit_respected(result, concurrency)
        assert_concurrency_limit_hit(result, concurrency)

    def test_with_prefill_concurrency(self, cli: AIPerfCLI):
        """Test timing mode with prefill concurrency limit.

        Mathematical basis:
        - TTFT = 5ms, so max_prefill_concurrent = QPS × 0.005
        - For prefill_concurrency=2 to be hit, need QPS >= 400
        - Using QPS=500 gives max_prefill = 2.5 > 2 ✓
        """
        prefill_concurrency = 2
        qps = 500.0  # 500 × 0.005 = 2.5 > 2
        config = TimingTestConfig(
            num_sessions=30,
            qps=qps,
            prefill_concurrency=prefill_concurrency,
            osl=50,  # Need longer OSL for consistent timing
        )

        # Validate test parameters will hit the prefill limit
        assert config.will_hit_prefill_limit(), (
            f"Test config won't hit prefill limit: "
            f"expected_max={config.expected_max_prefill_concurrent:.1f}, "
            f"prefill_concurrency={prefill_concurrency}"
        )

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.num_sessions)
        assert_concurrency_limit_respected(result, prefill_concurrency, prefill=True)
        assert_concurrency_limit_hit(result, prefill_concurrency, prefill=True)

    def test_multi_turn_with_concurrency(self, cli: AIPerfCLI):
        """Test multi-turn conversations with concurrency."""
        config = TimingTestConfig(
            num_sessions=10,
            qps=100.0,
            turns_per_session=4,
            concurrency=4,  # 100 × 0.055 = 5.5 > 4 ✓
            osl=50,  # Need longer OSL to hit concurrency limits
        )

        assert config.will_hit_concurrency_limit()

        cmd = self.build_command(config)
        result = cli.run_sync(cmd, timeout=config.timeout)

        assert_request_count(result, config.expected_requests)
        assert_concurrency_limit_hit(result, config.concurrency)
