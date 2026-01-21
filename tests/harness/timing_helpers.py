# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timing test helper functions and assertion utilities.

This module provides reusable helper functions for timing strategy tests:
- Session validation helpers
- Command builders for timing tests
- Assertion helpers for common test patterns
- Configuration prediction helpers for test design

These utilities eliminate boilerplate and ensure consistent testing patterns
across all timing strategies (constant, poisson, gamma, burst, user-centric, etc.).
"""

from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.component_integration.timing.conftest import TimingTestConfig
from tests.harness.analyzers import ConcurrencyAnalyzer, CreditFlowAnalyzer
from tests.harness.utils import AIPerfResults

# ============================================================================
# Session Validation Helpers
# ============================================================================


def verify_no_interleaving_within_session(
    credit_analyzer: CreditFlowAnalyzer,
) -> tuple[bool, str]:
    """Verify no interleaving of turns within a single session.

    Each session's turns should be strictly sequential - no overlapping.
    Uses CapturedPayload.timestamp_ns (monotonic) for timing comparisons.

    Args:
        credit_analyzer: Analyzer containing credit flow data

    Returns:
        Tuple of (passed, reason) where passed is True if no violations found
    """
    for session_id, payloads in credit_analyzer.credits_by_session.items():
        if len(payloads) < 2:
            continue

        sorted_payloads = sorted(payloads, key=lambda p: p.timestamp_ns)
        for i in range(1, len(sorted_payloads)):
            prev = sorted_payloads[i - 1]
            curr = sorted_payloads[i]

            if curr.timestamp_ns <= prev.timestamp_ns:
                return False, (
                    f"Session {session_id}: turn {curr.payload.turn_index} captured at "
                    f"{curr.timestamp_ns} before/same as turn {prev.payload.turn_index} "
                    f"at {prev.timestamp_ns}"
                )

    return True, "All sessions have sequential turns"


def verify_sessions_can_interleave(
    credit_analyzer: CreditFlowAnalyzer,
) -> tuple[bool, str]:
    """Verify that different sessions CAN interleave globally.

    This confirms that the system is truly concurrent - different sessions'
    requests can be processed simultaneously rather than blocking each other.

    Uses CapturedPayload.timestamp_ns (monotonic) for timing comparisons.

    Args:
        credit_analyzer: Analyzer containing credit flow data

    Returns:
        Tuple of (passed, reason) where passed is True if interleaving detected
    """
    all_credits = []
    for session_id, payloads in credit_analyzer.credits_by_session.items():
        for p in payloads:
            all_credits.append((p.timestamp_ns, session_id))

    if len(all_credits) < 2:
        return True, "Not enough credits to check interleaving"

    sorted_credits = sorted(all_credits, key=lambda x: x[0])

    transitions = 0
    for i in range(1, len(sorted_credits)):
        if sorted_credits[i][1] != sorted_credits[i - 1][1]:
            transitions += 1

    num_sessions = credit_analyzer.num_sessions
    min_expected = num_sessions - 1

    if transitions < min_expected:
        return False, (
            f"Only {transitions} session transitions, expected at least {min_expected}"
        )

    return True, f"{transitions} session transitions"


# ============================================================================
# Command Builders
# ============================================================================

# Default random seed for deterministic Poisson tests
DEFAULT_RANDOM_SEED = 42


def build_timing_command(
    config: TimingTestConfig,
    *,
    arrival_pattern: str | None = None,
    user_centric_rate: float | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    extra_args: str = "",
) -> str:
    """Build a CLI command for timing tests.

    This is the primary command builder for most timing tests. It supports
    all timing modes and patterns through flexible configuration.

    Args:
        config: Test configuration (num_sessions, qps, concurrency, etc.)
        arrival_pattern: Arrival pattern (constant, poisson, gamma)
        user_centric_rate: User-centric rate QPS (enables user-centric mode)
        random_seed: Random seed for deterministic Poisson timing (default: 42)
        extra_args: Additional CLI arguments

    Returns:
        CLI command string ready for cli.run_sync()

    Example:
        config = TimingTestConfig(num_sessions=20, qps=100.0)
        cmd = build_timing_command(config, arrival_pattern="constant")
        result = cli.run_sync(cmd)
    """
    cmd = f"""
        aiperf profile \\
            --model {defaults.model} \\
            --streaming \\
            --osl {config.osl} \\
            --extra-inputs ignore_eos:true \\
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
    """Build burst mode command (no rate limiting, concurrency-limited only).

    Burst mode sends all requests as fast as possible, limited only by the
    concurrency setting. This is useful for testing maximum throughput and
    concurrency enforcement without rate limiting.

    Args:
        config: Test configuration (must include concurrency setting)

    Returns:
        CLI command string for burst mode

    Example:
        config = TimingTestConfig(
            num_sessions=100,
            qps=0,  # No rate limiting
            concurrency=10,
        )
        cmd = build_burst_command(config)
        result = cli.run_sync(cmd)
    """
    cmd = f"""
        aiperf profile \\
            --model {defaults.model} \\
            --streaming \\
            --num-sessions {config.num_sessions} \\
            --concurrency {config.concurrency} \\
            --osl {config.osl} \\
            --extra-inputs ignore_eos:true \\
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

    Raises:
        AssertionError: If request count doesn't match expected
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

    Raises:
        AssertionError: If credits are not balanced (leaked or double-returned)
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

    Raises:
        AssertionError: If concurrency limit was exceeded
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

    Raises:
        AssertionError: If concurrency limit was not reached
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
    tolerance_pct: float = 30.0,
) -> None:
    """Assert requests were fairly distributed across workers.

    Uses LoadBalancingAnalyzer to verify fair distribution based on credit
    allocation across workers.

    Args:
        result: Test result
        tolerance_pct: Allowed deviation from perfect balance (default 30%)

    Raises:
        AssertionError: If load distribution is unfair
    """
    from tests.harness.analyzers import LoadBalancingAnalyzer

    analyzer = LoadBalancingAnalyzer(result)
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

    Raises:
        AssertionError: If any session doesn't have the expected turn count
    """
    runner: AIPerfRunnerResultWithSharedBus = result.runner_result
    analyzer = CreditFlowAnalyzer(runner)
    assert analyzer.session_credits_match(expected_turns), (
        f"Not all sessions have {expected_turns} credits. "
        f"Sessions: {analyzer.num_sessions}, Total credits: {analyzer.total_credits}"
    )


def assert_turn_indices_sequential(result: AIPerfResults) -> None:
    """Assert turn indices are sequential (0, 1, 2, ...) within each session.

    Args:
        result: Test result with credit flow data

    Raises:
        AssertionError: If turn indices are not sequential
    """
    runner: AIPerfRunnerResultWithSharedBus = result.runner_result
    analyzer = CreditFlowAnalyzer(runner)
    assert analyzer.turn_indices_sequential(), (
        "Turn indices are not sequential within sessions"
    )


# ============================================================================
# Test Configuration Prediction Helpers
# ============================================================================
# These helpers predict whether a test will hit concurrency limits based on
# the configuration. They help validate test design and understand expected
# behavior before running tests.


def assert_test_will_hit_concurrency_limit(
    config: TimingTestConfig,
    message: str = "",
) -> None:
    """Assert that the test configuration will hit the concurrency limit.

    This validates test design - if you're testing concurrency limiting,
    your test should actually exercise the limit.

    Args:
        config: Test configuration to analyze
        message: Optional context message

    Raises:
        AssertionError: If test won't hit concurrency limit
    """
    context = f"{message}: " if message else ""
    assert config.will_hit_concurrency_limit(), (
        f"{context}Test configuration will NOT hit concurrency limit {config.concurrency}. "
        f"Expected max concurrent: {config.expected_max_concurrent:.1f}, "
        f"QPS: {config.qps}, OSL: {config.osl}, Sessions: {config.num_sessions}"
    )


def assert_test_will_hit_prefill_limit(
    config: TimingTestConfig,
    message: str = "",
) -> None:
    """Assert that the test configuration will hit the prefill concurrency limit.

    This validates test design - if you're testing prefill limiting,
    your test should actually exercise the limit.

    Args:
        config: Test configuration to analyze
        message: Optional context message

    Raises:
        AssertionError: If test won't hit prefill limit
    """
    context = f"{message}: " if message else ""
    assert config.will_hit_prefill_limit(), (
        f"{context}Test configuration will NOT hit prefill limit {config.prefill_concurrency}. "
        f"Expected max prefill concurrent: {config.expected_max_prefill_concurrent:.1f}, "
        f"QPS: {config.qps}, Sessions: {config.num_sessions}"
    )
