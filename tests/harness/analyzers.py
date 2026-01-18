# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive analyzers for timing strategy testing.

This module provides reusable analyzer classes for verifying timing, concurrency,
load balancing, and statistical properties of AIPerf test results.

Analyzers:
    CreditFlowAnalyzer: Credit flow patterns (sent/returned/by session)
    TimingAnalyzer: Timing verification (stagger, gaps, intervals)
    StatisticalAnalyzer: Statistical distribution verification (exponential, gamma, constant)
    ConcurrencyAnalyzer: Concurrency validation (total and prefill)
    LoadBalancingAnalyzer: Load balancing distribution analysis
    InvariantChecker: Fundamental timing invariant validation

All analyzers use CapturedPayload.timestamp_ns (monotonic perf_counter) for timing
analysis instead of wall-clock time for consistency and accuracy.

Design Philosophy:
    - Each analyzer has a single responsibility
    - Analyzers are composable and can be used independently or together
    - All timing uses monotonic timestamps from CapturedPayload
    - Results are returned as (passed, reason) tuples for flexible testing
    - Statistical methods are based on rigorous mathematical foundations

Mathematical Foundations:
    - Poisson processes: inter-arrival times are exponentially distributed
    - Exponential distribution: Mean = Std = 1/λ, CV = 1.0
    - Gamma distribution: CV = 1/√smoothness
    - Jain's Fairness Index: standard metric for resource allocation fairness
    - Gini coefficient: measures inequality in distributions

Usage Example:
    # Analyze credit flow
    credit_analyzer = CreditFlowAnalyzer(runner_result)
    assert credit_analyzer.credits_balanced()
    assert credit_analyzer.session_credits_match(expected_turns=3)

    # Verify timing patterns
    timing_analyzer = TimingAnalyzer(results)
    gaps = timing_analyzer.get_credit_issue_times_ns()
    passed, reason = StatisticalAnalyzer.is_approximately_poisson(gaps, qps)

    # Check concurrency
    concurrency_analyzer = ConcurrencyAnalyzer(results)
    assert concurrency_analyzer.concurrency_within_limit(10)

    # Validate load balancing
    lb_analyzer = LoadBalancingAnalyzer(results)
    passed, reason = lb_analyzer.verify_sticky_routing()
    jfi = lb_analyzer.jains_fairness_index()

    # Run invariant checks
    checker = InvariantChecker(runner_result)
    results = checker.run_all_checks()
"""

from collections import defaultdict

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.credit.messages import CreditReturn, FirstToken
from aiperf.credit.structs import Credit
from tests.component_integration.conftest import AIPerfRunnerResultWithSharedBus
from tests.harness.fake_communication import CapturedPayload
from tests.harness.utils import AIPerfResults


class CreditFlowAnalyzer:
    """Analyze credit flow patterns from test results.

    Uses CapturedPayload.timestamp_ns (monotonic perf_counter) for timing analysis
    instead of Credit.issued_at_ns (wall-clock) for consistency with other timing
    measurements.
    """

    def __init__(self, runner_result: AIPerfRunnerResultWithSharedBus):
        self._result = runner_result
        self._credit_payloads: list[CapturedPayload] | None = None
        self._return_payloads: list[CapturedPayload] | None = None
        self._credits_by_session: dict[str, list[CapturedPayload]] | None = None
        self._returns_by_session: dict[str, list[CapturedPayload]] | None = None

    @property
    def credit_payloads(self) -> list[CapturedPayload]:
        """Get all credit payloads sent."""
        if self._credit_payloads is None:
            self._credit_payloads = self._result.payloads_by_type(Credit, sent=True)
        return self._credit_payloads

    @property
    def credits(self) -> list[Credit]:
        """Get all credits sent (convenience property)."""
        return [p.payload for p in self.credit_payloads]

    @property
    def return_payloads(self) -> list[CapturedPayload]:
        """Get all credit return payloads."""
        if self._return_payloads is None:
            self._return_payloads = self._result.payloads_by_type(
                CreditReturn, sent=True
            )
        return self._return_payloads

    @property
    def credit_returns(self) -> list[CreditReturn]:
        """Get all credit returns (convenience property)."""
        return [p.payload for p in self.return_payloads]

    @property
    def credits_by_session(self) -> dict[str, list[CapturedPayload]]:
        """Group credit payloads by session (x_correlation_id)."""
        if self._credits_by_session is None:
            self._credits_by_session = defaultdict(list)
            for p in self.credit_payloads:
                self._credits_by_session[p.payload.x_correlation_id].append(p)
        return self._credits_by_session

    @property
    def returns_by_session(self) -> dict[str, list[CapturedPayload]]:
        """Group credit return payloads by session."""
        if self._returns_by_session is None:
            self._returns_by_session = defaultdict(list)
            for p in self.return_payloads:
                self._returns_by_session[p.payload.credit.x_correlation_id].append(p)
        return self._returns_by_session

    @property
    def num_sessions(self) -> int:
        """Get number of unique sessions."""
        return len(self.credits_by_session)

    @property
    def total_credits(self) -> int:
        """Get total credits sent."""
        return len(self.credit_payloads)

    @property
    def total_returns(self) -> int:
        """Get total credit returns."""
        return len(self.return_payloads)

    def credits_balanced(self) -> bool:
        """Check if all credits have been returned."""
        return self.total_credits == self.total_returns

    def session_credits_match(self, expected_turns: int) -> bool:
        """Check if each session has the expected number of credits."""
        return all(
            len(payloads) == expected_turns
            for payloads in self.credits_by_session.values()
        )

    def turn_indices_sequential(self) -> bool:
        """Check if turn indices are sequential within each session."""
        for payloads in self.credits_by_session.values():
            sorted_payloads = sorted(payloads, key=lambda p: p.timestamp_ns)
            indices = [p.payload.turn_index for p in sorted_payloads]
            if indices != list(range(len(payloads))):
                return False
        return True

    def get_sorted_issue_times_ns(self) -> list[int]:
        """Get sorted list of all credit capture times (monotonic)."""
        return sorted(p.timestamp_ns for p in self.credit_payloads)

    def get_first_turn_issue_times_ns(self) -> list[int]:
        """Get sorted capture times for first turns only (monotonic)."""
        return sorted(
            p.timestamp_ns for p in self.credit_payloads if p.payload.turn_index == 0
        )

    def get_session_issue_times_ns(self, session_id: str) -> list[int]:
        """Get sorted capture times for a specific session (monotonic)."""
        return sorted(
            p.timestamp_ns for p in self.credits_by_session.get(session_id, [])
        )


class TimingAnalyzer:
    """Analyze timing patterns from captured Credit messages.

    Uses CapturedPayload from the communication bus for accurate timing
    measurement at the enforcement layer.
    """

    def __init__(self, results: AIPerfResults):
        self._results = results
        self._credit_flow: CreditFlowAnalyzer | None = None

    @property
    def credit_flow(self) -> CreditFlowAnalyzer:
        """Lazy-load CreditFlowAnalyzer."""
        if self._credit_flow is None:
            runner_result: AIPerfRunnerResultWithSharedBus = self._results.runner_result
            self._credit_flow = CreditFlowAnalyzer(runner_result)
        return self._credit_flow

    def get_credit_issue_times_ns(self) -> list[int]:
        """Get sorted credit issue times from Credit messages."""
        return self.credit_flow.get_sorted_issue_times_ns()

    def get_first_turn_issue_times_ns(self) -> list[int]:
        """Get sorted issue times for first turns only."""
        return self.credit_flow.get_first_turn_issue_times_ns()

    def get_issue_times_by_session(self) -> dict[str, list[int]]:
        """Group issue times by session."""
        return {
            session_id: self.credit_flow.get_session_issue_times_ns(session_id)
            for session_id in self.credit_flow.credits_by_session
        }

    @staticmethod
    def calculate_gaps_sec(times_ns: list[int]) -> list[float]:
        """Calculate inter-arrival gaps in seconds."""
        return [
            (times_ns[i] - times_ns[i - 1]) / NANOS_PER_SECOND
            for i in range(1, len(times_ns))
        ]

    @staticmethod
    def calculate_mean(values: list[float]) -> float:
        """Calculate mean of values."""
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def calculate_std(values: list[float]) -> float:
        """Calculate standard deviation of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance**0.5

    @staticmethod
    def calculate_cv(values: list[float]) -> float:
        """Calculate coefficient of variation (std / mean)."""
        mean = TimingAnalyzer.calculate_mean(values)
        if mean == 0:
            return 0.0
        return TimingAnalyzer.calculate_std(values) / mean


class StatisticalAnalyzer:
    """Statistical tests for timing distributions.

    Provides rigorous statistical methods for verifying timing distributions:
    - Exponential distribution checks for Poisson processes
    - Gamma distribution checks for smoothed/bursty arrivals
    - Constant rate verification
    - Independence testing (memoryless property)

    Mathematical foundations from:
    - Poisson process: inter-arrival times are exponentially distributed
    - Exponential distribution properties: Mean = 1/λ, Std = 1/λ, CV = 1.0
    - CDF property: P(X < mean) = 1 - e^(-1) ≈ 0.632
    - Gamma distribution: CV = 1/√smoothness
    """

    @staticmethod
    def verify_exponential_mean_std_cv(
        values: list[float],
        expected_rate: float,
        tolerance_pct: float = 20.0,
    ) -> tuple[bool, str]:
        """Verify exponential distribution properties: Mean ≈ Std ≈ 1/rate, CV ≈ 1.0.

        For exponential distribution with rate λ:
        - Mean = 1/λ
        - Standard deviation = 1/λ (same as mean!)
        - Coefficient of variation = 1.0

        Args:
            values: List of inter-arrival times
            expected_rate: Expected rate (λ)
            tolerance_pct: Tolerance as percentage

        Returns:
            Tuple of (passed, reason)
        """
        if len(values) < 20:
            return False, f"Need at least 20 samples, got {len(values)}"

        expected_mean = 1.0 / expected_rate
        actual_mean = sum(values) / len(values)
        actual_std = TimingAnalyzer.calculate_std(values)
        actual_cv = actual_std / actual_mean if actual_mean > 0 else 0

        results = []

        # Check mean
        mean_tolerance = expected_mean * (tolerance_pct / 100)
        if abs(actual_mean - expected_mean) > mean_tolerance:
            results.append(
                f"Mean {actual_mean:.4f} differs from expected {expected_mean:.4f}"
            )

        # Check std ≈ mean (key exponential property)
        if abs(actual_std - expected_mean) > mean_tolerance * 1.5:  # Slightly looser
            results.append(
                f"Std {actual_std:.4f} differs from expected {expected_mean:.4f}"
            )

        # Check CV ≈ 1.0
        if abs(actual_cv - 1.0) > 0.3:
            results.append(f"CV {actual_cv:.4f} differs from expected 1.0")

        if results:
            return False, "; ".join(results)

        return True, f"Mean={actual_mean:.4f}, Std={actual_std:.4f}, CV={actual_cv:.4f}"

    @staticmethod
    def verify_exponential_cdf_property(
        values: list[float],
        tolerance: float = 0.15,
    ) -> tuple[bool, str]:
        """Verify exponential CDF property: ~63.2% of values are below the mean.

        For exponential distribution: P(X < mean) = 1 - e^(-1) ≈ 0.6321

        This is a powerful test because it's independent of the rate parameter
        and only depends on the shape of the distribution.

        Args:
            values: List of inter-arrival times
            tolerance: Absolute tolerance for proportion check

        Returns:
            Tuple of (passed, reason)
        """
        import math

        if len(values) < 30:
            return False, f"Need at least 30 samples for CDF test, got {len(values)}"

        mean = sum(values) / len(values)
        values_below_mean = sum(1 for v in values if v < mean)
        proportion_below = values_below_mean / len(values)

        expected_proportion = 1 - math.exp(-1)  # ≈ 0.6321

        if abs(proportion_below - expected_proportion) > tolerance:
            return False, (
                f"Proportion below mean {proportion_below:.3f} differs from "
                f"expected {expected_proportion:.3f}"
            )

        return True, f"Proportion below mean: {proportion_below:.3f} (expected ≈0.632)"

    @staticmethod
    def verify_independence(
        values: list[float],
        max_correlation: float = 0.25,
    ) -> tuple[bool, str]:
        """Verify independence: consecutive inter-arrival times should be uncorrelated.

        For a true Poisson process (memoryless property), consecutive inter-arrival
        times should be independent, meaning low correlation.

        Args:
            values: List of inter-arrival times
            max_correlation: Maximum acceptable absolute correlation

        Returns:
            Tuple of (passed, reason)
        """
        if len(values) < 20:
            return (
                False,
                f"Need at least 20 samples for independence test, got {len(values)}",
            )

        # Calculate correlation between consecutive values
        n = len(values) - 1
        x = values[:-1]
        y = values[1:]

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        # Covariance
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n

        # Standard deviations
        std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

        if std_x == 0 or std_y == 0:
            return False, "Zero variance - cannot compute correlation"

        correlation = cov / (std_x * std_y)

        if abs(correlation) > max_correlation:
            return False, (
                f"Correlation {correlation:.4f} exceeds threshold {max_correlation} - "
                f"intervals may not be independent"
            )

        return True, f"Correlation: {correlation:.4f} (memoryless property holds)"

    @staticmethod
    def verify_index_of_dispersion(
        values: list[float],
        expected_rate: float,
        interval_duration: float = 0.1,
        tolerance: float = 0.5,
    ) -> tuple[bool, str]:
        """Verify Poisson event counts have index of dispersion ≈ 1.0.

        For a Poisson process, if we count events in fixed time intervals:
        - Variance of counts = Mean of counts (index of dispersion = 1.0)

        Args:
            values: List of inter-arrival times (timestamps relative to first)
            expected_rate: Expected rate
            interval_duration: Duration of each counting interval
            tolerance: Tolerance for index of dispersion deviation from 1.0

        Returns:
            Tuple of (passed, reason)
        """
        if len(values) < 30:
            return False, f"Need at least 30 samples, got {len(values)}"

        # Convert intervals to cumulative timestamps
        timestamps = [0.0]
        for interval in values:
            timestamps.append(timestamps[-1] + interval)

        total_duration = timestamps[-1]
        if total_duration < interval_duration * 5:
            return False, "Test duration too short for interval analysis"

        # Count events in each interval
        num_intervals = int(total_duration / interval_duration)
        if num_intervals < 10:
            return False, f"Not enough intervals ({num_intervals}) for dispersion test"

        event_counts = []
        for i in range(num_intervals):
            start = i * interval_duration
            end = start + interval_duration
            count = sum(1 for t in timestamps if start <= t < end)
            event_counts.append(count)

        if len(event_counts) < 10:
            return False, "Not enough intervals for dispersion test"

        mean_count = sum(event_counts) / len(event_counts)
        if mean_count < 1:
            return (
                False,
                f"Mean count {mean_count:.2f} too low for reliable dispersion test",
            )

        variance = sum((c - mean_count) ** 2 for c in event_counts) / len(event_counts)
        index_of_dispersion = variance / mean_count if mean_count > 0 else 0

        if abs(index_of_dispersion - 1.0) > tolerance:
            return False, (
                f"Index of dispersion {index_of_dispersion:.4f} differs from expected 1.0 "
                f"(variance={variance:.2f}, mean={mean_count:.2f})"
            )

        return (
            True,
            f"Index of dispersion: {index_of_dispersion:.4f} (variance/mean ratio)",
        )

    @staticmethod
    def comprehensive_poisson_check(
        values: list[float],
        expected_rate: float,
        tolerance_pct: float = 30.0,
    ) -> tuple[bool, str, dict[str, tuple[bool, str]]]:
        """Run all Poisson distribution checks and return comprehensive results.

        This runs multiple statistical tests and returns detailed results for each.
        A distribution passes if it meets the majority of tests with reasonable tolerances.

        Args:
            values: List of inter-arrival times
            expected_rate: Expected rate
            tolerance_pct: Base tolerance percentage

        Returns:
            Tuple of (overall_passed, summary, individual_results)
        """
        results: dict[str, tuple[bool, str]] = {}

        # 1. Mean/Std/CV check
        results["mean_std_cv"] = StatisticalAnalyzer.verify_exponential_mean_std_cv(
            values, expected_rate, tolerance_pct
        )

        # 2. CDF property (63.2% below mean)
        results["cdf_property"] = StatisticalAnalyzer.verify_exponential_cdf_property(
            values
        )

        # 3. Independence (memoryless property)
        results["independence"] = StatisticalAnalyzer.verify_independence(values)

        # 4. Index of dispersion
        results["dispersion"] = StatisticalAnalyzer.verify_index_of_dispersion(
            values, expected_rate
        )

        # Overall pass if at least 3 of 4 tests pass
        passes = sum(1 for passed, _ in results.values() if passed)
        overall_passed = passes >= 3

        if overall_passed:
            summary = f"Poisson check passed ({passes}/4 tests)"
        else:
            failed = [name for name, (passed, _) in results.items() if not passed]
            summary = f"Poisson check failed ({passes}/4 tests): {', '.join(failed)}"

        return overall_passed, summary, results

    @staticmethod
    def is_approximately_constant(
        values: list[float],
        expected: float,
        tolerance_pct: float = 20.0,
        max_cv: float = 0.6,
    ) -> tuple[bool, str]:
        """Check if values are approximately constant.

        Args:
            values: List of values to check
            expected: Expected value
            tolerance_pct: Tolerance as percentage of expected
            max_cv: Maximum acceptable coefficient of variation (default 0.6).
                    Higher values are needed for small sample sizes or when
                    test harness timing jitter is significant.

        Returns:
            Tuple of (passed, reason)
        """
        if not values:
            return False, "No values to analyze"

        mean = sum(values) / len(values)
        tolerance = expected * (tolerance_pct / 100)

        if abs(mean - expected) > tolerance:
            return (
                False,
                f"Mean {mean:.4f} differs from expected {expected:.4f} by more than {tolerance_pct}%",
            )

        # For constant rate, CV should be low
        cv = TimingAnalyzer.calculate_cv(values)
        if cv > max_cv:
            return (
                False,
                f"CV {cv:.4f} too high for constant rate (expected < {max_cv})",
            )

        return True, f"Mean={mean:.4f}, CV={cv:.4f}"

    @staticmethod
    def is_approximately_poisson(
        values: list[float],
        expected_rate: float,
        tolerance_pct: float = 30.0,
    ) -> tuple[bool, str]:
        """Check if values follow exponential distribution (Poisson process).

        For a Poisson process, inter-arrival times should be exponentially
        distributed with mean = 1/rate and CV ~ 1.0.

        Args:
            values: List of inter-arrival times
            expected_rate: Expected rate (requests per second)
            tolerance_pct: Tolerance for mean check

        Returns:
            Tuple of (passed, reason)
        """
        if not values:
            return False, "No values to analyze"

        expected_mean = 1.0 / expected_rate
        mean = sum(values) / len(values)
        tolerance = expected_mean * (tolerance_pct / 100)

        if abs(mean - expected_mean) > tolerance:
            return (
                False,
                f"Mean {mean:.4f} differs from expected {expected_mean:.4f} by more than {tolerance_pct}%",
            )

        # For exponential distribution, CV should be approximately 1.0
        cv = TimingAnalyzer.calculate_cv(values)
        if cv < 0.5 or cv > 1.5:
            return (
                False,
                f"CV {cv:.4f} outside range [0.5, 1.5] for exponential distribution",
            )

        return True, f"Mean={mean:.4f}, CV={cv:.4f}"

    @staticmethod
    def is_approximately_gamma(
        values: list[float],
        expected_rate: float,
        smoothness: float,
        tolerance_pct: float = 30.0,
    ) -> tuple[bool, str]:
        """Check if values follow Gamma distribution with given smoothness.

        For Gamma distribution with shape=smoothness:
        - Mean = 1/rate (same as Poisson)
        - CV = 1/sqrt(smoothness)

        Args:
            values: List of inter-arrival times
            expected_rate: Expected rate (requests per second)
            smoothness: Gamma shape parameter (1.0 = Poisson)
            tolerance_pct: Tolerance for mean and CV checks

        Returns:
            Tuple of (passed, reason)
        """
        if not values:
            return False, "No values to analyze"

        if len(values) < 20:
            return (
                False,
                f"Need at least 20 samples for Gamma analysis, got {len(values)}",
            )

        # Check mean
        expected_mean = 1.0 / expected_rate
        mean = sum(values) / len(values)
        mean_tolerance = expected_mean * (tolerance_pct / 100)

        if abs(mean - expected_mean) > mean_tolerance:
            return (
                False,
                f"Mean {mean:.4f} differs from expected {expected_mean:.4f} by more than {tolerance_pct}%",
            )

        # Check CV: for Gamma, CV = 1/sqrt(smoothness)
        expected_cv = 1.0 / (smoothness**0.5)
        cv = TimingAnalyzer.calculate_cv(values)

        # Allow wider tolerance for CV since it's more variable
        cv_tolerance = (
            expected_cv * (tolerance_pct / 100) * 1.5
        )  # 1.5x tolerance for CV
        if abs(cv - expected_cv) > cv_tolerance:
            return (
                False,
                f"CV {cv:.4f} differs from expected {expected_cv:.4f} (smoothness={smoothness})",
            )

        return True, f"Mean={mean:.4f}, CV={cv:.4f} (expected CV={expected_cv:.4f})"

    @staticmethod
    def verify_stagger(
        times_ns: list[int],
        expected_stagger_sec: float,
        tolerance_pct: float = 50.0,
    ) -> tuple[bool, str]:
        """Verify staggered timing pattern.

        Args:
            times_ns: Sorted list of issue times in nanoseconds
            expected_stagger_sec: Expected stagger interval
            tolerance_pct: Tolerance as percentage

        Returns:
            Tuple of (passed, reason)
        """
        if len(times_ns) < 2:
            return True, "Less than 2 events, stagger check not applicable"

        gaps = TimingAnalyzer.calculate_gaps_sec(times_ns)
        mean_gap = sum(gaps) / len(gaps)
        tolerance = expected_stagger_sec * (tolerance_pct / 100)

        if abs(mean_gap - expected_stagger_sec) > tolerance:
            return False, (
                f"Mean stagger {mean_gap:.4f}s differs from expected "
                f"{expected_stagger_sec:.4f}s by more than {tolerance_pct}%"
            )

        return True, f"Mean stagger={mean_gap:.4f}s"

    @staticmethod
    def verify_per_user_gaps(
        times_by_session: dict[str, list[int]],
        expected_gap_sec: float,
        tolerance_pct: float = 50.0,
    ) -> tuple[bool, str]:
        """Verify per-user gap timing.

        Args:
            times_by_session: Dict of session_id -> sorted issue times (ns)
            expected_gap_sec: Expected gap between same-user turns
            tolerance_pct: Tolerance as percentage

        Returns:
            Tuple of (passed, reason)
        """
        all_gaps = []
        for times in times_by_session.values():
            if len(times) >= 2:
                all_gaps.extend(TimingAnalyzer.calculate_gaps_sec(times))

        if not all_gaps:
            return True, "No multi-turn sessions to verify"

        mean_gap = sum(all_gaps) / len(all_gaps)
        tolerance = expected_gap_sec * (tolerance_pct / 100)

        if abs(mean_gap - expected_gap_sec) > tolerance:
            return False, (
                f"Mean per-user gap {mean_gap:.4f}s differs from expected "
                f"{expected_gap_sec:.4f}s by more than {tolerance_pct}%"
            )

        return True, f"Mean per-user gap={mean_gap:.4f}s"


class ConcurrencyAnalyzer:
    """Analyze concurrency patterns from captured Credit/CreditReturn/FirstToken messages.

    Uses CapturedPayload from the communication bus for accurate timing
    measurement at the enforcement layer (TimingManager) rather than the
    HTTP layer (Worker).
    """

    def __init__(self, results: AIPerfResults):
        self._results = results
        # Cache for intervals
        self._request_intervals: list[tuple[int, int]] | None = None
        self._prefill_intervals: list[tuple[int, int]] | None = None

    def get_request_intervals(self) -> list[tuple[int, int]]:
        """Get request intervals from Credit/CreditReturn messages.

        Measures at the credit layer using monotonic capture timestamps:
        - Request start = CapturedPayload.timestamp_ns for Credit (when Credit was sent)
        - Request end = CapturedPayload.timestamp_ns for CreditReturn (when worker returned credit)
        """
        if self._request_intervals is not None:
            return self._request_intervals

        runner_result: AIPerfRunnerResultWithSharedBus = self._results.runner_result

        # Get all Credits with their capture timestamps
        credit_payloads: list[CapturedPayload] = runner_result.payloads_by_type(
            Credit, sent=True
        )
        credit_send_times: dict[int, int] = {
            p.payload.id: p.timestamp_ns for p in credit_payloads
        }

        # Get all CreditReturn messages with their capture timestamps
        credit_return_payloads: list[CapturedPayload] = runner_result.payloads_by_type(
            CreditReturn, sent=True
        )
        credit_return_times: dict[int, int] = {
            p.payload.credit.id: p.timestamp_ns for p in credit_return_payloads
        }

        # Build intervals by matching credit.id
        intervals = []
        for credit_id, send_time_ns in credit_send_times.items():
            if credit_id in credit_return_times:
                intervals.append((send_time_ns, credit_return_times[credit_id]))

        self._request_intervals = intervals
        return intervals

    def get_prefill_intervals(self) -> list[tuple[int, int]]:
        """Get prefill intervals from Credit/FirstToken messages.

        Measures at the prefill enforcement layer using monotonic capture timestamps:
        - Prefill start = CapturedPayload.timestamp_ns for Credit (when Credit was sent)
        - Prefill end = CapturedPayload.timestamp_ns for FirstToken (when worker sent FirstToken)
        """
        if self._prefill_intervals is not None:
            return self._prefill_intervals

        runner_result: AIPerfRunnerResultWithSharedBus = self._results.runner_result

        # Get all Credits with their capture timestamps
        credit_payloads: list[CapturedPayload] = runner_result.payloads_by_type(
            Credit, sent=True
        )
        credit_send_times: dict[int, int] = {
            p.payload.id: p.timestamp_ns for p in credit_payloads
        }

        # Get all FirstToken messages with their capture timestamps
        first_token_payloads: list[CapturedPayload] = runner_result.payloads_by_type(
            FirstToken, sent=True
        )
        first_token_times: dict[int, int] = {
            p.payload.credit_id: p.timestamp_ns for p in first_token_payloads
        }

        # Build intervals by matching credit_id
        intervals = []
        for credit_id, send_time_ns in credit_send_times.items():
            if credit_id in first_token_times:
                intervals.append((send_time_ns, first_token_times[credit_id]))

        self._prefill_intervals = intervals
        return intervals

    def _calculate_max_concurrent_from_events(
        self, events: list[tuple[int, int]]
    ) -> int:
        """Calculate maximum concurrency from a list of (time, delta) events."""
        if not events:
            return 0

        # Sort by time, starts before ends at same time
        events.sort(key=lambda x: (x[0], -x[1]))

        max_concurrent = 0
        current = 0
        for _, delta in events:
            current += delta
            max_concurrent = max(max_concurrent, current)

        return max_concurrent

    def get_max_concurrent(self) -> int:
        """Calculate maximum concurrency from Credit/CreditReturn messages."""
        events = []
        for start, end in self.get_request_intervals():
            events.append((start, 1))  # +1 at credit issue
            events.append((end, -1))  # -1 at credit return

        return self._calculate_max_concurrent_from_events(events)

    def get_max_prefill_concurrent(self) -> int:
        """Calculate maximum concurrent prefills from Credit/FirstToken messages."""
        events = []
        for start, prefill_end in self.get_prefill_intervals():
            events.append((start, 1))  # +1 at prefill start (Credit issued)
            events.append((prefill_end, -1))  # -1 at prefill end (FirstToken sent)

        return self._calculate_max_concurrent_from_events(events)

    def concurrency_within_limit(self, limit: int) -> bool:
        """Check if concurrency stayed within the specified limit."""
        return self.get_max_concurrent() <= limit

    def prefill_concurrency_within_limit(self, limit: int) -> bool:
        """Check if prefill concurrency stayed within the specified limit."""
        return self.get_max_prefill_concurrent() <= limit


class LoadBalancingAnalyzer:
    """Analyze load balancing distribution from captured Credit messages.

    Uses CapturedPayload.receiver_identity to determine which worker received
    each credit.
    """

    def __init__(self, results: AIPerfResults):
        self._results = results
        self._credits_by_worker: dict[str, list[CapturedPayload]] | None = None
        self._sessions_by_worker: dict[str, set[str]] | None = None

    @property
    def credits_by_worker(self) -> dict[str, list[CapturedPayload]]:
        """Group credit payloads by worker (receiver_identity)."""
        if self._credits_by_worker is None:
            self._credits_by_worker = defaultdict(list)
            runner_result: AIPerfRunnerResultWithSharedBus = self._results.runner_result
            for p in runner_result.payloads_by_type(Credit, sent=True):
                if p.receiver_identity:
                    self._credits_by_worker[p.receiver_identity].append(p)
        return self._credits_by_worker

    @property
    def sessions_by_worker(self) -> dict[str, set[str]]:
        """Group unique sessions (x_correlation_ids) by worker."""
        if self._sessions_by_worker is None:
            self._sessions_by_worker = defaultdict(set)
            for worker_id, payloads in self.credits_by_worker.items():
                for p in payloads:
                    self._sessions_by_worker[worker_id].add(p.payload.x_correlation_id)
        return self._sessions_by_worker

    @property
    def num_workers(self) -> int:
        """Get number of workers that received credits."""
        return len(self.credits_by_worker)

    @property
    def total_credits(self) -> int:
        """Get total credits sent across all workers."""
        return sum(len(payloads) for payloads in self.credits_by_worker.values())

    @property
    def total_sessions(self) -> int:
        """Get total unique sessions across all workers."""
        all_sessions = set()
        for sessions in self.sessions_by_worker.values():
            all_sessions.update(sessions)
        return len(all_sessions)

    def credits_per_worker(self) -> dict[str, int]:
        """Get credit count per worker."""
        return {
            worker_id: len(payloads)
            for worker_id, payloads in self.credits_by_worker.items()
        }

    def sessions_per_worker(self) -> dict[str, int]:
        """Get session count per worker."""
        return {
            worker_id: len(sessions)
            for worker_id, sessions in self.sessions_by_worker.items()
        }

    def first_turns_per_worker(self) -> dict[str, int]:
        """Get first turn (turn_index=0) count per worker."""
        first_turns: dict[str, int] = defaultdict(int)
        for worker_id, payloads in self.credits_by_worker.items():
            for p in payloads:
                if p.payload.turn_index == 0:
                    first_turns[worker_id] += 1
        return dict(first_turns)

    def verify_sticky_routing(self) -> tuple[bool, str]:
        """Verify all turns of each session went to the same worker.

        Returns:
            Tuple of (passed, reason)
        """
        session_to_workers: dict[str, set[str]] = defaultdict(set)
        for worker_id, payloads in self.credits_by_worker.items():
            for p in payloads:
                session_to_workers[p.payload.x_correlation_id].add(worker_id)

        violations = []
        for session_id, workers in session_to_workers.items():
            if len(workers) > 1:
                violations.append(
                    f"Session {session_id[:8]}... routed to {len(workers)} workers: {workers}"
                )

        if violations:
            return False, f"Sticky routing violations: {violations[:3]}"
        return (
            True,
            f"All {len(session_to_workers)} sessions correctly routed to single workers",
        )

    def verify_fair_distribution(
        self,
        tolerance_pct: float = 30.0,
    ) -> tuple[bool, str]:
        """Verify credits are fairly distributed across workers.

        Fair distribution means each worker receives approximately
        total_credits / num_workers credits.

        Args:
            tolerance_pct: Maximum allowed deviation from expected as percentage

        Returns:
            Tuple of (passed, reason)
        """
        if self.num_workers == 0:
            return False, "No workers received credits"

        if self.total_credits == 0:
            return False, "No credits sent"

        expected_per_worker = self.total_credits / self.num_workers
        tolerance = expected_per_worker * (tolerance_pct / 100)

        credits = self.credits_per_worker()
        deviations = []
        for worker_id, count in credits.items():
            deviation = abs(count - expected_per_worker)
            if deviation > tolerance:
                deviations.append(
                    f"{worker_id}: {count} credits (expected ~{expected_per_worker:.1f}, "
                    f"deviation {deviation:.1f} > tolerance {tolerance:.1f})"
                )

        if deviations:
            return False, f"Unfair distribution: {deviations}"

        counts = list(credits.values())
        actual_max_deviation = max(abs(c - expected_per_worker) for c in counts)
        return True, (
            f"Fair distribution: {self.num_workers} workers, {self.total_credits} credits, "
            f"expected ~{expected_per_worker:.1f}/worker, max deviation {actual_max_deviation:.1f}"
        )

    def verify_first_turn_distribution(
        self,
        tolerance_pct: float = 30.0,
    ) -> tuple[bool, str]:
        """Verify first turns (new sessions) are fairly distributed.

        This specifically tests the load balancing algorithm's fairness for
        new session assignment, independent of sticky routing effects.

        Args:
            tolerance_pct: Maximum allowed deviation from expected as percentage

        Returns:
            Tuple of (passed, reason)
        """
        first_turns = self.first_turns_per_worker()
        total_first_turns = sum(first_turns.values())

        if total_first_turns == 0:
            return False, "No first turns found"

        expected_per_worker = total_first_turns / self.num_workers
        tolerance = expected_per_worker * (tolerance_pct / 100)

        deviations = []
        for worker_id, count in first_turns.items():
            deviation = abs(count - expected_per_worker)
            if deviation > tolerance:
                deviations.append(
                    f"{worker_id}: {count} first turns (expected ~{expected_per_worker:.1f})"
                )

        if deviations:
            return False, f"Unfair first turn distribution: {deviations}"

        actual_max_deviation = max(
            abs(c - expected_per_worker) for c in first_turns.values()
        )
        return True, (
            f"Fair first turn distribution: {total_first_turns} sessions across "
            f"{self.num_workers} workers, max deviation {actual_max_deviation:.1f}"
        )

    def get_distribution_stats(self) -> dict[str, float]:
        """Get distribution statistics for analysis."""
        credits = list(self.credits_per_worker().values())
        if not credits:
            return {"mean": 0, "std": 0, "cv": 0, "min": 0, "max": 0}

        mean = sum(credits) / len(credits)
        variance = sum((c - mean) ** 2 for c in credits) / len(credits)
        std = variance**0.5
        cv = std / mean if mean > 0 else 0

        return {
            "mean": mean,
            "std": std,
            "cv": cv,
            "min": min(credits),
            "max": max(credits),
        }

    def jains_fairness_index(self) -> float:
        """Calculate Jain's Fairness Index for credit distribution.

        Jain's Fairness Index (JFI) is a standard metric for measuring fairness
        in resource allocation, widely used in networking and distributed systems.

        Formula: J(x) = (Σxᵢ)² / (n × Σxᵢ²)

        Properties:
        - Range: [1/n, 1] where n is the number of workers
        - 1.0 = perfectly fair (all workers got equal share)
        - 1/n = maximally unfair (one worker got everything)
        - Scale-independent (doubling all allocations doesn't change index)

        Reference:
            Jain, R., Chiu, D., Hawe, W. (1984). "A Quantitative Measure of
            Fairness and Discrimination for Resource Allocation in Shared
            Computer Systems". DEC Research Report TR-301.

        Returns:
            Jain's Fairness Index in range [1/n, 1]
        """
        credits = list(self.credits_per_worker().values())
        if not credits:
            return 0.0

        n = len(credits)
        if n == 1:
            return 1.0  # Single worker is trivially fair

        sum_x = sum(credits)
        sum_x_squared = sum(x * x for x in credits)

        if sum_x_squared == 0:
            return 1.0  # All zeros is fair

        return (sum_x * sum_x) / (n * sum_x_squared)

    def max_min_ratio(self) -> float:
        """Calculate max-min ratio (imbalance ratio).

        A simple metric showing how much more work the busiest worker
        did compared to the least busy worker.

        Returns:
            Ratio of max/min credits. 1.0 = perfectly balanced.
            Returns inf if min is 0.
        """
        credits = list(self.credits_per_worker().values())
        if not credits:
            return 0.0

        min_credits = min(credits)
        max_credits = max(credits)

        if min_credits == 0:
            return float("inf")

        return max_credits / min_credits

    def gini_coefficient(self) -> float:
        """Calculate Gini coefficient for credit distribution.

        The Gini coefficient measures inequality, commonly used in economics
        but also applicable to load balancing fairness.

        Formula: G = (Σᵢ Σⱼ |xᵢ - xⱼ|) / (2n² × mean)

        Properties:
        - Range: [0, 1]
        - 0 = perfect equality (all workers got equal share)
        - 1 = maximum inequality (one worker got everything)

        Returns:
            Gini coefficient in range [0, 1]
        """
        credits = list(self.credits_per_worker().values())
        if not credits:
            return 0.0

        n = len(credits)
        if n == 1:
            return 0.0  # Single worker has no inequality

        mean = sum(credits) / n
        if mean == 0:
            return 0.0

        # Calculate sum of absolute differences
        total_diff = sum(abs(xi - xj) for xi in credits for xj in credits)

        return total_diff / (2 * n * n * mean)

    def verify_jains_fairness(
        self,
        min_fairness: float = 0.9,
    ) -> tuple[bool, str]:
        """Verify Jain's Fairness Index meets threshold.

        Args:
            min_fairness: Minimum acceptable JFI (default 0.9 = highly fair)

        Returns:
            Tuple of (passed, reason)
        """
        jfi = self.jains_fairness_index()

        if jfi < min_fairness:
            return False, (
                f"Jain's Fairness Index {jfi:.4f} below threshold {min_fairness}"
            )

        return True, f"Jain's Fairness Index: {jfi:.4f} (threshold: {min_fairness})"


class InvariantChecker:
    """Comprehensive invariant checking for timing correctness.

    This class provides systematic validation of fundamental timing invariants
    that must hold regardless of timing mode, rate, or configuration.

    Moved from test_invariants.py to enable reuse across all timing tests.
    """

    def __init__(self, runner_result: AIPerfRunnerResultWithSharedBus):
        self._result = runner_result
        self._credit_analyzer = CreditFlowAnalyzer(runner_result)

    def check_credit_id_uniqueness(self) -> tuple[bool, str]:
        """Verify all credit IDs are unique within each phase.

        Credit IDs are scoped per phase (warmup/profiling), so they restart at 0
        for each phase. Duplicates within the same phase would indicate a bug.
        """
        # Group credits by phase
        credits_by_phase: dict[str, list[int]] = defaultdict(list)
        for credit in self._credit_analyzer.credits:
            credits_by_phase[credit.phase].append(credit.id)

        violations = []
        for phase, credit_ids in credits_by_phase.items():
            unique_ids = set(credit_ids)
            if len(credit_ids) != len(unique_ids):
                duplicates = [cid for cid in unique_ids if credit_ids.count(cid) > 1]
                violations.append(f"{phase}: duplicates {duplicates[:3]}")

        if violations:
            return False, f"Duplicate credit IDs found: {violations}"

        total = sum(len(ids) for ids in credits_by_phase.values())
        phases = list(credits_by_phase.keys())
        return True, f"All {total} credit IDs unique within phases {phases}"

    def check_credit_return_matching(self) -> tuple[bool, str]:
        """Verify each credit return matches a valid credit within the same phase.

        Every CreditReturn should reference a (phase, credit_id) pair that was issued.
        Credit IDs are unique per phase, so we must match within phase context.
        """
        # Build (phase, id) tuples for issued credits
        issued = {(c.phase, c.id) for c in self._credit_analyzer.credits}
        # Build (phase, id) tuples for returned credits
        returned = {
            (cr.credit.phase, cr.credit.id)
            for cr in self._credit_analyzer.credit_returns
        }

        # Check for returns without matching credits
        orphan_returns = returned - issued
        if orphan_returns:
            return (
                False,
                f"Returns for non-existent credits: {list(orphan_returns)[:5]}",
            )

        # Check for unreturned credits
        unreturned = issued - returned
        if unreturned:
            return False, f"Credits never returned: {list(unreturned)[:5]}"

        return True, f"All {len(issued)} credits properly matched with returns"

    def check_no_double_returns(self) -> tuple[bool, str]:
        """Verify no credit is returned more than once within its phase.

        Double returns would indicate a serious bug in credit tracking.
        Credit IDs are scoped per phase, so we check (phase, id) pairs.
        """
        # Count returns by (phase, id) pair
        return_keys = [
            (cr.credit.phase, cr.credit.id)
            for cr in self._credit_analyzer.credit_returns
        ]
        key_counts: dict[tuple, int] = defaultdict(int)
        for key in return_keys:
            key_counts[key] += 1

        double_returns = [key for key, count in key_counts.items() if count > 1]
        if double_returns:
            return False, f"Credits returned multiple times: {double_returns[:5]}"

        return True, f"No double returns among {len(return_keys)} credit returns"

    def check_timestamp_monotonicity(self) -> tuple[bool, str]:
        """Verify credit issue timestamps are strictly increasing.

        Non-monotonic timestamps would indicate clock issues or race conditions.
        Uses capture timestamps (monotonic perf_counter) not wall-clock time.
        """
        times = self._credit_analyzer.get_sorted_issue_times_ns()

        if len(times) < 2:
            return True, "Not enough credits to check monotonicity"

        violations = []
        for i in range(1, len(times)):
            if times[i] <= times[i - 1]:
                violations.append(
                    f"credit[{i}] at {times[i]} <= credit[{i - 1}] at {times[i - 1]}"
                )

        if violations:
            return False, f"Non-monotonic timestamps: {violations[:5]}"

        return True, f"All {len(times)} timestamps are strictly increasing"

    def check_turn_index_correctness(self) -> tuple[bool, str]:
        """Verify turn indices are correct within each session.

        Turn indices should:
        - Start at 0
        - Be sequential (0, 1, 2, ...)
        - Have no gaps or duplicates
        """
        violations = []
        for session_id, payloads in self._credit_analyzer.credits_by_session.items():
            sorted_payloads = sorted(payloads, key=lambda p: p.timestamp_ns)
            indices = [p.payload.turn_index for p in sorted_payloads]

            expected = list(range(len(payloads)))
            if indices != expected:
                violations.append(
                    f"Session {session_id[:8]}...: got indices {indices}, expected {expected}"
                )

        if violations:
            return False, f"Turn index violations: {violations[:3]}"

        return True, "All sessions have correct turn indices"

    def check_session_metadata_consistency(self) -> tuple[bool, str]:
        """Verify session metadata is consistent across all credits in a session.

        All credits for a session should have:
        - Same x_correlation_id
        - Same worker assignment (if sticky routing)
        """
        violations = []
        for session_id, payloads in self._credit_analyzer.credits_by_session.items():
            # Check x_correlation_id consistency (should be the session_id by definition)
            for p in payloads:
                if p.payload.x_correlation_id != session_id:
                    violations.append(
                        f"Session {session_id[:8]}...: credit has wrong x_correlation_id"
                    )
                    break

        if violations:
            return False, f"Metadata inconsistencies: {violations[:3]}"

        return (
            True,
            f"All {len(self._credit_analyzer.credits_by_session)} sessions have consistent metadata",
        )

    def check_return_after_issue(self) -> tuple[bool, str]:
        """Verify each credit return timestamp is after its issue timestamp.

        Returns happening before issues would indicate serious timing bugs.
        Credit IDs are scoped per phase, so we use (phase, id) as the key.
        """
        # Build lookup from (phase, credit_id) to issue timestamp
        credit_payloads = self._result.payloads_by_type(Credit, sent=True)
        issue_times = {
            (p.payload.phase, p.payload.id): p.timestamp_ns for p in credit_payloads
        }

        # Check each return
        return_payloads = self._result.payloads_by_type(CreditReturn, sent=True)
        violations = []
        for p in return_payloads:
            key = (p.payload.credit.phase, p.payload.credit.id)
            if key in issue_times and p.timestamp_ns < issue_times[key]:
                violations.append(
                    f"Credit {key}: returned at {p.timestamp_ns} before issued at {issue_times[key]}"
                )

        if violations:
            return False, f"Return-before-issue violations: {violations[:3]}"

        return True, "All returns are after their corresponding issues"

    def run_all_checks(self) -> list[tuple[str, bool, str]]:
        """Run all invariant checks and return results."""
        checks = [
            ("credit_id_uniqueness", self.check_credit_id_uniqueness),
            ("credit_return_matching", self.check_credit_return_matching),
            ("no_double_returns", self.check_no_double_returns),
            ("timestamp_monotonicity", self.check_timestamp_monotonicity),
            ("turn_index_correctness", self.check_turn_index_correctness),
            ("session_metadata_consistency", self.check_session_metadata_consistency),
            ("return_after_issue", self.check_return_after_issue),
        ]

        results = []
        for name, check in checks:
            passed, reason = check()
            results.append((name, passed, reason))

        return results


def verify_no_interleaving_within_session(
    credit_analyzer: CreditFlowAnalyzer,
) -> tuple[bool, str]:
    """Verify no interleaving of turns within a single session.

    Each session's turns should be strictly sequential - no overlapping.
    Uses CapturedPayload.timestamp_ns (monotonic) for timing comparisons.
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

    Uses CapturedPayload.timestamp_ns (monotonic) for timing comparisons.
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
