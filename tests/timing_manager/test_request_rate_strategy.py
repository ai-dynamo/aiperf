# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive unit tests for the RequestRateStrategy class.
"""

import math

import numpy as np
import pytest
from scipy import stats

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase, RequestRateMode, TimingMode
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.models import CreditPhaseStats
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.request_rate_strategy import (
    ConcurrencyBurstRateGenerator,
    ConstantRateGenerator,
    PoissonRateGenerator,
    RequestRateStrategy,
)
from tests.timing_manager.conftest import (
    MockCreditManager,
    profiling_phase_stats_from_config,
)
from tests.utils.time_traveler import TimeTraveler


def request_rate_config(
    request_rate: float,
    request_count: int,
    request_rate_mode: RequestRateMode = RequestRateMode.POISSON,
    concurrency: int | None = None,
    random_seed: int | None = 42,
) -> tuple[TimingManagerConfig, CreditPhaseStats]:
    config = TimingManagerConfig(
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=request_rate,
        request_count=request_count,
        request_rate_mode=request_rate_mode,
        concurrency=concurrency,
        random_seed=random_seed,
    )
    return config, profiling_phase_stats_from_config(config)


def concurrency_config(
    concurrency: int,
    request_count: int = 10,
    request_rate: float | None = None,
    request_rate_mode: RequestRateMode = RequestRateMode.CONCURRENCY_BURST,
    random_seed: int | None = 42,
) -> tuple[TimingManagerConfig, CreditPhaseStats]:
    config = TimingManagerConfig(
        timing_mode=TimingMode.REQUEST_RATE,
        request_rate=request_rate,
        request_count=request_count,
        concurrency=concurrency,
        request_rate_mode=request_rate_mode,
        random_seed=random_seed,
    )
    return config, profiling_phase_stats_from_config(config)


@pytest.mark.asyncio
class TestRequestRateStrategyPoissonDistribution:
    """Tests for verifying Poisson distribution behavior in RequestRateStrategy."""

    async def run_poisson_rate(
        self,
        request_rate: float,
        request_count: int,
        random_seed: int,
        mock_credit_manager: MockCreditManager,
    ) -> list[int]:
        """Run the Poisson rate execution."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate=request_rate,
            request_count=request_count,
            request_rate_mode=RequestRateMode.POISSON,
            random_seed=random_seed,
        )
        phase_stats = profiling_phase_stats_from_config(config)

        strategy = RequestRateStrategy(config, mock_credit_manager)
        await strategy._execute_single_phase(phase_stats)

        # Wait for all background tasks to complete
        await strategy.wait_for_tasks()

        return mock_credit_manager.dropped_timestamps

    async def run_poisson_rate_event_counts(
        self,
        request_rate: float,
        request_count: int,
        random_seed: int,
        interval_duration: float,
        mock_credit_manager: MockCreditManager,
    ) -> tuple[np.floating, np.ndarray]:
        """Run the Poisson rate execution and return the event counts in each interval."""
        dropped_timestamps = await self.run_poisson_rate(
            request_rate, request_count, random_seed, mock_credit_manager
        )

        timestamps_sec = np.array(dropped_timestamps) / NANOS_PER_SECOND

        # Create time bins from start to end of execution
        start_time = timestamps_sec[0]
        end_time = timestamps_sec[-1]
        num_intervals = int((end_time - start_time) / interval_duration) + 1

        # Count events in each interval
        event_counts = []
        for i in range(num_intervals):
            interval_start = start_time + i * interval_duration
            interval_end = interval_start + interval_duration

            # Count events in this interval
            events_in_interval = np.sum(
                (timestamps_sec >= interval_start) & (timestamps_sec < interval_end)
            )
            event_counts.append(events_in_interval)

        return np.mean(event_counts), np.array(event_counts)

    async def test_poisson_rate_follows_exponential_distribution(
        self, mock_credit_manager, time_traveler
    ):
        """Test that _execute_poisson_rate generates inter-arrival times following exponential distribution."""
        # Set up strategy with Poisson mode
        request_rate = 20.0
        request_count = 20_000
        dropped_timestamps = await self.run_poisson_rate(
            request_rate, request_count, 42, mock_credit_manager
        )

        expected_mean_interval = 1.0 / request_rate

        # Calculate inter-arrival times
        dropped_timestamps = mock_credit_manager.dropped_timestamps
        assert len(dropped_timestamps) == request_count, (
            f"Expected {request_count} credits, got {len(dropped_timestamps)}"
        )

        inter_arrival_times = []
        for i in range(1, len(dropped_timestamps)):
            interval = dropped_timestamps[i] - dropped_timestamps[i - 1]
            inter_arrival_times.append(interval / NANOS_PER_SECOND)

        inter_arrival_times = np.array(inter_arrival_times)

        # Statistical tests for exponential distribution
        # 1. Test mean: For exponential distribution with rate λ, mean = 1/λ
        actual_mean = np.mean(inter_arrival_times)
        assert (
            abs(actual_mean - expected_mean_interval) < expected_mean_interval * 0.2
        ), (
            f"Mean inter-arrival time {actual_mean:.4f} deviates too much from expected {expected_mean_interval:.4f}"
        )

        # 2. Test standard deviation: For exponential distribution, std = mean
        actual_std = np.std(inter_arrival_times)
        expected_std = expected_mean_interval
        assert abs(actual_std - expected_std) < expected_std * 0.3, (
            f"Standard deviation {actual_std:.4f} deviates too much from expected {expected_std:.4f}"
        )

        # 3. Test coefficient of variation: For exponential distribution, CV = 1
        cv = actual_std / actual_mean
        assert abs(cv - 1.0) < 0.2, (
            f"Coefficient of variation {cv:.4f} should be close to 1.0 for exponential distribution"
        )

        # 4. Test that ~63.2% of values are less than the mean (exponential CDF property)
        values_below_mean = np.sum(inter_arrival_times < actual_mean)
        proportion_below_mean = values_below_mean / len(inter_arrival_times)
        expected_proportion = 1 - math.exp(-1)  # ≈ 0.632
        assert abs(proportion_below_mean - expected_proportion) < 0.1, (
            f"Proportion below mean {proportion_below_mean:.3f} should be close to {expected_proportion:.3f}"
        )

    async def test_poisson_rate_independence_of_intervals(
        self, mock_credit_manager, time_traveler
    ):
        """Test that inter-arrival times in Poisson process are independent (low correlation)."""

        request_rate = 15.0
        request_count = 15_000
        dropped_timestamps = await self.run_poisson_rate(
            request_rate, request_count, 123, mock_credit_manager
        )

        # Calculate inter-arrival times
        inter_arrival_times = []
        for i in range(1, len(dropped_timestamps)):
            interval = dropped_timestamps[i] - dropped_timestamps[i - 1]
            inter_arrival_times.append(interval / NANOS_PER_SECOND)

        inter_arrival_times = np.array(inter_arrival_times)

        # Test independence by checking correlation between consecutive intervals
        # For independent events, correlation should be close to 0
        correlation = np.corrcoef(inter_arrival_times[:-1], inter_arrival_times[1:])[
            0, 1
        ]
        # Allow for some variance due to finite sample size, but correlation should be low
        assert abs(correlation) < 0.2, (
            f"Correlation between consecutive intervals {correlation:.4f} indicates lack of independence"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_mean(
        self, mock_credit_manager, time_traveler
    ):
        """Test that event counts in fixed time intervals follow a valid Poisson distribution with mean = λt."""

        request_rate = 1_000
        request_count = 25_000
        interval_duration = 0.5

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, interval_duration, mock_credit_manager
        )

        # Mean should be close to λt (rate × interval_duration)
        expected_events_per_interval = request_rate * interval_duration
        assert (
            abs(actual_mean - expected_events_per_interval)
            < expected_events_per_interval * 0.3
        ), (
            f"Mean event count {actual_mean:.4f} deviates too much from expected {expected_events_per_interval:.4f}"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_variance(
        self, mock_credit_manager, time_traveler
    ):
        """Test that event counts in fixed time intervals follow a valid Poisson distribution with variance = mean."""
        request_rate = 1_000
        request_count = 25_000

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, 0.5, mock_credit_manager
        )

        # For Poisson distribution, variance = mean
        actual_variance = np.var(event_counts)
        assert abs(actual_variance - actual_mean) < actual_mean * 0.4, (
            f"Variance {actual_variance:.4f} should be close to mean {actual_mean:.4f} for Poisson distribution"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_index_of_dispersion(
        self, mock_credit_manager, time_traveler
    ):
        """Test that event counts in fixed time intervals follow a valid Poisson distribution with index of dispersion ≈ 1."""
        request_rate = 1_000
        request_count = 25_000

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, 0.5, mock_credit_manager
        )

        # For Poisson distribution, variance/mean ≈ 1
        actual_variance = np.var(event_counts)
        index_of_dispersion = actual_variance / actual_mean
        assert abs(index_of_dispersion - 1.0) < 0.4, (
            f"Index of dispersion {index_of_dispersion:.4f} should be close to 1.0 for Poisson distribution"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_ks_test(
        self, mock_credit_manager, time_traveler
    ):
        """Test that event counts in fixed time intervals pass Kolmogorov-Smirnov test for goodness of fit."""
        request_rate = 1_000
        request_count = 25_000

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, 0.5, mock_credit_manager
        )

        # Kolmogorov-Smirnov test for goodness of fit
        # Compare empirical distribution to theoretical Poisson distribution

        max_count = int(np.max(event_counts))
        theoretical_pmf = stats.poisson.pmf(np.arange(max_count + 1), actual_mean)

        # Calculate empirical distribution
        unique_counts, frequencies = np.unique(event_counts, return_counts=True)
        empirical_pmf = frequencies / len(event_counts)

        # Pad empirical PMF to match theoretical length if needed
        empirical_pmf_padded = np.zeros(max_count + 1)
        empirical_pmf_padded[unique_counts] = empirical_pmf

        # Calculate cumulative distributions
        theoretical_cdf = np.cumsum(theoretical_pmf)
        empirical_cdf = np.cumsum(empirical_pmf_padded)

        # Kolmogorov-Smirnov statistic
        ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))

        # Critical value for 95% confidence level (approximation for large n)
        critical_value = 1.36 / np.sqrt(len(event_counts))

        assert ks_statistic < critical_value, (
            f"KS statistic {ks_statistic:.4f} exceeds critical value {critical_value:.4f}, "
            f"indicating poor fit to Poisson distribution"
        )

    async def test_poisson_rate_event_counts_follow_poisson_distribution_ratio_property(
        self, mock_credit_manager, time_traveler
    ):
        """Test that the distribution has the right shape characteristics"""
        request_rate = 1_000
        request_count = 25_000

        actual_mean, event_counts = await self.run_poisson_rate_event_counts(
            request_rate, request_count, 789, 0.5, mock_credit_manager
        )

        # For Poisson distribution, P(X = k+1) / P(X = k) = λ / (k+1)
        unique_counts, frequencies = np.unique(event_counts, return_counts=True)
        probabilities = frequencies / len(event_counts)

        # Test ratio property for consecutive values where both have sufficient frequency
        valid_ratios = []
        for i in range(len(unique_counts) - 1):
            k = unique_counts[i]
            if (
                probabilities[i] > 0.01 and probabilities[i + 1] > 0.01
            ):  # Both have reasonable frequency
                actual_ratio = probabilities[i + 1] / probabilities[i]
                expected_ratio = actual_mean / (k + 1)
                if expected_ratio > 0.1:  # Only test ratios that should be meaningful
                    valid_ratios.append((actual_ratio, expected_ratio))

        # Check that most ratios are close to expected (allow some variance due to sampling)
        ratio_errors = [
            abs(actual - expected) / expected for actual, expected in valid_ratios
        ]
        avg_ratio_error = np.mean(ratio_errors)
        assert avg_ratio_error < 0.5, (
            f"Average ratio error {avg_ratio_error:.4f} indicates poor fit to Poisson distribution"
        )


@pytest.mark.asyncio
class TestRequestRateStrategyMaxConcurrency:
    """Tests for max concurrency support in RequestRateStrategy."""

    @pytest.mark.parametrize(
        "concurrency, expected_semaphore_value",
        [
            (5, 5),
            (10, 10),
            (100, 100),
            (300_000, 300_000),
            (None, None),
        ],
    )
    async def test_semaphore_creation_with_concurrency(
        self,
        mock_credit_manager: MockCreditManager,
        concurrency: int,
        expected_semaphore_value: int | None,
    ):
        """Test that semaphore is created when concurrency is specified."""
        config, _ = request_rate_config(
            request_rate=10.0, request_count=10, concurrency=concurrency
        )
        strategy = RequestRateStrategy(config, mock_credit_manager)

        if expected_semaphore_value is None:
            assert strategy._semaphore is None
        else:
            assert strategy._semaphore is not None
            assert strategy._semaphore._value == expected_semaphore_value

    async def test_semaphore_acquisition_during_credit_drop(
        self, mock_credit_manager: MockCreditManager, time_traveler: TimeTraveler
    ):
        """Test that semaphore is acquired before each credit drop."""
        config, phase_stats = concurrency_config(concurrency=5, request_count=10)
        strategy, mock_semaphore = mock_credit_manager.create_strategy(
            config, RequestRateStrategy, auto_return_delay=1.0
        )

        await strategy._execute_single_phase(phase_stats)
        await strategy.wait_for_tasks()

        # Verify that acquire was called for each credit drop
        assert mock_semaphore.acquire_count == 10

    async def test_credit_return_releases_semaphore(
        self, mock_credit_manager: MockCreditManager, time_traveler: TimeTraveler
    ):
        """Test that credit return properly releases the semaphore."""

        config, _ = request_rate_config(
            request_rate=10.0, request_count=10, concurrency=2
        )

        strategy = RequestRateStrategy(config, mock_credit_manager)

        original_semaphore_value = strategy._semaphore._value
        # Simulate credit return
        credit_return = CreditReturnMessage(
            service_id="test-service",
            phase=CreditPhase.PROFILING,
        )
        await strategy._on_credit_return(credit_return)

        # Check that release was called
        assert strategy._semaphore._value == original_semaphore_value + 1

    async def test_single_concurrency_serializes_execution(
        self, mock_credit_manager: MockCreditManager, time_traveler: TimeTraveler
    ):
        """Test that concurrency=1 ensures credits are processed one at a time."""
        config, phase_stats = concurrency_config(concurrency=1, request_count=5)

        strategy, mock_semaphore = mock_credit_manager.create_strategy(
            config, RequestRateStrategy, auto_return_delay=1.0
        )

        await strategy._execute_single_phase(phase_stats)
        await strategy.wait_for_tasks()

        # With concurrency=1 we should have to wait for all but the first credit
        assert mock_semaphore.wait_count == config.request_count - 1
        assert mock_semaphore.acquire_count == config.request_count

    async def test_concurrency_boundary_conditions(
        self, mock_credit_manager: MockCreditManager, time_traveler: TimeTraveler
    ):
        """Test concurrency behavior at boundary conditions."""
        # Test exactly at concurrency limit
        config, phase_stats = concurrency_config(concurrency=5, request_count=5)

        strategy, mock_semaphore = mock_credit_manager.create_strategy(
            config, RequestRateStrategy, auto_return_delay=1.0
        )

        await strategy._execute_single_phase(phase_stats)
        await strategy.wait_for_tasks()

        # Should process all credits without blocking since count equals concurrency
        assert mock_semaphore.acquire_count == 5
        assert mock_semaphore.wait_count == 0

        # Final semaphore value should be 0 (all acquired, none released yet in this test)
        assert mock_semaphore.value == 0


class TestConcurrencyBurstRateGeneratorExceptions:
    """Tests for ConcurrencyBurstRateGenerator initialization exceptions."""

    def test_concurrency_none_raises_value_error(self):
        """Test that ConcurrencyBurstRateGenerator raises ValueError when concurrency is None."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
            concurrency=None,
            request_rate=None,
        )

        with pytest.raises(ValueError):
            ConcurrencyBurstRateGenerator(config)

    @pytest.mark.parametrize("invalid_concurrency", [0, -1, -5, -100])
    def test_concurrency_less_than_one_raises_value_error(
        self, invalid_concurrency: int
    ):
        """Test that ConcurrencyBurstRateGenerator raises ValueError when concurrency is less than 1."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
            concurrency=invalid_concurrency,
            request_rate=None,
        )

        with pytest.raises(ValueError):
            ConcurrencyBurstRateGenerator(config)

    @pytest.mark.parametrize("invalid_request_rate", [1.0, 10.5, 100, 0.1])
    def test_request_rate_not_none_raises_value_error(
        self, invalid_request_rate: float
    ):
        """Test that ConcurrencyBurstRateGenerator raises ValueError when request_rate is not None."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
            concurrency=5,
            request_rate=invalid_request_rate,
        )

        with pytest.raises(ValueError):
            ConcurrencyBurstRateGenerator(config)

    def test_valid_configuration_succeeds(self):
        """Test that ConcurrencyBurstRateGenerator initializes successfully with valid configuration."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
            concurrency=10,
            request_rate=None,
        )

        # Should not raise any exception
        generator = ConcurrencyBurstRateGenerator(config)

        # Verify next_interval always returns 0 for burst mode
        for _ in range(10):
            assert generator.next_interval() == 0


class TestPoissonRateGeneratorExceptions:
    """Tests for PoissonRateGenerator initialization exceptions."""

    def test_request_rate_none_raises_value_error(self):
        """Test that PoissonRateGenerator raises ValueError when request_rate is None."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.POISSON,
            request_rate=None,
        )

        with pytest.raises(ValueError):
            PoissonRateGenerator(config)

    @pytest.mark.parametrize("invalid_request_rate", [0, -1, -5.0, -100.5, 0.0])
    def test_request_rate_zero_or_negative_raises_value_error(
        self, invalid_request_rate: float
    ):
        """Test that PoissonRateGenerator raises ValueError when request_rate is zero or negative."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.POISSON,
            request_rate=invalid_request_rate,
        )

        with pytest.raises(ValueError):
            PoissonRateGenerator(config)

    @pytest.mark.parametrize("valid_request_rate", [0.1, 1.0, 10.5, 100, 1000])
    def test_valid_configuration_succeeds(self, valid_request_rate: float):
        """Test that PoissonRateGenerator initializes successfully with valid configuration."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.POISSON,
            request_rate=valid_request_rate,
            random_seed=42,
        )

        # Should not raise any exception
        generator = PoissonRateGenerator(config)

        # Verify next_interval returns positive values for Poisson distribution
        for _ in range(10):
            interval = generator.next_interval()
            assert interval > 0


class TestConstantRateGeneratorExceptions:
    """Tests for ConstantRateGenerator initialization exceptions."""

    def test_request_rate_none_raises_value_error(self):
        """Test that ConstantRateGenerator raises ValueError when request_rate is None."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.CONSTANT,
            request_rate=None,
        )

        with pytest.raises(ValueError):
            ConstantRateGenerator(config)

    @pytest.mark.parametrize("invalid_request_rate", [0, -1, -5.0, -100.5, 0.0])
    def test_request_rate_zero_or_negative_raises_value_error(
        self, invalid_request_rate: float
    ):
        """Test that ConstantRateGenerator raises ValueError when request_rate is zero or negative."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.CONSTANT,
            request_rate=invalid_request_rate,
        )

        with pytest.raises(ValueError):
            ConstantRateGenerator(config)

    @pytest.mark.parametrize("valid_request_rate", [0.1, 1.0, 10.5, 100, 1000])
    def test_valid_configuration_succeeds(self, valid_request_rate: float):
        """Test that ConstantRateGenerator initializes successfully with valid configuration."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.CONSTANT,
            request_rate=valid_request_rate,
        )

        # Should not raise any exception
        generator = ConstantRateGenerator(config)

        # Verify next_interval returns the expected constant value
        expected_interval = 1.0 / valid_request_rate
        for _ in range(10):
            interval = generator.next_interval()
            assert interval == expected_interval


@pytest.mark.asyncio
class TestRequestRateStrategyEarlyExit:
    """Test for the early exit fix that prevents unnecessary final sleep."""

    async def test_early_exit_prevents_unnecessary_final_sleep(
        self, mock_credit_manager: MockCreditManager, time_traveler: TimeTraveler
    ):
        """Test that execution stops immediately after sending all credits, without extra sleep."""
        config, phase_stats = request_rate_config(
            request_rate=1.0,  # 1 second intervals
            request_count=2,
            request_rate_mode=RequestRateMode.CONSTANT,
            concurrency=1,
        )

        strategy, mock_semaphore = mock_credit_manager.create_strategy(
            config, RequestRateStrategy, auto_return_delay=0.1
        )

        start_time = time_traveler.time()
        await strategy._execute_single_phase(phase_stats)
        await strategy.wait_for_tasks()
        end_time = time_traveler.time()

        # Verify all credits were sent
        assert phase_stats.sent == 2

        # Should take 1 second (no delay for first, 1 second for 2nd, and no final sleep)
        assert end_time - start_time == 1.0
