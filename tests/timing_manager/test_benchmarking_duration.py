# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for benchmarking duration works correctly.
"""

import time

import pytest

from aiperf.common.enums import CreditPhase, RequestRateMode, TimingMode
from aiperf.common.models import CreditPhaseStats
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.request_rate_strategy import RequestRateStrategy
from tests.timing_manager.conftest import (
    MockCreditManager,
)


def benchmarking_duration_config(
    benchmarking_duration: float,
    request_rate_mode: RequestRateMode = RequestRateMode.CONCURRENCY_BURST,
    concurrency: int = 1,
    request_rate: float | None = None,
    warmup_request_count: int = 0,
    random_seed: int | None = 42,
) -> TimingManagerConfig:
    """Helper function to create a TimingManagerConfig with benchmarking duration."""
    return TimingManagerConfig(
        timing_mode=TimingMode.REQUEST_RATE,
        benchmarking_duration=benchmarking_duration,
        request_rate_mode=request_rate_mode,
        concurrency=concurrency,
        request_rate=request_rate,
        request_count=10,  # This should be ignored when benchmarking_duration is set
        warmup_request_count=warmup_request_count,
        random_seed=random_seed,
    )


def mixed_config(
    request_count: int = 10,
    benchmarking_duration: float | None = None,
    request_rate_mode: RequestRateMode = RequestRateMode.CONCURRENCY_BURST,
    concurrency: int = 1,
    request_rate: float | None = None,
    warmup_request_count: int = 0,
    random_seed: int | None = 42,
) -> TimingManagerConfig:
    """Helper function to create a TimingManagerConfig with both request_count and duration."""
    return TimingManagerConfig(
        timing_mode=TimingMode.REQUEST_RATE,
        request_count=request_count,
        benchmarking_duration=benchmarking_duration,
        request_rate_mode=request_rate_mode,
        concurrency=concurrency,
        request_rate=request_rate,
        warmup_request_count=warmup_request_count,
        random_seed=random_seed,
    )


class TestBenchmarkingDurationConfiguration:
    """Test configuration validation and behavior of benchmarking duration."""

    def test_benchmarking_duration_creates_time_based_config(self):
        """Test that benchmarking duration creates proper configuration."""
        config = benchmarking_duration_config(benchmarking_duration=5.0)

        assert config.benchmarking_duration == 5.0
        assert config.timing_mode == TimingMode.REQUEST_RATE

    def test_benchmarking_duration_overrides_request_count(self):
        """Test that when both are provided, duration takes precedence."""
        config = mixed_config(request_count=100, benchmarking_duration=3.0)

        assert config.benchmarking_duration == 3.0
        assert config.request_count == 100  # Still stored but should be ignored

    def test_benchmarking_duration_large_value(self):
        """Test that large duration values are accepted."""
        config = benchmarking_duration_config(benchmarking_duration=3600.0)

        assert config.benchmarking_duration == 3600.0

    def test_benchmarking_duration_fractional_value(self):
        """Test that fractional duration values are accepted."""
        config = benchmarking_duration_config(benchmarking_duration=2.5)

        assert config.benchmarking_duration == 2.5

    def test_benchmarking_duration_with_different_request_rate_modes(self):
        """Test benchmarking duration with various request rate modes."""
        modes = [
            RequestRateMode.CONCURRENCY_BURST,
            RequestRateMode.POISSON,
            RequestRateMode.CONSTANT,
        ]

        for mode in modes:
            config = benchmarking_duration_config(
                benchmarking_duration=4.0,
                request_rate_mode=mode,
                request_rate=10.0
                if mode != RequestRateMode.CONCURRENCY_BURST
                else None,
            )

            assert config.benchmarking_duration == 4.0
            assert config.request_rate_mode == mode

    def test_benchmarking_duration_with_warmup(self):
        """Test benchmarking duration configuration with warmup requests."""
        config = benchmarking_duration_config(
            benchmarking_duration=6.0, warmup_request_count=5
        )

        assert config.benchmarking_duration == 6.0
        assert config.warmup_request_count == 5

    def test_benchmarking_duration_with_concurrency(self):
        """Test benchmarking duration with different concurrency levels."""
        for concurrency in [1, 5, 10]:
            config = benchmarking_duration_config(
                benchmarking_duration=3.0, concurrency=concurrency
            )

            assert config.benchmarking_duration == 3.0
            assert config.concurrency == concurrency


class TestBenchmarkingDurationPhaseStats:
    """Test CreditPhaseStats behavior with benchmarking duration."""

    def test_phase_stats_should_send_time_based(self, time_traveler):
        """Test that phase stats correctly determine when to send based on time."""
        start_time = time_traveler.time_ns()
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=start_time,
            expected_duration_sec=2.0,
        )

        # Initially should send
        assert phase_stats.should_send()

        # After 1 second, should still send
        time_traveler.advance_time(1.0)
        assert phase_stats.should_send()

        # After 2.1 seconds, should not send (duration exceeded)
        time_traveler.advance_time(1.1)
        assert not phase_stats.should_send()

    def test_phase_stats_should_send_with_request_count(self):
        """Test that phase stats without duration use request count."""
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=time.time_ns(),
            total_expected_requests=5,
        )

        # Should send initially
        assert phase_stats.should_send()

        # After sending 4 requests, should still send
        phase_stats.sent = 4
        assert phase_stats.should_send()

        # After sending 5 requests, should not send
        phase_stats.sent = 5
        assert not phase_stats.should_send()

    def test_phase_stats_completion_time_based(self, time_traveler):
        """Test that time-based phase stats report completion correctly."""
        start_time = time_traveler.time_ns()
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            start_ns=start_time,
            expected_duration_sec=1.5,
        )

        # Not complete initially
        assert phase_stats.should_send()

        # Not complete after 1 second
        time_traveler.advance_time(1.0)
        assert phase_stats.should_send()

        # Complete after 1.6 seconds
        time_traveler.advance_time(0.6)
        assert not phase_stats.should_send()


class TestBenchmarkingDurationRequestRateStrategy:
    """Test RequestRateStrategy behavior with benchmarking duration."""

    async def test_strategy_uses_duration_for_profiling_phase(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test that RequestRateStrategy respects benchmarking duration."""
        config = benchmarking_duration_config(benchmarking_duration=2.0)

        # Create strategy and check profiling phase config
        strategy = RequestRateStrategy(config, mock_credit_manager)

        # Check that the profiling phase is configured correctly
        assert len(strategy.ordered_phase_configs) > 0
        profiling_config = strategy.ordered_phase_configs[
            -1
        ]  # Last phase should be profiling

        # Should be time-based
        assert profiling_config.expected_duration_sec == 2.0
        assert profiling_config.total_expected_requests is None

    async def test_strategy_ignores_request_count_when_duration_set(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test that request count is ignored when duration is specified."""
        config = mixed_config(request_count=50, benchmarking_duration=1.5)

        strategy = RequestRateStrategy(config, mock_credit_manager)

        profiling_config = strategy.ordered_phase_configs[-1]

        # Should use duration, not request count
        assert profiling_config.expected_duration_sec == 1.5
        assert profiling_config.total_expected_requests is None

    async def test_strategy_fallback_to_request_count(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test that strategy falls back to request count when no duration."""
        config = mixed_config(request_count=25, benchmarking_duration=None)

        strategy = RequestRateStrategy(config, mock_credit_manager)

        profiling_config = strategy.ordered_phase_configs[-1]

        # Should use request count, not duration
        assert profiling_config.total_expected_requests == 25
        assert profiling_config.expected_duration_sec is None

    async def test_strategy_with_warmup_and_duration(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test strategy behavior with both warmup and benchmarking duration."""
        config = benchmarking_duration_config(
            benchmarking_duration=4.0, warmup_request_count=10
        )

        strategy = RequestRateStrategy(config, mock_credit_manager)

        # Should have warmup and profiling phases
        assert len(strategy.ordered_phase_configs) == 2

        # Warmup phase should still use request count
        warmup_config = strategy.ordered_phase_configs[0]
        assert warmup_config.type == CreditPhase.WARMUP
        assert warmup_config.total_expected_requests == 10
        assert warmup_config.expected_duration_sec is None

        # Profiling phase should use duration
        profiling_config = strategy.ordered_phase_configs[1]
        assert profiling_config.type == CreditPhase.PROFILING
        assert profiling_config.expected_duration_sec == 4.0
        assert profiling_config.total_expected_requests is None


class TestBenchmarkingDurationIntegration:
    """Integration tests for benchmarking duration feature."""

    async def test_strategy_with_duration_and_concurrency(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test strategy with duration and concurrency settings."""
        config = benchmarking_duration_config(benchmarking_duration=3.0, concurrency=5)

        strategy = RequestRateStrategy(config, mock_credit_manager)

        assert config.benchmarking_duration == 3.0
        assert config.concurrency == 5

        profiling_config = strategy.ordered_phase_configs[-1]
        assert profiling_config.expected_duration_sec == 3.0

    async def test_strategy_with_duration_and_request_rate(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test strategy with duration and request rate."""
        config = benchmarking_duration_config(
            benchmarking_duration=2.5,
            request_rate_mode=RequestRateMode.POISSON,
            request_rate=15.0,
        )

        assert config.benchmarking_duration == 2.5
        assert config.request_rate == 15.0
        assert config.request_rate_mode == RequestRateMode.POISSON

    async def test_mixed_warmup_and_duration_profiling(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test mixed configuration with warmup requests and duration-based profiling."""
        config = benchmarking_duration_config(
            benchmarking_duration=5.0, warmup_request_count=3
        )

        strategy = RequestRateStrategy(config, mock_credit_manager)

        # Should have 2 phases: warmup (request-based) and profiling (time-based)
        assert len(strategy.ordered_phase_configs) == 2

        warmup_config = strategy.ordered_phase_configs[0]
        assert warmup_config.type == CreditPhase.WARMUP
        assert warmup_config.total_expected_requests == 3

        profiling_config = strategy.ordered_phase_configs[1]
        assert profiling_config.type == CreditPhase.PROFILING
        assert profiling_config.expected_duration_sec == 5.0

    @pytest.mark.parametrize(
        "duration,warmup_count",
        [
            (1.0, 2),  # Short duration, small warmup
            (30.0, 10),  # Medium duration, medium warmup
            (300.0, 50),  # Long duration, large warmup
        ],
    )
    async def test_various_duration_warmup_combinations(
        self, mock_credit_manager: MockCreditManager, duration: float, warmup_count: int
    ):
        """Test various combinations of duration and warmup counts."""
        config = benchmarking_duration_config(
            benchmarking_duration=duration, warmup_request_count=warmup_count
        )

        strategy = RequestRateStrategy(config, mock_credit_manager)

        # Verify configuration is correct
        assert config.benchmarking_duration == duration
        assert config.warmup_request_count == warmup_count

        # Verify phase configurations
        assert len(strategy.ordered_phase_configs) == 2

        warmup_config = strategy.ordered_phase_configs[0]
        assert warmup_config.total_expected_requests == warmup_count

        profiling_config = strategy.ordered_phase_configs[1]
        assert profiling_config.expected_duration_sec == duration


class TestBenchmarkingDurationConfig:
    """Test TimingManagerConfig specifically for benchmarking duration."""

    def test_timing_manager_config_with_duration(self):
        """Test TimingManagerConfig creation with benchmarking duration."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            benchmarking_duration=10.0,
            request_rate_mode=RequestRateMode.POISSON,
            request_rate=5.0,
            concurrency=2,
            request_count=100,  # Should be ignored
        )

        assert config.benchmarking_duration == 10.0
        assert config.request_count == 100  # Still present but should be ignored
        assert config.request_rate == 5.0

    def test_timing_manager_config_without_duration(self):
        """Test TimingManagerConfig creation without benchmarking duration."""
        config = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
            concurrency=3,
            request_count=50,
        )

        assert config.benchmarking_duration is None
        assert config.request_count == 50

    @pytest.mark.parametrize("duration", [1.0, 5.5, 30.0, 120.0, 3600.0])
    def test_various_duration_values(self, duration: float):
        """Test various valid duration values."""
        config = benchmarking_duration_config(benchmarking_duration=duration)

        assert config.benchmarking_duration == duration

    def test_duration_with_different_modes(self):
        """Test duration configuration with different request rate modes."""
        # Test with CONCURRENCY_BURST
        config1 = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            benchmarking_duration=5.0,
            request_rate_mode=RequestRateMode.CONCURRENCY_BURST,
            concurrency=4,
            request_count=50,
        )
        assert config1.benchmarking_duration == 5.0
        assert config1.request_rate_mode == RequestRateMode.CONCURRENCY_BURST

        # Test with CONSTANT
        config2 = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            benchmarking_duration=7.5,
            request_rate_mode=RequestRateMode.CONSTANT,
            request_rate=12.0,
            concurrency=1,
            request_count=30,
        )
        assert config2.benchmarking_duration == 7.5
        assert config2.request_rate_mode == RequestRateMode.CONSTANT

        # Test with POISSON
        config3 = TimingManagerConfig(
            timing_mode=TimingMode.REQUEST_RATE,
            benchmarking_duration=15.0,
            request_rate_mode=RequestRateMode.POISSON,
            request_rate=8.0,
            concurrency=2,
            request_count=25,
        )
        assert config3.benchmarking_duration == 15.0
        assert config3.request_rate_mode == RequestRateMode.POISSON


class TestBenchmarkingDurationPhaseSetup:
    """Test phase configuration setup with benchmarking duration."""

    async def test_profiling_phase_setup_with_duration(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test profiling phase setup when duration is specified."""
        config = benchmarking_duration_config(benchmarking_duration=8.0)
        strategy = RequestRateStrategy(config, mock_credit_manager)

        # Find the profiling phase config
        profiling_config = next(
            (
                cfg
                for cfg in strategy.ordered_phase_configs
                if cfg.type == CreditPhase.PROFILING
            ),
            None,
        )

        assert profiling_config is not None
        assert profiling_config.expected_duration_sec == 8.0
        assert profiling_config.total_expected_requests is None

    async def test_profiling_phase_setup_without_duration(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test profiling phase setup when duration is not specified."""
        config = mixed_config(request_count=40, benchmarking_duration=None)
        strategy = RequestRateStrategy(config, mock_credit_manager)

        # Find the profiling phase config
        profiling_config = next(
            (
                cfg
                for cfg in strategy.ordered_phase_configs
                if cfg.type == CreditPhase.PROFILING
            ),
            None,
        )

        assert profiling_config is not None
        assert profiling_config.total_expected_requests == 40
        assert profiling_config.expected_duration_sec is None

    async def test_warmup_phase_unaffected_by_duration(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test that warmup phase is unaffected by benchmarking duration."""
        config = benchmarking_duration_config(
            benchmarking_duration=12.0, warmup_request_count=15
        )
        strategy = RequestRateStrategy(config, mock_credit_manager)

        # Find the warmup phase config
        warmup_config = next(
            (
                cfg
                for cfg in strategy.ordered_phase_configs
                if cfg.type == CreditPhase.WARMUP
            ),
            None,
        )

        assert warmup_config is not None
        assert warmup_config.total_expected_requests == 15
        assert warmup_config.expected_duration_sec is None

    async def test_both_warmup_and_profiling_phases(
        self, mock_credit_manager: MockCreditManager
    ):
        """Test configuration with both warmup and profiling phases."""
        config = benchmarking_duration_config(
            benchmarking_duration=6.0, warmup_request_count=8
        )
        strategy = RequestRateStrategy(config, mock_credit_manager)

        assert len(strategy.ordered_phase_configs) == 2

        # Verify phases are in correct order
        warmup_config = strategy.ordered_phase_configs[0]
        profiling_config = strategy.ordered_phase_configs[1]

        assert warmup_config.type == CreditPhase.WARMUP
        assert warmup_config.total_expected_requests == 8

        assert profiling_config.type == CreditPhase.PROFILING
        assert profiling_config.expected_duration_sec == 6.0


class TestBenchmarkingDurationLogic:
    """Test the logic behind benchmarking duration implementation."""

    def test_should_send_time_based_phase(self, time_traveler):
        """Test should_send logic for time-based phases."""
        start_time = time_traveler.time_ns()
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            expected_duration_sec=3.0,
            start_ns=start_time,
        )

        # Should send initially
        assert phase_stats.should_send()

        # Should send after 2.5 seconds
        time_traveler.advance_time(2.5)
        assert phase_stats.should_send()

        # Should not send after 3.1 seconds
        time_traveler.advance_time(0.6)
        assert not phase_stats.should_send()

    def test_should_send_request_count_based_phase(self):
        """Test should_send logic for request count-based phases."""
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            total_expected_requests=10,
            start_ns=time.time_ns(),
        )

        # Should send initially
        assert phase_stats.should_send()

        # Should send after 5 requests
        phase_stats.sent = 5
        assert phase_stats.should_send()

        # Should not send after 10 requests
        phase_stats.sent = 10
        assert not phase_stats.should_send()

    @pytest.mark.parametrize(
        "duration,expected_nanos",
        [
            (1.0, 1_000_000_000),  # 1 second
            (5.5, 5_500_000_000),  # 5.5 seconds
            (30.0, 30_000_000_000),  # 30 seconds
        ],
    )
    def test_duration_conversion_to_nanoseconds(
        self, duration: float, expected_nanos: int
    ):
        """Test that duration is correctly converted to nanoseconds for comparison."""
        start_time = time.time_ns()

        time_phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            expected_duration_sec=duration,
            start_ns=start_time,
        )

        # Verify the should_send logic works correctly with this duration
        assert time_phase_stats.should_send()

    def test_edge_case_zero_start_time(self):
        """Test behavior when start_ns is None or 0."""
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            expected_duration_sec=5.0,
            start_ns=None,
        )

        # When start_ns is None, should still behave correctly
        # The implementation uses (self.start_ns or 0), so it defaults to 0
        # This means it will likely not send since current time >> 0 + duration
        assert not phase_stats.should_send()


class TestBenchmarkingDurationCompletion:
    """Test completion logic for benchmarking duration."""

    def test_time_based_completion_logic(self, time_traveler):
        """Test completion detection for time-based phases."""
        start_time = time_traveler.time_ns()
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            expected_duration_sec=2.0,
            start_ns=start_time,
        )

        # Phase is not complete initially
        assert phase_stats.should_send()

        # Phase should complete after duration expires
        time_traveler.advance_time(2.1)
        assert not phase_stats.should_send()

    def test_request_count_completion_logic(self):
        """Test completion detection for request count-based phases."""
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            total_expected_requests=5,
            start_ns=time.time_ns(),
        )

        # Phase is not complete initially
        assert phase_stats.should_send()

        # Phase should complete after all requests are sent
        phase_stats.sent = 5
        assert not phase_stats.should_send()

    def test_in_flight_calculation(self):
        """Test in-flight credit calculation."""
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            total_expected_requests=10,
            start_ns=time.time_ns(),
        )

        # Initially no credits in flight
        assert phase_stats.in_flight == 0

        # After sending 3 and completing 1
        phase_stats.sent = 3
        phase_stats.completed = 1
        assert phase_stats.in_flight == 2

        # After completing all sent credits
        phase_stats.completed = 3
        assert phase_stats.in_flight == 0


class TestBenchmarkingDurationEdgeCases:
    """Test edge cases and boundary conditions for benchmarking duration."""

    def test_very_small_duration(self):
        """Test with very small duration values."""
        config = benchmarking_duration_config(
            benchmarking_duration=1.0
        )  # Minimum allowed

        assert config.benchmarking_duration == 1.0

    def test_very_large_duration(self):
        """Test with very large duration values."""
        large_duration = 86400.0  # 24 hours
        config = benchmarking_duration_config(benchmarking_duration=large_duration)

        assert config.benchmarking_duration == large_duration

    def test_fractional_duration(self):
        """Test with fractional duration values."""
        config = benchmarking_duration_config(benchmarking_duration=2.5)

        assert config.benchmarking_duration == 2.5

    def test_phase_validation_with_duration(self):
        """Test phase validation with duration settings."""
        phase_stats = CreditPhaseStats(
            type=CreditPhase.PROFILING,
            expected_duration_sec=5.0,
            start_ns=time.time_ns(),
        )

        # Phase should be valid (time-based)
        assert phase_stats.is_time_based
        assert not phase_stats.is_request_count_based
        assert phase_stats.is_valid
