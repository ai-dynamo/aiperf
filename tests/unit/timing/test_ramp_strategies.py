# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ramp strategies."""

import pytest
from pydantic import ValidationError

from aiperf.timing.ramping import (
    BaseRampStrategy as RampStrategy,
)
from aiperf.timing.ramping import (
    ExponentialStrategy,
    LinearStrategy,
    PoissonStrategy,
    RampConfig,
    RampStrategyFactory,
    RampType,
)


def linear_config(
    start: float, target: float, duration_sec: float, step_size: float | None = None
) -> RampConfig:
    return RampConfig(
        ramp_type=RampType.LINEAR,
        start=start,
        target=target,
        duration_sec=duration_sec,
        step_size=step_size,
    )


def exponential_config(
    start: float, target: float, duration_sec: float, exponent: float = 2.0
) -> RampConfig:
    return RampConfig(
        ramp_type=RampType.EXPONENTIAL,
        start=start,
        target=target,
        duration_sec=duration_sec,
        exponent=exponent,
    )


def poisson_config(start: float, target: float, duration_sec: float) -> RampConfig:
    return RampConfig(
        ramp_type=RampType.POISSON,
        start=start,
        target=target,
        duration_sec=duration_sec,
    )


class TestLinearStrategy:
    def test_protocol_compliance(self) -> None:
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=10.0))
        assert isinstance(strategy, RampStrategy)

    def test_start_target_properties(self) -> None:
        strategy = LinearStrategy(linear_config(start=5, target=50, duration_sec=10.0))
        assert strategy.start == 5
        assert strategy.target == 50

    @pytest.mark.parametrize(
        "start,target,current,expected_next",
        [
            (1, 100, 100, None),  # at target returns None
            (1, 100, 1, 2),  # ramp up increments by one
            (100, 1, 100, 99),  # ramp down decrements by one
            (50, 50, 50, None),  # start equals target returns None
        ],
    )
    def test_next_step_values(
        self, start: int, target: int, current: int, expected_next: int | None
    ) -> None:
        strategy = LinearStrategy(
            linear_config(start=start, target=target, duration_sec=10.0)
        )
        result = strategy.next_step(current, elapsed_sec=0.0)
        if expected_next is None:
            assert result is None
        else:
            assert result is not None
            _, next_val = result
            assert next_val == expected_next

    @pytest.mark.parametrize(
        "start,target,duration,expected_interval",
        [
            (1, 100, 9.9, 9.9 / 99),  # ramp up
            (100, 1, 9.9, 9.9 / 99),  # ramp down
            (1, 500, 1.0, 1.0 / 499),  # precise timing
        ],
    )
    def test_interval_calculation(
        self, start: int, target: int, duration: float, expected_interval: float
    ) -> None:
        strategy = LinearStrategy(
            linear_config(start=start, target=target, duration_sec=duration)
        )
        result = strategy.next_step(start, elapsed_sec=0.0)
        assert result is not None
        delay, _ = result
        assert abs(delay - expected_interval) < 0.000001

    def test_precise_timing_self_corrects(self) -> None:
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=10.0))

        result1 = strategy.next_step(1, elapsed_sec=0.0)
        assert result1 is not None
        delay1, _ = result1
        assert abs(delay1 - 10.0 * (1 / 99)) < 0.0001

        result2 = strategy.next_step(50, elapsed_sec=5.0)
        assert result2 is not None
        delay2, _ = result2
        assert abs(delay2 - (10.0 * (50 / 99) - 5.0)) < 0.0001

    def test_small_ramp_single_step(self) -> None:
        strategy = LinearStrategy(linear_config(start=1, target=2, duration_sec=1.0))
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert delay == 1.0
        assert next_val == 2

    def test_full_ramp_simulation(self) -> None:
        strategy = LinearStrategy(linear_config(start=1, target=10, duration_sec=9.0))
        current = 1
        values = [current]
        while True:
            result = strategy.next_step(current, elapsed_sec=0.0)
            if result is None:
                break
            _, current = result
            values.append(current)
        assert values == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


class TestLinearStrategyWithStepSize:
    @pytest.mark.parametrize(
        "start,target,current,expected_next",
        [
            (1, 100, 1, 11),  # ramp up step
            (100, 1, 100, 90),  # ramp down step
            (1, 100, 95, 100),  # clamp to target (ramp up)
            (100, 1, 5, 1),  # clamp to target (ramp down)
        ],
    )
    def test_step_size_behavior(
        self, start: int, target: int, current: int, expected_next: int
    ) -> None:
        strategy = LinearStrategy(
            linear_config(start=start, target=target, duration_sec=10.0, step_size=10)
        )
        result = strategy.next_step(current, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == expected_next

    def test_precise_timing_calculation(self) -> None:
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=10.0, step_size=10)
        )
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert next_val == 11
        assert abs(delay - 10.0 * (10 / 99)) < 0.0001

    def test_precise_timing_self_corrects(self) -> None:
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=10.0, step_size=10)
        )

        result1 = strategy.next_step(1, elapsed_sec=0.0)
        assert result1 is not None
        delay1, _ = result1
        assert abs(delay1 - 10.0 * (10 / 99)) < 0.0001

        result2 = strategy.next_step(51, elapsed_sec=5.0)
        assert result2 is not None
        delay2, next_val = result2
        assert next_val == 61
        assert abs(delay2 - (10.0 * (60 / 99) - 5.0)) < 0.0001

    def test_full_ramp_simulation_with_step_size(self) -> None:
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=4.0, step_size=25)
        )
        current = 1
        values = [current]
        while True:
            result = strategy.next_step(current, elapsed_sec=0.0)
            if result is None:
                break
            _, current = result
            values.append(current)
        assert values == [1, 26, 51, 76, 100]


class TestExponentialStrategy:
    def test_protocol_compliance(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        assert isinstance(strategy, RampStrategy)

    @pytest.mark.parametrize("exponent", [1.0, 0.5])
    def test_invalid_exponent_raises(self, exponent: float) -> None:
        with pytest.raises(ValidationError, match="greater than 1"):
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=exponent)

    @pytest.mark.parametrize(
        "current,expected_none",
        [
            (100, True),  # at target
            (150, True),  # above target (overshoot)
        ],
    )
    def test_returns_none_at_or_above_target(
        self, current: int, expected_none: bool
    ) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        result = strategy.next_step(current, elapsed_sec=0.0)
        assert (result is None) == expected_none

    def test_always_increments_by_one(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 2

        result2 = strategy.next_step(50, elapsed_sec=0.5)
        assert result2 is not None
        _, next_val2 = result2
        assert next_val2 == 51

    def test_delays_decrease_over_time(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        delays = []
        current = 1
        elapsed = 0.0

        for _ in range(10):
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            delays.append(delay)
            elapsed += delay

        for i in range(1, len(delays)):
            assert delays[i] <= delays[i - 1] + 0.001

    def test_first_delay_is_longest(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        first_delay, _ = result
        assert first_delay > 0.09

    def test_last_delay_is_shortest(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        result = strategy.next_step(99, elapsed_sec=0.99)
        assert result is not None
        last_delay, next_val = result
        assert next_val == 100
        assert last_delay < 0.02

    def test_higher_exponent_slower_start(self) -> None:
        strategy_low = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        strategy_high = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=3.0)
        )

        result_low = strategy_low.next_step(1, elapsed_sec=0.0)
        result_high = strategy_high.next_step(1, elapsed_sec=0.0)
        assert result_low is not None and result_high is not None
        assert result_high[0] > result_low[0]

    def test_full_ramp_simulation(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        current = 1
        elapsed = 0.0
        values = [current]

        while current < 100:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            elapsed += delay
            values.append(current)

        assert values == list(range(1, 101))
        assert abs(elapsed - 1.0) < 0.01

    def test_total_time_matches_duration(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=100, duration_sec=1.0, exponent=2.0)
        )
        current = 1
        elapsed = 0.0
        total_delay = 0.0

        while current < 100:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            total_delay += delay
            elapsed += delay

        assert abs(total_delay - 1.0) < 0.001

    def test_ramp_down_decrements_by_one(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=100, target=1, duration_sec=1.0, exponent=2.0)
        )
        result = strategy.next_step(100, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert next_val == 99

    def test_ramp_down_delays_decrease(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=100, target=1, duration_sec=1.0, exponent=2.0)
        )
        delays = []
        current = 100
        elapsed = 0.0

        for _ in range(10):
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            delays.append(delay)
            elapsed += delay

        for i in range(1, len(delays)):
            assert delays[i] <= delays[i - 1] + 0.001

    def test_ramp_down_full_simulation(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=100, target=1, duration_sec=1.0, exponent=2.0)
        )
        current = 100
        elapsed = 0.0
        values = [current]

        while current > 1:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            elapsed += delay
            values.append(current)

        assert values == list(range(100, 0, -1))
        assert abs(elapsed - 1.0) < 0.01

    def test_returns_none_below_target_ramp_down(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=100, target=1, duration_sec=1.0, exponent=2.0)
        )
        result = strategy.next_step(0, elapsed_sec=0.5)
        assert result is None


class TestStrategyEdgeCases:
    @pytest.mark.parametrize(
        "strategy",
        [
            LinearStrategy(linear_config(start=1, target=1_000_000, duration_sec=100.0)),
            LinearStrategy(linear_config(start=1, target=1_000_000, duration_sec=100.0, step_size=10)),
            ExponentialStrategy(exponential_config(start=1, target=1_000_000, duration_sec=100.0, exponent=2.0)),
            PoissonStrategy(poisson_config(start=1, target=1_000, duration_sec=100.0)),
        ],
    )  # fmt: skip
    def test_handles_large_values(self, strategy: RampStrategy) -> None:
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert next_val > 1
        assert delay > 0

    @pytest.mark.parametrize(
        "strategy",
        [
            LinearStrategy(linear_config(start=1, target=100, duration_sec=0.001)),
            LinearStrategy(linear_config(start=1, target=100, duration_sec=0.001, step_size=10)),
        ],
    )  # fmt: skip
    def test_handles_very_small_duration(self, strategy: RampStrategy) -> None:
        result = strategy.next_step(1, elapsed_sec=0.0)
        assert result is not None
        delay, next_val = result
        assert delay <= 0.001
        assert next_val > 1

    def test_poisson_very_small_duration_returns_few_events(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=0.001)
        )
        result = strategy.next_step(1, elapsed_sec=0.0)
        if result is None:
            assert len(strategy._event_times) == 0
        else:
            delay, _ = result
            assert delay >= 0


class TestRampStrategyFactory:
    def test_factory_creates_linear_strategy(self) -> None:
        config = linear_config(start=1, target=100, duration_sec=10.0)
        strategy = RampStrategyFactory.create_instance(config)
        assert isinstance(strategy, LinearStrategy)
        assert strategy.start == 1
        assert strategy.target == 100

    def test_factory_creates_linear_strategy_with_step_size(self) -> None:
        config = linear_config(start=1, target=100, duration_sec=10.0, step_size=10)
        strategy = RampStrategyFactory.create_instance(config)
        assert isinstance(strategy, LinearStrategy)

    def test_factory_creates_exponential_strategy(self) -> None:
        config = exponential_config(
            start=1, target=100, duration_sec=10.0, exponent=2.0
        )
        strategy = RampStrategyFactory.create_instance(config)
        assert isinstance(strategy, ExponentialStrategy)


class TestValueAt:
    @pytest.mark.parametrize(
        "start,target,elapsed,expected",
        [
            (10, 100, 0.0, 10.0),  # start value at elapsed=0
            (1, 101, 5.0, 51.0),  # midpoint value at half duration
            (100, 1, 5.0, 50.5),  # ramp down midpoint
        ],
    )
    def test_linear_value_at(
        self, start: int, target: int, elapsed: float, expected: float
    ) -> None:
        strategy = LinearStrategy(
            linear_config(start=start, target=target, duration_sec=10.0)
        )
        value = strategy.value_at(elapsed)
        assert value is not None
        assert abs(value - expected) < 0.01

    @pytest.mark.parametrize("elapsed", [10.0, 15.0])
    def test_linear_value_at_returns_none_at_completion(self, elapsed: float) -> None:
        strategy = LinearStrategy(linear_config(start=1, target=100, duration_sec=10.0))
        assert strategy.value_at(elapsed) is None

    def test_exponential_value_at_slow_start(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=101, duration_sec=10.0, exponent=2.0)
        )
        value = strategy.value_at(5.0)
        assert value is not None
        assert value < 51.0
        assert abs(value - 26.0) < 0.1

    def test_exponential_value_at_accelerates(self) -> None:
        strategy = ExponentialStrategy(
            exponential_config(start=1, target=101, duration_sec=10.0, exponent=2.0)
        )
        value = strategy.value_at(8.0)
        assert value is not None
        assert abs(value - 65.0) < 0.1

    def test_linear_with_step_size_value_at_interpolates(self) -> None:
        strategy = LinearStrategy(
            linear_config(start=1, target=101, duration_sec=10.0, step_size=25)
        )
        value = strategy.value_at(5.0)
        assert value is not None
        assert abs(value - 51.0) < 0.01

    @pytest.mark.parametrize("elapsed", [0.0, 5.0])
    def test_value_at_returns_none_for_zero_range(self, elapsed: float) -> None:
        strategy = LinearStrategy(linear_config(start=50, target=50, duration_sec=10.0))
        assert strategy.value_at(elapsed) is None

    @pytest.mark.parametrize("elapsed", [0.001, 0.01])
    def test_value_at_handles_very_small_duration(self, elapsed: float) -> None:
        strategy = LinearStrategy(
            linear_config(start=1, target=100, duration_sec=0.001)
        )
        assert strategy.value_at(elapsed) is None

    def test_higher_exponent_slower_value_progress(self) -> None:
        strategy_exp2 = ExponentialStrategy(
            exponential_config(start=1, target=101, duration_sec=10.0, exponent=2.0)
        )
        strategy_exp3 = ExponentialStrategy(
            exponential_config(start=1, target=101, duration_sec=10.0, exponent=3.0)
        )
        value_exp2 = strategy_exp2.value_at(5.0)
        value_exp3 = strategy_exp3.value_at(5.0)
        assert value_exp2 is not None and value_exp3 is not None
        assert value_exp3 < value_exp2


class TestPoissonStrategy:
    def test_protocol_compliance(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )
        assert isinstance(strategy, RampStrategy)

    def test_start_target_properties(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=5, target=50, duration_sec=10.0)
        )
        assert strategy.start == 5
        assert strategy.target == 50

    def test_returns_none_when_complete(self) -> None:
        strategy = PoissonStrategy(poisson_config(start=1, target=3, duration_sec=1.0))
        result1 = strategy.next_step(1, elapsed_sec=0.0)
        assert result1 is not None
        result2 = strategy.next_step(2, elapsed_sec=0.5)
        assert result2 is not None
        result3 = strategy.next_step(3, elapsed_sec=1.0)
        assert result3 is None

    @pytest.mark.parametrize(
        "start,target,current,check_fn",
        [
            (1, 100, 1, lambda v: 1 < v <= 100),  # ramp up
            (100, 1, 100, lambda v: 1 <= v < 100),  # ramp down
        ],
    )
    def test_ramp_direction(
        self,
        start: int,
        target: int,
        current: int,
        check_fn: callable,
    ) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=start, target=target, duration_sec=10.0)
        )
        result = strategy.next_step(current, elapsed_sec=0.0)
        assert result is not None
        _, next_val = result
        assert check_fn(next_val)

    def test_full_ramp_simulation(self) -> None:
        strategy = PoissonStrategy(poisson_config(start=1, target=10, duration_sec=9.0))
        current = 1
        values = [current]
        while True:
            result = strategy.next_step(current, elapsed_sec=0.0)
            if result is None:
                break
            _, current = result
            values.append(current)

        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]
        assert values[-1] == 10

    def test_total_time_matches_duration(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )
        current = 1
        elapsed = 0.0
        total_delay = 0.0

        while current < 100:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            total_delay += delay
            elapsed += delay

        assert abs(total_delay - 10.0) < 0.001

    def test_intervals_are_variable(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=1, target=20, duration_sec=10.0)
        )
        delays = []
        current = 1
        elapsed = 0.0

        for _ in range(10):
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            delays.append(delay)
            elapsed += delay

        unique_delays = set(round(d, 6) for d in delays)
        assert len(unique_delays) > 1, "Poisson intervals should vary"

    def test_deterministic_with_same_seed(self) -> None:
        strategy1 = PoissonStrategy(
            poisson_config(start=1, target=10, duration_sec=5.0)
        )
        strategy2 = PoissonStrategy(
            poisson_config(start=1, target=10, duration_sec=5.0)
        )

        times1 = strategy1._event_times
        times2 = strategy2._event_times

        assert len(times1) == len(times2)
        for t1, t2 in zip(times1, times2, strict=True):
            assert abs(t1 - t2) < 1e-10

    def test_already_at_target(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=50, target=50, duration_sec=10.0)
        )
        result = strategy.next_step(50, elapsed_sec=0.0)
        assert result is None

    @pytest.mark.parametrize(
        "start,target,check_monotonic,check_boundary",
        [
            (1.0, 10.7, lambda a, b: a <= b, lambda v, t: v <= t),  # ramp up
            (10.7, 1.0, lambda a, b: a >= b, lambda v, t: v >= t),  # ramp down
        ],
    )
    def test_fractional_range(
        self,
        start: float,
        target: float,
        check_monotonic: callable,
        check_boundary: callable,
    ) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=start, target=target, duration_sec=5.0)
        )
        assert len(strategy._event_times) >= 1
        for i in range(1, len(strategy._values)):
            assert check_monotonic(strategy._values[i - 1], strategy._values[i])
        assert strategy._values[0] == start
        assert check_boundary(strategy._values[-1], target)


class TestPoissonStrategyValueAt:
    def test_value_at_start(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=10, target=100, duration_sec=10.0)
        )
        value = strategy.value_at(0.0)
        assert value is not None
        assert value == 10.0

    @pytest.mark.parametrize("elapsed", [10.0, 15.0])
    def test_value_at_returns_none_at_completion(self, elapsed: float) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )
        assert strategy.value_at(elapsed) is None

    @pytest.mark.parametrize("elapsed", [0.0, 5.0])
    def test_value_at_returns_none_for_zero_range(self, elapsed: float) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=50, target=50, duration_sec=10.0)
        )
        assert strategy.value_at(elapsed) is None

    def test_value_at_is_step_function(self) -> None:
        strategy = PoissonStrategy(poisson_config(start=1, target=10, duration_sec=9.0))
        event_times = strategy._event_times
        values = strategy._values

        if event_times:
            just_before = event_times[0] - 0.001
            if just_before > 0:
                value = strategy.value_at(just_before)
                assert value == values[0]

            just_after = event_times[0] + 0.001
            if just_after < 9.0:
                value = strategy.value_at(just_after)
                assert value == values[1]

    def test_value_at_returns_valid_value_near_end(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=1, target=100, duration_sec=10.0)
        )
        value = strategy.value_at(9.999)
        assert value is not None
        assert 1 <= value <= 100

    def test_value_at_handles_ramp_down(self) -> None:
        strategy = PoissonStrategy(
            poisson_config(start=100, target=1, duration_sec=10.0)
        )
        value_start = strategy.value_at(0.0)
        assert value_start == 100

        value_mid = strategy.value_at(5.0)
        assert value_mid is not None
        assert value_mid < 100

    def test_value_at_consistent_with_next_step(self) -> None:
        strategy = PoissonStrategy(poisson_config(start=1, target=10, duration_sec=5.0))
        trajectory: list[tuple[float, float]] = [(0.0, 1.0)]
        elapsed = 0.0
        current = 1
        event_times = list(strategy._event_times)

        while True:
            result = strategy.next_step(current, elapsed_sec=elapsed)
            if result is None:
                break
            delay, current = result
            elapsed += delay
            trajectory.append((elapsed, float(current)))

        strategy2 = PoissonStrategy(
            poisson_config(start=1, target=10, duration_sec=5.0)
        )
        for i, event_time in enumerate(event_times):
            value = strategy2.value_at(event_time + 0.0001)
            if value is not None:
                expected = trajectory[i + 1][1]
                assert value == expected


class TestPoissonStrategyFactory:
    def test_factory_creates_poisson_strategy(self) -> None:
        config = poisson_config(start=1, target=100, duration_sec=10.0)
        strategy = RampStrategyFactory.create_instance(config)
        assert isinstance(strategy, PoissonStrategy)
        assert strategy.start == 1
        assert strategy.target == 100
