# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic value ramper with pluggable strategies.

Provides a Ramper class that manages the polling loop and applies values
via a setter function. The strategy is selected via RampConfig.ramp_type
and created by the factory.

Supports two modes:
- Discrete mode (default): Uses next_step() for value-centric ramping with +1 steps
- Continuous mode (update_interval set): Uses value_at() for time-centric ramping

Example:
    ```python
    from aiperf.timing.ramp import Ramper, RampConfig
    from aiperf.plugin.enums import RampType

    # Discrete mode: +1 steps with timing derived from value count
    config = RampConfig(
        ramp_type=RampType.LINEAR,
        start=1,
        target=100,
        duration_sec=30.0,
    )
    ramper = Ramper(setter=concurrency_manager.set_limit, config=config)

    # Continuous mode: update every 2 seconds with interpolated values
    config = RampConfig(
        ramp_type=RampType.LINEAR,
        start=1.0,
        target=100.0,
        duration_sec=30.0,
        update_interval=2.0,
    )
    ramper = Ramper(setter=rate_limiter.set_rate, config=config)

    # Start the ramp (returns task)
    task = ramper.start()
    await task  # Wait for completion

    # Or stop early:
    ramper.stop()  # Cancels task, stays at current value
    ```
"""

import asyncio
import bisect
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from pydantic import ConfigDict, Field

from aiperf.common import random_generator as rng
from aiperf.common.models import AIPerfBaseModel
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, RampType

# =============================================================================
# RampConfig - Configuration for ramp strategies
# =============================================================================


class RampConfig(AIPerfBaseModel):
    """Configuration for ramp strategies.

    Values are floats throughout the ramp system. Callers needing integer values
    (e.g., concurrency limits) should cast in their setter function.
    """

    model_config = ConfigDict(frozen=True)

    ramp_type: RampType = Field(..., description="The ramp algorithm to use.")
    start: float = Field(..., gt=0, description="Starting value for the ramp.")
    target: float = Field(..., gt=0, description="Target value for the ramp.")
    duration_sec: float = Field(
        ..., gt=0, description="Duration of the ramp in seconds."
    )
    update_interval: float | None = Field(
        default=None,
        gt=0,
        description="Time between updates in seconds. When set, uses value_at() for continuous sampling.",
    )
    step_size: float | None = Field(
        default=None,
        gt=0,
        description="Step size for LinearStrategy in discrete mode (default: 1).",
    )
    exponent: float | None = Field(
        default=None,
        gt=1.0,
        description="Exponent for ExponentialStrategy (default: 2.0). Must be > 1.0.",
    )


# =============================================================================
# RampStrategyProtocol - Ramp strategies
# =============================================================================


@runtime_checkable
class RampStrategyProtocol(Protocol):
    """Protocol for ramp algorithms."""

    def __init__(self, config: RampConfig) -> None: ...

    start: float
    """Starting value for the ramp."""

    target: float
    """Target value for the ramp."""

    def next_step(
        self, current: float, elapsed_sec: float
    ) -> tuple[float, float] | None:
        """Return (delay_sec, next_value) or None if ramp complete. For discrete stepping.

        Returns the next value to apply and how long to wait before applying it.
        """
        ...

    def value_at(self, elapsed_sec: float) -> float | None:
        """Return value at elapsed time, or None if complete. For continuous sampling."""
        ...


# ==============================================================================
# Ramper
# ==============================================================================


class Ramper:
    """Generic value ramper with pluggable strategies.

    Manages the polling loop and applies values via setter function.
    The strategy is created from config via plugins.

    On completion, sets the target value to ensure we reach the destination.
    On stop/cancel, stays at current value (caller decides what to do next).
    """

    def __init__(
        self,
        setter: Callable[[float], None],
        config: RampConfig,
    ) -> None:
        """Initialize the ramper.

        Args:
            setter: Function to call with each new value (float).
            config: Ramp configuration (ramp_type selects the strategy).
        """
        self._setter = setter
        self._config = config
        RampClass = plugins.get_class(PluginType.RAMP, config.ramp_type)
        self._strategy = RampClass(config=config)
        self._task: asyncio.Task | None = None

    @property
    def is_running(self) -> bool:
        """Return True if the ramp task is currently running."""
        return self._task is not None and not self._task.done()

    def start(self) -> asyncio.Task:
        """Start ramping in background task.

        Returns:
            Task that completes when ramp finishes or is stopped.
        """
        if self._config.update_interval is not None:
            self._task = asyncio.create_task(self._run_continuous())
        else:
            self._task = asyncio.create_task(self._run_discrete())
        return self._task

    def stop(self) -> None:
        """Stop the ramp early (stays at current value).

        Safe to call multiple times or if ramp already completed.
        """
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def _run_discrete(self) -> None:
        """Execute discrete ramp loop using next_step().

        Polls the strategy for (delay, next_value) pairs and applies them.
        On completion, sets target value. On cancellation, stays at current.
        """
        current = self._strategy.start
        ramp_start = time.perf_counter()

        # Set initial value
        self._setter(current)

        try:
            while True:
                elapsed = time.perf_counter() - ramp_start
                result = self._strategy.next_step(current, elapsed)
                if result is None:
                    # Ramp complete - ensure target value is set (handles edge cases
                    # like timing drift or start == target)
                    if current != self._strategy.target:
                        self._setter(self._strategy.target)
                    break

                delay, next_value = result
                await asyncio.sleep(delay)

                current = next_value
                self._setter(current)

        except asyncio.CancelledError:
            # On cancel, stay at current value - caller decides what to do next
            pass

    async def _run_continuous(self) -> None:
        """Execute continuous ramp loop using value_at().

        Sleeps for update_interval, then queries value_at() for interpolated value.
        On completion, sets target value. On cancellation, stays at current.
        """
        update_interval = self._config.update_interval
        ramp_start = time.perf_counter()

        # Set initial value
        self._setter(self._strategy.start)

        try:
            while True:
                await asyncio.sleep(update_interval)

                elapsed = time.perf_counter() - ramp_start
                value = self._strategy.value_at(elapsed)

                if value is None:
                    # Ramp complete - set target value
                    self._setter(self._strategy.target)
                    break

                self._setter(value)

        except asyncio.CancelledError:
            # On cancel, stay at current value - caller decides what to do next
            pass


# =============================================================================
# BaseRampStrategy - Base class for ramp strategies
# =============================================================================


class BaseRampStrategy(ABC):
    """Base class with common ramp logic. Config computed once in __init__.

    Subclasses override _compute_next_value for discrete mode and optionally:
    - _apply_curve: modify time-to-progress mapping for non-linear curves
    - _time_to_value_progress: inverse of _apply_curve for continuous sampling
    """

    def __init__(self, config: RampConfig) -> None:
        self._start = config.start
        self._target = config.target
        self._duration = config.duration_sec
        self._range = abs(config.target - config.start)
        self._direction = (
            0.0 if self._range == 0 else (1.0 if config.target > config.start else -1.0)
        )

    @property
    def start(self) -> float:
        """Starting value for the ramp."""
        return self._start

    @property
    def target(self) -> float:
        """Target value for the ramp."""
        return self._target

    def next_step(
        self, current: float, elapsed_sec: float
    ) -> tuple[float, float] | None:
        """Compute next (delay, value) or None if ramp complete.

        For discrete stepping mode. Returns the next value and delay until it should
        be applied. Uses _compute_next_value to determine discrete steps.
        """
        if current == self._target or self._range == 0:
            return None

        # Check if overshot (can happen with external modifications)
        if (self._direction > 0 and current > self._target) or (
            self._direction < 0 and current < self._target
        ):
            return None

        next_val = self._compute_next_value(current)
        progress = min(1.0, max(0.0, abs(next_val - self._start) / self._range))
        time_at_next = self._duration * self._apply_curve(progress)
        delay = max(0.0, time_at_next - elapsed_sec)

        return (delay, next_val)

    def value_at(self, elapsed_sec: float) -> float | None:
        """Return interpolated value at elapsed time, or None if complete.

        For continuous sampling mode. Uses linear interpolation by default,
        with _apply_curve modifying the time-to-progress mapping for non-linear
        ramps (e.g., exponential ease-in).
        """
        if self._range == 0:
            return None

        if elapsed_sec >= self._duration:
            return None

        # Compute time progress [0, 1]
        time_progress = (
            min(1.0, max(0.0, elapsed_sec / self._duration))
            if self._duration > 0
            else 1.0
        )

        # Apply inverse curve to get value progress
        # For linear: value_progress = time_progress
        # For exponential ease-in: slower value progress early, faster later
        value_progress = self._time_to_value_progress(time_progress)

        # Interpolate value
        return self._start + (self._range * self._direction * value_progress)

    @abstractmethod
    def _compute_next_value(self, current: float) -> float:
        """Compute the next value in the ramp. For discrete stepping."""
        ...

    def _apply_curve(self, progress: float) -> float:
        """Transform value progress [0,1] to time fraction [0,1]. Default: linear."""
        return progress

    def _time_to_value_progress(self, time_progress: float) -> float:
        """Transform time progress [0,1] to value progress [0,1]. Default: linear.

        Used by continuous sampling mode to compute value at any point in time.
        For non-linear strategies, this must be the mathematical inverse of
        _apply_curve. Override in subclasses that have non-linear curves.
        """
        return time_progress


# =============================================================================
# LinearStrategy - Linear ramp strategy
# =============================================================================


class LinearStrategy(BaseRampStrategy):
    """Linear ramp: steps by step_size (default 1) at evenly spaced intervals."""

    def __init__(self, config: RampConfig) -> None:
        super().__init__(config)
        self._step_size = config.step_size if config.step_size is not None else 1.0

    def _compute_next_value(self, current: float) -> float:
        next_val = current + (self._step_size * self._direction)
        if self._direction > 0:
            return min(next_val, self._target)
        return max(next_val, self._target)


# =============================================================================
# ExponentialStrategy - Exponential ramp strategy
# =============================================================================


class ExponentialStrategy(BaseRampStrategy):
    """Exponential ease-in: slow start accelerating to target."""

    def __init__(self, config: RampConfig) -> None:
        super().__init__(config)
        exponent = config.exponent if config.exponent is not None else 2.0
        if exponent <= 1.0:
            raise ValueError(f"exponent must be > 1.0, got {exponent}")
        self._exponent = exponent
        self._inv_exponent = 1.0 / exponent

    def _compute_next_value(self, current: float) -> float:
        return current + self._direction

    def _apply_curve(self, progress: float) -> float:
        """Map value progress to time progress (ease-in: fast time early)."""
        return progress**self._inv_exponent

    def _time_to_value_progress(self, time_progress: float) -> float:
        """Map time progress to value progress (ease-in: slow value early)."""
        return time_progress**self._exponent


# =============================================================================
# PoissonStrategy - Poisson ramp strategy
# =============================================================================


class PoissonStrategy(BaseRampStrategy):
    """Poisson ramp: exponentially-distributed intervals normalized to fit duration.

    Pre-generates a trajectory using a Poisson process (exponentially-distributed
    inter-arrival times), then normalizes to guarantee completion. This provides:

    - Poisson-like burstiness: Variable intervals from exponential distribution
    - Guaranteed completion: Times scaled to fit duration, values scaled to reach target
    - Stochastic step count: Number of events emerges from the process
    - Deterministic replay: Same seed produces identical trajectories

    The trajectory is a step function - values jump at event times and stay constant
    between them, matching how Poisson processes actually behave.
    """

    def __init__(self, config: RampConfig) -> None:
        super().__init__(config)
        self._rng = rng.derive("timing.ramp.poisson")
        self._step_index = 0

        if self._range == 0 or self._duration == 0:
            self._event_times: list[float] = []
            self._values: list[float] = [self._start]
            return

        # Generate Poisson process with rate λ = range / duration
        # Event count is stochastic - generate until we exceed duration
        expected_rate = self._range / self._duration
        raw_intervals: list[float] = []
        cumulative = 0.0

        while cumulative < self._duration:
            interval = self._rng.expovariate(expected_rate)
            raw_intervals.append(interval)
            cumulative += interval

        # Scale intervals to fit exactly in duration
        time_scale = self._duration / cumulative

        # Scale step size to guarantee reaching target
        num_events = len(raw_intervals)
        step_size = self._range / num_events

        # Build timeline: scale intervals and step values toward target
        self._values = [self._start]
        self._event_times = []
        cumulative = 0.0
        for i, interval in enumerate(raw_intervals):
            cumulative += interval * time_scale
            # Use exact target for final event to avoid floating point drift
            value = (
                self._target
                if i == num_events - 1
                else self._start + step_size * (i + 1) * self._direction
            )
            self._event_times.append(cumulative)
            self._values.append(value)

    def _compute_next_value(self, current: float) -> float:
        """Abstract method stub - PoissonStrategy uses pre-computed trajectory."""
        raise NotImplementedError("PoissonStrategy uses pre-computed trajectory")

    def next_step(
        self, current: float, elapsed_sec: float
    ) -> tuple[float, float] | None:
        """Return next (delay, value) from pre-computed trajectory."""
        if self._step_index >= len(self._event_times):
            return None

        target_time = self._event_times[self._step_index]
        delay = max(0.0, target_time - elapsed_sec)
        next_val = self._values[self._step_index + 1]
        self._step_index += 1

        return (delay, next_val)

    def value_at(self, elapsed_sec: float) -> float | None:
        """Return value at elapsed time from pre-computed trajectory (step function)."""
        if not self._event_times or elapsed_sec >= self._duration:
            return None

        # Binary search: find first event AFTER elapsed_sec
        idx = bisect.bisect_right(self._event_times, elapsed_sec)
        return self._values[idx]
