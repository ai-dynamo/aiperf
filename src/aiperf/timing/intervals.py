# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interval generators for credit timing.

Modes for inter-arrival time generation:
- Poisson: Exponential distribution for realistic traffic patterns
- Gamma: Gamma distribution for realistic traffic patterns with tunable burstiness
- Constant: Fixed intervals for deterministic benchmarks
- ConcurrencyBurst: Zero delay, throughput controlled by concurrency semaphore
"""

from typing import Protocol, runtime_checkable

from pydantic import Field
from typing_extensions import Self

from aiperf.common import random_generator as rng
from aiperf.common.models import AIPerfBaseModel
from aiperf.plugin.enums import ArrivalPattern
from aiperf.timing.config import CreditPhaseConfig

# ==============================================================================
# Configuration
# ==============================================================================


class IntervalGeneratorConfig(AIPerfBaseModel):
    """Configuration for interval generators."""

    arrival_pattern: ArrivalPattern = Field(
        ..., description="The arrival pattern to use."
    )
    request_rate: float | None = Field(
        default=None, description="The request rate (requests per second)."
    )
    arrival_smoothness: float | None = Field(
        default=None,
        description="The smoothness parameter for gamma distribution arrivals.",
    )

    @classmethod
    def from_phase_config(cls, phase_config: CreditPhaseConfig) -> Self:
        """Create an IntervalGeneratorConfig from a CreditPhaseConfig."""
        return cls(
            arrival_pattern=phase_config.arrival_pattern,
            request_rate=phase_config.request_rate,
            arrival_smoothness=phase_config.arrival_smoothness,
        )


# ==============================================================================
# Protocol
# ==============================================================================


@runtime_checkable
class IntervalGeneratorProtocol(Protocol):
    """Protocol for a interval generator that generates the next interval for a request rate.

    Supports dynamic rate adjustment via set_rate() for rate ramping during phases.
    """

    def __init__(self, config: IntervalGeneratorConfig) -> None: ...

    def next_interval(self) -> float: ...

    def set_rate(self, new_rate: float) -> None:
        """Update the request rate dynamically for rate ramping.

        Args:
            new_rate: New request rate (requests per second, must be > 0).

        Note:
            Change takes effect on the next next_interval() call.
        """
        ...


# ==============================================================================
# Implementations
# ==============================================================================


def _validate_request_rate(
    request_rate: float | None, arrival_pattern: ArrivalPattern
) -> None:
    if request_rate is None or request_rate <= 0:
        raise ValueError(
            f"Request rate {request_rate} must be set and greater than 0 for {arrival_pattern!r}"
        )


class PoissonIntervalGenerator:
    """Poisson process with exponential inter-arrival times.

    Models real-world traffic patterns where requests arrive randomly but at
    a consistent average rate. This tests how the system handles natural
    load variability rather than artificially even spacing.
    """

    def __init__(self, config: IntervalGeneratorConfig) -> None:
        """Initialize with request rate from config."""
        _validate_request_rate(config.request_rate, config.arrival_pattern)

        self._rng = rng.derive("timing.request.poisson_interval")
        self._request_rate: float = config.request_rate

    @property
    def rate(self) -> float:
        """Current request rate (requests per second)."""
        return self._request_rate

    def set_rate(self, new_rate: float) -> None:
        """Update the request rate dynamically."""
        if new_rate <= 0:
            raise ValueError(f"Rate must be > 0, got {new_rate}")
        self._request_rate = new_rate

    def next_interval(self) -> float:
        """Generate exponentially distributed inter-arrival time."""
        return self._rng.expovariate(self._request_rate)


class GammaIntervalGenerator:
    """Gamma distribution with tunable smoothness for inter-arrival times.

    Generalizes Poisson arrivals by adding a smoothness parameter (Gamma shape):
    - smoothness = 1.0: Equivalent to Poisson (exponential inter-arrivals)
    - smoothness < 1.0: More bursty/clustered arrivals (higher variance)
    - smoothness > 1.0: More regular/smooth arrivals (lower variance)

    This allows benchmarking with realistic traffic patterns that match
    vLLM's burstiness parameter (same value = same distribution).
    """

    def __init__(self, config: IntervalGeneratorConfig) -> None:
        """Initialize with request rate and smoothness from config."""
        _validate_request_rate(config.request_rate, config.arrival_pattern)

        self._rng = rng.derive("timing.request.gamma_interval")
        self._request_rate: float = config.request_rate
        # Default smoothness to 1.0 (equivalent to Poisson) if not specified
        self._smoothness: float = config.arrival_smoothness or 1.0
        self._update_gamma_params()

    def _update_gamma_params(self) -> None:
        """Update gamma distribution parameters based on current rate and smoothness."""
        # shape = smoothness (controls burstiness)
        # scale = 1 / (rate * smoothness) to maintain correct mean = 1/rate
        self._gamma_shape = self._smoothness
        self._gamma_scale = 1.0 / (self._request_rate * self._smoothness)

    @property
    def rate(self) -> float:
        """Current request rate (requests per second)."""
        return self._request_rate

    @property
    def smoothness(self) -> float:
        """Current smoothness parameter (Gamma shape)."""
        return self._smoothness

    def set_rate(self, new_rate: float) -> None:
        """Update the request rate dynamically."""
        if new_rate <= 0:
            raise ValueError(f"Rate must be > 0, got {new_rate}")
        self._request_rate = new_rate
        self._update_gamma_params()

    def next_interval(self) -> float:
        """Generate gamma distributed inter-arrival time."""
        return self._rng.gammavariate(self._gamma_shape, self._gamma_scale)


class ConstantIntervalGenerator:
    """Constant inter-arrival times with fixed intervals.

    Produces perfectly regular intervals (period = 1/rate).
    Useful for deterministic benchmarks and baseline measurements.
    """

    def __init__(self, config: IntervalGeneratorConfig) -> None:
        """Initialize with request rate from config."""
        _validate_request_rate(config.request_rate, config.arrival_pattern)
        self._request_rate: float = config.request_rate
        self._period: float = 1.0 / config.request_rate

    @property
    def rate(self) -> float:
        """Current request rate (requests per second)."""
        return self._request_rate

    def set_rate(self, new_rate: float) -> None:
        """Update the request rate dynamically."""
        if new_rate <= 0:
            raise ValueError(f"Rate must be > 0, got {new_rate}")
        self._request_rate = new_rate
        self._period = 1.0 / new_rate

    def next_interval(self) -> float:
        """Return fixed inter-arrival time."""
        return self._period


class ConcurrencyBurstIntervalGenerator:
    """Burst arrival pattern with zero delay (concurrency-driven throughput).

    Issues credits immediately, limited only by concurrency semaphore.
    This maximizes throughput by keeping exactly N requests in flight at all
    times. Effective rate depends on server response time.

    Note:
        set_rate() is a no-op for burst mode since throughput is controlled
        by concurrency, not request rate.
    """

    def __init__(self, config: IntervalGeneratorConfig) -> None:
        pass

    @property
    def rate(self) -> float:
        """Current request rate (always 0 for burst mode - rate not applicable)."""
        return 0.0

    def set_rate(self, new_rate: float) -> None:
        """No-op for burst mode (throughput controlled by concurrency, not rate)."""
        pass

    def next_interval(self) -> float:
        """Return zero (no delay between credits)."""
        return 0.0
