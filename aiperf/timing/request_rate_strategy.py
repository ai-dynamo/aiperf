# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random

from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import TimingMode
from aiperf.common.enums.timing_enums import RequestRateMode
from aiperf.common.factories import RequestRateGeneratorFactory
from aiperf.common.messages import CreditReturnMessage
from aiperf.common.models import CreditPhaseStats
from aiperf.common.protocols import RequestRateGeneratorProtocol
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
    CreditManagerProtocol,
)


@CreditIssuingStrategyFactory.register(TimingMode.REQUEST_RATE)
class RequestRateStrategy(CreditIssuingStrategy):
    """
    Strategy for issuing credits based on a specified request rate. Optionally, a max concurrency limit can be specified.

    Supports three modes:
    - CONSTANT: Issues credits at a constant rate with fixed intervals
    - POISSON: Issues credits using a Poisson process with exponentially distributed intervals
    - CONCURRENCY_BURST: Issues credits as soon as possible, up to a max concurrency limit. Only allowed when a request rate is not specified.
    """

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__(config=config, credit_manager=credit_manager)
        self._request_rate_generator = RequestRateGeneratorFactory.create_instance(
            config
        )
        # If the user has provided a concurrency, use a semaphore to limit the maximum number of concurrent requests
        self._semaphore: asyncio.Semaphore | None = (
            asyncio.Semaphore(value=config.concurrency) if config.concurrency else None
        )

    async def _execute_single_phase(self, phase_stats: CreditPhaseStats) -> None:
        """Execute credit drops based on the request rate generator, optionally with a max concurrency limit."""

        loop_count = 0
        while phase_stats.should_send():
            loop_count += 1

            # Ensure we have an available credit before dropping
            if self._semaphore:
                await self._semaphore.acquire()
                if self.is_trace_enabled:
                    self.trace(f"Acquired credit drop semaphore: {self._semaphore!r}")
                if not phase_stats.should_send():
                    # Check one last time to see if we should still send a credit in case the
                    # time-based phase expired while we were waiting for the semaphore.
                    self._semaphore.release()
                    if self.is_trace_enabled:
                        self.trace(
                            f"Released semaphore after should_send returned False: {self._semaphore!r}"
                        )
                    break

            await self.credit_manager.drop_credit(credit_phase=phase_stats.type)
            phase_stats.sent += 1
            # Check if we should break out of the loop before we sleep for the next interval.
            # This is to ensure we don't sleep for any unnecessary time, which could cause race conditions.
            if not phase_stats.should_send():
                break

            next_interval = self._request_rate_generator.next_interval()
            if next_interval > 0:
                await asyncio.sleep(next_interval)

    async def _on_credit_return(self, message: CreditReturnMessage) -> None:
        """Process a credit return message. If concurrency is enabled, release the semaphore to allow another credit to be issued."""

        # Release the semaphore to allow another credit to be issued,
        # then call the superclass to handle the credit return like normal
        if self._semaphore:
            self._semaphore.release()
            if self.is_trace_enabled:
                self.trace(f"Credit return released semaphore: {self._semaphore!r}")
        await super()._on_credit_return(message)


@implements_protocol(RequestRateGeneratorProtocol)
@RequestRateGeneratorFactory.register(RequestRateMode.POISSON)
class PoissonRateGenerator:
    """
    Generator for Poisson process (exponential inter-arrival times).

    In a Poisson process with rate λ (requests per second), the inter-arrival times
    are exponentially distributed with parameter λ. This attempts to model more
    realistic traffic patterns where requests arrive randomly but at a consistent
    average rate.
    """

    def __init__(self, config: TimingManagerConfig) -> None:
        if config.request_rate is None or config.request_rate <= 0:
            raise ValueError(
                f"Request rate {config.request_rate} must be set and greater than 0 for {config.request_rate_mode!r}"
            )
        # Initialize random number generator for reproducibility
        self._rng = (
            random.Random(config.random_seed) if config.random_seed else random.Random()
        )
        self._request_rate: float = config.request_rate

    def next_interval(self) -> float:
        """
        Generate the next inter-arrival time for a Poisson process.

        For Poisson process, inter-arrival times are exponentially distributed.
        random.expovariate(lambd) generates exponentially distributed random numbers
        where lambd is the rate parameter (requests per second)
        """
        return self._rng.expovariate(self._request_rate)


@implements_protocol(RequestRateGeneratorProtocol)
@RequestRateGeneratorFactory.register(RequestRateMode.CONSTANT)
class ConstantRateGenerator:
    """
    Generator for constant rate (fixed inter-arrival times).

    This generates a fixed inter-arrival time for each request.
    """

    def __init__(self, config: TimingManagerConfig) -> None:
        if config.request_rate is None or config.request_rate <= 0:
            raise ValueError(
                f"Request rate {config.request_rate} must be set and greater than 0 for {config.request_rate_mode!r}"
            )
        self._period: float = 1.0 / config.request_rate

    def next_interval(self) -> float:
        """
        Generate the next inter-arrival time for a constant rate.
        """
        return self._period


@implements_protocol(RequestRateGeneratorProtocol)
@RequestRateGeneratorFactory.register(RequestRateMode.CONCURRENCY_BURST)
class ConcurrencyBurstRateGenerator:
    """
    Generator for concurrency-burst rate (no delay between requests).
    """

    def __init__(self, config: TimingManagerConfig) -> None:
        if config.concurrency is None or config.concurrency < 1:
            raise ValueError(
                f"Concurrency {config.concurrency} must be set and greater than 0 for {config.request_rate_mode!r}"
            )
        if config.request_rate is not None:
            raise ValueError(
                f"Request rate {config.request_rate} should be None for {config.request_rate_mode!r}"
            )

    def next_interval(self) -> float:
        """
        Generate the next inter-arrival time for a concurrency-burst rate.

        This will always return 0, as the requests should be issued as soon as possible.
        """
        return 0
