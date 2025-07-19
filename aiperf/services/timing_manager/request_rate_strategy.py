# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
import time

from aiperf.common.enums import RequestRateMode, TimingMode
from aiperf.common.exceptions import InvalidStateError
from aiperf.common.mixins import AsyncTaskManagerMixin
from aiperf.common.models import CreditPhaseStats
from aiperf.services.timing_manager.config import TimingManagerConfig
from aiperf.services.timing_manager.credit_issuing_strategy import (
    CreditIssuingStrategy,
    CreditIssuingStrategyFactory,
    CreditManagerProtocol,
)


@CreditIssuingStrategyFactory.register(TimingMode.REQUEST_RATE)
class RequestRateStrategy(CreditIssuingStrategy, AsyncTaskManagerMixin):
    """
    Strategy for issuing credits based on a specified request rate.

    Supports two modes:
    - CONSTANT: Issues credits at a constant rate with fixed intervals
    - POISSON: Issues credits using a Poisson process with exponentially distributed intervals
    """

    def __init__(
        self, config: TimingManagerConfig, credit_manager: CreditManagerProtocol
    ):
        super().__init__(config=config, credit_manager=credit_manager)

        if config.request_rate is None:
            raise InvalidStateError("Request rate is not set")
        if config.request_count < 1:
            raise InvalidStateError("Request count must be at least 1")

        self._request_rate = config.request_rate
        self._request_rate_mode = config.request_rate_mode

        # Initialize random number generator for reproducibility
        self._random = (
            random.Random(config.random_seed) if config.random_seed else random.Random()
        )

    async def _execute_single_phase(self, phase_stats: CreditPhaseStats) -> None:
        """Execute a single phase. This will not return until the phase sending is complete."""
        # Issue credit drops at the specified rate
        if self._request_rate_mode == RequestRateMode.CONSTANT:
            await self._execute_constant_rate(phase_stats)
        elif self._request_rate_mode == RequestRateMode.POISSON:
            await self._execute_poisson_rate(phase_stats)
        else:
            raise InvalidStateError(
                f"Unsupported request rate mode: {self._request_rate_mode}"
            )

    async def _execute_constant_rate(self, phase_stats: CreditPhaseStats) -> None:
        """Execute credit drops at a constant rate."""

        # The effective time between each credit drop is the inverse of the request rate.
        period_sec = 1.0 / self._request_rate

        # We start by sending the first credit immediately.
        next_drop_at = time.perf_counter()

        while phase_stats.should_send:
            wait_sec = next_drop_at - time.perf_counter()
            if wait_sec > 0:
                await asyncio.sleep(wait_sec)

            self.execute_async(
                self.credit_manager.drop_credit(credit_phase=phase_stats.type)
            )
            phase_stats.sent += 1

            # Instead of naively sleeping for a constant period_sec, we are scheduling the
            # next drop to happen at exactly (next_drop_at + period_sec). This ensures that
            # we do not slowly drift over time based on slight variances in the asyncio.sleep
            # or executing the credit drop task.
            next_drop_at += period_sec

    async def _execute_poisson_rate(self, phase_stats: CreditPhaseStats) -> None:
        """Execute credit drops using Poisson process (exponential inter-arrival times).

        In a Poisson process with rate λ (requests per second), the inter-arrival times
        are exponentially distributed with parameter λ. This models realistic traffic
        patterns where requests arrive randomly but at a consistent average rate.
        """
        while phase_stats.should_send:
            # For Poisson process, inter-arrival times are exponentially distributed.
            # random.expovariate(lambd) generates exponentially distributed random numbers
            # where lambd is the rate parameter (requests per second)
            wait_duration_sec = self._random.expovariate(self._request_rate)

            if wait_duration_sec > 0:
                await asyncio.sleep(wait_duration_sec)

            self.execute_async(
                self.credit_manager.drop_credit(credit_phase=phase_stats.type)
            )
            phase_stats.sent += 1
