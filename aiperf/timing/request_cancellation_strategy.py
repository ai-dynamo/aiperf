# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.timing.config import TimingManagerConfig


class RequestCancellationStrategy:
    """
    Strategy for determining which requests to cancel and when to cancel them.

    Provides deterministic cancellation decisions when a random seed is provided,
    enabling reproducible benchmarking scenarios with request cancellation.
    """

    def __init__(self, config: TimingManagerConfig):
        """Initialize the request cancellation strategy.

        Args:
            config: The timing manager configuration containing cancellation parameters.
        """
        self.config = config

        self._rng = random.Random(config.random_seed)

        self._cancellation_rate = config.request_cancellation_rate / 100.0
        self._cancellation_delay_ns = int(
            config.request_cancellation_delay * NANOS_PER_SECOND
        )

    def should_cancel_request(self) -> bool:
        """Determine if a request should be cancelled.

        Returns:
            True if the request should be cancelled, False otherwise.
        """
        if self._cancellation_rate == 0.0:
            return False

        return self._rng.random() < self._cancellation_rate

    def get_cancellation_delay_ns(self) -> int:
        """Get the delay in nanoseconds after which the request should be cancelled.

        Returns:
            The cancellation delay in nanoseconds.
        """
        return self._cancellation_delay_ns

    @property
    def is_cancellation_enabled(self) -> bool:
        """Check if request cancellation is enabled.

        Returns:
            True if both cancellation rate and delay are configured, False otherwise.
        """
        return self.config.request_cancellation_rate > 0.0
