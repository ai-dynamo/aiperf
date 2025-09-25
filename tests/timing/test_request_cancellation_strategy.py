# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.timing.config import TimingManagerConfig
from aiperf.timing.request_cancellation_strategy import RequestCancellationStrategy


class TestRequestCancellationStrategy:
    """Tests for the RequestCancellationStrategy class."""

    def test_cancellation_disabled_by_default(self):
        """Test that cancellation is disabled when not configured."""
        config = TimingManagerConfig()
        strategy = RequestCancellationStrategy(config)

        assert not strategy.is_cancellation_enabled
        assert not strategy.should_cancel_request()
        assert strategy.get_cancellation_delay_ns() == 0

    def test_cancellation_enabled_with_parameters(self):
        """Test that cancellation is enabled when parameters are configured."""
        config = TimingManagerConfig(
            request_cancellation_rate=0.5,
            request_cancellation_delay=2.0,
        )
        strategy = RequestCancellationStrategy(config)

        assert strategy.is_cancellation_enabled
        assert strategy.get_cancellation_delay_ns() == int(2.0 * NANOS_PER_SECOND)

    def test_deterministic_behavior_with_seed(self):
        """Test that cancellation decisions are deterministic with a seed."""
        config = TimingManagerConfig(
            request_cancellation_rate=50,
            request_cancellation_delay=1.0,
            random_seed=42,
        )

        # Create two strategies with the same seed
        strategy1 = RequestCancellationStrategy(config)
        strategy2 = RequestCancellationStrategy(config)

        # Generate the same sequence of decisions
        decisions1 = [strategy1.should_cancel_request() for _ in range(100)]
        decisions2 = [strategy2.should_cancel_request() for _ in range(100)]

        # Should be identical
        assert decisions1 == decisions2

        # Should have roughly 50% cancellation rate (within reasonable bounds)
        cancellation_count = sum(decisions1)
        assert 30 <= cancellation_count <= 70  # Allow some variance

    def test_zero_cancellation_rate(self):
        """Test that zero cancellation rate never cancels requests."""
        config = TimingManagerConfig(
            request_cancellation_rate=0.0,
            request_cancellation_delay=1.0,
        )
        strategy = RequestCancellationStrategy(config)

        # Should never cancel with 0.0 rate
        for _ in range(100):
            assert not strategy.should_cancel_request()

    def test_full_cancellation_rate(self):
        """Test that 100 cancellation rate always cancels requests."""
        config = TimingManagerConfig(
            request_cancellation_rate=100,
            request_cancellation_delay=1.0,
        )
        strategy = RequestCancellationStrategy(config)

        for _ in range(100):
            assert strategy.should_cancel_request()
