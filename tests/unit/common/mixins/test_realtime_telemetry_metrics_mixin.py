# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aiperf.common.config import ServiceConfig
from aiperf.common.hooks import AIPerfHook
from aiperf.common.messages import RealtimeTelemetryMetricsMessage
from aiperf.common.mixins.realtime_telemetry_metrics_mixin import (
    RealtimeTelemetryMetricsMixin,
)
from aiperf.common.models import MetricResult


class TestRealtimeTelemetryMetricsMixin:
    """Test suite for RealtimeTelemetryMetricsMixin functionality."""

    @pytest.fixture
    def mocked_mixin(self):
        """Create a RealtimeTelemetryMetricsMixin instance with mocked dependencies."""
        service_config = ServiceConfig()
        mock_controller = MagicMock()

        # Mock the MessageBusClientMixin.__init__ to avoid initialization issues
        with patch(
            "aiperf.common.mixins.message_bus_mixin.MessageBusClientMixin.__init__",
            return_value=None,
        ):
            mixin = RealtimeTelemetryMetricsMixin(
                service_config=service_config, controller=mock_controller
            )
            # Manually set attributes that would be set by parent __init__
            mixin._controller = mock_controller
            mixin._telemetry_metrics = []
            mixin.run_hooks = AsyncMock()
            mixin.debug = MagicMock()

        return mixin

    def test_mixin_initialization(self, mocked_mixin):
        """Test that mixin initializes with correct attributes."""
        assert hasattr(mocked_mixin, "_controller")
        assert hasattr(mocked_mixin, "_telemetry_metrics")
        assert hasattr(mocked_mixin, "_telemetry_metrics_lock")
        assert mocked_mixin._telemetry_metrics == []

    @pytest.mark.asyncio
    async def test_on_realtime_telemetry_metrics_stores_metrics(self, mocked_mixin):
        """Test that telemetry metrics are stored when message is received."""
        metrics = [
            MetricResult(tag="gpu_util", header="GPU Utilization", unit="%", avg=75.0),
            MetricResult(
                tag="gpu_memory", header="GPU Memory Used", unit="GB", avg=8.5
            ),
        ]

        message = RealtimeTelemetryMetricsMessage(
            service_id="records_manager", metrics=metrics
        )

        await mocked_mixin._on_realtime_telemetry_metrics(message)

        # Verify metrics were stored
        assert mocked_mixin._telemetry_metrics == metrics

    @pytest.mark.asyncio
    async def test_on_realtime_telemetry_metrics_triggers_hook(self, mocked_mixin):
        """Test that receiving telemetry metrics triggers the appropriate hook."""

        metrics = [
            MetricResult(tag="gpu_util", header="GPU Utilization", unit="%", avg=75.0)
        ]

        message = RealtimeTelemetryMetricsMessage(
            service_id="records_manager", metrics=metrics
        )

        await mocked_mixin._on_realtime_telemetry_metrics(message)

        # Verify hook was triggered with correct arguments
        mocked_mixin.run_hooks.assert_called_once_with(
            AIPerfHook.ON_REALTIME_TELEMETRY_METRICS, metrics=metrics
        )

    @pytest.mark.asyncio
    async def test_on_realtime_telemetry_metrics_replaces_previous_metrics(
        self, mocked_mixin
    ):
        """Test that new metrics replace previous metrics (not append)."""
        # Set initial metrics
        initial_metrics = [
            MetricResult(tag="old_metric", header="Old Metric", unit="ms", avg=10.0)
        ]
        mocked_mixin._telemetry_metrics = initial_metrics

        # Receive new metrics
        new_metrics = [
            MetricResult(tag="new_metric", header="New Metric", unit="%", avg=50.0)
        ]
        message = RealtimeTelemetryMetricsMessage(
            service_id="records_manager", metrics=new_metrics
        )

        await mocked_mixin._on_realtime_telemetry_metrics(message)

        # Verify old metrics were replaced, not appended
        assert mocked_mixin._telemetry_metrics == new_metrics
        assert len(mocked_mixin._telemetry_metrics) == 1

    @pytest.mark.asyncio
    async def test_on_realtime_telemetry_metrics_with_empty_list(self, mocked_mixin):
        """Test that receiving empty metrics list is handled correctly."""
        message = RealtimeTelemetryMetricsMessage(
            service_id="records_manager", metrics=[]
        )

        await mocked_mixin._on_realtime_telemetry_metrics(message)

        # Should store empty list and still trigger hook
        assert mocked_mixin._telemetry_metrics == []
        mocked_mixin.run_hooks.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_access_with_lock(self, mocked_mixin):
        """Test that the lock protects concurrent access to telemetry metrics."""

        # Track lock acquisition order
        lock_acquired_order = []

        async def acquire_lock_and_update(metrics_value, delay):
            """Helper to simulate concurrent updates."""
            async with mocked_mixin._telemetry_metrics_lock:
                lock_acquired_order.append(metrics_value)
                await asyncio.sleep(delay)
                mocked_mixin._telemetry_metrics = [
                    MetricResult(
                        tag=f"metric_{metrics_value}",
                        header=f"Metric {metrics_value}",
                        unit="ms",
                        avg=float(metrics_value),
                    )
                ]

        # Start two concurrent operations
        await asyncio.gather(
            acquire_lock_and_update(1, 0.01), acquire_lock_and_update(2, 0.005)
        )

        # Both should have acquired the lock (order doesn't matter for this test)
        assert len(lock_acquired_order) == 2
        assert set(lock_acquired_order) == {1, 2}

        # Final value should be from the last completed operation
        assert len(mocked_mixin._telemetry_metrics) == 1

    @pytest.mark.asyncio
    async def test_multiple_metrics_handling(self, mocked_mixin):
        """Test handling of message with multiple metrics."""
        metrics = [
            MetricResult(
                tag=f"metric_{i}", header=f"Metric {i}", unit="ms", avg=float(i)
            )
            for i in range(10)
        ]

        message = RealtimeTelemetryMetricsMessage(
            service_id="records_manager", metrics=metrics
        )

        await mocked_mixin._on_realtime_telemetry_metrics(message)

        # All metrics should be stored
        assert len(mocked_mixin._telemetry_metrics) == 10
        assert mocked_mixin._telemetry_metrics == metrics
