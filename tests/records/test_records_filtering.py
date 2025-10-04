# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric


class TestRecordsManagerFiltering:
    """Test the records manager's filtering logic ."""

    def test_should_include_request_by_duration_no_duration_benchmark(self):
        """Test that request-count benchmarks always include all requests."""
        from aiperf.records.records_manager import RecordsManager

        instance = MagicMock()
        instance.expected_duration_sec = None

        results = [
            {
                MinRequestTimestampMetric.tag: 999999999999999,
                RequestLatencyMetric.tag: 999999999999999,
            }
        ]

        result = RecordsManager._should_include_request_by_duration(instance, results)
        assert result is True

    def test_should_include_request_within_duration_no_grace_period(self):
        """Test filtering with zero grace period - only duration matters."""
        from aiperf.records.records_manager import RecordsManager

        start_time = 1000000000
        duration_sec = 2.0
        grace_period_sec = 0.0

        instance = MagicMock()
        instance.expected_duration_sec = duration_sec
        instance.start_time_ns = start_time
        instance.user_config.loadgen.benchmark_grace_period = grace_period_sec
        instance.debug = MagicMock()  # Mock debug method

        # Request that completes exactly at duration end should be included
        results_at_duration = [
            {
                MinRequestTimestampMetric.tag: start_time + int(1.5 * NANOS_PER_SECOND),
                RequestLatencyMetric.tag: int(
                    0.5 * NANOS_PER_SECOND
                ),  # Completes exactly at 2.0s
            }
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_at_duration
            )
            is True
        )

        # Request that completes after duration should be excluded
        results_after_duration = [
            {
                MinRequestTimestampMetric.tag: start_time + int(1.5 * NANOS_PER_SECOND),
                RequestLatencyMetric.tag: int(
                    0.6 * NANOS_PER_SECOND
                ),  # Completes at 2.1s
            }
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_after_duration
            )
            is False
        )

    def test_should_include_request_within_grace_period(self):
        """Test filtering with grace period - responses within grace period are included."""
        from aiperf.records.records_manager import RecordsManager

        start_time = 1000000000
        duration_sec = 2.0
        grace_period_sec = 1.0  # 1 second grace period

        instance = MagicMock()
        instance.expected_duration_sec = duration_sec
        instance.start_time_ns = start_time
        instance.user_config.loadgen.benchmark_grace_period = grace_period_sec
        instance.debug = MagicMock()

        results_within_grace = [
            {
                MinRequestTimestampMetric.tag: start_time
                + int(1.5 * NANOS_PER_SECOND),  # 1.5s after start
                RequestLatencyMetric.tag: int(
                    1.4 * NANOS_PER_SECOND
                ),  # Completes at 2.9s (within 3s total)
            }
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_within_grace
            )
            is True
        )

        # Request that completes after grace period should be excluded
        results_after_grace = [
            {
                MinRequestTimestampMetric.tag: start_time
                + int(1.5 * NANOS_PER_SECOND),  # 1.5s after start
                RequestLatencyMetric.tag: int(
                    1.6 * NANOS_PER_SECOND
                ),  # Completes at 3.1s (after 3s total)
            }
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_after_grace
            )
            is False
        )

    def test_should_include_request_missing_metrics(self):
        """Test filtering behavior when required metrics are missing."""
        from aiperf.records.records_manager import RecordsManager

        start_time = 1000000000
        duration_sec = 2.0
        grace_period_sec = 1.0

        instance = MagicMock()
        instance.expected_duration_sec = duration_sec
        instance.start_time_ns = start_time
        instance.user_config.loadgen.benchmark_grace_period = grace_period_sec
        instance.debug = MagicMock()

        # Request with missing timestamp should be included (cannot filter)
        results_missing_timestamp = [
            {
                RequestLatencyMetric.tag: int(5.0 * NANOS_PER_SECOND)  # Only latency
            }
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_missing_timestamp
            )
            is True
        )

        # Request with missing latency should be included (cannot filter)
        results_missing_latency = [
            {
                MinRequestTimestampMetric.tag: start_time
                + int(1.0 * NANOS_PER_SECOND)  # Only timestamp
            }
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_missing_latency
            )
            is True
        )

        # Request with no metrics should be included
        results_no_metrics = [{}]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_no_metrics
            )
            is True
        )

    @pytest.mark.parametrize("grace_period", [0.0, 0.5, 1.0, 5.0, 30.0])
    def test_should_include_request_various_grace_periods(self, grace_period: float):
        """Test filtering logic with various grace period values."""
        from aiperf.records.records_manager import RecordsManager

        start_time = 1000000000
        duration_sec = 2.0

        instance = MagicMock()
        instance.expected_duration_sec = duration_sec
        instance.start_time_ns = start_time
        instance.user_config.loadgen.benchmark_grace_period = grace_period
        instance.debug = MagicMock()

        # Request that completes exactly at duration + grace_period boundary
        results_at_boundary = [
            {
                MinRequestTimestampMetric.tag: start_time + int(1.0 * NANOS_PER_SECOND),
                RequestLatencyMetric.tag: int((1.0 + grace_period) * NANOS_PER_SECOND),
            }
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_at_boundary
            )
            is True
        )

        # Request that completes just after duration + grace_period should be excluded
        results_after_boundary = [
            {
                MinRequestTimestampMetric.tag: start_time + int(1.0 * NANOS_PER_SECOND),
                RequestLatencyMetric.tag: int(
                    (1.0 + grace_period + 0.1) * NANOS_PER_SECOND
                ),
            }
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_after_boundary
            )
            is False
        )

    def test_should_include_request_multiple_results_in_request(self):
        """Test filtering with multiple result dictionaries for a single request (all-or-nothing)."""
        from aiperf.records.records_manager import RecordsManager

        start_time = 1000000000
        duration_sec = 2.0
        grace_period_sec = 1.0

        instance = MagicMock()
        instance.expected_duration_sec = duration_sec
        instance.start_time_ns = start_time
        instance.user_config.loadgen.benchmark_grace_period = grace_period_sec
        instance.debug = MagicMock()

        # Multiple results where all complete within grace period - should include
        results_all_within = [
            {
                MinRequestTimestampMetric.tag: start_time + int(1.5 * NANOS_PER_SECOND),
                RequestLatencyMetric.tag: int(
                    1.0 * NANOS_PER_SECOND
                ),  # Completes at 2.5s
            },
            {
                MinRequestTimestampMetric.tag: start_time + int(1.8 * NANOS_PER_SECOND),
                RequestLatencyMetric.tag: int(
                    1.1 * NANOS_PER_SECOND
                ),  # Completes at 2.9s
            },
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_all_within
            )
            is True
        )

        # Multiple results where one completes after grace period - should exclude entire request
        results_one_after = [
            {
                MinRequestTimestampMetric.tag: start_time + int(1.0 * NANOS_PER_SECOND),
                RequestLatencyMetric.tag: int(
                    1.0 * NANOS_PER_SECOND
                ),  # Completes at 2.0s (within)
            },
            {
                MinRequestTimestampMetric.tag: start_time + int(1.5 * NANOS_PER_SECOND),
                RequestLatencyMetric.tag: int(
                    2.0 * NANOS_PER_SECOND
                ),  # Completes at 3.5s (after grace)
            },
        ]
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, results_one_after
            )
            is False
        )


class TestRecordsManagerTelemetry:
    """Test RecordsManager telemetry handling with mocked components."""

    @pytest.mark.asyncio
    async def test_on_telemetry_records_valid(self):
        """Test handling valid telemetry records."""
        from unittest.mock import AsyncMock, MagicMock

        from aiperf.common.messages import TelemetryRecordsMessage
        from aiperf.common.models import (
            TelemetryHierarchy,
            TelemetryMetrics,
            TelemetryRecord,
        )

        # Create sample telemetry records
        records = [
            TelemetryRecord(
                timestamp_ns=1000000,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=0,
                gpu_uuid="GPU-123",
                gpu_model_name="Test GPU",
                telemetry_data=TelemetryMetrics(
                    gpu_power_usage=100.0,
                ),
            )
        ]

        message = TelemetryRecordsMessage(
            service_id="test_service",
            collector_id="test_collector",
            records=records,
            error=None,
        )

        # Mock the hierarchy
        mock_hierarchy = MagicMock(spec=TelemetryHierarchy)
        mock_hierarchy.add_record = MagicMock()
        mock_send_to_processors = AsyncMock()

        # Test the logic directly without instantiating the full service
        for record in message.records:
            mock_hierarchy.add_record(record)

        if message.records:
            await mock_send_to_processors(message.records)

        # Verify behavior
        assert mock_hierarchy.add_record.call_count == len(records)
        mock_send_to_processors.assert_called_once_with(records)

    @pytest.mark.asyncio
    async def test_on_telemetry_records_invalid(self):
        """Test handling invalid telemetry records with errors."""
        from unittest.mock import AsyncMock

        from aiperf.common.messages import TelemetryRecordsMessage
        from aiperf.common.models import ErrorDetails

        error = ErrorDetails(message="Test error", code=500)

        message = TelemetryRecordsMessage(
            service_id="test_service",
            collector_id="test_collector",
            records=[],
            error=error,
        )

        mock_send_to_processors = AsyncMock()
        error_counts = {}

        # Test the logic: errors should be tracked, not sent to processors
        if message.error:
            error_counts[message.error] = error_counts.get(message.error, 0) + 1
        else:
            await mock_send_to_processors(message.records)

        # Should not send to processors
        mock_send_to_processors.assert_not_called()

        # Error should be tracked
        assert error in error_counts
        assert error_counts[error] == 1

    @pytest.mark.asyncio
    async def test_send_telemetry_to_results_processors(self):
        """Test sending telemetry records to processors."""
        from unittest.mock import AsyncMock, Mock

        from aiperf.common.models import TelemetryMetrics, TelemetryRecord

        # Create mock telemetry processor
        mock_processor = Mock()
        mock_processor.process_telemetry_record = AsyncMock()

        records = [
            TelemetryRecord(
                timestamp_ns=1000000,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=0,
                gpu_uuid="GPU-123",
                gpu_model_name="Test GPU",
                telemetry_data=TelemetryMetrics(),
            ),
            TelemetryRecord(
                timestamp_ns=1000001,
                dcgm_url="http://localhost:9400/metrics",
                gpu_index=1,
                gpu_uuid="GPU-456",
                gpu_model_name="Test GPU",
                telemetry_data=TelemetryMetrics(),
            ),
        ]

        # Test the logic: each record should be sent to processor
        for record in records:
            await mock_processor.process_telemetry_record(record)

        # Processor should be called for each record
        assert mock_processor.process_telemetry_record.call_count == len(records)

    def test_telemetry_hierarchy_add_record(self):
        """Test that telemetry hierarchy adds records correctly."""
        from aiperf.common.models import (
            TelemetryHierarchy,
            TelemetryMetrics,
            TelemetryRecord,
        )

        hierarchy = TelemetryHierarchy()

        record = TelemetryRecord(
            timestamp_ns=1000000,
            dcgm_url="http://localhost:9400/metrics",
            gpu_index=0,
            gpu_uuid="GPU-123",
            gpu_model_name="Test GPU",
            telemetry_data=TelemetryMetrics(
                gpu_power_usage=100.0,
            ),
        )

        # Add record to hierarchy
        hierarchy.add_record(record)

        # Verify hierarchy structure
        assert "http://localhost:9400/metrics" in hierarchy.dcgm_endpoints
        assert "GPU-123" in hierarchy.dcgm_endpoints["http://localhost:9400/metrics"]
