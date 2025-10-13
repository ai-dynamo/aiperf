# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models.record_models import MetricRecordMetadata
from aiperf.common.types import MetricTagT
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric

# Constants
START_TIME = 1000000000


# Helper functions
def create_mock_records_manager(
    start_time_ns: int,
    expected_duration_sec: float | None,
    grace_period_sec: float = 0.0,
) -> MagicMock:
    """Create a mock RecordsManager instance for testing filtering logic."""
    instance = MagicMock()
    instance.expected_duration_sec = expected_duration_sec
    instance.start_time_ns = start_time_ns
    instance.user_config.loadgen.benchmark_grace_period = grace_period_sec
    instance.debug = MagicMock()
    return instance


def create_metric_record_data(
    request_start_ns: int,
    request_end_ns: int,
    metrics: dict[MetricTagT, int | float] | None = None,
) -> MetricRecordsData:
    """Create a MetricRecordsData object with sensible defaults for testing."""
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=0,
            conversation_id="test",
            turn_index=0,
            request_start_ns=request_start_ns,
            request_end_ns=request_end_ns,
            worker_id="worker-1",
            record_processor_id="processor-1",
            benchmark_phase=CreditPhase.PROFILING,
        ),
        metrics=metrics or {},
    )


class TestRecordsManagerFiltering:
    """Test the records manager's filtering logic."""

    def test_should_include_request_by_duration_no_duration_benchmark(self):
        """Test that request-count benchmarks always include all requests."""
        from aiperf.records.records_manager import RecordsManager

        instance = create_mock_records_manager(
            start_time_ns=0,
            expected_duration_sec=None,
        )

        record_data = create_metric_record_data(
            request_start_ns=999999999999999,
            request_end_ns=999999999999999,
            metrics={
                MinRequestTimestampMetric.tag: 999999999999999,
                RequestLatencyMetric.tag: 999999999999999,
            },
        )

        result = RecordsManager._should_include_request_by_duration(
            instance, record_data
        )
        assert result is True

    def test_should_include_request_within_duration_no_grace_period(self):
        """Test filtering with zero grace period - only duration matters."""
        from aiperf.records.records_manager import RecordsManager

        instance = create_mock_records_manager(
            start_time_ns=START_TIME,
            expected_duration_sec=2.0,
            grace_period_sec=0.0,
        )

        # Request that completes exactly at duration end should be included
        request_start = START_TIME + int(1.5 * NANOS_PER_SECOND)
        request_latency = int(0.5 * NANOS_PER_SECOND)
        record_at_duration = create_metric_record_data(
            request_start_ns=request_start,
            request_end_ns=request_start + request_latency,  # Completes exactly at 2.0s
            metrics={
                MinRequestTimestampMetric.tag: request_start,
                RequestLatencyMetric.tag: request_latency,
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_at_duration
            )
            is True
        )

        # Request that completes after duration should be excluded
        request_start2 = START_TIME + int(1.5 * NANOS_PER_SECOND)
        request_latency2 = int(0.6 * NANOS_PER_SECOND)
        record_after_duration = create_metric_record_data(
            request_start_ns=request_start2,
            request_end_ns=request_start2 + request_latency2,  # Completes at 2.1s
            metrics={
                MinRequestTimestampMetric.tag: request_start2,
                RequestLatencyMetric.tag: request_latency2,
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_after_duration
            )
            is False
        )

    def test_should_include_request_within_grace_period(self):
        """Test filtering with grace period - responses within grace period are included."""
        from aiperf.records.records_manager import RecordsManager

        instance = create_mock_records_manager(
            start_time_ns=START_TIME,
            expected_duration_sec=2.0,
            grace_period_sec=1.0,
        )

        # Request that completes within grace period should be included
        request_start_within = START_TIME + int(1.5 * NANOS_PER_SECOND)
        request_latency_within = int(1.4 * NANOS_PER_SECOND)
        record_within_grace = create_metric_record_data(
            request_start_ns=request_start_within,
            request_end_ns=request_start_within
            + request_latency_within,  # Completes at 2.9s
            metrics={
                MinRequestTimestampMetric.tag: request_start_within,
                RequestLatencyMetric.tag: request_latency_within,
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_within_grace
            )
            is True
        )

        # Request that completes after grace period should be excluded
        request_start_after = START_TIME + int(1.5 * NANOS_PER_SECOND)
        request_latency_after = int(1.6 * NANOS_PER_SECOND)
        record_after_grace = create_metric_record_data(
            request_start_ns=request_start_after,
            request_end_ns=request_start_after
            + request_latency_after,  # Completes at 3.1s
            metrics={
                MinRequestTimestampMetric.tag: request_start_after,
                RequestLatencyMetric.tag: request_latency_after,
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_after_grace
            )
            is False
        )

    def test_should_include_request_missing_metrics(self):
        """Test filtering behavior when required metrics are missing."""
        from aiperf.records.records_manager import RecordsManager

        instance = create_mock_records_manager(
            start_time_ns=START_TIME,
            expected_duration_sec=2.0,
            grace_period_sec=1.0,
        )

        # Request that ends after grace period should be excluded
        record_missing_timestamp = create_metric_record_data(
            request_start_ns=START_TIME,
            request_end_ns=START_TIME + int(5.0 * NANOS_PER_SECOND),  # After grace
            metrics={
                RequestLatencyMetric.tag: int(5.0 * NANOS_PER_SECOND)  # Only latency
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_missing_timestamp
            )
            is False
        )

        # Request that ends within grace period should be included
        record_missing_latency = create_metric_record_data(
            request_start_ns=START_TIME + int(1.0 * NANOS_PER_SECOND),
            request_end_ns=START_TIME + int(2.0 * NANOS_PER_SECOND),  # Within grace
            metrics={
                MinRequestTimestampMetric.tag: START_TIME + int(1.0 * NANOS_PER_SECOND)
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_missing_latency
            )
            is True
        )

        # Request with no metrics should be included if it ends within grace period
        record_no_metrics = create_metric_record_data(
            request_start_ns=START_TIME,
            request_end_ns=START_TIME + int(1.0 * NANOS_PER_SECOND),  # Within grace
            metrics={},
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_no_metrics
            )
            is True
        )

    @pytest.mark.parametrize("grace_period", [0.0, 0.5, 1.0, 5.0, 30.0])
    def test_should_include_request_various_grace_periods(self, grace_period: float):
        """Test filtering logic with various grace period values."""
        from aiperf.records.records_manager import RecordsManager

        instance = create_mock_records_manager(
            start_time_ns=START_TIME,
            expected_duration_sec=2.0,
            grace_period_sec=grace_period,
        )

        # Request that completes exactly at duration + grace_period boundary
        request_start_at = START_TIME + int(1.0 * NANOS_PER_SECOND)
        request_latency_at = int((1.0 + grace_period) * NANOS_PER_SECOND)
        record_at_boundary = create_metric_record_data(
            request_start_ns=request_start_at,
            request_end_ns=request_start_at + request_latency_at,
            metrics={
                MinRequestTimestampMetric.tag: request_start_at,
                RequestLatencyMetric.tag: request_latency_at,
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_at_boundary
            )
            is True
        )

        # Request that completes just after duration + grace_period should be excluded
        request_start_after = START_TIME + int(1.0 * NANOS_PER_SECOND)
        request_latency_after = int((1.0 + grace_period + 0.1) * NANOS_PER_SECOND)
        record_after_boundary = create_metric_record_data(
            request_start_ns=request_start_after,
            request_end_ns=request_start_after + request_latency_after,
            metrics={
                MinRequestTimestampMetric.tag: request_start_after,
                RequestLatencyMetric.tag: request_latency_after,
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_after_boundary
            )
            is False
        )

    def test_should_include_request_multiple_results_in_request(self):
        """Test filtering with multiple result dictionaries for a single request (all-or-nothing)."""
        from aiperf.records.records_manager import RecordsManager

        instance = create_mock_records_manager(
            start_time_ns=START_TIME,
            expected_duration_sec=2.0,
            grace_period_sec=1.0,
        )

        # Request where the latest response completes within grace period - should include
        request_start_within = START_TIME + int(1.5 * NANOS_PER_SECOND)
        record_all_within = create_metric_record_data(
            request_start_ns=request_start_within,
            request_end_ns=START_TIME
            + int(2.9 * NANOS_PER_SECOND),  # Latest completion time
            metrics={
                MinRequestTimestampMetric.tag: request_start_within,
                RequestLatencyMetric.tag: int(1.0 * NANOS_PER_SECOND),
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_all_within
            )
            is True
        )

        # Request where one response completes after grace period - should exclude entire request
        request_start_after = START_TIME + int(1.0 * NANOS_PER_SECOND)
        record_one_after = create_metric_record_data(
            request_start_ns=request_start_after,
            request_end_ns=START_TIME
            + int(3.5 * NANOS_PER_SECOND),  # Latest completion time (after grace)
            metrics={
                MinRequestTimestampMetric.tag: request_start_after,
                RequestLatencyMetric.tag: int(1.0 * NANOS_PER_SECOND),
            },
        )
        assert (
            RecordsManager._should_include_request_by_duration(
                instance, record_one_after
            )
            is False
        )
