# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import OutputConfig, UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import MetricType
from aiperf.common.exceptions import NoMetricValue, PostProcessorDisabled
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_dicts import MetricArray, MetricResultsDict
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from aiperf.post_processors.timeslice_metric_results_processor import (
    TimesliceMetricResultsProcessor,
)
from tests.unit.post_processors.conftest import create_metric_records_message


class TestTimesliceMetricResultsProcessor:
    """Test cases for TimesliceMetricResultsProcessor."""

    def test_initialization_without_slice_duration_raises_exception(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that processor initialization fails when slice_duration is not set."""
        # Ensure slice_duration is None
        mock_user_config.output.slice_duration = None

        with pytest.raises(PostProcessorDisabled, match="requires slice_duration"):
            TimesliceMetricResultsProcessor(mock_user_config)

    def test_initialization_with_slice_duration(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processor initialization sets up timeslice-specific data structures."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)

        assert hasattr(processor, "_timeslice_instances_maps")
        assert hasattr(processor, "_timeslice_results")
        assert hasattr(processor, "_slice_duration_ns")
        assert processor._slice_duration_ns == 1.0 * NANOS_PER_SECOND

    @pytest.mark.asyncio
    async def test_get_instances_map_requires_request_start_ns(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that get_instances_map raises ValueError when request_start_ns is None."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)

        with pytest.raises(ValueError, match="must be passed a request_start_ns"):
            await processor.get_instances_map(None)

    @pytest.mark.asyncio
    async def test_get_results_requires_request_start_ns(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that get_results raises ValueError when request_start_ns is None."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)

        with pytest.raises(ValueError, match="must be passed a request_start_ns"):
            await processor.get_results(None)

    @pytest.mark.asyncio
    async def test_process_result_separates_by_timeslice(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that metrics are separated into different timeslices based on timestamp."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)  # 1 second
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        # Process request in first timeslice (0.5 seconds)
        message1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            results=[{"test_record": 42.0}],
        )
        await processor.process_result(message1.to_data())

        # Process request in second timeslice (1.5 seconds)
        message2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            results=[{"test_record": 84.0}],
        )
        await processor.process_result(message2.to_data())

        # Verify results are in different timeslices
        assert 0 in processor._timeslice_results
        assert 1 in processor._timeslice_results
        assert list(processor._timeslice_results[0]["test_record"].data) == [42.0]
        assert list(processor._timeslice_results[1]["test_record"].data) == [84.0]

    @pytest.mark.asyncio
    async def test_process_result_accumulates_in_same_timeslice(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that metrics in the same timeslice are accumulated together."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)  # 1 second
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        # Process two requests in same timeslice (both in first second)
        message1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"test_record": 10.0}],
        )
        await processor.process_result(message1.to_data())

        message2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"test_record": 20.0}],
        )
        await processor.process_result(message2.to_data())

        # Verify results are accumulated in same timeslice
        assert 0 in processor._timeslice_results
        assert list(processor._timeslice_results[0]["test_record"].data) == [10.0, 20.0]

    @pytest.mark.asyncio
    async def test_process_result_aggregate_metric_per_timeslice(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that aggregate metrics work correctly per timeslice."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)  # 1 second
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}

        # First timeslice - two requests
        message1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 5}],
        )
        await processor.process_result(message1.to_data())

        message2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 3}],
        )
        await processor.process_result(message2.to_data())

        # Second timeslice - one request
        message3 = create_metric_records_message(
            x_request_id="test-3",
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 7}],
        )
        await processor.process_result(message3.to_data())

        # Verify aggregate counts are separate per timeslice
        assert processor._timeslice_results[0][RequestCountMetric.tag] == 8
        assert processor._timeslice_results[1][RequestCountMetric.tag] == 7

    @pytest.mark.asyncio
    async def test_timeslice_boundary_conditions(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test behavior at timeslice boundaries."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)  # 1 second
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        # Request at 0.999s (should be in timeslice 0)
        message1 = create_metric_records_message(
            x_request_id="test-1",
            request_start_ns=int(0.999 * NANOS_PER_SECOND),
            results=[{"test_record": 1.0}],
        )
        await processor.process_result(message1.to_data())

        # Request at 1.0s (should be in timeslice 1)
        message2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=int(1.0 * NANOS_PER_SECOND),
            results=[{"test_record": 2.0}],
        )
        await processor.process_result(message2.to_data())

        # Request at 1.001s (should be in timeslice 1)
        message3 = create_metric_records_message(
            x_request_id="test-3",
            request_start_ns=int(1.001 * NANOS_PER_SECOND),
            results=[{"test_record": 3.0}],
        )
        await processor.process_result(message3.to_data())

        # Verify proper separation at boundaries
        assert list(processor._timeslice_results[0]["test_record"].data) == [1.0]
        assert list(processor._timeslice_results[1]["test_record"].data) == [2.0, 3.0]

    @pytest.mark.asyncio
    async def test_update_derived_metrics_per_timeslice(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that derived metrics are computed per timeslice."""

        def mock_derive_func(results_dict: MetricResultsDict):
            # Simple derive func that returns a constant based on existence of data
            return 100.0 if results_dict else 0.0

        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor.derive_funcs = {RequestThroughputMetric.tag: mock_derive_func}

        # Set up some dummy results in different timeslices
        processor._timeslice_results[0]["base_metric"] = 42
        processor._timeslice_results[1]["base_metric"] = 84

        await processor.update_derived_metrics()

        # Verify derived metrics are computed for each timeslice
        assert processor._timeslice_results[0][RequestThroughputMetric.tag] == 100.0
        assert processor._timeslice_results[1][RequestThroughputMetric.tag] == 100.0

    @pytest.mark.asyncio
    async def test_update_derived_metrics_handles_no_metric_value(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that NoMetricValue exceptions are caught and logged gracefully per timeslice."""

        def failing_derive_func(results_dict: MetricResultsDict):
            raise NoMetricValue("Cannot derive value")

        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor.derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}
        processor._timeslice_results[0]["base_metric"] = 42

        with patch.object(processor, "debug") as mock_debug:
            # NoMetricValue should be caught and logged, not raised
            await processor.update_derived_metrics()

            # Verify no derived metric was added (exception was caught)
            assert RequestThroughputMetric.tag not in processor._timeslice_results[0]
            # Verify the exception was logged via debug
            mock_debug.assert_called_once()
            assert "No metric value" in str(mock_debug.call_args)

    @pytest.mark.asyncio
    async def test_update_derived_metrics_handles_value_error(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that derived metrics handle ValueError exceptions gracefully."""

        def failing_derive_func(results_dict: MetricResultsDict):
            raise ValueError("Calculation error")

        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor.derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}
        processor._timeslice_results[0]["base_metric"] = 42

        with patch.object(processor, "warning") as mock_warning:
            await processor.update_derived_metrics()

            # Verify no derived metric was added
            assert RequestThroughputMetric.tag not in processor._timeslice_results[0]
            mock_warning.assert_called()

    @pytest.mark.asyncio
    async def test_summarize_returns_dict_of_timeslices(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test summarize returns dict mapping timeslice indices to metric results."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}

        # Set up results in multiple timeslices
        # Values are in nanoseconds (internal unit), summarize() converts to ms (display unit)
        processor._timeslice_results[0][RequestLatencyMetric.tag] = MetricArray()
        processor._timeslice_results[0][RequestLatencyMetric.tag].append(42_000_000.0)

        processor._timeslice_results[1][RequestLatencyMetric.tag] = MetricArray()
        processor._timeslice_results[1][RequestLatencyMetric.tag].append(84_000_000.0)

        # Set up the instances map (used by _create_metric_result)
        # The parent class _create_metric_result uses self._instances_map
        processor._instances_map = {RequestLatencyMetric.tag: RequestLatencyMetric()}

        results = await processor.summarize()

        # Verify structure: dict of timeslice_index -> list[MetricResult]
        assert isinstance(results, dict)
        assert 0 in results
        assert 1 in results
        assert isinstance(results[0], list)
        assert isinstance(results[1], list)
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert isinstance(results[0][0], MetricResult)
        assert isinstance(results[1][0], MetricResult)
        assert results[0][0].tag == RequestLatencyMetric.tag
        assert results[1][0].tag == RequestLatencyMetric.tag
        # Verify the actual values are in display units (ms)
        assert results[0][0].avg == 42.0
        assert results[1][0].avg == 84.0
        assert results[0][0].unit == "ms"
        assert results[1][0].unit == "ms"

    @pytest.mark.asyncio
    async def test_summarize_with_empty_timeslices(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test summarize handles empty timeslices correctly."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)

        # No data processed
        results = await processor.summarize()

        # Should return empty dict
        assert isinstance(results, dict)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_multiple_timeslices_with_different_slice_duration(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that a different slice_duration value works correctly."""
        # Test with 500ms slices (different from default 1000ms)
        mock_user_config.output = OutputConfig(slice_duration=0.5)
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        # Process requests across multiple 0.5s slices
        for i in range(4):
            message = create_metric_records_message(
                x_request_id=f"test-{i}",
                request_start_ns=int((i * 0.5 + 0.25) * NANOS_PER_SECOND),
                results=[{"test_record": float(i)}],
            )
            await processor.process_result(message.to_data())

        # Should have 4 different timeslices (0, 1, 2, 3)
        assert len(processor._timeslice_results) == 4
        for i in range(4):
            assert i in processor._timeslice_results
            assert list(processor._timeslice_results[i]["test_record"].data) == [
                float(i)
            ]

    @pytest.mark.asyncio
    async def test_timeslice_instances_map_creates_separate_instances(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test that each timeslice gets its own metric instances."""
        mock_user_config.output = OutputConfig(slice_duration=1.0)
        processor = TimesliceMetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}

        # Get instances for two different timestamps in different timeslices
        request_start_ns_1 = int(0.5 * NANOS_PER_SECOND)
        request_start_ns_2 = int(1.5 * NANOS_PER_SECOND)

        instances_map_0 = await processor.get_instances_map(request_start_ns_1)
        instances_map_1 = await processor.get_instances_map(request_start_ns_2)

        # Verify they are different instances
        assert instances_map_0 is not instances_map_1
        assert (
            instances_map_0[RequestCountMetric.tag]
            is not instances_map_1[RequestCountMetric.tag]
        )
