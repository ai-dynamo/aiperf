# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.enums import MetricType
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult
from aiperf.metrics.metric_dicts import MetricArray, MetricResultsDict
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor
from tests.post_processors.conftest import create_metric_records_message


class TestMetricResultsProcessor:
    """Test cases for MetricResultsProcessor."""

    def test_initialization(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processor initialization sets up necessary data structures."""
        processor = MetricResultsProcessor(mock_user_config)

        assert isinstance(processor.derive_funcs, dict)
        assert isinstance(processor._results, dict)
        assert isinstance(processor._tags_to_types, dict)
        assert isinstance(processor._instances_map, dict)
        assert isinstance(processor._tags_to_aggregate_funcs, dict)

    @pytest.mark.asyncio
    async def test_process_result_record_metric(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing result for record metric accumulates values in the array."""
        processor = MetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        message = create_metric_records_message(
            x_request_id="test-1",
            results=[{"test_record": 42.0}],
        )
        await processor.process_result(message.to_data())

        assert "test_record" in processor._results
        assert isinstance(processor._results["test_record"], MetricArray)
        assert list(processor._results["test_record"].data) == [42.0]

        # New data should expand the array
        message2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=1_000_000_001,
            results=[{"test_record": 84.0}],
        )
        await processor.process_result(message2.to_data())
        assert list(processor._results["test_record"].data) == [42.0, 84.0]

    @pytest.mark.asyncio
    async def test_process_result_record_metric_list_values(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing record metric with list values extends the array."""
        processor = MetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        # Process list of values
        message = create_metric_records_message(
            x_request_id="test-1",
            results=[{"test_record": [10.0, 20.0, 30.0]}],
        )
        await processor.process_result(message.to_data())

        assert "test_record" in processor._results
        assert isinstance(processor._results["test_record"], MetricArray)
        assert list(processor._results["test_record"].data) == [10.0, 20.0, 30.0]

    @pytest.mark.asyncio
    async def test_process_result_aggregate_metric(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processing result for aggregate metric updates aggregated value."""
        processor = MetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._instances_map = {RequestCountMetric.tag: RequestCountMetric()}

        # Process two values and ensure they are accumulated
        message1 = create_metric_records_message(
            x_request_id="test-1",
            results=[{RequestCountMetric.tag: 5}],
        )
        await processor.process_result(message1.to_data())
        assert processor._results[RequestCountMetric.tag] == 5

        message2 = create_metric_records_message(
            x_request_id="test-2",
            request_start_ns=1_000_000_001,
            results=[{RequestCountMetric.tag: 3}],
        )
        await processor.process_result(message2.to_data())
        assert processor._results[RequestCountMetric.tag] == 8

    @pytest.mark.asyncio
    async def test_update_derived_metrics(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test derived metrics are computed correctly."""

        def mock_derive_func(results_dict: MetricResultsDict):
            return 100.0

        processor = MetricResultsProcessor(mock_user_config)
        processor.derive_funcs = {RequestThroughputMetric.tag: mock_derive_func}

        await processor.update_derived_metrics()

        assert processor._results[RequestThroughputMetric.tag] == 100.0

    @pytest.mark.asyncio
    async def test_update_derived_metrics_handles_no_metric_value(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test derived metrics gracefully handle NoMetricValue exceptions."""

        def failing_derive_func(results_dict: MetricResultsDict):
            raise NoMetricValue("Cannot derive value")

        processor = MetricResultsProcessor(mock_user_config)
        processor.derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}

        with patch.object(processor, "debug") as mock_debug:
            await processor.update_derived_metrics()

            assert RequestThroughputMetric.tag not in processor._results
            mock_debug.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_derived_metrics_handles_value_error_exception(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test derived metrics gracefully handle ValueError exceptions."""

        def failing_derive_func(results_dict: MetricResultsDict):
            raise ValueError("Calculation error")

        processor = MetricResultsProcessor(mock_user_config)
        processor.derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}

        with patch.object(processor, "warning") as mock_warning:
            await processor.update_derived_metrics()

            assert RequestThroughputMetric.tag not in processor._results
            mock_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test summarize returns list of MetricResult objects."""
        processor = MetricResultsProcessor(mock_user_config)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}
        processor._instances_map = {RequestLatencyMetric.tag: RequestLatencyMetric()}

        processor._results[RequestLatencyMetric.tag] = MetricArray()
        processor._results[RequestLatencyMetric.tag].append(42.0)

        results = await processor.summarize()

        assert len(results) == 1
        assert isinstance(results[0], MetricResult)
        assert results[0].tag == RequestLatencyMetric.tag

    @pytest.mark.asyncio
    async def test_full_metrics(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test full_metrics returns the complete results dict including derived metrics."""

        def mock_derive_func(results_dict: MetricResultsDict):
            return 200.0

        processor = MetricResultsProcessor(mock_user_config)
        processor.derive_funcs = {RequestThroughputMetric.tag: mock_derive_func}
        processor._results["base_metric"] = 100.0

        full_results = await processor.full_metrics()

        assert "base_metric" in full_results
        assert RequestThroughputMetric.tag in full_results
        assert full_results["base_metric"] == 100.0
        assert full_results[RequestThroughputMetric.tag] == 200.0

    def test_create_metric_result_from_scalar(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test creating MetricResult from scalar value."""
        processor = MetricResultsProcessor(mock_user_config)
        processor._instances_map = {RequestLatencyMetric.tag: RequestLatencyMetric()}

        result = processor._create_metric_result(RequestLatencyMetric.tag, 42)

        assert isinstance(result, MetricResult)
        assert result.tag == RequestLatencyMetric.tag
        assert result.header == RequestLatencyMetric.header
        assert result.unit == str(RequestLatencyMetric.unit)
        assert result.avg == 42
        assert result.count == 1

    def test_create_metric_result_from_metric_array(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test creating MetricResult from MetricArray."""
        processor = MetricResultsProcessor(mock_user_config)
        processor._instances_map = {RequestLatencyMetric.tag: RequestLatencyMetric()}
        metric_array = MetricArray()
        metric_array.extend([10.0, 20.0, 30.0])

        expected_result = MetricResult(
            tag=RequestLatencyMetric.tag,
            header=RequestLatencyMetric.header,
            unit=str(RequestLatencyMetric.unit),
            avg=20.0,
            count=3,
        )
        metric_array.to_result = Mock(return_value=expected_result)

        result = processor._create_metric_result(RequestLatencyMetric.tag, metric_array)

        assert result == expected_result
        metric_array.to_result.assert_called_once_with(
            RequestLatencyMetric.tag,
            RequestLatencyMetric.header,
            str(RequestLatencyMetric.unit),
        )

    def test_create_metric_result_invalid_type(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test creating MetricResult with invalid value type raises a ValueError."""
        processor = MetricResultsProcessor(mock_user_config)

        processor._instances_map = {RequestLatencyMetric.tag: RequestLatencyMetric()}
        with pytest.raises(ValueError, match="Unexpected values type"):
            processor._create_metric_result(
                RequestLatencyMetric.tag, {"invalid": "dict"}
            )
