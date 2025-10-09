# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.base_record_metric import BaseRecordMetric
from aiperf.metrics.metric_dicts import MetricRecordDict
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from tests.conftest import (
    DEFAULT_LAST_RESPONSE_NS,
    DEFAULT_START_TIME_NS,
)
from tests.post_processors.conftest import (
    setup_mock_registry_sequences,
)


class FailingMetricNoValue(BaseRecordMetric[int]):
    """Test metric that raises NoMetricValue exception."""

    tag = "failing_metric_no_value"

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        raise NoMetricValue("No value available")


class FailingMetricValueError(BaseRecordMetric[int]):
    """Test metric that raises ValueError exception."""

    tag = "failing_metric_value_error"

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        raise ValueError("Something went wrong")


class DoubleLatencyTestMetric(BaseRecordMetric[int]):
    """Test metric that depends on other metrics' results."""

    tag = "double_latency_test_metric"

    def __init__(self):
        super().__init__()
        self.base_metric_tag = RequestLatencyMetric.tag

    def _parse_record(
        self, record: ParsedResponseRecord, record_metrics: MetricRecordDict
    ) -> int:
        # Use the base metric result and multiply by 2
        base_value = record_metrics.get(RequestLatencyMetric.tag, 0)
        return base_value * 2  # type: ignore


class TestMetricRecordProcessor:
    """Test cases for MetricRecordProcessor."""

    def test_initialization_caches_parse_functions(
        self, mock_metric_registry: Mock, mock_user_config: UserConfig
    ) -> None:
        """Test processor initialization caches parse functions for valid and error metrics."""
        setup_mock_registry_sequences(
            mock_metric_registry, [RequestLatencyMetric], [ErrorRequestCountMetric]
        )

        processor = MetricRecordProcessor(mock_user_config)

        # Verify valid parse and error parse functions were cached
        assert len(processor.valid_parse_funcs) == 1
        assert processor.valid_parse_funcs[0][0] == RequestLatencyMetric.tag
        assert (
            processor.valid_parse_funcs[0][1]
            == mock_metric_registry.get_instance(RequestLatencyMetric.tag).parse_record
        )

        assert len(processor.error_parse_funcs) == 1
        assert processor.error_parse_funcs[0][0] == ErrorRequestCountMetric.tag
        assert (
            processor.error_parse_funcs[0][1]
            == mock_metric_registry.get_instance(
                ErrorRequestCountMetric.tag
            ).parse_record
        )

        assert mock_metric_registry.tags_applicable_to.call_count == 2
        assert mock_metric_registry.create_dependency_order_for.call_count == 2

    @pytest.mark.asyncio
    async def test_process_valid_record(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test processing a valid record uses valid parse functions."""
        setup_mock_registry_sequences(mock_metric_registry, [RequestLatencyMetric], [])

        processor = MetricRecordProcessor(mock_user_config)
        result = await processor.process_record(sample_parsed_record)

        assert isinstance(result, MetricRecordDict)
        assert RequestLatencyMetric.tag in result

        expected_latency = DEFAULT_LAST_RESPONSE_NS - DEFAULT_START_TIME_NS
        assert result[RequestLatencyMetric.tag] == expected_latency

    @pytest.mark.asyncio
    async def test_process_error_record(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        error_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test processing an error record uses error parse functions."""
        setup_mock_registry_sequences(
            mock_metric_registry, [], [ErrorRequestCountMetric]
        )

        processor = MetricRecordProcessor(mock_user_config)
        result = await processor.process_record(error_parsed_record)

        assert isinstance(result, MetricRecordDict)
        assert result[ErrorRequestCountMetric.tag] == 1

    @pytest.mark.asyncio
    async def test_process_record_multiple_metrics(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test processing record with multiple metrics in sequence."""
        setup_mock_registry_sequences(
            mock_metric_registry, [RequestLatencyMetric, RequestCountMetric], []
        )

        processor = MetricRecordProcessor(mock_user_config)
        result = await processor.process_record(sample_parsed_record)

        assert len(result) == 2
        assert RequestLatencyMetric.tag in result
        assert RequestCountMetric.tag in result
        assert result[RequestCountMetric.tag] == 1

        expected_latency = DEFAULT_LAST_RESPONSE_NS - DEFAULT_START_TIME_NS
        assert result[RequestLatencyMetric.tag] == expected_latency

    @pytest.mark.asyncio
    async def test_process_record_handles_no_metric_value_exception(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test graceful handling of NoMetricValue exception by logging as a debug message."""
        setup_mock_registry_sequences(mock_metric_registry, [FailingMetricNoValue], [])

        processor = MetricRecordProcessor(mock_user_config)

        with patch.object(processor, "debug") as mock_debug:
            result = await processor.process_record(sample_parsed_record)

            assert isinstance(result, MetricRecordDict)
            assert FailingMetricNoValue.tag not in result
            mock_debug.assert_called_once()
            assert f"No metric value for metric '{FailingMetricNoValue.tag}'" in str(
                mock_debug.call_args
            )

    @pytest.mark.asyncio
    async def test_process_record_handles_value_error_exception(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test graceful handling of ValueError exceptions during metric parsing by logging as a warning message."""
        setup_mock_registry_sequences(
            mock_metric_registry, [FailingMetricValueError], []
        )

        processor = MetricRecordProcessor(mock_user_config)

        with patch.object(processor, "warning") as mock_warning:
            result = await processor.process_record(sample_parsed_record)

            assert isinstance(result, MetricRecordDict)
            assert FailingMetricValueError.tag not in result
            mock_warning.assert_called_once()
            assert (
                f"Error parsing record for metric '{FailingMetricValueError.tag}'"
                in str(mock_warning.call_args)
            )

    @pytest.mark.asyncio
    async def test_process_record_mixed_success_failure(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test processing record with mix of successful and failing metrics."""
        setup_mock_registry_sequences(
            mock_metric_registry, [RequestLatencyMetric], [FailingMetricNoValue]
        )

        processor = MetricRecordProcessor(mock_user_config)

        with patch.object(processor, "debug"):
            result = await processor.process_record(sample_parsed_record)

            assert len(result) == 1
            assert RequestLatencyMetric.tag in result
            assert FailingMetricNoValue.tag not in result

            expected_latency = DEFAULT_LAST_RESPONSE_NS - DEFAULT_START_TIME_NS
            assert result[RequestLatencyMetric.tag] == expected_latency

    @pytest.mark.asyncio
    async def test_process_record_with_dependencies(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test processing record with dependent metrics executes sequentially."""
        setup_mock_registry_sequences(
            mock_metric_registry, [RequestLatencyMetric, DoubleLatencyTestMetric], []
        )

        processor = MetricRecordProcessor(mock_user_config)
        result = await processor.process_record(sample_parsed_record)

        assert len(result) == 2
        assert RequestLatencyMetric.tag in result
        assert DoubleLatencyTestMetric.tag in result

        # Dependent metric value is 2x the base metric value (see DoubleLatencyTestMetric)
        assert (
            result[DoubleLatencyTestMetric.tag] == result[RequestLatencyMetric.tag] * 2
        )

    @pytest.mark.asyncio
    async def test_process_record_empty_metrics(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        sample_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test processing record when no metrics are configured."""
        processor = MetricRecordProcessor(mock_user_config)
        result = await processor.process_record(sample_parsed_record)

        assert isinstance(result, MetricRecordDict)
        assert len(result) == 0
