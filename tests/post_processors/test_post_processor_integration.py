# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration unit tests for post-processing pipeline."""

from unittest.mock import Mock

import pytest

from aiperf.common.config import UserConfig
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.metric_dicts import MetricArray
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from aiperf.post_processors.metric_results_processor import MetricResultsProcessor
from tests.post_processors.conftest import (
    create_metric_records_message,
    create_results_processor_with_metrics,
    setup_mock_registry_sequences,
)

TEST_LATENCY_VALUES = [100.0, 150.0, 200.0]
TEST_REQUEST_COUNT = 100
TEST_DURATION_SECONDS = 10
EXPECTED_THROUGHPUT = TEST_REQUEST_COUNT / TEST_DURATION_SECONDS


@pytest.mark.asyncio
class TestPostProcessorIntegration:
    """Integration tests focusing on key processor handoffs."""

    async def test_record_to_results_data_flow(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test data flows correctly from record processor to results processor."""
        results_processor = create_results_processor_with_metrics(
            mock_user_config, RequestLatencyMetric, RequestCountMetric
        )
        message = create_metric_records_message(
            x_request_id="test-1",
            results=[{RequestLatencyMetric.tag: 100.0, RequestCountMetric.tag: 1}],
        )

        await results_processor.process_result(message)

        assert RequestLatencyMetric.tag in results_processor._results
        assert isinstance(
            results_processor._results[RequestLatencyMetric.tag], MetricArray
        )
        assert list(results_processor._results[RequestLatencyMetric.tag].data) == [
            100.0
        ]

        assert results_processor._results[RequestCountMetric.tag] == 1

    async def test_multiple_batches_accumulation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test accumulation across multiple record batches."""
        results_processor = create_results_processor_with_metrics(
            mock_user_config, RequestLatencyMetric
        )

        for idx, value in enumerate(TEST_LATENCY_VALUES):
            message = create_metric_records_message(
                x_request_id=f"test-{idx}",
                timestamp_ns=1_000_000_000 + idx,
                x_correlation_id=f"test-correlation-{idx}",
                results=[{RequestLatencyMetric.tag: value}],
            )
            await results_processor.process_result(message)

        assert RequestLatencyMetric.tag in results_processor._results
        accumulated_data = list(
            results_processor._results[RequestLatencyMetric.tag].data
        )
        assert accumulated_data == TEST_LATENCY_VALUES

    async def test_error_metrics_isolation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
        error_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test that error and valid metrics are processed separately."""
        setup_mock_registry_sequences(
            mock_metric_registry, [], [ErrorRequestCountMetric]
        )

        record_processor = MetricRecordProcessor(mock_user_config)

        assert len(record_processor.error_parse_funcs) == 1
        assert len(record_processor.valid_parse_funcs) == 0

        result = await record_processor.process_record(error_parsed_record)
        assert ErrorRequestCountMetric.tag in result
        assert result[ErrorRequestCountMetric.tag] == 1

    async def test_derived_metrics_computation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test derived metrics are computed from accumulated results."""
        setup_mock_registry_sequences(
            mock_metric_registry, [RequestThroughputMetric], []
        )

        results_processor = MetricResultsProcessor(mock_user_config)

        results_processor._results[RequestCountMetric.tag] = TEST_REQUEST_COUNT
        results_processor._results[BenchmarkDurationMetric.tag] = (
            TEST_DURATION_SECONDS * NANOS_PER_SECOND
        )

        await results_processor.update_derived_metrics()

        assert RequestThroughputMetric.tag in results_processor._results
        assert (
            results_processor._results[RequestThroughputMetric.tag]
            == EXPECTED_THROUGHPUT
        )

    async def test_complete_pipeline_summary(
        self,
        mock_metric_registry: Mock,
        mock_user_config: UserConfig,
    ) -> None:
        """Test complete pipeline produces proper summary results."""
        results_processor = create_results_processor_with_metrics(
            mock_user_config, RequestLatencyMetric
        )

        results_processor._results[RequestLatencyMetric.tag] = MetricArray()
        results_processor._results[RequestLatencyMetric.tag].extend(TEST_LATENCY_VALUES)

        summary = await results_processor.summarize()

        assert isinstance(summary, list)
        assert all(hasattr(result, "tag") for result in summary)
        assert all(hasattr(result, "avg") for result in summary)
        assert all(hasattr(result, "count") for result in summary)
