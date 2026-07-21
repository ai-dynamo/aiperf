# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration unit tests for the metrics record/accumulator pipeline."""

from unittest.mock import Mock

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.models import ParsedResponseRecord
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from tests.unit.post_processors.conftest import (
    create_metric_records_data,
    setup_mock_registry_sequences,
)

TEST_LATENCY_VALUES_NS = [100_000_000.0, 150_000_000.0, 200_000_000.0]
TEST_REQUEST_COUNT = 100
TEST_DURATION_SECONDS = 10
EXPECTED_THROUGHPUT = TEST_REQUEST_COUNT / TEST_DURATION_SECONDS


@pytest.mark.asyncio
class TestPostProcessorIntegration:
    """Integration tests focusing on key metric handoffs."""

    async def test_record_to_accumulator_data_flow(self, mock_run) -> None:
        """MetricRecordsData flows into the accumulator summary."""
        accumulator = MetricsAccumulator(mock_run)
        message = create_metric_records_data(
            x_request_id="test-1",
            results=[
                {RequestLatencyMetric.tag: 100_000_000.0, RequestCountMetric.tag: 1}
            ],
        )

        await accumulator.process_record(message)
        summary = await accumulator.summarize()

        assert summary.results[RequestLatencyMetric.tag].avg == pytest.approx(100.0)
        assert summary.results[RequestLatencyMetric.tag].count == 1
        assert summary.results[RequestCountMetric.tag].avg == 1

    async def test_multiple_batches_accumulation(self, mock_run) -> None:
        """The accumulator summarizes values across multiple record batches."""
        accumulator = MetricsAccumulator(mock_run)

        for idx, value in enumerate(TEST_LATENCY_VALUES_NS):
            message = create_metric_records_data(
                x_request_id=f"test-{idx}",
                request_start_ns=1_000_000_000 + idx,
                x_correlation_id=f"test-correlation-{idx}",
                results=[{RequestLatencyMetric.tag: value}],
            )
            await accumulator.process_record(message)

        summary = await accumulator.summarize()
        latency = summary.results[RequestLatencyMetric.tag]

        assert latency.count == len(TEST_LATENCY_VALUES_NS)
        assert latency.avg == pytest.approx(150.0)
        assert latency.min == pytest.approx(100.0)
        assert latency.max == pytest.approx(200.0)

    async def test_error_metrics_isolation(
        self,
        mock_metric_registry: Mock,
        mock_run,
        error_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Error and valid metrics are parsed by separate record-processor paths."""
        setup_mock_registry_sequences(
            mock_metric_registry, [], [ErrorRequestCountMetric]
        )

        record_processor = MetricRecordProcessor(mock_run)

        assert len(record_processor.error_parse_funcs) == 1
        assert len(record_processor.valid_parse_funcs) == 0

        from tests.unit.post_processors.conftest import create_metric_metadata

        metadata = create_metric_metadata()
        result = await record_processor.process_record(error_parsed_record, metadata)
        assert ErrorRequestCountMetric.tag in result.metrics
        assert result.metrics[ErrorRequestCountMetric.tag] == 1

    async def test_derived_metrics_computation(self, mock_run) -> None:
        """Derived metrics are computed from accumulated aggregate metrics."""
        accumulator = MetricsAccumulator(mock_run)
        await accumulator.process_record(
            create_metric_records_data(
                x_request_id="test-1",
                results=[
                    {
                        RequestCountMetric.tag: TEST_REQUEST_COUNT,
                        MinRequestTimestampMetric.tag: 0,
                        MaxResponseTimestampMetric.tag: (
                            TEST_DURATION_SECONDS * NANOS_PER_SECOND
                        ),
                    }
                ],
            )
        )

        summary = await accumulator.summarize()

        assert RequestThroughputMetric.tag in summary.results
        assert summary.results[RequestThroughputMetric.tag].avg == pytest.approx(
            EXPECTED_THROUGHPUT
        )

    async def test_complete_pipeline_summary(self, mock_run) -> None:
        """The accumulator produces typed summary results."""
        accumulator = MetricsAccumulator(mock_run)

        for idx, value in enumerate(TEST_LATENCY_VALUES_NS):
            await accumulator.process_record(
                create_metric_records_data(
                    x_request_id=f"test-{idx}",
                    request_start_ns=1_000_000_000 + idx,
                    results=[{RequestLatencyMetric.tag: value}],
                )
            )

        summary = await accumulator.summarize()

        assert isinstance(summary, AccumulatorMetricsSummary)
        assert all(hasattr(result, "tag") for result in summary.results.values())
        assert all(hasattr(result, "avg") for result in summary.results.values())
        assert all(hasattr(result, "count") for result in summary.results.values())
