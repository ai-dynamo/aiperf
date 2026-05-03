# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration unit tests for post-processing pipeline."""

from unittest.mock import Mock

import numpy as np
import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import AggregationKind, MetricType
from aiperf.common.models import MetricResult, ParsedResponseRecord
from aiperf.config import AIPerfConfig
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.benchmark_duration_metric import BenchmarkDurationMetric
from aiperf.metrics.types.error_request_count import ErrorRequestCountMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from aiperf.post_processors.metric_record_processor import MetricRecordProcessor
from tests.unit.post_processors.conftest import (
    _make_run,
    create_accumulator_with_metrics,
    create_metric_metadata,
    create_metric_records_message,
    setup_mock_registry_sequences,
)

TEST_LATENCY_VALUES = [100.0, 150.0, 200.0]
TEST_REQUEST_COUNT = 100
TEST_DURATION_SECONDS = 10
EXPECTED_THROUGHPUT = TEST_REQUEST_COUNT / TEST_DURATION_SECONDS


@pytest.mark.asyncio
class TestPostProcessorIntegration:
    """Integration tests focusing on key processor handoffs.

    Adapted from the metrics-accumulator branch to use the k8s
    ``MetricsAccumulator(run=...)`` constructor and msgspec
    ``MetricRecordsData`` wire envelopes. The behavior under test is the
    same: data flowing from per-record processing into the accumulator's
    column store, accumulating across batches, and surfacing in summaries.
    """

    async def test_record_to_results_data_flow(
        self,
        mock_metric_registry: Mock,
        mock_user_config: AIPerfConfig,
    ) -> None:
        """Test data flows correctly from record processor to accumulator."""
        accumulator = create_accumulator_with_metrics(
            mock_user_config, RequestLatencyMetric, RequestCountMetric
        )
        message = create_metric_records_message(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestLatencyMetric.tag: 100.0, RequestCountMetric.tag: 1}],
        )

        await accumulator.process_record(message.to_data())

        # RECORD metric stored in column store numeric column
        assert RequestLatencyMetric.tag in accumulator._column_store.numeric_tags()
        values = accumulator._column_store.numeric(RequestLatencyMetric.tag)
        assert list(values[~np.isnan(values)]) == [100.0]

        # AGGREGATE metric also stored in column store
        assert RequestCountMetric.tag in accumulator._column_store.numeric_tags()
        agg_values = accumulator._column_store.numeric(RequestCountMetric.tag)
        assert list(agg_values[~np.isnan(agg_values)]) == [1.0]

    async def test_multiple_batches_accumulation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: AIPerfConfig,
    ) -> None:
        """Test accumulation across multiple record batches."""
        accumulator = create_accumulator_with_metrics(
            mock_user_config, RequestLatencyMetric
        )

        for idx, value in enumerate(TEST_LATENCY_VALUES):
            message = create_metric_records_message(
                x_request_id=f"test-{idx}",
                session_num=idx,
                request_start_ns=1_000_000_000 + idx,
                x_correlation_id=f"test-correlation-{idx}",
                results=[{RequestLatencyMetric.tag: value}],
            )
            await accumulator.process_record(message.to_data())

        assert RequestLatencyMetric.tag in accumulator._column_store.numeric_tags()
        values = accumulator._column_store.numeric(RequestLatencyMetric.tag)
        accumulated_data = list(values[~np.isnan(values)])
        assert accumulated_data == TEST_LATENCY_VALUES

    async def test_error_metrics_isolation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: AIPerfConfig,
        error_parsed_record: ParsedResponseRecord,
    ) -> None:
        """Test that error and valid metrics are processed separately."""
        setup_mock_registry_sequences(
            mock_metric_registry, [], [ErrorRequestCountMetric]
        )

        record_processor = MetricRecordProcessor(run=_make_run(mock_user_config))

        assert len(record_processor.error_parse_funcs) == 1
        assert len(record_processor.valid_parse_funcs) == 0

        metadata = create_metric_metadata()
        result = await record_processor.process_record(error_parsed_record, metadata)
        assert ErrorRequestCountMetric.tag in result
        assert result[ErrorRequestCountMetric.tag] == 1

    async def test_derived_metrics_computation(
        self,
        mock_metric_registry: Mock,
        mock_user_config: AIPerfConfig,
    ) -> None:
        """Test derived metrics are computed from accumulated results."""
        setup_mock_registry_sequences(
            mock_metric_registry, [RequestThroughputMetric], []
        )

        accumulator = MetricsAccumulator(run=_make_run(mock_user_config))

        # Set up RequestCount as aggregate with values in column store
        accumulator._tags_to_types = {
            RequestCountMetric.tag: MetricType.AGGREGATE,
        }
        accumulator._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        accumulator._metric_classes = {
            RequestCountMetric.tag: RequestCountMetric,
            RequestThroughputMetric.tag: RequestThroughputMetric,
            BenchmarkDurationMetric.tag: BenchmarkDurationMetric,
        }
        accumulator._column_store.ingest(
            idx=0,
            record_metrics={RequestCountMetric.tag: float(TEST_REQUEST_COUNT)},
            start_ns=1_000_000_000.0,
            end_ns=1_100_000_000.0,
            generation_start_ns=None,
        )

        # BenchmarkDuration is a DERIVED metric; add it as a derive func that
        # returns a constant, then let RequestThroughput compute from both.
        benchmark_ns = TEST_DURATION_SECONDS * NANOS_PER_SECOND

        def benchmark_duration_derive(results_dict: MetricResultsDict) -> int:
            return benchmark_ns

        accumulator._derive_funcs = {
            BenchmarkDurationMetric.tag: benchmark_duration_derive,
            **accumulator._derive_funcs,
        }

        full_results = await accumulator.full_metrics()

        assert RequestThroughputMetric.tag in full_results
        assert isinstance(full_results[RequestThroughputMetric.tag], MetricResult)
        assert full_results[RequestThroughputMetric.tag].avg == EXPECTED_THROUGHPUT

    async def test_complete_pipeline_summary(
        self,
        mock_metric_registry: Mock,
        mock_user_config: AIPerfConfig,
    ) -> None:
        """Test complete pipeline produces proper summary results."""
        accumulator = create_accumulator_with_metrics(
            mock_user_config, RequestLatencyMetric
        )

        # Add data via process_record
        for i, v in enumerate(TEST_LATENCY_VALUES):
            msg = create_metric_records_message(
                x_request_id=f"test-{i}",
                session_num=i,
                request_start_ns=1_000_000_000 + i,
                results=[{RequestLatencyMetric.tag: v}],
            )
            await accumulator.process_record(msg.to_data())

        summary = await accumulator.summarize()

        assert hasattr(summary, "results")
        assert isinstance(summary.results, dict)
        assert all(hasattr(result, "tag") for result in summary.results.values())
        assert all(hasattr(result, "avg") for result in summary.results.values())
        assert all(hasattr(result, "count") for result in summary.results.values())
