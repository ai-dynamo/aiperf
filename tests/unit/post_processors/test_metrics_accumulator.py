# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MetricsAccumulator."""

from __future__ import annotations

import math
from unittest.mock import Mock, patch

import numpy as np
import pytest

from aiperf.common.accumulator_protocols import ExportContext, SummaryContext
from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import AggregationKind, CreditPhase, MetricType
from aiperf.common.exceptions import NoMetricValue
from aiperf.common.models import MetricResult, TimesliceResult
from aiperf.metrics.accumulator import (
    _AGGREGATE_FUNCS,
    AccumulatorMetricsSummary,
    MetricsAccumulator,
)
from aiperf.metrics.column_store import ColumnStore
from aiperf.metrics.metric_dicts import MetricResultsDict, metric_result_from_array
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from tests.unit.post_processors.conftest import (
    create_accumulator_with_metrics,
    create_metric_metadata,
    create_metric_records_data,
)


class TestMetricsAccumulator:
    """Test cases for MetricsAccumulator."""

    def test_initialization(self, mock_metric_registry: Mock, mock_run) -> None:
        """Test processor initialization sets up necessary data structures."""
        processor = MetricsAccumulator(mock_run)

        assert isinstance(processor._derive_funcs, dict)
        assert isinstance(processor._column_store, ColumnStore)
        assert isinstance(processor._tags_to_types, dict)
        assert isinstance(processor._aggregation_kinds, dict)

    @pytest.mark.asyncio
    async def test_process_record_record_metric(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test processing record metric stores values in column store."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        message = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(message)

        assert "test_record" in processor._column_store.numeric_tags()
        values = processor._column_store.numeric("test_record")
        assert list(values[~np.isnan(values)]) == [42.0]

        # New data should expand the column store
        message2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=1_000_000_001,
            results=[{"test_record": 84.0}],
        )
        await processor.process_record(message2)
        values = processor._column_store.numeric("test_record")
        assert list(values[~np.isnan(values)]) == [42.0, 84.0]

    @pytest.mark.asyncio
    async def test_profile_metric_duration_coverage_is_phase_scoped(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Warmup observations cannot hide profiling metrics that stopped early."""
        processor = MetricsAccumulator(mock_run)
        phase_start_ns = 100 * NANOS_PER_SECOND

        profiling = create_metric_records_data(
            session_num=0,
            request_start_ns=147 * NANOS_PER_SECOND,
            request_end_ns=149 * NANOS_PER_SECOND,
            results=[
                {"time_to_first_token": NANOS_PER_SECOND},
                {"inter_token_latency": 100_000_000},
            ],
        )
        warmup_metadata = create_metric_metadata(
            session_num=0,
            benchmark_phase=CreditPhase.WARMUP,
            request_start_ns=197 * NANOS_PER_SECOND,
            request_end_ns=199 * NANOS_PER_SECOND,
        )
        warmup = create_metric_records_data(
            metadata=warmup_metadata,
            results=[
                {"time_to_first_token": NANOS_PER_SECOND},
                {"inter_token_latency": 100_000_000},
            ],
        )
        await processor.process_record(profiling)
        await processor.process_record(warmup)

        coverage = processor.profile_metric_duration_coverage(
            ExportContext(
                start_ns=phase_start_ns,
                phase=CreditPhase.PROFILING,
            ),
            phase_name="profiling",
            expected_duration_seconds=100.0,
            required_ratio=0.98,
        )

        assert coverage.ttft_ratio == pytest.approx(0.48)
        assert coverage.inter_token_latency_ratio == pytest.approx(0.49)
        assert coverage.passed is False

    @pytest.mark.asyncio
    async def test_profile_metric_duration_coverage_accepts_threshold_boundary(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """A last TTFT exactly at the threshold and a later ITL record pass."""
        processor = MetricsAccumulator(mock_run)
        phase_start_ns = 100 * NANOS_PER_SECOND
        record = create_metric_records_data(
            session_num=0,
            request_start_ns=197 * NANOS_PER_SECOND,
            request_end_ns=199 * NANOS_PER_SECOND,
            results=[
                {"time_to_first_token": NANOS_PER_SECOND},
                {"inter_token_latency": 100_000_000},
            ],
        )
        await processor.process_record(record)

        coverage = processor.profile_metric_duration_coverage(
            ExportContext(
                start_ns=phase_start_ns,
                phase=CreditPhase.PROFILING,
            ),
            phase_name="profiling",
            expected_duration_seconds=100.0,
            required_ratio=0.98,
        )

        assert coverage.ttft_ratio == pytest.approx(0.98)
        assert coverage.inter_token_latency_ratio == pytest.approx(0.99)
        assert coverage.passed is True

    @pytest.mark.asyncio
    async def test_profile_metric_duration_coverage_requires_itl(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """TTFT alone cannot make an interactive streaming run valid."""
        processor = MetricsAccumulator(mock_run)
        phase_start_ns = 100 * NANOS_PER_SECOND
        record = create_metric_records_data(
            session_num=0,
            request_start_ns=198 * NANOS_PER_SECOND,
            request_end_ns=199 * NANOS_PER_SECOND,
            results=[{"time_to_first_token": 0}],
        )
        await processor.process_record(record)

        coverage = processor.profile_metric_duration_coverage(
            ExportContext(
                start_ns=phase_start_ns,
                phase=CreditPhase.PROFILING,
            ),
            phase_name="profiling",
            expected_duration_seconds=100.0,
            required_ratio=0.98,
        )

        assert coverage.ttft_ratio == pytest.approx(0.98)
        assert coverage.inter_token_latency_ratio == 0.0
        assert coverage.passed is False

    @pytest.mark.asyncio
    async def test_process_record_record_metric_list_values(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test processing record metric with list values stores in ragged series."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_record": MetricType.RECORD}

        message = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            results=[{"test_record": [10.0, 20.0, 30.0]}],
        )
        await processor.process_record(message)

        assert "test_record" in processor._column_store.ragged_tags()
        ragged = processor._column_store.ragged("test_record")
        assert list(ragged.values) == [10.0, 20.0, 30.0]

    @pytest.mark.asyncio
    async def test_process_record_aggregate_metric(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test processing aggregate metric stores values in column store."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }

        message1 = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestCountMetric.tag: 5}],
        )
        await processor.process_record(message1)

        assert RequestCountMetric.tag in processor._column_store.numeric_tags()
        values = processor._column_store.numeric(RequestCountMetric.tag)
        assert list(values[~np.isnan(values)]) == [5.0]

        message2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=1_000_000_001,
            results=[{RequestCountMetric.tag: 3}],
        )
        await processor.process_record(message2)
        values = processor._column_store.numeric(RequestCountMetric.tag)
        assert list(values[~np.isnan(values)]) == [5.0, 3.0]

    @pytest.mark.asyncio
    async def test_aggregate_sum_computed_at_summary_time(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test aggregate SUM values are computed vectorized from stored values."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric}

        for i in range(3):
            msg = create_metric_records_data(
                x_request_id=f"test-{i}",
                session_num=i,
                request_start_ns=1_000_000_000 + i,
                results=[{RequestCountMetric.tag: 5}],
            )
            await processor.process_record(msg)

        results = processor._compute_results()
        assert results[RequestCountMetric.tag].avg == 15.0

    @pytest.mark.asyncio
    async def test_record_count(self, mock_metric_registry: Mock, mock_run) -> None:
        """Test record_count derives from column store."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {}

        msg1 = create_metric_records_data(x_request_id="test-1", session_num=0)
        msg2 = create_metric_records_data(
            x_request_id="test-2", session_num=1, request_start_ns=1_000_000_001
        )

        await processor.process_record(msg1)
        await processor.process_record(msg2)

        assert processor.record_count == 2

    @pytest.mark.asyncio
    async def test_export_results_separates_warmup_and_profiling_with_reused_session_num(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Warmup/profiling credit ids restart at 0; accumulator rows must not collide."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}
        processor._metric_classes = {RequestLatencyMetric.tag: RequestLatencyMetric}

        warmup_msg = create_metric_records_data(
            session_num=0,
            benchmark_phase=CreditPhase.WARMUP,
            request_start_ns=1_000_000_000,
            request_end_ns=1_100_000_000,
            results=[{RequestLatencyMetric.tag: 100_000_000.0}],
        )
        profiling_msg = create_metric_records_data(
            session_num=0,
            benchmark_phase=CreditPhase.PROFILING,
            request_start_ns=2_000_000_000,
            request_end_ns=2_200_000_000,
            results=[{RequestLatencyMetric.tag: 200_000_000.0}],
        )

        await processor.process_record(warmup_msg)
        await processor.process_record(profiling_msg)

        assert processor.record_count == 2

        warmup = await processor.export_results(ExportContext(phase=CreditPhase.WARMUP))
        profiling = await processor.export_results(
            ExportContext(phase=CreditPhase.PROFILING)
        )

        assert warmup.results[RequestLatencyMetric.tag].avg == pytest.approx(100.0)
        assert profiling.results[RequestLatencyMetric.tag].avg == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_summarize_realtime_phase_scoped_excludes_warmup(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Realtime summarize scoped to PROFILING must exclude warmup records so
        the live view is not diluted by warmup latencies/counts. Pre-fix,
        summarize built no phase mask and averaged warmup+profiling together."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {
            RequestLatencyMetric.tag: MetricType.RECORD,
            RequestCountMetric.tag: MetricType.AGGREGATE,
        }
        processor._metric_classes = {
            RequestLatencyMetric.tag: RequestLatencyMetric,
            RequestCountMetric.tag: RequestCountMetric,
        }
        processor._aggregation_kinds = {RequestCountMetric.tag: AggregationKind.SUM}

        warmup_msg = create_metric_records_data(
            session_num=0,
            benchmark_phase=CreditPhase.WARMUP,
            request_start_ns=1_000_000_000,
            request_end_ns=1_100_000_000,
            results=[
                {
                    RequestLatencyMetric.tag: 100_000_000.0,
                    RequestCountMetric.tag: 1,
                }
            ],
        )
        profiling_msg = create_metric_records_data(
            session_num=0,
            benchmark_phase=CreditPhase.PROFILING,
            request_start_ns=2_000_000_000,
            request_end_ns=2_200_000_000,
            results=[
                {
                    RequestLatencyMetric.tag: 200_000_000.0,
                    RequestCountMetric.tag: 1,
                }
            ],
        )

        await processor.process_record(warmup_msg)
        await processor.process_record(profiling_msg)

        assert processor.record_count == 2

        summary = await processor.summarize(SummaryContext(phase=CreditPhase.PROFILING))

        # Pre-fix: no phase mask → warmup+profiling averaged to 150.0.
        assert summary.results[RequestLatencyMetric.tag].avg == pytest.approx(200.0)
        # Request count reflects only the single profiling record, not both.
        assert summary.results[RequestCountMetric.tag].avg == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_process_record_metric_accumulates_values(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)

        for idx, value in enumerate([42_000_000.0, 84_000_000.0]):
            message = create_metric_records_data(
                x_request_id=f"test-{idx}",
                request_start_ns=1_000_000_000 + idx,
                request_end_ns=1_100_000_000 + idx,
                results=[{RequestLatencyMetric.tag: value}],
            )
            await accumulator.process_record(message)

        summary = await accumulator.summarize()
        result = summary.results[RequestLatencyMetric.tag]

        assert accumulator.record_count == 2
        assert result.unit == "ms"
        assert result.count == 2
        assert result.avg == pytest.approx(63.0)
        assert result.sum == pytest.approx(126.0)

    @pytest.mark.asyncio
    async def test_process_record_list_metric_summarizes_distribution(
        self, mock_run
    ) -> None:
        accumulator = MetricsAccumulator(mock_run)
        message = create_metric_records_data(
            x_request_id="test-1",
            results=[
                {
                    InterChunkLatencyMetric.tag: [
                        10_000_000.0,
                        20_000_000.0,
                        30_000_000.0,
                    ]
                }
            ],
        )

        await accumulator.process_record(message)

        summary = await accumulator.summarize()
        result = summary.results[InterChunkLatencyMetric.tag]
        assert result.count == 3
        assert result.sum == pytest.approx(60.0)
        assert result.min == pytest.approx(10.0)
        assert result.max == pytest.approx(30.0)

    @pytest.mark.asyncio
    async def test_process_aggregate_metric_sums_values(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)

        for idx, value in enumerate([5, 3]):
            message = create_metric_records_data(
                x_request_id=f"test-{idx}",
                request_start_ns=1_000_000_000 + idx,
                results=[{RequestCountMetric.tag: value}],
            )
            await accumulator.process_record(message)

        summary = await accumulator.summarize()

        assert summary.results[RequestCountMetric.tag].avg == 8

    @pytest.mark.asyncio
    async def test_derived_metrics_are_computed_from_accumulated_results(
        self, mock_run
    ) -> None:
        accumulator = MetricsAccumulator(mock_run)
        message = create_metric_records_data(
            x_request_id="test-1",
            results=[
                {
                    RequestCountMetric.tag: 100,
                    MinRequestTimestampMetric.tag: 0,
                    MaxResponseTimestampMetric.tag: 10 * NANOS_PER_SECOND,
                }
            ],
        )

        await accumulator.process_record(message)
        summary = await accumulator.summarize()

        assert summary.results[RequestThroughputMetric.tag].avg == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_derived_metrics_ignore_no_metric_value(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)

        def failing_derive_func(results_dict: MetricResultsDict) -> float:
            raise NoMetricValue("Cannot derive value")

        accumulator._derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}

        with patch.object(accumulator, "debug") as mock_debug:
            summary = await accumulator.summarize()

        assert RequestThroughputMetric.tag not in summary.results
        assert any(
            "No metric value for derived metric" in str(call.args[0])
            for call in mock_debug.call_args_list
        )

    @pytest.mark.asyncio
    async def test_derived_metrics_warn_on_unexpected_exception(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)

        def failing_derive_func(results_dict: MetricResultsDict) -> float:
            raise ValueError("Calculation error")

        accumulator._derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}

        with patch.object(accumulator, "warning") as mock_warning:
            summary = await accumulator.summarize()

        assert RequestThroughputMetric.tag not in summary.results
        mock_warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_returns_typed_summary(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)
        await accumulator.process_record(
            create_metric_records_data(
                x_request_id="test-1",
                results=[{RequestLatencyMetric.tag: 42_000_000.0}],
            )
        )

        summary = await accumulator.summarize()

        assert isinstance(summary, AccumulatorMetricsSummary)
        assert summary.results[RequestLatencyMetric.tag].unit == "ms"
        assert summary.results[RequestLatencyMetric.tag].avg == pytest.approx(42.0)


class TestComputeResultsWindowBounds:
    """Test that _compute_results propagates window bounds to derived metrics."""

    @pytest.mark.asyncio
    async def test_window_bounds_set_on_scalar_dict(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Window bounds passed to _compute_results reach the derived-metric scalar dict."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric}

        captured: list[MetricResultsDict] = []

        def spy_derive(results_dict: MetricResultsDict) -> float:
            captured.append(results_dict)
            return 42.0

        processor._derive_funcs = {RequestThroughputMetric.tag: spy_derive}
        processor._metric_classes[RequestThroughputMetric.tag] = RequestThroughputMetric

        msg = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestCountMetric.tag: 10}],
        )
        await processor.process_record(msg)

        processor._compute_results(
            window_start_ns=1_000_000_000, window_end_ns=5_000_000_000
        )

        assert len(captured) == 1
        assert captured[0].window_start_ns == 1_000_000_000
        assert captured[0].window_end_ns == 5_000_000_000

    @pytest.mark.asyncio
    async def test_compute_results_for_mask_forwards_window_bounds(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """compute_results_for_mask forwards window bounds to _compute_results."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric}

        captured: list[MetricResultsDict] = []

        def spy_derive(results_dict: MetricResultsDict) -> float:
            captured.append(results_dict)
            return 42.0

        processor._derive_funcs = {RequestThroughputMetric.tag: spy_derive}
        processor._metric_classes[RequestThroughputMetric.tag] = RequestThroughputMetric

        msg = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestCountMetric.tag: 10}],
        )
        await processor.process_record(msg)

        mask = np.ones(processor._column_store.count, dtype=bool)
        processor.compute_results_for_mask(
            mask, window_start_ns=2_000_000_000, window_end_ns=8_000_000_000
        )

        assert len(captured) == 1
        assert captured[0].window_start_ns == 2_000_000_000
        assert captured[0].window_end_ns == 8_000_000_000


class TestAggregationKind:
    """Test AggregationKind enum and vectorized aggregate functions."""

    def test_sum(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert _AGGREGATE_FUNCS[AggregationKind.SUM](values) == 10.0

    def test_max(self) -> None:
        values = np.array([1.0, 4.0, 2.0, 3.0])
        assert _AGGREGATE_FUNCS[AggregationKind.MAX](values) == 4.0

    def test_min(self) -> None:
        values = np.array([3.0, 1.0, 4.0, 2.0])
        assert _AGGREGATE_FUNCS[AggregationKind.MIN](values) == 1.0

    def test_aggregate_kind_on_request_count(self) -> None:
        assert RequestCountMetric.aggregation_kind == AggregationKind.SUM

    def test_aggregate_kind_on_min_request_timestamp(self) -> None:
        from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric

        assert MinRequestTimestampMetric.aggregation_kind == AggregationKind.MIN

    def test_aggregate_kind_on_max_response_timestamp(self) -> None:
        from aiperf.metrics.types.max_response_metric import (
            MaxResponseTimestampMetric,
        )

        assert MaxResponseTimestampMetric.aggregation_kind == AggregationKind.MAX


class TestQueryTimeRange:
    @pytest.mark.asyncio
    async def test_empty(self, mock_metric_registry: Mock, mock_run) -> None:
        processor = MetricsAccumulator(mock_run)
        mask = processor.query_time_range(0, 10_000)
        assert len(mask) == 0

    @pytest.mark.asyncio
    async def test_single_record_inside(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {}
        record = create_metric_records_data(
            x_request_id="test-1", session_num=0, request_start_ns=5_000
        )
        await processor.process_record(record)
        mask = processor.query_time_range(0, 10_000)
        assert mask.sum() == 1

    @pytest.mark.asyncio
    async def test_single_record_outside(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {}
        record = create_metric_records_data(
            x_request_id="test-1", session_num=0, request_start_ns=15_000
        )
        await processor.process_record(record)
        mask = processor.query_time_range(0, 10_000)
        assert mask.sum() == 0

    @pytest.mark.asyncio
    async def test_boundary_inclusive_start_exclusive_end(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {}
        record1 = create_metric_records_data(
            x_request_id="test-1", session_num=0, request_start_ns=1_000
        )
        record2 = create_metric_records_data(
            x_request_id="test-2", session_num=1, request_start_ns=2_000
        )
        await processor.process_record(record1)
        await processor.process_record(record2)
        # [1_000, 2_000) should include 1_000 but exclude 2_000
        mask = processor.query_time_range(1_000, 2_000)
        assert mask.sum() == 1
        assert mask[0] is np.True_
        assert mask[1] is np.False_

    @pytest.mark.asyncio
    async def test_multiple_records_filtering(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {}
        for i, ts in enumerate([100, 200, 300, 400, 500]):
            r = create_metric_records_data(
                x_request_id=f"test-{i}", session_num=i, request_start_ns=ts
            )
            await processor.process_record(r)

        mask = processor.query_time_range(200, 400)
        assert mask.sum() == 2
        np.testing.assert_array_equal(np.where(mask)[0], [1, 2])


class TestSummarize:
    @pytest.mark.asyncio
    async def test_summarize_returns_metrics_summary(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test summarize returns AccumulatorMetricsSummary wrapping MetricResult objects."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}
        processor._metric_classes = {RequestLatencyMetric.tag: RequestLatencyMetric}

        # Inject data via process_record
        msg = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            results=[{RequestLatencyMetric.tag: 42.0}],
        )
        await processor.process_record(msg)

        summary = await processor.summarize()

        assert isinstance(summary, AccumulatorMetricsSummary)
        assert RequestLatencyMetric.tag in summary.results
        # Also includes effective_concurrency + effective_decode_throughput from sweep injection
        assert len(summary.results) >= 1
        assert isinstance(summary.results[RequestLatencyMetric.tag], MetricResult)
        assert summary.timeslices is None

    @pytest.mark.asyncio
    async def test_summarize_with_derived_metrics(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test derived metrics are computed during summarize."""

        def mock_derive_func(results_dict: MetricResultsDict) -> float:
            return 100.0

        processor = MetricsAccumulator(mock_run)
        processor._derive_funcs = {RequestThroughputMetric.tag: mock_derive_func}
        processor._metric_classes = {
            RequestThroughputMetric.tag: RequestThroughputMetric
        }

        summary = await processor.summarize()

        assert isinstance(summary, AccumulatorMetricsSummary)
        assert RequestThroughputMetric.tag in summary.results

    @pytest.mark.asyncio
    async def test_summarize_derived_handles_no_metric_value(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test derived metrics gracefully handle NoMetricValue."""

        def failing_derive_func(results_dict: MetricResultsDict) -> float:
            raise NoMetricValue("Cannot derive value")

        processor = MetricsAccumulator(mock_run)
        processor._derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}
        processor._metric_classes = {}

        with patch.object(processor, "debug") as mock_debug:
            summary = await processor.summarize()
            assert RequestThroughputMetric.tag not in summary.results
            mock_debug.assert_called()

    @pytest.mark.asyncio
    async def test_summarize_derived_handles_value_error(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test derived metrics gracefully handle ValueError."""

        def failing_derive_func(results_dict: MetricResultsDict) -> float:
            raise ValueError("Calculation error")

        processor = MetricsAccumulator(mock_run)
        processor._derive_funcs = {RequestThroughputMetric.tag: failing_derive_func}
        processor._metric_classes = {}

        with patch.object(processor, "warning") as mock_warning:
            summary = await processor.summarize()
            assert RequestThroughputMetric.tag not in summary.results
            mock_warning.assert_called()


class TestTimesliceSummarize:
    @pytest.mark.asyncio
    async def test_summarize_with_timeslices(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test summarize produces timeslice results when slice_duration is set."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric}

        # Process records in two different 1-second windows
        msg1 = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            request_end_ns=int(0.6 * NANOS_PER_SECOND),
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg1)

        msg2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            request_end_ns=int(2.5 * NANOS_PER_SECOND),
            results=[{"test_record": 84.0}],
        )
        await processor.process_record(msg2)

        summary = await processor.summarize()

        assert isinstance(summary, AccumulatorMetricsSummary)
        assert summary.timeslices is not None
        assert len(summary.timeslices) == 2
        # Each timeslice should have results
        assert len(summary.timeslices[0].metric_results) > 0
        assert len(summary.timeslices[1].metric_results) > 0

    @pytest.mark.asyncio
    async def test_summarize_no_timeslices_without_config(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test summarize returns None timeslices when slice_duration is not set."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric}

        msg = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg)

        summary = await processor.summarize()
        assert summary.timeslices is None

    @pytest.mark.asyncio
    async def test_timeslice_accumulation(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test that values within same timeslice are accumulated."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric}

        # Two records in same 1-second window
        msg1 = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"test_record": 10.0}],
        )
        await processor.process_record(msg1)

        msg2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"test_record": 20.0}],
        )
        await processor.process_record(msg2)

        summary = await processor.summarize()
        assert summary.timeslices is not None
        # Both should be in the same timeslice
        assert len(summary.timeslices) == 1

    @pytest.mark.asyncio
    async def test_timeslice_aggregate_metrics(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test aggregate metrics use vectorized AggregationKind per timeslice."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._aggregation_kinds = {
            RequestCountMetric.tag: AggregationKind.SUM,
        }
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric}

        # First timeslice: 5 + 3 = 8
        msg1 = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            request_end_ns=int(0.6 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 5}],
        )
        await processor.process_record(msg1)

        msg2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            request_end_ns=int(0.8 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 3}],
        )
        await processor.process_record(msg2)

        # Second timeslice: 7
        msg3 = create_metric_records_data(
            x_request_id="test-3",
            session_num=2,
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            request_end_ns=int(2.5 * NANOS_PER_SECOND),
            results=[{RequestCountMetric.tag: 7}],
        )
        await processor.process_record(msg3)

        summary = await processor.summarize()
        assert summary.timeslices is not None
        assert len(summary.timeslices) == 2

        # Each timeslice should have aggregated separately via SUM
        ts0_results = summary.timeslices[0].metric_results
        ts1_results = summary.timeslices[1].metric_results
        assert ts0_results[RequestCountMetric.tag].avg == 8  # 5 + 3
        assert ts1_results[RequestCountMetric.tag].avg == 7

    @pytest.mark.asyncio
    async def test_timeslice_max_aggregate(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test MAX aggregation per timeslice."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"max_ts": MetricType.AGGREGATE}
        processor._aggregation_kinds = {"max_ts": AggregationKind.MAX}
        processor._metric_classes = {"max_ts": RequestCountMetric}

        msg1 = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"max_ts": 100}],
        )
        await processor.process_record(msg1)

        msg2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"max_ts": 300}],
        )
        await processor.process_record(msg2)

        summary = await processor.summarize()
        assert summary.timeslices is not None
        ts0_results = summary.timeslices[0].metric_results
        assert ts0_results["max_ts"].avg == 300.0  # MAX of 100, 300

    @pytest.mark.asyncio
    async def test_timeslice_min_aggregate(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test MIN aggregation per timeslice."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"min_ts": MetricType.AGGREGATE}
        processor._aggregation_kinds = {"min_ts": AggregationKind.MIN}
        processor._metric_classes = {"min_ts": RequestCountMetric}

        msg1 = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            request_start_ns=int(0.3 * NANOS_PER_SECOND),
            results=[{"min_ts": 500}],
        )
        await processor.process_record(msg1)

        msg2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"min_ts": 200}],
        )
        await processor.process_record(msg2)

        summary = await processor.summarize()
        assert summary.timeslices is not None
        ts0_results = summary.timeslices[0].metric_results
        assert ts0_results["min_ts"].avg == 200.0  # MIN of 500, 200

    @pytest.mark.asyncio
    async def test_compute_timeslices_populates_window_bounds(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test _compute_timeslices populates window bounds on each TimesliceResult."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric}

        msg1 = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            request_end_ns=int(0.6 * NANOS_PER_SECOND),
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg1)

        msg2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            request_end_ns=int(2.5 * NANOS_PER_SECOND),
            results=[{"test_record": 84.0}],
        )
        await processor.process_record(msg2)

        summary = await processor.summarize()

        assert summary.timeslices is not None
        assert len(summary.timeslices) == 2
        ts0 = summary.timeslices[0]
        ts1 = summary.timeslices[1]

        # Windows should be consecutive 1-second bins
        assert ts0.end_ns == ts1.start_ns
        assert ts1.end_ns - ts1.start_ns == NANOS_PER_SECOND
        # is_complete defaults to None (complete) when window_end <= max(end_ns)
        assert ts0.is_complete is None
        assert ts1.is_complete is None

    @pytest.mark.asyncio
    async def test_timeslices_none_without_config(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Test summary.timeslices is None when slice_duration is not set."""
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric}

        msg = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            results=[{"test_record": 42.0}],
        )
        await processor.process_record(msg)

        summary = await processor.summarize()
        assert summary.timeslices is None

    @pytest.mark.asyncio
    async def test_last_timeslice_clips_to_run_end(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """The last slice's window_end clips to max(end_ns) and is flagged
        is_complete=False, matching the server-metrics export pattern. Without
        this, sweep metrics on the trailing slice get diluted by phantom idle
        padding past the actual run end."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_record": MetricType.RECORD}
        processor._metric_classes = {"test_record": RequestLatencyMetric}

        # Run extends from 0.5s to 1.7s — slice 1 [1.5, 2.5) overshoots.
        msg1 = create_metric_records_data(
            x_request_id="test-1",
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            request_end_ns=int(1.4 * NANOS_PER_SECOND),
            results=[{"test_record": 1.0}],
        )
        await processor.process_record(msg1)

        msg2 = create_metric_records_data(
            x_request_id="test-2",
            session_num=1,
            request_start_ns=int(1.5 * NANOS_PER_SECOND),
            request_end_ns=int(1.7 * NANOS_PER_SECOND),
            results=[{"test_record": 2.0}],
        )
        await processor.process_record(msg2)

        summary = await processor.summarize()
        assert summary.timeslices is not None
        assert len(summary.timeslices) == 2

        ts0 = summary.timeslices[0]
        ts1 = summary.timeslices[1]
        # First slice fully within the run → complete (is_complete = None).
        assert ts0.is_complete is None
        assert ts0.end_ns - ts0.start_ns == NANOS_PER_SECOND
        # Last slice is clipped to max(end_ns)=1.7s → partial (is_complete=False).
        assert ts1.is_complete is False
        assert ts1.start_ns == int(1.5 * NANOS_PER_SECOND)
        assert ts1.end_ns == int(1.7 * NANOS_PER_SECOND)
        # Critically, the partial duration is shorter than slice_duration.
        assert ts1.end_ns - ts1.start_ns < NANOS_PER_SECOND


class TestMetricsSummary:
    def test_to_json(self) -> None:
        summary = AccumulatorMetricsSummary(
            results={
                "test": MetricResult(
                    tag="test", header="Test", unit="ms", avg=42.0, count=1
                )
            }
        )
        json_data = summary.to_json()
        assert "results" in json_data
        assert len(json_data["results"]) == 1

    def test_to_json_with_timeslices(self) -> None:
        summary = AccumulatorMetricsSummary(
            results={},
            timeslices=[
                TimesliceResult(
                    start_ns=0,
                    end_ns=1,
                    metric_results={
                        "test": MetricResult(
                            tag="test", header="Test", unit="ms", avg=42.0, count=1
                        )
                    },
                )
            ],
        )
        json_data = summary.to_json()
        assert "timeslices" in json_data
        assert isinstance(json_data["timeslices"], list)
        assert len(json_data["timeslices"]) == 1

    def test_to_csv(self) -> None:
        summary = AccumulatorMetricsSummary(
            results={
                "test": MetricResult(
                    tag="test", header="Test", unit="ms", avg=42.0, count=1
                )
            }
        )
        csv_data = summary.to_csv()
        assert len(csv_data) == 1

    def test_to_csv_with_timeslices(self) -> None:
        summary = AccumulatorMetricsSummary(
            results={
                "test": MetricResult(
                    tag="test", header="Test", unit="ms", avg=42.0, count=1
                )
            },
            timeslices=[
                TimesliceResult(
                    start_ns=0,
                    end_ns=1,
                    metric_results={
                        "ts_test": MetricResult(
                            tag="ts_test",
                            header="TS Test",
                            unit="ms",
                            avg=10.0,
                            count=1,
                        )
                    },
                )
            ],
        )
        csv_data = summary.to_csv()
        # 1 overall result + 1 timeslice result
        assert len(csv_data) == 2
        assert csv_data[1]["timeslice"] == 0


class TestProtocolConformance:
    def test_satisfies_accumulator_protocol(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        from aiperf.common.accumulator_protocols import AccumulatorProtocol

        processor = MetricsAccumulator(mock_run)
        assert isinstance(processor, AccumulatorProtocol)

    def test_summary_satisfies_accumulator_result(self) -> None:
        from aiperf.common.accumulator_protocols import AccumulatorResult

        summary = AccumulatorMetricsSummary(results={})
        assert isinstance(summary, AccumulatorResult)


class TestMetricResultFromArray:
    """Test metric_result_from_array computes correct statistics."""

    def test_single_value(self) -> None:
        """Single-element array: all stats equal the value."""
        arr = np.array([5.0], dtype=np.float64)
        r = metric_result_from_array("test", "Test", "ms", arr, 5.0)
        assert r.tag == "test"
        assert r.header == "Test"
        assert r.unit == "ms"
        assert r.count == 1
        assert r.min == 5.0
        assert r.max == 5.0
        assert r.avg == 5.0
        assert r.std == 0.0
        assert r.p50 == 5.0

    def test_five_values(self) -> None:
        """Five evenly-spaced values: known min/max/avg/p50."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        r = metric_result_from_array("t", "T", "u", arr, 15.0)
        assert r.count == 5
        assert r.min == 1.0
        assert r.max == 5.0
        assert r.avg == 3.0
        assert r.p50 == 3.0
        np.testing.assert_allclose(r.std, np.std([1.0, 2.0, 3.0, 4.0, 5.0]))

    def test_hundred_values(self) -> None:
        """1..100: verify percentile interpolation on a larger dataset."""
        values = list(range(1, 101))
        arr = np.array(values, dtype=np.float64)
        r = metric_result_from_array("t", "T", "u", arr, float(sum(values)))
        assert r.count == 100
        assert r.min == 1.0
        assert r.max == 100.0
        assert r.avg == 50.5
        assert r.p50 == 50.5
        np.testing.assert_allclose(r.p1, 1.99)
        np.testing.assert_allclose(r.p99, 99.01)

    def test_sorts_in_place(self) -> None:
        """Verify the function sorts the input array in-place."""
        arr = np.array([5.0, 1.0, 3.0], dtype=np.float64)
        metric_result_from_array("t", "T", "u", arr, 9.0)
        np.testing.assert_array_equal(arr, [1.0, 3.0, 5.0])


# ---------------------------------------------------------------------------
# Helpers for timeslice sweep metric tests
# ---------------------------------------------------------------------------


def _make_sweep_metric_classes():
    """Create minimal metric classes needed for sweep-based timeslice tests."""
    from aiperf.common.enums import MetricType

    class FakeLatency:
        tag = "request_latency"
        type = MetricType.RECORD
        header = "Request Latency"
        unit = "ms"

    class FakeOutputTokens:
        tag = "output_sequence_length"
        type = MetricType.RECORD
        header = "Output Tokens"
        unit = "tokens"

    class FakeTTFT:
        tag = "time_to_first_token"
        type = MetricType.RECORD
        header = "Time To First Token"
        unit = "ns"

    class FakeISL:
        tag = "input_sequence_length"
        type = MetricType.RECORD
        header = "Input Sequence Length"
        unit = "tokens"

    return FakeLatency, FakeOutputTokens, FakeTTFT, FakeISL


class TestTimesliceSweepMetrics:
    """Tests for sweep-based effective_concurrency and effective_decode_throughput in timeslices."""

    @pytest.mark.asyncio
    async def test_timeslice_has_effective_concurrency_and_throughput(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """All sweep metrics are present in every timeslice with correct tag/unit."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls
        )

        # One request: 0.5s start, 0.8s end, 10 output tokens, 50ms TTFT
        msg = create_metric_records_data(
            session_num=0,
            request_start_ns=int(0.5 * NANOS_PER_SECOND),
            request_end_ns=int(0.8 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 300_000_000.0,
                    "output_sequence_length": 10.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.timeslices is not None
        for ts in summary.timeslices:
            ts_results = ts.metric_results
            assert "effective_concurrency" in ts_results
            assert "effective_decode_throughput" in ts_results
            assert "effective_prefill_throughput" in ts_results
            ec = ts_results["effective_concurrency"]
            et = ts_results["effective_decode_throughput"]
            ept = ts_results["effective_prefill_throughput"]
            assert ec.tag == "effective_concurrency"
            assert ec.unit == "requests"
            assert et.tag == "effective_decode_throughput"
            assert et.unit == "tokens/sec"
            assert ept.tag == "effective_prefill_throughput"
            assert ept.unit == "tokens/sec"

    @pytest.mark.asyncio
    async def test_timeslice_effective_concurrency_overlapping_requests(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Overlapping requests in a timeslice produce avg concurrency > 1."""
        mock_run.cfg.artifacts.slice_duration = 2.0
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls
        )

        # Two overlapping requests within the same 2s timeslice
        # Request A: [0.1s, 1.5s)  Request B: [0.5s, 1.8s)
        for i, (start, end) in enumerate(
            [(0.1, 1.5), (0.5, 1.8)],
        ):
            msg = create_metric_records_data(
                session_num=i,
                request_start_ns=int(start * NANOS_PER_SECOND),
                request_end_ns=int(end * NANOS_PER_SECOND),
                results=[
                    {
                        "request_latency": (end - start) * NANOS_PER_SECOND,
                        "output_sequence_length": 5.0,
                        "time_to_first_token": 10_000_000.0,
                    }
                ],
            )
            await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.timeslices is not None
        ts0 = summary.timeslices[0].metric_results
        assert ts0["effective_concurrency"].avg > 1.0

    @pytest.mark.asyncio
    async def test_timeslice_effective_throughput_nonzero(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Records with output_tokens and TTFT produce nonzero throughput."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls
        )

        msg = create_metric_records_data(
            session_num=0,
            request_start_ns=int(0.1 * NANOS_PER_SECOND),
            request_end_ns=int(0.9 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 800_000_000.0,
                    "output_sequence_length": 100.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.timeslices is not None
        ts0 = summary.timeslices[0].metric_results
        assert ts0["effective_decode_throughput"].avg > 0.0

    @pytest.mark.asyncio
    async def test_timeslice_sweep_metrics_zero_throughput_without_tokens(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Without output_tokens, throughput avg is 0 but concurrency is nonzero."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        latency_cls, _, _, _ = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(mock_run, latency_cls)

        msg = create_metric_records_data(
            session_num=0,
            request_start_ns=int(0.2 * NANOS_PER_SECOND),
            request_end_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"request_latency": 500_000_000.0}],
        )
        await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.timeslices is not None
        ts0 = summary.timeslices[0].metric_results
        assert ts0["effective_decode_throughput"].avg == 0.0
        assert ts0["effective_concurrency"].avg > 0.0

    @pytest.mark.asyncio
    async def test_timeslice_sweep_metrics_multiple_slices(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Records across 3 slices each have distinct sweep metric values."""
        mock_run.cfg.artifacts.slice_duration = 1.0
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls
        )

        # 3 non-overlapping requests, one per 1s slice
        records = [
            (0, 0.1, 0.9, 800e6, 10.0, 50e6),
            (1, 1.1, 1.9, 800e6, 20.0, 50e6),
            (2, 2.1, 2.9, 800e6, 30.0, 50e6),
        ]
        for session_num, start, end, latency, tokens, ttft in records:
            msg = create_metric_records_data(
                session_num=session_num,
                request_start_ns=int(start * NANOS_PER_SECOND),
                request_end_ns=int(end * NANOS_PER_SECOND),
                results=[
                    {
                        "request_latency": latency,
                        "output_sequence_length": tokens,
                        "time_to_first_token": ttft,
                    }
                ],
            )
            await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.timeslices is not None
        assert len(summary.timeslices) == 3

        # Each slice should have its own sweep metrics
        for ts_idx in range(3):
            ts = summary.timeslices[ts_idx].metric_results
            assert "effective_concurrency" in ts
            assert "effective_decode_throughput" in ts
            assert ts["effective_concurrency"].avg > 0.0
            assert ts["effective_decode_throughput"].avg > 0.0

        # Throughput should scale with token count (more tokens → higher throughput)
        # Since request durations are identical, throughput is proportional to tokens
        t0 = summary.timeslices[0].metric_results["effective_decode_throughput"].avg
        t1 = summary.timeslices[1].metric_results["effective_decode_throughput"].avg
        t2 = summary.timeslices[2].metric_results["effective_decode_throughput"].avg
        assert t1 > t0
        assert t2 > t1


class TestOverallSweepMetrics:
    """Tests for sweep-based effective_concurrency and effective_decode_throughput in overall results."""

    @pytest.mark.asyncio
    async def test_overall_has_effective_concurrency_and_throughput(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """All sweep metrics are present in the overall results with correct tag/unit."""
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls
        )

        msg = create_metric_records_data(
            session_num=0,
            request_start_ns=int(0.1 * NANOS_PER_SECOND),
            request_end_ns=int(0.9 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 800_000_000.0,
                    "output_sequence_length": 50.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg)

        summary = await acc.summarize()
        assert "effective_concurrency" in summary.results
        assert "effective_decode_throughput" in summary.results
        assert "effective_prefill_throughput" in summary.results
        ec = summary.results["effective_concurrency"]
        et = summary.results["effective_decode_throughput"]
        ept = summary.results["effective_prefill_throughput"]
        assert ec.tag == "effective_concurrency"
        assert ec.unit == "requests"
        assert et.tag == "effective_decode_throughput"
        assert et.unit == "tokens/sec"
        assert ept.tag == "effective_prefill_throughput"
        assert ept.unit == "tokens/sec"

    @pytest.mark.asyncio
    async def test_overall_effective_concurrency_overlapping_requests(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Overlapping requests produce avg concurrency > 1 in overall results."""
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls
        )

        for i, (start, end) in enumerate([(0.1, 1.5), (0.5, 1.8)]):
            msg = create_metric_records_data(
                session_num=i,
                request_start_ns=int(start * NANOS_PER_SECOND),
                request_end_ns=int(end * NANOS_PER_SECOND),
                results=[
                    {
                        "request_latency": (end - start) * NANOS_PER_SECOND,
                        "output_sequence_length": 5.0,
                        "time_to_first_token": 10_000_000.0,
                    }
                ],
            )
            await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.results["effective_concurrency"].avg > 1.0

    @pytest.mark.asyncio
    async def test_overall_effective_throughput_nonzero(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Records with output_tokens and TTFT produce nonzero overall throughput."""
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls
        )

        msg = create_metric_records_data(
            session_num=0,
            request_start_ns=int(0.1 * NANOS_PER_SECOND),
            request_end_ns=int(0.9 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 800_000_000.0,
                    "output_sequence_length": 100.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.results["effective_decode_throughput"].avg > 0.0

    @pytest.mark.asyncio
    async def test_overall_zero_throughput_without_tokens(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Without output_tokens, throughput avg is 0 but concurrency is nonzero."""
        latency_cls, _, _, _ = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(mock_run, latency_cls)

        msg = create_metric_records_data(
            session_num=0,
            request_start_ns=int(0.2 * NANOS_PER_SECOND),
            request_end_ns=int(0.7 * NANOS_PER_SECOND),
            results=[{"request_latency": 500_000_000.0}],
        )
        await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.results["effective_decode_throughput"].avg == 0.0
        assert summary.results["effective_concurrency"].avg > 0.0

    @pytest.mark.asyncio
    async def test_overall_sweep_metrics_not_present_when_empty(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """No sweep metrics when no records have been ingested."""
        latency_cls, _, _, _ = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(mock_run, latency_cls)

        summary = await acc.summarize()
        assert "effective_concurrency" not in summary.results
        assert "effective_decode_throughput" not in summary.results
        assert "effective_prefill_throughput" not in summary.results

    @pytest.mark.asyncio
    async def test_overall_effective_prefill_throughput_nonzero(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Records with ISL and TTFT produce nonzero prefill throughput."""
        latency_cls, output_cls, ttft_cls, isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls, isl_cls
        )

        msg = create_metric_records_data(
            session_num=0,
            request_start_ns=int(0.1 * NANOS_PER_SECOND),
            request_end_ns=int(0.9 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 800_000_000.0,
                    "output_sequence_length": 100.0,
                    "time_to_first_token": 50_000_000.0,
                    "input_sequence_length": 200.0,
                }
            ],
        )
        await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.results["effective_prefill_throughput"].avg > 0.0

    @pytest.mark.asyncio
    async def test_overall_zero_prefill_throughput_without_isl(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Without input_sequence_length metric, prefill throughput avg is 0."""
        latency_cls, output_cls, ttft_cls, _isl_cls = _make_sweep_metric_classes()
        acc = create_accumulator_with_metrics(
            mock_run, latency_cls, output_cls, ttft_cls
        )

        msg = create_metric_records_data(
            session_num=0,
            request_start_ns=int(0.2 * NANOS_PER_SECOND),
            request_end_ns=int(0.7 * NANOS_PER_SECOND),
            results=[
                {
                    "request_latency": 500_000_000.0,
                    "output_sequence_length": 50.0,
                    "time_to_first_token": 50_000_000.0,
                }
            ],
        )
        await acc.process_record(msg)

        summary = await acc.summarize()
        assert summary.results["effective_prefill_throughput"].avg == 0.0


class TestListMetricBackendSwitch:
    """Verify the AIPERF_METRICS_LIST_BACKEND env-flag swaps the ICL storage
    backend between RaggedSeries (default, exact, replay-capable) and the
    crick.TDigest sketch (bounded memory, approximate percentiles, no replay).
    """

    @pytest.mark.asyncio
    async def test_default_backend_is_ragged(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        from aiperf.metrics.ragged_series import RaggedSeries

        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {"test_list": MetricType.RECORD}

        message = create_metric_records_data(
            session_num=0, results=[{"test_list": [1.0, 2.0, 3.0]}]
        )
        await processor.process_record(message)

        backend = processor._column_store.ragged("test_list")
        assert isinstance(backend, RaggedSeries)
        assert backend.SUPPORTS_PER_RECORD_REPLAY is True
        assert list(backend.values) == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_tdigest_backend_via_env_flag(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        from aiperf.metrics.list_metric_aggregation import TDigestListMetricAggregator

        with patch(
            "aiperf.common.environment.Environment.METRICS.LIST_BACKEND",
            "tdigest",
        ):
            processor = MetricsAccumulator(mock_run)
            processor._tags_to_types = {"test_list": MetricType.RECORD}

            message = create_metric_records_data(
                session_num=0, results=[{"test_list": [10.0, 20.0, 30.0, 40.0]}]
            )
            await processor.process_record(message)

            backend = processor._column_store.ragged("test_list")
            assert isinstance(backend, TDigestListMetricAggregator)
            assert backend.SUPPORTS_PER_RECORD_REPLAY is False
            # Sketch retains exact running stats even though it can't replay.
            assert backend.sum == 100.0
            assert len(backend) == 4

    @pytest.mark.asyncio
    async def test_tdigest_summary_stats_match_ragged_within_tolerance(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Summary stats from the t-digest backend match the ragged backend's
        exact stats within the t-digest's documented percentile error band."""
        rng = np.random.default_rng(42)
        # Ten records each with 100 log-normal ICL samples — well above the
        # t-digest's centroid count, so it has to do real work.
        chunk_lists = [
            rng.lognormal(mean=np.log(30.0), sigma=0.5, size=100).tolist()
            for _ in range(10)
        ]

        async def _run(backend_name: str) -> MetricResult:
            with patch(
                "aiperf.common.environment.Environment.METRICS.LIST_BACKEND",
                backend_name,
            ):
                processor = MetricsAccumulator(mock_run)
                processor._tags_to_types = {"inter_chunk_latency": MetricType.RECORD}
                processor._metric_classes = {
                    "inter_chunk_latency": Mock(
                        header="ICL", unit="ms", tag="inter_chunk_latency"
                    )
                }
                for i, lst in enumerate(chunk_lists):
                    msg = create_metric_records_data(
                        session_num=i, results=[{"inter_chunk_latency": lst}]
                    )
                    await processor.process_record(msg)
                results = processor._compute_results()
                return results["inter_chunk_latency"]

        ragged_result = await _run("ragged")
        tdigest_result = await _run("tdigest")

        # Exact stats: sum, count, min, max should match exactly (Welford + side-channel).
        assert tdigest_result.count == ragged_result.count
        assert tdigest_result.sum == pytest.approx(ragged_result.sum, rel=1e-9)
        assert tdigest_result.min == pytest.approx(ragged_result.min, rel=1e-9)
        assert tdigest_result.max == pytest.approx(ragged_result.max, rel=1e-9)
        assert tdigest_result.avg == pytest.approx(ragged_result.avg, rel=1e-9)
        # Percentiles: at 1k samples the t-digest's tail error is naturally
        # looser than the asymptotic <0.05% claim (which holds at 50M samples).
        # Body percentiles tighten quickly; tail (p95, p99) can drift a few
        # percent until centroid count saturates.
        for pct, tol in (("p50", 0.01), ("p90", 0.02), ("p95", 0.03), ("p99", 0.05)):
            r_val = getattr(ragged_result, pct)
            t_val = getattr(tdigest_result, pct)
            assert t_val == pytest.approx(r_val, rel=tol), (
                f"{pct} drift outside {tol * 100:.0f}% band: "
                f"ragged={r_val} tdigest={t_val}"
            )

    @pytest.mark.asyncio
    async def test_tdigest_skips_per_record_replay_in_sweeps(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Under tdigest, ``_get_icl_data`` returns None so sweep helpers fall
        through to their request-level (non-ICL) implementations. Verifies the
        capability-flag check, not the sweep math itself."""
        from aiperf.metrics.accumulator_sweeps import _get_icl_data

        with patch(
            "aiperf.common.environment.Environment.METRICS.LIST_BACKEND",
            "tdigest",
        ):
            processor = MetricsAccumulator(mock_run)
            processor._tags_to_types = {"inter_chunk_latency": MetricType.RECORD}

            msg = create_metric_records_data(
                session_num=0,
                results=[{"inter_chunk_latency": [10.0, 20.0, 30.0]}],
            )
            await processor.process_record(msg)

            # ICL is recorded but the backend doesn't support replay.
            assert "inter_chunk_latency" in processor._column_store.ragged_tags()
            assert _get_icl_data(processor._column_store) is None


class TestMetadataColumnEncoding:
    """Verify per-record metadata routes to the right column backing:
    bool fields → uint8 (sentinel 255 = missing), low-cardinality strings →
    int16 codes + per-tag interning table, high-cardinality strings → raw list.
    """

    @pytest.mark.asyncio
    async def test_bool_metadata_round_trip(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        from aiperf.common.models.record_models import MetricRecordMetadata

        processor = MetricsAccumulator(mock_run)

        # Three records, varying was_cancelled
        for i, cancelled in enumerate((False, True, False)):
            meta = MetricRecordMetadata(
                session_num=i,
                request_start_ns=1_000_000_000 + i,
                request_end_ns=1_100_000_000 + i,
                worker_id="worker-1",
                record_processor_id="processor-1",
                benchmark_phase="profiling",
                was_cancelled=cancelled,
            )
            msg = create_metric_records_data(metadata=meta)
            await processor.process_record(msg)

        store = processor._column_store
        # has_error and was_cancelled should now be in _metadata_bool, not _metadata_numeric
        assert "was_cancelled" in store._metadata_bool
        assert "has_error" in store._metadata_bool
        assert "was_cancelled" not in store._metadata_numeric
        assert "has_error" not in store._metadata_numeric

        was_cancelled_col = store.metadata_bool("was_cancelled")
        # uint8 column: 0=False, 1=True
        assert list(was_cancelled_col[:3]) == [0, 1, 0]

        # uint8 storage = 1 byte/record; float64 would have been 8 bytes.
        # At 3 records the column is at initial_capacity (1024) so size is
        # dominated by the buffer header — but the dtype is what matters.
        assert was_cancelled_col.dtype == np.uint8

    @pytest.mark.asyncio
    async def test_categorical_metadata_intern_pool(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        processor = MetricsAccumulator(mock_run)

        worker_ids = ["worker_a", "worker_b", "worker_a", "worker_c", "worker_b"]
        for i, wid in enumerate(worker_ids):
            msg = create_metric_records_data(
                session_num=i,
                request_start_ns=1_000_000_000 + i,
                worker_id=wid,
            )
            await processor.process_record(msg)

        store = processor._column_store
        assert "worker_id" in store._metadata_categorical
        assert "worker_id" not in store._metadata_string

        codes = store.metadata_categorical("worker_id")
        assert codes.dtype == np.int32
        assert len(codes) == 5

        # Round-trip via the reverse-lookup helper
        cats = store.metadata_category_strings("worker_id")
        decoded = [cats[c] for c in codes[:5]]
        assert decoded == worker_ids
        # Pool collapses to 3 unique strings even though 5 records were ingested
        assert len(cats) == 3
        assert set(cats) == {"worker_a", "worker_b", "worker_c"}

    @pytest.mark.asyncio
    async def test_uuid_routing_drops_request_id_categoricalises_correlation_and_conversation(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """``x_request_id`` is dropped (cardinality == n_records, no grouping
        value); ``x_correlation_id`` and ``conversation_id`` route to
        categorical so per-conversation / per-template grouping analyzers
        can find them via ``unique_categorical_values`` /
        ``mask_for_categorical``."""
        processor = MetricsAccumulator(mock_run)

        msg = create_metric_records_data(
            session_num=0,
            x_request_id="req-deadbeef",
            x_correlation_id="corr-cafebabe",
            conversation_id="conv-12345",
        )
        await processor.process_record(msg)

        store = processor._column_store
        # x_request_id no longer stored anywhere — exporters read it off
        # the live record, not the ColumnStore.
        assert "x_request_id" not in store._metadata_string
        assert "x_request_id" not in store._metadata_categorical
        # Other two UUIDs are now categorical (not raw strings)
        assert "x_correlation_id" in store._metadata_categorical
        assert "conversation_id" in store._metadata_categorical
        assert "x_correlation_id" not in store._metadata_string
        assert "conversation_id" not in store._metadata_string

    @pytest.mark.asyncio
    async def test_categorical_grouping_accessors(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """``unique_categorical_values`` and ``mask_for_categorical`` enable
        per-X grouping analyses (e.g. per-conversation latency CDF)."""
        processor = MetricsAccumulator(mock_run)

        # Three "conversations" interleaved across six records
        correlations = ["conv_a", "conv_b", "conv_a", "conv_c", "conv_b", "conv_a"]
        for i, cid in enumerate(correlations):
            msg = create_metric_records_data(
                session_num=i,
                request_start_ns=1_000_000_000 + i,
                x_correlation_id=cid,
            )
            await processor.process_record(msg)

        store = processor._column_store
        # Enumerate unique values
        unique = store.unique_categorical_values("x_correlation_id")
        assert set(unique) == {"conv_a", "conv_b", "conv_c"}

        # Boolean mask per group — feeds compute_results_for_mask
        mask_a = store.mask_for_categorical("x_correlation_id", "conv_a")
        assert mask_a.dtype == np.bool_
        assert list(mask_a) == [True, False, True, False, False, True]
        assert mask_a.sum() == 3

        mask_b = store.mask_for_categorical("x_correlation_id", "conv_b")
        assert list(mask_b) == [False, True, False, False, True, False]

        # Unknown value returns an empty mask (no false positives via missing-sentinel)
        mask_unknown = store.mask_for_categorical("x_correlation_id", "conv_zzz")
        assert mask_unknown.sum() == 0

        # Unknown tag also returns empty (not KeyError)
        mask_no_tag = store.mask_for_categorical("nonexistent_tag", "anything")
        assert mask_no_tag.sum() == 0


class TestDerivedLatencyMetrics:
    """Verify summarize() emits effective_latency and credit_to_start_latency
    from stored timestamps + metadata."""

    @pytest.mark.asyncio
    async def test_credit_to_start_and_effective_latency_present(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        from aiperf.common.models.record_models import MetricRecordMetadata

        processor = MetricsAccumulator(mock_run)
        # Fixed 5 ms credit→start gap, 100 ms total request → effective = 105 ms
        for i in range(50):
            meta = MetricRecordMetadata(
                session_num=i,
                request_start_ns=1_000_000_000 + i * 200_000_000,
                request_end_ns=1_000_000_000 + i * 200_000_000 + 100_000_000,
                credit_issued_ns=1_000_000_000 + i * 200_000_000 - 5_000_000,
                worker_id="w1",
                record_processor_id="rp1",
                benchmark_phase="profiling",
                turn_index=0,
            )
            msg = create_metric_records_data(metadata=meta)
            await processor.process_record(msg)

        summary = await processor.summarize()
        assert "credit_to_start_latency" in summary.results
        assert "effective_latency" in summary.results

        c2s = summary.results["credit_to_start_latency"]
        assert c2s.unit == "ms"
        assert c2s.count == 50
        assert c2s.avg == pytest.approx(5.0, abs=1e-9)
        assert c2s.min == pytest.approx(5.0, abs=1e-9)
        assert c2s.max == pytest.approx(5.0, abs=1e-9)

        eff = summary.results["effective_latency"]
        assert eff.unit == "ms"
        assert eff.count == 50
        assert eff.avg == pytest.approx(105.0, abs=1e-9)


class TestErrorAdjustedPercentiles:
    """Issue #688: per-record latency percentiles where errored requests are
    modeled as ``+inf`` so the band correctly flips to ``inf`` once it crosses
    into the failure region.

    The implementation uses ``np.percentile(..., method="nearest")`` because
    the default linear interpolation produces ``nan`` at boundaries that
    straddle a finite sample and ``+inf`` (IEEE 754: ``inf - inf == nan``).
    See PR #825 review thread on metric_dicts.py:214.
    """

    @pytest.mark.asyncio
    async def test_adj_percentiles_flip_to_inf_at_10_percent_error_rate(
        self,
        mock_metric_registry: Mock,
        mock_run,
    ) -> None:
        """The worked example from issue #688: 10 records, 1 errored. Spec
        says adj_p95 should report ``inf``; the buggy ``method="linear"`` would
        return NaN, and ``method="lower"`` would silently return finite."""
        from aiperf.common.enums import MetricFlags
        from aiperf.common.messages.inference_messages import MetricRecordsData
        from aiperf.common.models.error_models import ErrorDetails
        from aiperf.common.models.record_models import MetricRecordMetadata
        from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric

        # Sanity-check that the metric class actually carries the opt-in flag.
        # This is what makes the inflation kick in.
        assert RequestLatencyMetric.has_flags(
            MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS
        )

        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}
        processor._metric_classes = {RequestLatencyMetric.tag: RequestLatencyMetric}
        # 9 successful records all reporting 100 ns, plus 1 errored record.
        for i in range(9):
            meta = MetricRecordMetadata(
                session_num=i,
                request_start_ns=1_000_000_000 + i * 1_000_000,
                request_end_ns=1_000_000_000 + i * 1_000_000 + 100,
                worker_id="w1",
                record_processor_id="rp1",
                benchmark_phase="profiling",
                turn_index=0,
            )
            await processor.process_record(
                MetricRecordsData(
                    metadata=meta,
                    metrics={"request_latency": 100},  # ns
                    error=None,
                )
            )
        # One errored record (no metric value emitted, but has_error=True).
        meta_err = MetricRecordMetadata(
            session_num=9,
            request_start_ns=1_000_000_009,
            request_end_ns=1_000_000_009,
            worker_id="w1",
            record_processor_id="rp1",
            benchmark_phase="profiling",
            turn_index=0,
        )
        await processor.process_record(
            MetricRecordsData(
                metadata=meta_err,
                metrics={},
                error=ErrorDetails(code=500, type="ServerError", message="boom"),
            )
        )

        results = processor._compute_results()
        rl = results.get("request_latency")
        assert rl is not None, "request_latency should be present"
        # Successes: avg/p50 unaffected on the regular metric.
        assert rl.avg == pytest.approx(100.0, abs=1e-9)
        assert rl.p50 == pytest.approx(100.0, abs=1e-9)

        # The adjusted distribution lives in its own MetricResult tagged
        # ``adj_request_latency`` — full p1..p99 band, count, sum, avg, min, max.
        adj = results.get("adj_request_latency")
        assert adj is not None, (
            "adj_request_latency should be emitted as a separate MetricResult "
            "(not as fields on request_latency); see issue #688 design notes."
        )
        # Header comes from the parent metric class.
        assert "(error-adjusted)" in adj.header
        # Full distribution shape: 9 percentiles + count/sum/avg/min/max.
        assert adj.count == 10  # 9 success + 1 error
        assert math.isinf(adj.sum), "sum is inf with one inf-inflated sample"
        assert math.isinf(adj.avg), "avg is inf with one inf-inflated sample"
        assert adj.min == pytest.approx(100.0)  # finite — least value
        assert math.isinf(adj.max), "max is inf when any error present"
        assert adj.std is None  # std mathematically undefined with inf
        # 10 samples, 1 inf: method="nearest" rounds the rank to the closest
        # integer index. At 10% error rate the boundary lands as follows
        # (rank = q/100 × 9):
        #   p50  rank=4.5 → idx 4 → 100 (finite)
        #   p90  rank=8.1 → idx 8 → 100 (finite — still in success band)
        #   p95  rank=8.55 → idx 9 → inf (crosses into failure)
        #   p99  rank=8.91 → idx 9 → inf
        # This matches issue #688's worked-example table exactly.
        assert adj.p50 == pytest.approx(100.0)
        assert adj.p90 == pytest.approx(100.0)
        assert math.isinf(adj.p95), f"adj p95 should be inf, got {adj.p95!r}"
        assert math.isinf(adj.p99), f"adj p99 should be inf, got {adj.p99!r}"
        # Critically — NOT NaN. method="nearest" avoids the linear-interp bug.
        assert not math.isnan(adj.p95), "adj p95 must not be nan (linear-interp bug)"
        assert not math.isnan(adj.p99), "adj p99 must not be nan"
        # adj_p* sidecar fields removed — request_latency carries no adj fields.
        assert not hasattr(rl, "adj_p50") or rl.adj_p50 is None
        assert not hasattr(rl, "adj_p95") or rl.adj_p95 is None

    @pytest.mark.asyncio
    async def test_adj_percentiles_absent_when_no_errors(
        self,
        mock_metric_registry: Mock,
        mock_run,
    ) -> None:
        """No errors → no inflation → adj_<tag> MetricResult is not emitted."""
        from aiperf.common.messages.inference_messages import MetricRecordsData
        from aiperf.common.models.record_models import MetricRecordMetadata
        from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric

        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestLatencyMetric.tag: MetricType.RECORD}
        processor._metric_classes = {RequestLatencyMetric.tag: RequestLatencyMetric}
        for i in range(20):
            meta = MetricRecordMetadata(
                session_num=i,
                request_start_ns=1_000_000_000 + i,
                request_end_ns=1_000_000_000 + i + 100,
                worker_id="w1",
                record_processor_id="rp1",
                benchmark_phase="profiling",
                turn_index=0,
            )
            await processor.process_record(
                MetricRecordsData(
                    metadata=meta,
                    metrics={"request_latency": 100},
                    error=None,
                )
            )
        results = processor._compute_results()
        rl = results.get("request_latency")
        assert rl is not None
        # Regular percentiles populated.
        assert rl.p95 == pytest.approx(100.0)
        # No adj_* MetricResult emitted when there are no errors to inflate.
        assert "adj_request_latency" not in results

    @pytest.mark.asyncio
    async def test_decode_duration_emits_separate_error_adjusted_metric(
        self,
        mock_metric_registry: Mock,
        mock_run,
    ) -> None:
        """Decode duration remains success-only, while ``adj_decode_duration``
        carries the failure-inflated distribution when errored records exist."""
        from aiperf.common.enums import MetricFlags
        from aiperf.common.messages.inference_messages import MetricRecordsData
        from aiperf.common.models.error_models import ErrorDetails
        from aiperf.common.models.record_models import MetricRecordMetadata
        from aiperf.metrics.types.decode_duration_metric import DecodeDurationMetric

        assert DecodeDurationMetric.has_flags(
            MetricFlags.PERCENTILE_INCLUDES_FAILED_REQUESTS
        )

        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {DecodeDurationMetric.tag: MetricType.RECORD}
        processor._metric_classes = {DecodeDurationMetric.tag: DecodeDurationMetric}
        for i in range(9):
            meta = MetricRecordMetadata(
                session_num=i,
                request_start_ns=1_000_000_000 + i * 1_000_000,
                request_end_ns=1_000_000_000 + i * 1_000_000 + 100,
                worker_id="w1",
                record_processor_id="rp1",
                benchmark_phase="profiling",
                turn_index=0,
            )
            await processor.process_record(
                MetricRecordsData(
                    metadata=meta,
                    metrics={DecodeDurationMetric.tag: 100},
                    error=None,
                )
            )

        meta_err = MetricRecordMetadata(
            session_num=9,
            request_start_ns=1_000_009_000,
            request_end_ns=1_000_009_000,
            worker_id="w1",
            record_processor_id="rp1",
            benchmark_phase="profiling",
            turn_index=0,
        )
        await processor.process_record(
            MetricRecordsData(
                metadata=meta_err,
                metrics={},
                error=ErrorDetails(code=500, type="ServerError", message="boom"),
            )
        )

        results = processor._compute_results()
        decode_duration = results.get(DecodeDurationMetric.tag)
        assert decode_duration is not None
        assert decode_duration.count == 9
        assert decode_duration.p95 == pytest.approx(100.0)

        adj = results.get("adj_decode_duration")
        assert adj is not None
        assert adj.count == 10
        assert adj.min == pytest.approx(100.0)
        assert math.isinf(adj.max)
        assert math.isinf(adj.p95)


class TestPhaseScopedExportCoarseClock:
    """Phase-scoped exports must not drop records on coarse-clock boundary ties.

    Windows ``time.time_ns()`` before Python 3.13 advances in ~0.5-15.6ms
    ticks, so a straggler request that completes right as the phase ends can
    be stamped with ``request_start_ns == requests_end_ns``. The final export
    window is phase-scoped; the credit-phase tag alone must select records.
    """

    PHASE_START_NS = 1_783_362_364_086_542_600
    REQUESTS_END_NS = 1_783_362_364_094_539_300

    def _accumulator(self, mock_run) -> MetricsAccumulator:
        processor = MetricsAccumulator(mock_run)
        processor._tags_to_types = {RequestCountMetric.tag: MetricType.AGGREGATE}
        processor._metric_classes = {RequestCountMetric.tag: RequestCountMetric}
        return processor

    async def _ingest(
        self,
        processor: MetricsAccumulator,
        starts: list[int],
        phase: CreditPhase = CreditPhase.PROFILING,
    ) -> None:
        for i, start_ns in enumerate(starts):
            msg = create_metric_records_data(
                session_num=i,
                benchmark_phase=phase,
                request_start_ns=start_ns,
                request_end_ns=start_ns + 1_000_000,
                results=[{RequestCountMetric.tag: 1}],
            )
            await processor.process_record(msg)

    @pytest.mark.asyncio
    async def test_export_results_straggler_stamped_at_window_end_included(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """A record with start_ns == requests_end_ns counts in the phase export."""
        processor = self._accumulator(mock_run)
        starts = [
            self.PHASE_START_NS,  # coarse clock: same tick as phase start
            self.PHASE_START_NS + 120_000,
            self.PHASE_START_NS + 150_000,
            self.PHASE_START_NS + 200_000,
            self.REQUESTS_END_NS,  # coarse clock: same tick as phase end
        ]
        await self._ingest(processor, starts)

        summary = await processor.export_results(
            ExportContext(
                start_ns=self.PHASE_START_NS,
                end_ns=self.REQUESTS_END_NS,
                phase=CreditPhase.PROFILING,
            )
        )
        assert summary.results[RequestCountMetric.tag].avg == 5

    @pytest.mark.asyncio
    async def test_export_results_phase_tag_still_excludes_other_phases(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Dropping the time bounds must not leak other phases into the export."""
        processor = self._accumulator(mock_run)
        await self._ingest(
            processor, [self.PHASE_START_NS - 5_000_000], phase=CreditPhase.WARMUP
        )
        await self._ingest(processor, [self.PHASE_START_NS + 100_000])

        summary = await processor.export_results(
            ExportContext(
                start_ns=self.PHASE_START_NS,
                end_ns=self.REQUESTS_END_NS,
                phase=CreditPhase.PROFILING,
            )
        )
        assert summary.results[RequestCountMetric.tag].avg == 1

    @pytest.mark.asyncio
    async def test_phaseless_window_keeps_half_open_end_semantics(
        self, mock_metric_registry: Mock, mock_run
    ) -> None:
        """Realtime rolling windows (no phase) stay [start, end) — no overlap."""
        processor = self._accumulator(mock_run)
        starts = [
            self.PHASE_START_NS,
            self.PHASE_START_NS + 120_000,
            self.REQUESTS_END_NS,  # on the window end boundary: excluded
        ]
        await self._ingest(processor, starts)

        mask = processor._mask_for_export_context(
            ExportContext(start_ns=self.PHASE_START_NS, end_ns=self.REQUESTS_END_NS)
        )
        assert mask is not None
        assert int(mask.sum()) == 2
