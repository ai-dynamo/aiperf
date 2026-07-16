# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.exceptions import NoMetricValue
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.accumulator_models import AccumulatorMetricsSummary
from aiperf.metrics.metric_dicts import MetricResultsDict
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from tests.unit.post_processors.conftest import create_metric_records_message


class TestMetricsAccumulator:
    """Tests for the accumulator-backed metric summary engine."""

    def test_initialization(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)

        assert accumulator.record_count == 0
        assert accumulator.column_store.count == 0
        assert accumulator._network_rtt_ns is None
        assert isinstance(accumulator._derive_funcs, dict)
        assert isinstance(accumulator._tags_to_types, dict)
        assert isinstance(accumulator._metric_classes, dict)

    @pytest.mark.asyncio
    async def test_process_record_metric_accumulates_values(self, mock_run) -> None:
        accumulator = MetricsAccumulator(mock_run)

        for idx, value in enumerate([42_000_000.0, 84_000_000.0]):
            message = create_metric_records_message(
                x_request_id=f"test-{idx}",
                request_start_ns=1_000_000_000 + idx,
                request_end_ns=1_100_000_000 + idx,
                results=[{RequestLatencyMetric.tag: value}],
            )
            await accumulator.process_record(message.to_data())

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
        message = create_metric_records_message(
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

        await accumulator.process_record(message.to_data())

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
            message = create_metric_records_message(
                x_request_id=f"test-{idx}",
                request_start_ns=1_000_000_000 + idx,
                results=[{RequestCountMetric.tag: value}],
            )
            await accumulator.process_record(message.to_data())

        summary = await accumulator.summarize()

        assert summary.results[RequestCountMetric.tag].avg == 8

    @pytest.mark.asyncio
    async def test_derived_metrics_are_computed_from_accumulated_results(
        self, mock_run
    ) -> None:
        accumulator = MetricsAccumulator(mock_run)
        message = create_metric_records_message(
            x_request_id="test-1",
            results=[
                {
                    RequestCountMetric.tag: 100,
                    MinRequestTimestampMetric.tag: 0,
                    MaxResponseTimestampMetric.tag: 10 * NANOS_PER_SECOND,
                }
            ],
        )

        await accumulator.process_record(message.to_data())
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
            create_metric_records_message(
                x_request_id="test-1",
                results=[{RequestLatencyMetric.tag: 42_000_000.0}],
            ).to_data()
        )

        summary = await accumulator.summarize()

        assert isinstance(summary, AccumulatorMetricsSummary)
        assert summary.results[RequestLatencyMetric.tag].unit == "ms"
        assert summary.results[RequestLatencyMetric.tag].avg == pytest.approx(42.0)

