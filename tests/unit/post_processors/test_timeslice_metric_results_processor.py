# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from tests.unit.post_processors.conftest import create_metric_records_message


class TestMetricsAccumulatorTimeslices:
    """Timeslice coverage for the accumulator-backed summary path."""

    def test_initialization_without_slice_duration_disables_timeslices(
        self, mock_run
    ) -> None:
        mock_run.cfg.artifacts.slice_duration = None
        accumulator = MetricsAccumulator(mock_run)

        assert accumulator._slice_duration_ns is None

    def test_initialization_with_slice_duration(self, mock_run) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        accumulator = MetricsAccumulator(mock_run)

        assert accumulator._slice_duration_ns == NANOS_PER_SECOND

    @pytest.mark.asyncio
    async def test_process_record_separates_by_timeslice(self, mock_run) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        accumulator = MetricsAccumulator(mock_run)

        await accumulator.process_record(
            create_metric_records_message(
                x_request_id="test-1",
                request_start_ns=int(0.5 * NANOS_PER_SECOND),
                request_end_ns=int(0.6 * NANOS_PER_SECOND),
                results=[{RequestLatencyMetric.tag: 42_000_000.0}],
            ).to_data()
        )
        await accumulator.process_record(
            create_metric_records_message(
                x_request_id="test-2",
                request_start_ns=int(1.5 * NANOS_PER_SECOND),
                request_end_ns=int(1.6 * NANOS_PER_SECOND),
                results=[{RequestLatencyMetric.tag: 84_000_000.0}],
            ).to_data()
        )

        timeslices = (await accumulator.summarize()).timeslices

        assert timeslices is not None
        assert len(timeslices) == 2
        assert timeslices[0].metric_results[RequestLatencyMetric.tag].avg == 42.0
        assert timeslices[1].metric_results[RequestLatencyMetric.tag].avg == 84.0

    @pytest.mark.asyncio
    async def test_process_record_accumulates_in_same_timeslice(self, mock_run) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        accumulator = MetricsAccumulator(mock_run)

        for idx, value in enumerate([10_000_000.0, 20_000_000.0]):
            await accumulator.process_record(
                create_metric_records_message(
                    x_request_id=f"test-{idx}",
                    request_start_ns=int((0.3 + idx * 0.4) * NANOS_PER_SECOND),
                    request_end_ns=int((0.35 + idx * 0.4) * NANOS_PER_SECOND),
                    results=[{RequestLatencyMetric.tag: value}],
                ).to_data()
            )

        timeslices = (await accumulator.summarize()).timeslices

        assert timeslices is not None
        assert len(timeslices) == 1
        result = timeslices[0].metric_results[RequestLatencyMetric.tag]
        assert result.count == 2
        assert result.avg == pytest.approx(15.0)

    @pytest.mark.asyncio
    async def test_aggregate_metric_per_timeslice(self, mock_run) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        accumulator = MetricsAccumulator(mock_run)

        records = [
            (0.5, 5),
            (0.7, 3),
            (1.5, 7),
        ]
        for idx, (start_s, count) in enumerate(records):
            await accumulator.process_record(
                create_metric_records_message(
                    x_request_id=f"test-{idx}",
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int((start_s + 0.1) * NANOS_PER_SECOND),
                    results=[{RequestCountMetric.tag: count}],
                ).to_data()
            )

        timeslices = (await accumulator.summarize()).timeslices

        assert timeslices is not None
        assert timeslices[0].metric_results[RequestCountMetric.tag].avg == 8
        assert timeslices[1].metric_results[RequestCountMetric.tag].avg == 7

    @pytest.mark.asyncio
    async def test_timeslice_boundary_conditions(self, mock_run) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        accumulator = MetricsAccumulator(mock_run)

        records = [
            (0.999, 1_000_000.0),
            (1.0, 2_000_000.0),
            (1.001, 3_000_000.0),
        ]
        for idx, (start_s, value) in enumerate(records):
            await accumulator.process_record(
                create_metric_records_message(
                    x_request_id=f"test-{idx}",
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int((start_s + 0.01) * NANOS_PER_SECOND),
                    results=[{RequestLatencyMetric.tag: value}],
                ).to_data()
            )

        timeslices = (await accumulator.summarize()).timeslices

        assert timeslices is not None
        assert len(timeslices) == 1
        result = timeslices[0].metric_results[RequestLatencyMetric.tag]
        assert result.count == 3
        assert result.avg == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_derived_metrics_are_computed_per_timeslice(self, mock_run) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        accumulator = MetricsAccumulator(mock_run)

        records = [
            (0.5, 1.5, 5),
            (1.5, 2.5, 10),
        ]
        for idx, (start_s, end_s, count) in enumerate(records):
            await accumulator.process_record(
                create_metric_records_message(
                    x_request_id=f"test-{idx}",
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int(end_s * NANOS_PER_SECOND),
                    results=[
                        {
                            RequestCountMetric.tag: count,
                            MinRequestTimestampMetric.tag: start_s * NANOS_PER_SECOND,
                            MaxResponseTimestampMetric.tag: end_s * NANOS_PER_SECOND,
                        }
                    ],
                ).to_data()
            )

        timeslices = (await accumulator.summarize()).timeslices

        assert timeslices is not None
        assert timeslices[0].metric_results[RequestThroughputMetric.tag].avg == 5.0
        assert timeslices[1].metric_results[RequestThroughputMetric.tag].avg == 10.0

    @pytest.mark.asyncio
    async def test_summarize_without_records_returns_no_timeslices(
        self, mock_run
    ) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        accumulator = MetricsAccumulator(mock_run)

        summary = await accumulator.summarize()

        assert summary.timeslices is None

    @pytest.mark.asyncio
    async def test_multiple_timeslices_with_different_slice_duration(
        self, mock_run
    ) -> None:
        mock_run.cfg.artifacts.slice_duration = 0.5
        accumulator = MetricsAccumulator(mock_run)

        for i in range(4):
            start_s = i * 0.5 + 0.25
            await accumulator.process_record(
                create_metric_records_message(
                    x_request_id=f"test-{i}",
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int((start_s + 0.05) * NANOS_PER_SECOND),
                    results=[{RequestLatencyMetric.tag: float(i * 1_000_000)}],
                ).to_data()
            )

        timeslices = (await accumulator.summarize()).timeslices

        assert timeslices is not None
        assert len(timeslices) == 4
        assert [
            ts.metric_results[RequestLatencyMetric.tag].avg for ts in timeslices
        ] == [0.0, 1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_trailing_partial_timeslice_is_flagged(self, mock_run) -> None:
        mock_run.cfg.artifacts.slice_duration = 1.0
        accumulator = MetricsAccumulator(mock_run)

        await accumulator.process_record(
            create_metric_records_message(
                x_request_id="test-1",
                request_start_ns=int(0.5 * NANOS_PER_SECOND),
                request_end_ns=int(0.6 * NANOS_PER_SECOND),
                results=[{RequestLatencyMetric.tag: 42_000_000.0}],
            ).to_data()
        )

        timeslices = (await accumulator.summarize()).timeslices

        assert timeslices is not None
        assert timeslices[0].is_complete is False
        assert timeslices[0].end_ns == int(0.6 * NANOS_PER_SECOND)
