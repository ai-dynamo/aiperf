# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MetricsAccumulator.process_record() and query_time_range()."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.common.accumulator_protocols import AccumulatorProtocol
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.metrics.accumulator import MetricsAccumulator
from tests.unit.post_processors.conftest import create_metric_metadata


def _make_accumulator(run) -> MetricsAccumulator:
    """Construct a MetricsAccumulator without metric-class side-effects."""
    return MetricsAccumulator(run)


def _make_record(request_start_ns: int, session_num: int = 0) -> MetricRecordsData:
    """Create a minimal MetricRecordsData with a given timestamp."""
    return MetricRecordsData(
        metadata=create_metric_metadata(
            session_num=session_num,
            request_start_ns=request_start_ns,
            request_end_ns=request_start_ns + 1_000_000,
        ),
        metrics={},
    )


@pytest.fixture
def processor(mock_run) -> MetricsAccumulator:
    return _make_accumulator(mock_run)


class TestMetricsAccumulatorProtocol:
    def test_satisfies_accumulator_protocol(
        self, processor: MetricsAccumulator
    ) -> None:
        assert isinstance(processor, AccumulatorProtocol)


class TestProcessRecord:
    @pytest.mark.asyncio
    async def test_process_record_stores_record(
        self, processor: MetricsAccumulator
    ) -> None:
        record = _make_record(1_000, session_num=0)
        await processor.process_record(record)
        assert processor.record_count == 1

    @pytest.mark.asyncio
    async def test_process_record_multiple(self, processor: MetricsAccumulator) -> None:
        records = [
            _make_record(ts, session_num=i)
            for i, ts in enumerate((1_000, 2_000, 3_000))
        ]
        for r in records:
            await processor.process_record(r)
        assert processor.record_count == 3


class TestQueryTimeRange:
    @pytest.mark.asyncio
    async def test_empty(self, processor: MetricsAccumulator) -> None:
        mask = processor.query_time_range(0, 10_000)
        assert len(mask) == 0

    @pytest.mark.asyncio
    async def test_single_record_inside(self, processor: MetricsAccumulator) -> None:
        await processor.process_record(_make_record(5_000, session_num=0))
        mask = processor.query_time_range(0, 10_000)
        assert mask.sum() == 1

    @pytest.mark.asyncio
    async def test_single_record_outside(self, processor: MetricsAccumulator) -> None:
        await processor.process_record(_make_record(15_000, session_num=0))
        mask = processor.query_time_range(0, 10_000)
        assert mask.sum() == 0

    @pytest.mark.asyncio
    async def test_boundary_inclusive_start(
        self, processor: MetricsAccumulator
    ) -> None:
        await processor.process_record(_make_record(1_000, session_num=0))
        # [1_000, 2_000) should include 1_000
        mask = processor.query_time_range(1_000, 2_000)
        assert mask.sum() == 1

    @pytest.mark.asyncio
    async def test_boundary_exclusive_end(self, processor: MetricsAccumulator) -> None:
        await processor.process_record(_make_record(2_000, session_num=0))
        # [1_000, 2_000) should NOT include 2_000
        mask = processor.query_time_range(1_000, 2_000)
        assert mask.sum() == 0

    @pytest.mark.asyncio
    async def test_multiple_records_filtering(
        self, processor: MetricsAccumulator
    ) -> None:
        timestamps = [100, 200, 300, 400, 500]
        for i, ts in enumerate(timestamps):
            await processor.process_record(_make_record(ts, session_num=i))

        mask = processor.query_time_range(200, 400)
        assert mask.sum() == 2
        indices = np.where(mask)[0]
        np.testing.assert_array_equal(indices, [1, 2])

    @pytest.mark.asyncio
    async def test_equal_start_end_returns_empty(
        self, processor: MetricsAccumulator
    ) -> None:
        await processor.process_record(_make_record(100, session_num=0))
        mask = processor.query_time_range(100, 100)
        assert mask.sum() == 0
