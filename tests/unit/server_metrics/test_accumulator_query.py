# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsAccumulator.process_record() and query_time_range()."""

from __future__ import annotations

import numpy as np
import pytest

from aiperf.common.accumulator_protocols import AccumulatorProtocol
from aiperf.common.models.server_metrics_models import ServerMetricsRecord
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator
from tests.unit.conftest import make_run_from_cli


def _make_server_metrics_record(timestamp_ns: int) -> ServerMetricsRecord:
    """Create a minimal ServerMetricsRecord with a given timestamp."""
    return ServerMetricsRecord(
        endpoint_url="http://localhost:8081/metrics",
        timestamp_ns=timestamp_ns,
        endpoint_latency_ns=5_000_000,
        metrics={},
    )


@pytest.fixture
def accumulator() -> ServerMetricsAccumulator:
    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            endpoint_type=EndpointType.CHAT,
            streaming=False,
        )
    )
    return ServerMetricsAccumulator(run=run)


class TestServerMetricsAccumulatorConformance:
    def test_satisfies_accumulator_protocol(
        self, accumulator: ServerMetricsAccumulator
    ) -> None:
        assert isinstance(accumulator, AccumulatorProtocol)


class TestProcessRecord:
    @pytest.mark.asyncio
    async def test_process_record_stores_timestamp_and_adds_to_hierarchy(
        self, accumulator: ServerMetricsAccumulator
    ) -> None:
        record = _make_server_metrics_record(1_000)
        await accumulator.process_record(record)

        assert len(accumulator._timestamps_ns) == 1
        assert accumulator._timestamps_ns[0] == 1_000


class TestQueryTimeRange:
    @pytest.mark.asyncio
    async def test_empty(self, accumulator: ServerMetricsAccumulator) -> None:
        mask = accumulator.query_time_range(0, 10_000)
        assert len(mask) == 0

    @pytest.mark.asyncio
    async def test_single_record_inside(
        self, accumulator: ServerMetricsAccumulator
    ) -> None:
        await accumulator.process_record(_make_server_metrics_record(5_000))
        mask = accumulator.query_time_range(0, 10_000)
        assert mask.sum() == 1

    @pytest.mark.asyncio
    async def test_single_record_outside(
        self, accumulator: ServerMetricsAccumulator
    ) -> None:
        await accumulator.process_record(_make_server_metrics_record(15_000))
        mask = accumulator.query_time_range(0, 10_000)
        assert mask.sum() == 0

    @pytest.mark.asyncio
    async def test_boundary_inclusive_start(
        self, accumulator: ServerMetricsAccumulator
    ) -> None:
        await accumulator.process_record(_make_server_metrics_record(1_000))
        mask = accumulator.query_time_range(1_000, 2_000)
        assert mask.sum() == 1

    @pytest.mark.asyncio
    async def test_boundary_exclusive_end(
        self, accumulator: ServerMetricsAccumulator
    ) -> None:
        await accumulator.process_record(_make_server_metrics_record(2_000))
        mask = accumulator.query_time_range(1_000, 2_000)
        assert mask.sum() == 0

    @pytest.mark.asyncio
    async def test_multiple_records_filtering(
        self, accumulator: ServerMetricsAccumulator
    ) -> None:
        timestamps = [100, 200, 300, 400, 500]
        for ts in timestamps:
            await accumulator.process_record(_make_server_metrics_record(ts))

        mask = accumulator.query_time_range(200, 400)
        assert mask.sum() == 2
        np.testing.assert_array_equal(np.where(mask)[0], [1, 2])

    @pytest.mark.asyncio
    async def test_equal_start_end_returns_empty(
        self, accumulator: ServerMetricsAccumulator
    ) -> None:
        await accumulator.process_record(_make_server_metrics_record(100))
        mask = accumulator.query_time_range(100, 100)
        assert mask.sum() == 0
