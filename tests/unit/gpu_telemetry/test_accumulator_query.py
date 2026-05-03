# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GPUTelemetryAccumulator.process_record() and query_time_range()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from aiperf.common.accumulator_protocols import AccumulatorProtocol
from aiperf.config import AIPerfConfig, BenchmarkRun
from aiperf.gpu_telemetry.accumulator import GPUTelemetryAccumulator
from aiperf.plugin.enums import EndpointType
from tests.unit.post_processors.conftest import make_telemetry_record


def _make_run(config: AIPerfConfig) -> BenchmarkRun:
    return BenchmarkRun(benchmark_id="test", cfg=config, artifact_dir=Path("/tmp/test"))


@pytest.fixture
def accumulator(mock_metric_registry) -> GPUTelemetryAccumulator:
    config = AIPerfConfig(
        models=["test-model"],
        endpoint={
            "urls": ["http://localhost:8000/v1/chat/completions"],
            "type": EndpointType.CHAT,
            "streaming": False,
        },
        datasets=[
            {
                "name": "default",
                "type": "synthetic",
                "entries": 100,
                "prompts": {"isl": 128, "osl": 64},
            }
        ],
        phases=[
            {"name": "default", "type": "concurrency", "requests": 10, "concurrency": 1}
        ],
    )
    mock_pub = Mock()
    mock_pub.publish = AsyncMock()
    return GPUTelemetryAccumulator(
        run=_make_run(config),
        pub_client=mock_pub,
    )


class TestGPUTelemetryAccumulatorProtocol:
    def test_satisfies_accumulator_protocol(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        assert isinstance(accumulator, AccumulatorProtocol)


class TestProcessRecord:
    @pytest.mark.asyncio
    async def test_process_record_stores_timestamp_and_adds_to_hierarchy(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        record = make_telemetry_record(timestamp_ns=1_000)
        await accumulator.process_record(record)

        assert len(accumulator._timestamps_ns) == 1
        assert accumulator._timestamps_ns[0] == 1_000
        assert len(accumulator._hierarchy.dcgm_endpoints) > 0


class TestQueryTimeRange:
    @pytest.mark.asyncio
    async def test_empty(self, accumulator: GPUTelemetryAccumulator) -> None:
        mask = accumulator.query_time_range(0, 10_000)
        assert len(mask) == 0

    @pytest.mark.asyncio
    async def test_single_record_inside(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        await accumulator.process_record(make_telemetry_record(timestamp_ns=5_000))
        mask = accumulator.query_time_range(0, 10_000)
        assert mask.sum() == 1

    @pytest.mark.asyncio
    async def test_single_record_outside(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        await accumulator.process_record(make_telemetry_record(timestamp_ns=15_000))
        mask = accumulator.query_time_range(0, 10_000)
        assert mask.sum() == 0

    @pytest.mark.asyncio
    async def test_boundary_inclusive_start(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        await accumulator.process_record(make_telemetry_record(timestamp_ns=1_000))
        mask = accumulator.query_time_range(1_000, 2_000)
        assert mask.sum() == 1

    @pytest.mark.asyncio
    async def test_boundary_exclusive_end(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        await accumulator.process_record(make_telemetry_record(timestamp_ns=2_000))
        mask = accumulator.query_time_range(1_000, 2_000)
        assert mask.sum() == 0

    @pytest.mark.asyncio
    async def test_multiple_records_filtering(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        timestamps = [100, 200, 300, 400, 500]
        for ts in timestamps:
            await accumulator.process_record(make_telemetry_record(timestamp_ns=ts))

        mask = accumulator.query_time_range(200, 400)
        assert mask.sum() == 2
        np.testing.assert_array_equal(np.where(mask)[0], [1, 2])

    @pytest.mark.asyncio
    async def test_equal_start_end_returns_empty(
        self, accumulator: GPUTelemetryAccumulator
    ) -> None:
        await accumulator.process_record(make_telemetry_record(timestamp_ns=100))
        mask = accumulator.query_time_range(100, 100)
        assert mask.sum() == 0
