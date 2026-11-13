# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import MetricRecordMetadata
from aiperf.metrics.accumulator import MetricsAccumulator
from tests.unit.conftest import make_benchmark_run

NS_PER_S = 1_000_000_000
T0 = 1_780_000_000 * NS_PER_S


def _record(index: int, itl: list[int]) -> MetricRecordsData:
    start = T0 + index * NS_PER_S
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=index,
            turn_index=0,
            request_start_ns=start,
            request_end_ns=start + NS_PER_S,
            worker_id="worker",
            record_processor_id="rp",
            benchmark_phase=CreditPhase.PROFILING,
        ),
        metrics={
            "request_latency": NS_PER_S,
            "request_count": 1,
            "output_sequence_length": len(itl) + 1,
            "inter_token_latency": itl,
        },
        error=None,
    )


async def _ingest(n_records: int) -> MetricsAccumulator:
    acc = MetricsAccumulator(make_benchmark_run())
    rng = np.random.default_rng(7)
    for index in range(n_records):
        itl = rng.integers(1, 100, size=int(rng.integers(0, 6))).tolist()
        await acc.process_record(_record(index, itl))
    return acc


@pytest.mark.parametrize(
    "n_bins",
    [
        param(5, id="int16_keys"),
        param(40_000, id="int32_keys"),
    ],
)  # fmt: skip
def test_slice_ragged_index_matches_per_slice_mask(n_bins: int) -> None:
    acc = asyncio.run(_ingest(30))
    rng = np.random.default_rng(3)
    # Records in no slice (-1) and a record past the index's range must drop out.
    record_bin = rng.integers(-1, min(n_bins, 6), size=29).astype(np.int32)

    index = acc._build_slice_ragged_index(record_bin, n_bins)

    backend = acc.column_store.ragged("inter_token_latency")
    order, bounds = index["inter_token_latency"]
    assert order.dtype == np.int32
    assert len(bounds) == n_bins + 1
    for slice_idx in range(n_bins):
        mask = np.zeros(acc.column_store.count, dtype=bool)
        mask[:29] = record_bin == slice_idx
        expected = backend.get_values_for_mask(mask)
        actual = backend.values[order[bounds[slice_idx] : bounds[slice_idx + 1]]]
        assert np.array_equal(actual, expected)
