# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase
from aiperf.common.messages import MetricRecordsData
from aiperf.common.models import MetricRecordMetadata
from aiperf.metrics.accumulator import MetricsAccumulator
from tests.unit.conftest import make_benchmark_run

NS_PER_S = 1_000_000_000
T0 = 1_780_000_000 * NS_PER_S


def _record(index: int, second: int, phase: CreditPhase) -> MetricRecordsData:
    start, end = T0 + second * NS_PER_S, T0 + (second + 1) * NS_PER_S
    return MetricRecordsData(
        metadata=MetricRecordMetadata(
            session_num=index,
            turn_index=0,
            request_start_ns=start,
            request_end_ns=end,
            worker_id="worker",
            record_processor_id="rp",
            benchmark_phase=phase,
        ),
        metrics={
            "request_latency": NS_PER_S,
            "request_count": 1,
            "output_sequence_length": 100,
            "input_sequence_length": 50,
            "min_request_timestamp": start,
            "max_response_timestamp": end,
        },
        error=None,
    )


def test_last_slice_ends_with_the_exported_phase() -> None:
    asyncio.run(_run_last_slice_ends_with_the_exported_phase())


async def _run_last_slice_ends_with_the_exported_phase() -> None:
    acc = MetricsAccumulator(
        make_benchmark_run(extra={"artifacts": {"slice_duration": 10}})
    )
    for second in range(15):
        await acc.process_record(_record(second, second, CreditPhase.PROFILING))
    for second in range(15, 40):
        await acc.process_record(_record(second, second, CreditPhase.WARMUP))

    summary = await acc.export_results(
        ExportContext(
            start_ns=T0, end_ns=T0 + 15 * NS_PER_S, phase=CreditPhase.PROFILING
        )
    )

    last = summary.timeslices[-1]
    assert (last.start_ns, last.end_ns) == (T0 + 10 * NS_PER_S, T0 + 15 * NS_PER_S)
    assert last.is_complete is False
    assert last.metric_results["request_throughput"].avg == 1.0
