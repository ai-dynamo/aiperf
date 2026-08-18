# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete named-phase identity remains intact in manager-owned aggregation."""

import pytest

from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase, PrometheusMetricType
from aiperf.common.models.server_metrics_models import (
    MetricFamily,
    MetricSample,
    ServerMetricsRecord,
)
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import EndpointType
from aiperf.server_metrics.accumulator import ServerMetricsAccumulator
from tests.unit.conftest import make_run_from_cli


def _record(
    timestamp_ns: int,
    *,
    phase_index: int,
    profiling_index: int,
    phase_name: str,
) -> ServerMetricsRecord:
    return ServerMetricsRecord(
        endpoint_url="http://server:8000/metrics",
        timestamp_ns=timestamp_ns,
        benchmark_phase=CreditPhase.PROFILING,
        phase_index=phase_index,
        profiling_index=profiling_index,
        phase_name=phase_name,
        phase_kind="profiling",
        metrics={
            "vllm:num_requests_running": MetricFamily(
                type=PrometheusMetricType.GAUGE,
                description="running",
                samples=[MetricSample(value=float(timestamp_ns))],
            )
        },
    )


@pytest.mark.asyncio
async def test_export_results_contains_exact_named_phase_summaries() -> None:
    accumulator = ServerMetricsAccumulator(
        run=make_run_from_cli(
            CLIConfig(
                model_names=["model"],
                endpoint_type=EndpointType.CHAT,
                urls=["http://server:8000/v1/chat/completions"],
            )
        )
    )
    for record in (
        _record(10, phase_index=0, profiling_index=0, phase_name="baseline"),
        _record(20, phase_index=0, profiling_index=0, phase_name="baseline"),
        _record(30, phase_index=1, profiling_index=1, phase_name="main"),
        _record(40, phase_index=1, profiling_index=1, phase_name="main"),
    ):
        await accumulator.process_record(record)

    results = await accumulator.export_results(ExportContext(start_ns=10, end_ns=41))

    assert results is not None
    assert [result.phase_name for result in results.phase_results] == [
        "baseline",
        "main",
    ]
    assert [result.phase_index for result in results.phase_results] == [0, 1]
    assert all(result.endpoint_summaries for result in results.phase_results)
