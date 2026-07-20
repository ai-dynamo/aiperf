# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``generate_realtime_metrics`` fan-out over accumulators.

A persistently failing accumulator must not crash the realtime tick, but it
must leave a trail: the failure is logged (warning) with the accumulator's
class name so a stale dashboard/log block is diagnosable.
"""

import logging
from unittest.mock import Mock

import pytest

from aiperf.common.accumulator_protocols import SummaryContext
from aiperf.common.enums import CreditPhase
from aiperf.common.models import MetricResult
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.records.records_manager_processing import generate_realtime_metrics
from tests.unit.post_processors.conftest import (
    create_accumulator_with_metrics,
    create_metric_records_data,
)


def _metric(tag: str) -> MetricResult:
    return MetricResult(tag=tag, header=tag, unit="ms", avg=1.0)


class HealthyAccumulator:
    """Accumulator whose summarize returns a plain list of MetricResult."""

    async def summarize(self, ctx: SummaryContext | None = None) -> list[MetricResult]:
        return [_metric("time_to_first_token")]


class ExplodingAccumulator:
    """Accumulator whose summarize always raises."""

    async def summarize(self, ctx: SummaryContext | None = None) -> list[MetricResult]:
        raise RuntimeError("summarize blew up")


@pytest.mark.asyncio
async def test_generate_realtime_metrics_failing_accumulator_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.WARNING, logger="aiperf.records.records_manager_processing"
    )

    flat = await generate_realtime_metrics(
        [HealthyAccumulator(), ExplodingAccumulator()]
    )

    assert [m.tag for m in flat] == ["time_to_first_token"]
    assert "ExplodingAccumulator" in caplog.text
    assert "summarize blew up" in caplog.text


@pytest.mark.asyncio
async def test_generate_realtime_metrics_all_healthy_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(
        logging.WARNING, logger="aiperf.records.records_manager_processing"
    )

    flat = await generate_realtime_metrics([HealthyAccumulator()])

    assert [m.tag for m in flat] == ["time_to_first_token"]
    assert caplog.text == ""


@pytest.fixture
def _mock_registry_for_accumulator(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Minimal MetricRegistry stub so MetricsAccumulator construction does not
    pull in globally-registered derived metrics (mirrors
    post_processors.conftest.mock_metric_registry for the realtime path)."""
    mock_registry = Mock()
    mock_registry.tags_applicable_to.return_value = []
    mock_registry.create_dependency_order_for.return_value = []
    mock_registry.get_instance.return_value = Mock()
    mock_registry.all_classes.return_value = []
    mock_registry.all_tags.return_value = []
    monkeypatch.setattr(
        "aiperf.post_processors.base_metrics_processor.MetricRegistry",
        mock_registry,
    )
    return mock_registry


@pytest.mark.asyncio
async def test_generate_realtime_metrics_excludes_warmup_records(
    _mock_registry_for_accumulator: Mock, benchmark_run
) -> None:
    """End-to-end: generate_realtime_metrics is PROFILING-scoped, so a warmup
    record must not dilute the live MetricResult. Pre-fix, summarize() ran with
    no phase mask and averaged the warmup + profiling latencies together."""
    accumulator = create_accumulator_with_metrics(benchmark_run, RequestLatencyMetric)

    warmup = create_metric_records_data(
        session_num=0,
        benchmark_phase=CreditPhase.WARMUP,
        request_start_ns=1_000_000_000,
        request_end_ns=1_100_000_000,
        results=[{RequestLatencyMetric.tag: 100_000_000.0}],
    )
    profiling = create_metric_records_data(
        session_num=0,
        benchmark_phase=CreditPhase.PROFILING,
        request_start_ns=2_000_000_000,
        request_end_ns=2_200_000_000,
        results=[{RequestLatencyMetric.tag: 200_000_000.0}],
    )
    await accumulator.process_record(warmup)
    await accumulator.process_record(profiling)

    flat = await generate_realtime_metrics([accumulator])

    latency = next(m for m in flat if m.tag == RequestLatencyMetric.tag)
    assert latency.avg == pytest.approx(200.0)
