# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``generate_realtime_metrics`` fan-out over accumulators.

A persistently failing accumulator must not crash the realtime tick, but it
must leave a trail: the failure is logged (warning) with the accumulator's
class name so a stale dashboard/log block is diagnosable.
"""

import logging
from unittest.mock import Mock, patch

import pytest

from aiperf.common.accumulator_protocols import ExportContext, SummaryContext
from aiperf.common.enums import CreditPhase
from aiperf.common.models import MetricResult
from aiperf.common.models.credit_models import PhaseRecordsStats
from aiperf.metrics.accumulator_sweeps import compute_sweep_curves
from aiperf.metrics.types.inter_chunk_latency_metric import InterChunkLatencyMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.plugin.enums import UIType
from aiperf.records.records_manager import _render_realtime_block
from aiperf.records.records_manager_processing import generate_realtime_metrics
from tests.unit.post_processors.conftest import (
    create_accumulator_with_metrics,
    create_metric_metadata,
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


@pytest.mark.asyncio
async def test_generate_realtime_metrics_can_scope_to_phase_index(
    _mock_registry_for_accumulator: Mock, benchmark_run
) -> None:
    accumulator = create_accumulator_with_metrics(benchmark_run, RequestLatencyMetric)

    first_metadata = create_metric_metadata(
        session_num=0,
        benchmark_phase=CreditPhase.PROFILING,
        request_start_ns=1_000_000_000,
        request_end_ns=1_100_000_000,
    ).model_copy(update={"phase_index": 0})
    second_metadata = create_metric_metadata(
        session_num=0,
        benchmark_phase=CreditPhase.PROFILING,
        request_start_ns=2_000_000_000,
        request_end_ns=2_300_000_000,
    ).model_copy(update={"phase_index": 2})
    first = create_metric_records_data(
        metadata=first_metadata,
        results=[{RequestLatencyMetric.tag: 100_000_000.0}],
    )
    second = create_metric_records_data(
        metadata=second_metadata,
        results=[{RequestLatencyMetric.tag: 300_000_000.0}],
    )
    await accumulator.process_record(first)
    await accumulator.process_record(second)

    flat = await generate_realtime_metrics([accumulator], phase_index=2)

    latency = next(m for m in flat if m.tag == RequestLatencyMetric.tag)
    assert latency.avg == pytest.approx(300.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase_index", [None, 2])
@pytest.mark.parametrize("scenario", [None, "inferencex-agentx-mvp"])
@pytest.mark.parametrize("ui", [UIType.NONE, UIType.SIMPLE, UIType.DASHBOARD])
async def test_live_metrics_skip_unused_work_and_preserve_final_export(
    _mock_registry_for_accumulator: Mock,
    benchmark_run,
    phase_index: int | None,
    scenario: str | None,
    ui: UIType,
) -> None:
    benchmark_run.cfg.artifacts.slice_duration = 1.0
    benchmark_run.cfg.scenario = scenario
    benchmark_run.cfg.runtime.ui = ui
    accumulator = create_accumulator_with_metrics(
        benchmark_run, RequestLatencyMetric, InterChunkLatencyMetric
    )
    accumulator.set_network_rtt_ns(1_000_000.0)
    for idx, phase in enumerate(
        [CreditPhase.WARMUP, CreditPhase.PROFILING, CreditPhase.PROFILING]
    ):
        start_ns = (idx + 1) * 2_000_000_000
        metadata = create_metric_metadata(
            session_num=idx,
            benchmark_phase=phase,
            request_start_ns=start_ns,
            request_end_ns=start_ns + 600_000_000,
        ).model_copy(update={"phase_index": idx})
        await accumulator.process_record(
            create_metric_records_data(
                metadata=metadata,
                results=[
                    {
                        RequestLatencyMetric.tag: 600_000_000.0,
                        InterChunkLatencyMetric.tag: [300_000_000.0, 100_000_000.0],
                    }
                ],
            )
        )

    ctx = ExportContext(phase=CreditPhase.PROFILING, phase_index=phase_index)
    before = await accumulator.export_results(ctx)
    assert before.timeslices
    assert "effective_concurrency" in before.results
    assert "network_adjusted_request_latency" in before.results

    # Failing at the expensive entry point proves it is skipped, not discarded.
    with (
        patch.object(
            accumulator,
            "_compute_timeslices",
            side_effect=AssertionError("live timeslices"),
        ) as timeslices,
        patch(
            "aiperf.metrics.accumulator.compute_sweep_curves",
            wraps=compute_sweep_curves,
        ) as sweeps,
    ):
        live = await generate_realtime_metrics([accumulator], phase_index=phase_index)
        timeslices.assert_not_called()
    expected = before.results.copy()
    if ui != UIType.DASHBOARD:
        sweeps.assert_not_called()
        sweep_tags = {
            tag
            for tag in expected
            if (
                tag.startswith(("effective_", "active_")) and tag != "effective_latency"
            )
            or tag == "tokens_in_flight"
        }
        assert len(sweep_tags) == 14
        expected = {
            tag: value for tag, value in expected.items() if tag not in sweep_tags
        }
    else:
        sweeps.assert_called_once()
    assert {metric.tag: metric for metric in live} == expected
    stats = PhaseRecordsStats(
        phase=CreditPhase.PROFILING,
        start_ns=2_000_000_000,
        records_end_ns=7_000_000_000,
        success_records=2,
    )
    assert _render_realtime_block(live, stats, None) == _render_realtime_block(
        list(before.results.values()), stats, None
    )

    after = await accumulator.export_results(ctx)
    assert after == before
