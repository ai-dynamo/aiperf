# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import uuid

import pytest

from aiperf.common.constants import NANOS_PER_MILLIS, NANOS_PER_SECOND
from aiperf.metrics.accumulator import MetricsAccumulator
from aiperf.metrics.types.e2e_normalized_interactivity_metrics import (
    E2ENormalizedInteractivityP75Metric,
    E2ENormalizedInteractivityP90Metric,
)
from aiperf.metrics.types.max_response_metric import MaxResponseTimestampMetric
from aiperf.metrics.types.min_request_metric import MinRequestTimestampMetric
from aiperf.metrics.types.replay_sched_lag_metrics import (
    ReplaySchedDegradedMetric,
    ReplaySchedLagP50Metric,
    ReplaySchedLagP90Metric,
    ReplaySchedLagP99Metric,
    ReplaySendScheduleOffsetMetric,
)
from aiperf.metrics.types.request_count_metric import RequestCountMetric
from aiperf.metrics.types.request_latency_metric import RequestLatencyMetric
from aiperf.metrics.types.request_throughput_metric import RequestThroughputMetric
from tests.unit.post_processors.conftest import create_metric_records_data


@pytest.fixture
def fixed_schedule_slice_run():
    """Real (unmocked) fixed-schedule BenchmarkRun with --slice-duration set,
    so FIXED_SCHEDULE_ONLY metrics survive the applicability filters."""
    from aiperf.config import BenchmarkConfig, BenchmarkRun
    from aiperf.plugin.enums import EndpointType

    cfg = BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {
                "type": EndpointType.COMPLETIONS,
                "urls": ["http://localhost:8000/v1"],
                "streaming": False,
            },
            "datasets": [{"name": "default", "type": "synthetic"}],
            "phases": [{"name": "profiling", "type": "fixed_schedule"}],
            "artifacts": {"slice_duration": 1.0},
        }
    )
    return BenchmarkRun(
        benchmark_id=uuid.uuid4().hex,
        cfg=cfg,
        artifact_dir=cfg.artifacts.dir,
        random_seed=None,
        variables={},
    )


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
            create_metric_records_data(
                x_request_id="test-1",
                request_start_ns=int(0.5 * NANOS_PER_SECOND),
                request_end_ns=int(0.6 * NANOS_PER_SECOND),
                results=[{RequestLatencyMetric.tag: 42_000_000.0}],
            )
        )
        await accumulator.process_record(
            create_metric_records_data(
                x_request_id="test-2",
                request_start_ns=int(1.5 * NANOS_PER_SECOND),
                request_end_ns=int(1.6 * NANOS_PER_SECOND),
                results=[{RequestLatencyMetric.tag: 84_000_000.0}],
            )
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
                create_metric_records_data(
                    x_request_id=f"test-{idx}",
                    request_start_ns=int((0.3 + idx * 0.4) * NANOS_PER_SECOND),
                    request_end_ns=int((0.35 + idx * 0.4) * NANOS_PER_SECOND),
                    results=[{RequestLatencyMetric.tag: value}],
                )
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
                create_metric_records_data(
                    x_request_id=f"test-{idx}",
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int((start_s + 0.1) * NANOS_PER_SECOND),
                    results=[{RequestCountMetric.tag: count}],
                )
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
                create_metric_records_data(
                    x_request_id=f"test-{idx}",
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int((start_s + 0.01) * NANOS_PER_SECOND),
                    results=[{RequestLatencyMetric.tag: value}],
                )
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
                create_metric_records_data(
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
                )
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
                create_metric_records_data(
                    x_request_id=f"test-{i}",
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int((start_s + 0.05) * NANOS_PER_SECOND),
                    results=[{RequestLatencyMetric.tag: float(i * 1_000_000)}],
                )
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
            create_metric_records_data(
                x_request_id="test-1",
                request_start_ns=int(0.5 * NANOS_PER_SECOND),
                request_end_ns=int(0.6 * NANOS_PER_SECOND),
                results=[{RequestLatencyMetric.tag: 42_000_000.0}],
            )
        )

        timeslices = (await accumulator.summarize()).timeslices

        assert timeslices is not None
        assert timeslices[0].is_complete is False
        assert timeslices[0].end_ns == int(0.6 * NANOS_PER_SECOND)


class TestMetricsAccumulatorRunScopedDerivedMetrics:
    """The replay send-lag family is anchored to the run-global minimum offset
    (``timeslice_derivable = False``); re-deriving it per slice would re-anchor
    each slice at its own minimum and erase cumulative schedule drift, so the
    accumulator excludes it from per-slice derivation while keeping it in the
    overall summary."""

    def test_run_scoped_tags_excluded_from_timeslice_derivation(
        self, fixed_schedule_slice_run
    ) -> None:
        accumulator = MetricsAccumulator(fixed_schedule_slice_run)

        run_scoped_tags = {
            ReplaySchedLagP50Metric.tag,
            ReplaySchedLagP90Metric.tag,
            ReplaySchedLagP99Metric.tag,
            ReplaySchedDegradedMetric.tag,
            E2ENormalizedInteractivityP90Metric.tag,
            E2ENormalizedInteractivityP75Metric.tag,
        }
        # The run-scoped family derives at run level but is skipped per slice.
        assert run_scoped_tags <= set(accumulator._derive_funcs)
        assert accumulator._non_timeslice_derived_tags == run_scoped_tags

    @pytest.mark.asyncio
    async def test_run_scoped_tags_never_derived_in_timeslice_results(
        self, fixed_schedule_slice_run
    ) -> None:
        accumulator = MetricsAccumulator(fixed_schedule_slice_run)

        # Two records straddling the 1s slice boundary so timeslices are built.
        for i, start_s in enumerate((0.2, 1.2)):
            await accumulator.process_record(
                create_metric_records_data(
                    x_request_id=f"req-{i}",
                    request_start_ns=int(start_s * NANOS_PER_SECOND),
                    request_end_ns=int((start_s + 0.05) * NANOS_PER_SECOND),
                    results=[
                        {ReplaySendScheduleOffsetMetric.tag: i * NANOS_PER_MILLIS}
                    ],
                )
            )

        summary = await accumulator.summarize()

        assert summary.timeslices is not None
        assert len(summary.timeslices) >= 1
        for ts in summary.timeslices:
            assert ReplaySchedLagP50Metric.tag not in ts.metric_results
            assert ReplaySchedLagP90Metric.tag not in ts.metric_results
            assert ReplaySchedLagP99Metric.tag not in ts.metric_results
            assert ReplaySchedDegradedMetric.tag not in ts.metric_results
