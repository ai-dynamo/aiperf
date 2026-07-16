# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from aiperf.accuracy.accumulator import AccuracyAccumulator
from aiperf.accuracy.models import AccuracyRecordsData, AccuracySummary
from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import PostProcessorDisabled
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_accumulator() -> AccuracyAccumulator:
    return AccuracyAccumulator(
        run=make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=False,
            accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
        )
    )


def _record(
    *,
    timestamp_ns: int,
    phase: CreditPhase = CreditPhase.PROFILING,
    task: str | None = "math",
    passed: bool = True,
    unparsed: bool = False,
    session_num: int = 0,
    grader_name: str = "exact_match",
) -> AccuracyRecordsData:
    return AccuracyRecordsData(
        session_num=session_num,
        worker_id="worker-0",
        benchmark_phase=phase,
        timestamp_ns=timestamp_ns,
        task=task,
        grader_name=grader_name,
        passed=passed,
        unparsed=unparsed,
        confidence=1.0,
        expected="A",
        actual="A",
        explanation="matched",
    )


async def _seed(acc: AccuracyAccumulator, records: list[AccuracyRecordsData]) -> None:
    for record in records:
        await acc.process_record(record)


@pytest.mark.asyncio
class TestAccuracyAccumulator:
    async def test_disabled_raises(self) -> None:
        with pytest.raises(PostProcessorDisabled):
            AccuracyAccumulator(
                run=make_benchmark_run(
                    model_names=["test-model"],
                    endpoint_type=EndpointType.COMPLETIONS,
                    streaming=False,
                )
            )

    async def test_export_results_profiling_overall_and_per_task(self) -> None:
        acc = _make_accumulator()
        await _seed(
            acc,
            [
                _record(timestamp_ns=10, task="math", passed=True),
                _record(timestamp_ns=20, task="math", passed=False, unparsed=True),
                _record(timestamp_ns=30, task="algebra", passed=True),
                _record(timestamp_ns=40, task="algebra", passed=True, unparsed=True),
                # WARMUP record must be excluded from PROFILING scope.
                _record(
                    timestamp_ns=5,
                    phase=CreditPhase.WARMUP,
                    task="math",
                    passed=False,
                ),
            ],
        )

        summary = await acc.export_results(ExportContext(phase=CreditPhase.PROFILING))

        assert isinstance(summary, AccuracySummary)
        assert summary.total_evaluated == 4
        assert summary.total_passed == 3
        assert summary.overall_unparsed == 2
        assert summary.accuracy_rate == pytest.approx(0.75)
        assert summary.grader_name == "exact_match"

        assert set(summary.per_task) == {"math", "algebra"}

        math = summary.per_task["math"]
        assert math.total == 2
        assert math.passed == 1
        assert math.unparsed == 1
        assert math.accuracy_rate == pytest.approx(0.5)
        assert math.unparsed_rate == pytest.approx(0.5)

        algebra = summary.per_task["algebra"]
        assert algebra.total == 2
        assert algebra.passed == 2
        assert algebra.unparsed == 1
        assert algebra.accuracy_rate == pytest.approx(1.0)
        assert algebra.unparsed_rate == pytest.approx(0.5)

    async def test_warmup_scope_isolated(self) -> None:
        acc = _make_accumulator()
        await _seed(
            acc,
            [
                _record(timestamp_ns=5, phase=CreditPhase.WARMUP, passed=True),
                _record(timestamp_ns=15, phase=CreditPhase.PROFILING, passed=False),
            ],
        )

        warmup = await acc.export_results(ExportContext(phase=CreditPhase.WARMUP))
        assert warmup is not None
        assert warmup.total_evaluated == 1
        assert warmup.total_passed == 1
        assert warmup.accuracy_rate == pytest.approx(1.0)

    async def test_phase_with_no_records_returns_none(self) -> None:
        acc = _make_accumulator()
        await _seed(
            acc,
            [_record(timestamp_ns=15, phase=CreditPhase.PROFILING)],
        )

        assert await acc.export_results(ExportContext(phase=CreditPhase.WARMUP)) is None

    async def test_empty_returns_none(self) -> None:
        acc = _make_accumulator()
        assert await acc.export_results(ExportContext(phase=None)) is None

    async def test_task_none_counts_overall_but_absent_from_per_task(self) -> None:
        acc = _make_accumulator()
        await _seed(
            acc,
            [
                _record(timestamp_ns=10, task=None, passed=True),
                _record(timestamp_ns=20, task="math", passed=False),
            ],
        )

        summary = await acc.export_results(ExportContext(phase=CreditPhase.PROFILING))
        assert summary is not None
        assert summary.total_evaluated == 2
        assert summary.total_passed == 1
        assert set(summary.per_task) == {"math"}

    async def test_summarize_is_phase_agnostic(self) -> None:
        acc = _make_accumulator()
        await _seed(
            acc,
            [
                _record(timestamp_ns=5, phase=CreditPhase.WARMUP, passed=True),
                _record(timestamp_ns=15, phase=CreditPhase.PROFILING, passed=False),
            ],
        )

        summary = await acc.summarize()
        assert summary is not None
        assert summary.total_evaluated == 2
        assert summary.total_passed == 1
        assert summary.accuracy_rate == pytest.approx(0.5)

    async def test_query_time_range_slices_by_timestamp(self) -> None:
        acc = _make_accumulator()
        await _seed(
            acc,
            [
                _record(timestamp_ns=10),
                _record(timestamp_ns=20),
                _record(timestamp_ns=30),
                _record(timestamp_ns=40),
            ],
        )

        sliced = acc.query_time_range(20, 40)
        assert [r.timestamp_ns for r in sliced] == [20, 30]
