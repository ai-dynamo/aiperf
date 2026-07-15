# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.accuracy.accuracy_results_processor import AccuracyResultsProcessor
from aiperf.common.accumulator_protocols import ExportContext
from aiperf.common.enums import CreditPhase
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_processor() -> AccuracyResultsProcessor:
    return AccuracyResultsProcessor(
        run=make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=False,
            accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
        )
    )


def _seed(
    processor: AccuracyResultsProcessor,
    *,
    phase: CreditPhase = CreditPhase.PROFILING,
    overall_total: int = 0,
    overall_correct: int = 0,
    overall_unparsed: int = 0,
    task_total: dict[str, int] | None = None,
    task_correct: dict[str, int] | None = None,
    task_unparsed: dict[str, int] | None = None,
) -> None:
    """Seed the per-phase counters for a single phase."""
    processor._overall_total[phase] = overall_total
    processor._overall_correct[phase] = overall_correct
    processor._overall_unparsed[phase] = overall_unparsed
    for task, count in (task_total or {}).items():
        processor._task_total[phase][task] = count
    for task, count in (task_correct or {}).items():
        processor._task_correct[phase][task] = count
    for task, count in (task_unparsed or {}).items():
        processor._task_unparsed[phase][task] = count


@pytest.mark.asyncio
class TestAccuracyResultsProcessorSummarize:
    async def test_empty_returns_no_results(self) -> None:
        processor = _make_processor()
        results = await processor.summarize()
        assert results == []

    async def test_overall_metric_values(self) -> None:
        processor = _make_processor()
        _seed(processor, overall_total=10, overall_correct=7)

        results = await processor.summarize()

        overall = next(r for r in results if r.tag == "accuracy.overall")
        assert overall.current == pytest.approx(0.7)
        assert overall.count == 10
        assert overall.sum == 7
        assert overall.unit == "ratio"

    async def test_task_metrics_sorted_alphabetically(self) -> None:
        processor = _make_processor()
        _seed(
            processor,
            overall_total=4,
            overall_correct=3,
            task_total={"zebra": 2, "algebra": 2},
            task_correct={"zebra": 1, "algebra": 2},
        )

        results = await processor.summarize()
        task_results = [r for r in results if r.tag.startswith("accuracy.task.")]

        assert task_results[0].tag == "accuracy.task.algebra"
        assert task_results[1].tag == "accuracy.task.zebra"

    async def test_task_metric_accuracy_calculation(self) -> None:
        processor = _make_processor()
        _seed(
            processor,
            overall_total=5,
            overall_correct=3,
            task_total={"math": 5},
            task_correct={"math": 3},
        )

        results = await processor.summarize()

        task = next(r for r in results if r.tag == "accuracy.task.math")
        assert task.current == pytest.approx(0.6)
        assert task.count == 5
        assert task.sum == 3
        assert task.header == "Accuracy (math)"

    async def test_overall_not_emitted_when_no_results_processed(self) -> None:
        processor = _make_processor()
        _seed(processor, task_total={"math": 3}, task_correct={"math": 2})

        results = await processor.summarize()

        tags = [r.tag for r in results]
        assert "accuracy.overall" not in tags
        assert "accuracy.unparsed" not in tags
        assert "accuracy.task.math" in tags

    async def test_multiple_tasks_each_get_own_metric(self) -> None:
        processor = _make_processor()
        _seed(
            processor,
            overall_total=6,
            overall_correct=4,
            task_total={"history": 2, "biology": 2, "physics": 2},
            task_correct={"history": 1, "biology": 1, "physics": 1},
        )

        results = await processor.summarize()
        task_tags = {r.tag for r in results if r.tag.startswith("accuracy.task.")}

        assert task_tags == {
            "accuracy.task.history",
            "accuracy.task.biology",
            "accuracy.task.physics",
        }

    async def test_unparsed_overall_emitted_when_records_processed(self) -> None:
        processor = _make_processor()
        _seed(processor, overall_total=10, overall_correct=7, overall_unparsed=3)

        results = await processor.summarize()

        unparsed = next(r for r in results if r.tag == "accuracy.unparsed")
        assert unparsed.sum == 3
        assert unparsed.count == 10
        assert unparsed.current == pytest.approx(0.3)

    async def test_unparsed_per_task_emitted(self) -> None:
        processor = _make_processor()
        _seed(
            processor,
            overall_total=5,
            overall_correct=3,
            task_total={"math": 5},
            task_correct={"math": 3},
            task_unparsed={"math": 2},
        )

        results = await processor.summarize()

        unparsed_task = next(
            r for r in results if r.tag == "accuracy.unparsed.task.math"
        )
        assert unparsed_task.sum == 2
        assert unparsed_task.count == 5
        assert unparsed_task.current == pytest.approx(0.4)

    async def test_unparsed_zero_when_all_conforming(self) -> None:
        processor = _make_processor()
        _seed(
            processor,
            overall_total=5,
            overall_correct=5,
            task_total={"math": 5},
            task_correct={"math": 5},
        )

        results = await processor.summarize()

        unparsed = next(r for r in results if r.tag == "accuracy.unparsed")
        assert unparsed.sum == 0
        assert unparsed.current == pytest.approx(0.0)

    async def test_export_results_scopes_to_phase(self) -> None:
        """export_results(ctx) must isolate the requested phase so warmup grades
        do not leak into the profiling accuracy summary (and vice versa)."""
        processor = _make_processor()
        # One correct warmup record, one incorrect profiling record.
        _seed(processor, phase=CreditPhase.WARMUP, overall_total=1, overall_correct=1)
        _seed(
            processor, phase=CreditPhase.PROFILING, overall_total=1, overall_correct=0
        )

        profiling = await processor.export_results(
            ExportContext(phase=CreditPhase.PROFILING)
        )
        overall = next(r for r in profiling if r.tag == "accuracy.overall")
        assert overall.count == 1
        assert overall.current == pytest.approx(0.0)

        warmup = await processor.export_results(ExportContext(phase=CreditPhase.WARMUP))
        overall = next(r for r in warmup if r.tag == "accuracy.overall")
        assert overall.count == 1
        assert overall.current == pytest.approx(1.0)

        # summarize() (phase-agnostic) still sees both records combined.
        combined = await processor.summarize()
        overall = next(r for r in combined if r.tag == "accuracy.overall")
        assert overall.count == 2
        assert overall.current == pytest.approx(0.5)
