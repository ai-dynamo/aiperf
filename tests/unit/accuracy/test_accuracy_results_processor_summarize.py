# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.accuracy.accuracy_results_processor import AccuracyResultsProcessor
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType


def _make_processor() -> AccuracyResultsProcessor:
    user_config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.MMLU),
    )
    return AccuracyResultsProcessor(user_config=user_config)


@pytest.mark.asyncio
class TestAccuracyResultsProcessorSummarize:
    async def test_empty_returns_no_results(self) -> None:
        processor = _make_processor()
        results = await processor.summarize()
        assert results == []

    async def test_overall_metric_values(self) -> None:
        processor = _make_processor()
        processor._overall_total = 10
        processor._overall_correct = 7

        results = await processor.summarize()

        overall = next(r for r in results if r.tag == "accuracy.overall")
        assert overall.current == pytest.approx(0.7)
        assert overall.count == 10
        assert overall.sum == 7
        assert overall.unit == "ratio"

    async def test_task_metrics_sorted_alphabetically(self) -> None:
        processor = _make_processor()
        processor._overall_total = 4
        processor._overall_correct = 3
        processor._task_total["zebra"] = 2
        processor._task_total["algebra"] = 2
        processor._task_correct["zebra"] = 1
        processor._task_correct["algebra"] = 2

        results = await processor.summarize()
        task_results = [r for r in results if r.tag.startswith("accuracy.task.")]

        assert task_results[0].tag == "accuracy.task.algebra"
        assert task_results[1].tag == "accuracy.task.zebra"

    async def test_task_metric_accuracy_calculation(self) -> None:
        processor = _make_processor()
        processor._overall_total = 5
        processor._overall_correct = 3
        processor._task_total["math"] = 5
        processor._task_correct["math"] = 3

        results = await processor.summarize()

        task = next(r for r in results if r.tag == "accuracy.task.math")
        assert task.current == pytest.approx(0.6)
        assert task.count == 5
        assert task.sum == 3
        assert task.header == "Accuracy (math)"

    async def test_overall_not_emitted_when_no_results_processed(self) -> None:
        processor = _make_processor()
        processor._task_total["math"] = 3
        processor._task_correct["math"] = 2

        results = await processor.summarize()

        tags = [r.tag for r in results]
        assert "accuracy.overall" not in tags
        assert "accuracy.unparsed" not in tags
        assert "accuracy.task.math" in tags

    async def test_multiple_tasks_each_get_own_metric(self) -> None:
        processor = _make_processor()
        processor._overall_total = 6
        processor._overall_correct = 4
        for task in ("history", "biology", "physics"):
            processor._task_total[task] = 2
            processor._task_correct[task] = 1

        results = await processor.summarize()
        task_tags = {r.tag for r in results if r.tag.startswith("accuracy.task.")}

        assert task_tags == {
            "accuracy.task.history",
            "accuracy.task.biology",
            "accuracy.task.physics",
        }

    async def test_unparsed_overall_emitted_when_records_processed(self) -> None:
        processor = _make_processor()
        processor._overall_total = 10
        processor._overall_correct = 7
        processor._overall_unparsed = 3

        results = await processor.summarize()

        unparsed = next(r for r in results if r.tag == "accuracy.unparsed")
        assert unparsed.sum == 3
        assert unparsed.count == 10
        assert unparsed.current == pytest.approx(0.3)

    async def test_unparsed_per_task_emitted(self) -> None:
        processor = _make_processor()
        processor._overall_total = 5
        processor._overall_correct = 3
        processor._task_total["math"] = 5
        processor._task_correct["math"] = 3
        processor._task_unparsed["math"] = 2

        results = await processor.summarize()

        unparsed_task = next(
            r for r in results if r.tag == "accuracy.unparsed.task.math"
        )
        assert unparsed_task.sum == 2
        assert unparsed_task.count == 5
        assert unparsed_task.current == pytest.approx(0.4)

    async def test_unparsed_zero_when_all_conforming(self) -> None:
        processor = _make_processor()
        processor._overall_total = 5
        processor._overall_correct = 5
        processor._task_total["math"] = 5
        processor._task_correct["math"] = 5

        results = await processor.summarize()

        unparsed = next(r for r in results if r.tag == "accuracy.unparsed")
        assert unparsed.sum == 0
        assert unparsed.current == pytest.approx(0.0)
