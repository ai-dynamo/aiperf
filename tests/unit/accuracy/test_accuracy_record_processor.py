# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.accuracy_results_processor import AccuracyResultsProcessor
from aiperf.accuracy.models import BenchmarkProblem, GradingResult
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.post_processors.conftest import create_metric_metadata


def _make_user_config() -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.MMLU),
    )


def _make_processor(monkeypatch, user_config: UserConfig) -> AccuracyRecordProcessor:
    mock_grader_cls = MagicMock()
    mock_grader_cls.return_value = MagicMock()

    # Only grader lookups remain in accuracy_record_processor; benchmark loading
    # moved to benchmark_loader and is bypassed by setting processor.problems directly.
    monkeypatch.setattr(
        "aiperf.accuracy.accuracy_record_processor.plugins.get_class",
        lambda plugin_type, name: mock_grader_cls,
    )
    monkeypatch.setattr(
        "aiperf.accuracy.accuracy_record_processor.plugins.get_metadata",
        lambda *_args, **_kwargs: {"default_grader": "multiple_choice"},
    )

    return AccuracyRecordProcessor(service_id="test", user_config=user_config)


def _make_problem(ground_truth: str = "A", task: str = "test_task") -> BenchmarkProblem:
    return BenchmarkProblem(
        prompt="Which is correct?",
        ground_truth=ground_truth,
        task=task,
    )


@pytest.mark.asyncio
class TestAccuracyRecordProcessorSessionBounds:
    async def test_process_record_wraps_when_session_num_exceeds_dataset(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        """session_num >= dataset size wraps via modulo so the correct problem is graded."""
        user_config = _make_user_config()
        processor = _make_processor(monkeypatch, user_config)
        processor.problems = [_make_problem(ground_truth="A")]

        grading_result = GradingResult(
            correct=True,
            confidence=1.0,
            reasoning="Correct",
            extracted_answer="A",
            ground_truth="A",
        )
        processor.grader.grade = AsyncMock(return_value=grading_result)

        # session_num=1 wraps to index 0 (the only problem)
        metadata = create_metric_metadata(session_num=1)
        result = await processor.process_record(sample_parsed_record, metadata)

        assert result["accuracy.correct"] == 1.0
        processor.grader.grade.assert_awaited_once_with("Hello world", "A")

    async def test_process_record_wraps_to_correct_problem(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        """With N problems, session_num=N+1 grades problem at index 1."""
        user_config = _make_user_config()
        processor = _make_processor(monkeypatch, user_config)
        processor.problems = [
            _make_problem(ground_truth="A"),
            _make_problem(ground_truth="B"),
            _make_problem(ground_truth="C"),
        ]

        grading_result = GradingResult(
            correct=False,
            confidence=1.0,
            reasoning="Wrong",
            extracted_answer="A",
            ground_truth="B",
        )
        processor.grader.grade = AsyncMock(return_value=grading_result)

        # session_num=4 % 3 = index 1 (ground_truth="B")
        metadata = create_metric_metadata(session_num=4)
        result = await processor.process_record(sample_parsed_record, metadata)

        assert result["accuracy.correct"] == 0.0
        processor.grader.grade.assert_awaited_once_with("Hello world", "B")

    async def test_process_record_last_valid_session_num_succeeds(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        user_config = _make_user_config()
        processor = _make_processor(monkeypatch, user_config)
        processor.problems = [
            _make_problem(ground_truth="A"),
            _make_problem(ground_truth="B"),
        ]

        grading_result = GradingResult(
            correct=True,
            confidence=1.0,
            reasoning="Correct",
            extracted_answer="B",
            ground_truth="B",
        )
        processor.grader.grade = AsyncMock(return_value=grading_result)

        metadata = create_metric_metadata(session_num=1)
        result = await processor.process_record(sample_parsed_record, metadata)

        assert result["accuracy.correct"] == 1.0


def _make_results_processor(
    monkeypatch, user_config: UserConfig
) -> AccuracyResultsProcessor:
    # AccuracyResultsProcessor.__init__ no longer calls plugins directly;
    # benchmark loading is in benchmark_loader and bypassed by setting _problems directly.
    return AccuracyResultsProcessor(user_config=user_config)


def _make_record_data(session_num: int, correct: float = 1.0) -> MetricRecordsData:
    return MetricRecordsData(
        metadata=create_metric_metadata(session_num=session_num),
        metrics={"accuracy.correct": correct},
    )


@pytest.mark.asyncio
class TestAccuracyResultsProcessorSessionBounds:
    async def test_process_result_wraps_when_session_num_exceeds_dataset(self) -> None:
        """session_num >= dataset size wraps via modulo so the correct task is recorded."""
        processor = _make_results_processor(None, _make_user_config())
        processor._problems = [_make_problem(task="algebra")]

        # session_num=1 wraps to index 0 (the only problem, task="algebra")
        await processor.process_result(_make_record_data(session_num=1))

        assert processor._task_total["algebra"] == 1
        assert processor._overall_total == 1

    async def test_process_result_wraps_to_correct_task(self) -> None:
        """With N problems, session_num=N+1 accumulates under the task at index 1."""
        processor = _make_results_processor(None, _make_user_config())
        processor._problems = [
            _make_problem(task="algebra"),
            _make_problem(task="history"),
            _make_problem(task="biology"),
        ]

        # session_num=4 % 3 = index 1 → task="history"
        await processor.process_result(_make_record_data(session_num=4))

        assert processor._task_total["history"] == 1
        assert processor._task_total.get("algebra", 0) == 0

    async def test_process_result_last_valid_session_num_succeeds(self) -> None:
        processor = _make_results_processor(None, _make_user_config())
        processor._problems = [
            _make_problem(ground_truth="A"),
            _make_problem(ground_truth="B"),
        ]

        await processor.process_result(_make_record_data(session_num=1, correct=1.0))

        assert processor._overall_total == 1
        assert processor._overall_correct == 1
        assert processor._task_correct["test_task"] == 1
