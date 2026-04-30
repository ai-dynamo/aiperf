# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.accuracy_results_processor import AccuracyResultsProcessor
from aiperf.accuracy.models import GradingResult
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models.dataset_models import ConversationMetadata, DatasetMetadata
from aiperf.plugin.enums import (
    AccuracyBenchmarkType,
    DatasetSamplingStrategy,
    EndpointType,
)
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

    monkeypatch.setattr(
        "aiperf.accuracy.accuracy_record_processor.plugins.get_class",
        lambda plugin_type, name: mock_grader_cls,
    )
    monkeypatch.setattr(
        "aiperf.accuracy.accuracy_record_processor.plugins.get_metadata",
        lambda *_args, **_kwargs: {"default_grader": "multiple_choice"},
    )

    return AccuracyRecordProcessor(service_id="test", user_config=user_config)


def _make_results_processor(user_config: UserConfig) -> AccuracyResultsProcessor:
    return AccuracyResultsProcessor(user_config=user_config)


def _make_dataset_metadata(
    ground_truths: list[str], tasks: list[str]
) -> DatasetMetadata:
    assert len(ground_truths) == len(tasks)
    conversations = [
        ConversationMetadata(
            conversation_id=f"conv-{i}",
            accuracy_ground_truth=gt,
            accuracy_task=task,
        )
        for i, (gt, task) in enumerate(zip(ground_truths, tasks, strict=True))
    ]
    return DatasetMetadata(
        conversations=conversations,
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _make_record_data(
    session_num: int, correct: float = 1.0, unparsed: float = 0.0
) -> MetricRecordsData:
    return MetricRecordsData(
        metadata=create_metric_metadata(session_num=session_num),
        metrics={"accuracy.correct": correct, "accuracy.unparsed": unparsed},
    )


class TestAccuracyRecordProcessorOnDatasetConfigured:
    def test_populates_ground_truths_from_metadata(self, monkeypatch) -> None:
        processor = _make_processor(monkeypatch, _make_user_config())
        metadata = _make_dataset_metadata(["A", "B", "C"], ["t1", "t2", "t3"])

        processor.on_dataset_configured(metadata)

        assert processor._ground_truths == ["A", "B", "C"]

    def test_skips_conversations_without_accuracy_fields(self, monkeypatch) -> None:
        processor = _make_processor(monkeypatch, _make_user_config())
        conversations = [
            ConversationMetadata(conversation_id="plain"),  # no accuracy fields
            ConversationMetadata(
                conversation_id="accurate",
                accuracy_ground_truth="B",
                accuracy_task="math",
            ),
        ]
        metadata = DatasetMetadata(
            conversations=conversations,
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        processor.on_dataset_configured(metadata)

        assert processor._ground_truths == ["B"]


@pytest.mark.asyncio
class TestAccuracyRecordProcessorSessionBounds:
    async def test_process_record_wraps_when_session_num_exceeds_dataset(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        """session_num >= dataset size wraps via modulo so the correct problem is graded."""
        processor = _make_processor(monkeypatch, _make_user_config())
        processor._ground_truths = ["A"]

        grading_result = GradingResult(
            correct=True,
            confidence=1.0,
            reasoning="Correct",
            extracted_answer="A",
            ground_truth="A",
        )
        processor.grader.grade = AsyncMock(return_value=grading_result)

        # session_num=1 wraps to index 0 (the only ground truth)
        metadata = create_metric_metadata(session_num=1)
        result = await processor.process_record(sample_parsed_record, metadata)

        assert result["accuracy.correct"] == 1.0
        assert result["accuracy.unparsed"] == 0.0
        processor.grader.grade.assert_awaited_once_with("Hello world", "A")

    async def test_process_record_wraps_to_correct_problem(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        """With N problems, session_num=N+1 grades problem at index 1."""
        processor = _make_processor(monkeypatch, _make_user_config())
        processor._ground_truths = ["A", "B", "C"]

        grading_result = GradingResult(
            correct=False,
            unparsed=True,
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
        assert result["accuracy.unparsed"] == 1.0
        processor.grader.grade.assert_awaited_once_with("Hello world", "B")

    async def test_process_record_last_valid_session_num_succeeds(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        processor = _make_processor(monkeypatch, _make_user_config())
        processor._ground_truths = ["A", "B"]

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
        assert result["accuracy.unparsed"] == 0.0

    async def test_process_record_raises_if_not_configured(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        """process_record must raise if on_dataset_configured was never called."""
        processor = _make_processor(monkeypatch, _make_user_config())
        metadata = create_metric_metadata(session_num=0)

        with pytest.raises(RuntimeError, match="dataset not configured"):
            await processor.process_record(sample_parsed_record, metadata)


class TestAccuracyResultsProcessorOnDatasetConfigured:
    def test_populates_tasks_from_metadata(self) -> None:
        processor = _make_results_processor(_make_user_config())
        metadata = _make_dataset_metadata(["A", "B"], ["algebra", "history"])

        processor.on_dataset_configured(metadata)

        assert processor._tasks == ["algebra", "history"]

    def test_skips_conversations_without_accuracy_task(self) -> None:
        processor = _make_results_processor(_make_user_config())
        conversations = [
            ConversationMetadata(conversation_id="plain"),
            ConversationMetadata(
                conversation_id="accurate",
                accuracy_ground_truth="B",
                accuracy_task="math",
            ),
        ]
        metadata = DatasetMetadata(
            conversations=conversations,
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        processor.on_dataset_configured(metadata)

        assert processor._tasks == ["math"]


@pytest.mark.asyncio
class TestAccuracyResultsProcessorSessionBounds:
    async def test_process_result_wraps_when_session_num_exceeds_dataset(self) -> None:
        """session_num >= dataset size wraps via modulo so the correct task is recorded."""
        processor = _make_results_processor(_make_user_config())
        processor._tasks = ["algebra"]

        # session_num=1 wraps to index 0 (the only task, "algebra")
        await processor.process_result(_make_record_data(session_num=1))

        assert processor._task_total["algebra"] == 1
        assert processor._overall_total == 1

    async def test_process_result_wraps_to_correct_task(self) -> None:
        """With N problems, session_num=N+1 accumulates under the task at index 1."""
        processor = _make_results_processor(_make_user_config())
        processor._tasks = ["algebra", "history", "biology"]

        # session_num=4 % 3 = index 1 → task="history"
        await processor.process_result(_make_record_data(session_num=4))

        assert processor._task_total["history"] == 1
        assert processor._task_total.get("algebra", 0) == 0

    async def test_process_result_last_valid_session_num_succeeds(self) -> None:
        processor = _make_results_processor(_make_user_config())
        processor._tasks = ["test_task", "test_task"]

        await processor.process_result(_make_record_data(session_num=1, correct=1.0))

        assert processor._overall_total == 1
        assert processor._overall_correct == 1
        assert processor._task_correct["test_task"] == 1

    async def test_process_result_raises_if_not_configured(self) -> None:
        """process_result must raise if on_dataset_configured was never called."""
        processor = _make_results_processor(_make_user_config())

        with pytest.raises(RuntimeError, match="dataset not configured"):
            await processor.process_result(_make_record_data(session_num=0))

    async def test_process_result_increments_overall_unparsed(self) -> None:
        processor = _make_results_processor(_make_user_config())
        processor._tasks = ["algebra"]

        await processor.process_result(
            _make_record_data(session_num=0, correct=1.0, unparsed=1.0)
        )

        assert processor._overall_unparsed == 1
        assert processor._overall_total == 1

    async def test_process_result_increments_task_unparsed(self) -> None:
        processor = _make_results_processor(_make_user_config())
        processor._tasks = ["algebra"]

        await processor.process_result(
            _make_record_data(session_num=0, correct=0.0, unparsed=1.0)
        )

        assert processor._task_unparsed["algebra"] == 1

    async def test_process_result_does_not_increment_unparsed_when_conforming(
        self,
    ) -> None:
        processor = _make_results_processor(_make_user_config())
        processor._tasks = ["algebra"]

        await processor.process_result(
            _make_record_data(session_num=0, correct=1.0, unparsed=0.0)
        )

        assert processor._overall_unparsed == 0
        assert processor._task_unparsed.get("algebra", 0) == 0

    async def test_process_result_missing_unparsed_key_treated_as_conforming(
        self,
    ) -> None:
        """Records without accuracy.unparsed (e.g. from older graders) count as conforming."""
        processor = _make_results_processor(_make_user_config())
        processor._tasks = ["algebra"]
        data = MetricRecordsData(
            metadata=create_metric_metadata(session_num=0),
            metrics={"accuracy.correct": 1.0},  # no accuracy.unparsed key
        )

        await processor.process_result(data)

        assert processor._overall_unparsed == 0
