# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.accuracy_results_processor import AccuracyResultsProcessor
from aiperf.accuracy.models import AccuracyRecordsData, GradingResult
from aiperf.common.enums import CreditPhase
from aiperf.common.messages.inference_messages import MetricRecordsData
from aiperf.common.models.dataset_models import ConversationMetadata, DatasetMetadata
from aiperf.config import BenchmarkRun
from aiperf.plugin.enums import (
    AccuracyBenchmarkType,
    DatasetSamplingStrategy,
    EndpointType,
)
from tests.unit.conftest import make_benchmark_run
from tests.unit.post_processors.conftest import create_metric_metadata


def _make_run() -> BenchmarkRun:
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
    )


def _make_processor(monkeypatch) -> AccuracyRecordProcessor:
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

    return AccuracyRecordProcessor(run=_make_run(), service_id="test")


def _make_accuracy_accumulator() -> AccuracyResultsProcessor:
    return AccuracyResultsProcessor(run=_make_run())


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
        metrics={"accuracy_correct": correct, "accuracy_unparsed": unparsed},
    )


class TestAccuracyRecordProcessorOnDatasetConfigured:
    def test_populates_ground_truths_from_metadata(self, monkeypatch) -> None:
        processor = _make_processor(monkeypatch)
        metadata = _make_dataset_metadata(["A", "B", "C"], ["t1", "t2", "t3"])

        processor.on_dataset_configured(metadata)

        assert processor._ground_truths == ["A", "B", "C"]
        assert processor._tasks == ["t1", "t2", "t3"]

    def test_skips_conversations_without_accuracy_fields(self, monkeypatch) -> None:
        processor = _make_processor(monkeypatch)
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
        processor = _make_processor(monkeypatch)
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

        assert isinstance(result, AccuracyRecordsData)
        assert result.passed is True
        assert result.unparsed is False
        processor.grader.grade.assert_awaited_once_with("Hello world", "A")

    async def test_process_record_maps_all_grading_and_metadata_fields(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        """Every AccuracyRecordsData field maps from the grader result / metadata."""
        processor = _make_processor(monkeypatch)
        processor._ground_truths = ["A", "B", "C"]
        processor._tasks = ["t0", "t1", "t2"]

        grading_result = GradingResult(
            correct=False,
            unparsed=True,
            confidence=0.42,
            reasoning="Wrong answer",
            extracted_answer="A",
            ground_truth="B",
        )
        processor.grader.grade = AsyncMock(return_value=grading_result)

        # session_num=4 % 3 = index 1 -> ground_truth="B", task="t1"
        metadata = create_metric_metadata(
            session_num=4,
            worker_id="worker-9",
            request_end_ns=1_234_567_890,
            benchmark_phase=CreditPhase.PROFILING,
        )
        result = await processor.process_record(sample_parsed_record, metadata)

        assert isinstance(result, AccuracyRecordsData)
        assert result.session_num == 4
        assert result.worker_id == "worker-9"
        assert result.benchmark_phase == CreditPhase.PROFILING
        assert result.timestamp_ns == 1_234_567_890
        assert result.task == "t1"
        assert result.grader_name == "multiple_choice"
        assert result.passed is False
        assert result.unparsed is True
        assert result.confidence == 0.42
        assert result.expected == "B"
        assert result.actual == "A"
        assert result.reasoning == "Wrong answer"
        processor.grader.grade.assert_awaited_once_with("Hello world", "B")

    async def test_process_record_task_none_when_no_tasks(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        """task is None when the dataset carried no task labels."""
        processor = _make_processor(monkeypatch)
        processor._ground_truths = ["A", "B"]
        processor._tasks = []

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

        assert isinstance(result, AccuracyRecordsData)
        assert result.task is None
        assert result.passed is True

    async def test_process_record_raises_if_not_configured(
        self, monkeypatch, sample_parsed_record
    ) -> None:
        """process_record must raise if on_dataset_configured was never called."""
        processor = _make_processor(monkeypatch)
        metadata = create_metric_metadata(session_num=0)

        with pytest.raises(RuntimeError, match="dataset not configured"):
            await processor.process_record(sample_parsed_record, metadata)


class TestLogGradingDetail:
    """``_log_grading_detail`` surfaces the grader's reasoning/extracted answer
    that otherwise never reaches metrics — the diagnostic that turns a
    "100% unparsed" report from a guess into a one-line answer."""

    def _result(self) -> GradingResult:
        return GradingResult(
            correct=False,
            unparsed=True,
            confidence=0.0,
            reasoning="LCB grader failed: sandboxed exec failed: daemonic ...",
            extracted_answer="```python\nprint(1)\n```",
            ground_truth="<lcb test cases>",
        )

    def test_verbose_logs_reason_at_info(self, monkeypatch) -> None:
        processor = _make_processor(monkeypatch)
        processor._verbose = True
        logged: list[str] = []
        monkeypatch.setattr(
            processor, "info", lambda m: logged.append(m() if callable(m) else m)
        )
        processor._log_grading_detail(0, "some response", self._result())
        assert len(logged) == 1
        assert "sandboxed exec failed" in logged[0]
        assert "unparsed=True" in logged[0]

    def test_non_verbose_debug_disabled_is_noop(self, monkeypatch) -> None:
        """No verbose flag and debug off → skip entirely (no info, no debug)."""
        processor = _make_processor(monkeypatch)
        processor._verbose = False
        monkeypatch.setattr(
            type(processor), "is_debug_enabled", property(lambda _s: False)
        )
        info_calls, debug_calls = [], []
        monkeypatch.setattr(processor, "info", lambda m: info_calls.append(m))
        monkeypatch.setattr(processor, "debug", lambda m: debug_calls.append(m))
        processor._log_grading_detail(0, "some response", self._result())
        assert info_calls == []
        assert debug_calls == []

    def test_non_verbose_debug_enabled_logs_at_debug(self, monkeypatch) -> None:
        processor = _make_processor(monkeypatch)
        processor._verbose = False
        monkeypatch.setattr(
            type(processor), "is_debug_enabled", property(lambda _s: True)
        )
        logged: list[str] = []
        monkeypatch.setattr(
            processor, "debug", lambda m: logged.append(m() if callable(m) else m)
        )
        processor._log_grading_detail(0, "some response", self._result())
        assert len(logged) == 1
        assert "sandboxed exec failed" in logged[0]


class TestAccuracyResultsProcessorOnDatasetConfigured:
    def test_populates_tasks_from_metadata(self) -> None:
        processor = _make_accuracy_accumulator()
        metadata = _make_dataset_metadata(["A", "B"], ["algebra", "history"])

        processor.on_dataset_configured(metadata)

        assert processor._tasks == ["algebra", "history"]

    def test_skips_conversations_without_accuracy_task(self) -> None:
        processor = _make_accuracy_accumulator()
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
    async def test_process_record_wraps_when_session_num_exceeds_dataset(self) -> None:
        """session_num >= dataset size wraps via modulo so the correct task is recorded."""
        processor = _make_accuracy_accumulator()
        processor._tasks = ["algebra"]

        # session_num=1 wraps to index 0 (the only task, "algebra")
        await processor.process_record(_make_record_data(session_num=1))

        assert processor._task_total[CreditPhase.PROFILING]["algebra"] == 1
        assert processor._overall_total[CreditPhase.PROFILING] == 1

    async def test_process_record_wraps_to_correct_task(self) -> None:
        """With N problems, session_num=N+1 accumulates under the task at index 1."""
        processor = _make_accuracy_accumulator()
        processor._tasks = ["algebra", "history", "biology"]

        # session_num=4 % 3 = index 1 → task="history"
        await processor.process_record(_make_record_data(session_num=4))

        assert processor._task_total[CreditPhase.PROFILING]["history"] == 1
        assert processor._task_total[CreditPhase.PROFILING].get("algebra", 0) == 0

    async def test_process_record_last_valid_session_num_succeeds(self) -> None:
        processor = _make_accuracy_accumulator()
        processor._tasks = ["test_task", "test_task"]

        await processor.process_record(_make_record_data(session_num=1, correct=1.0))

        assert processor._overall_total[CreditPhase.PROFILING] == 1
        assert processor._overall_correct[CreditPhase.PROFILING] == 1
        assert processor._task_correct[CreditPhase.PROFILING]["test_task"] == 1

    async def test_process_record_raises_if_not_configured(self) -> None:
        """process_record must raise if on_dataset_configured was never called."""
        processor = _make_accuracy_accumulator()

        with pytest.raises(RuntimeError, match="dataset not configured"):
            await processor.process_record(_make_record_data(session_num=0))

    async def test_process_record_increments_overall_unparsed(self) -> None:
        processor = _make_accuracy_accumulator()
        processor._tasks = ["algebra"]

        await processor.process_record(
            _make_record_data(session_num=0, correct=1.0, unparsed=1.0)
        )

        assert processor._overall_unparsed[CreditPhase.PROFILING] == 1
        assert processor._overall_total[CreditPhase.PROFILING] == 1

    async def test_process_record_increments_task_unparsed(self) -> None:
        processor = _make_accuracy_accumulator()
        processor._tasks = ["algebra"]

        await processor.process_record(
            _make_record_data(session_num=0, correct=0.0, unparsed=1.0)
        )

        assert processor._task_unparsed[CreditPhase.PROFILING]["algebra"] == 1

    async def test_process_record_does_not_increment_unparsed_when_conforming(
        self,
    ) -> None:
        processor = _make_accuracy_accumulator()
        processor._tasks = ["algebra"]

        await processor.process_record(
            _make_record_data(session_num=0, correct=1.0, unparsed=0.0)
        )

        assert processor._overall_unparsed[CreditPhase.PROFILING] == 0
        assert processor._task_unparsed[CreditPhase.PROFILING].get("algebra", 0) == 0

    async def test_process_record_missing_unparsed_key_treated_as_conforming(
        self,
    ) -> None:
        """Records without accuracy_unparsed (e.g. from older graders) count as conforming."""
        processor = _make_accuracy_accumulator()
        processor._tasks = ["algebra"]
        data = MetricRecordsData(
            metadata=create_metric_metadata(session_num=0),
            metrics={"accuracy_correct": 1.0},  # no accuracy_unparsed key
        )

        await processor.process_record(data)

        assert processor._overall_unparsed[CreditPhase.PROFILING] == 0
