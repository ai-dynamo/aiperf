# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.models import AccuracyRecordsData, GradingResult
from aiperf.common.enums import CreditPhase
from aiperf.common.models.dataset_models import ConversationMetadata, DatasetMetadata
from aiperf.common.models.record_models import ParsedResponse, ParsedResponseRecord
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


class TestAccuracyRecordProcessorInit:
    def test_raises_when_accuracy_not_enabled(self, monkeypatch) -> None:
        """PostProcessorDisabled raised when accuracy mode is off."""
        from aiperf.common.exceptions import PostProcessorDisabled

        run = make_benchmark_run(
            model_names=["m"],
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=False,
        )
        with pytest.raises(PostProcessorDisabled):
            AccuracyRecordProcessor(run=run, service_id="test")


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

    def test_tasks_stay_aligned_with_ground_truths_when_labels_are_sparse(
        self, monkeypatch
    ) -> None:
        """Graded conversations missing a task label keep a None slot so the
        session_num modulo maps to the correct task instead of being shifted."""
        processor = _make_processor(monkeypatch)
        conversations = [
            ConversationMetadata(
                conversation_id="c0", accuracy_ground_truth="A", accuracy_task="t0"
            ),
            ConversationMetadata(
                conversation_id="c1", accuracy_ground_truth="B"
            ),  # graded, no task label
            ConversationMetadata(
                conversation_id="c2", accuracy_ground_truth="C", accuracy_task="t2"
            ),
        ]
        metadata = DatasetMetadata(
            conversations=conversations,
            sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
        )

        processor.on_dataset_configured(metadata)

        assert processor._ground_truths == ["A", "B", "C"]
        # Index-aligned with ground truths; the label-less conversation is None,
        # not dropped (which would have mismapped c1 -> "t2").
        assert processor._tasks == ["t0", None, "t2"]


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
            conversation_id="session_000004",
            x_request_id="req-abc",
        )
        result = await processor.process_record(sample_parsed_record, metadata)

        assert isinstance(result, AccuracyRecordsData)
        assert result.session_num == 4
        assert result.conversation_id == "session_000004"
        assert result.x_request_id == "req-abc"
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
        assert result.explanation == "Wrong answer"
        # Full model output captured; no separate reasoning channel here.
        assert result.model_output == "Hello world"
        assert result.model_thinking is None
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

    async def test_process_record_grades_reasoning_model_on_content_only(
        self, monkeypatch
    ) -> None:
        """Grader receives only the answer content, not reasoning + content.

        Regression test for https://github.com/ai-dynamo/aiperf/issues/1136:
        reasoning models returned 0% because the CoT preamble was concatenated
        with the final answer before exact-match comparison.
        """
        from aiperf.common.models.record_models import ReasoningResponseData

        processor = _make_processor(monkeypatch)
        processor._ground_truths = ["True"]

        grading_result = GradingResult(
            correct=True,
            confidence=1.0,
            reasoning="Correct",
            extracted_answer="True",
            ground_truth="True",
        )
        processor.grader.grade = AsyncMock(return_value=grading_result)

        reasoning_record = MagicMock(spec=ParsedResponseRecord)
        reasoning_record.content_responses = [
            ParsedResponse(
                perf_ns=0,
                data=ReasoningResponseData(
                    reasoning="Thinking Process:\n\n1. Analyze the request... True",
                    content="\n\nTrue",
                ),
            ),
        ]

        metadata = create_metric_metadata(session_num=0)
        result = await processor.process_record(reasoning_record, metadata)

        assert result.passed is True
        # Grader must have received only the answer content, not the CoT preamble.
        processor.grader.grade.assert_awaited_once_with("\n\nTrue", "True")
        assert result.model_output == "\n\nTrue"
        assert (
            result.model_thinking
            == "Thinking Process:\n\n1. Analyze the request... True"
        )

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


class TestExtractOutputAndThinking:
    """`_extract_output_and_thinking` splits answer content from reasoning."""

    @staticmethod
    def _record(data_list: list[str]) -> ParsedResponseRecord:
        record = MagicMock(spec=ParsedResponseRecord)
        record.content_responses = [
            ParsedResponse(perf_ns=i, data=d) for i, d in enumerate(data_list)
        ]
        return record

    def test_text_only_output_no_thinking(self) -> None:
        from aiperf.common.models.record_models import TextResponseData

        record = self._record(
            [TextResponseData(text="Hello"), TextResponseData(text=" world")]
        )
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == "Hello world"
        assert thinking is None

    def test_reasoning_split_into_output_and_thinking(self) -> None:
        from aiperf.common.models.record_models import ReasoningResponseData

        record = self._record(
            [
                ReasoningResponseData(
                    content="The answer is (B)", reasoning="Let me think... "
                ),
                ReasoningResponseData(content=" final.", reasoning="step two."),
            ]
        )
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == "The answer is (B) final."
        assert thinking == "Let me think... step two."

    def test_empty_record_yields_empty_output_and_none_thinking(self) -> None:
        record = self._record([])
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == ""
        assert thinking is None

    def test_reasoning_only_content_none_falls_back_to_reasoning(self) -> None:
        """content=None with reasoning present → reasoning used as model_output fallback."""
        from aiperf.common.models.record_models import ReasoningResponseData

        record = self._record(
            [ReasoningResponseData(content=None, reasoning="Thinking... True")]
        )
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == "Thinking... True"
        assert thinking == "Thinking... True"

    def test_reasoning_only_content_empty_falls_back_to_reasoning(self) -> None:
        """content='' is treated as missing; reasoning used as model_output fallback."""
        from aiperf.common.models.record_models import ReasoningResponseData

        record = self._record(
            [ReasoningResponseData(content="", reasoning="Thinking... True")]
        )
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == "Thinking... True"
        assert thinking == "Thinking... True"

    def test_unknown_subclass_uses_get_text_not_content_attr(self) -> None:
        """Non-Reasoning/ToolCall subclasses go through get_text(), not .content.

        Regression: the old getattr-based implementation would have read the
        .content attribute directly, returning the wrong value for any
        BaseResponseData subclass whose .content differs from .get_text().
        """
        from dataclasses import dataclass

        from aiperf.common.models.record_models import BaseResponseData

        @dataclass(slots=True)
        class CustomResponseData(BaseResponseData):
            content: str = "wrong"

            def get_text(self) -> str:
                return "correct"

        record = self._record([CustomResponseData()])
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == "correct"
        assert thinking is None

    def test_none_data_in_response_is_skipped(self) -> None:
        """resp.data=None entries are skipped without error."""
        from aiperf.common.models.record_models import TextResponseData

        record = MagicMock(spec=ParsedResponseRecord)
        record.content_responses = [
            ParsedResponse(perf_ns=0, data=None),
            ParsedResponse(perf_ns=1, data=TextResponseData(text="hello")),
        ]
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == "hello"
        assert thinking is None

    def test_tool_call_response_includes_content_and_tool_call_text(self) -> None:
        """ToolCallResponseData appends content then tool_call_text to model_output."""
        from aiperf.common.models.record_models import ToolCallResponseData

        record = self._record(
            [
                ToolCallResponseData(
                    tool_call_text='{"name":"fn"}', content="Sure, calling:"
                )
            ]
        )
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == 'Sure, calling:{"name":"fn"}'
        assert thinking is None

    def test_tool_call_response_no_content_uses_tool_call_text_only(self) -> None:
        """ToolCallResponseData with content=None only adds tool_call_text."""
        from aiperf.common.models.record_models import ToolCallResponseData

        record = self._record(
            [ToolCallResponseData(tool_call_text='{"name":"fn"}', content=None)]
        )
        output, thinking = AccuracyRecordProcessor._extract_output_and_thinking(record)
        assert output == '{"name":"fn"}'
        assert thinking is None
