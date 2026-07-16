# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import Field
from typing_extensions import TypedDict

from aiperf.common.enums import CreditPhase
from aiperf.common.models.base_models import AIPerfBaseModel

# Summary/metric tags (the ``accuracy.`` dot namespace) emitted by
# AccuracyResultsProcessor.summarize() and read by the accuracy exporters.
ACCURACY_OVERALL_TAG = "accuracy.overall"
ACCURACY_TASK_TAG_PREFIX = "accuracy.task."
ACCURACY_UNPARSED_TAG = "accuracy.unparsed"
ACCURACY_UNPARSED_TASK_TAG_PREFIX = "accuracy.unparsed.task."
ACCURACY_METRIC_PREFIX = "accuracy."

# Per-record TRANSPORT keys (NOT metric tags): AccuracyRecordProcessor writes
# these into each record's MetricRecordDict to hand the grade to
# AccuracyResultsProcessor.process_record. They use an ``accuracy_`` (underscore)
# prefix so they live outside the ``accuracy.`` (dot) summary namespace and can
# never be mistaken for -- or collide with -- a summary/metric tag. Keeping them
# in the dot namespace is what let the registered ``accuracy.unparsed`` transport
# metric shadow the ``accuracy.unparsed`` summary tag; the underscore separates them.
ACCURACY_RECORD_CORRECT_KEY = "accuracy_correct"
ACCURACY_RECORD_UNPARSED_KEY = "accuracy_unparsed"


def accuracy_task_tag(task: str) -> str:
    """Build the MetricResult.tag for a per-task accuracy result."""
    return f"{ACCURACY_TASK_TAG_PREFIX}{task}"


def accuracy_unparsed_task_tag(task: str) -> str:
    """Build the MetricResult.tag for a per-task unparsed-count result."""
    return f"{ACCURACY_UNPARSED_TASK_TAG_PREFIX}{task}"


class AccuracyChatMessage(TypedDict):
    """A single OpenAI-compatible chat message used in accuracy benchmark prompts."""

    role: Literal["system", "user", "assistant"]
    content: str


class GradingResult(AIPerfBaseModel):
    """Result of grading a single LLM response against ground truth."""

    correct: bool = Field(description="Whether the response was graded as correct")
    unparsed: bool = Field(
        default=False,
        description="True when the model output did not match the expected format "
        "(e.g. 'The answer is B.' instead of 'B') and a regex fallback was used. "
        "A correct unparsed response is still scored as correct.",
    )
    confidence: float = Field(
        ge=0, le=1, description="Confidence score of the grading (0.0 to 1.0)"
    )
    reasoning: str = Field(description="Explanation of the grading decision")
    extracted_answer: str = Field(
        description="Answer extracted from the model response"
    )
    ground_truth: str = Field(description="Expected correct answer")


class BenchmarkProblem(AIPerfBaseModel):
    """A single problem from an accuracy benchmark dataset."""

    prompt: str = Field(description="The prompt to send to the LLM")
    ground_truth: str = Field(description="The expected correct answer")
    task: str = Field(description="The task or subtask name within the benchmark")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional problem metadata"
    )
    raw_messages: list[AccuracyChatMessage] | None = Field(
        default=None,
        description="Pre-formatted OpenAI-compatible messages array for the chat endpoint. "
        "Assigned verbatim to Turn.raw_messages when building the dataset, matching "
        "lighteval's chat format. The flat 'prompt' field is still used for the "
        "completions endpoint. "
        "AccuracyChatMessage narrows the shape to {role, content} — accuracy benchmarks "
        "only produce these two shapes. The type broadens to dict[str, Any] at "
        "Turn.raw_messages because that field also accepts tool-call and multi-modal "
        "messages from other callers (e.g. MooncakeTrace).",
    )


class AccuracyRecordsData(AIPerfBaseModel):
    """Per-graded-response record that flows on the dedicated ``accuracy`` channel.

    Mirrors the record-type pattern used by ``ServerMetricsRecord`` and
    ``TelemetryRecord``: ``record_type`` is a plain ``ClassVar`` (not a Pydantic
    field) read by the routing layer via ``getattr(record, "record_type")``.
    """

    record_type: ClassVar[str] = "accuracy"

    session_num: int = Field(
        description="Conversation/session index this response came from, used to map to task"
    )
    worker_id: str = Field(
        description="ID of the record processor that produced this record"
    )
    benchmark_phase: CreditPhase = Field(
        description="Benchmark phase active when grading completed (warmup vs profiling)"
    )
    timestamp_ns: int = Field(
        description="Nanosecond wall-clock timestamp when grading completed"
    )
    task: str | None = Field(
        default=None,
        description="Accuracy task/subtask name (e.g. an MMLU subtask); "
        "None when the dataset has no task label",
    )
    grader_name: str = Field(description="Which grader scored this response")
    passed: bool = Field(
        description="Whether the response was graded correct (maps from GradingResult.correct)"
    )
    unparsed: bool = Field(
        default=False,
        description="Whether the model output needed a regex fallback "
        "(maps from GradingResult.unparsed)",
    )
    confidence: float = Field(ge=0, le=1, description="Grading confidence (0.0 to 1.0)")
    expected: str = Field(
        description="Ground-truth answer (maps from GradingResult.ground_truth)"
    )
    actual: str = Field(
        description="Answer extracted from the model response "
        "(maps from GradingResult.extracted_answer)"
    )
    reasoning: str = Field(description="Grader's explanation of the grading decision")


class TaskAccuracyStats(AIPerfBaseModel):
    """Per-task accuracy rollup."""

    total: int = Field(description="Total responses evaluated for this task")
    passed: int = Field(description="Number graded correct for this task")
    unparsed: int = Field(
        description="Number that needed a regex fallback for this task"
    )
    accuracy_rate: float = Field(
        description="passed/total for this task, 0.0 when total==0"
    )
    unparsed_rate: float = Field(
        description="unparsed/total for this task, 0.0 when total==0"
    )


class AccuracySummary(AIPerfBaseModel):
    """Accumulator result payload; structured replacement for today's list[MetricResult]."""

    total_evaluated: int = Field(
        description="Total responses evaluated across all tasks"
    )
    total_passed: int = Field(
        description="Total responses graded correct across all tasks"
    )
    accuracy_rate: float = Field(
        description="total_passed/total_evaluated, 0.0 when total_evaluated==0"
    )
    overall_unparsed: int = Field(
        description="Total responses that needed a regex fallback across all tasks"
    )
    grader_name: str | None = Field(
        default=None, description="Grader that scored these responses, if uniform"
    )
    per_task: dict[str, TaskAccuracyStats] = Field(
        default_factory=dict, description="Per-task accuracy rollups keyed by task name"
    )

    def to_json(self) -> dict[str, Any]:
        """Return a plain-dict representation of the summary."""
        return self.model_dump()

    def to_csv(self) -> list[dict[str, Any]]:
        """Return one row per task (sorted by name) plus a trailing OVERALL row.

        Columns: task, total, passed, unparsed, accuracy_rate, unparsed_rate.
        Mirrors today's ``accuracy_results.csv`` shape (per-task rows + overall).
        """
        rows: list[dict[str, Any]] = [
            {
                "task": task,
                "total": stats.total,
                "passed": stats.passed,
                "unparsed": stats.unparsed,
                "accuracy_rate": stats.accuracy_rate,
                "unparsed_rate": stats.unparsed_rate,
            }
            for task, stats in sorted(self.per_task.items())
        ]
        overall_unparsed_rate = (
            self.overall_unparsed / self.total_evaluated
            if self.total_evaluated
            else 0.0
        )
        rows.append(
            {
                "task": "OVERALL",
                "total": self.total_evaluated,
                "passed": self.total_passed,
                "unparsed": self.overall_unparsed,
                "accuracy_rate": self.accuracy_rate,
                "unparsed_rate": overall_unparsed_rate,
            }
        )
        return rows


class ProcessAccuracyResult(AIPerfBaseModel):
    """Wire wrapper for a processed accuracy summary - mirrors ProcessServerMetricsResult."""

    results: AccuracySummary | None = Field(
        default=None, description="The processed accuracy summary"
    )
