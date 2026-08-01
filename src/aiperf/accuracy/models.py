# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field
from typing_extensions import TypedDict

from aiperf.common.enums import CreditPhase
from aiperf.common.models import RecordData
from aiperf.common.models.base_models import AIPerfBaseModel

if TYPE_CHECKING:
    from aiperf.common.models import MetricResult

# Summary/metric tags (the ``accuracy.`` dot namespace) materialized from the
# dedicated-channel ``AccuracySummary`` and read by the accuracy exporters.
ACCURACY_OVERALL_TAG = "accuracy.overall"
ACCURACY_TASK_TAG_PREFIX = "accuracy.task."
ACCURACY_UNPARSED_TAG = "accuracy.unparsed"
ACCURACY_UNPARSED_TASK_TAG_PREFIX = "accuracy.unparsed.task."
ACCURACY_METRIC_PREFIX = "accuracy."


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


class AccuracyRecordsData(RecordData):
    """Per-graded-response record that rides the generic ``RecordsMessage`` envelope.

    ``record_type`` is a SERIALIZED ``Literal`` discriminator field (not a
    ClassVar) so AutoRoutedModel reconstructs the concrete type across the ZMQ
    boundary; the routing layer still reads it via ``getattr(record, "record_type")``.
    """

    record_type: Literal["accuracy"] = Field(
        default="accuracy",
        description="Serialized discriminator routing this record to the accuracy "
        "channel for wire reconstruction.",
    )

    session_num: int = Field(
        ge=0,
        description="Conversation/session index this response came from, used to map to task",
    )
    conversation_id: str | None = Field(
        default=None,
        description="Stable id of the benchmark problem/conversation this response "
        "answered; the key to look up the full prompt in inputs.json",
    )
    x_request_id: str | None = Field(
        default=None,
        description="Unique per-request id (X-Request-ID) for tracing this exact "
        "graded response back to the raw records",
    )
    worker_id: str = Field(description="ID of the worker that produced this record")
    benchmark_phase: CreditPhase = Field(
        description="Benchmark phase active when grading completed (warmup vs profiling)"
    )
    timestamp_ns: int = Field(
        ge=0, description="Nanosecond wall-clock timestamp when grading completed"
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
    explanation: str = Field(
        description="The grader's explanation of WHY it scored this response "
        "correct/incorrect (distinct from model_thinking, which is the model's own "
        "reasoning). Maps from GradingResult.reasoning",
    )
    model_output: str = Field(
        default="",
        description="Full model response content (the answer text the model "
        "returned, excluding any separate reasoning channel). Always populated by "
        "AccuracyRecordProcessor in production",
    )
    model_thinking: str | None = Field(
        default=None,
        description="Model's reasoning/thinking content (reasoning_content) when "
        "the model emitted a separate reasoning channel; None otherwise",
    )


class TaskAccuracyStats(AIPerfBaseModel):
    """Per-task accuracy rollup."""

    total: int = Field(ge=0, description="Total responses evaluated for this task")
    passed: int = Field(ge=0, description="Number graded correct for this task")
    unparsed: int = Field(
        ge=0, description="Number that needed a regex fallback for this task"
    )
    accuracy_rate: float = Field(
        ge=0, le=1, description="passed/total for this task, 0.0 when total==0"
    )
    unparsed_rate: float = Field(
        ge=0, le=1, description="unparsed/total for this task, 0.0 when total==0"
    )


class AccuracySummary(AIPerfBaseModel):
    """Accumulator result payload; structured replacement for today's list[MetricResult]."""

    total_evaluated: int = Field(
        ge=0, description="Total responses evaluated across all tasks"
    )
    total_passed: int = Field(
        ge=0, description="Total responses graded correct across all tasks"
    )
    accuracy_rate: float = Field(
        ge=0,
        le=1,
        description="total_passed/total_evaluated, 0.0 when total_evaluated==0",
    )
    overall_unparsed: int = Field(
        ge=0,
        description="Total responses that needed a regex fallback across all tasks",
    )
    grader_name: str | None = Field(
        default=None,
        description="Grader that scored these responses, taken from the first "
        "record in scope",
    )
    per_task: dict[str, TaskAccuracyStats] = Field(
        default_factory=dict, description="Per-task accuracy rollups keyed by task name"
    )

    def to_metric_results(self) -> list[MetricResult]:
        """Legacy ``accuracy.*`` MetricResult representation for byte-identical export.

        Reproduces the legacy ``AccuracyResultsProcessor._build_results`` output
        field-for-field so every legacy exporter (perf CSV/JSON + the dedicated
        accuracy CSV/console) renders identical bytes when these results are
        injected into ``ProfileResults.records``.

        Emitted in this exact order (load-bearing for byte-exact JSON/CSV):
        overall, tasks sorted, unparsed overall, unparsed tasks sorted.
        """
        from aiperf.common.enums import MetricConsoleGroup
        from aiperf.common.models import MetricResult

        results: list[MetricResult] = []

        if self.total_evaluated > 0:
            results.append(
                MetricResult(
                    tag=ACCURACY_OVERALL_TAG,
                    header="Accuracy (Overall)",
                    unit="ratio",
                    count=self.total_evaluated,
                    current=self.total_passed / self.total_evaluated,
                    sum=self.total_passed,
                    console_group=MetricConsoleGroup.NONE,
                )
            )

        for task in sorted(self.per_task):
            stats = self.per_task[task]
            results.append(
                MetricResult(
                    tag=accuracy_task_tag(task),
                    header=f"Accuracy ({task})",
                    unit="ratio",
                    count=stats.total,
                    current=stats.passed / stats.total if stats.total else 0.0,
                    sum=stats.passed,
                    console_group=MetricConsoleGroup.NONE,
                )
            )

        if self.total_evaluated > 0:
            results.append(
                MetricResult(
                    tag=ACCURACY_UNPARSED_TAG,
                    header="Accuracy Unparsed (Overall)",
                    unit="ratio",
                    count=self.total_evaluated,
                    current=self.overall_unparsed / self.total_evaluated,
                    sum=self.overall_unparsed,
                    console_group=MetricConsoleGroup.NONE,
                )
            )

        for task in sorted(self.per_task):
            stats = self.per_task[task]
            results.append(
                MetricResult(
                    tag=accuracy_unparsed_task_tag(task),
                    header=f"Accuracy Unparsed ({task})",
                    unit="ratio",
                    count=stats.total,
                    current=stats.unparsed / stats.total if stats.total else 0.0,
                    sum=stats.unparsed,
                    console_group=MetricConsoleGroup.NONE,
                )
            )

        return results


class ProcessAccuracyResult(AIPerfBaseModel):
    """Wire wrapper for a processed accuracy summary - mirrors ProcessServerMetricsResult."""

    results: AccuracySummary | None = Field(
        default=None, description="The processed accuracy summary"
    )
