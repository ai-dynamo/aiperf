# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

from pydantic import Field
from typing_extensions import TypedDict

from aiperf.common.models.base_models import AIPerfBaseModel

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
