# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

from pydantic import Field

from aiperf.common.config.base_config import BaseConfig
from aiperf.common.config.cli_parameter import CLIParameter
from aiperf.common.config.groups import Groups
from aiperf.plugin.enums import AccuracyBenchmarkType, AccuracyGraderType


class AccuracyConfig(BaseConfig):
    """Configuration for accuracy benchmarking mode."""

    _CLI_GROUP = Groups.ACCURACY

    benchmark: Annotated[
        AccuracyBenchmarkType | None,
        Field(
            description="Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). "
            "When set, enables accuracy benchmarking mode alongside performance profiling.",
        ),
        CLIParameter(
            name=("--accuracy-benchmark",),
            group=_CLI_GROUP,
        ),
    ] = None

    tasks: Annotated[
        list[str] | None,
        Field(
            description="Specific tasks or subtasks within the benchmark to evaluate "
            "(e.g., specific MMLU subjects). If not set, all tasks are included.",
        ),
        CLIParameter(
            name=("--accuracy-tasks",),
            group=_CLI_GROUP,
        ),
    ] = None

    n_shots: Annotated[
        int,
        Field(
            ge=0,
            le=8,
            description="Number of few-shot examples to include in the prompt. "
            "0 means zero-shot evaluation. Maximum 8.",
        ),
        CLIParameter(
            name=("--accuracy-n-shots",),
            group=_CLI_GROUP,
        ),
    ] = 0

    enable_cot: Annotated[
        bool,
        Field(
            description="Enable chain-of-thought prompting for accuracy evaluation. "
            "Adds reasoning instructions to the prompt.",
        ),
        CLIParameter(
            name=("--accuracy-enable-cot",),
            group=_CLI_GROUP,
        ),
    ] = False

    grader: Annotated[
        AccuracyGraderType | None,
        Field(
            description="Override the default grader for the selected benchmark "
            "(e.g., exact_match, math, multiple_choice, code_execution). "
            "If not set, uses the benchmark's default grader.",
        ),
        CLIParameter(
            name=("--accuracy-grader",),
            group=_CLI_GROUP,
        ),
    ] = None

    system_prompt: Annotated[
        str | None,
        Field(
            description="Custom system prompt to use for accuracy evaluation. "
            "Overrides any benchmark-specific system prompt.",
        ),
        CLIParameter(
            name=("--accuracy-system-prompt",),
            group=_CLI_GROUP,
        ),
    ] = None

    verbose: Annotated[
        bool,
        Field(
            description="Enable verbose output for accuracy evaluation, "
            "showing per-problem grading details.",
        ),
        CLIParameter(
            name=("--accuracy-verbose",),
            group=_CLI_GROUP,
        ),
    ] = False

    @property
    def enabled(self) -> bool:
        """Whether accuracy benchmarking mode is enabled."""
        return self.benchmark is not None
