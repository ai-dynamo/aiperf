# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-run and accuracy benchmarking configuration models.

Split out of ``models.py`` so the public module stays under the ergonomics
file-size cap. Re-exported via :mod:`aiperf.config.models`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field

from aiperf.common.enums import ConvergenceMode, ConvergenceStat
from aiperf.config._base import BaseConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, AccuracyGraderType


class MultiRunConfig(BaseConfig):
    """Configuration for multi-run benchmarking with statistical reporting.

    When num_runs > 1, AIPerf executes multiple benchmark runs and computes
    aggregate statistics (mean, std, confidence intervals) across runs.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    # Upper limit of 10 runs balances statistical validity with practical considerations:
    # - Statistical: 10 samples provide reasonable confidence intervals (t-distribution)
    # - Practical: Limits total benchmark time (10 runs can take hours for long benchmarks)
    # - Diminishing returns: Confidence interval width decreases with sqrt(n), so gains
    #   beyond 10 runs are marginal compared to the additional time investment
    # - Resource efficiency: Reduces compute/GPU costs while maintaining statistical rigor
    num_runs: Annotated[
        int,
        Field(
            ge=1,
            le=10,
            default=1,
            description="Number of profile runs to execute for confidence reporting. "
            "When 1, runs a single benchmark. "
            "When >1, computes aggregate statistics across runs.",
        ),
    ]

    cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            default=0.0,
            description="Cooldown duration in seconds between profile runs. "
            "Allows the system to stabilize between runs.",
        ),
    ]

    confidence_level: Annotated[
        float,
        Field(
            gt=0,
            lt=1,
            default=0.95,
            description="Confidence level for computing confidence intervals (0-1). "
            "Common values: 0.90 (90%%), 0.95 (95%%), 0.99 (99%%).",
        ),
    ]

    set_consistent_seed: Annotated[
        bool,
        Field(
            default=True,
            description="Auto-set random seed if not specified for workload consistency.",
        ),
    ]

    disable_warmup_after_first: Annotated[
        bool,
        Field(
            default=True,
            description="Disable warmup for runs after the first. "
            "When true, only the first run includes warmup for steady-state measurement.",
        ),
    ]

    convergence_metric: Annotated[
        str | None,
        Field(
            default=None,
            description="Target metric name for adaptive convergence stopping. "
            "When set, enables adaptive mode that stops early once the metric stabilizes.",
        ),
    ]

    convergence_stat: Annotated[
        ConvergenceStat,
        Field(
            default=ConvergenceStat.AVG,
            description="Statistic to evaluate for convergence when using ci_width or cv mode.",
        ),
    ]

    convergence_threshold: Annotated[
        float,
        Field(
            gt=0,
            lt=1,
            default=0.10,
            description="Threshold for convergence detection.",
        ),
    ]

    convergence_mode: Annotated[
        ConvergenceMode,
        Field(
            default=ConvergenceMode.CI_WIDTH,
            description="Statistical method for convergence detection (ci_width, cv, distribution).",
        ),
    ]


class AccuracyConfig(BaseConfig):
    """Configuration for accuracy benchmarking mode.

    When benchmark is set, enables accuracy evaluation alongside
    performance profiling using standard benchmarks (MMLU, AIME, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    benchmark: Annotated[
        AccuracyBenchmarkType | None,
        Field(
            default=None,
            description="Accuracy benchmark to run (e.g., mmlu, aime, hellaswag). "
            "When set, enables accuracy benchmarking mode alongside performance profiling.",
        ),
    ]

    tasks: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Specific tasks or subtasks within the benchmark to evaluate "
            "(e.g., specific MMLU subjects). If not set, all tasks are included.",
        ),
    ]

    n_shots: Annotated[
        int,
        Field(
            ge=0,
            le=8,
            default=0,
            description="Number of few-shot examples to include in the prompt. "
            "0 means zero-shot evaluation. Maximum 8.",
        ),
    ]

    enable_cot: Annotated[
        bool,
        Field(
            default=False,
            description="Enable chain-of-thought prompting for accuracy evaluation. "
            "Adds reasoning instructions to the prompt.",
        ),
    ]

    grader: Annotated[
        AccuracyGraderType | None,
        Field(
            default=None,
            description="Override the default grader for the selected benchmark "
            "(e.g., exact_match, math, multiple_choice, code_execution). "
            "If not set, uses the benchmark's default grader.",
        ),
    ]

    system_prompt: Annotated[
        str | None,
        Field(
            default=None,
            description="Custom system prompt to use for accuracy evaluation. "
            "Overrides any benchmark-specific system prompt.",
        ),
    ]

    verbose: Annotated[
        bool,
        Field(
            default=False,
            description="Enable verbose output for accuracy evaluation, "
            "showing per-problem grading details.",
        ),
    ]

    @property
    def enabled(self) -> bool:
        """Whether accuracy benchmarking mode is enabled."""
        return self.benchmark is not None
