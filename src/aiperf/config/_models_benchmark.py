# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-run and accuracy benchmarking configuration models.

Split out of ``models.py`` so the public module stays under the ergonomics
file-size cap. Re-exported via :mod:`aiperf.config.models`.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import ConfigDict, Field, field_validator

from aiperf.common.enums import ConvergenceMode, ConvergenceStat, SweepMode
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

    parameter_sweep_cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            default=0.0,
            description="Cooldown duration in seconds between sweep variations. "
            "Honored by MultiRunOrchestrator when iterating plan.configs (one per "
            "sweep variation). Distinct from cooldown_seconds, which gates trial-"
            "to-trial cooldown within a single variation. Surfaced via "
            "--parameter-sweep-cooldown-seconds.",
        ),
    ]

    parameter_sweep_same_seed: Annotated[
        bool,
        Field(
            default=False,
            description="If true, every sweep variation reuses the same random seed "
            "(correlated comparisons). If false (default), each variation derives a "
            "unique seed `base_seed + variation.index`. Surfaced via "
            "--parameter-sweep-same-seed; the cross-field validator on AIPerfConfig "
            "requires --random-seed when true.",
        ),
    ]

    mode: Annotated[
        SweepMode,
        Field(
            default=SweepMode.REPEATED,
            description="Iteration order for sweep + multi-trial composition. "
            "'repeated' (default): trials outer, variations inner - "
            "artifact tree is <base>/profile_runs/trial_NNNN/<variation>/profile_runs/run_NNNN/. "
            "'independent': variations outer, trials inner - artifact tree is "
            "<base>/<variation>/profile_runs/run_NNNN/. "
            "Both modes produce the same total runs and same sweep_aggregate/ "
            "output. Adaptive convergence (--convergence-metric) is "
            "incompatible with 'repeated'.",
        ),
    ]

    adaptive_search: Annotated[
        Any,
        Field(
            default=None,
            description=(
                "Adaptive outer-loop configuration (Bayesian Optimization). "
                "Typed AdaptiveSearchConfig but expressed as Any to avoid a "
                "circular import between aiperf.config and aiperf.orchestrator "
                "(adaptive_search.py imports OptimizationDirection). The "
                "field_validator below coerces dicts to AdaptiveSearchConfig at "
                "validation time. Set by the v1 converter when --search-* flags "
                "are present. Mutually exclusive with the top-level `sweep` "
                "block; build_benchmark_plan enforces that. When set, "
                "MultiRunOrchestrator.execute dispatches to "
                "execute_adaptive_search instead of grid-mode paths."
            ),
        ),
    ] = None

    @field_validator("adaptive_search", mode="before")
    @classmethod
    def _coerce_adaptive_search(cls, value: Any) -> Any:
        # Lazy-import AdaptiveSearchConfig at validation time -- declaring it as
        # a top-level type would create a cycle with `aiperf.orchestrator`.
        if value is None:
            return None
        from aiperf.config.adaptive_search import AdaptiveSearchConfig

        if isinstance(value, AdaptiveSearchConfig):
            return value
        if isinstance(value, dict):
            return AdaptiveSearchConfig.model_validate(value)
        return value

    post_process: Annotated[
        Any,
        Field(
            default=None,
            description=(
                "Optional PostProcessSpec emitted by a grid Search Recipe. Threads "
                "through to BenchmarkPlan and is consumed by aggregate_sweep_and_export "
                "to produce a derived artifact (e.g. a TTFT-vs-ISL curve fit). "
                "Typed Any to avoid a circular import with aiperf.search_recipes; "
                "the field_validator below coerces dicts to PostProcessSpec at "
                "validation time."
            ),
        ),
    ] = None

    @field_validator("post_process", mode="before")
    @classmethod
    def _coerce_post_process(cls, value: Any) -> Any:
        if value is None:
            return None
        from aiperf.search_recipes._base import PostProcessSpec

        if isinstance(value, PostProcessSpec):
            return value
        if isinstance(value, dict):
            return PostProcessSpec.model_validate(value)
        return value

    sla_filters: Annotated[
        list[Any],
        Field(
            default_factory=list,
            description=(
                "SLA filters threaded from a grid Search Recipe into "
                "SweepAnalyzer.compute. BO recipes carry SLA filters on "
                "AdaptiveSearchConfig.sla_filters instead; this field is the grid "
                "carrier. Typed list[Any] to avoid a circular import with "
                "aiperf.config.adaptive_search; the field_validator below coerces "
                "dict items to SLAFilter."
            ),
        ),
    ]

    @field_validator("sla_filters", mode="before")
    @classmethod
    def _coerce_sla_filters(cls, value: Any) -> Any:
        if value is None:
            return []
        from aiperf.config.adaptive_search import SLAFilter

        if not isinstance(value, list):
            return value
        coerced: list[Any] = []
        for item in value:
            if isinstance(item, SLAFilter):
                coerced.append(item)
            elif isinstance(item, dict):
                coerced.append(SLAFilter.model_validate(item))
            else:
                coerced.append(item)
        return coerced


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
            description="Accuracy benchmark to run. When set, enables accuracy "
            "benchmarking alongside performance profiling. AIME variants: 'aime' "
            "is the legacy combined set (deprecated for new runs); prefer the "
            "year-pinned 'aime24' or 'aime25' for reproducibility.",
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
