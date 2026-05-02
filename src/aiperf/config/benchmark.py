# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BenchmarkPlan and BenchmarkRun models for sweep/multi-run orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiperf.common.enums import (
    ConvergenceMode,
    ConvergenceStat,
    GPUTelemetryMode,
    SweepMode,
)
from aiperf.config.adaptive_search import AdaptiveSearchConfig, SLAFilter
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import SweepVariation
from aiperf.config.zmq import BaseZMQCommunicationConfig
from aiperf.plugin.enums import CustomDatasetType, DatasetSamplingStrategy
from aiperf.search_recipes._base import PostProcessSpec


class BenchmarkPlan(BaseModel):
    """Output of config loading: expanded configs + execution preferences.

    For a simple config with no sweep/multi_run, contains a single config
    and trials=1. For sweeps, contains one config per variation.
    The orchestrator iterates configs x trials.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    configs: list[BenchmarkConfig] = Field(
        description="Expanded benchmark configs, one per sweep variation.",
        min_length=1,
    )
    variations: list[SweepVariation] = Field(
        default_factory=list,
        description="Parallel to configs: metadata per sweep variation.",
    )
    trials: Annotated[
        int,
        Field(
            ge=1,
            le=20,
            default=1,
            description="Number of trials per config (from multi_run.num_runs).",
        ),
    ]
    cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            default=0.0,
            description="Cooldown between runs in seconds.",
        ),
    ]
    confidence_level: Annotated[
        float,
        Field(
            gt=0,
            lt=1,
            default=0.95,
            description="Confidence level for aggregate statistics.",
        ),
    ]
    set_consistent_seed: Annotated[
        bool,
        Field(
            default=True,
            description="Auto-set random seed for workload consistency.",
        ),
    ]
    disable_warmup_after_first: Annotated[
        bool,
        Field(
            default=True,
            description="Disable warmup for runs after the first.",
        ),
    ]
    convergence_metric: str | None = Field(
        default=None,
        description="Target metric name for adaptive convergence stopping. "
        "When set, enables adaptive mode that stops early once the metric stabilizes.",
    )
    convergence_stat: ConvergenceStat = Field(
        default=ConvergenceStat.AVG,
        description="Statistic to evaluate for convergence (avg, p50, p90, etc.).",
    )
    convergence_threshold: Annotated[
        float,
        Field(
            gt=0,
            lt=1,
            default=0.10,
            description="Threshold for convergence detection.",
        ),
    ]
    convergence_mode: ConvergenceMode = Field(
        default=ConvergenceMode.CI_WIDTH,
        description="Statistical method for convergence detection.",
    )
    export_level: str = Field(
        default="summary",
        description="Export level for record-level data (summary, records, raw).",
    )
    export_jsonl_file: str | None = Field(
        default=None,
        description="Path to JSONL export file for distribution convergence mode.",
    )
    # Sweep-specific config from kubernetes.sweep_models. Typed Any here
    # because the config layer must not import from the kubernetes layer
    # at runtime (one-way layering: kubernetes -> config). Validated at
    # the AIPerfSweepSpec boundary; attached by sweep_controller.plan_builder.
    failure_policy: Any = Field(
        default=None,
        description="Sweep failure-policy config (FailurePolicy); None for non-sweep plans.",
    )
    convergence_config: Any = Field(
        default=None,
        description="Sweep convergence config (ConvergenceConfig); None for non-sweep plans.",
    )
    parameter_sweep_cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            default=0.0,
            description=(
                "Cooldown seconds to sleep between sweep variations. "
                "Honored by MultiRunOrchestrator.execute when iterating "
                "plan.configs."
            ),
        ),
    ]
    parameter_sweep_same_seed: bool = Field(
        default=False,
        description=(
            "If true, every sweep variation reuses the same seed for "
            "correlated comparisons. If false (default), each variation "
            "derives a unique seed (base_seed + variation.index)."
        ),
    )
    parameter_sweep_mode: SweepMode = Field(
        default=SweepMode.REPEATED,
        description=(
            "Iteration order for sweep + multi-trial. 'repeated' "
            "(default) iterates trials outer, variations inner. "
            "'independent' iterates variations outer, trials inner. "
            "Dispatched in MultiRunOrchestrator.execute."
        ),
    )
    adaptive_search: AdaptiveSearchConfig | None = Field(
        default=None,
        description=(
            "Adaptive outer-loop configuration (e.g. Bayesian Optimization). "
            "None for non-adaptive plans. When set, MultiRunOrchestrator.execute "
            "dispatches to execute_adaptive_search instead of grid-mode paths."
        ),
    )
    post_process: PostProcessSpec | None = Field(
        default=None,
        description=(
            "Optional PostProcessSpec emitted by a grid Search Recipe. Carried "
            "through from MultiRunConfig.post_process; consumed by "
            "aggregate_sweep_and_export to write a derived artifact under "
            "sweep_aggregate/."
        ),
    )
    sla_filters: list[SLAFilter] = Field(
        default_factory=list,
        description=(
            "SLA filters threaded from a grid Search Recipe into "
            "SweepAnalyzer.compute. BO recipes carry SLA filters on "
            "AdaptiveSearchConfig.sla_filters instead."
        ),
    )

    @property
    def use_adaptive(self) -> bool:
        """True if convergence-based adaptive stopping is configured."""
        return self.convergence_metric is not None

    @property
    def is_single_run(self) -> bool:
        """True if this plan has exactly one config and one trial."""
        return len(self.configs) == 1 and self.trials <= 1

    @property
    def is_sweep(self) -> bool:
        """True when build_benchmark_plan expanded a sweep (multiple variations).

        Used by _cli_runner_helpers and the operator-mode gate to detect
        sweep-in-flight without re-counting plan.configs at every callsite.
        """
        return len(self.configs) > 1

    @property
    def is_adaptive_search(self) -> bool:
        """True when an adaptive outer loop (BO) is configured.

        Distinct from is_sweep (which checks for a multi-variation grid).
        Sweep-aware code paths continue to branch on is_sweep without change;
        outer-loop dispatch is handled separately in
        MultiRunOrchestrator.execute. Both can be False (single-point run).
        Both being True is forbidden by build_benchmark_plan.
        """
        return self.adaptive_search is not None

    @model_validator(mode="after")
    def _check_repeated_incompatible_with_convergence(self) -> BenchmarkPlan:
        if self.use_adaptive and self.parameter_sweep_mode == SweepMode.REPEATED:
            raise ValueError(
                "parameter_sweep_mode='repeated' is incompatible with "
                "convergence-based stopping (--convergence-metric). The "
                "trial-outer iteration order has no place to evaluate "
                "convergence per-cell. Use 'independent' for adaptive "
                "sweeps, or remove --convergence-metric."
            )
        return self


class ResolvedConfig(BaseModel):
    """Runtime-computed state populated after construction.

    Holds values discovered or computed during startup that don't belong
    in the static YAML config. Accessed via ``run.resolved``.
    """

    tokenizer_names: dict[str, str] | None = Field(
        default=None,
        description="Mapping of model names to resolved tokenizer names. "
        "Used by services to skip redundant alias resolution.",
    )
    gpu_telemetry_mode: GPUTelemetryMode = Field(
        default=GPUTelemetryMode.SUMMARY,
        description="Resolved GPU telemetry mode. "
        "Set at runtime based on telemetry discovery.",
    )
    artifact_dir_created: bool = Field(
        default=False,
        description="Whether the artifact directory tree has been created.",
    )
    dataset_file_paths: dict[str, Path] | None = Field(
        default=None,
        description="Validated absolute paths for file-based datasets. "
        "Used by dataset composers to skip redundant path validation.",
    )
    total_expected_duration: float | None = Field(
        default=None,
        description="Sum of phase durations in seconds. "
        "None if any phase lacks a duration.",
    )
    gpu_custom_metrics: list[tuple] | None = Field(
        default=None,
        description="Parsed custom GPU metrics from CSV. "
        "Cached to avoid re-parsing in child processes.",
    )
    gpu_dcgm_mappings: dict[str, str] | None = Field(
        default=None,
        description="DCGM field-to-metric-name mappings from custom CSV. "
        "Cached to avoid re-parsing in child processes.",
    )
    dataset_types: dict[str, CustomDatasetType] | None = Field(
        default=None,
        description="Detected CustomDatasetType per file-based dataset name. "
        "Resolved via can_load detection or explicit format mapping.",
    )
    dataset_sampling_strategies: dict[str, DatasetSamplingStrategy] | None = Field(
        default=None,
        description="Resolved DatasetSamplingStrategy per dataset name. "
        "Uses loader's preferred strategy when config uses the default.",
    )
    dataset_has_timing_data: dict[str, bool] | None = Field(
        default=None,
        description="Whether each file-based dataset has timing data "
        "(timestamps/delays). Determined by inspecting the first "
        "record for timestamp or delay fields.",
    )
    dataset_total_records: dict[str, int] | None = Field(
        default=None,
        description="Total non-empty line count per file-based dataset. "
        "Used for total_expected_requests in fixed_schedule phases.",
    )
    dataset_session_count: dict[str, int] | None = Field(
        default=None,
        description="Unique session/conversation count per file-based dataset. "
        "For single-turn: equals total_records. For multi-turn: "
        "count of unique session_id values.",
    )
    comm_config: BaseZMQCommunicationConfig | None = Field(
        default=None,
        description="Pre-built ZMQ communication config. "
        "Avoids rebuilding in every service's CommunicationMixin.",
    )


class BenchmarkRun(BaseModel):
    """Per-iteration wrapper: single config + run metadata.

    Built by the orchestrator for each (variation, trial) pair.
    Serialized to JSON for subprocess execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    benchmark_id: str = Field(description="Unique ID for this benchmark run.")
    cfg: BenchmarkConfig = Field(description="The benchmark config for this run.")
    variation: SweepVariation | None = Field(
        default=None, description="Sweep variation metadata, if applicable."
    )
    trial: Annotated[
        int,
        Field(ge=0, default=0, description="Zero-based trial index."),
    ]
    artifact_dir: Path = Field(description="Directory for this run's artifacts.")
    label: str = Field(default="", description="Human-readable run label.")
    resolved: ResolvedConfig = Field(
        default_factory=ResolvedConfig,
        description="Runtime-computed state populated after construction.",
    )

    @property
    def comm_config(self) -> BaseZMQCommunicationConfig:
        """Build the ZMQ communication config for this run.

        Delegates to config.comm_config which caches the result.
        """
        return self.cfg.comm_config
