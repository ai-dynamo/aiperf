# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BenchmarkPlan and BenchmarkRun models for sweep/multi-run orchestration."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiperf.common.enums import (
    GPUTelemetryMode,
    SweepMode,
)
from aiperf.common.redact import build_cli_command
from aiperf.config.base import BaseConfig
from aiperf.config.comm import BaseZMQCommunicationConfig
from aiperf.config.config import BenchmarkConfig
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    SweepConfig,
    SweepVariation,
    _GridSweepBase,
)
from aiperf.config.sweep.multi_run import MultiRunConfig
from aiperf.plugin.enums import (
    CustomDatasetType,
    DatasetSamplingStrategy,
)

# `aiperf.config.plot.PlotEnvelopeConfig` is the runtime type of `plot`, but
# importing it at module top would cycle: aiperf.config.plot ->
# aiperf.plot.core -> aiperf.common.mixins -> aiperf.common.messages, which
# can be mid-init when this module loads. The field is typed ``Any``; the
# build_benchmark_plan copy step preserves the typed instance from
# AIPerfConfig.plot.


def _new_uuid() -> str:
    return str(uuid.uuid4())


# Inlined from aiperf.kubernetes.sweep_models to keep aiperf.config free of
# kubernetes imports. Identical surface; orchestrator/cluster code that needs
# to share this type can re-import it from here.
class FailurePolicy(BaseConfig):
    """Failure handling policy for the sweep."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    on_child_failure: Literal["continue", "abort"] = Field(
        default="continue",
        description=(
            "continue: failed child becomes a status entry, advance to next variation. "
            "abort: any failure terminates the sweep with phase=Failed."
        ),
    )
    max_failures: int = Field(
        default=0,
        ge=0,
        description=(
            "Hard failure budget for the entire sweep. 0 = unbounded "
            "(no early-abort on count). When >0, the orchestrator stops "
            "scheduling new children once failedRuns >= maxFailures and "
            "the sweep terminates with phase=Failed. Independent of "
            "terminal-phase resolution: a sweep with 0 < failedRuns < total "
            "and the threshold not exceeded ends as PartiallyFailed."
        ),
    )


class BenchmarkPlan(BaseModel):
    """Output of config loading: expanded configs + execution preferences.

    For a simple config with no sweep/multi_run, contains a single config
    and trials=1. For sweeps, contains one config per variation.
    The orchestrator iterates configs x trials.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    sweep_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this outer sweep (UUID4). "
        "Stable across all variations and trials of one plan; threaded "
        "into every BenchmarkRun.sweep_id so every per-run JSON export "
        "carries the same sweep_id. A non-sweep / single-run plan still "
        "gets a fresh sweep_id (sweep of size 1).",
    )
    configs: list[BenchmarkConfig] = Field(
        description="Expanded benchmark configs, one per sweep variation.",
        min_length=1,
    )
    variations: list[SweepVariation] = Field(
        default_factory=list,
        description="Parallel to configs: metadata per sweep variation.",
    )
    variation_seeds: list[int | None] = Field(
        default_factory=list,
        description="Per-variation random seed (None when no base seed set). "
        "Length matches `configs`/`variations`. Variation 0 inherits the "
        "envelope `random_seed`; for grid/scenario/zip sweeps variation N "
        "gets `random_seed + N` unless `sweep.same_seed=True`. "
        "Adaptive/Sobol/LHS sweeps always derive `+N`; non-sweep plans get "
        "the base seed verbatim.",
    )
    trials: Annotated[
        int,
        Field(
            ge=1,
            le=10,
            default=1,
            description="Number of trials per config (from multi_run.num_runs).",
        ),
    ]
    cooldown_seconds: Annotated[
        float,
        Field(
            ge=0,
            le=86400,
            default=0.0,
            description="Cooldown between runs in seconds. Capped at 24h to "
            "surface typos like `1e18` at config-load time rather than "
            "blocking the orchestrator inside ``asyncio.sleep``.",
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
    no_sweep_table: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to suppress the per-cell streaming sweep "
            "table emitted via the AIPerf logger during sweeps. Forwarded "
            "from AIPerfConfig at plan-build time so cli_runner._execute_multi_benchmark "
            "can decide whether to wire up SweepTableLogger.",
        ),
    ]
    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            ge=0,
            description="Envelope-level base random seed. Per-variation seeds "
            "live on `variation_seeds`.",
        ),
    ]
    variables: Annotated[
        dict[str, object],
        Field(
            default_factory=dict,
            description="Envelope-level Jinja substitution variables, captured "
            "for downstream reporting / artifact metadata.",
        ),
    ]
    plot: Annotated[
        Any,
        Field(
            default=None,
            description=(
                "Envelope-level plot configuration applied to every variation's "
                "per-run plot pass and the cross-variation aggregate pass. "
                "Mirrors AIPerfConfig.plot after Form-A resolution. None means "
                "the existing fallback chain (~/.aiperf/plot_config.yaml -> "
                "shipped default) applies. Runtime type: PlotEnvelopeConfig | None."
            ),
        ),
    ]
    export_level: str = Field(
        default="summary",
        description="Export level for record-level data (summary, records, raw).",
    )
    export_jsonl_file: str | None = Field(
        default=None,
        description="Path to JSONL export file for distribution convergence mode.",
    )
    multi_run: Annotated[
        MultiRunConfig,
        Field(
            default_factory=MultiRunConfig,
            description="Trial mechanics for each variation (num_runs, convergence, etc.).",
        ),
    ]
    sweep: Annotated[
        SweepConfig | None,
        Field(
            default=None,
            description=(
                "Sweep configuration when this plan was built from an envelope with a "
                "`sweep:` block. None means single-point (no sweep)."
            ),
        ),
    ]
    failure_policy: Annotated[
        FailurePolicy | None,
        Field(
            default=None,
            description="Failure handling policy (only meaningful in cluster context).",
        ),
    ]

    @property
    def use_adaptive(self) -> bool:
        """True if convergence-based adaptive trial stopping is configured."""
        return self.multi_run.convergence is not None

    @property
    def is_single_run(self) -> bool:
        """True if this plan executes exactly one profile run end-to-end.

        Adaptive search (BO) plans carry a single starting config but the
        planner mutates it across iterations — those are NOT single runs and
        must dispatch to the multi-run orchestrator so the planner is wired.
        """
        return (
            len(self.configs) == 1 and self.trials == 1 and not self.is_adaptive_search
        )

    @property
    def is_sweep(self) -> bool:
        """True when build_benchmark_plan expanded a sweep (multiple variations).

        Used by cli_runner._multi_run and the operator-mode gate to detect
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
        Both being True cannot arise from build_benchmark_plan, which emits
        a single config for adaptive_search runs.
        """
        return isinstance(self.sweep, AdaptiveSearchSweep)

    @model_validator(mode="after")
    def _check_repeated_incompatible_with_convergence(self) -> Self:
        if (
            self.use_adaptive
            and isinstance(self.sweep, _GridSweepBase)
            and self.sweep.iteration_order == SweepMode.REPEATED
        ):
            raise ValueError(
                "iteration_order='repeated' is incompatible with adaptive trial "
                "convergence (multi_run.convergence). Use 'independent' instead."
            )
        return self

    @model_validator(mode="after")
    def _check_trials_matches_num_runs_cap(self) -> Self:
        """Reject ``trials`` exceeding what user-facing ``num_runs`` allows.

        ``BenchmarkPlan.trials`` accepts ``le=10`` and ``MultiRunConfig.num_runs``
        accepts ``le=10`` — the two caps must stay aligned so a programmatic
        plan can't bypass user-facing config validation by writing a
        higher trial count directly into the plan. This validator surfaces a
        cap drift as a config error instead of silently honoring it.
        """
        num_runs_cap = MultiRunConfig.model_fields["num_runs"].metadata
        # Pull the ``le=...`` constraint from the field metadata; bail if
        # the cap is not declared (defensive — existing schema sets it).
        cap_value: int | None = None
        for entry in num_runs_cap:
            value = getattr(entry, "le", None)
            if value is not None:
                cap_value = int(value)
                break
        if cap_value is not None and self.trials > cap_value:
            raise ValueError(
                f"trials ({self.trials}) exceeds MultiRunConfig.num_runs cap "
                f"({cap_value}). Lower trials or update both caps in lockstep."
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
    dataset_root_count: dict[str, Annotated[int, Field(ge=0)]] | None = Field(
        default=None,
        description="Root-conversation count per file-based dataset. "
        "For forking datasets (dag_jsonl): sessions not referenced by any "
        "other session's forks/spawns/pre_session_spawns lists — these are "
        "the entries the loader actually samples standalone. "
        "Other dataset types are not populated.",
    )
    dataset_is_forking: dict[str, bool] | None = Field(
        default=None,
        description="Whether each file-based dataset can fork (DAG branches). "
        "Today only dag_jsonl carries fork semantics; other custom datasets "
        "(single_turn, multi_turn, mooncake_trace, ...) are linear.",
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
    sweep_id: str | None = Field(
        default=None,
        description="UUID of the outer sweep this run belongs to "
        "(BenchmarkPlan.sweep_id). None when the run was constructed "
        "outside the multi-run orchestrator (e.g. ad-hoc CLI single-run "
        "paths that don't go through BenchmarkPlan).",
    )
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
    cli_command: str | None = Field(
        default_factory=build_cli_command,
        description="The redacted CLI command that launched this run, captured "
        "from sys.argv at construction time. Surfaced into RunInfo so a reader "
        "of `profile_export_aiperf.json` alone can see how the benchmark was "
        "invoked. None when constructed outside a CLI context where sys.argv "
        "has nothing to redact (e.g., a programmatic test fixture that passes "
        "cli_command=None explicitly).",
    )
    random_seed: int | None = Field(
        default=None,
        ge=0,
        description="Random seed for this run (sourced from "
        "BenchmarkPlan.variation_seeds at construction time). None when "
        "no envelope-level random_seed was set.",
    )
    variables: dict[str, Any] = Field(
        default_factory=dict,
        description="Envelope-level Jinja substitution variables propagated "
        "from AIPerfConfig.variables, exposed to run-time renderers (e.g. "
        "artifacts.user_files). Empty dict when no envelope-level variables "
        "were set.",
    )
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
