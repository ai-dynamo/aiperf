# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the AIPerfSweep CRD.

AIPerfSweep is the parent CR that owns child AIPerfJob CRs and orchestrates
parameter sweeps and multi-run trials. The orchestration loop runs in a
dedicated sweep-controller pod, not in the kopf operator. See
docs/superpowers/specs/2026-04-25-k8s-sweeps-design.md.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field, model_validator

from aiperf.config._base import BaseConfig
from aiperf.config.sweep import SweepConfig

__all__ = [
    "AIPerfJobTemplate",
    "AIPerfSweepSpec",
    "ConvergenceConfig",
    "FailurePolicy",
    "MultiRunConfig",
]


class MultiRunConfig(BaseConfig):
    """Per-variation trial configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    trials: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Fixed trials per variation. Must be unset when `convergence` is set.",
    )
    cooldown_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Sleep duration between trials within a variation.",
    )
    auto_set_seed: bool = Field(
        default=True,
        description="Auto-set random seed for workload consistency across trials.",
    )
    disable_warmup_after_first: bool = Field(
        default=True,
        description="Skip warmup on trials 2..N for steady-state measurement.",
    )


class ConvergenceConfig(BaseConfig):
    """Per-variation adaptive early-stop configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    metric: str = Field(
        ...,
        description="Metric name from docs/metrics-reference.md (e.g., ttft_p99).",
    )
    criterion: Literal["cv_threshold"] = Field(
        default="cv_threshold",
        description="Convergence criterion. v1 supports cv_threshold only.",
    )
    min_runs: int = Field(
        default=3,
        ge=2,
        description="Minimum trials per variation before convergence is checked.",
    )
    max_runs: int = Field(
        default=10,
        ge=2,
        description="Maximum trials per variation; hard cap regardless of convergence.",
    )
    threshold: float = Field(
        default=0.05,
        gt=0,
        lt=1,
        description="Criterion threshold. For cv_threshold, the coefficient-of-variation cap.",
    )

    @model_validator(mode="after")
    def _validate_run_bounds(self) -> ConvergenceConfig:
        if self.min_runs > self.max_runs:
            raise ValueError(
                f"convergence.min_runs ({self.min_runs}) must be <= "
                f"convergence.max_runs ({self.max_runs}). Either lower min_runs "
                f"or raise max_runs to allow the convergence check to run."
            )
        return self


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
        description="0 = unbounded. Otherwise, terminate sweep when failed count reaches this value.",
    )


class AIPerfJobTemplate(BaseConfig):
    """Wrapper around an AIPerfJobSpec stamped onto every child."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="ObjectMeta merged into every child (labels, annotations).",
    )
    spec: dict[str, Any] = Field(
        ...,
        description="AIPerfJobSpec used as the child stamp. Must not contain sweep:/multi_run:.",
    )


class AIPerfSweepSpec(BaseConfig):
    """Top-level spec for an AIPerfSweep CR."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    sweep: SweepConfig | None = Field(
        default=None,
        description="Variation generator (grid | scenarios). Reuses aiperf.config.sweep.SweepConfig.",
    )
    multi_run: MultiRunConfig | None = Field(
        default=None,
        description="Per-variation trial configuration. Required when `convergence` is set.",
    )
    convergence: ConvergenceConfig | None = Field(
        default=None,
        description="Per-variation adaptive early-stop. Requires `multiRun`. Composes with `sweep`.",
    )
    failure_policy: FailurePolicy = Field(
        default_factory=FailurePolicy,
        description="When and whether to abort on child failure.",
    )
    cancel: bool = Field(
        default=False,
        description="Cooperative cancel: signals the current child and skips remaining variations.",
    )
    ttl_seconds_after_finished: int | None = Field(
        default=None,
        ge=0,
        description="Parent CR retention after terminal phase; children use their own TTL.",
    )
    template: AIPerfJobTemplate = Field(
        ...,
        description="Child stamp; spec is an AIPerfJobSpec.",
    )

    @model_validator(mode="after")
    def _validate_axis_combination(self) -> AIPerfSweepSpec:
        # Rule 1: at least one of sweep, multiRun, convergence must be set.
        if self.sweep is None and self.multi_run is None and self.convergence is None:
            raise ValueError(
                "AIPerfSweep requires at least one of `sweep`, `multiRun`, or `convergence`. "
                "For a single benchmark, use AIPerfJob via `aiperf kube profile`."
            )
        # Rule 2: convergence requires multiRun set, with multi_run.trials unset.
        if self.convergence is not None:
            if self.multi_run is None:
                raise ValueError(
                    "`convergence` requires `multiRun` to be set "
                    "(for cooldown/seed/warmup config)."
                )
            if self.multi_run.trials is not None:
                raise ValueError(
                    "`multiRun.trials` must be unset when `convergence` is set; "
                    "convergence.maxRuns governs the per-cell trial cap."
                )
        # Rule 4: template.spec must not contain sweep-axis keys at any
        # level the user can mistakenly nest them at: template.spec.{sweep,
        # multi_run, multiRun, convergence} OR template.spec.benchmark.{...}.
        # Sweep-axis keys belong at AIPerfSweep.spec, not stamped onto every
        # child.
        forbidden_keys = ("sweep", "multi_run", "multiRun", "convergence")
        template_spec = self.template.spec or {}
        for forbidden in forbidden_keys:
            if forbidden in template_spec:
                raise ValueError(
                    f"`template.spec.{forbidden}` is not permitted on AIPerfSweep. "
                    f"Set `spec.{forbidden}` at the top level instead."
                )
        benchmark = template_spec.get("benchmark") or {}
        for forbidden in forbidden_keys:
            if forbidden in benchmark:
                raise ValueError(
                    f"`template.spec.benchmark.{forbidden}` is not permitted on AIPerfSweep. "
                    f"Set `spec.{forbidden}` at the top level instead."
                )
        return self
