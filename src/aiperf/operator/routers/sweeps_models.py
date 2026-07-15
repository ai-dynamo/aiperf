# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic response models for the operator's AIPerfSweep router.

Schemas are deliberately a superset of the apiserver shapes; the router
synthesizes equivalent payloads for archived (PVC-only) sweeps so the
client never has to branch on ``source``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from aiperf.common.enums import SweepType
from aiperf.operator.routers.jobs_models import JobPodSummary


class DimensionInfo(BaseModel):
    """One swept dimension and the values it takes across the sweep."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="Dimension name (e.g. 'concurrency').")
    values: list[Any] = Field(
        description="Values the dimension takes across the sweep, in spec order."
    )


class SpecSummary(BaseModel):
    """Compact summary of the sweep's structural spec for the UI detail page."""

    model_config = ConfigDict(extra="forbid")
    sweep_type: SweepType = Field(description="Variation generator kind.")
    dimensions: list[DimensionInfo] = Field(
        description="Swept dimensions and their value lists."
    )
    multi_run: dict[str, Any] | None = Field(
        default=None,
        description="multiRun config snapshot (trials, cooldown, ...) or None.",
    )
    convergence: dict[str, Any] | None = Field(
        default=None,
        description="convergence config snapshot or None.",
    )


class SweepSummary(BaseModel):
    """One row in the /sweeps list response and embedded in detail."""

    model_config = ConfigDict(extra="forbid")
    namespace: str = Field(description="CR namespace.")
    name: str = Field(description="CR name.")
    source: Literal["live", "archived", "both"] = Field(
        description="Origin of the record: live CR, archived PVC dir, or both."
    )
    phase: str = Field(description="Parent phase.")
    total_variations: int = Field(description="Total variations from spec/aggregate.")
    completed_runs: int = Field(
        description="Sum of children in terminal-success phase."
    )
    failed_runs: int = Field(description="Sum of children in terminal-failure phase.")
    cancelled_runs: int = Field(
        default=0,
        description=(
            "Sum of children in terminal ``cancelled`` phase. Kept separate "
            "from ``failed_runs`` so user-cancelled children are not "
            "counted as failures. UIs gating on 'any non-success terminal' "
            "should sum ``failed_runs + cancelled_runs``."
        ),
    )
    age_seconds: int = Field(description="Seconds since CR/dir creation.")
    model: str | None = Field(
        default=None, description="Primary model name from template snapshot."
    )
    started_at: str | None = Field(
        default=None,
        description="ISO-8601 ``status.startedAt`` stamped by the operator on phase transition.",
    )
    completed_at: str | None = Field(
        default=None,
        description="ISO-8601 ``status.completedAt`` stamped by the operator at terminal phase.",
    )
    api_url: str | None = Field(
        default=None,
        description="Operator-side API base URL for cross-process result fetches.",
    )
    results_available: bool = Field(
        default=False,
        description="True once the operator has stamped ``status.resultsAvailable``.",
    )
    current_child_ref: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Pointer to the in-flight child for live drill-down. Shape: "
            "``{name, index, label}``. Null when no child is active."
        ),
    )
    run_states: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Per-state run counts rolled up from children. Keys: ``pending``, "
            "``running``, ``completed``, ``failed``, ``cancelled``."
        ),
    )


class SweepListResponse(BaseModel):
    """Body of GET /api/v1/sweeps."""

    model_config = ConfigDict(extra="forbid")
    sweeps: list[SweepSummary] = Field(default_factory=list)


class ChildJobRef(BaseModel):
    """Pointer to a child AIPerfJob inside a cell's children list."""

    model_config = ConfigDict(extra="forbid")
    namespace: str
    name: str
    trial_index: int | None = None
    phase: str | None = None


class CellEntry(BaseModel):
    """One sweep cell (variation) with per-cell aggregates and child links."""

    model_config = ConfigDict(extra="forbid")
    variation_index: int = Field(description="Index from expand_sweep().")
    variation_label: str = Field(description="Human-readable variation label.")
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured dimension values for this cell.",
    )
    trials_completed: int = Field(default=0)
    trials_failed: int = Field(default=0)
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="metric_name -> stat_name -> value for this cell.",
    )
    children: list[ChildJobRef] = Field(default_factory=list)


class CellAggregatesResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}/cells."""

    model_config = ConfigDict(extra="forbid")
    dimensions: list[DimensionInfo] = Field(default_factory=list)
    cells: list[CellEntry] = Field(default_factory=list)
    source: Literal["live", "archived", "both"] = Field(
        description="Origin of the cell data: live (synthesized from per-child summaries), "
        "archived (read from aggregate.json), or both."
    )


class SweepDetailResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}."""

    model_config = ConfigDict(extra="forbid")
    sweep: SweepSummary
    status: dict[str, Any] = Field(default_factory=dict)
    spec_summary: SpecSummary
    children: list[dict[str, Any]] = Field(
        default_factory=list,
        description="ActiveJobSummary dicts (alias-keyed) for the sweep's children.",
    )
    pods: list[JobPodSummary] = Field(
        default_factory=list,
        description=(
            "Sweep-controller pod summaries (one row per pod under the sweep's "
            "JobSet, identified by ``jobset.sigs.k8s.io/jobset-name=aiperf-<name>``). "
            "Empty for archived sweeps whose CR has been deleted, since the "
            "controller pod is also gone in that state."
        ),
    )


class SweepEpochSummary(BaseModel):
    """One epoch entry in a sweep's history listing."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    epoch: str = Field(description="Sweep run epoch, decimal seconds.")
    is_latest: bool = Field(
        description="True iff this epoch matches the sweep's latest.txt pointer."
    )
    mtime_epoch: int = Field(
        description="Filesystem mtime of the epoch dir, seconds since epoch."
    )
    file_count: int = Field(
        description="Number of immediate children under the epoch dir."
    )


class SweepEpochsResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}/epochs."""

    model_config = ConfigDict(extra="forbid")
    epochs: list[SweepEpochSummary] = Field(default_factory=list)


class ChildrenManifestEntry(BaseModel):
    """One row in the per-epoch children manifest."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    namespace: str = Field(description="Child AIPerfJob namespace.")
    name: str = Field(description="Child AIPerfJob CR name.")
    variation_index: int = Field(description="Variation index from expand_sweep().")
    variation_label: str = Field(
        default="", description="Human-readable variation label."
    )
    trial_index: int | None = Field(
        default=None, description="Trial index within the variation, if multi-trial."
    )
    child_run_epoch: str = Field(
        description="Child job run epoch on disk (decimal seconds)."
    )


class ChildrenManifestResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}/children."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    sweep_run_epoch: str = Field(description="Sweep epoch this manifest belongs to.")
    children: list[ChildrenManifestEntry] = Field(default_factory=list)
