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
from aiperf.common.finite import FiniteFloat
from aiperf.operator.routers.jobs_models import JobPodSummary

# Safety net on the projected Pareto front. A multi-objective run's front is
# bounded by its iteration count, so this only bites pathological configs; it
# exists so one sweep cannot make the polled detail route unbounded in size.
# Mirrors the ``status.runs[]`` truncation marker on AIPerfSweep.
MAX_BEST_TRIALS = 20


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
    objectives: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Adaptive-search objectives (metric / stat / direction), or None for "
            "generator sweep types that have no declared objective. This is what "
            "defines the sweep's winner -- without it the UI has to guess from "
            "whichever metric the chart happens to be showing."
        ),
    )
    sla_filters: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "SLA constraints applied at trial scoring time (metricTag / stat / op / "
            "threshold), or None. A variation that breaches any of these is "
            "infeasible and must not be reported as the winner however good its "
            "objective value looks."
        ),
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
    total_variations: int = Field(
        ge=0, description="Total variations from spec/aggregate."
    )
    completed_runs: int = Field(
        ge=0, description="Sum of children in terminal-success phase."
    )
    failed_runs: int = Field(
        ge=0, description="Sum of children in terminal-failure phase."
    )
    cancelled_runs: int = Field(
        default=0,
        ge=0,
        description=(
            "Sum of children in terminal ``cancelled`` phase. Kept separate "
            "from ``failed_runs`` so user-cancelled children are not "
            "counted as failures. UIs gating on 'any non-success terminal' "
            "should sum ``failed_runs + cancelled_runs``."
        ),
    )
    age_seconds: int = Field(ge=0, description="Seconds since CR/dir creation.")
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
    namespace: str = Field(description="Child AIPerfJob namespace.")
    name: str = Field(description="Child AIPerfJob CR name.")
    trial_index: int | None = Field(
        default=None,
        ge=0,
        description="Trial index within the variation; None when single-trial.",
    )
    phase: str | None = Field(
        default=None, description="Child lifecycle phase; None when not yet observed."
    )


class CellEntry(BaseModel):
    """One sweep cell (variation) with per-cell aggregates and child links."""

    model_config = ConfigDict(extra="forbid")
    variation_index: int = Field(ge=0, description="Index from expand_sweep().")
    variation_label: str = Field(description="Human-readable variation label.")
    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured dimension values for this cell.",
    )
    trials_completed: int = Field(
        default=0,
        ge=0,
        description="Trials for this cell that reached terminal success.",
    )
    trials_failed: int = Field(
        default=0,
        ge=0,
        description="Trials for this cell that reached terminal failure.",
    )
    metrics: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="metric_name -> stat_name -> value for this cell.",
    )
    children: list[ChildJobRef] = Field(
        default_factory=list, description="Child AIPerfJob pointers for this cell."
    )


class CellAggregatesResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}/cells."""

    model_config = ConfigDict(extra="forbid")
    dimensions: list[DimensionInfo] = Field(default_factory=list)
    cells: list[CellEntry] = Field(default_factory=list)
    source: Literal["live", "archived", "both"] = Field(
        description="Origin of the cell data: live (synthesized from per-child summaries), "
        "archived (read from aggregate.json), or both."
    )


class SearchObjective(BaseModel):
    """One optimization target, as recorded in ``search_history.json``'s config."""

    model_config = ConfigDict(extra="forbid")
    metric: str = Field(
        description="Metric tag being optimized, e.g. 'request_throughput'."
    )
    stat: str = Field(description="Statistic on the metric ('avg', 'p50', 'p95', ...).")
    direction: str = Field(
        description=(
            "'maximize' or 'minimize'. Lowercased here on purpose: the artifact "
            "serializes ``OptimizationDirection.name`` (uppercase) while "
            "``SpecSummary.objectives[].direction`` serializes the enum value "
            "(lowercase). Two spellings of one enum in one response would make "
            "every client write a case-insensitive comparison, so the API picks one."
        )
    )


class SearchSLABreach(BaseModel):
    """The SLA filter that first failed at an infeasible probe."""

    model_config = ConfigDict(extra="forbid")
    metric_tag: str | None = Field(
        default=None, description="Metric tag the breached filter constrains."
    )
    stat: str | None = Field(
        default=None, description="Statistic the filter compares ('p95', 'avg', ...)."
    )
    op: str | None = Field(
        default=None, description="Comparison mnemonic: 'lt', 'le', 'gt', or 'ge'."
    )
    threshold: FiniteFloat | None = Field(
        default=None, description="Threshold the statistic was compared against."
    )
    observed: FiniteFloat | None = Field(
        default=None,
        description=(
            "Mean of the statistic across the iteration's successful trials. "
            "Display-only: the feasibility verdict itself uses ANY-trial-passes "
            "semantics, not this average. Null when no trial measured the metric."
        ),
    )


class SearchBoundaryEdge(BaseModel):
    """One side of the empirical SLA-feasibility boundary on the swept axis."""

    model_config = ConfigDict(extra="forbid")
    value: FiniteFloat | None = Field(
        default=None, description="Swept-dimension value at this edge."
    )
    iteration_idx: int | None = Field(
        default=None, ge=0, description="Iteration that observed this edge."
    )
    objective_value: FiniteFloat | None = Field(
        default=None,
        description="Objective at the same probe, for context. Feasible edge only.",
    )
    first_breach: SearchSLABreach | None = Field(
        default=None,
        description="Which SLA filter failed first. Infeasible edge only.",
    )


class SearchBoundarySummary(BaseModel):
    """Empirical SLA boundary along a 1-D search axis.

    Absent for multi-dimensional searches: "the highest value that still
    passed" is only well-defined when there is a single axis to order by.
    """

    model_config = ConfigDict(extra="forbid")
    swept_dim_path: str = Field(
        description="Dotted config path of the swept dimension, e.g. "
        "'phases.profiling.concurrency'."
    )
    feasible_max: SearchBoundaryEdge | None = Field(
        default=None,
        description="Highest swept value that satisfied every SLA filter. Null "
        "when no probe passed.",
    )
    infeasible_min: SearchBoundaryEdge | None = Field(
        default=None,
        description="Lowest swept value that violated an SLA filter. Null when no "
        "probe failed.",
    )
    boundary_type: str | None = Field(
        default=None,
        description=(
            "'smooth' or 'cliff', written only by the smooth-isotonic planner. "
            "'cliff' means the planner is reporting an honest bracket rather than "
            "a single boundary point, because the response is discontinuous there."
        ),
    )
    binding_constraint: str | None = Field(
        default=None,
        description="'<metric_tag>:<stat>' of the SLA that defines the boundary, "
        "when the planner identified one.",
    )


class SearchBestTrial(BaseModel):
    """One planner-selected winner from ``search_history.json``'s best_trials."""

    model_config = ConfigDict(extra="forbid")
    iteration_idx: int = Field(
        ge=0,
        description=(
            "Zero-based iteration index. Equals the variation index for adaptive "
            "sweeps, so clients can join this row against the variations table."
        ),
    )
    objective_values: list[FiniteFloat | None] | None = Field(
        default=None,
        description=(
            "One value per entry of ``objectives``, in the same order. Clients "
            "index this POSITIONALLY, so an entry the scorer could not produce "
            "is carried as an explicit null rather than dropped: compacting the "
            "vector shifts every later objective onto the wrong label. Null "
            "means 'this objective was not measured for this trial' and must be "
            "rendered as such, never as a neighbouring objective's value. The "
            "outer null means the trial was not scored at all."
        ),
    )
    variation_values: dict[str, Any] = Field(
        default_factory=dict,
        description="Swept parameter values that produced this trial, keyed by "
        "dotted config path.",
    )
    feasible: bool = Field(
        default=False,
        description="Whether this trial satisfied every configured SLA filter.",
    )
    feasible_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of scored iterations across the run that were feasible. Zero "
            "means NO iteration was both scored and feasible, so the planner "
            "ranked the full pool instead -- the winner is then a least-bad "
            "point, not a servable one, and must not be presented as optimal."
        ),
    )
    pareto_rank: int = Field(
        default=0,
        ge=0,
        description="Always 0 today; reserved for non-dominated sorting.",
    )


class SweepSearchSummary(BaseModel):
    """Planner-authored verdict for an adaptive sweep.

    Projected from the ``search_history.json`` artifact, which is the only
    place the planner's own feasibility verdict, stopping reason, and SLA
    boundary are recorded. Everything here is a *measurement the planner
    already made*; clients must prefer it over re-deriving the same
    quantities from per-variation metrics, because two implementations of
    one rule diverge in exactly the cases that matter (a point that breaches
    an SLA by a hair, a metric that was never measured).

    The full trajectory (``iterations[]``, ``config``) is deliberately NOT
    projected here -- it reaches ~100 KB on a long run and the raw artifact
    is already downloadable from the sweep artifacts routes. This model
    carries only what a summary view needs.
    """

    model_config = ConfigDict(extra="forbid")
    convergence_reason: str | None = Field(
        default=None,
        description=(
            "Verbatim planner stop reason, e.g. 'max_iterations', "
            "'improvement_patience', 'monotonic_precision_reached'. Null means "
            "the file was written mid-loop or the run ended abnormally "
            "(cancelled, crashed)."
        ),
    )
    stop_kind: Literal["converged", "budget_exhausted", "incomplete"] = Field(
        default="incomplete",
        description=(
            "Classification of ``convergence_reason`` so clients do not each "
            "re-parse a growing string enum. 'budget_exhausted' = ran out of "
            "iterations; 'converged' = a convergence rule fired (including the "
            "'unknown' clean-exit fallback, where the planner stopped but "
            "recorded no reason); 'incomplete' = no reason recorded at all."
        ),
    )
    iteration_count: int = Field(
        default=0,
        ge=0,
        description="Number of iterations recorded in the trajectory.",
    )
    feasible_iteration_count: int = Field(
        default=0,
        ge=0,
        description="How many of those iterations satisfied every SLA filter.",
    )
    objectives: list[SearchObjective] = Field(
        default_factory=list,
        description=(
            "Objectives the planner optimized, read from the artifact rather "
            "than the spec so an archived sweep whose CR is gone still labels "
            "its own objective values."
        ),
    )
    best_trials: list[SearchBestTrial] = Field(
        default_factory=list,
        description=(
            "Planner-selected winner(s): length 1 for single-objective runs, the "
            "Pareto front for multi-objective. Empty until an iteration produces "
            "a usable objective."
        ),
    )
    best_trials_truncated: bool = Field(
        default=False,
        description=f"True when the Pareto front exceeded {MAX_BEST_TRIALS} entries "
        "and was cut. Download the artifact for the full front.",
    )
    sla_filter_count: int = Field(
        default=0,
        ge=0,
        description=(
            "How many SLA filters the search was configured with. Zero makes "
            "every ``feasible`` flag in this response VACUOUSLY true -- the "
            "exporter defaults the verdict to true when nothing constrains it, "
            "so ``feasible_max`` degenerates to 'highest value tried'. Clients "
            "must check this before rendering any feasibility claim, or an "
            "unconstrained search will appear to have passed an SLA it never had."
        ),
    )
    boundary_summary: SearchBoundarySummary | None = Field(
        default=None,
        description="Empirical SLA boundary on the swept axis; null for "
        "multi-dimensional searches and for runs with no iterations.",
    )
    recipe: str | None = Field(
        default=None,
        description="Search recipe that authored the configuration, when one did.",
    )


class SweepDetailResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}."""

    model_config = ConfigDict(extra="forbid")
    sweep: SweepSummary
    status: dict[str, Any] = Field(default_factory=dict)
    spec_summary: SpecSummary
    search_summary: SweepSearchSummary | None = Field(
        default=None,
        description=(
            "Planner verdict for adaptive sweeps. Null for grid-family sweeps "
            "(which have no planner), for adaptive sweeps whose trajectory has "
            "not been harvested to the operator PVC yet, and whenever the "
            "artifact is unreadable -- a missing verdict degrades the page, it "
            "never fails the request."
        ),
    )
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
        ge=0, description="Filesystem mtime of the epoch dir, seconds since epoch."
    )
    file_count: int = Field(
        ge=0, description="Number of immediate children under the epoch dir."
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
    variation_index: int = Field(
        ge=0, description="Variation index from expand_sweep()."
    )
    variation_label: str = Field(
        default="", description="Human-readable variation label."
    )
    variation_values: str = Field(
        default="",
        description=(
            "Swept parameter values as a JSON object string, e.g. "
            '{"phases.profiling.concurrency":17}. Same encoding as '
            "AIPerfSweep.status.runs[].values. Empty when the sweep predates "
            "this field or the child carried no variation."
        ),
    )
    trial_index: int | None = Field(
        default=None,
        ge=0,
        description="Trial index within the variation, if multi-trial.",
    )
    child_run_epoch: str = Field(
        description="Child job run epoch on disk (decimal seconds)."
    )


class ChildrenManifestResponse(BaseModel):
    """Body of GET /api/v1/sweeps/{ns}/{name}/children."""

    model_config = ConfigDict(populate_by_name=True, alias_generator=to_camel)

    sweep_run_epoch: str = Field(description="Sweep epoch this manifest belongs to.")
    children: list[ChildrenManifestEntry] = Field(default_factory=list)


class CreateSweepRequest(BaseModel):
    """Body of POST /api/v1/sweeps: create an AIPerfSweep from a manifest dict."""

    model_config = ConfigDict(extra="forbid")
    manifest: dict[str, Any] = Field(
        description="Full AIPerfSweep manifest, shaped like `kubectl apply -f` input."
    )


class CreateSweepResponse(BaseModel):
    """Response from POST /api/v1/sweeps."""

    model_config = ConfigDict(extra="forbid")
    namespace: str = Field(description="Namespace of the created AIPerfSweep.")
    name: str = Field(description="Name of the created AIPerfSweep.")
    uid: str | None = Field(default=None, description="UID of the created AIPerfSweep.")
