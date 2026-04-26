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
    sweep_type: Literal["grid", "scenarios"] = Field(
        description="Variation generator kind."
    )
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
    age_seconds: int = Field(description="Seconds since CR/dir creation.")
    model: str | None = Field(
        default=None, description="Primary model name from template snapshot."
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
