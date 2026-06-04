# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for multi-tier SLO boundary search.

Defines the tier configuration, per-tier mutable bracket state, per-tier
output record, and the top-level boundary summary for multi-tier searches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.config.sweep.adaptive import SLOTier

__all__ = ["BracketState", "MultiTierBoundarySummary", "SLOTier", "TierResult"]


@dataclass(slots=True)
class BracketState:
    """Per-tier mutable bracket tracking state."""

    tier: SLOTier
    """The tier this bracket state belongs to."""
    feasible_max: int | None = None
    """Highest concurrency where all tier filters passed."""
    infeasible_min: int | None = None
    """Lowest concurrency where at least one tier filter failed."""
    converged: bool = False
    """Whether this tier's boundary has been resolved to within precision."""
    convergence_reason: str | None = None
    """Reason string for this tier's convergence (or None if still searching)."""
    binding_constraint: dict[str, Any] | None = field(default=None)
    """The SLA filter that first fails as concurrency increases."""
    probe_count: int = 0
    """Number of probes that contributed to this tier's bracket."""
    non_monotonic_warning: bool = False
    """True if a non-monotonic observation was detected for this tier."""


class TierResult(AIPerfBaseModel):
    """Per-tier output record for search_history.json."""

    label: str = Field(description="Tier identifier")
    boundary_concurrency: int | None = Field(
        description="Discovered max concurrency under SLA for this tier"
    )
    convergence_status: str = Field(
        description="One of: converged, partial, no_pass_in_range, no_failure_in_range",
    )
    convergence_reason: str | None = Field(
        default=None,
        description="Detailed reason string matching existing planner reasons",
    )
    binding_constraint: dict[str, Any] | None = Field(
        default=None,
        description="First-failing SLA filter at the boundary",
    )
    bracket_lower: int | None = Field(
        description="Current feasible_max (lower bracket bound)",
    )
    bracket_upper: int | None = Field(
        description="Current infeasible_min (upper bracket bound)",
    )
    confidence_interval: dict[str, float] | None = Field(
        default=None,
        description="Boundary CI from replicate phase: {low, high}",
    )
    probe_count: int = Field(description="Probes that informed this tier's bracket")
    filters: list[dict[str, Any]] = Field(
        description="Echo of the SLA filters that define this tier",
    )


class MultiTierBoundarySummary(AIPerfBaseModel):
    """Top-level boundary summary for multi-tier searches."""

    swept_dim_path: str = Field(description="Dotted path of the swept dimension")
    tiers: list[TierResult] = Field(description="Per-tier boundary results")
    total_probe_count: int = Field(description="Total probes across all tiers")
    ordering_detected: bool = Field(
        description="Whether monotonic tier ordering was exploited",
    )
    ordering_pairs: list[dict[str, str]] | None = Field(
        default=None,
        description="Detected ordering pairs: [{strict: label, lenient: label}]",
    )
