# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metadata schemas for orchestrator plugin categories.

Re-exported from schemas.py so plugins.yaml references like
``metadata_class: aiperf.plugin.schema.schemas:ConvergenceCriterionMetadata``
keep resolving - `aiperf.plugin.schema.schemas.ConvergenceCriterionMetadata`
is still importable via the re-export.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["ConvergenceCriterionMetadata", "SearchPlannerMetadata"]


class ConvergenceCriterionMetadata(BaseModel):
    """Metadata schema for convergence criterion plugins.

    Declares statistical-method capabilities so the CLI/config layer can
    validate `--convergence-metric` / `--convergence-stat` against the chosen
    criterion before the plugin is imported.

    Referenced by: categories.yaml convergence_criterion.metadata_class
    Used in: plugins.yaml convergence_criterion entries
    """

    min_samples: int = Field(
        ge=1,
        description="Minimum number of successful runs required before convergence can trigger.",
    )
    requires_confidence_level: bool = Field(
        default=False,
        description="Whether the criterion consumes plan.confidence_level (e.g. CI-width does, CV doesn't).",
    )
    requires_jsonl_export: bool = Field(
        default=False,
        description="Whether the criterion reads per-request metrics from JSONL exports (e.g. distribution does).",
    )
    metric_kinds: list[str] = Field(
        default_factory=lambda: ["continuous"],
        description=(
            "Kinds of metrics this criterion handles. One or more of "
            "'continuous', 'counts', 'categorical'."
        ),
    )


class SearchPlannerMetadata(BaseModel):
    """Metadata schema for search planner plugins.

    Declares dimension-type and objective-direction support so the CLI/config
    layer can validate `--search-space` shape against the chosen planner
    before the planner (and its heavy soft-dep imports) is loaded.

    Referenced by: categories.yaml search_planner.metadata_class
    Used in: plugins.yaml search_planner entries
    """

    supports_continuous: bool = Field(
        description="Whether the planner accepts Real-valued search-space dimensions.",
    )
    supports_discrete: bool = Field(
        description="Whether the planner accepts Integer-valued search-space dimensions.",
    )
    supports_categorical: bool = Field(
        default=False,
        description="Whether the planner accepts Categorical search-space dimensions.",
    )
    requires_initial_samples: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Minimum number of initial random/Sobol samples required before the "
            "planner's model is fit. None when the planner has no warm-up phase."
        ),
    )
    compatible_objective_directions: list[str] = Field(
        default_factory=lambda: ["maximize", "minimize"],
        description="Objective directions the planner can optimize. Lower-case strings.",
    )
    requires_extras: list[str] = Field(
        default_factory=list,
        description=(
            "Names of pyproject.toml extras (e.g. ['optuna']) required to install "
            "this planner's heavy dependencies. Informational; the planner "
            "class itself owns the ImportError surface."
        ),
    )
